"""Trace-driven load layer (Phase 2).

Turns the cluster-averaged Alibaba 2018 ``machine_usage`` trace into a
per-slot, per-timestep utilization array that drives the existing thermal
model. Responsibilities:

1. Load the Parquet file and convert ``cpu_util_percent`` from a [0, 100]
   scale to a utilization fraction ``u`` in [0, 1]. CPU is the primary load
   driver; the other columns are loaded and kept available but unused for now.
2. Resample the 300 s-stride signal up to the 1 s sim timestep. The
   interpolation method is a clearly named function dispatched by config name,
   so it is easy to swap later.
3. Apply a fixed, deterministically-seeded per-slot variation so the slots are
   not all identical, then clip to [0, 1].

The result is precomputed into a ``[n_slots, n_timesteps]`` array for speed;
``get_utilization(slot, t)`` is a convenience accessor over that array.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Columns kept available for later phases even though only CPU drives load now.
AUX_COLUMNS = ("mem_util_percent", "net_in", "net_out", "disk_io_percent")


# --- Interpolation methods --------------------------------------------------
# Each interpolator maps source samples (spaced ``stride`` seconds apart) onto a
# 1 s grid. Register new methods here and select them by name from config.

def interpolate_linear(values: FloatArray, stride: int) -> FloatArray:
    """Linear interpolation of ``values`` onto a 1 s grid.

    Source sample ``i`` sits at time ``i * stride``. The output covers every
    integer second from 0 up to and including the last source time, so two
    source points ``stride`` seconds apart yield ``stride + 1`` samples
    (sharing the endpoint with the next interval).
    """
    n = values.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return values.astype(np.float64, copy=True)
    src_t = np.arange(n) * stride
    dst_t = np.arange(src_t[-1] + 1)
    return np.interp(dst_t, src_t, values).astype(np.float64)


INTERPOLATORS: dict[str, Callable[[FloatArray, int], FloatArray]] = {
    "linear": interpolate_linear,
}


@dataclass(frozen=True)
class TraceConfig:
    """Configuration for the trace-driven load layer."""

    path: str
    source_stride_s: int = 300
    cpu_column: str = "cpu_util_percent"
    interpolation: str = "linear"
    seed: int = 1729
    scale_spread: float = 0.08
    offset_spread: float = 0.01

    @classmethod
    def from_config(cls, cfg: dict) -> "TraceConfig":
        """Build a TraceConfig from the `trace` section of the config dict."""
        var = cfg.get("variation", {})
        return cls(
            path=str(cfg["path"]),
            source_stride_s=int(cfg["source_stride_s"]),
            cpu_column=str(cfg.get("cpu_column", "cpu_util_percent")),
            interpolation=str(cfg.get("interpolation", "linear")),
            seed=int(var.get("seed", 1729)),
            scale_spread=float(var.get("scale_spread", 0.08)),
            offset_spread=float(var.get("offset_spread", 0.01)),
        )


def load_cpu_fraction(path: str | Path, cpu_column: str = "cpu_util_percent") -> FloatArray:
    """Read the Parquet trace and return CPU utilization as a fraction in [0, 1].

    The conversion [0, 100] -> [0, 1] happens exactly once, here. (It is then
    clipped defensively so a noisy >100 sample cannot leak a >1 fraction.)
    """
    df = pd.read_parquet(path)
    if cpu_column not in df.columns:
        raise KeyError(f"trace column {cpu_column!r} not found in {path}")
    pct = df[cpu_column].to_numpy(dtype=np.float64)
    return np.clip(pct / 100.0, 0.0, 1.0)


def per_slot_variation(
    n_slots: int, seed: int, scale_spread: float, offset_spread: float
) -> tuple[FloatArray, FloatArray]:
    """Deterministic per-slot (multiplier, offset) pair.

    Drawn once from a seeded RNG so the values are identical on every run — no
    seed drift between runs. ``multiplier ~ 1 ± scale_spread`` and
    ``offset ~ ± offset_spread``, both uniform.
    """
    rng = np.random.default_rng(seed)
    multiplier = 1.0 + rng.uniform(-scale_spread, scale_spread, size=n_slots)
    offset = rng.uniform(-offset_spread, offset_spread, size=n_slots)
    return multiplier, offset


class TraceLoader:
    """Precomputed per-slot utilization driver over a trace signal.

    Construct directly from a base fraction array (handy for tests) or via
    :meth:`from_config`, which loads the Parquet file named in the config.
    """

    def __init__(
        self,
        base_fraction: FloatArray,
        n_slots: int,
        *,
        source_stride_s: int = 300,
        interpolation: str = "linear",
        seed: int = 1729,
        scale_spread: float = 0.08,
        offset_spread: float = 0.01,
    ) -> None:
        if interpolation not in INTERPOLATORS:
            raise ValueError(
                f"unknown interpolation {interpolation!r}; "
                f"available: {sorted(INTERPOLATORS)}"
            )
        self.n_slots = int(n_slots)
        self.source_stride_s = int(source_stride_s)
        self.interpolation = interpolation

        # Keep the raw points (and their 1 s timestamps) for validation plots.
        self.source_fraction: FloatArray = np.clip(
            np.asarray(base_fraction, dtype=np.float64), 0.0, 1.0
        )
        self.source_times: FloatArray = (
            np.arange(self.source_fraction.shape[0]) * self.source_stride_s
        ).astype(np.float64)

        # 1 s resolution cluster-average signal, shared by all slots.
        self.signal_1s: FloatArray = INTERPOLATORS[interpolation](
            self.source_fraction, self.source_stride_s
        )

        # Fixed per-slot transform, then clip to [0, 1].
        self.multiplier, self.offset = per_slot_variation(
            self.n_slots, seed, scale_spread, offset_spread
        )
        util = (
            self.signal_1s[None, :] * self.multiplier[:, None]
            + self.offset[:, None]
        )
        self.utilization: FloatArray = np.clip(util, 0.0, 1.0)

    @property
    def n_timesteps(self) -> int:
        return self.utilization.shape[1]

    def get_utilization(self, slot_index: int, t: int) -> float:
        """Utilization of one slot at sim time ``t`` (seconds)."""
        return float(self.utilization[slot_index, t])

    def utilization_at(self, t: int) -> FloatArray:
        """Per-slot utilization column at sim time ``t`` (seconds)."""
        return self.utilization[:, t]

    @classmethod
    def from_config(cls, config: dict, n_slots: int) -> "TraceLoader":
        """Build a TraceLoader from a parsed config dict and the rack size."""
        tcfg = TraceConfig.from_config(config["trace"])
        base = load_cpu_fraction(tcfg.path, tcfg.cpu_column)
        return cls(
            base,
            n_slots,
            source_stride_s=tcfg.source_stride_s,
            interpolation=tcfg.interpolation,
            seed=tcfg.seed,
            scale_spread=tcfg.scale_spread,
            offset_spread=tcfg.offset_spread,
        )

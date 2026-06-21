"""Per-node burstiness layer (Phase 2b).

The cluster-averaged Alibaba trace is too smooth: averaging erased the
node-to-node heterogeneity, so every slot tracks the same gentle signal and the
rack never gets hot enough for cooling control to matter. This layer injects
deterministic, *independent* per-node load bursts on top of the Phase 2
baseline utilization, so individual slots intermittently spike toward full load
while the cluster average stays moderate — recreating the heterogeneity and
pushing some nodes into the 75–85 °C regime.

A burst is a temporary additive elevation of one slot's utilization
(``baseline + burst``, then clipped to [0, 1]). Bursts are generated per slot
from an independent, deterministically-seeded RNG (``seed + slot_index``), so:

* slots burst at different times -> nodes diverge from one another (the point);
* runs are byte-for-byte reproducible with zero seed drift;
* the whole layer is toggleable (``enabled: false`` recovers the exact Phase 2
  baseline).

Physical note: the thermal model relaxes with τ ≈ 60 s, so a burst much shorter
than that barely moves temperature — the node sheds heat as fast as it arrives.
Burst durations are therefore on the order of minutes (several × τ) and
magnitudes lift utilization into the ~0.85–1.0 range, so sustained bursts
genuinely heat nodes rather than just making the utilization plot look spiky.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .trace_loader import TraceLoader

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BurstConfig:
    """Configuration for the per-node burst-injection layer."""

    enabled: bool = True
    seed: int = 2024
    arrival_per_hour: float = 2.0      # expected burst starts per slot per hour
    magnitude_min: float = 0.5         # additive elevation added to baseline u
    magnitude_max: float = 0.7
    duration_min_s: int = 180          # burst length [s] — must be several × τ
    duration_max_s: int = 480

    @classmethod
    def from_config(cls, cfg: dict) -> "BurstConfig":
        """Build a BurstConfig from the `burstiness` section of the config."""
        mag = cfg.get("magnitude", [0.5, 0.7])
        dur = cfg.get("duration_s", [180, 480])
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            seed=int(cfg.get("seed", 2024)),
            arrival_per_hour=float(cfg.get("arrival_per_hour", 2.0)),
            magnitude_min=float(mag[0]),
            magnitude_max=float(mag[1]),
            duration_min_s=int(dur[0]),
            duration_max_s=int(dur[1]),
        )


def generate_slot_elevation(
    n_timesteps: int, rng: np.random.Generator, cfg: BurstConfig
) -> FloatArray:
    """Additive burst elevation (length ``n_timesteps``) for a single slot.

    Bursts are laid down sequentially and never overlap: each burst occupies
    ``[start, start + duration)`` and the next start is the previous end plus an
    exponentially-distributed idle gap (mean ``3600 / arrival_per_hour`` s).
    """
    elevation = np.zeros(n_timesteps, dtype=np.float64)
    if n_timesteps == 0 or cfg.arrival_per_hour <= 0.0:
        return elevation

    mean_gap = 3600.0 / cfg.arrival_per_hour
    t = rng.exponential(mean_gap)  # first burst start
    while t < n_timesteps:
        start = int(t)
        duration = int(
            rng.integers(cfg.duration_min_s, cfg.duration_max_s, endpoint=True)
        )
        magnitude = rng.uniform(cfg.magnitude_min, cfg.magnitude_max)
        end = min(n_timesteps, start + duration)
        elevation[start:end] = magnitude
        t = end + rng.exponential(mean_gap)
    return elevation


def inject_bursts(baseline: FloatArray, cfg: BurstConfig) -> FloatArray:
    """Return ``baseline`` with per-slot bursts added and clipped to [0, 1].

    With ``cfg.enabled == False`` the baseline is returned unchanged (an exact
    copy), so the smooth Phase 2 behaviour is recovered bit-for-bit.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    if not cfg.enabled:
        return baseline.copy()

    n_slots, n_timesteps = baseline.shape
    out = baseline.copy()
    for slot in range(n_slots):
        # Independent stream per slot -> divergence; deterministic -> no drift.
        rng = np.random.default_rng(cfg.seed + slot)
        out[slot] += generate_slot_elevation(n_timesteps, rng, cfg)
    return np.clip(out, 0.0, 1.0)


class BurstyTraceLoader:
    """Trace loader with the per-node burstiness layer applied.

    Composes a Phase 2 :class:`TraceLoader` and exposes the same accessor
    interface (``utilization``, ``utilization_at``, ``n_timesteps``,
    ``n_slots``), plus ``baseline_utilization`` for off-vs-on comparison.
    """

    def __init__(self, loader: TraceLoader, cfg: BurstConfig) -> None:
        self.loader = loader
        self.config = cfg
        self.n_slots = loader.n_slots
        self.baseline_utilization: FloatArray = loader.utilization
        self.utilization: FloatArray = inject_bursts(loader.utilization, cfg)

    @property
    def n_timesteps(self) -> int:
        return self.utilization.shape[1]

    def get_utilization(self, slot_index: int, t: int) -> float:
        return float(self.utilization[slot_index, t])

    def utilization_at(self, t: int) -> FloatArray:
        return self.utilization[:, t]

    @classmethod
    def from_config(cls, config: dict, n_slots: int) -> "BurstyTraceLoader":
        """Build the baseline TraceLoader then apply the burstiness layer."""
        loader = TraceLoader.from_config(config, n_slots)
        cfg = BurstConfig.from_config(config.get("burstiness", {}))
        return cls(loader, cfg)

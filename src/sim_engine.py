"""Simulation engine facade for the visualization data layer (Phase 8).

This is the backend contract for a (separately built) visualization. It exposes
the room simulation as a stream of plain, JSON-serializable frames — no plotting,
no frontend. Teammates can consume it two ways:

* **Recorded run** — ``scripts/export_run.py`` writes a single JSON file
  (``metadata`` + ``topology`` + ``frames``) the viz loads and replays.
* **Live** — import ``SimulationEngine``, call ``reset()`` / ``step(demand)``,
  and read each returned frame dict (e.g. wrap it in your own web server).

Every value in a frame is a built-in (float/int/str/bool/list), so
``json.dumps(frame)`` always works. See ``SCHEMA.md`` for the field contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .burstiness import BurstConfig, inject_bursts
from .room import Room
from .trace_loader import (
    TraceConfig,
    interpolate_linear,
    load_cpu_fraction,
    per_slot_variation,
)

SCHEMA_VERSION = "1.0"

# Repo root, so the default trace path resolves regardless of CWD.
_ROOT = Path(__file__).resolve().parents[1]


def generate_room_demand(
    config: dict, n_ticks: int, scale: float = 1.0
) -> NDArray[np.float64]:
    """Realistic aggregate room demand: sum of one bursty stream per (rack, slot).

    Built only for the requested window (no full-length allocation). Deterministic
    given the trace/burstiness seeds in config, so recorded runs are reproducible.
    """
    n_slots = int(config["rack"]["n_slots"])
    n_racks = int(config["room"]["n_air_racks"]) + int(config["room"]["n_liquid_racks"])
    total_slots = n_racks * n_slots

    tcfg = TraceConfig.from_config(config["trace"])
    base = load_cpu_fraction(_ROOT / tcfg.path, tcfg.cpu_column)
    signal = interpolate_linear(base, tcfg.source_stride_s)[:n_ticks]
    mult, off = per_slot_variation(total_slots, tcfg.seed,
                                   tcfg.scale_spread, tcfg.offset_spread)
    util = np.clip(signal[None, :] * mult[:, None] + off[:, None], 0.0, 1.0)
    bursty = inject_bursts(util, BurstConfig.from_config(config["burstiness"]))
    return bursty.sum(axis=0) * scale


class SimulationEngine:
    """Drives the room and emits JSON-serializable per-tick frames."""

    def __init__(self, config: dict, route: bool = True) -> None:
        self.config = config
        self.route = route
        self.reset()

    def reset(self) -> None:
        """Rebuild the room and rewind to tick 0 (deterministic)."""
        self.room = Room(self.config, route=self.route)
        self.t = 0

    # --- static descriptors --------------------------------------------------

    def metadata(self, output_stride_s: int = 1) -> dict:
        """Run-level metadata: units, thresholds, energy params, seeds, policy."""
        ca = self.config["cooling_agent"]
        zones = {
            k: {"kind": z["kind"], "t_supply_c": float(z["T_supply"])}
            for k, z in self.config["zones"].items()
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "dt_s": float(self.config["sim"]["dt"]),
            "output_stride_s": int(output_stride_s),
            "routing": "liquid-first" if self.route else "naive-even-split",
            "thresholds_c": {
                "warn": float(ca["warn_c"]),
                "critical": float(ca["critical_c"]),
                "release": float(ca["release_c"]),
            },
            "fan_max": float(ca["fan_max"]),
            "zones": zones,
            "energy": {
                "exponent": float(self.config["energy"]["exponent"]),
                "rated_w": {k: float(v)
                            for k, v in self.config["energy"]["rated_w"].items()},
            },
            "seeds": {
                "trace": int(self.config["trace"]["variation"]["seed"]),
                "burstiness": int(self.config["burstiness"]["seed"]),
            },
            "units": {
                "temperature": "celsius", "power": "watts",
                "omega": "fan/pump speed multiplier", "util": "fraction 0-1",
            },
        }

    def topology(self) -> dict:
        """Static room layout (rack IDs, zones, slot count)."""
        return self.room.topology()

    # --- stepping ------------------------------------------------------------

    def step(self, demand_total: float, include_slots: bool = False) -> dict:
        """Advance one tick under a scalar room demand; return a frame dict."""
        res = self.room.step(float(demand_total), self.t)
        frame = {
            "t": self.t,
            "room": {
                "demand": round(float(demand_total), 1),
                "it_w": round(res.it_power_w, 1),
                "cooling_w": round(res.cooling_power_w, 1),
                "pue": round(res.pue, 4),
                "tpi": round(self.room.tpi, 4),
                "dropped": round(res.dropped, 2),
                "throttles": int(res.throttles),
                "max_temp_air": round(res.max_temp_air, 2),
                "max_temp_liquid": round(res.max_temp_liquid, 2),
            },
            "racks": self.room.rack_states(include_slots=include_slots),
        }
        self.t += 1
        return frame

"""Simulation Runner — OODA loop orchestrator.

Ties all four agents together and drives the simulation at a configurable
tick rate. Maintains two parallel simulation tracks:
  - BASELINE:     round-robin routing, fixed fans, no shedding
  - MULTI_AGENT:  LTI routing, predictive cooling, Arbiter negotiation

Both tracks are always running so comparison data is always available
for the analytics charts.
"""
from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cooling_agent import CoolingAgent
from src.datacenter_layout import RACKS, to_topology_dict
from src.rack import Rack
from src.agents.arbiter import ArbiterAgent, ArbiterConfig
from src.agents.router import RouterAgent
from src.agents.scheduler import SchedulerAgent, SchedulerConfig
from src.agents.thermal_sentinel import ThermalSentinel

# ── Configuration ────────────────────────────────────────────────────────────

CONFIG_PATH = ROOT / "config" / "sim_config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_rack(config: dict, zone_key: str) -> Rack:
    return Rack.from_config(config, zone_key)


# ── KPI accumulators ─────────────────────────────────────────────────────────

class KPITracker:
    def __init__(self) -> None:
        self.energy_j: float = 0.0
        self.violations_s: int = 0
        self.ticks: int = 0
        self.shed_ticks: int = 0
        # History for charts (keep last 300 ticks)
        self.energy_history: Deque[float] = deque(maxlen=300)
        self.peak_temp_history: Deque[float] = deque(maxlen=300)
        self.violation_history: Deque[int] = deque(maxlen=300)
        self.fan_history: Deque[float] = deque(maxlen=300)
        self.load_history: Deque[float] = deque(maxlen=300)

    def update(self, total_power_w: float, peak_temp: float,
               dt: float, throttle: bool, fan: float, load: float) -> None:
        self.energy_j += total_power_w * dt
        self.ticks += 1
        violated = 1 if peak_temp > 85.0 else 0
        self.violations_s += violated
        if throttle:
            self.shed_ticks += 1
        self.energy_history.append(round(self.energy_j / 3600000, 4))
        self.peak_temp_history.append(round(peak_temp, 1))
        self.violation_history.append(violated)
        self.fan_history.append(round(fan, 2))
        self.load_history.append(round(load, 3))

    @property
    def energy_kwh(self) -> float:
        return round(self.energy_j / 3600000, 4)

    @property
    def task_preservation_pct(self) -> float:
        if self.ticks == 0:
            return 100.0
        return round(100.0 * (1.0 - self.shed_ticks / self.ticks), 1)

    def to_dict(self) -> dict:
        return {
            "energy_kwh": self.energy_kwh,
            "violations_s": self.violations_s,
            "task_preservation_pct": self.task_preservation_pct,
            "history": {
                "energy": list(self.energy_history),
                "peak_temp": list(self.peak_temp_history),
                "violations": list(self.violation_history),
                "fan_speed": list(self.fan_history),
                "load": list(self.load_history),
            }
        }


# ── Simulation Runner ─────────────────────────────────────────────────────────

class SimulationRunner:
    """Drives the simulation at a desired tick rate and returns JSON frames."""

    def __init__(self) -> None:
        self.config = _load_config()
        self._tick = 0
        self.mode: str = "multi_agent"  # "baseline" or "multi_agent"
        self._running = False
        self._log: Deque[str] = deque(maxlen=100)

        # Scheduler & Router (shared — both modes read the same load)
        sched_cfg = SchedulerConfig(
            **self.config.get("scheduler", {})
        )
        self.scheduler = SchedulerAgent(sched_cfg)
        rack_ids = [r.id for r in RACKS]
        self.router = RouterAgent(rack_ids)

        # Arbiter
        arb_cfg_raw = self.config.get("arbiter", {})
        self.arbiter = ArbiterAgent(ArbiterConfig(
            critical_c=float(arb_cfg_raw.get("critical_c", self.config["cooling_agent"]["critical_c"])),
            release_c=float(arb_cfg_raw.get("release_c", self.config["cooling_agent"]["release_c"])),
            p3_shed=float(arb_cfg_raw.get("shed_p3_pct", 1.0)),
            p2_shed=float(arb_cfg_raw.get("shed_p2_pct", 0.5)),
        ))

        # Build one Sentinel (= Rack + CoolingAgent) per rack for each mode
        self.sentinels_ma:   Dict[str, ThermalSentinel] = {}
        self.sentinels_base: Dict[str, ThermalSentinel] = {}
        self.kpi_ma   = KPITracker()
        self.kpi_base = KPITracker()

        for rack_def in RACKS:
            rack_ma   = _build_rack(self.config, rack_def.zone_key)
            rack_base = _build_rack(self.config, rack_def.zone_key)
            agent     = CoolingAgent.from_config(self.config)
            agent_b   = CoolingAgent.from_config(self.config)
            self.sentinels_ma[rack_def.id]   = ThermalSentinel(rack_ma,   agent,   rack_def.id)
            self.sentinels_base[rack_def.id] = ThermalSentinel(rack_base, agent_b, rack_def.id)

        self.topology = to_topology_dict()
        self._last_frame: Optional[dict] = None

    # ── Public ───────────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        assert mode in ("baseline", "multi_agent")
        self.mode = mode
        self._log.append(f"[t={self._tick:05d}] [SYSTEM] Mode switched → {mode.upper()}")

    def tick(self) -> dict:
        """Advance one simulation step. Returns the full state frame."""
        self._tick += 1
        t = self._tick
        dt = float(self.config["sim"]["dt"])

        # 1. Observe ── sample network load
        self.scheduler.tick()
        load = self.scheduler.effective_load

        # Build per-slot utilization for each rack (vary slightly per rack)
        rng = np.random.default_rng(t)

        def make_util(base_load: float, n_slots: int, rack_idx: int) -> np.ndarray:
            noise = 0.06 * rng.standard_normal(n_slots)
            rack_bias = 0.04 * np.sin(2 * np.pi * rack_idx / len(RACKS) + t / 120)
            return np.clip(base_load + rack_bias + noise, 0.05, 1.0)

        # 2. Orient + Decide + Act ── advance both simulation tracks
        peek_temps_ma = {rid: s.peak_temp for rid, s in self.sentinels_ma.items()}

        # Packet routing
        n_pkts = max(1, int(load * 20))
        priority_counts = {
            p: max(0, int(n_pkts * self.scheduler.config.priority_split[p]))
            for p in range(4)
        }
        self.router.route(n_pkts, priority_counts, peek_temps_ma,
                          multi_agent=(self.mode == "multi_agent"))

        # Arbitration
        sentinel_dicts_ma = {rid: s.to_dict() for rid, s in self.sentinels_ma.items()}
        new_logs = self.arbiter.evaluate(
            sentinel_dicts_ma, self.scheduler, t,
            multi_agent=(self.mode == "multi_agent")
        )
        for lg in new_logs:
            self._log.append(lg)

        # Step both tracks
        racks_out = []
        total_power_ma = 0.0
        total_power_base = 0.0
        peak_ma_global = 22.0
        peak_base_global = 22.0
        fan_ma_sum = 0.0

        for idx, rack_def in enumerate(RACKS):
            rid = rack_def.id
            util = make_util(load, self.config["rack"]["n_slots"], idx)

            # Multi-agent track
            dec_ma = self.sentinels_ma[rid].step(util, multi_agent=True)
            sm = self.sentinels_ma[rid]
            slot_powers_ma = sm.slot_powers
            total_power_ma += sum(slot_powers_ma)
            peak_ma_global = max(peak_ma_global, sm.peak_temp)
            fan_ma_sum += dec_ma.omega_fan

            # Baseline track
            dec_base = self.sentinels_base[rid].step(util, multi_agent=False)
            sb = self.sentinels_base[rid]
            total_power_base += sum(sb.slot_powers)
            peak_base_global = max(peak_base_global, sb.peak_temp)

            # Build rack output (use the currently active mode for display)
            active = sm if self.mode == "multi_agent" else sb
            active_dec = dec_ma if self.mode == "multi_agent" else dec_base
            racks_out.append({
                "id": rid,
                "kind": rack_def.kind,
                "aisle": rack_def.aisle,
                "peak_temp": active.peak_temp,
                "avg_temp": active.avg_temp,
                "omega_fan": active_dec.omega_fan,
                "throttle_active": active_dec.throttle_request,
                "state": active_dec.state,
                "slots": [
                    {
                        "id": i + 1,
                        "temp": active.slot_temps[i],
                        "util": active.slot_utils[i],
                        "power": active.slot_powers[i],
                    }
                    for i in range(len(active.slot_temps))
                ],
            })

        # 3. KPI update
        avg_fan_ma = fan_ma_sum / len(RACKS)
        self.kpi_ma.update(total_power_ma, peak_ma_global, dt,
                           self.arbiter._throttle_active, avg_fan_ma, load)
        self.kpi_base.update(total_power_base, peak_base_global, dt,
                             False, 1.0, load)

        # 4. Compose frame
        frame = {
            "tick": t,
            "mode": self.mode,
            "racks": racks_out,
            "topology": {
                **self.topology,
                "packets": self.router.to_dict()["packets"],
            },
            "agents": {
                "scheduler": self.scheduler.to_dict(),
                "router": self.router.to_dict(),
                "arbiter": self.arbiter.to_dict(),
                "sentinel_summary": {
                    rid: self.sentinels_ma[rid].to_dict()
                    for rid in list(self.sentinels_ma)[:3]  # top 3 for brevity
                },
            },
            "kpis": {
                "active": self.kpi_ma.to_dict() if self.mode == "multi_agent" else self.kpi_base.to_dict(),
                "multi_agent": self.kpi_ma.to_dict(),
                "baseline": self.kpi_base.to_dict(),
            },
            "log": list(self._log)[-20:],
        }
        self._last_frame = frame
        return frame

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.__init__()

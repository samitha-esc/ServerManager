"""Server-room composition (Phase 7) — the full agent stack over many racks.

A `Room` is a mix of air and liquid racks. Each tick it runs the whole control
stack end to end:

    cross-zone router  ->  per-rack Cooling Agent (fans/pump)
                       ->  per-rack Workload Agent (priority shedding)
                       ->  thermal step  ->  IT + cooling energy accounting

Routing is liquid-first (fill the efficient liquid pool, spill to air); set
``route=False`` for the naive even-split baseline. Cooling and workload agents
are per rack (each holds its own latch / deferral buffer). Energy is tracked as
IT power (compute) + cooling power (fans ∝ ω³, pumps), giving a room PUE.

Scale note: this is the "start medium, design for large" form — racks are looped
as Python objects (fine to ~dozens of racks). The per-rack loop body in `step`
is the single vectorization point: batching state into ``[R, n_slots]`` arrays
later collapses the loop into array ops without changing the control logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cooling_agent import CoolingAgent
from .cross_zone_router import split_liquid_first
from .energy import EnergyModel
from .rack import Rack
from .workload_agent import WorkloadAgent


@dataclass(frozen=True)
class RoomConfig:
    """Server-room composition parameters (from the `room` config section)."""

    n_air_racks: int = 35
    n_liquid_racks: int = 15
    liquid_util_cap: float = 1.0   # mirrors routing.liquid_util_cap
    demand_scale: float = 1.0

    @classmethod
    def from_config(cls, config: dict) -> "RoomConfig":
        room = config.get("room", {})
        routing = config.get("routing", {})
        return cls(
            n_air_racks=int(room.get("n_air_racks", 35)),
            n_liquid_racks=int(room.get("n_liquid_racks", 15)),
            liquid_util_cap=float(routing.get("liquid_util_cap", 1.0)),
            demand_scale=float(room.get("demand_scale", 1.0)),
        )


@dataclass
class RoomStepResult:
    """Per-tick room aggregates (logged; cheap, no per-slot history)."""

    t: int
    max_temp_air: float
    max_temp_liquid: float
    it_power_w: float
    cooling_power_w: float
    pue: float
    liquid_load: float
    air_load: float
    dropped: float
    throttles: int          # racks asserting THROTTLE_REQUEST this tick


@dataclass
class _Unit:
    """One rack plus its dedicated cooling + workload agents."""

    rack: Rack
    cooling: CoolingAgent
    workload: WorkloadAgent
    kind: str
    rack_id: str = ""
    last: dict = field(default_factory=dict)  # latest per-rack snapshot (viz)


class Room:
    """A room of air + liquid racks running the full control stack."""

    def __init__(self, config: dict, route: bool = True) -> None:
        self.config = config
        self.rc = RoomConfig.from_config(config)
        self.n_slots = int(config["rack"]["n_slots"])
        self.dt = float(config["sim"]["dt"])
        self.route = route
        self.energy = EnergyModel.from_config(config)

        def make(kind: str) -> _Unit:
            return _Unit(
                rack=Rack.from_config(config, kind),
                cooling=CoolingAgent.from_config(config),
                workload=WorkloadAgent.from_config(config),
                kind=kind,
            )

        self.liquid: list[_Unit] = [make("liquid") for _ in range(self.rc.n_liquid_racks)]
        self.air: list[_Unit] = [make("air") for _ in range(self.rc.n_air_racks)]
        for i, u in enumerate(self.liquid):
            u.rack_id = f"liquid-{i:02d}"
        for i, u in enumerate(self.air):
            u.rack_id = f"air-{i:02d}"

        self.liquid_capacity = len(self.liquid) * self.n_slots * self.rc.liquid_util_cap
        self.air_capacity = float(len(self.air) * self.n_slots)

        # Cumulative energy [J] and dropped load.
        self.it_energy_j = 0.0
        self.cooling_energy_j = 0.0
        self.total_dropped = 0.0

    # --- placement ----------------------------------------------------------

    def _place(self, demand_total: float) -> tuple[float, float, float]:
        """Split room demand into (liquid_total, air_total, dropped)."""
        if self.route:
            return split_liquid_first(demand_total, self.liquid_capacity,
                                      self.air_capacity)
        # Naive: even per-slot load across every rack/slot, regardless of zone.
        total_slots = (len(self.air) + len(self.liquid)) * self.n_slots
        per_slot = demand_total / total_slots if total_slots else 0.0
        per_slot = min(per_slot, 1.0)
        liquid_total = per_slot * len(self.liquid) * self.n_slots
        air_total = per_slot * len(self.air) * self.n_slots
        dropped = max(0.0, demand_total - self.liquid_capacity - self.air_capacity)
        return liquid_total, air_total, dropped

    # --- per-tick -----------------------------------------------------------

    def step(self, demand_total: float, t: int) -> RoomStepResult:
        """Advance the whole room one tick under a scalar room demand."""
        D = demand_total * self.rc.demand_scale
        liquid_total, air_total, dropped = self._place(D)
        self.total_dropped += dropped

        it_w = 0.0
        cool_w = 0.0
        throttles = 0
        max_air = 0.0
        max_liq = 0.0

        # --- vectorization point: this per-rack loop batches to [R, n_slots] ---
        for units, pool_total in ((self.liquid, liquid_total), (self.air, air_total)):
            if not units:
                continue
            cap = self.rc.liquid_util_cap if units[0].kind == "liquid" else 1.0
            per_slot = min(cap, pool_total / (len(units) * self.n_slots))
            u = np.full(self.n_slots, per_slot, dtype=np.float64)
            for unit in units:
                cd = unit.cooling.control(unit.rack, u)
                wd = unit.workload.step(u, cd, unit.rack, t)
                t_node, _, p = unit.rack.step(wd.effective_utilization)
                rack_it = float(p.sum())
                rack_cool = self.energy.cooling_power_w(unit.kind, unit.rack.zone.omega_fan)
                it_w += rack_it
                cool_w += rack_cool
                throttles += int(cd.throttle_request)
                if unit.kind == "liquid":
                    max_liq = max(max_liq, float(t_node.max()))
                else:
                    max_air = max(max_air, float(t_node.max()))
                # Per-rack snapshot for the visualization data layer.
                unit.last = {
                    "id": unit.rack_id,
                    "kind": unit.kind,
                    "max_temp": round(float(t_node.max()), 2),
                    "mean_temp": round(float(t_node.mean()), 2),
                    "omega": round(float(unit.rack.zone.omega_fan), 3),
                    "throttle": bool(cd.throttle_request),
                    "state": cd.state,
                    "util": round(float(np.mean(wd.effective_utilization)), 3),
                    "it_w": round(rack_it, 1),
                    "cooling_w": round(rack_cool, 1),
                }

        self.it_energy_j += it_w * self.dt
        self.cooling_energy_j += cool_w * self.dt

        return RoomStepResult(
            t=t,
            max_temp_air=max_air,
            max_temp_liquid=max_liq,
            it_power_w=it_w,
            cooling_power_w=cool_w,
            pue=EnergyModel.pue(it_w, cool_w),
            liquid_load=liquid_total,
            air_load=air_total,
            dropped=dropped,
            throttles=throttles,
        )

    # --- aggregates ---------------------------------------------------------

    @property
    def n_racks(self) -> int:
        return len(self.air) + len(self.liquid)

    # --- visualization data layer ------------------------------------------

    def topology(self) -> dict:
        """Static room layout (emit once): rack IDs, zones, slot count."""
        return {
            "n_slots_per_rack": self.n_slots,
            "n_air_racks": len(self.air),
            "n_liquid_racks": len(self.liquid),
            "racks": [
                {"id": u.rack_id, "kind": u.kind, "zone": u.rack.zone.name,
                 "index": i}
                for i, u in enumerate((*self.liquid, *self.air))
            ],
        }

    def rack_states(self, include_slots: bool = False) -> list[dict]:
        """Per-rack snapshot from the latest step (JSON-serializable).

        With ``include_slots`` each record also carries the 15 per-slot
        temperatures (for in-rack heatmaps).
        """
        out: list[dict] = []
        for u in (*self.liquid, *self.air):
            rec = dict(u.last)
            if include_slots:
                rec["slot_temps"] = [round(float(x), 2) for x in u.rack.t_node]
            out.append(rec)
        return out

    @property
    def total_energy_j(self) -> float:
        return self.it_energy_j + self.cooling_energy_j

    @property
    def pue(self) -> float:
        """Cumulative room PUE over the run so far."""
        return EnergyModel.pue(self.it_energy_j, self.cooling_energy_j)

    @property
    def tpi(self) -> float:
        """Room-wide Task Preservation Index across all workload agents."""
        executed = sum(u.workload.total_executed for u in (*self.air, *self.liquid))
        expired = sum(u.workload.total_expired for u in (*self.air, *self.liquid))
        denom = executed + expired
        return 1.0 if denom <= 0 else executed / denom

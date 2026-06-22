"""Cross-zone workload router (Phase 6) — liquid-first placement, spill to air.

The cluster has two racks: a cool, finite-capacity **liquid** rack and a hotter
**air** rack. Each tick the router takes the incoming cluster demand and decides
how much runs on each zone, **conserving total load** (work is relocated between
zones, never created or dropped):

    liquid_total = min(demand, liquid_capacity)   # fill the cool zone first
    air_total    = demand - liquid_total          # spill the overflow to air

Liquid is preferred because it runs far cooler (Phase 3: ~58 °C at full load vs.
air's ~82 °C), so air is used only as a fallback when liquid is at capacity.
``liquid_util_cap`` sets how full each liquid slot may run before spilling
(1.0 = fill to compute capacity; a lower value holds thermal margin and spills
to air sooner). Each zone's assigned total is spread evenly across its slots.

The router is a pure placement layer: it returns per-rack utilization arrays and
touches neither the thermal substrate nor the cooling/workload agents.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RoutingConfig:
    """Tunable parameters for the cross-zone router (all from config)."""

    liquid_util_cap: float = 1.0   # max per-slot utilization on liquid before spill

    def __post_init__(self) -> None:
        if not 0.0 < self.liquid_util_cap <= 1.0:
            raise ValueError(
                f"liquid_util_cap must be in (0, 1], got {self.liquid_util_cap}"
            )

    @classmethod
    def from_config(cls, cfg: dict) -> "RoutingConfig":
        """Build from the ``routing`` section of the config dict."""
        return cls(liquid_util_cap=float(cfg.get("liquid_util_cap", 1.0)))


@dataclass(frozen=True)
class RoutingDecision:
    """One routing output: how this tick's demand was placed across zones."""

    u_liquid: FloatArray   # per-slot utilization for the liquid rack
    u_air: FloatArray      # per-slot utilization for the air rack
    demand_total: float    # total cluster demand offered this tick
    liquid_total: float    # load placed on liquid
    air_total: float       # load spilled to air
    dropped: float         # load that fit in neither zone (cluster overload)


def split_liquid_first(
    demand_total: float, liquid_capacity: float, air_capacity: float
) -> tuple[float, float, float]:
    """Liquid-first split of a demand total across two pools (conserving load).

    Fills liquid up to its capacity, spills the overflow to air up to air's
    capacity, and reports any remainder as ``dropped`` (cluster overload).
    Returns ``(liquid_total, air_total, dropped)``. Shared by the single-pair
    router and the room-scale composition so the policy lives in one place.
    """
    liquid_total = min(demand_total, liquid_capacity)
    air_total = demand_total - liquid_total
    dropped = max(0.0, air_total - air_capacity)
    air_total = min(air_total, air_capacity)
    return liquid_total, air_total, dropped


def _distribute_evenly(total: float, n_slots: int, cap: float) -> FloatArray:
    """Spread ``total`` evenly across ``n_slots``, each slot capped at ``cap``.

    Assumes ``total <= n_slots * cap`` (the caller guarantees this); any tiny
    floating-point overshoot is clipped.
    """
    if total <= 0.0:
        return np.zeros(n_slots, dtype=np.float64)
    per_slot = total / n_slots
    return np.clip(np.full(n_slots, per_slot, dtype=np.float64), 0.0, cap)


class CrossZoneRouter:
    """Liquid-first cross-zone placement over a 2-rack air+liquid cluster."""

    def __init__(self, config: RoutingConfig, n_slots: int) -> None:
        self.config = config
        self.n_slots = int(n_slots)
        # Cumulative accounting (conservation / spill diagnostics).
        self.total_demand: float = 0.0
        self.total_liquid: float = 0.0
        self.total_air: float = 0.0
        self.total_dropped: float = 0.0

    def route(self, demand: ArrayLike) -> RoutingDecision:
        """Place this tick's per-slot ``demand`` across liquid then air.

        ``demand`` is the incoming cluster demand (per-slot utilization from the
        trace/burstiness layer). Returns per-rack utilization arrays whose total
        equals the offered demand (minus any ``dropped`` cluster overload).
        """
        d = np.asarray(demand, dtype=np.float64)
        demand_total = float(d.sum())

        liquid_capacity = self.n_slots * self.config.liquid_util_cap
        air_capacity = float(self.n_slots)  # air slots cap at 1.0

        liquid_total, air_total, dropped = split_liquid_first(
            demand_total, liquid_capacity, air_capacity
        )

        u_liquid = _distribute_evenly(liquid_total, self.n_slots,
                                      self.config.liquid_util_cap)
        u_air = _distribute_evenly(air_total, self.n_slots, 1.0)

        self.total_demand += demand_total
        self.total_liquid += liquid_total
        self.total_air += air_total
        self.total_dropped += dropped

        return RoutingDecision(
            u_liquid=u_liquid,
            u_air=u_air,
            demand_total=demand_total,
            liquid_total=liquid_total,
            air_total=air_total,
            dropped=dropped,
        )

    @classmethod
    def from_config(cls, config: dict, n_slots: int | None = None) -> "CrossZoneRouter":
        """Build a CrossZoneRouter from a parsed config dict and the rack size."""
        if n_slots is None:
            n_slots = int(config["rack"]["n_slots"])
        return cls(RoutingConfig.from_config(config.get("routing", {})), n_slots)

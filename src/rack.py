"""Rack abstraction: N vertical slots belonging to a cooling zone.

A `CoolingZone` owns the cooling-side parameters (heat-removal coefficient,
supply temperature, fan model) and a `kind` discriminator. Today we only
instantiate one air-cooled zone, but the abstraction is deliberately generic
so a liquid-cooled zone can be added later purely through config — nothing
zone-specific is hardcoded in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .power import PowerModel
from .thermal import ThermalParams, thermal_step

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CoolingZone:
    """Cooling parameters shared by every rack in the zone.

    Each zone owns its own ``alpha`` (vertical recirculation strength): air
    has a hot exhaust column rising through the rack (``alpha > 0``), while
    liquid removes heat at the component and has near-zero recirculation.
    ``omega_fan`` is the actuation multiplier — a fan for air, a pump for
    liquid — but the interface is identical so control works across zones.
    """

    name: str
    kind: str  # "air" or "liquid" — discriminator only, no Python branching
    K_cool: float
    T_supply: float
    alpha: float
    omega_fan: float = 1.0

    @classmethod
    def from_config(cls, cfg: dict) -> "CoolingZone":
        """Build a CoolingZone from one entry of the `zones` config section."""
        return cls(
            name=str(cfg["name"]),
            kind=str(cfg["kind"]),
            K_cool=float(cfg["K_cool"]),
            T_supply=float(cfg["T_supply"]),
            alpha=float(cfg["alpha"]),
            omega_fan=float(cfg.get("omega_fan", 1.0)),
        )


@dataclass
class Rack:
    """A vertical stack of `n_slots` thermal nodes in a cooling zone.

    Slot 0 is the bottom of the rack; slot `n_slots - 1` is the top. State is
    held in `t_node` and advanced one timestep at a time via `step`.
    """

    n_slots: int
    zone: CoolingZone
    power_model: PowerModel
    thermal: ThermalParams
    dt: float
    initial_temp: float = 22.0
    t_node: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset every node back to the initial temperature."""
        self.t_node = np.full(self.n_slots, self.initial_temp, dtype=np.float64)

    def power(self, utilization: ArrayLike) -> FloatArray:
        """Per-slot power (watts) for a scalar or per-slot utilization."""
        u = np.broadcast_to(
            np.asarray(utilization, dtype=np.float64), (self.n_slots,)
        )
        return self.power_model.power(u)

    def step(self, utilization: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Advance the rack one timestep under the given utilization.

        `utilization` may be a scalar (applied to every slot) or a per-slot
        array. Returns (t_node, t_inlet, power) after the update.
        """
        p = self.power(utilization)
        self.t_node, t_inlet = thermal_step(
            self.t_node,
            p,
            params=self.thermal,
            k_cool=self.zone.K_cool,
            t_supply=self.zone.T_supply,
            alpha=self.zone.alpha,
            omega_fan=self.zone.omega_fan,
            dt=self.dt,
        )
        return self.t_node, t_inlet, p

    @property
    def tau(self) -> float:
        """First-order thermal time constant [s]: C / (K_cool * omega_fan)."""
        return self.thermal.C / (self.zone.K_cool * self.zone.omega_fan)

    @classmethod
    def from_config(cls, config: dict, zone_key: str = "air") -> "Rack":
        """Construct a Rack from a parsed config dict and a zone key."""
        zone = CoolingZone.from_config(config["zones"][zone_key])
        return cls(
            n_slots=int(config["rack"]["n_slots"]),
            zone=zone,
            power_model=PowerModel.from_config(config["power"]),
            thermal=ThermalParams.from_config(config["thermal"]),
            dt=float(config["sim"]["dt"]),
            initial_temp=float(config["rack"].get("initial_temp", zone.T_supply)),
        )

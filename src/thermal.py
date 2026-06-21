"""Lumped RC thermal model with vertical recirculation coupling.

Each slot is a single thermal node updated with explicit (forward Euler)
integration at a fixed timestep `dt`:

    T_node[n, t+1] = T_node[n, t]
        + ( P[n, t] * K_heat
            - (T_node[n, t] - T_inlet[n, t]) * K_cool * omega_fan ) * dt / C

The recirculation coupling is the heart of the model: warm exhaust from lower
slots is partially re-ingested by the slots above them, so the inlet seen by
slot `n` rises with the accumulated warmth of every slot below it:

    T_inlet[n, t] = T_supply + alpha * sum_{m < n} max(0, T_node[m, t] - T_supply)

This produces a monotonic bottom-to-top temperature gradient under uniform load.
Slot index 0 is the bottom of the rack; index N-1 is the top.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ThermalParams:
    """Per-node RC parameters shared across a rack.

    Note: ``alpha`` (recirculation strength) is NOT here — it is a property of
    the cooling zone, not the node, because liquid cooling removes heat at the
    component and has little or no rising exhaust-air column. It lives on
    :class:`~src.rack.CoolingZone` and is passed into :func:`thermal_step`.
    """

    K_heat: float = 1.0
    C: float = 540.0

    @classmethod
    def from_config(cls, cfg: dict) -> "ThermalParams":
        """Build ThermalParams from the `thermal` section of the config dict."""
        return cls(
            K_heat=float(cfg["K_heat"]),
            C=float(cfg["C"]),
        )


def compute_inlet(
    t_node: FloatArray,
    t_supply: float,
    alpha: float,
) -> FloatArray:
    """Inlet temperature per slot given current node temps (recirculation).

    Vectorized across the rack. For each slot `n` the inlet is the supply
    temperature plus a fraction `alpha` of the cumulative positive warmth of
    all slots strictly below it (m < n).
    """
    excess = np.maximum(0.0, t_node - t_supply)
    # Exclusive prefix sum: below[n] = sum_{m < n} excess[m].
    below = np.empty_like(excess)
    below[0] = 0.0
    np.cumsum(excess[:-1], out=below[1:])
    return t_supply + alpha * below


def thermal_step(
    t_node: FloatArray,
    power: FloatArray,
    *,
    params: ThermalParams,
    k_cool: float,
    t_supply: float,
    alpha: float,
    omega_fan: float,
    dt: float,
) -> tuple[FloatArray, FloatArray]:
    """Advance node temperatures by one timestep.

    ``alpha`` (recirculation strength) is a per-zone parameter supplied by the
    caller. Returns the new node temperatures and the inlet temperatures that
    were used for this step (handy for logging/validation).
    """
    t_inlet = compute_inlet(t_node, t_supply, alpha)
    heat_in = power * params.K_heat
    heat_out = (t_node - t_inlet) * k_cool * omega_fan
    t_next = t_node + (heat_in - heat_out) * dt / params.C
    return t_next, t_inlet

"""Utilization -> power (watts).

Uses the OpenDC / SPECpower-style curve so the model is grounded in published
server power data and can be cross-checked later:

    P(u) = p_idle + (p_max - p_idle) * (2*u - u**r)

The returned value is the node's total electrical draw, which equals its heat
input. Idle consumption is already included in the curve, so callers must NOT
add a separate idle term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PowerModel:
    """SPECpower-style utilization -> power curve."""

    p_idle: float = 120.0
    p_max: float = 400.0
    r: float = 1.4

    def power(self, utilization: ArrayLike) -> NDArray[np.float64]:
        """Return power in watts for utilization(s) in [0, 1].

        Accepts a scalar or array; utilization is clipped to [0, 1] so that
        out-of-range inputs degrade gracefully instead of producing nonsense.
        """
        u = np.clip(np.asarray(utilization, dtype=np.float64), 0.0, 1.0)
        return self.p_idle + (self.p_max - self.p_idle) * (2.0 * u - u**self.r)

    @classmethod
    def from_config(cls, cfg: dict) -> "PowerModel":
        """Build a PowerModel from the `power` section of the config dict."""
        return cls(
            p_idle=float(cfg["p_idle"]),
            p_max=float(cfg["p_max"]),
            r=float(cfg["r"]),
        )

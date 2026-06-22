"""Cooling-energy / PUE accounting (Phase 7).

A thin accounting layer on top of the substrate — it changes no physics. Two
power streams are tracked per rack per tick:

* **IT power** — the useful compute draw, the sum of the SPECpower curve over the
  rack's slots at their effective utilization (already computed by ``Rack.step``).
* **Cooling power** — the fan (air) or pump (liquid) electrical draw, modelled by
  the affinity (fan) law ``P_cool = rated_w[kind] * omega ** exponent``. With the
  default cubic exponent, driving a fan to omega = 2.5 costs ``2.5**3 ≈ 15.6×``
  its rated power — which is why maxing fans is expensive and why routing work to
  the efficient liquid zone (low rated pump power, rarely ramped) saves energy.

Facility power = IT + cooling; **PUE = facility / IT**. Energy is the time
integral (``power * dt`` summed over ticks). Cooling rated powers are per zone
*kind* and come from config, so adding a new cooling technology is config-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# Illustrative defaults (tunable in config). Per-rack, at omega = 1:
# a ~6 kW rack (15 slots × 400 W) gets ~500 W air fans (~8 % overhead, PUE ~1.08)
# or ~150 W liquid pump (~2.5 %). The cubic law makes a maxed air fan (omega 2.5)
# draw ~7.8 kW — more than the IT itself — so emergencies are costly.
DEFAULT_RATED_W = {"air": 500.0, "liquid": 150.0}
DEFAULT_EXPONENT = 3.0


@dataclass(frozen=True)
class EnergyModel:
    """Maps actuation (omega) and IT draw to cooling power, energy, and PUE."""

    rated_w: Mapping[str, float]      # per zone kind: cooling power at omega = 1
    exponent: float = DEFAULT_EXPONENT  # affinity-law exponent (3 = cubic)

    def cooling_power_w(self, kind: str, omega: float) -> float:
        """Cooling (fan/pump) electrical power [W] for a rack at this omega."""
        if kind not in self.rated_w:
            raise KeyError(f"no cooling rated power configured for zone kind {kind!r}")
        return self.rated_w[kind] * (omega ** self.exponent)

    @staticmethod
    def pue(it_w: float, cooling_w: float) -> float:
        """Power Usage Effectiveness = (IT + cooling) / IT. 1.0 is ideal."""
        if it_w <= 0.0:
            return float("nan")
        return (it_w + cooling_w) / it_w

    @classmethod
    def from_config(cls, config: dict) -> "EnergyModel":
        """Build from the ``energy`` section of the config dict."""
        e = config.get("energy", {})
        rated = e.get("rated_w", DEFAULT_RATED_W)
        return cls(
            rated_w={str(k): float(v) for k, v in rated.items()},
            exponent=float(e.get("exponent", DEFAULT_EXPONENT)),
        )

"""Cooling Agent (Phase 4) — predictive, zoned fan control with hysteresis.

The agent regulates a single rack by reading its temperature and writing the
zone's ``omega_fan`` actuation multiplier. It is *predictive*: decisions are made
on a short-horizon temperature forecast (the existing thermal model rolled
forward under the current load), not on the instantaneous temperature.

State machine, evaluated on the predicted hottest-slot temperature:

* **Safe**    (< warn):            base fan speed (omega_fan = 1.0).
* **Warning** (warn .. critical):  fan ramps linearly 1.0 -> fan_max.
* **Critical** (> critical):       assert THROTTLE_REQUEST, hold fan at fan_max.

Hysteresis latches the critical state: once entered, it holds until the
predicted temperature falls below ``release_c`` (72 °C), *not* merely back below
``critical_c`` (80 °C) — this prevents fan/throttle chatter at the 80 °C edge.

Hard boundary (Phase 4): THROTTLE_REQUEST is decided, exposed, and logged, but
it has **no effect on load**. Acting on it (shedding/deferring work) is the
Workload Agent's job in a later phase. The agent's only real actuation this phase
is ``omega_fan``, which feeds the thermal model and closes the cooling loop. The
throttle flag feeds nothing.

The agent holds no rack reference and no "air"/single-rack assumptions: the rack
is passed in per call, and per-rack latch state lives in the instance, so a
second instance can regulate another rack/zone unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .rack import Rack
from .thermal import thermal_step

# State labels (the critical state is the one that asserts the throttle).
SAFE = "safe"
WARNING = "warning"
CRITICAL = "critical"


@dataclass(frozen=True)
class CoolingAgentConfig:
    """Tunable parameters for the Cooling Agent (all from config)."""

    horizon_s: float = 15.0     # prediction look-ahead [s]
    warn_c: float = 75.0        # safe -> warning threshold
    critical_c: float = 80.0    # warning -> critical (throttle) threshold
    fan_max: float = 2.5        # omega_fan at/above the critical edge
    release_c: float = 72.0     # hysteresis release: latch clears below this
    fan_base: float = 1.0       # omega_fan in the safe zone

    @classmethod
    def from_config(cls, cfg: dict) -> "CoolingAgentConfig":
        """Build from the `cooling_agent` section of the config dict."""
        return cls(
            horizon_s=float(cfg["horizon_s"]),
            warn_c=float(cfg["warn_c"]),
            critical_c=float(cfg["critical_c"]),
            fan_max=float(cfg["fan_max"]),
            release_c=float(cfg["release_c"]),
            fan_base=float(cfg.get("fan_base", 1.0)),
        )


@dataclass(frozen=True)
class CoolingDecision:
    """One control output: what the agent decided for this timestep."""

    omega_fan: float          # actuation written to the rack (real effect)
    throttle_request: bool    # THROTTLE_REQUEST flag (logged only this phase)
    state: str                # SAFE / WARNING / CRITICAL
    predicted_temp: float     # the forecast the decision was made on


class CoolingAgent:
    """Predictive zoned fan controller with a latched critical state."""

    def __init__(self, config: CoolingAgentConfig) -> None:
        self.config = config
        self._latched = False  # per-rack hysteresis latch for the critical state

    @property
    def latched(self) -> bool:
        """Whether the critical/throttle state is currently latched."""
        return self._latched

    def predict_temperature(self, rack: Rack, utilization) -> float:
        """Forecast the hottest-slot temperature ``horizon_s`` ahead.

        Rolls the existing thermal model forward on a *copy* of the rack state,
        holding the current load and the rack's current ``omega_fan`` constant.
        Does not mutate the rack.
        """
        t_node = rack.t_node.copy()
        power = rack.power(utilization)  # held constant over the horizon
        steps = max(1, int(round(self.config.horizon_s / rack.dt)))
        for _ in range(steps):
            t_node, _ = thermal_step(
                t_node,
                power,
                params=rack.thermal,
                k_cool=rack.zone.K_cool,
                t_supply=rack.zone.T_supply,
                alpha=rack.zone.alpha,
                omega_fan=rack.zone.omega_fan,
                dt=rack.dt,
            )
        return float(t_node.max())

    def decide(self, predicted_temp: float) -> CoolingDecision:
        """Pure decision from a predicted temperature + the latch state.

        Updates the hysteresis latch, then maps to (omega_fan, throttle, state).
        Separated from prediction/actuation so it is trivially unit-testable.
        """
        cfg = self.config

        # Update the latch: set above critical, clear below release, else hold.
        if predicted_temp > cfg.critical_c:
            self._latched = True
        elif predicted_temp < cfg.release_c:
            self._latched = False

        if self._latched:
            # Critical state held by hysteresis: max fans + throttle asserted.
            return CoolingDecision(cfg.fan_max, True, CRITICAL, predicted_temp)

        if predicted_temp < cfg.warn_c:
            return CoolingDecision(cfg.fan_base, False, SAFE, predicted_temp)

        # Warning band: ramp fan linearly from base (at warn) to max (at critical).
        span = cfg.critical_c - cfg.warn_c
        frac = 0.0 if span <= 0 else (predicted_temp - cfg.warn_c) / span
        frac = min(max(frac, 0.0), 1.0)
        omega = cfg.fan_base + frac * (cfg.fan_max - cfg.fan_base)
        return CoolingDecision(omega, False, WARNING, predicted_temp)

    def control(self, rack: Rack, utilization) -> CoolingDecision:
        """Predict, decide, and write ``omega_fan`` to the rack.

        Returns the decision (including the logged-only throttle flag). The
        caller then advances the substrate with ``rack.step(utilization)``; the
        new fan setting takes effect on that step. The throttle flag is NOT
        applied to load anywhere — that is the Workload Agent's job later.
        """
        predicted = self.predict_temperature(rack, utilization)
        decision = self.decide(predicted)
        # Write the fan via the existing actuation hook (zone is frozen).
        rack.zone = replace(rack.zone, omega_fan=decision.omega_fan)
        return decision

    @classmethod
    def from_config(cls, config: dict) -> "CoolingAgent":
        """Build a CoolingAgent from a parsed config dict."""
        return cls(CoolingAgentConfig.from_config(config["cooling_agent"]))

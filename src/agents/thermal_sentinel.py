"""Thermal Sentinel Agent.

Wraps the existing Rack + CoolingAgent from src/ and adds an ML predictor.
Manages one Rack per call; the SimulationRunner instantiates one Sentinel
per rack and calls step() each tick.

In BASELINE mode: CoolingAgent is NOT used (omega_fan stays at 1.0).
In MULTI-AGENT mode: CoolingAgent provides predictive fan control.
"""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rack import Rack
from src.cooling_agent import CoolingAgent, CoolingDecision, SAFE, WARNING, CRITICAL

try:
    from sklearn.linear_model import Ridge
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class ThermalSentinel:
    """Thermal Sentinel — per-rack thermal management + ML prediction.

    Wraps a Rack and its CoolingAgent. On each tick:
      1. (multi-agent only) CoolingAgent.control() adjusts omega_fan.
      2. Rack.step() advances the thermal model one timestep.
      3. Appends to history buffer.
      4. Trains/updates Ridge regressor when enough history exists.
    """

    HISTORY_LEN = 300
    ML_WARMUP = 60

    def __init__(self, rack: Rack, agent: CoolingAgent, rack_id: str) -> None:
        self.rack = rack
        self.agent = agent
        self.rack_id = rack_id

        # History: list of (util, omega_fan, peak_temp)
        self._history: Deque[Tuple[float, float, float]] = deque(maxlen=self.HISTORY_LEN)
        self._ml_model: Optional[object] = None
        self._ml_trained: bool = False

        # State
        self.last_decision: Optional[CoolingDecision] = None
        self.peak_temp: float = rack.initial_temp
        self.avg_temp: float = rack.initial_temp
        self.slot_temps: List[float] = [rack.initial_temp] * rack.n_slots
        self.slot_utils: List[float] = [0.0] * rack.n_slots
        self.slot_powers: List[float] = [0.0] * rack.n_slots
        self.ml_prediction: float = rack.initial_temp

    def step(self, utilization: np.ndarray, multi_agent: bool) -> CoolingDecision:
        """Advance one thermal step. Returns CoolingDecision."""
        if multi_agent:
            decision = self.agent.control(self.rack, utilization)
        else:
            # Baseline: fixed fan, no agent
            decision = CoolingDecision(
                omega_fan=1.0,
                throttle_request=False,
                state=SAFE,
                predicted_temp=self.rack.t_node.max(),
            )
            self.rack.zone = replace(self.rack.zone, omega_fan=1.0)

        t_node, t_inlet, power = self.rack.step(utilization)

        self.slot_temps = [round(float(t), 2) for t in t_node]
        self.slot_utils = [round(float(u), 3) for u in utilization]
        self.slot_powers = [round(float(p), 1) for p in power]
        self.peak_temp = float(t_node.max())
        self.avg_temp = float(t_node.mean())
        self.last_decision = decision

        # Update ML history
        self._history.append((float(utilization.mean()), decision.omega_fan, self.peak_temp))
        self._update_ml()

        return decision

    def _update_ml(self) -> None:
        """Retrain Ridge regressor on rolling history."""
        if not _SKLEARN_AVAILABLE:
            self.ml_prediction = self.last_decision.predicted_temp if self.last_decision else self.peak_temp
            return
        n = len(self._history)
        if n < self.ML_WARMUP + 15:
            self.ml_prediction = round(self.last_decision.predicted_temp, 1) if self.last_decision else self.peak_temp
            return
        hist = list(self._history)
        # Features: [util, omega_fan, temp] at time t → target: temp at t+15
        X, y = [], []
        horizon = 15
        for i in range(len(hist) - horizon):
            u, fan, temp = hist[i]
            _, _, future_temp = hist[i + horizon]
            X.append([u, fan, temp])
            y.append(future_temp)
        X_arr = np.array(X)
        y_arr = np.array(y)
        if self._ml_model is None:
            self._ml_model = Ridge(alpha=1.0)
        self._ml_model.fit(X_arr, y_arr)
        self._ml_trained = True

        # Predict from latest state
        u, fan, temp = hist[-1]
        pred = float(self._ml_model.predict([[u, fan, temp]])[0])
        self.ml_prediction = round(pred, 1)

    def to_dict(self) -> dict:
        dec = self.last_decision
        return {
            "rack_id": self.rack_id,
            "state": dec.state if dec else SAFE,
            "predicted_temp": round(dec.predicted_temp, 1) if dec else round(self.peak_temp, 1),
            "ml_prediction": self.ml_prediction,
            "ml_trained": self._ml_trained,
            "peak_temp": round(self.peak_temp, 2),
            "avg_temp": round(self.avg_temp, 2),
            "omega_fan": round(dec.omega_fan, 3) if dec else 1.0,
            "throttle_request": dec.throttle_request if dec else False,
        }

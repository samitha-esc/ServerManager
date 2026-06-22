"""Arbiter Agent — the negotiation engine.

Resolves the conflict between Scheduler (wants maximum throughput) and
Thermal Sentinel (wants low temperature).

When any rack's Thermal Sentinel predicts > critical_c (80°C):
  1. Arbiter sends a THROTTLE_REQUEST signal to the Scheduler.
  2. Scheduler performs greedy shedding: drops P3 first, then partial P2.
  3. Arbiter logs the negotiation event.

When temperatures recover below release_c (72°C):
  1. Arbiter releases the throttle.
  2. Scheduler restores full traffic.

In BASELINE mode: Arbiter does nothing — no shedding, no negotiation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional


@dataclass
class ArbiterConfig:
    critical_c: float = 80.0
    release_c: float = 72.0
    p3_shed: float = 1.0    # fraction of P3 traffic dropped
    p2_shed: float = 0.5    # fraction of P2 traffic dropped


class ArbiterAgent:
    """Negotiation engine — mediates between Scheduler and Sentinel."""

    def __init__(self, config: ArbiterConfig) -> None:
        self.config = config
        self._throttle_active: bool = False
        self._log: Deque[str] = deque(maxlen=50)
        self.shed_pct: float = 0.0
        self.negotiation_status: str = "NOMINAL"

    def evaluate(
        self,
        sentinel_states: Dict[str, dict],  # rack_id → sentinel.to_dict()
        scheduler,                          # SchedulerAgent instance
        tick: int,
        multi_agent: bool,
    ) -> List[str]:
        """Evaluate thermal state and issue/release throttle as needed.

        Returns list of new log lines generated this tick.
        """
        if not multi_agent:
            # Baseline: always nominal, never throttle
            if self._throttle_active:
                scheduler.release_throttle()
                self._throttle_active = False
            self.negotiation_status = "BASELINE-MODE"
            self.shed_pct = 0.0
            return []

        logs: List[str] = []
        # Find hottest predicted temperature across all racks
        max_predicted = max(
            (s["predicted_temp"] for s in sentinel_states.values()),
            default=22.0,
        )
        hottest_rack = max(
            sentinel_states, key=lambda r: sentinel_states[r]["predicted_temp"],
            default="?",
        )

        if not self._throttle_active and max_predicted > self.config.critical_c:
            # Assert throttle
            self._throttle_active = True
            scheduler.apply_throttle(
                p3_shed=self.config.p3_shed,
                p2_shed=self.config.p2_shed,
            )
            self.shed_pct = self.config.p3_shed * 100
            self.negotiation_status = "THROTTLE-ACTIVE"
            msg = (
                f"[t={tick:05d}] [ARBITER] → [SCHEDULER]: THROTTLE_REQUEST signed. "
                f"Hottest rack {hottest_rack} predicted {max_predicted:.1f}°C. "
                f"Shedding P3={self.config.p3_shed*100:.0f}% "
                f"P2={self.config.p2_shed*100:.0f}%"
            )
            logs.append(msg)
            self._log.append(msg)

        elif self._throttle_active and max_predicted < self.config.release_c:
            # Release throttle
            self._throttle_active = False
            scheduler.release_throttle()
            self.shed_pct = 0.0
            self.negotiation_status = "NOMINAL"
            msg = (
                f"[t={tick:05d}] [ARBITER] → [SCHEDULER]: THROTTLE released. "
                f"All racks predicted below {self.config.release_c:.0f}°C."
            )
            logs.append(msg)
            self._log.append(msg)

        elif self._throttle_active:
            # Holding
            self.negotiation_status = f"THROTTLE-HOLD ({max_predicted:.1f}°C)"

        else:
            self.negotiation_status = "NOMINAL"

        # Emit periodic Sentinel updates to log
        if tick % 30 == 0:
            hottest = sentinel_states.get(hottest_rack, {})
            state = hottest.get("state", "safe").upper()
            msg = (
                f"[t={tick:05d}] [SENTINEL] {hottest_rack}: "
                f"actual {hottest.get('peak_temp',0):.1f}°C "
                f"| predicted {max_predicted:.1f}°C "
                f"| ML {hottest.get('ml_prediction',0):.1f}°C "
                f"| state={state}"
            )
            logs.append(msg)
            self._log.append(msg)

        return logs

    def get_log(self) -> List[str]:
        """Return recent negotiation log lines (newest first)."""
        return list(reversed(self._log))

    def to_dict(self) -> dict:
        return {
            "throttle_active": self._throttle_active,
            "negotiation_status": self.negotiation_status,
            "shed_pct": round(self.shed_pct, 1),
            "recent_log": list(self._log)[-5:],
        }

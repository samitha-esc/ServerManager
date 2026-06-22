"""Router Agent.

Assigns incoming packet batches to rack/slot targets.

MULTI-AGENT mode — Least Thermal Intensity (LTI) Routing:
  Sorts racks by current peak temperature (ascending) and assigns packets
  to the coolest available rack first. This actively steers heat away from
  hot zones and is the core routing intelligence.

BASELINE mode — Round-Robin:
  Distributes load evenly across all racks regardless of temperature.
  No thermal awareness — produces uneven heating and more violations.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Packet:
    id: str
    priority: int    # 0=Critical, 1=High, 2=Medium, 3=Low
    target_rack: str = ""
    x: float = 0.0  # animation position (0-1 along the path)
    lane_y: float = 0.0  # y offset within the traffic lane
    color: str = "#ef4444"

PRIORITY_COLORS = {0: "#ef4444", 1: "#f59e0b", 2: "#14b8a6", 3: "#6b7280"}
LOAD_PER_PACKET = 0.08   # each packet contributes this fraction of load


class RouterAgent:
    """Router — maps incoming packets to racks using LTI or Round-Robin."""

    def __init__(self, rack_ids: List[str]) -> None:
        self.rack_ids = rack_ids
        self._rr_cycle = itertools.cycle(rack_ids)
        self._packet_counter = 0
        self.routed_total: int = 0
        self.current_policy: str = "ROUND-ROBIN"
        # Live packet list for animation (only the most recent N)
        self.packets: List[Packet] = []

    def route(
        self,
        n_packets: int,
        priority_counts: Dict[int, int],
        peak_temps: Dict[str, float],
        multi_agent: bool,
    ) -> List[Packet]:
        """Assign packets to racks. Returns new Packet objects."""
        if multi_agent:
            self.current_policy = "DESCENDING-THERMAL (LTI)"
            ordered = sorted(self.rack_ids, key=lambda r: peak_temps.get(r, 99.0))
        else:
            self.current_policy = "ROUND-ROBIN"
            ordered = self.rack_ids

        new_packets: List[Packet] = []
        rack_cycle = itertools.cycle(ordered)

        for priority, count in priority_counts.items():
            for _ in range(count):
                self._packet_counter += 1
                target = next(rack_cycle)
                pkt = Packet(
                    id=f"pkt_{self._packet_counter}",
                    priority=priority,
                    target_rack=target,
                    x=0.0,
                    lane_y=random.uniform(0.15, 0.85),
                    color=PRIORITY_COLORS[priority],
                )
                new_packets.append(pkt)

        self.routed_total += len(new_packets)
        # Advance existing packets; cull finished ones; add new ones
        surviving = [p for p in self.packets if p.x < 1.0]
        for p in surviving:
            p.x = min(1.0, p.x + 0.07)
        self.packets = surviving + new_packets
        return new_packets

    def to_dict(self) -> dict:
        return {
            "policy": self.current_policy,
            "routed_total": self.routed_total,
            "in_flight": len(self.packets),
            "packets": [
                {"id": p.id, "priority": p.priority, "target": p.target_rack,
                 "x": round(p.x, 3), "lane_y": round(p.lane_y, 3), "color": p.color}
                for p in self.packets[-40:]  # send at most 40 to frontend
            ],
        }

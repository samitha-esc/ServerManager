"""Datacenter topology — 5-Floor × 10-Rack layout.

50 racks arranged as:
  Floors 1-3: air-cooled   (30 racks)  — rows labeled F1-F3
  Floors 4-5: liquid-cooled (20 racks) — rows labeled F4-F5

Each floor has 10 racks arranged in a single horizontal row on the canvas.
Network: one leaf switch per floor connected to two spine switches (replaced
by the floor concept in the UI, but kept internally for packet routing logic).

Canvas: 1000 × 700.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RackDef:
    id: str
    kind: str          # "air" or "liquid"
    floor: int         # 1-5
    position: int      # position within floor, 1-10
    leaf: str          # which leaf switch (one per floor)
    x: float           # canvas x (0-1000)
    y: float           # canvas y (0-700)
    zone_key: str = "air"

    @property
    def aisle(self) -> str:
        """Backwards-compat alias used by simulation_runner."""
        return f"F{self.floor}"


@dataclass
class SwitchDef:
    id: str
    kind: str   # "leaf" or "floor"
    x: float
    y: float
    connects_to: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layout — 5 Floors × 10 Racks
# ---------------------------------------------------------------------------
#
#  Floor 1 (air)    — y ≈ 170  — R01-R10
#  Floor 2 (air)    — y ≈ 280  — R11-R20
#  Floor 3 (air)    — y ≈ 390  — R21-R30
#  Floor 4 (liquid) — y ≈ 500  — R31-R40
#  Floor 5 (liquid) — y ≈ 600  — R41-R50
#
# ---------------------------------------------------------------------------

FLOOR_CONFIG = [
    # (floor_num, kind,     zone_key,  canvas_y)
    (1, "air",    "air",    170),
    (2, "air",    "air",    280),
    (3, "air",    "air",    390),
    (4, "liquid", "liquid", 490),
    (5, "liquid", "liquid", 590),
]

RACKS: List[RackDef] = []
SWITCHES: List[SwitchDef] = []

# Build floor leaf switches and racks
_rack_num = 1
for floor_num, kind, zone_key, fy in FLOOR_CONFIG:
    leaf_id = f"FLOOR-{floor_num}"
    leaf_x = 50.0
    SWITCHES.append(SwitchDef(leaf_id, "floor", leaf_x, fy))

    for pos in range(1, 11):
        rack_id = f"R{_rack_num:02d}"
        # Distribute 10 racks across x=120 to x=980
        rx = 120 + (pos - 1) * 95
        RACKS.append(RackDef(
            id=rack_id, kind=kind, floor=floor_num, position=pos,
            leaf=leaf_id, x=rx, y=fy, zone_key=zone_key
        ))
        _rack_num += 1

# Maps for quick lookup
RACK_MAP = {r.id: r for r in RACKS}


def get_edges():
    """Return (src_id, dst_id) pairs for leaf ↔ rack connections."""
    edges = []
    for rack in RACKS:
        edges.append((rack.leaf, rack.id))
    return edges


def get_node_positions():
    positions = {}
    for sw in SWITCHES:
        positions[sw.id] = {"x": sw.x, "y": sw.y, "kind": sw.kind}
    for rack in RACKS:
        positions[rack.id] = {"x": rack.x, "y": rack.y, "kind": rack.kind}
    return positions


def to_topology_dict():
    """Serialisable summary for the frontend."""
    return {
        "nodes": [
            {
                "id": r.id, "kind": r.kind,
                "floor": r.floor, "position": r.position,
                "aisle": r.aisle, "leaf": r.leaf,
                "x": r.x, "y": r.y,
            }
            for r in RACKS
        ] + [
            {"id": s.id, "kind": s.kind, "x": s.x, "y": s.y}
            for s in SWITCHES
        ],
        "edges": get_edges(),
        "floors": [
            {
                "floor": fn, "kind": kind, "zone_key": zk, "y": fy,
                "label": f"FLOOR {fn} — {'AIR' if kind == 'air' else 'LIQUID'} COOLING",
            }
            for fn, kind, zk, fy in FLOOR_CONFIG
        ],
    }

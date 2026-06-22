"""Datacenter topology definition for THERMOS simulation.

Defines a 10-rack facility with a standard Leaf-Spine network:
  - 7 air-cooled racks  (R01-R07) across Aisles A and B
  - 3 liquid-cooled racks (R08-R10) in Aisle C
  - 3 leaf switches (one per aisle) + 2 spine switches

All (x, y) coordinates are in a 1000×700 canvas space.
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
    aisle: str         # "A", "B", or "C"
    leaf: str          # which leaf switch this rack connects to
    x: float           # canvas x (0-1000)
    y: float           # canvas y (0-700)
    zone_key: str = "air"  # key into sim_config zones block


@dataclass
class SwitchDef:
    id: str
    kind: str   # "leaf" or "spine"
    x: float
    y: float
    connects_to: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layout — Leaf-Spine Topology
# ---------------------------------------------------------------------------
#
#  SPINE-01 (x=350)        SPINE-02 (x=650)
#      |   \             /     |
#   LEAF-A  LEAF-B   LEAF-B  LEAF-C
#  (Aisle A) (Aisle B)      (Aisle C)
#
#  Aisles A+B: air-cooled racks (gold)
#  Aisle C:    liquid-cooled racks (teal)
#
# ---------------------------------------------------------------------------

SWITCHES: List[SwitchDef] = [
    SwitchDef("SPINE-01", "spine", 350, 80,  ["LEAF-A", "LEAF-B", "LEAF-C"]),
    SwitchDef("SPINE-02", "spine", 650, 80,  ["LEAF-A", "LEAF-B", "LEAF-C"]),
    SwitchDef("LEAF-A",   "leaf",  200, 200, ["SPINE-01", "SPINE-02"]),
    SwitchDef("LEAF-B",   "leaf",  500, 200, ["SPINE-01", "SPINE-02"]),
    SwitchDef("LEAF-C",   "leaf",  800, 200, ["SPINE-01", "SPINE-02"]),
]

RACKS: List[RackDef] = [
    # Aisle A — air (4 racks, left cluster)
    RackDef("R01", "air", "A", "LEAF-A",  120, 340, "air"),
    RackDef("R02", "air", "A", "LEAF-A",  180, 300, "air"),
    RackDef("R03", "air", "A", "LEAF-A",  150, 420, "air"),
    RackDef("R04", "air", "A", "LEAF-A",  240, 370, "air"),
    # Aisle B — air (3 racks, center cluster)
    RackDef("R05", "air", "B", "LEAF-B",  460, 310, "air"),
    RackDef("R06", "air", "B", "LEAF-B",  540, 350, "air"),
    RackDef("R07", "air", "B", "LEAF-B",  500, 430, "air"),
    # Aisle C — liquid (3 racks, right cluster)
    RackDef("R08", "liquid", "C", "LEAF-C", 760, 340, "liquid"),
    RackDef("R09", "liquid", "C", "LEAF-C", 840, 300, "liquid"),
    RackDef("R10", "liquid", "C", "LEAF-C", 820, 420, "liquid"),
]

# Maps rack id -> RackDef for quick lookup
RACK_MAP = {r.id: r for r in RACKS}

# Connection edges for rendering (spine ↔ leaf, leaf ↔ rack)
def get_edges():
    """Return list of (src_id, dst_id) pairs for all topology links."""
    edges = []
    for sw in SWITCHES:
        if sw.kind == "spine":
            for leaf_id in sw.connects_to:
                edges.append((sw.id, leaf_id))
    for rack in RACKS:
        edges.append((rack.leaf, rack.id))
    return edges


def get_node_positions():
    """Return dict {id: {x, y, kind}} for all switches and racks."""
    positions = {}
    for sw in SWITCHES:
        positions[sw.id] = {"x": sw.x, "y": sw.y, "kind": sw.kind}
    for rack in RACKS:
        positions[rack.id] = {"x": rack.x, "y": rack.y, "kind": rack.kind}
    return positions


def to_topology_dict():
    """Serialisable summary of the static topology for the frontend."""
    return {
        "nodes": [
            {"id": r.id, "kind": r.kind, "aisle": r.aisle,
             "leaf": r.leaf, "x": r.x, "y": r.y}
            for r in RACKS
        ] + [
            {"id": s.id, "kind": s.kind, "x": s.x, "y": s.y}
            for s in SWITCHES
        ],
        "edges": get_edges(),
    }

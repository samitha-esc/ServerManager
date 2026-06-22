"""Tests for the visualization data layer (Phase 8)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sim_engine import SimulationEngine, generate_room_demand  # noqa: E402


@pytest.fixture
def config() -> dict:
    with open(ROOT / "config" / "sim_config.yaml", "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # Small room keeps the tests fast.
    cfg = {**cfg, "room": {**cfg["room"], "n_air_racks": 4, "n_liquid_racks": 2}}
    return cfg


def test_frame_is_json_serializable(config):
    eng = SimulationEngine(config)
    frame = eng.step(20.0)
    # Must serialize with no custom encoder.
    json.dumps(frame)
    assert frame["t"] == 0
    assert set(frame) == {"t", "room", "racks"}


def test_room_aggregate_fields_present(config):
    eng = SimulationEngine(config)
    room = eng.step(20.0)["room"]
    for key in ("demand", "it_w", "cooling_w", "pue", "tpi", "dropped",
                "throttles", "max_temp_air", "max_temp_liquid"):
        assert key in room


def test_per_rack_records_match_topology(config):
    eng = SimulationEngine(config)
    frame = eng.step(20.0)
    topo = eng.topology()
    assert len(frame["racks"]) == len(topo["racks"]) == 6
    ids_frame = {r["id"] for r in frame["racks"]}
    ids_topo = {r["id"] for r in topo["racks"]}
    assert ids_frame == ids_topo
    for r in frame["racks"]:
        for key in ("id", "kind", "max_temp", "omega", "throttle", "state",
                    "util", "it_w", "cooling_w"):
            assert key in r


def test_slots_optional(config):
    eng = SimulationEngine(config)
    n_slots = config["rack"]["n_slots"]
    no_slots = eng.step(20.0, include_slots=False)
    assert "slot_temps" not in no_slots["racks"][0]
    with_slots = eng.step(20.0, include_slots=True)
    assert len(with_slots["racks"][0]["slot_temps"]) == n_slots


def test_metadata_has_contract_fields(config):
    meta = SimulationEngine(config).metadata(output_stride_s=5)
    assert meta["schema_version"]
    assert meta["output_stride_s"] == 5
    assert meta["thresholds_c"] == {"warn": 75.0, "critical": 80.0, "release": 72.0}
    assert "air" in meta["zones"] and "liquid" in meta["zones"]
    assert "trace" in meta["seeds"]
    json.dumps(meta)  # serializable


def test_tick_advances_and_reset_rewinds(config):
    eng = SimulationEngine(config)
    eng.step(20.0)
    eng.step(20.0)
    assert eng.t == 2
    eng.reset()
    assert eng.t == 0


def test_runs_are_deterministic(config):
    demand = generate_room_demand(config, n_ticks=30, scale=1.5)
    a = SimulationEngine(config)
    b = SimulationEngine(config)
    fa = [a.step(float(d)) for d in demand][-1]
    fb = [b.step(float(d)) for d in demand][-1]
    assert json.dumps(fa) == json.dumps(fb)


def test_generate_room_demand_length_and_positive(config):
    d = generate_room_demand(config, n_ticks=50, scale=1.0)
    assert d.shape[0] == 50
    assert (d >= 0).all()


def test_naive_vs_routed_differ(config):
    demand = generate_room_demand(config, n_ticks=40, scale=2.0)
    routed = SimulationEngine(config, route=True)
    naive = SimulationEngine(config, route=False)
    fr = [routed.step(float(d)) for d in demand][-1]
    fn = [naive.step(float(d)) for d in demand][-1]
    # Routing keeps the hottest air rack cooler than an even split.
    assert fr["room"]["max_temp_air"] <= fn["room"]["max_temp_air"]
    assert routed.metadata()["routing"] != naive.metadata()["routing"]

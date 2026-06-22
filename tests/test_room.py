"""Tests for the server-room composition (Phase 7)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.room import Room, RoomConfig  # noqa: E402


@pytest.fixture
def config() -> dict:
    with open(ROOT / "config" / "sim_config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _small_config(config: dict, n_air=4, n_liquid=4) -> dict:
    cfg = {**config, "room": {**config.get("room", {}),
                              "n_air_racks": n_air, "n_liquid_racks": n_liquid,
                              "demand_scale": 1.0}}
    return cfg


def _run(room: Room, demand: float, ticks: int):
    results = []
    for t in range(ticks):
        results.append(room.step(demand, t))
    return results


# --- Construction -----------------------------------------------------------

def test_room_builds_with_configured_rack_counts(config):
    room = Room(_small_config(config, 6, 3))
    assert len(room.air) == 6
    assert len(room.liquid) == 3
    assert room.n_racks == 9


def test_room_config_from_config(config):
    rc = RoomConfig.from_config(config)
    assert rc.n_air_racks == 35
    assert rc.n_liquid_racks == 15


# --- Stepping / energy accounting -------------------------------------------

def test_step_produces_finite_state_and_valid_pue(config):
    room = Room(_small_config(config))
    res = _run(room, demand=40.0, ticks=50)[-1]
    assert np.isfinite(res.max_temp_air) and np.isfinite(res.max_temp_liquid)
    assert res.it_power_w > 0
    assert res.cooling_power_w > 0
    assert res.pue >= 1.0
    assert room.pue >= 1.0
    assert room.total_energy_j == pytest.approx(room.it_energy_j + room.cooling_energy_j)


def test_routing_conserves_load_within_capacity(config):
    room = Room(_small_config(config, 4, 4), route=True)
    # 4 liquid racks * 15 slots = 60 capacity; demand 50 fits in liquid alone.
    res = room.step(50.0, 0)
    assert res.liquid_load + res.air_load == pytest.approx(50.0, abs=1e-9)
    assert res.dropped == 0.0


def test_liquid_first_keeps_air_idle_at_low_demand(config):
    room = Room(_small_config(config, 4, 4), route=True)
    res = room.step(40.0, 0)  # < liquid capacity 60
    assert res.air_load == pytest.approx(0.0)
    assert res.liquid_load == pytest.approx(40.0)


def test_overflow_spills_to_air(config):
    room = Room(_small_config(config, 4, 4), route=True)
    res = room.step(90.0, 0)  # > liquid capacity 60 -> 30 spills to air
    assert res.liquid_load == pytest.approx(60.0)
    assert res.air_load == pytest.approx(30.0)


# --- The payoff: routing beats naive on temperature AND cooling energy ------

def test_routed_cooler_and_cheaper_than_naive(config):
    cfg = _small_config(config, n_air=4, n_liquid=4)
    demand, ticks = 90.0, 600   # high enough that naive drives air fans to ramp
    routed = Room(cfg, route=True)
    naive = Room(cfg, route=False)
    r_last = _run(routed, demand, ticks)[-1]
    n_last = _run(naive, demand, ticks)[-1]

    # Naive loads air racks evenly with liquid -> hotter air -> fans ramp.
    assert r_last.max_temp_air < n_last.max_temp_air
    # Cube-law fan energy: keeping air cool makes routing cheaper to cool.
    assert routed.cooling_energy_j < naive.cooling_energy_j
    # Both keep liquid safe.
    assert r_last.max_temp_liquid < 75.0


def test_tpi_property_available(config):
    room = Room(_small_config(config, 3, 3))
    _run(room, demand=30.0, ticks=20)
    assert 0.0 <= room.tpi <= 1.0

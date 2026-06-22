"""Tests for the cross-zone workload router (Phase 6)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cross_zone_router import (  # noqa: E402
    CrossZoneRouter,
    RoutingConfig,
)


@pytest.fixture
def config() -> dict:
    with open(ROOT / "config" / "sim_config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _router(n_slots=15, cap=1.0) -> CrossZoneRouter:
    return CrossZoneRouter(RoutingConfig(liquid_util_cap=cap), n_slots)


# --- Conservation -----------------------------------------------------------

def test_routing_conserves_total_load():
    router = _router(10)
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = rng.uniform(0.0, 1.0, 10)
        dec = router.route(d)
        assert dec.u_liquid.sum() + dec.u_air.sum() == pytest.approx(d.sum(), abs=1e-9)
        assert dec.dropped == 0.0  # well within 2-zone capacity


def test_routing_conserves_even_at_high_demand():
    router = _router(8)
    d = np.full(8, 0.95)  # sum 7.6, fits across 16 slots
    dec = router.route(d)
    assert dec.u_liquid.sum() + dec.u_air.sum() == pytest.approx(d.sum(), abs=1e-9)


# --- Liquid-first behaviour --------------------------------------------------

def test_below_capacity_air_stays_idle():
    router = _router(15)
    d = np.full(15, 0.4)  # sum 6.0 < liquid capacity 15
    dec = router.route(d)
    assert dec.air_total == pytest.approx(0.0)
    np.testing.assert_array_equal(dec.u_air, np.zeros(15))
    assert dec.liquid_total == pytest.approx(d.sum())


def test_above_capacity_spills_overflow_to_air():
    router = _router(10)  # liquid capacity = 10 (cap 1.0)
    d = np.full(10, 1.0)  # sum 10.0 exactly fills liquid; add more below
    # Push demand above liquid capacity by scaling.
    dec = router.route(d * 1.5)  # total 15 > 10 -> 5 spills to air
    assert dec.liquid_total == pytest.approx(10.0)
    assert dec.air_total == pytest.approx(5.0)
    assert dec.u_liquid.max() <= 1.0 + 1e-12
    assert dec.u_air.max() <= 1.0 + 1e-12


def test_liquid_util_cap_spills_earlier():
    # With cap 0.5, liquid capacity halves, so spill starts at lower demand.
    router = _router(10, cap=0.5)  # liquid capacity = 5.0
    d = np.full(10, 0.7)  # sum 7.0 > 5.0 -> 2.0 spills
    dec = router.route(d)
    assert dec.liquid_total == pytest.approx(5.0)
    assert dec.air_total == pytest.approx(2.0)
    assert dec.u_liquid.max() <= 0.5 + 1e-12


def test_cluster_overload_is_reported_as_dropped():
    router = _router(5)  # total capacity = 10 (5 liquid + 5 air)
    d = np.full(5, 1.0)
    dec = router.route(d * 3.0)  # total 15 > 10 -> 5 dropped
    assert dec.liquid_total == pytest.approx(5.0)
    assert dec.air_total == pytest.approx(5.0)
    assert dec.dropped == pytest.approx(5.0)


# --- Caps / ranges ----------------------------------------------------------

def test_no_slot_exceeds_unit_utilization():
    router = _router(12)
    dec = router.route(np.full(12, 1.0) * 1.8)  # heavy demand
    assert dec.u_liquid.max() <= 1.0 + 1e-12
    assert dec.u_air.max() <= 1.0 + 1e-12
    assert dec.u_liquid.min() >= 0.0
    assert dec.u_air.min() >= 0.0


def test_zero_demand_idle_both_zones():
    router = _router(15)
    dec = router.route(np.zeros(15))
    np.testing.assert_array_equal(dec.u_liquid, np.zeros(15))
    np.testing.assert_array_equal(dec.u_air, np.zeros(15))


# --- Cumulative accounting --------------------------------------------------

def test_cumulative_totals_track_routing():
    router = _router(10)
    d = np.full(10, 0.6)  # sum 6 < capacity -> all liquid
    for _ in range(5):
        router.route(d)
    assert router.total_demand == pytest.approx(30.0)
    assert router.total_liquid == pytest.approx(30.0)
    assert router.total_air == pytest.approx(0.0)


# --- Config -----------------------------------------------------------------

def test_invalid_liquid_util_cap_rejected():
    with pytest.raises(ValueError, match="liquid_util_cap"):
        RoutingConfig(liquid_util_cap=0.0)
    with pytest.raises(ValueError, match="liquid_util_cap"):
        RoutingConfig(liquid_util_cap=1.5)


def test_from_config(config):
    router = CrossZoneRouter.from_config(config)
    assert router.n_slots == int(config["rack"]["n_slots"])
    assert router.config.liquid_util_cap == 1.0

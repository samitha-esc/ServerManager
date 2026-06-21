"""Unit tests for the per-node burstiness layer (Phase 2b)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.burstiness import (  # noqa: E402
    BurstConfig,
    BurstyTraceLoader,
    generate_slot_elevation,
    inject_bursts,
)
from src.rack import Rack  # noqa: E402
from src.trace_loader import TraceLoader  # noqa: E402


@pytest.fixture
def config() -> dict:
    with open(ROOT / "config" / "sim_config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def baseline() -> np.ndarray:
    # A flat, moderate baseline so burst effects are unambiguous.
    return np.full((15, 4 * 3600), 0.4, dtype=np.float64)


def _cfg(**kw) -> BurstConfig:
    base = dict(enabled=True, seed=2024, arrival_per_hour=4.0,
                magnitude_min=0.5, magnitude_max=0.7,
                duration_min_s=180, duration_max_s=480)
    base.update(kw)
    return BurstConfig(**base)


# --- Clipping ---------------------------------------------------------------

def test_burst_utilization_never_exceeds_one(baseline):
    out = inject_bursts(baseline, _cfg(magnitude_min=0.8, magnitude_max=1.5))
    assert out.max() <= 1.0
    assert out.min() >= 0.0


def test_real_loader_bursts_stay_in_unit_range(config):
    loader = TraceLoader.from_config(config, int(config["rack"]["n_slots"]))
    cfg = BurstConfig.from_config(config["burstiness"])
    out = inject_bursts(loader.utilization, cfg)
    assert out.max() <= 1.0
    assert out.min() >= 0.0


# --- Determinism ------------------------------------------------------------

def test_bursts_are_deterministic(baseline):
    a = inject_bursts(baseline, _cfg())
    b = inject_bursts(baseline, _cfg())
    assert np.array_equal(a, b)


def test_bursty_loader_deterministic_across_two_loads(config):
    n = int(config["rack"]["n_slots"])
    l1 = BurstyTraceLoader.from_config(config, n)
    l2 = BurstyTraceLoader.from_config(config, n)
    assert np.array_equal(l1.utilization, l2.utilization)


def test_different_seed_changes_bursts(baseline):
    a = inject_bursts(baseline, _cfg(seed=1))
    b = inject_bursts(baseline, _cfg(seed=2))
    assert not np.array_equal(a, b)


# --- Independence between slots ---------------------------------------------

def test_slots_get_different_burst_patterns(baseline):
    out = inject_bursts(baseline, _cfg())
    elev = out - baseline  # per-slot elevation
    # No two slots should have identical burst timelines.
    for i in range(elev.shape[0]):
        for j in range(i + 1, elev.shape[0]):
            assert not np.array_equal(elev[i], elev[j]), f"slots {i},{j} identical"


def test_slots_burst_at_different_times(baseline):
    # At a typical instant, not every slot is bursting in lockstep: the set of
    # bursting slots changes over the window.
    out = inject_bursts(baseline, _cfg())
    bursting = out > baseline + 1e-9  # [n_slots, n_steps] bool
    # The count of simultaneously-bursting slots varies (not constant / all).
    counts = bursting.sum(axis=0)
    assert counts.max() < bursting.shape[0], "all slots burst at once somewhere"
    assert counts.std() > 0.0, "bursting count never varies -> lockstep"


# --- Toggle off recovers the baseline ---------------------------------------

def test_disabled_returns_exact_baseline(baseline):
    out = inject_bursts(baseline, _cfg(enabled=False))
    assert np.array_equal(out, baseline)
    assert out is not baseline  # a copy, not the same object


def test_disabled_loader_matches_phase2_baseline(config):
    n = int(config["rack"]["n_slots"])
    base_loader = TraceLoader.from_config(config, n)
    cfg_off = BurstConfig.from_config({**config["burstiness"], "enabled": False})
    bursty = BurstyTraceLoader(base_loader, cfg_off)
    assert np.array_equal(bursty.utilization, base_loader.utilization)


# --- Elevation generator edge cases -----------------------------------------

def test_elevation_zero_length():
    rng = np.random.default_rng(0)
    assert generate_slot_elevation(0, rng, _cfg()).shape == (0,)


def test_elevation_zero_arrival_means_no_bursts():
    rng = np.random.default_rng(0)
    elev = generate_slot_elevation(3600, rng, _cfg(arrival_per_hour=0.0))
    assert np.all(elev == 0.0)


def test_elevation_bursts_are_within_magnitude_range():
    rng = np.random.default_rng(0)
    elev = generate_slot_elevation(4 * 3600, rng, _cfg())
    nonzero = elev[elev > 0]
    assert nonzero.size > 0
    assert nonzero.min() >= 0.5
    assert nonzero.max() <= 0.7


# --- The control problem now exists -----------------------------------------

def test_at_least_one_slot_crosses_temperature_threshold(config):
    """With default bursts, at least one slot must exceed 75 °C — this is the
    test that proves a cooling-control problem now exists."""
    rack = Rack.from_config(config, zone_key="air")
    loader = TraceLoader.from_config(config, rack.n_slots)
    cfg = BurstConfig.from_config(config["burstiness"])
    n_window = min(4 * 3600, loader.n_timesteps)
    util = inject_bursts(loader.utilization[:, :n_window], cfg)

    rack.reset()
    peak = 0.0
    for t in range(n_window):
        t_node, _, _ = rack.step(util[:, t])
        peak = max(peak, float(t_node.max()))
    assert peak > 75.0, f"no slot crossed 75 °C (peak {peak:.1f})"

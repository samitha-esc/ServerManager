"""Unit tests for the trace-driven load layer (Phase 2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trace_loader import (  # noqa: E402
    TraceLoader,
    interpolate_linear,
    load_cpu_fraction,
    per_slot_variation,
)


@pytest.fixture
def config() -> dict:
    with open(ROOT / "config" / "sim_config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --- Interpolation length ---------------------------------------------------

def test_interpolation_step_count_between_two_points():
    # Two points 300 s apart -> 301 one-second samples (inclusive endpoints).
    out = interpolate_linear(np.array([0.2, 0.8]), stride=300)
    assert out.shape[0] == 301
    assert out[0] == pytest.approx(0.2)
    assert out[-1] == pytest.approx(0.8)
    # Midpoint of a linear ramp.
    assert out[150] == pytest.approx(0.5)


def test_interpolation_length_for_many_points():
    n_src = 11
    vals = np.linspace(0.1, 0.9, n_src)
    out = interpolate_linear(vals, stride=300)
    assert out.shape[0] == (n_src - 1) * 300 + 1


def test_loader_timestep_count_matches_formula():
    base = np.linspace(0.2, 0.6, 5)  # 5 source points
    loader = TraceLoader(base, n_slots=4, source_stride_s=300)
    assert loader.n_timesteps == (5 - 1) * 300 + 1


# --- Clipping to [0, 1] -----------------------------------------------------

def test_utilization_never_exceeds_one_even_with_extreme_variation():
    # Base near the top of the range with a large positive multiplier+offset.
    base = np.full(5, 0.98)
    loader = TraceLoader(
        base, n_slots=20, scale_spread=0.5, offset_spread=0.2, seed=7
    )
    assert loader.utilization.max() <= 1.0
    assert loader.utilization.min() >= 0.0


def test_loader_clips_out_of_range_base_signal():
    # A noisy >1 / <0 base must be clipped before and after variation.
    base = np.array([1.5, -0.3, 0.4])
    loader = TraceLoader(base, n_slots=3, scale_spread=0.1, offset_spread=0.05)
    assert loader.utilization.max() <= 1.0
    assert loader.utilization.min() >= 0.0


def test_real_trace_cpu_fraction_in_unit_range(config):
    frac = load_cpu_fraction(
        ROOT / config["trace"]["path"], config["trace"]["cpu_column"]
    )
    assert frac.min() >= 0.0
    assert frac.max() <= 1.0


def test_real_loader_utilization_never_exceeds_one(config):
    frac = load_cpu_fraction(
        ROOT / config["trace"]["path"], config["trace"]["cpu_column"]
    )
    loader = TraceLoader(frac, n_slots=int(config["rack"]["n_slots"]))
    assert loader.utilization.max() <= 1.0


# --- Determinism ------------------------------------------------------------

def test_per_slot_variation_is_deterministic():
    a = per_slot_variation(15, seed=1729, scale_spread=0.08, offset_spread=0.01)
    b = per_slot_variation(15, seed=1729, scale_spread=0.08, offset_spread=0.01)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_two_loads_produce_identical_utilization():
    base = np.linspace(0.1, 0.7, 6)
    l1 = TraceLoader(base, n_slots=15, seed=1729)
    l2 = TraceLoader(base, n_slots=15, seed=1729)
    assert np.array_equal(l1.utilization, l2.utilization)


def test_different_seed_changes_variation():
    base = np.linspace(0.1, 0.7, 6)
    l1 = TraceLoader(base, n_slots=15, seed=1)
    l2 = TraceLoader(base, n_slots=15, seed=2)
    assert not np.array_equal(l1.multiplier, l2.multiplier)


def test_slots_are_not_all_identical():
    base = np.full(4, 0.5)
    loader = TraceLoader(base, n_slots=10, scale_spread=0.08, offset_spread=0.01)
    # At least two slots must differ at a given time.
    column = loader.utilization[:, 0]
    assert column.std() > 0.0


# --- Empty / short traces handled gracefully --------------------------------

def test_empty_trace_yields_zero_timesteps():
    loader = TraceLoader(np.array([]), n_slots=15)
    assert loader.n_timesteps == 0
    assert loader.utilization.shape == (15, 0)


def test_single_point_trace_yields_one_timestep():
    loader = TraceLoader(np.array([0.5]), n_slots=15)
    assert loader.n_timesteps == 1
    # Constant signal -> the single column is just the per-slot transform.
    assert loader.utilization.shape == (15, 1)


def test_interpolate_linear_handles_empty_and_single():
    assert interpolate_linear(np.array([]), stride=300).shape[0] == 0
    assert interpolate_linear(np.array([0.3]), stride=300).shape[0] == 1


# --- Accessor consistency ---------------------------------------------------

def test_get_utilization_matches_array():
    base = np.linspace(0.2, 0.6, 4)
    loader = TraceLoader(base, n_slots=5)
    assert loader.get_utilization(2, 100) == pytest.approx(
        loader.utilization[2, 100]
    )
    assert np.array_equal(loader.utilization_at(100), loader.utilization[:, 100])

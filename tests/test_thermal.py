"""Unit tests for the thermal foundation: power curve, gradient, convergence."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.power import PowerModel  # noqa: E402
from src.rack import Rack  # noqa: E402
from src.thermal import compute_inlet  # noqa: E402


@pytest.fixture
def config() -> dict:
    with open(ROOT / "config" / "sim_config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def rack(config: dict) -> Rack:
    return Rack.from_config(config, zone_key="air")


# --- Power curve ------------------------------------------------------------

def test_power_at_zero_utilization_is_idle():
    pm = PowerModel(p_idle=120.0, p_max=400.0, r=1.4)
    # P(0) = p_idle + (p_max - p_idle) * (0 - 0) = p_idle
    assert pm.power(0.0) == pytest.approx(120.0)


def test_power_at_full_utilization_is_pmax():
    pm = PowerModel(p_idle=120.0, p_max=400.0, r=1.4)
    # P(1) = p_idle + (p_max - p_idle) * (2 - 1) = p_max
    assert pm.power(1.0) == pytest.approx(400.0)


def test_power_is_monotonic_in_utilization():
    pm = PowerModel(p_idle=120.0, p_max=400.0, r=1.4)
    u = np.linspace(0.0, 1.0, 50)
    p = pm.power(u)
    assert np.all(np.diff(p) > 0)


def test_power_clips_out_of_range_inputs():
    pm = PowerModel()
    assert pm.power(-0.5) == pytest.approx(pm.power(0.0))
    assert pm.power(1.5) == pytest.approx(pm.power(1.0))


# --- Recirculation / vertical gradient --------------------------------------

def test_inlet_bottom_slot_equals_supply():
    t_node = np.array([60.0, 62.0, 64.0])
    inlet = compute_inlet(t_node, t_supply=22.0, alpha=0.006)
    # Bottom slot has nothing below it -> inlet is exactly the supply temp.
    assert inlet[0] == pytest.approx(22.0)


def test_inlet_increases_up_the_rack():
    t_node = np.array([60.0, 60.0, 60.0, 60.0])
    inlet = compute_inlet(t_node, t_supply=22.0, alpha=0.01)
    assert np.all(np.diff(inlet) > 0)


def test_monotonic_vertical_gradient_under_uniform_load(rack: Rack):
    # Drive a steady uniform 70% load well past the time constant.
    for _ in range(int(10 * rack.tau)):
        rack.step(0.70)
    temps = rack.t_node
    # Every slot must be strictly warmer than the one below it.
    assert np.all(np.diff(temps) > 0), temps


# --- Steady-state convergence -----------------------------------------------

def test_steady_state_convergence(rack: Rack):
    prev = rack.t_node.copy()
    last_change = np.inf
    for _ in range(int(12 * rack.tau)):
        rack.step(0.70)
        last_change = float(np.max(np.abs(rack.t_node - prev)))
        prev = rack.t_node.copy()
    # Per-step change should have decayed to essentially nothing.
    assert last_change < 1e-3, f"did not converge, last step change={last_change}"


def test_steady_state_in_calibrated_band(rack: Rack):
    for _ in range(int(12 * rack.tau)):
        rack.step(0.70)
    temps = rack.t_node
    # Phase 2b calibration: at steady 70% uniform load node temps sit ~68–78 °C
    # (was ~60–63 °C pre-recalibration). The trace baseline (~0.4) is cooler;
    # this hotter 70% scenario leaves headroom for bursts to reach 80 °C.
    assert 67.0 <= temps.min(), f"bottom slot too cold: {temps.min():.1f}"
    assert temps.max() <= 78.0, f"top slot too hot: {temps.max():.1f}"


def test_analytic_steady_state_matches_simulation(rack: Rack):
    # Bottom slot inlet == supply, so analytic rise = P*K_heat / (K_cool*omega).
    # Derive K_cool from the (re)calibrated config rather than hardcoding it.
    p70 = rack.power_model.power(0.70)
    rise = p70 * rack.thermal.K_heat / (rack.zone.K_cool * rack.zone.omega_fan)
    expected_bottom = rack.zone.T_supply + rise
    # With Phase 2b calibration (K_cool=7) this lands ~71 °C.
    assert 68.0 <= expected_bottom <= 74.0
    # And it must match what the simulation actually settles to at the bottom.
    for _ in range(int(12 * rack.tau)):
        rack.step(0.70)
    assert rack.t_node[0] == pytest.approx(expected_bottom, abs=0.2)

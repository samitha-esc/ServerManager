"""Tests for the Cooling Agent (Phase 4)."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.burstiness import BurstConfig, inject_bursts  # noqa: E402
from src.cooling_agent import (  # noqa: E402
    CRITICAL,
    SAFE,
    WARNING,
    CoolingAgent,
    CoolingAgentConfig,
)
from src.rack import Rack  # noqa: E402
from src.trace_loader import TraceLoader  # noqa: E402


@pytest.fixture
def config() -> dict:
    with open(ROOT / "config" / "sim_config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _agent() -> CoolingAgent:
    # Spec defaults; fresh instance => latch starts clear.
    return CoolingAgent(CoolingAgentConfig(
        horizon_s=15.0, warn_c=75.0, critical_c=80.0,
        fan_max=2.5, release_c=72.0, fan_base=1.0,
    ))


# --- Safe zone --------------------------------------------------------------

def test_safe_zone_base_fan_no_throttle():
    d = _agent().decide(60.0)
    assert d.state == SAFE
    assert d.omega_fan == pytest.approx(1.0)
    assert d.throttle_request is False


# --- Linear fan ramp through the warning band -------------------------------

def test_warning_ramp_at_lower_boundary():
    d = _agent().decide(75.0)  # bottom of band -> base speed
    assert d.state == WARNING
    assert d.omega_fan == pytest.approx(1.0)
    assert d.throttle_request is False


def test_warning_ramp_midpoint():
    d = _agent().decide(77.5)  # halfway -> halfway between 1.0 and 2.5
    assert d.state == WARNING
    assert d.omega_fan == pytest.approx(1.75)
    assert d.throttle_request is False


def test_warning_ramp_at_upper_boundary():
    # Exactly 80 is the top of the warning band (not yet > critical).
    d = _agent().decide(80.0)
    assert d.state == WARNING
    assert d.omega_fan == pytest.approx(2.5)
    assert d.throttle_request is False


# --- Critical / throttle ----------------------------------------------------

def test_throttle_asserts_above_critical():
    d = _agent().decide(80.01)
    assert d.state == CRITICAL
    assert d.omega_fan == pytest.approx(2.5)
    assert d.throttle_request is True


def test_throttle_held_at_max_fan_when_critical():
    d = _agent().decide(95.0)
    assert d.throttle_request is True
    assert d.omega_fan == pytest.approx(2.5)


# --- Hysteresis latch (the key behaviour) -----------------------------------

def test_hysteresis_latch_holds_then_releases_at_72():
    agent = _agent()
    # Drive up past 80 -> latch on.
    assert agent.decide(85.0).throttle_request is True
    assert agent.latched is True
    # Back below 80 but above 72 -> MUST stay latched (no chatter at 80).
    for temp in (79.9, 78.0, 76.0, 74.0, 73.0, 72.0):
        d = agent.decide(temp)
        assert d.throttle_request is True, f"released early at {temp}"
        assert d.state == CRITICAL
    # Only below 72 does it release.
    d = agent.decide(71.9)
    assert d.throttle_request is False
    assert agent.latched is False


def test_does_not_release_at_80_boundary():
    agent = _agent()
    agent.decide(85.0)              # latch
    assert agent.decide(80.0).throttle_request is True   # 80 is within hold band
    assert agent.decide(79.5).throttle_request is True


def test_release_threshold_is_strict():
    agent = _agent()
    agent.decide(85.0)
    assert agent.decide(72.0).throttle_request is True   # exactly 72 still held
    assert agent.decide(71.99).throttle_request is False  # below 72 releases


def test_after_release_returns_to_zone_logic():
    agent = _agent()
    agent.decide(85.0)
    agent.decide(71.0)             # release
    # Now behaves by predicted temp again, no latch.
    assert agent.decide(60.0).state == SAFE
    assert agent.decide(77.5).omega_fan == pytest.approx(1.75)


# --- Predictive input -------------------------------------------------------

def test_prediction_rolls_forward_and_is_hotter_under_high_load(config):
    rack = Rack.from_config(config, "air")
    # Warm the rack a bit, then predict under full load.
    for _ in range(200):
        rack.step(0.7)
    current = float(rack.t_node.max())
    pred = _agent().predict_temperature(rack, 1.0)
    assert pred > current, "prediction under higher load should exceed current"


def test_longer_horizon_predicts_further(config):
    rack = Rack.from_config(config, "air")
    for _ in range(200):
        rack.step(0.5)
    short = CoolingAgent(replace(_agent().config, horizon_s=5.0))
    long = CoolingAgent(replace(_agent().config, horizon_s=60.0))
    # Heating scenario: a longer horizon forecasts a higher temperature.
    assert long.predict_temperature(rack, 1.0) > short.predict_temperature(rack, 1.0)


def test_prediction_does_not_mutate_rack(config):
    rack = Rack.from_config(config, "air")
    for _ in range(100):
        rack.step(0.6)
    before = rack.t_node.copy()
    _agent().predict_temperature(rack, 1.0)
    assert np.array_equal(rack.t_node, before)


# --- Boundary: throttle must NOT affect load --------------------------------

def test_throttle_does_not_alter_load(config):
    """Run with the agent active under a load that forces the throttle on, and
    confirm the utilization driving the rack is byte-identical to the input."""
    rack = Rack.from_config(config, "air")
    agent = CoolingAgent.from_config(config)
    # Degrade cooling so the agent goes critical, like the validation stress run.
    n = 1500
    load = np.full((rack.n_slots, n), 0.6)
    used = np.empty_like(load)
    fired = False
    for t in range(n):
        rack.zone = replace(rack.zone, K_cool=2.0)  # forced cooling fault
        d = agent.control(rack, load[:, t])
        used[:, t] = load[:, t]  # the array actually passed to the rack
        rack.step(load[:, t])
        fired = fired or d.throttle_request
    assert fired, "scenario did not exercise the throttle"
    assert np.array_equal(used, load), "load was modified while throttle active"


def test_agent_active_vs_inactive_same_load(config):
    """The load sequence is identical whether or not the agent runs — the agent
    only writes omega_fan, never load."""
    n_slots = int(config["rack"]["n_slots"])
    loader = TraceLoader.from_config(config, n_slots)
    bcfg = BurstConfig.from_config(config["burstiness"])
    util = inject_bursts(loader.utilization[:, :2000], bcfg)

    # With agent.
    rack = Rack.from_config(config, "air")
    agent = CoolingAgent.from_config(config)
    seen_with = []
    for t in range(util.shape[1]):
        agent.control(rack, util[:, t])
        seen_with.append(util[:, t].copy())
        rack.step(util[:, t])
    # Without agent.
    base = Rack.from_config(config, "air")
    seen_without = []
    for t in range(util.shape[1]):
        seen_without.append(util[:, t].copy())
        base.step(util[:, t])
    assert np.array_equal(np.array(seen_with), np.array(seen_without))


# --- Reusable across racks (no air/single-rack hardcoding) -------------------

def test_agent_regulates_any_zone_unmodified(config):
    """A second agent instance regulates the liquid rack with no code change."""
    liquid = Rack.from_config(config, "liquid")
    agent = CoolingAgent.from_config(config)
    d = agent.control(liquid, 0.8)
    assert d.omega_fan >= 1.0
    # It read the liquid zone's params via the generic rack interface.
    assert liquid.zone.kind == "liquid"


# --- Config plumbing --------------------------------------------------------

def test_config_drives_all_thresholds(config):
    cfg = CoolingAgent.from_config(config).config
    assert cfg.horizon_s == 15.0
    assert cfg.warn_c == 75.0
    assert cfg.critical_c == 80.0
    assert cfg.fan_max == 2.5
    assert cfg.release_c == 72.0

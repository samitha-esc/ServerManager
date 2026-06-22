"""Unit tests for the Workload Agent (Phase 5)."""

from __future__ import annotations

import sys
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
    CoolingAgent,
    CoolingDecision,
)
from src.rack import Rack  # noqa: E402
from src.trace_loader import TraceLoader  # noqa: E402
from src.workload_agent import (  # noqa: E402
    P0_CRITICAL,
    P1_HIGH,
    P2_MEDIUM,
    P3_LOW,
    DeferralBuffer,
    WorkloadAgent,
    WorkloadAgentConfig,
)


@pytest.fixture
def config() -> dict:
    with open(ROOT / "config" / "sim_config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _no_throttle(temp: float = 60.0) -> CoolingDecision:
    return CoolingDecision(omega_fan=1.0, throttle_request=False,
                           state=SAFE, predicted_temp=temp)


def _throttle(temp: float = 82.0) -> CoolingDecision:
    return CoolingDecision(omega_fan=2.5, throttle_request=True,
                           state=CRITICAL, predicted_temp=temp)


# --- Priority split ---------------------------------------------------------

def test_priority_fractions_sum_to_original():
    cfg = WorkloadAgentConfig()
    agent = WorkloadAgent(cfg, n_slots=5)
    u = np.array([0.8, 0.6, 0.4, 0.9, 0.7])
    bands = agent.split_by_priority(u)
    assert len(bands) == 4
    total = sum(bands)
    np.testing.assert_allclose(total, u, atol=1e-12)


def test_custom_priority_fractions():
    cfg = WorkloadAgentConfig(priority_fractions=(0.5, 0.3, 0.1, 0.1))
    agent = WorkloadAgent(cfg, n_slots=3)
    u = np.array([1.0, 1.0, 1.0])
    bands = agent.split_by_priority(u)
    np.testing.assert_allclose(bands[0], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(bands[3], [0.1, 0.1, 0.1])


# --- LTI routing -----------------------------------------------------------

def test_lti_route_steers_load_to_cooler_slots():
    cfg = WorkloadAgentConfig()
    agent = WorkloadAgent(cfg, n_slots=5)
    u = np.full(5, 0.6)
    bands = agent.split_by_priority(u)
    t_node = np.array([70.0, 65.0, 60.0, 55.0, 50.0])  # slot 4 coolest
    routed = agent.lti_route(bands, t_node)
    # Coolest slot (4) should get more load than hottest slot (0).
    assert routed[4] > routed[0]


def test_lti_route_conserves_total_exactly():
    cfg = WorkloadAgentConfig()
    agent = WorkloadAgent(cfg, n_slots=10)
    u = np.random.default_rng(42).uniform(0.2, 0.8, 10)
    bands = agent.split_by_priority(u)
    t_node = np.linspace(40, 80, 10)
    routed = agent.lti_route(bands, t_node)
    # Strict conservation: routing relocates work, never drops it.
    assert routed.sum() == pytest.approx(u.sum(), abs=1e-9)


def test_lti_route_conserves_total_even_with_saturation():
    # High demand forces some slots to clip at 1.0; overflow must spill, not
    # vanish. Total stays conserved as long as it fits (sum <= n).
    cfg = WorkloadAgentConfig()
    agent = WorkloadAgent(cfg, n_slots=6)
    u = np.full(6, 0.9)  # sum 5.4, fits in capacity 6
    bands = agent.split_by_priority(u)
    t_node = np.array([90.0, 88.0, 70.0, 60.0, 55.0, 50.0])
    routed = agent.lti_route(bands, t_node)
    assert routed.max() <= 1.0 + 1e-12
    assert routed.sum() == pytest.approx(u.sum(), abs=1e-9)


def test_lti_route_clips_to_unit_range():
    cfg = WorkloadAgentConfig()
    agent = WorkloadAgent(cfg, n_slots=5)
    u = np.full(5, 0.95)
    bands = agent.split_by_priority(u)
    t_node = np.array([80.0, 75.0, 70.0, 65.0, 60.0])
    routed = agent.lti_route(bands, t_node)
    assert routed.max() <= 1.0
    assert routed.min() >= 0.0


# --- Nominal passthrough (no throttle) --------------------------------------

def test_nominal_mode_passthrough_by_default(config):
    rack = Rack.from_config(config, "air")
    agent = WorkloadAgent.from_config(config)  # lti_enabled defaults to False
    u = np.full(rack.n_slots, 0.5)
    for _ in range(100):
        rack.step(0.5)
    # No throttle visible (delay=1) and LTI off -> transparent pass-through.
    d = agent.step(u, _no_throttle(), rack, t=0)
    assert d.routing == "passthrough"
    assert d.throttle_active is False
    # Effective load is byte-identical to the input (baseline unchanged).
    np.testing.assert_array_equal(d.effective_utilization, u)


def test_nominal_mode_lti_when_enabled(config):
    rack = Rack.from_config(config, "air")
    cfg = WorkloadAgentConfig.from_config({**config["workload_agent"],
                                           "lti_enabled": True})
    agent = WorkloadAgent(cfg, rack.n_slots)
    u = np.full(rack.n_slots, 0.5)
    for _ in range(100):
        rack.step(0.5)
    d = agent.step(u, _no_throttle(), rack, t=0)
    assert d.routing == "lti"
    # LTI conserves total load (relocates, never drops).
    assert d.effective_utilization.sum() == pytest.approx(u.sum(), abs=1e-9)


# --- Throttle shedding -----------------------------------------------------

def test_p0_fully_preserved_during_throttle():
    cfg = WorkloadAgentConfig()
    agent = WorkloadAgent(cfg, n_slots=3)
    rack_temps = np.array([78.0, 79.0, 80.0])

    # Prime the delay buffer: tick 0 sends throttle, tick 1 sees it.
    u = np.array([1.0, 1.0, 1.0])

    # Tick 0: send throttle signal.
    from src.rack import CoolingZone, Rack as R
    from src.thermal import ThermalParams
    from src.power import PowerModel
    rack = R(n_slots=3, zone=CoolingZone("test", "air", 7.0, 22.0, 0.006),
             power_model=PowerModel(), thermal=ThermalParams(), dt=1.0)
    rack.t_node = rack_temps.copy()
    agent.step(u, _throttle(), rack, t=0)

    # Tick 1: throttle is now visible.
    rack.t_node = rack_temps.copy()
    d = agent.step(u, _throttle(), rack, t=1)
    assert d.throttle_active is True
    assert d.routing == "throttled"
    # P0 fraction (0.20) of each slot's load should be fully preserved.
    p0_load = float(u.sum() * cfg.priority_fractions[P0_CRITICAL])
    assert d.effective_utilization.sum() >= p0_load - 0.01


def test_p1_reduced_15pct_above_83c():
    cfg = WorkloadAgentConfig(p1_throttle_temp=83.0, p1_throttle_fraction=0.15)
    agent = WorkloadAgent(cfg, n_slots=2)

    from src.rack import CoolingZone, Rack as R
    from src.thermal import ThermalParams
    from src.power import PowerModel
    rack = R(n_slots=2, zone=CoolingZone("test", "air", 7.0, 22.0, 0.006),
             power_model=PowerModel(), thermal=ThermalParams(), dt=1.0)

    u = np.array([0.8, 0.8])
    # Prime delay.
    rack.t_node = np.array([84.0, 70.0])
    agent.step(u, _throttle(), rack, t=0)

    # Tick 1: throttle visible.
    rack.t_node = np.array([84.0, 70.0])
    d = agent.step(u, _throttle(), rack, t=1)
    # Slot 0 is above 83°C: P1 load should be reduced by 15%.
    # Slot 1 is below: full P1.
    assert d.shed_by_priority[P1_HIGH] > 0
    # The shed amount should be ~15% of slot 0's P1 load.
    p1_slot0 = 0.8 * cfg.priority_fractions[P1_HIGH]
    expected_shed = p1_slot0 * 0.15
    assert d.shed_by_priority[P1_HIGH] == pytest.approx(expected_shed, abs=0.01)


def test_p2_and_p3_fully_deferred():
    cfg = WorkloadAgentConfig()
    agent = WorkloadAgent(cfg, n_slots=2)

    from src.rack import CoolingZone, Rack as R
    from src.thermal import ThermalParams
    from src.power import PowerModel
    rack = R(n_slots=2, zone=CoolingZone("test", "air", 7.0, 22.0, 0.006),
             power_model=PowerModel(), thermal=ThermalParams(), dt=1.0)

    u = np.array([0.8, 0.8])
    rack.t_node = np.array([78.0, 78.0])
    agent.step(u, _throttle(), rack, t=0)

    rack.t_node = np.array([78.0, 78.0])
    d = agent.step(u, _throttle(), rack, t=1)
    # P2 and P3 should be fully deferred.
    p2_expected = float(u.sum() * cfg.priority_fractions[P2_MEDIUM])
    p3_expected = float(u.sum() * cfg.priority_fractions[P3_LOW])
    assert d.shed_by_priority[P2_MEDIUM] == pytest.approx(p2_expected, abs=0.01)
    assert d.shed_by_priority[P3_LOW] == pytest.approx(p3_expected, abs=0.01)
    assert agent.buffer.size > 0


# --- Deferral buffer -------------------------------------------------------

def test_buffer_p2_expires_after_limit():
    buf = DeferralBuffer()
    buf.push(P2_MEDIUM, t=0, slot=0, load=0.3)
    buf.push(P2_MEDIUM, t=10, slot=1, load=0.2)
    # Expire at t=50: entry at t=0 is 50s old (> 45s limit), but t=10 is 40s.
    expired = buf.expire(t=50, p2_limit_s=45)
    assert expired == pytest.approx(0.3)
    assert buf.p2_count == 1


def test_buffer_p3_held_until_throttle_clears():
    buf = DeferralBuffer()
    buf.push(P3_LOW, t=0, slot=0, load=0.5)
    # While throttle active: P3 should NOT drain.
    headroom = np.array([1.0])
    drained = buf.drain(t=100, throttle_active=True, n_slots=1, headroom=headroom)
    assert drained[0] == 0.0
    assert buf.p3_count == 1
    # After throttle clears: P3 drains.
    drained = buf.drain(t=200, throttle_active=False, n_slots=1, headroom=headroom)
    assert drained[0] == pytest.approx(0.5)
    assert buf.p3_count == 0


def test_buffer_drain_respects_headroom():
    buf = DeferralBuffer()
    buf.push(P2_MEDIUM, t=0, slot=0, load=0.8)
    headroom = np.array([0.3])  # only 0.3 capacity available
    drained = buf.drain(t=10, throttle_active=False, n_slots=1, headroom=headroom)
    # Can't fit 0.8 into 0.3 headroom — stays in buffer.
    assert drained[0] == 0.0
    assert buf.p2_count == 1


# --- Propagation delay ------------------------------------------------------

def test_throttle_signal_delayed_by_one_tick():
    cfg = WorkloadAgentConfig(propagation_delay_ticks=1)
    agent = WorkloadAgent(cfg, n_slots=2)

    from src.rack import CoolingZone, Rack as R
    from src.thermal import ThermalParams
    from src.power import PowerModel
    rack = R(n_slots=2, zone=CoolingZone("test", "air", 7.0, 22.0, 0.006),
             power_model=PowerModel(), thermal=ThermalParams(), dt=1.0)

    u = np.array([0.6, 0.6])
    # Tick 0: send throttle — should NOT be visible yet.
    d0 = agent.step(u, _throttle(), rack, t=0)
    assert d0.throttle_active is False

    # Tick 1: throttle from tick 0 is now visible.
    d1 = agent.step(u, _no_throttle(), rack, t=1)
    assert d1.throttle_active is True


# --- Hysteresis alignment ---------------------------------------------------

def test_hysteresis_stays_in_mitigation_until_72c(config):
    """Agent mirrors CoolingAgent's latch: stays throttled until < 72°C."""
    rack = Rack.from_config(config, "air")
    agent = WorkloadAgent.from_config(config)
    u = np.full(rack.n_slots, 0.6)

    # Warm the rack.
    for _ in range(300):
        rack.step(0.7)

    # Send throttle for 3 ticks so it's visible.
    for t in range(3):
        agent.step(u, _throttle(), rack, t=t)

    # Now send clear — but because of the 1-tick delay, tick 3 still sees
    # the throttle from tick 2.
    d = agent.step(u, _no_throttle(), rack, t=3)
    assert d.throttle_active is True  # delayed: still seeing throttle


# --- Task Preservation Index ------------------------------------------------

def test_tpi_perfect_when_no_shedding():
    cfg = WorkloadAgentConfig()
    agent = WorkloadAgent(cfg, n_slots=3)

    from src.rack import CoolingZone, Rack as R
    from src.thermal import ThermalParams
    from src.power import PowerModel
    rack = R(n_slots=3, zone=CoolingZone("test", "air", 7.0, 22.0, 0.006),
             power_model=PowerModel(), thermal=ThermalParams(), dt=1.0)

    u = np.array([0.5, 0.5, 0.5])
    for t in range(10):
        agent.step(u, _no_throttle(), rack, t=t)
    # No shedding in nominal mode — TPI should be exactly 1.0.
    assert agent.tpi == pytest.approx(1.0)


def _bench_rack(n_slots: int, temp: float):
    from src.rack import CoolingZone, Rack as R
    from src.thermal import ThermalParams
    from src.power import PowerModel
    rack = R(n_slots=n_slots, zone=CoolingZone("test", "air", 7.0, 22.0, 0.006),
             power_model=PowerModel(), thermal=ThermalParams(), dt=1.0)
    rack.t_node = np.full(n_slots, temp)
    return rack


def test_deferral_alone_does_not_lower_tpi():
    """Work parked in the buffer (not yet expired) must NOT count against TPI."""
    cfg = WorkloadAgentConfig(p2_defer_limit_s=10_000)  # nothing expires in window
    agent = WorkloadAgent(cfg, n_slots=3)
    rack = _bench_rack(3, 78.0)
    u = np.array([0.8, 0.8, 0.8])
    for t in range(30):           # throttle active: P2/P3 deferred, none expire
        rack.t_node = np.full(3, 78.0)
        agent.step(u, _throttle(), rack, t=t)
    assert agent.buffer.size > 0          # work is genuinely parked
    assert agent.total_expired == 0.0
    assert agent.tpi == pytest.approx(1.0)  # in-flight work is not penalised


def test_tpi_drops_only_when_work_expires():
    cfg = WorkloadAgentConfig()  # p2_defer_limit_s = 45
    agent = WorkloadAgent(cfg, n_slots=3)
    rack = _bench_rack(3, 78.0)
    u = np.array([0.8, 0.8, 0.8])
    # Throttle continuously well past the 45 s P2 expiry window.
    for t in range(120):
        rack.t_node = np.full(3, 78.0)
        agent.step(u, _throttle(), rack, t=t)
    assert agent.total_expired > 0.0
    assert agent.tpi < 1.0
    # Matches the definition exactly: executed / (executed + expired).
    expected = agent.total_executed / (agent.total_executed + agent.total_expired)
    assert agent.tpi == pytest.approx(expected)


# --- Integration: full loop with CoolingAgent + WorkloadAgent ---------------

def test_full_integration_loop(config):
    """Run the full simulation loop: Trace → Burst → CoolingAgent → WorkloadAgent → Rack."""
    rack = Rack.from_config(config, "air")
    cooling = CoolingAgent.from_config(config)
    workload = WorkloadAgent.from_config(config)

    loader = TraceLoader.from_config(config, rack.n_slots)
    bcfg = BurstConfig.from_config(config["burstiness"])
    n_win = min(500, loader.n_timesteps)
    util = inject_bursts(loader.utilization[:, :n_win], bcfg)

    for t in range(n_win):
        cd = cooling.control(rack, util[:, t])
        wd = workload.step(util[:, t], cd, rack, t)
        rack.step(wd.effective_utilization)

    # The simulation should complete without error.
    assert np.all(np.isfinite(rack.t_node))
    # TPI should be reasonable (most work preserved).
    assert workload.tpi > 0.3


def test_workload_agent_reduces_peak_under_stress(config):
    """Under degraded cooling, the WorkloadAgent should reduce peak temp."""
    from dataclasses import replace

    n_steps = 2000

    # Without workload agent.
    rack_no = Rack.from_config(config, "air")
    cool_no = CoolingAgent.from_config(config)
    peak_no = 0.0
    for t in range(n_steps):
        rack_no.zone = replace(rack_no.zone, K_cool=2.0)
        cd = cool_no.control(rack_no, 0.6)
        rack_no.step(0.6)
        peak_no = max(peak_no, float(rack_no.t_node.max()))

    # With workload agent.
    rack_wa = Rack.from_config(config, "air")
    cool_wa = CoolingAgent.from_config(config)
    work_wa = WorkloadAgent.from_config(config)
    peak_wa = 0.0
    for t in range(n_steps):
        rack_wa.zone = replace(rack_wa.zone, K_cool=2.0)
        u = np.full(rack_wa.n_slots, 0.6)
        cd = cool_wa.control(rack_wa, u)
        wd = work_wa.step(u, cd, rack_wa, t)
        rack_wa.step(wd.effective_utilization)
        peak_wa = max(peak_wa, float(rack_wa.t_node.max()))

    # Workload agent should reduce peak temperature.
    assert peak_wa < peak_no, (
        f"WorkloadAgent did not help: peak_wa={peak_wa:.1f} >= peak_no={peak_no:.1f}"
    )


# --- Config plumbing --------------------------------------------------------

def test_priority_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        WorkloadAgentConfig(priority_fractions=(0.2, 0.2, 0.2, 0.2))


def test_priority_fractions_must_have_four_entries():
    with pytest.raises(ValueError, match="4 entries"):
        WorkloadAgentConfig(priority_fractions=(0.5, 0.5))


def test_priority_fractions_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        WorkloadAgentConfig(priority_fractions=(0.6, 0.6, -0.1, -0.1))


def test_config_drives_all_params(config):
    agent = WorkloadAgent.from_config(config)
    assert agent.config.lti_enabled is False
    assert agent.config.priority_fractions == (0.20, 0.25, 0.30, 0.25)
    assert agent.config.p1_throttle_fraction == 0.15
    assert agent.config.p1_throttle_temp == 83.0
    assert agent.config.p2_defer_limit_s == 45
    assert agent.config.propagation_delay_ticks == 1


# --- Baseline regression: default agent must not change nominal behaviour -----

def test_default_agent_preserves_nominal_baseline(config):
    """With default config (LTI off) and no throttle ever firing, the rack with
    the WorkloadAgent must behave byte-identically to the cooling-only baseline."""
    n_slots = int(config["rack"]["n_slots"])
    loader = TraceLoader.from_config(config, n_slots)
    bcfg = BurstConfig.from_config(config["burstiness"])
    util = inject_bursts(loader.utilization[:, :1500], bcfg)

    # Cooling-only baseline (Phase 4 behaviour).
    rack_base = Rack.from_config(config, "air")
    cool_base = CoolingAgent.from_config(config)
    base_temps = np.empty((n_slots, util.shape[1]))
    for t in range(util.shape[1]):
        cool_base.control(rack_base, util[:, t])
        rack_base.step(util[:, t])
        base_temps[:, t] = rack_base.t_node

    # Cooling + default WorkloadAgent.
    rack_wa = Rack.from_config(config, "air")
    cool_wa = CoolingAgent.from_config(config)
    work_wa = WorkloadAgent.from_config(config)
    wa_temps = np.empty((n_slots, util.shape[1]))
    for t in range(util.shape[1]):
        cd = cool_wa.control(rack_wa, util[:, t])
        wd = work_wa.step(util[:, t], cd, rack_wa, t)
        # Nominal pass-through: effective load equals the raw input exactly.
        np.testing.assert_array_equal(wd.effective_utilization, util[:, t])
        rack_wa.step(wd.effective_utilization)
        wa_temps[:, t] = rack_wa.t_node

    np.testing.assert_array_equal(base_temps, wa_temps)
    assert work_wa.tpi == pytest.approx(1.0, abs=1e-9)  # nothing dropped

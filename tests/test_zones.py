"""Tests for the heterogeneous air/liquid cooling fabric (Phase 3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.burstiness import BurstConfig, inject_bursts  # noqa: E402
from src.rack import CoolingZone, Rack  # noqa: E402
from src.trace_loader import TraceLoader  # noqa: E402

# Golden air-zone values from Phase 2b — the alpha refactor must not move them.
PHASE2B_AIR_PEAK = 81.70
PHASE2B_AIR_70PCT = (70.86, 75.13)  # (bottom, top) steady at uniform 70% load


@pytest.fixture
def config() -> dict:
    with open(ROOT / "config" / "sim_config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _steady(rack: Rack, u: float, steps: int) -> np.ndarray:
    rack.reset()
    t_node = rack.t_node
    for _ in range(steps):
        t_node, _, _ = rack.step(u)
    return t_node.copy()


def _bursty_temps(config: dict, zone_key: str) -> np.ndarray:
    """Drive a rack of the given zone with the canonical bursty load."""
    rack = Rack.from_config(config, zone_key=zone_key)
    loader = TraceLoader.from_config(config, rack.n_slots)
    bcfg = BurstConfig.from_config(config["burstiness"])
    n_window = min(4 * 3600, loader.n_timesteps)
    util = inject_bursts(loader.utilization[:, :n_window], bcfg)
    rack.reset()
    temps = np.empty((rack.n_slots, n_window))
    for t in range(n_window):
        t_node, _, _ = rack.step(util[:, t])
        temps[:, t] = t_node
    return temps


# --- Air regression: unchanged after the alpha refactor ---------------------

def test_air_steady_state_unchanged_after_alpha_refactor(config):
    rack = Rack.from_config(config, zone_key="air")
    prof = _steady(rack, 0.70, steps=int(20 * rack.tau))
    assert prof[0] == pytest.approx(PHASE2B_AIR_70PCT[0], abs=0.05)
    assert prof[-1] == pytest.approx(PHASE2B_AIR_70PCT[1], abs=0.05)


def test_air_alpha_still_sourced_as_0006(config):
    # alpha now lives on the zone, but the air value is unchanged.
    zone = CoolingZone.from_config(config["zones"]["air"])
    assert zone.alpha == pytest.approx(0.006)


def test_air_bursty_peak_unchanged(config):
    temps = _bursty_temps(config, "air")
    assert temps.max() == pytest.approx(PHASE2B_AIR_PEAK, abs=0.01)


# --- Liquid stays safe while air overheats ----------------------------------

def test_liquid_safe_while_air_crosses_80_under_same_load(config):
    air = _bursty_temps(config, "air")
    liquid = _bursty_temps(config, "liquid")
    assert air.max() > 80.0, f"air should cross 80 °C (got {air.max():.1f})"
    assert liquid.max() < 75.0, f"liquid should stay safe (got {liquid.max():.1f})"


def test_no_liquid_slot_crosses_warning(config):
    liquid = _bursty_temps(config, "liquid")
    n_cross = int((liquid.max(axis=1) > 75.0).sum())
    assert n_cross == 0, f"{n_cross} liquid slots crossed 75 °C"


# --- Liquid gradient near-zero, well below air ------------------------------

def test_liquid_gradient_near_zero_and_below_air(config):
    air = Rack.from_config(config, zone_key="air")
    liquid = Rack.from_config(config, zone_key="liquid")
    prof_air = _steady(air, 0.70, steps=int(20 * air.tau))
    prof_liq = _steady(liquid, 0.70, steps=int(20 * air.tau))
    air_grad = abs(prof_air[-1] - prof_air[0])
    liq_grad = abs(prof_liq[-1] - prof_liq[0])
    assert liq_grad < 0.5, f"liquid gradient not flat ({liq_grad:.3f} °C)"
    assert liq_grad < air_grad, "liquid gradient should be below air's"


# --- Both zones are pure config, no zone-specific Python branching -----------

def test_both_zones_instantiate_from_config(config):
    air = Rack.from_config(config, zone_key="air")
    liquid = Rack.from_config(config, zone_key="liquid")
    assert air.zone.kind == "air"
    assert liquid.zone.kind == "liquid"
    # Liquid genuinely differs in physics, sourced from config.
    assert liquid.zone.K_cool > air.zone.K_cool
    assert liquid.zone.T_supply < air.zone.T_supply
    assert liquid.zone.alpha < air.zone.alpha


def test_kind_does_not_drive_physics(config):
    """A 'liquid'-kind zone given air's numbers must behave exactly like air —
    proving the physics is config-driven with no branching on `kind`."""
    air = Rack.from_config(config, zone_key="air")
    # Same numeric params as air, but labelled liquid.
    air_cfg = dict(config["zones"]["air"])
    air_cfg["kind"] = "liquid"
    air_cfg["name"] = "liquid-but-air-params"
    masquerade_cfg = {**config, "zones": {**config["zones"], "x": air_cfg}}
    masquerade = Rack.from_config(masquerade_cfg, zone_key="x")

    prof_air = _steady(air, 0.70, steps=600)
    prof_masq = _steady(masquerade, 0.70, steps=600)
    assert np.array_equal(prof_air, prof_masq), "kind label altered the physics"


def test_arbitrary_new_zone_works_from_config_only(config):
    """Adding a brand-new zone is purely config — no code change needed."""
    new_zone = {
        "name": "warm-water", "kind": "liquid",
        "K_cool": 30.0, "T_supply": 30.0, "alpha": 0.001, "omega_fan": 1.0,
    }
    cfg = {**config, "zones": {**config["zones"], "warm": new_zone}}
    rack = Rack.from_config(cfg, zone_key="warm")
    prof = _steady(rack, 0.70, steps=600)
    assert np.all(np.isfinite(prof))
    assert prof.min() > new_zone["T_supply"]  # warmer than supply under load

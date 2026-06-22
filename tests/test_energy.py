"""Tests for the cooling-energy / PUE model (Phase 7)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.energy import EnergyModel  # noqa: E402


@pytest.fixture
def config() -> dict:
    with open(ROOT / "config" / "sim_config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _model() -> EnergyModel:
    return EnergyModel(rated_w={"air": 500.0, "liquid": 150.0}, exponent=3.0)


def test_cooling_power_at_rated_omega():
    m = _model()
    assert m.cooling_power_w("air", 1.0) == pytest.approx(500.0)
    assert m.cooling_power_w("liquid", 1.0) == pytest.approx(150.0)


def test_cooling_power_follows_cube_law():
    m = _model()
    # omega 2.5 -> 2.5**3 = 15.625x rated.
    assert m.cooling_power_w("air", 2.5) == pytest.approx(500.0 * 2.5 ** 3)
    assert m.cooling_power_w("air", 2.0) == pytest.approx(500.0 * 8.0)


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        _model().cooling_power_w("immersion", 1.0)


def test_pue_basic():
    assert EnergyModel.pue(1000.0, 100.0) == pytest.approx(1.1)
    assert EnergyModel.pue(2000.0, 0.0) == pytest.approx(1.0)


def test_pue_zero_it_is_nan():
    assert math.isnan(EnergyModel.pue(0.0, 50.0))


def test_liquid_far_cheaper_than_maxed_air():
    m = _model()
    # A maxed air fan dwarfs a steady liquid pump.
    assert m.cooling_power_w("air", 2.5) > 10 * m.cooling_power_w("liquid", 1.0)


def test_from_config(config):
    m = EnergyModel.from_config(config)
    assert m.exponent == 3.0
    assert m.rated_w["air"] == 500.0
    assert m.rated_w["liquid"] == 150.0

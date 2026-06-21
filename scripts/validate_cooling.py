"""Validate the Cooling Agent (Phase 4).

Two scenarios on a single air rack:

  NOMINAL — the real bursty load with the agent active (spec config untouched).
  The predictive fan ramp fully contains the load, so the throttle never fires.
  This is a *correct* finding, not a failure: fans alone suffice here. We report
  the peak with vs. without the agent to show the fan ramp removes heat.

  DEGRADED — a controlled cooling-fault stress test (validation artifact only,
  does NOT change default behaviour). A CRAC/fan failure reduces the air zone's
  K_cool for a window, driving the agent into the critical state so we can
  visualise the throttle assert -> hold (latched in the 72-80 band) -> release at
  predicted < 72. Load is a smooth controlled profile so the latch is legible.

Hard boundary (Phase 4): the THROTTLE_REQUEST is decided/logged only — it never
touches utilization or load in either scenario. The only real actuation is the
fan (omega_fan), which feeds the thermal model.

Plots saved to outputs/:
  cooling_nominal_fan_effect.png — peak temp with vs. without the agent (+ fan).
  cooling_degraded_throttle.png  — temp (actual+predicted) + omega_fan overlay
                                   with 75/80/72 lines, throttle timeline beneath.

Run from the project root:
    python scripts/validate_cooling.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass

import yaml  # noqa: E402

from src.burstiness import BurstConfig, inject_bursts  # noqa: E402
from src.cooling_agent import CoolingAgent  # noqa: E402
from src.rack import Rack  # noqa: E402
from src.trace_loader import TraceLoader  # noqa: E402

CONFIG_PATH = ROOT / "config" / "sim_config.yaml"
OUTPUT_DIR = ROOT / "outputs"
NOMINAL_HOURS = 4.0

# --- Degraded stress scenario (validation artifact only) --------------------
# A controlled CRAC/fan-failure timeline on a smooth load, sized purely to drive
# the agent into critical and exercise the hysteresis latch. None of this changes
# the default simulation or the agent config.
STRESS_T = 6000               # [s] scenario length
STRESS_LOAD = 0.55            # smooth constant load (legible trajectory)
FAILURE_START = 1000          # CRAC fault begins
PARTIAL_START = 2500          # cooling partially recovers (parks temp in-band)
REPAIR_START = 4000           # cooling fully restored -> release
K_FAIL = 1.8                  # degraded K_cool during the fault (critical)
K_PARTIAL = 2.5               # partial-recovery K_cool (parks temp in 72-80 band)


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def count_events(throttle: np.ndarray) -> int:
    """Number of rising edges (False->True) in the throttle signal."""
    return int(np.sum((~np.r_[False, throttle[:-1]]) & throttle))


def run_nominal(config: dict, util: np.ndarray):
    """Bursty load, agent active vs. fans fixed at 1.0."""
    n_slots, n = util.shape
    # With agent.
    rack = Rack.from_config(config, "air")
    agent = CoolingAgent.from_config(config)
    Tmax = np.empty(n)
    fan = np.empty(n)
    thr = np.zeros(n, dtype=bool)
    for t in range(n):
        d = agent.control(rack, util[:, t])
        rack.step(util[:, t])
        Tmax[t] = rack.t_node.max()
        fan[t] = d.omega_fan
        thr[t] = d.throttle_request
    # Without agent (fans fixed at 1.0).
    base = Rack.from_config(config, "air")
    Tbase = np.empty(n)
    for t in range(n):
        base.step(util[:, t])
        Tbase[t] = base.t_node.max()
    return Tmax, fan, thr, Tbase


def _stress_kcool(t: int, nom_k: float) -> float:
    if t < FAILURE_START:
        return nom_k
    if t < PARTIAL_START:
        return K_FAIL
    if t < REPAIR_START:
        return K_PARTIAL
    return nom_k


def run_degraded(config: dict):
    """Controlled CRAC-failure stress test; returns logged series + load used."""
    rack = Rack.from_config(config, "air")
    agent = CoolingAgent.from_config(config)
    nom_k = rack.zone.K_cool
    Tmax = np.empty(STRESS_T)
    pred = np.empty(STRESS_T)
    fan = np.empty(STRESS_T)
    thr = np.zeros(STRESS_T, dtype=bool)
    load_used = np.empty(STRESS_T)
    for t in range(STRESS_T):
        rack.zone = replace(rack.zone, K_cool=_stress_kcool(t, nom_k))
        load_used[t] = STRESS_LOAD  # record what actually drove the rack
        d = agent.control(rack, STRESS_LOAD)
        rack.step(STRESS_LOAD)
        Tmax[t] = rack.t_node.max()
        pred[t] = d.predicted_temp
        fan[t] = d.omega_fan
        thr[t] = d.throttle_request
    return Tmax, pred, fan, thr, load_used


def plot_nominal(Tmax, fan, Tbase, cfg) -> Path:
    n = Tmax.shape[0]
    t_min = np.arange(n) / 60.0
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(t_min, Tbase, color="tab:red", lw=0.8, alpha=0.7,
            label=f"no agent (fans=1.0): peak {Tbase.max():.1f} °C")
    ax.plot(t_min, Tmax, color="tab:green", lw=0.9,
            label=f"cooling agent: peak {Tmax.max():.1f} °C")
    ax.axhline(cfg.warn_c, color="darkorange", ls="--", lw=1.2, label="75 °C warning")
    ax.axhline(cfg.critical_c, color="red", ls="--", lw=1.2, label="80 °C critical")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("hottest-slot temperature [°C]")
    ax.set_title("Nominal bursty load — fan ramp removes heat (throttle never needed)")
    ax.legend(loc="lower right")
    ax2 = ax.twinx()
    ax2.plot(t_min, fan, color="tab:blue", lw=0.7, alpha=0.5)
    ax2.set_ylabel("omega_fan", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylim(0.9, cfg.fan_max + 0.2)
    fig.tight_layout()
    out = OUTPUT_DIR / "cooling_nominal_fan_effect.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_degraded(Tmax, pred, fan, thr, cfg) -> Path:
    n = Tmax.shape[0]
    t_min = np.arange(n) / 60.0
    fig, (ax_t, ax_s) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_t.plot(t_min, Tmax, color="tab:red", lw=1.1, label="actual hottest slot")
    ax_t.plot(t_min, pred, color="tab:purple", lw=1.0, ls=":",
              label=f"predicted (+{cfg.horizon_s:.0f} s, agent acts on this)")
    ax_t.axhline(cfg.critical_c, color="red", ls="--", lw=1.2, label="80 °C critical")
    ax_t.axhline(cfg.warn_c, color="darkorange", ls="--", lw=1.2, label="75 °C warning")
    ax_t.axhline(cfg.release_c, color="seagreen", ls="--", lw=1.2,
                 label="72 °C release (hysteresis)")
    ax_t.set_ylabel("temperature [°C]")
    ax_t.set_title("Degraded cooling (CRAC fault) — throttle asserts, holds, releases at 72")
    ax_t.legend(loc="upper right", fontsize=8)
    axf = ax_t.twinx()
    axf.plot(t_min, fan, color="tab:blue", lw=1.0, alpha=0.6)
    axf.set_ylabel("omega_fan", color="tab:blue")
    axf.tick_params(axis="y", labelcolor="tab:blue")
    axf.set_ylim(0.9, cfg.fan_max + 0.2)

    ax_s.fill_between(t_min, 0, thr.astype(float), step="pre",
                      color="firebrick", alpha=0.85)
    ax_s.set_ylim(-0.1, 1.1)
    ax_s.set_yticks([0, 1])
    ax_s.set_yticklabels(["clear", "THROTTLE"])
    ax_s.set_xlabel("time [min]")
    ax_s.set_ylabel("request")
    ax_s.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "cooling_degraded_throttle.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    n_slots = int(config["rack"]["n_slots"])
    agent_cfg = CoolingAgent.from_config(config).config

    # ---- Nominal scenario ---------------------------------------------------
    loader = TraceLoader.from_config(config, n_slots)
    bcfg = BurstConfig.from_config(config["burstiness"])
    n_win = min(int(NOMINAL_HOURS * 3600), loader.n_timesteps)
    util = inject_bursts(loader.utilization[:, :n_win], bcfg)

    Tmax, fan, thr, Tbase = run_nominal(config, util)
    nom_events = count_events(thr)
    print("=== NOMINAL (real bursty load, spec config) ===")
    print(f"  peak WITHOUT agent (fans=1.0): {Tbase.max():.2f} °C")
    print(f"  peak WITH agent              : {Tmax.max():.2f} °C")
    print(f"  fan range                    : {fan.min():.2f}–{fan.max():.2f}")
    print(f"  throttle events              : {nom_events} "
          f"(fans contain the load; throttle correctly not needed)")
    assert Tmax.max() < Tbase.max(), "fan ramp did not remove heat"
    assert nom_events == 0, "throttle unexpectedly fired under nominal load"

    # ---- Degraded stress scenario -------------------------------------------
    dT, dpred, dfan, dthr, dload = run_degraded(config)
    deg_events = count_events(dthr)
    on = np.where(dthr)[0]
    assert_t = int(on[0])
    release_t = int(on[-1] + 1)
    park = dT[PARTIAL_START + 200:REPAIR_START - 10]
    park_throttled = bool(dthr[PARTIAL_START + 200:REPAIR_START - 10].all())
    # Hold-in-band proof: throttle stays asserted while predicted is in (72, 80).
    hold_in_band = int(((dpred > agent_cfg.release_c)
                        & (dpred < agent_cfg.critical_c) & dthr).sum())
    rebound = bool(dthr[release_t:].any())

    print("\n=== DEGRADED (controlled CRAC-fault stress test) ===")
    print(f"  peak temperature             : {dT.max():.2f} °C")
    print(f"  throttle events              : {deg_events}")
    print(f"  assert  @ {assert_t} s: predicted {dpred[assert_t]:.1f} °C "
          f"(> {agent_cfg.critical_c:.0f}), actual {dT[assert_t]:.1f} °C")
    print(f"  hold    : parked at {park.mean():.1f} °C in the 72–80 band for "
          f"{hold_in_band} s with throttle latched (did NOT release at 80)")
    print(f"  release @ {release_t} s: predicted {dpred[release_t]:.1f} °C "
          f"(< {agent_cfg.release_c:.0f}), actual {dT[release_t]:.1f} °C")
    print(f"  rebound after release        : {rebound}")

    # Hysteresis assertions.
    assert deg_events == 1, f"expected one clean throttle cycle, got {deg_events}"
    assert dpred[assert_t] > agent_cfg.critical_c, "asserted below critical"
    assert park_throttled and park.mean() < agent_cfg.critical_c, \
        "throttle did not hold while parked below 80 °C"
    assert hold_in_band > 0, "no latched hold in the 72–80 band"
    assert dpred[release_t] < agent_cfg.release_c, "released above 72 °C predicted"
    assert not rebound, "throttle re-fired after release (chatter)"

    # Boundary regression: throttle never altered the load.
    assert np.array_equal(dload, np.full(STRESS_T, STRESS_LOAD)), \
        "load was modified — throttle must not affect load this phase"
    print("  boundary: load unchanged by throttle ✓")

    p1 = plot_nominal(Tmax, fan, Tbase, agent_cfg)
    p2 = plot_degraded(dT, dpred, dfan, dthr, agent_cfg)
    for p in (p1, p2):
        print(f"  saved {p.relative_to(ROOT)}")

    print("\nAll cooling-agent validation checks passed.")


if __name__ == "__main__":
    main()

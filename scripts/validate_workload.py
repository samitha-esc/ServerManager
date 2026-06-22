"""Validate the Workload Agent (Phase 5).

Two scenarios on a single air rack:

  NOMINAL — the real bursty load with both agents active (spec config untouched).
  Shows LTI routing steering load toward cooler slots and flattening the thermal
  gradient, while total throughput is preserved.

  DEGRADED — a controlled CRAC-fault stress test (same fault timeline as
  validate_cooling.py). The CoolingAgent asserts THROTTLE_REQUEST, and the
  WorkloadAgent responds by shedding P2/P3 load, reducing peak temperature.
  We report the Task Preservation Index (TPI) and compare peak temperatures
  with vs. without the workload agent.

Plots saved to outputs/:
  workload_nominal_lti.png       — effective vs. original utilization + temp.
  workload_degraded_shedding.png — per-priority load, buffer depth, temp, TPI.

Run from the project root:
    python scripts/validate_workload.py
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
from src.workload_agent import (  # noqa: E402
    PRIORITY_NAMES,
    WorkloadAgent,
)

CONFIG_PATH = ROOT / "config" / "sim_config.yaml"
OUTPUT_DIR = ROOT / "outputs"
NOMINAL_HOURS = 4.0

# --- Degraded stress scenario (same as validate_cooling.py) -----------------
STRESS_T = 6000
STRESS_LOAD = 0.55
FAILURE_START = 1000
PARTIAL_START = 2500
REPAIR_START = 4000
K_FAIL = 1.8
K_PARTIAL = 2.5


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _stress_kcool(t: int, nom_k: float) -> float:
    if t < FAILURE_START:
        return nom_k
    if t < PARTIAL_START:
        return K_FAIL
    if t < REPAIR_START:
        return K_PARTIAL
    return nom_k


# ---------------------------------------------------------------------------
# NOMINAL scenario
# ---------------------------------------------------------------------------

def run_nominal(config: dict, util: np.ndarray):
    """Bursty load, both agents active."""
    n_slots, n = util.shape

    # With both agents.
    rack = Rack.from_config(config, "air")
    cooling = CoolingAgent.from_config(config)
    workload = WorkloadAgent.from_config(config)

    Tmax = np.empty(n)
    eff_total = np.empty(n)
    orig_total = np.empty(n)
    fan = np.empty(n)

    for t in range(n):
        cd = cooling.control(rack, util[:, t])
        wd = workload.step(util[:, t], cd, rack, t)
        rack.step(wd.effective_utilization)
        Tmax[t] = rack.t_node.max()
        eff_total[t] = wd.effective_utilization.sum()
        orig_total[t] = wd.original_utilization.sum()
        fan[t] = cd.omega_fan

    # Without workload agent (cooling only).
    rack_co = Rack.from_config(config, "air")
    cool_co = CoolingAgent.from_config(config)
    Tmax_co = np.empty(n)
    for t in range(n):
        cool_co.control(rack_co, util[:, t])
        rack_co.step(util[:, t])
        Tmax_co[t] = rack_co.t_node.max()

    return Tmax, Tmax_co, eff_total, orig_total, fan, workload.tpi


# ---------------------------------------------------------------------------
# DEGRADED scenario
# ---------------------------------------------------------------------------

def run_degraded(config: dict):
    """CRAC-fault stress: with vs. without workload agent."""
    n_slots = int(config["rack"]["n_slots"])

    # --- WITH workload agent ---
    rack = Rack.from_config(config, "air")
    cooling = CoolingAgent.from_config(config)
    workload = WorkloadAgent.from_config(config)
    nom_k = rack.zone.K_cool

    Tmax = np.empty(STRESS_T)
    eff_total = np.empty(STRESS_T)
    thr = np.zeros(STRESS_T, dtype=bool)
    buffer_depth = np.empty(STRESS_T)
    shed_p = np.zeros((4, STRESS_T))

    for t in range(STRESS_T):
        rack.zone = replace(rack.zone, K_cool=_stress_kcool(t, nom_k))
        u = np.full(n_slots, STRESS_LOAD)
        cd = cooling.control(rack, u)
        wd = workload.step(u, cd, rack, t)
        rack.step(wd.effective_utilization)
        Tmax[t] = rack.t_node.max()
        eff_total[t] = wd.effective_utilization.sum()
        thr[t] = wd.throttle_active
        buffer_depth[t] = wd.deferred_load
        for p in range(4):
            shed_p[p, t] = wd.shed_by_priority[p]

    tpi = workload.tpi
    total_expired = workload.total_expired

    # --- WITHOUT workload agent (cooling only) ---
    rack_no = Rack.from_config(config, "air")
    cool_no = CoolingAgent.from_config(config)
    Tmax_no = np.empty(STRESS_T)
    for t in range(STRESS_T):
        rack_no.zone = replace(rack_no.zone, K_cool=_stress_kcool(t, nom_k))
        cool_no.control(rack_no, STRESS_LOAD)
        rack_no.step(STRESS_LOAD)
        Tmax_no[t] = rack_no.t_node.max()

    return (Tmax, Tmax_no, eff_total, thr, buffer_depth,
            shed_p, tpi, total_expired)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_nominal(Tmax, Tmax_co, eff_total, orig_total, fan) -> Path:
    n = Tmax.shape[0]
    t_min = np.arange(n) / 60.0

    fig, (ax_t, ax_u) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # Temperature panel.
    ax_t.plot(t_min, Tmax_co, color="tab:red", lw=0.8, alpha=0.7,
              label=f"cooling-only: peak {Tmax_co.max():.1f} °C")
    ax_t.plot(t_min, Tmax, color="tab:green", lw=0.9,
              label=f"cooling + workload: peak {Tmax.max():.1f} °C")
    ax_t.axhline(75, color="darkorange", ls="--", lw=1, alpha=0.7, label="75 °C warn")
    ax_t.axhline(80, color="red", ls="--", lw=1, alpha=0.7, label="80 °C critical")
    ax_t.set_ylabel("hottest-slot temperature [°C]")
    ax_t.set_title("Nominal bursty load — LTI routing + priority-aware shedding")
    ax_t.legend(loc="lower right", fontsize=8)

    # Fan overlay.
    axf = ax_t.twinx()
    axf.plot(t_min, fan, color="tab:blue", lw=0.6, alpha=0.4)
    axf.set_ylabel("ω_fan", color="tab:blue")
    axf.tick_params(axis="y", labelcolor="tab:blue")

    # Utilization panel.
    ax_u.plot(t_min, orig_total, color="tab:gray", lw=0.7, alpha=0.6,
              label="original (sum)")
    ax_u.plot(t_min, eff_total, color="tab:blue", lw=0.8,
              label="effective (sum, after agent)")
    ax_u.set_ylabel("total utilization")
    ax_u.set_xlabel("time [min]")
    ax_u.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = OUTPUT_DIR / "workload_nominal_lti.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_degraded(Tmax, Tmax_no, eff_total, thr, buffer_depth,
                  shed_p, tpi) -> Path:
    n = Tmax.shape[0]
    t_min = np.arange(n) / 60.0

    fig, axes = plt.subplots(
        4, 1, figsize=(13, 12), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1, 1]}
    )
    ax_t, ax_s, ax_b, ax_thr = axes

    # Temperature panel.
    ax_t.plot(t_min, Tmax_no, color="tab:red", lw=0.8, alpha=0.7,
              label=f"cooling-only: peak {Tmax_no.max():.1f} °C")
    ax_t.plot(t_min, Tmax, color="tab:green", lw=0.9,
              label=f"+ workload agent: peak {Tmax.max():.1f} °C")
    ax_t.axhline(80, color="red", ls="--", lw=1, alpha=0.6, label="80 °C critical")
    ax_t.axhline(75, color="darkorange", ls="--", lw=1, alpha=0.6, label="75 °C warn")
    ax_t.axhline(72, color="seagreen", ls="--", lw=1, alpha=0.6, label="72 °C release")
    ax_t.set_ylabel("hottest-slot temp [°C]")
    ax_t.set_title(
        f"Degraded cooling (CRAC fault) — TPI = {tpi:.3f}"
    )
    ax_t.legend(loc="upper right", fontsize=7)

    # Per-priority shed panel.
    colours = ["tab:red", "tab:orange", "tab:blue", "tab:gray"]
    bottom = np.zeros(n)
    for p in range(4):
        ax_s.fill_between(t_min, bottom, bottom + shed_p[p], alpha=0.7,
                          color=colours[p], label=PRIORITY_NAMES[p])
        bottom += shed_p[p]
    ax_s.set_ylabel("load shed")
    ax_s.legend(loc="upper right", fontsize=7, ncol=2)

    # Buffer depth panel.
    ax_b.fill_between(t_min, 0, buffer_depth, color="tab:purple", alpha=0.6)
    ax_b.set_ylabel("buffer depth")

    # Throttle timeline.
    ax_thr.fill_between(t_min, 0, thr.astype(float), step="pre",
                        color="firebrick", alpha=0.85)
    ax_thr.set_ylim(-0.1, 1.1)
    ax_thr.set_yticks([0, 1])
    ax_thr.set_yticklabels(["clear", "THROTTLE"])
    ax_thr.set_xlabel("time [min]")

    fig.tight_layout()
    out = OUTPUT_DIR / "workload_degraded_shedding.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    n_slots = int(config["rack"]["n_slots"])

    # ---- Nominal scenario ---------------------------------------------------
    loader = TraceLoader.from_config(config, n_slots)
    bcfg = BurstConfig.from_config(config["burstiness"])
    n_win = min(int(NOMINAL_HOURS * 3600), loader.n_timesteps)
    util = inject_bursts(loader.utilization[:, :n_win], bcfg)

    Tmax, Tmax_co, eff, orig, fan, tpi_nom = run_nominal(config, util)

    print("=== NOMINAL (real bursty load, both agents active) ===")
    print(f"  peak cooling-only          : {Tmax_co.max():.2f} °C")
    print(f"  peak cooling + workload    : {Tmax.max():.2f} °C")
    print(f"  TPI                        : {tpi_nom:.4f}")
    print(f"  effective util range        : {eff.min():.2f}–{eff.max():.2f}")

    # ---- Degraded scenario --------------------------------------------------
    (dT, dT_no, d_eff, d_thr, d_buf,
     d_shed, d_tpi, d_expired) = run_degraded(config)

    print(f"\n=== DEGRADED (CRAC fault, workload agent active) ===")
    print(f"  peak cooling-only          : {dT_no.max():.2f} °C")
    print(f"  peak + workload agent      : {dT.max():.2f} °C")
    print(f"  temperature reduction      : {dT_no.max() - dT.max():.2f} °C")
    print(f"  TPI                        : {d_tpi:.4f}")
    print(f"  total expired load         : {d_expired:.4f}")
    print(f"  throttle active ticks      : {d_thr.sum()}")

    # Assertions.
    assert dT.max() < dT_no.max(), (
        f"WorkloadAgent did not reduce peak: {dT.max():.1f} >= {dT_no.max():.1f}"
    )
    assert d_tpi > 0.3, f"TPI too low: {d_tpi:.3f}"

    # Plots.
    p1 = plot_nominal(Tmax, Tmax_co, eff, orig, fan)
    p2 = plot_degraded(dT, dT_no, d_eff, d_thr, d_buf, d_shed, d_tpi)
    for p in (p1, p2):
        print(f"  saved {p.relative_to(ROOT)}")

    print("\nAll workload-agent validation checks passed.")


if __name__ == "__main__":
    main()

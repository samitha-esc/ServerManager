"""Validate the heterogeneous air/liquid cooling fabric (Phase 3).

Drives one air rack and one liquid rack with the *identical* bursty utilization
array (bursts ON — the stress case) over a multi-hour window, plus a clean
uniform-load steady-state run to expose each zone's vertical gradient. Saves
plots to outputs/:

  1. zones_air_vs_liquid_temperature.png — overlaid per-slot temperature, air vs.
     liquid under the same load, with the 75/80 °C thresholds. Air crosses both;
     liquid stays cool.
  2. zones_gradient_profiles.png — steady-state slot-vs-temperature for both
     zones side by side: air is monotonic, liquid is near-flat.

Reports: air peak, liquid peak (identical load); air vs. liquid vertical
gradient; and the air-zone regression check (air unchanged from Phase 2b).

Run from the project root:
    python scripts/validate_zones.py
"""

from __future__ import annotations

import sys
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
from src.rack import Rack  # noqa: E402
from src.trace_loader import TraceLoader  # noqa: E402

CONFIG_PATH = ROOT / "config" / "sim_config.yaml"
OUTPUT_DIR = ROOT / "outputs"
WINDOW_HOURS = 4.0
WARN_C = 75.0
CRIT_C = 80.0
# Golden values from Phase 2b (air zone) — the alpha refactor must not move them.
PHASE2B_AIR_PEAK = 81.70
PHASE2B_AIR_70PCT = (70.86, 75.13)  # (bottom, top) steady at uniform 70% load


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def drive(rack: Rack, utilization: np.ndarray) -> np.ndarray:
    """Run the rack over a [n_slots, n_steps] utilization, return temp log."""
    n_slots, n_steps = utilization.shape
    rack.reset()
    temps = np.empty((n_slots, n_steps), dtype=np.float64)
    for t in range(n_steps):
        t_node, _, _ = rack.step(utilization[:, t])
        temps[:, t] = t_node
    return temps


def steady_profile(rack: Rack, u: float, steps: int) -> np.ndarray:
    """Run a uniform load to steady state, return the per-slot temperatures."""
    rack.reset()
    t_node = rack.t_node
    for _ in range(steps):
        t_node, _, _ = rack.step(u)
    return t_node.copy()


def plot_temperature(temps_air, temps_liq, n_slots) -> Path:
    t_min = np.arange(temps_air.shape[1]) / 60.0
    fig, ax = plt.subplots(figsize=(12, 6))
    for n in range(n_slots):
        ax.plot(t_min, temps_air[n], color="tab:red", lw=0.6, alpha=0.55,
                label="air" if n == 0 else None)
        ax.plot(t_min, temps_liq[n], color="tab:blue", lw=0.6, alpha=0.7,
                label="liquid" if n == 0 else None)
    ax.axhline(WARN_C, color="darkorange", ls="--", lw=1.4, label=f"{WARN_C:.0f} °C warning")
    ax.axhline(CRIT_C, color="red", ls="--", lw=1.4, label=f"{CRIT_C:.0f} °C critical")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("node temperature [°C]")
    ax.set_title("Air vs. liquid under identical bursty load — liquid stays cool")
    ax.legend(loc="center right")
    fig.tight_layout()
    out = OUTPUT_DIR / "zones_air_vs_liquid_temperature.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_gradients(prof_air, prof_liq) -> Path:
    n_slots = prof_air.shape[0]
    slots = np.arange(n_slots)
    fig, (ax_a, ax_l) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    ax_a.plot(prof_air, slots, "o-", color="tab:red")
    ax_a.set_title(f"Air zone — monotonic gradient ({prof_air[-1] - prof_air[0]:+.2f} °C)")
    ax_a.set_xlabel("steady-state temperature [°C]")
    ax_a.set_ylabel("slot (0 = bottom)")
    ax_a.grid(True, alpha=0.3)

    ax_l.plot(prof_liq, slots, "o-", color="tab:blue")
    ax_l.set_title(f"Liquid zone — near-flat ({prof_liq[-1] - prof_liq[0]:+.2f} °C)")
    ax_l.set_xlabel("steady-state temperature [°C]")
    ax_l.grid(True, alpha=0.3)

    fig.suptitle("Steady-state vertical gradient (uniform 70% load)")
    fig.tight_layout()
    out = OUTPUT_DIR / "zones_gradient_profiles.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    n_slots = int(config["rack"]["n_slots"])

    air = Rack.from_config(config, zone_key="air")
    liquid = Rack.from_config(config, zone_key="liquid")
    print(f"Air   : K_cool={air.zone.K_cool}, T_supply={air.zone.T_supply}, "
          f"alpha={air.zone.alpha}, τ={air.tau:.0f}s")
    print(f"Liquid: K_cool={liquid.zone.K_cool}, T_supply={liquid.zone.T_supply}, "
          f"alpha={liquid.zone.alpha}, τ={liquid.tau:.0f}s")

    # Identical bursty load for both zones (the stress case).
    loader = TraceLoader.from_config(config, n_slots)
    bcfg = BurstConfig.from_config(config["burstiness"])
    n_window = min(int(WINDOW_HOURS * 3600), loader.n_timesteps)
    util = inject_bursts(loader.utilization[:, :n_window], bcfg)

    temps_air = drive(air, util)
    temps_liq = drive(liquid, util)
    air_peak = float(temps_air.max())
    liq_peak = float(temps_liq.max())

    # Clean steady-state gradient under uniform 70% load.
    prof_air = steady_profile(air, 0.70, steps=int(20 * air.tau))
    prof_liq = steady_profile(liquid, 0.70, steps=int(20 * air.tau))
    air_grad = float(prof_air[-1] - prof_air[0])
    liq_grad = float(prof_liq[-1] - prof_liq[0])

    print("\n--- Results (identical bursty load) ---")
    print(f"Air    peak temperature : {air_peak:.2f} °C  "
          f"(slots >75: {int((temps_air.max(axis=1) > WARN_C).sum())}/{n_slots}, "
          f">80: {int((temps_air.max(axis=1) > CRIT_C).sum())}/{n_slots})")
    print(f"Liquid peak temperature : {liq_peak:.2f} °C  "
          f"(slots >75: {int((temps_liq.max(axis=1) > WARN_C).sum())}/{n_slots})")
    print(f"Air    vertical gradient: {air_grad:+.3f} °C (steady 70%, monotonic)")
    print(f"Liquid vertical gradient: {liq_grad:+.3f} °C (steady 70%, near-flat)")

    # --- Air-zone regression: refactor must not change air behaviour ----------
    reg_peak_ok = abs(air_peak - PHASE2B_AIR_PEAK) < 0.01
    reg_70_ok = (abs(prof_air[0] - PHASE2B_AIR_70PCT[0]) < 0.05
                 and abs(prof_air[-1] - PHASE2B_AIR_70PCT[1]) < 0.05)
    print("\n--- Air regression vs. Phase 2b ---")
    print(f"Bursty peak {air_peak:.2f} °C == {PHASE2B_AIR_PEAK} °C : {reg_peak_ok}")
    print(f"70% steady ({prof_air[0]:.2f}/{prof_air[-1]:.2f}) == "
          f"{PHASE2B_AIR_70PCT} : {reg_70_ok}")
    assert reg_peak_ok and reg_70_ok, "AIR REGRESSION FAILED — refactor changed air"

    # Sanity on the physical story.
    assert liq_peak < WARN_C, f"liquid not safe ({liq_peak:.1f} °C)"
    assert air_peak > CRIT_C, f"air did not cross 80 °C ({air_peak:.1f} °C)"
    assert abs(liq_grad) < 0.5 and abs(liq_grad) < abs(air_grad), "liquid gradient not flat"

    p1 = plot_temperature(temps_air, temps_liq, n_slots)
    p2 = plot_gradients(prof_air, prof_liq)
    for p in (p1, p2):
        print(f"  saved {p.relative_to(ROOT)}")

    print("\nAll zone validation checks passed.")


if __name__ == "__main__":
    main()

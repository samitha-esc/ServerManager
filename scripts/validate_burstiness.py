"""Validate the per-node burstiness layer (Phase 2b).

Drives the calibrated rack over a multi-hour window with bursts enabled and
saves plots to outputs/:

  1. burstiness_utilization.png — per-slot utilization heatmap, showing slots
     spiking independently at different times (not in lockstep).
  2. burstiness_temperature.png — per-slot node temperature over time with the
     75 °C and 80 °C thresholds drawn, showing nodes diverging and crossing 75.
  3. burstiness_baseline_vs_bursty.png — per-slot peak temperature, bursts off
     vs. on, making the heating explicit.

Reports the new peak temperature, how many slots cross 75 °C, and the spread
between the hottest and coolest slot's peak (the divergence number).

Run from the project root:
    python scripts/validate_burstiness.py
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

from src.burstiness import BurstConfig, BurstyTraceLoader, inject_bursts  # noqa: E402
from src.rack import Rack  # noqa: E402
from src.trace_loader import TraceLoader  # noqa: E402

CONFIG_PATH = ROOT / "config" / "sim_config.yaml"
OUTPUT_DIR = ROOT / "outputs"
WINDOW_HOURS = 4.0
WARN_C = 75.0
CRIT_C = 80.0


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


def plot_utilization(util: np.ndarray) -> Path:
    n_slots, n_steps = util.shape
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        util,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        extent=(0, n_steps / 60.0, -0.5, n_slots - 0.5),
        interpolation="nearest",
    )
    ax.set_xlabel("time [min]")
    ax.set_ylabel("slot (0 = bottom)")
    ax.set_yticks(range(n_slots))
    ax.set_title("Per-slot utilization with bursts — slots spike independently")
    fig.colorbar(im, ax=ax, label="utilization u")
    fig.tight_layout()
    out = OUTPUT_DIR / "burstiness_utilization.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_temperature(temps: np.ndarray, rack: Rack) -> Path:
    n_slots, n_steps = temps.shape
    t_min = np.arange(n_steps) / 60.0
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("turbo")
    for n in range(n_slots):
        ax.plot(t_min, temps[n], color=cmap(n / max(1, n_slots - 1)), lw=0.8)
    ax.axhline(WARN_C, color="darkorange", ls="--", lw=1.4, label=f"{WARN_C:.0f} °C warning")
    ax.axhline(CRIT_C, color="red", ls="--", lw=1.4, label=f"{CRIT_C:.0f} °C critical")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("node temperature [°C]")
    ax.set_title("Per-slot node temperature under bursty load — nodes diverge, cross 75 °C")
    ax.legend(loc="lower right")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_slots - 1))
    fig.colorbar(sm, ax=ax, label="slot (0 = bottom)")
    fig.tight_layout()
    out = OUTPUT_DIR / "burstiness_temperature.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_comparison(peaks_base: np.ndarray, peaks_burst: np.ndarray) -> Path:
    n_slots = peaks_base.shape[0]
    slots = np.arange(n_slots)
    width = 0.4
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(slots - width / 2, peaks_base, width, label="bursts OFF (baseline)",
           color="tab:blue")
    ax.bar(slots + width / 2, peaks_burst, width, label="bursts ON",
           color="tab:red")
    ax.axhline(WARN_C, color="darkorange", ls="--", lw=1.3, label=f"{WARN_C:.0f} °C warning")
    ax.axhline(CRIT_C, color="red", ls="--", lw=1.3, label=f"{CRIT_C:.0f} °C critical")
    ax.set_xlabel("slot (0 = bottom)")
    ax.set_ylabel("peak temperature [°C]")
    ax.set_xticks(slots)
    ax.set_title("Peak temperature per slot — baseline vs. bursty")
    ax.legend(loc="lower right", ncol=2)
    ax.set_ylim(40, CRIT_C + 8)
    fig.tight_layout()
    out = OUTPUT_DIR / "burstiness_baseline_vs_bursty.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    rack = Rack.from_config(config, zone_key="air")

    # Baseline (Phase 2) and bursty utilization over the same window.
    loader = TraceLoader.from_config(config, rack.n_slots)
    cfg = BurstConfig.from_config(config["burstiness"])
    n_window = min(int(WINDOW_HOURS * 3600), loader.n_timesteps)
    base_util = loader.utilization[:, :n_window]
    burst_util = inject_bursts(base_util, cfg)

    print(f"Window: {n_window} s ({n_window / 3600:.1f} h), {rack.n_slots} slots, "
          f"τ ≈ {rack.tau:.0f} s")
    print(f"Burst config: arrival {cfg.arrival_per_hour}/h, "
          f"magnitude {cfg.magnitude_min}-{cfg.magnitude_max}, "
          f"duration {cfg.duration_min_s}-{cfg.duration_max_s} s "
          f"({cfg.duration_min_s / rack.tau:.0f}-{cfg.duration_max_s / rack.tau:.0f} × τ)")

    umax = float(burst_util.max())
    print(f"\nBursty utilization: min {burst_util.min():.3f}, max {umax:.3f}")
    assert umax <= 1.0, f"utilization exceeded 1.0 ({umax})"

    temps_base = drive(rack, base_util)
    temps_burst = drive(rack, burst_util)

    peaks_base = temps_base.max(axis=1)
    peaks_burst = temps_burst.max(axis=1)
    n_cross = int((peaks_burst > WARN_C).sum())
    peak_overall = float(peaks_burst.max())
    divergence = float(peaks_burst.max() - peaks_burst.min())
    # Instantaneous spread: how far apart slots get at the same moment.
    inst_spread = float((temps_burst.max(axis=0) - temps_burst.min(axis=0)).max())

    print("\n--- Results ---")
    print(f"Baseline (bursts off) peak temperature : {temps_base.max():.2f} °C")
    print(f"Bursty   (bursts on)  peak temperature : {peak_overall:.2f} °C")
    print(f"Slots crossing {WARN_C:.0f} °C at any point   : {n_cross} / {rack.n_slots}")
    print(f"Slots crossing {CRIT_C:.0f} °C at any point   : "
          f"{int((peaks_burst > CRIT_C).sum())} / {rack.n_slots}")
    print(f"Divergence (hottest − coolest peak)    : {divergence:.2f} °C")
    print(f"Max instantaneous slot-to-slot spread  : {inst_spread:.2f} °C")

    # The Phase 2b control problem must now exist.
    assert n_cross >= 1, "no slot crossed the 75 °C warning threshold"
    assert inst_spread > 3.0, f"nodes did not visibly diverge ({inst_spread:.1f} °C)"

    p1 = plot_utilization(burst_util)
    p2 = plot_temperature(temps_burst, rack)
    p3 = plot_comparison(peaks_base, peaks_burst)
    for p in (p1, p2, p3):
        print(f"  saved {p.relative_to(ROOT)}")

    print("\nAll burstiness validation checks passed.")


if __name__ == "__main__":
    main()

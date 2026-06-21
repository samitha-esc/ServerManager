"""Validate the single-rack thermal model and save plots to outputs/.

Two scenarios:
  1. Constant 70% uniform load run to steady state. Plots temperature vs. time
     per slot and the steady-state slot-vs-temperature gradient. Asserts the
     steady-state gradient increases monotonically up the rack.
  2. A 30% -> 90% load step. Plots the thermal response and reports the measured
     time constant, checking for a sensible value and no oscillation/instability.

Run from the project root:
    python scripts/validate_thermal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Make `src` importable when run as a plain script from the project root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252; ensure unicode (°, τ) prints cleanly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover - older/odd streams
    pass

from src.rack import Rack  # noqa: E402

CONFIG_PATH = ROOT / "config" / "sim_config.yaml"
OUTPUT_DIR = ROOT / "outputs"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_constant_load(rack: Rack, utilization: float, steps: int) -> pd.DataFrame:
    """Run a constant uniform load, logging per-slot node temperatures."""
    rack.reset()
    records: list[np.ndarray] = []
    for _ in range(steps):
        t_node, _, _ = rack.step(utilization)
        records.append(t_node.copy())
    df = pd.DataFrame(records, columns=[f"slot_{n}" for n in range(rack.n_slots)])
    df.index.name = "t_s"
    return df


def run_load_step(
    rack: Rack, u_lo: float, u_hi: float, steps_each: int
) -> tuple[pd.DataFrame, int]:
    """Run u_lo to steady state, then step to u_hi. Returns log + step index."""
    rack.reset()
    records: list[np.ndarray] = []
    for _ in range(steps_each):
        t_node, _, _ = rack.step(u_lo)
        records.append(t_node.copy())
    step_index = len(records)
    for _ in range(steps_each):
        t_node, _, _ = rack.step(u_hi)
        records.append(t_node.copy())
    df = pd.DataFrame(records, columns=[f"slot_{n}" for n in range(rack.n_slots)])
    df.index.name = "t_s"
    return df, step_index


def measure_time_constant(
    response: np.ndarray, t0_value: float, steady_value: float
) -> float:
    """Time (in steps/seconds, dt=1) to reach 63.2% of the total rise."""
    target = t0_value + 0.632 * (steady_value - t0_value)
    crossings = np.where(response >= target)[0]
    return float(crossings[0]) if crossings.size else float("nan")


def plot_constant_load(df: pd.DataFrame, rack: Rack) -> Path:
    fig, (ax_t, ax_g) = plt.subplots(1, 2, figsize=(13, 5))

    cmap = plt.get_cmap("viridis")
    for n in range(rack.n_slots):
        ax_t.plot(
            df.index,
            df[f"slot_{n}"],
            color=cmap(n / max(1, rack.n_slots - 1)),
            lw=1.2,
        )
    ax_t.axhline(rack.zone.T_supply, color="grey", ls="--", lw=1, label="supply")
    ax_t.set_xlabel("time [s]")
    ax_t.set_ylabel("node temperature [°C]")
    ax_t.set_title("Constant 70% load — temperature vs. time per slot")
    ax_t.legend(loc="lower right")
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(0, rack.n_slots - 1)
    )
    fig.colorbar(sm, ax=ax_t, label="slot (0 = bottom)")

    steady = df.iloc[-1].to_numpy()
    slots = np.arange(rack.n_slots)
    ax_g.plot(steady, slots, "o-", color="firebrick")
    ax_g.set_xlabel("steady-state temperature [°C]")
    ax_g.set_ylabel("slot (0 = bottom)")
    ax_g.set_title("Steady-state vertical gradient")
    ax_g.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "constant_load_70pct.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_load_step(df: pd.DataFrame, step_index: int, rack: Rack, tau: float) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.get_cmap("plasma")
    for n in range(rack.n_slots):
        ax.plot(
            df.index,
            df[f"slot_{n}"],
            color=cmap(n / max(1, rack.n_slots - 1)),
            lw=1.2,
        )
    ax.axvline(step_index, color="black", ls=":", lw=1.2, label="load step 30%→90%")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("node temperature [°C]")
    ax.set_title(
        f"Load step 30% → 90% — measured τ ≈ {tau:.0f} s (analytic {rack.tau:.0f} s)"
    )
    ax.legend(loc="lower right")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, rack.n_slots - 1))
    fig.colorbar(sm, ax=ax, label="slot (0 = bottom)")
    fig.tight_layout()
    out = OUTPUT_DIR / "load_step_30_90.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def check_no_oscillation(response: np.ndarray) -> bool:
    """A first-order rise must be monotonic (no overshoot/oscillation)."""
    diffs = np.diff(response)
    # Allow tiny negative numerical noise; reject genuine ringing.
    return bool(np.all(diffs >= -1e-6))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    rack = Rack.from_config(config, zone_key="air")

    print(f"Rack: {rack.n_slots} slots in zone '{rack.zone.name}' "
          f"({rack.zone.kind}), analytic τ = {rack.tau:.1f} s")

    # --- Scenario 1: constant 70% load to steady state -----------------------
    steps = max(600, int(8 * rack.tau))
    df_const = run_constant_load(rack, utilization=0.70, steps=steps)
    steady = df_const.iloc[-1].to_numpy()

    gradient = np.diff(steady)
    monotonic = bool(np.all(gradient > 0))
    print("\n[Scenario 1] Constant 70% load")
    print(f"  bottom slot: {steady[0]:.2f} °C   top slot: {steady[-1]:.2f} °C")
    print(f"  top-to-bottom gradient: {steady[-1] - steady[0]:.2f} °C")
    print(f"  supply temp: {rack.zone.T_supply:.1f} °C")
    print(f"  monotonic increasing up the rack: {monotonic}")
    assert monotonic, "Steady-state gradient is NOT monotonically increasing up the rack"
    p1 = plot_constant_load(df_const, rack)
    print(f"  saved {p1.relative_to(ROOT)}")

    # --- Scenario 2: load step 30% -> 90% ------------------------------------
    steps_each = max(400, int(6 * rack.tau))
    df_step, step_index = run_load_step(rack, u_lo=0.30, u_hi=0.90, steps_each=steps_each)

    # Analyse the bottom slot's response to the step.
    probe = "slot_0"
    pre = df_step[probe].iloc[step_index - 1]
    post = df_step[probe].iloc[step_index:].to_numpy()
    final = df_step[probe].iloc[-1]
    tau_meas = measure_time_constant(post, pre, final)
    stable = check_no_oscillation(post)

    print("\n[Scenario 2] Load step 30% → 90%")
    print(f"  pre-step temp:  {pre:.2f} °C   post-step steady: {final:.2f} °C")
    print(f"  measured τ (63.2% rise): {tau_meas:.1f} s   analytic τ: {rack.tau:.1f} s")
    print(f"  monotonic / no oscillation: {stable}")
    assert stable, "Load-step response shows oscillation/instability"
    assert 30.0 <= tau_meas <= 120.0, f"time constant {tau_meas:.1f}s outside 30–120 s"
    p2 = plot_load_step(df_step, step_index, rack, tau_meas)
    print(f"  saved {p2.relative_to(ROOT)}")

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()

"""Drive the calibrated thermal model with the real Alibaba trace (Phase 2).

Runs the existing single-rack thermal model under the trace-derived per-slot
utilization over the first few simulated hours and saves plots to outputs/:

  1. trace_input_signal.png   — raw 300 s trace points vs. the 1 s interpolated
     curve, to confirm the interpolation is sane.
  2. trace_rack_temperature.png — per-slot node temperature over time under the
     real load, with the per-slot time-averaged gradient alongside.
  3. trace_thermal_lag.png    — a zoom of the instantaneous-equilibrium temp vs.
     the actual (lagged) temperature, visualising the thermal time constant.

Checks (assertions):
  * no utilization ever exceeds 1.0 after variation + clipping;
  * the vertical gradient is preserved (top slots hotter than bottom on average);
  * temperature follows load with the expected ~60 s lag (recovered by fitting
    the smoothing time constant that maps equilibrium temp to actual temp).

Run from the project root:
    python scripts/validate_trace.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252; ensure unicode (°, τ) prints cleanly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass

import yaml  # noqa: E402

from src.rack import Rack  # noqa: E402
from src.trace_loader import TraceLoader  # noqa: E402

CONFIG_PATH = ROOT / "config" / "sim_config.yaml"
OUTPUT_DIR = ROOT / "outputs"
WINDOW_HOURS = 3.0  # simulated window driven by the trace


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_window(rack: Rack, loader: TraceLoader, n_steps: int) -> np.ndarray:
    """Step the rack for n_steps, returning a [n_slots, n_steps] temp log."""
    rack.reset()
    temps = np.empty((rack.n_slots, n_steps), dtype=np.float64)
    for t in range(n_steps):
        t_node, _, _ = rack.step(loader.utilization_at(t))
        temps[:, t] = t_node
    return temps


def equilibrium_temp(rack: Rack, utilization: np.ndarray) -> np.ndarray:
    """Instantaneous steady-state temp a slot would reach at this utilization.

    For the bottom slot the inlet equals the supply temperature, so
    T_eq = T_supply + P(u) * K_heat / K_cool. The actual temperature relaxes
    toward this with the thermal time constant; the gap is the lag.
    """
    p = rack.power_model.power(utilization)
    return rack.zone.T_supply + p * rack.thermal.K_heat / rack.zone.K_cool


def _ema(target: np.ndarray, tau: float) -> np.ndarray:
    """First-order smoothing of `target` with time constant tau (dt = 1 s)."""
    out = np.empty_like(target)
    out[0] = target[0]
    a = 1.0 / tau
    for t in range(1, target.shape[0]):
        out[t] = out[t - 1] + a * (target[t - 1] - out[t - 1])
    return out


def recover_time_constant(
    actual: np.ndarray, eq: np.ndarray, warmup: int = 600
) -> tuple[float, float]:
    """Time constant whose EMA of equilibrium temp best fits the actual temp.

    Skips an initial warm-up (cold start from supply temp) so the transient
    doesn't bias the fit. Returns (best_tau, rmse).
    """
    best_tau, best_rmse = float("nan"), float("inf")
    for tau in range(5, 201):
        fit = _ema(eq, float(tau))
        rmse = float(np.sqrt(np.mean((fit[warmup:] - actual[warmup:]) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_tau = rmse, float(tau)
    return best_tau, best_rmse


def plot_input_signal(loader: TraceLoader, n_steps: int) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    t_1s = np.arange(n_steps)
    ax.plot(
        t_1s / 60.0,
        loader.signal_1s[:n_steps],
        color="tab:blue",
        lw=1.3,
        label="1 s linear interpolation",
        zorder=1,
    )
    mask = loader.source_times < n_steps
    ax.scatter(
        loader.source_times[mask] / 60.0,
        loader.source_fraction[mask],
        color="tab:red",
        s=28,
        zorder=2,
        label="raw 300 s trace points",
    )
    ax.set_xlabel("time [min]")
    ax.set_ylabel("CPU utilization fraction u")
    ax.set_ylim(0, 1)
    ax.set_title("Trace input signal — raw 300 s points vs. 1 s interpolation")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "trace_input_signal.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_rack_temperature(temps: np.ndarray, rack: Rack) -> Path:
    n_slots, n_steps = temps.shape
    t_min = np.arange(n_steps) / 60.0
    fig, (ax_t, ax_g) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1]}
    )

    cmap = plt.get_cmap("inferno")
    for n in range(n_slots):
        ax_t.plot(t_min, temps[n], color=cmap(n / max(1, n_slots - 1)), lw=0.9)
    ax_t.axhline(rack.zone.T_supply, color="grey", ls="--", lw=1, label="supply")
    ax_t.set_xlabel("time [min]")
    ax_t.set_ylabel("node temperature [°C]")
    ax_t.set_title("Per-slot node temperature under real Alibaba load")
    ax_t.legend(loc="lower right")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_slots - 1))
    fig.colorbar(sm, ax=ax_t, label="slot (0 = bottom)")

    mean_profile = temps.mean(axis=1)
    ax_g.plot(mean_profile, np.arange(n_slots), "o-", color="tab:blue")
    ax_g.set_xlabel("time-averaged temp [°C]")
    ax_g.set_ylabel("slot (0 = bottom)")
    ax_g.set_title("Vertical gradient")
    ax_g.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "trace_rack_temperature.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_thermal_lag(
    actual: np.ndarray, eq: np.ndarray, tau: float, window: tuple[int, int]
) -> Path:
    lo, hi = window
    t_min = np.arange(lo, hi) / 60.0
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t_min, eq[lo:hi], color="tab:orange", lw=1.4,
            label="instantaneous equilibrium T_eq(u)")
    ax.plot(t_min, actual[lo:hi], color="tab:blue", lw=1.6,
            label="actual node temperature (lagged)")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("temperature [°C]")
    ax.set_title(f"Thermal lag (bottom slot) — recovered τ ≈ {tau:.0f} s")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "trace_thermal_lag.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    rack = Rack.from_config(config, zone_key="air")
    loader = TraceLoader.from_config(config, rack.n_slots)

    n_window = min(int(WINDOW_HOURS * 3600), loader.n_timesteps)
    print(f"Trace: {loader.source_fraction.shape[0]} raw points @ "
          f"{loader.source_stride_s} s -> {loader.n_timesteps} samples @ 1 s")
    print(f"Driving rack for {n_window} s ({n_window / 3600:.1f} h), "
          f"{rack.n_slots} slots, analytic τ = {rack.tau:.0f} s")

    # Sanity: the [0,100]->[0,1] conversion + variation must never exceed 1.0.
    umax = float(loader.utilization.max())
    umin = float(loader.utilization.min())
    print(f"\nUtilization after variation+clip: min {umin:.3f}, max {umax:.3f}")
    assert umax <= 1.0, f"utilization exceeded 1.0 ({umax})"
    assert umin >= 0.0, f"utilization below 0.0 ({umin})"

    # Expected interpolated length, given source points and stride.
    n_src = loader.source_fraction.shape[0]
    expected_len = (n_src - 1) * loader.source_stride_s + 1
    assert loader.n_timesteps == expected_len, (
        f"interp length {loader.n_timesteps} != expected {expected_len}"
    )
    print(f"Interpolated length check: {loader.n_timesteps} == {expected_len} ✓")

    temps = run_window(rack, loader, n_window)

    # Gradient preserved: top slots hotter than bottom on time-average.
    mean_profile = temps.mean(axis=1)
    slope = float(np.polyfit(np.arange(rack.n_slots), mean_profile, 1)[0])
    top_hotter = mean_profile[-1] > mean_profile[0]
    print(f"\nGradient: bottom {mean_profile[0]:.2f} °C, top "
          f"{mean_profile[-1]:.2f} °C, profile slope {slope:+.3f} °C/slot")
    assert top_hotter, "top slot not hotter than bottom — gradient lost"
    assert slope > 0, f"vertical profile slope not positive ({slope:.3f})"

    # Lag: recover the smoothing time constant between equilibrium and actual.
    u_bottom = loader.utilization[0, :n_window]
    eq_bottom = equilibrium_temp(rack, u_bottom)
    tau_rec, rmse = recover_time_constant(temps[0], eq_bottom)
    print(f"Lag: recovered τ {tau_rec:.0f} s (analytic {rack.tau:.0f} s, "
          f"fit RMSE {rmse:.3f} °C)")
    assert 40.0 <= tau_rec <= 90.0, f"recovered τ {tau_rec:.0f}s far from ~60 s"

    peak = float(temps.max())
    peak_slot, peak_t = np.unravel_index(np.argmax(temps), temps.shape)
    print(f"\nPeak temperature: {peak:.2f} °C "
          f"(slot {peak_slot}, t = {peak_t} s)")

    p1 = plot_input_signal(loader, n_window)
    p2 = plot_rack_temperature(temps, rack)
    # Zoom the lag plot onto a busy mid-window minute range.
    lo = min(1500, n_window // 4)
    hi = min(lo + 1800, n_window)
    p3 = plot_thermal_lag(temps[0], eq_bottom, tau_rec, (lo, hi))
    for p in (p1, p2, p3):
        print(f"  saved {p.relative_to(ROOT)}")

    print("\nAll trace validation checks passed.")


if __name__ == "__main__":
    main()

"""Validate the composed full stack over a server room (Phase 7).

Runs the whole control stack — cross-zone router -> per-rack Cooling Agent
(fans/pump) -> per-rack Workload Agent -> thermal + energy accounting — over a
~50-rack room (35 air + 15 liquid), comparing two routing policies:

  ROUTED — liquid-first cross-zone routing (the full stack).
  NAIVE  — even split across all racks regardless of zone.

Two views:
  1. Time series at a busy demand: room air/liquid temperature and cooling power
     over time, routed vs. naive.
  2. Energy-vs-load sweep: steady-state cooling power across demand levels,
     showing routing's cooling-energy savings grow as the room gets busier (it
     keeps air racks out of the cube-law fan-ramp regime).

Reports peak temperatures, IT/cooling energy, PUE, TPI, and the cooling-energy
savings of routing vs. naive.

Run from the project root:
    python scripts/validate_room.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
from src.room import Room  # noqa: E402
from src.trace_loader import (  # noqa: E402
    TraceConfig,
    interpolate_linear,
    load_cpu_fraction,
    per_slot_variation,
)

CONFIG_PATH = ROOT / "config" / "sim_config.yaml"
OUTPUT_DIR = ROOT / "outputs"
WINDOW_S = 2700           # 45 min time-series (keeps the 50-rack run quick)
BUSY_SCALE = 1.7          # scale aggregate demand to a busy room (~0.7 util)
SWEEP_FRACS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # demand as fraction of total slots
SWEEP_TICKS = 400         # steady-state run per sweep point
WARN_C = 75.0
J_TO_KWH = 1.0 / 3.6e6


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_timeseries(config, demand_series, route):
    room = Room(config, route=route)
    n = demand_series.shape[0]
    air_T = np.empty(n)
    liq_T = np.empty(n)
    cool_kw = np.empty(n)
    for t in range(n):
        res = room.step(float(demand_series[t]), t)
        air_T[t] = res.max_temp_air
        liq_T[t] = res.max_temp_liquid
        cool_kw[t] = res.cooling_power_w / 1000.0
    return room, air_T, liq_T, cool_kw


def run_steady(config, demand, route, ticks):
    room = Room(config, route=route)
    res = None
    for t in range(ticks):
        res = room.step(demand, t)
    return res, room


def plot_timeseries(air_r, liq_r, air_n, liq_n) -> Path:
    t_min = np.arange(air_r.shape[0]) / 60.0
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(t_min, air_n, color="tab:red", lw=0.9,
            label=f"NAIVE hottest air rack: peak {air_n.max():.1f} °C")
    ax.plot(t_min, air_r, color="tab:green", lw=0.9,
            label=f"ROUTED hottest air rack: peak {air_r.max():.1f} °C")
    ax.plot(t_min, liq_r, color="tab:blue", lw=0.8, alpha=0.7,
            label=f"ROUTED hottest liquid rack: peak {liq_r.max():.1f} °C")
    ax.axhline(WARN_C, color="darkorange", ls="--", lw=1, label="75 °C warn")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("hottest-rack temperature [°C]")
    ax.set_title("Server room (50 racks) — routed keeps air racks off the ramp cliff")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = OUTPUT_DIR / "room_temps_routed_vs_naive.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_energy_sweep(fracs, cool_r, cool_n, pue_r, pue_n) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(fracs, cool_n, "o-", color="tab:red", label="naive")
    ax1.plot(fracs, cool_r, "o-", color="tab:green", label="routed (liquid-first)")
    ax1.set_xlabel("room demand (fraction of total slot capacity)")
    ax1.set_ylabel("steady cooling power [kW]")
    ax1.set_title("Cooling power vs. load — routing avoids the cube-law ramp")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(fracs, pue_n, "o-", color="tab:red", label="naive")
    ax2.plot(fracs, pue_r, "o-", color="tab:green", label="routed")
    ax2.set_xlabel("room demand (fraction of total slot capacity)")
    ax2.set_ylabel("steady-state PUE")
    ax2.set_title("PUE vs. load")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "room_energy_vs_load.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    n_slots = int(config["rack"]["n_slots"])
    n_racks = int(config["room"]["n_air_racks"]) + int(config["room"]["n_liquid_racks"])
    total_slots = n_racks * n_slots

    # Realistic aggregate room demand: sum of one bursty stream per (rack, slot),
    # built only for the window (avoids materialising the full-length trace for
    # all 750 slots).
    tcfg = TraceConfig.from_config(config["trace"])
    base = load_cpu_fraction(ROOT / tcfg.path, tcfg.cpu_column)
    signal = interpolate_linear(base, tcfg.source_stride_s)[:WINDOW_S]
    mult, off = per_slot_variation(total_slots, tcfg.seed,
                                   tcfg.scale_spread, tcfg.offset_spread)
    util = np.clip(signal[None, :] * mult[:, None] + off[:, None], 0.0, 1.0)
    bcfg = BurstConfig.from_config(config["burstiness"])
    bursty = inject_bursts(util, bcfg)
    demand_series = bursty.sum(axis=0) * BUSY_SCALE

    print(f"Room: {n_racks} racks ({config['room']['n_air_racks']} air + "
          f"{config['room']['n_liquid_racks']} liquid), {total_slots} slots, "
          f"liquid capacity {int(config['room']['n_liquid_racks']) * n_slots}")
    print(f"Busy time series: {WINDOW_S}s, demand mean {demand_series.mean():.0f} / "
          f"peak {demand_series.max():.0f} (capacity {total_slots})")

    # --- Time series ---------------------------------------------------------
    room_r, air_r, liq_r, cool_r_ts = run_timeseries(config, demand_series, route=True)
    room_n, air_n, liq_n, cool_n_ts = run_timeseries(config, demand_series, route=False)

    print("\n=== Time series (busy room) ===")
    print(f"  ROUTED: air peak {air_r.max():.1f} °C  liquid peak {liq_r.max():.1f} °C  "
          f"IT {room_r.it_energy_j * J_TO_KWH:.1f} kWh  cooling "
          f"{room_r.cooling_energy_j * J_TO_KWH:.2f} kWh  PUE {room_r.pue:.3f}  "
          f"TPI {room_r.tpi:.3f}")
    print(f"  NAIVE : air peak {air_n.max():.1f} °C  liquid peak {liq_n.max():.1f} °C  "
          f"IT {room_n.it_energy_j * J_TO_KWH:.1f} kWh  cooling "
          f"{room_n.cooling_energy_j * J_TO_KWH:.2f} kWh  PUE {room_n.pue:.3f}  "
          f"TPI {room_n.tpi:.3f}")
    cool_save = (room_n.cooling_energy_j - room_r.cooling_energy_j) / room_n.cooling_energy_j
    print(f"  cooling-energy saving (routed vs naive): {100 * cool_save:.1f}%")

    # --- Energy-vs-load sweep ------------------------------------------------
    print("\n=== Energy-vs-load sweep (steady state) ===")
    cool_r, cool_n, pue_r, pue_n = [], [], [], []
    for frac in SWEEP_FRACS:
        d = frac * total_slots
        res_r, _ = run_steady(config, d, True, SWEEP_TICKS)
        res_n, _ = run_steady(config, d, False, SWEEP_TICKS)
        cool_r.append(res_r.cooling_power_w / 1000.0)
        cool_n.append(res_n.cooling_power_w / 1000.0)
        pue_r.append(res_r.pue)
        pue_n.append(res_n.pue)
        print(f"  frac {frac:.2f}: cooling kW routed {cool_r[-1]:.2f} / naive "
              f"{cool_n[-1]:.2f} | PUE routed {pue_r[-1]:.3f} / naive {pue_n[-1]:.3f}")

    # --- Checks --------------------------------------------------------------
    assert air_r.max() < air_n.max(), "routing did not keep air cooler"
    assert room_r.cooling_energy_j <= room_n.cooling_energy_j + 1e-6, \
        "routing should not cost more cooling energy"
    assert liq_r.max() < 80.0, "liquid overheated"
    assert cool_r[-1] < cool_n[-1], "at high load routing should cool cheaper"

    p1 = plot_timeseries(air_r, liq_r, air_n, liq_n)
    p2 = plot_energy_sweep(SWEEP_FRACS, cool_r, cool_n, pue_r, pue_n)
    for p in (p1, p2):
        print(f"  saved {p.relative_to(ROOT)}")

    print("\nAll server-room validation checks passed.")


if __name__ == "__main__":
    main()

"""Validate cross-zone air/liquid routing (Phase 6).

Drives a 2-rack cluster (one air rack + one liquid rack, fixed fans to isolate
the routing effect) under a busy cluster demand, comparing two placement
policies:

  ROUTED — liquid-first fill, spill to air (the Phase 6 router). Liquid (cool,
  finite) carries the load; air is used only for the overflow once liquid is at
  capacity, so air is kept as a cold reserve.

  NAIVE — even 50/50 split of demand across both zones (no thermal awareness).

Demand is the existing trace+burstiness signal scaled to a cluster level so it
periodically exceeds one zone's capacity (DEMAND_SCALE) — that is what makes the
routing decision matter. The router conserves total load; it touches neither the
substrate nor the cooling/workload agents (those compose on top later).

Plots saved to outputs/:
  routing_air_vs_liquid_temp.png — air & liquid temperature, routed vs. naive.
  routing_load_split.png         — liquid vs. air (spill) load over time.

Run from the project root:
    python scripts/validate_routing.py
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
from src.cross_zone_router import CrossZoneRouter  # noqa: E402
from src.rack import Rack  # noqa: E402
from src.trace_loader import TraceLoader  # noqa: E402

CONFIG_PATH = ROOT / "config" / "sim_config.yaml"
OUTPUT_DIR = ROOT / "outputs"
WINDOW_HOURS = 4.0
# Cluster-demand scale: the trace is sized for one rack, so scale it up to a
# 2-rack cluster level where peaks exceed one zone's capacity and routing bites.
DEMAND_SCALE = 2.5
WARN_C = 75.0
CRIT_C = 80.0


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_routed(config, demand):
    """Liquid-first routing across an air + liquid rack (fixed fans)."""
    n_slots, n = demand.shape
    router = CrossZoneRouter.from_config(config, n_slots)
    air = Rack.from_config(config, "air")
    liquid = Rack.from_config(config, "liquid")
    Ta = np.empty(n)
    Tl = np.empty(n)
    liq_load = np.empty(n)
    air_load = np.empty(n)
    conserved = True
    for t in range(n):
        dec = router.route(demand[:, t])
        if abs((dec.u_liquid.sum() + dec.u_air.sum()) - demand[:, t].sum()) > 1e-9:
            conserved = False
        liquid.step(dec.u_liquid)
        air.step(dec.u_air)
        Ta[t] = air.t_node.max()
        Tl[t] = liquid.t_node.max()
        liq_load[t] = dec.liquid_total
        air_load[t] = dec.air_total
    return Ta, Tl, liq_load, air_load, conserved


def run_naive(config, demand):
    """Even 50/50 split across both zones (no thermal awareness)."""
    n_slots, n = demand.shape
    air = Rack.from_config(config, "air")
    liquid = Rack.from_config(config, "liquid")
    Ta = np.empty(n)
    Tl = np.empty(n)
    for t in range(n):
        half = demand[:, t].sum() * 0.5
        per_slot = min(1.0, half / n_slots)
        u = np.full(n_slots, per_slot)
        air.step(u)
        liquid.step(u)
        Ta[t] = air.t_node.max()
        Tl[t] = liquid.t_node.max()
    return Ta, Tl


def plot_temps(t_min, r_air, r_liq, n_air, n_liq) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(t_min, r_liq, color="tab:blue", lw=0.9,
            label=f"ROUTED liquid: peak {r_liq.max():.1f} °C")
    ax.plot(t_min, r_air, color="tab:cyan", lw=0.9,
            label=f"ROUTED air (reserve): peak {r_air.max():.1f} °C")
    ax.plot(t_min, n_air, color="tab:red", lw=0.8, alpha=0.7,
            label=f"NAIVE air (50/50): peak {n_air.max():.1f} °C")
    ax.plot(t_min, n_liq, color="tab:orange", lw=0.8, alpha=0.6,
            label=f"NAIVE liquid (50/50): peak {n_liq.max():.1f} °C")
    ax.axhline(WARN_C, color="darkorange", ls="--", lw=1, label="75 °C warn")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("hottest-slot temperature [°C]")
    ax.set_title("Cross-zone routing — liquid-first keeps air as a cool reserve")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = OUTPUT_DIR / "routing_air_vs_liquid_temp.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_load_split(t_min, liq_load, air_load, n_slots) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(t_min, 0, liq_load, color="tab:blue", alpha=0.6,
                    label="liquid load (filled first)")
    ax.fill_between(t_min, liq_load, liq_load + air_load, color="tab:red",
                    alpha=0.6, label="air load (spill / overflow)")
    ax.axhline(n_slots, color="black", ls="--", lw=1.2,
               label=f"liquid capacity = {n_slots}")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("cluster load placed [util units]")
    ax.set_title("Routed load split — liquid saturates, then spills to air")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = OUTPUT_DIR / "routing_load_split.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    n_slots = int(config["rack"]["n_slots"])

    loader = TraceLoader.from_config(config, n_slots)
    bcfg = BurstConfig.from_config(config["burstiness"])
    n = min(int(WINDOW_HOURS * 3600), loader.n_timesteps)
    demand = inject_bursts(loader.utilization[:, :n], bcfg) * DEMAND_SCALE

    r_air, r_liq, liq_load, air_load, conserved = run_routed(config, demand)
    n_air, n_liq = run_naive(config, demand)

    spill_ticks = int(np.sum(air_load > 1e-6))
    print(f"Cluster: air + liquid rack, {n_slots} slots each, demand scale "
          f"{DEMAND_SCALE} (peak {demand.sum(axis=0).max():.1f} vs liquid cap {n_slots})")
    print("\n=== ROUTED (liquid-first, spill to air) ===")
    print(f"  liquid peak : {r_liq.max():.2f} °C   air peak (reserve): {r_air.max():.2f} °C")
    print(f"  spill ticks : {spill_ticks}/{n} ({100 * spill_ticks / n:.0f}%)")
    print("=== NAIVE (even 50/50 split) ===")
    print(f"  liquid peak : {n_liq.max():.2f} °C   air peak: {n_air.max():.2f} °C")
    print(f"\n  air peak reduction (routed vs naive): "
          f"{n_air.max() - r_air.max():.2f} °C")
    print(f"  cluster peak routed/naive: {max(r_air.max(), r_liq.max()):.2f} / "
          f"{max(n_air.max(), n_liq.max()):.2f} °C")

    # Checks.
    assert conserved, "router did not conserve total load"
    assert r_air.max() < n_air.max(), "routing did not keep air cooler than naive"
    assert r_liq.max() < WARN_C, f"liquid exceeded safe zone ({r_liq.max():.1f} °C)"

    t_min = np.arange(n) / 60.0
    p1 = plot_temps(t_min, r_air, r_liq, n_air, n_liq)
    p2 = plot_load_split(t_min, liq_load, air_load, n_slots)
    for p in (p1, p2):
        print(f"  saved {p.relative_to(ROOT)}")

    print("\nAll cross-zone routing validation checks passed.")


if __name__ == "__main__":
    main()

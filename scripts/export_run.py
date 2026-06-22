"""Export a simulation run to a JSON file for the visualization team (Phase 8).

Runs the room simulation and writes a single JSON file:

    { "metadata": {...}, "topology": {...}, "frames": [ {...}, ... ] }

The physics advances every 1 s tick; frames are recorded every
``--stride`` seconds to keep the file small. Per-slot temperatures are included
only with ``--slots``. See SCHEMA.md for the field contract.

Usage (from the project root):
    python scripts/export_run.py                       # 30 min, 5 s stride, routed
    python scripts/export_run.py --minutes 60 --stride 10
    python scripts/export_run.py --naive --slots --out outputs/run_naive.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass

import yaml  # noqa: E402

from src.sim_engine import SimulationEngine, generate_room_demand  # noqa: E402

CONFIG_PATH = ROOT / "config" / "sim_config.yaml"
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_DEMAND_SCALE = 1.7  # busy room (see validate_room.py)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a simulation run as JSON.")
    ap.add_argument("--minutes", type=float, default=30.0, help="sim length [min]")
    ap.add_argument("--stride", type=int, default=5, help="record every N seconds")
    ap.add_argument("--slots", action="store_true", help="include per-slot temps")
    ap.add_argument("--naive", action="store_true", help="naive even split (vs liquid-first)")
    ap.add_argument("--scale", type=float, default=DEFAULT_DEMAND_SCALE, help="demand scale")
    ap.add_argument("--out", type=str, default=None, help="output JSON path")
    args = ap.parse_args()

    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    n_ticks = int(args.minutes * 60)
    engine = SimulationEngine(config, route=not args.naive)
    demand = generate_room_demand(config, n_ticks, scale=args.scale)

    frames = []
    for t in range(n_ticks):
        frame = engine.step(float(demand[t]), include_slots=args.slots)
        if t % args.stride == 0:
            frames.append(frame)

    run = {
        "metadata": engine.metadata(output_stride_s=args.stride),
        "topology": engine.topology(),
        "frames": frames,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (
        OUTPUT_DIR / f"run_{'naive' if args.naive else 'routed'}.json"
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(run, fh)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {len(frames)} frames ({engine.room.n_racks} racks, "
          f"{'per-slot' if args.slots else 'per-rack'}) over {args.minutes:.0f} min "
          f"@ {args.stride}s stride")
    print(f"  routing: {run['metadata']['routing']}")
    print(f"  -> {out_path.relative_to(ROOT)}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()

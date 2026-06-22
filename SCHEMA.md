# Visualization data contract

The backend exposes the server-room simulation as **plain JSON** — no plotting,
no frontend. Build the visualization against this contract; nothing here depends
on Python once you have a run file.

Two ways to consume it:

1. **Recorded run (simplest).** Generate a file and load it in the frontend:
   ```bash
   python scripts/export_run.py --minutes 30 --stride 5      # -> outputs/run_routed.json
   python scripts/export_run.py --naive                      # baseline for comparison
   python scripts/export_run.py --slots                      # add per-slot temps
   ```
   ```js
   const run = await fetch('run_routed.json').then(r => r.json());
   ```

2. **Live.** Import the engine and drive it from your own process/server:
   ```python
   from src.sim_engine import SimulationEngine, generate_room_demand
   import yaml
   cfg = yaml.safe_load(open('config/sim_config.yaml'))
   eng = SimulationEngine(cfg, route=True)
   demand = generate_room_demand(cfg, n_ticks=1800, scale=1.7)
   meta, topo = eng.metadata(), eng.topology()
   for d in demand:
       frame = eng.step(float(d))   # JSON-serializable dict, one per 1 s tick
   ```

The physics steps every **1 s**; recorded runs are **downsampled** by `stride`
(physics still runs every tick). Runs are **deterministic** given the seeds in
`metadata.seeds`.

## File structure

```jsonc
{
  "metadata": { ... },     // run-level constants (units, thresholds, seeds)
  "topology": { ... },     // static room layout (rack IDs, zones) — emit once
  "frames":   [ { ... } ]  // one object per recorded tick
}
```

### `metadata`

| field | type | meaning |
|---|---|---|
| `schema_version` | str | contract version (currently `"1.0"`) |
| `dt_s` | float | seconds per physics tick (1.0) |
| `output_stride_s` | int | seconds between recorded frames |
| `routing` | str | `"liquid-first"` or `"naive-even-split"` |
| `thresholds_c` | obj | `{ warn: 75, critical: 80, release: 72 }` — draw as reference lines |
| `fan_max` | float | max `omega` (2.5) |
| `zones` | obj | per zone kind: `{ kind, t_supply_c }` |
| `energy` | obj | `{ exponent, rated_w: { air, liquid } }` (cooling power = `rated_w[kind]·omega^exponent`) |
| `seeds` | obj | RNG seeds (reproducibility) |
| `units` | obj | unit labels (temperature=celsius, power=watts, …) |

### `topology`

| field | type | meaning |
|---|---|---|
| `n_slots_per_rack` | int | slots (servers) per rack (15) |
| `n_air_racks` / `n_liquid_racks` | int | counts per zone |
| `racks` | list | `{ id, kind, zone, index }` — `id` is stable (e.g. `"air-00"`, `"liquid-03"`) |

### `frames[i]`

| field | type | meaning |
|---|---|---|
| `t` | int | sim time [s] |
| `room` | obj | room aggregates (below) |
| `racks` | list | one record per rack (below), aligned to `topology.racks` by `id` |

**`frame.room`**

| field | type | meaning |
|---|---|---|
| `demand` | float | total offered load this tick [util units] |
| `it_w` | float | total compute (IT) power [W] |
| `cooling_w` | float | total fan + pump power [W] |
| `pue` | float | `(it_w + cooling_w) / it_w` |
| `tpi` | float | Task Preservation Index `executed/(executed+expired)` |
| `dropped` | float | demand that exceeded total cluster capacity |
| `throttles` | int | racks asserting THROTTLE_REQUEST this tick |
| `max_temp_air` / `max_temp_liquid` | float | hottest rack per zone [°C] |

**`frame.racks[j]`**

| field | type | meaning |
|---|---|---|
| `id` | str | stable rack id (matches `topology`) |
| `kind` | str | `"air"` or `"liquid"` |
| `max_temp` / `mean_temp` | float | rack hottest / mean slot temp [°C] |
| `omega` | float | fan/pump speed (1.0 = base, up to `fan_max`) |
| `throttle` | bool | THROTTLE_REQUEST asserted |
| `state` | str | `"safe"` \| `"warning"` \| `"critical"` (suggested colors: green / amber / red) |
| `util` | float | mean effective utilization [0–1] |
| `it_w` / `cooling_w` | float | this rack's compute / cooling power [W] |
| `slot_temps` | list[float] | per-slot temps [°C] — **only present with `--slots`** |

## Rendering hints

- Color racks by `state` (safe/warning/critical) or by `max_temp` vs
  `thresholds_c`. Slot 0 is the bottom of the rack; higher index = higher up.
- Air racks show a vertical gradient (bottom→top); liquid racks are near-flat.
- `omega` drives a fan-speed indicator; `throttle` a per-rack alarm badge.
- Room KPIs over time: `pue`, `cooling_w`, `tpi`, `max_temp_air`.
- Compare a `routed` vs `naive` run side by side to show the agents' benefit.

The contract is stable within a `schema_version`; additive fields will bump the
minor version.

# datacenter-sim

A discrete-time simulation of a data center where (later) AI agents will manage
cooling and workload routing, driven by the Alibaba 2018 cluster trace. The
build is deliberately incremental.

**Phase 1 — thermal foundation (complete).** A single rack with vertical
thermal coupling (recirculation), driven by synthetic load, plus a validation
script that proves it behaves sensibly.

**Phase 2 — trace-driven load (complete).** The same calibrated thermal model is
driven by the real Alibaba 2018 `machine_usage` trace instead of synthetic load.

**Phase 2b — per-node burstiness (complete).** Deterministic, independent
per-node load bursts are injected on top of the trace baseline, so individual
slots spike toward full load while the cluster average stays moderate — pushing
nodes into the 75–80 °C regime and making them diverge. The thermal model was
re-calibrated here (`K_cool` 9→7, `C` 540→420, τ held at 60 s) to give the
needed thermal headroom.

**Phase 3 — air/liquid cooling fabric (complete).** A second cooling zone —
**liquid** — is added alongside air, so the substrate models a heterogeneous
air/liquid fabric. This required one refactor: `alpha` (vertical recirculation
strength) was promoted from the global `thermal:` block to each zone, because it
is a cooling-fabric property, not a node property — liquid cools at the component
and has no rising exhaust column. The air zone is byte-for-byte unchanged.

**Phase 4 — Cooling Agent (this part).** The first control agent. A predictive,
zoned fan controller with a latched critical state regulates one air rack: it
reads temperature, forecasts ~15 s ahead, ramps `omega_fan` through the warning
band, and asserts a (logged-only) `THROTTLE_REQUEST` in the critical zone. The
substrate is untouched — the agent only writes the existing `omega_fan` hook.
There is still no Workload Agent, routing, task queue, or message bus, and the
throttle signal is **not** wired into load — those come in later phases.

## Layout

```
datacenter-sim/
  config/sim_config.yaml         # all tunable constants
  src/power.py                   # utilization -> power (watts), SPECpower curve
  src/thermal.py                 # RC thermal model + recirculation coupling
  src/rack.py                    # rack of N slots belonging to a cooling zone
  src/trace_loader.py            # Alibaba trace -> per-slot 1 s utilization
  src/burstiness.py              # independent per-node load bursts (Phase 2b)
  src/cooling_agent.py           # predictive zoned fan controller (Phase 4)
  scripts/validate_thermal.py    # Phase 1: synthetic-load scenarios
  scripts/validate_trace.py      # Phase 2: trace-driven thermal response
  scripts/validate_burstiness.py # Phase 2b: bursty thermal stress
  scripts/validate_zones.py      # Phase 3: air vs. liquid under same load
  scripts/validate_cooling.py    # Phase 4: cooling agent (nominal + degraded)
  tests/test_thermal.py          # power curve, gradient, convergence
  tests/test_trace_loader.py     # clipping, interpolation length, determinism
  tests/test_burstiness.py       # clipping, determinism, independence, 75 °C
  tests/test_zones.py            # air regression, liquid safety, flat gradient
  tests/test_cooling_agent.py    # zones, ramp, hysteresis latch, load boundary
  data/                          # (gitignored) holds the trace parquet
  outputs/                       # (gitignored) plots land here
```

## Setup & run

```bash
pip install -r requirements.txt
python -m pytest tests/ -q             # unit tests
python scripts/validate_thermal.py     # Phase 1 plots -> outputs/
python scripts/validate_trace.py       # Phase 2 plots -> outputs/
python scripts/validate_burstiness.py  # Phase 2b plots -> outputs/
python scripts/validate_zones.py       # Phase 3 plots -> outputs/
python scripts/validate_cooling.py     # Phase 4 plots -> outputs/
```

## The model

**Power** (`src/power.py`) — OpenDC / SPECpower-style curve so it is grounded in
published server power data and cross-checkable later:

```
P(u) = p_idle + (p_max - p_idle) * (2*u - u**r)
```

This single value is the node's heat input; idle is already included, so no
separate idle term is added anywhere.

**Thermal** (`src/thermal.py`) — each slot is a lumped RC node, integrated with
explicit forward Euler at `dt = 1 s`:

```
T_node[n, t+1] = T_node[n, t]
    + ( P[n,t] * K_heat
        - (T_node[n,t] - T_inlet[n,t]) * K_cool * omega_fan ) * dt / C
```

with the recirculation coupling — the important part — so higher slots inherit
the accumulated warmth of the slots below them:

```
T_inlet[n, t] = T_supply + alpha * sum_{m < n} max(0, T_node[m,t] - T_supply)
```

**Zones** (`src/rack.py`) — a rack has `N` slots and belongs to a `CoolingZone`
that owns its cooling parameters (`K_cool`, `T_supply`, `omega_fan`) and a
`kind`. Only one air-cooled zone is instantiated today, but nothing
zone-specific is hardcoded: a liquid-cooled zone can be added later purely by
appending an entry under `zones` in the config.

## Calibration rationale

> **Re-calibrated in Phase 2b.** The original Phase 1 calibration
> (`K_cool = 9`, `C = 540`) was physically sound but capped the whole rack at
> ~70 °C even at full load — too cool for cooling control to ever engage. Phase
> 2b lowered `K_cool` 9→7 and `C` 540→420 (holding `τ = C/K_cool = 60 s`) to
> give the model thermal headroom into the 75–85 °C regime. The reasoning below
> reflects the current values; `K_heat`, `alpha`, and `T_supply` are unchanged.

The targets were: at a steady **70% uniform load**, node temps settle in a
plausible hot band with supply/inlet around **22–27 °C**, a visible **monotonic
top-to-bottom gradient of a few degrees**, and a **thermal time constant of
~30–120 s**. The four constants were chosen as follows.

At 70% load the SPECpower curve gives `P(0.7) ≈ 342 W`. For the bottom slot the
inlet equals the supply temperature, so its steady-state rise above supply is
exactly `ΔT = P·K_heat / (K_cool·omega_fan)`.

- **`K_heat = 1.0`** — kept at unity so power enters the energy balance directly
  in watts; the cooling coefficient does the scaling. This keeps the two
  coefficients from being degenerate with each other.
- **`K_cool = 7.0` W/°C** — with `K_heat = 1` this puts the bottom-slot rise at
  `342 / 7 ≈ 49 °C`, so at 70% load the bottom slot lands at ~71 °C. The trace
  baseline load (~0.4) is cooler, landing the rack at ~60 °C, while sustained
  high load (u → 1) can reach ~84 °C — the headroom cooling control needs.
- **`T_supply = 22.0 °C`** — a conventional cold-aisle supply temperature, at the
  low end of the 22–27 °C window so the recirculating inlet has room to rise.
- **`alpha = 0.006`** — recirculation strength. Each lower slot contributes
  `alpha · (T_node − T_supply)` to the inlet of every slot above it. Summed over
  the stack this yields a **~4 °C** bottom-to-top gradient at 70% load,
  monotonic, with the top inlet still inside the 22–27 °C window. Small enough
  that the multiplicative compounding up the rack stays gentle.
- **`C = 420.0` J/°C** — sets the time constant `τ = C / (K_cool·omega_fan)
  = 420 / 7 = 60 s`, squarely in the requested 30–120 s range.

**Stability.** The explicit-Euler decay factor per step is
`1 − K_cool·omega_fan·dt/C = 1 − 7/420 ≈ 0.983`, comfortably inside `(−1, 1)`,
so the integration is stable and non-oscillatory by construction.

### Validation results

`scripts/validate_thermal.py` confirms the (re-)calibration:

| Quantity | Result |
|---|---|
| Bottom slot @ 70% steady | 70.9 °C |
| Top slot @ 70% steady | 75.1 °C |
| Top-to-bottom gradient | 4.3 °C, monotonic ↑ |
| Supply / inlet | 22 °C / up to ~26 °C |
| 30→90% step steady state | 76.6 °C |
| Analytic τ | 60 s |
| Measured τ (63.2% rise, 30→90% step) | 59 s |
| Oscillation / overshoot | none |

Plots are written to `outputs/`:
- `constant_load_70pct.png` — temperature vs. time per slot, and the
  steady-state slot-vs-temperature gradient.
- `load_step_30_90.png` — the 30% → 90% step response with the measured time
  constant.

## Phase 2 — trace-driven load

`src/trace_loader.py` turns the cluster-averaged Alibaba 2018 `machine_usage`
trace (`data/alibaba_machine_usage_300s.parquet`, 2243 samples at a 300 s
stride) into a per-slot, per-timestep utilization array that drives the existing
thermal model. The pipeline:

1. **Load & convert.** `cpu_util_percent` is the primary load driver; it is
   converted from a [0, 100] scale to a fraction `u ∈ [0, 1]` exactly once (and
   clipped defensively, so a noisy >100 sample cannot leak a >1 fraction). The
   other columns (`mem_util_percent`, `net_in`, `net_out`, `disk_io_percent`)
   are kept available but unused for now.
2. **Resample to 1 s.** The trace is at a 300 s stride but the sim runs at
   `dt = 1 s`, so the signal is interpolated up to 1 s resolution. The method is
   a clearly named function (`interpolate_linear`) selected via a `INTERPOLATORS`
   registry keyed by the config name, so it is easy to swap later. Two source
   points `stride` seconds apart yield `stride + 1` samples (shared endpoint), so
   a trace of `P` points becomes `(P - 1) · stride + 1` one-second samples.
3. **Per-slot variation.** The raw signal is cluster-averaged, so every slot
   would otherwise be identical. We apply a fixed, deterministically-seeded
   per-slot transform:

   ```
   u_slot[n, t] = clip( u_trace[t] * multiplier[n] + offset[n], 0, 1 )
   ```

   where `multiplier[n] ~ 1 ± scale_spread` and `offset[n] ~ ± offset_spread`
   are drawn **once** from a seeded NumPy RNG (`seed` from config). Because the
   draws are seeded and happen at construction, every run produces identical
   per-slot values — no seed drift between runs. The result is precomputed into a
   `[n_slots, n_timesteps]` array for speed; `get_utilization(slot, t)` and
   `utilization_at(t)` are convenience accessors over it.

All trace parameters live under the `trace:` block in `config/sim_config.yaml`
(file path, source stride, CPU column, interpolation name, and the variation
`seed` / `scale_spread` / `offset_spread`) — nothing is hardcoded in Python.

**Why this variation model.** A multiplier captures that some slots simply run
hotter workloads (a scale on the shared signal), while a small additive offset
breaks ties at low load. The defaults (`scale_spread = 0.08`, `offset_spread =
0.01`) are deliberately small: at the trace's mean utilization a ±8% spread in
`u` maps to only ~±1.7 °C of per-slot variation, less than the ~3.4 °C
recirculation gradient, so the bottom-to-top thermal gradient is preserved while
the slots are still visibly distinct.

### Validation results (`scripts/validate_trace.py`, first 3 simulated hours)

Numbers below reflect the Phase 2b re-calibration (`K_cool = 7`, `C = 420`).

| Check | Result |
|---|---|
| Utilization range after variation + clip | 0.141 – 0.855 (≤ 1.0 ✓) |
| Interpolated length vs. `(P−1)·stride+1` | 672601 = 672601 ✓ |
| Vertical gradient preserved | bottom 53.5 °C, top 57.0 °C, slope +0.21 °C/slot |
| Lag (recovered τ vs. analytic 60 s) | 59 s (EMA fit RMSE 0.003 °C) |
| Peak temperature under real load | **63.22 °C** (top slot) |

The lag is confirmed structurally: the update relaxes each node toward its
instantaneous equilibrium temperature `T_eq(u) = T_supply + P(u)·K_heat/K_cool`
with the thermal time constant, so the smoothing constant that best reconstructs
the simulated temperature from `T_eq` recovers τ ≈ 60 s. (A direct
cross-correlation can't resolve a 60 s lag here because the trace varies on a
300 s+ timescale — the lag is real but small relative to the signal's own
dynamics.)

Plots written to `outputs/`:
- `trace_input_signal.png` — raw 300 s trace points vs. the 1 s interpolated
  curve (confirms interpolation is sane).
- `trace_rack_temperature.png` — per-slot node temperature under real load, plus
  the time-averaged vertical gradient.
- `trace_thermal_lag.png` — instantaneous equilibrium vs. actual (lagged)
  temperature on a zoomed window, visualising the ~60 s lag.

## Phase 2b — per-node burstiness

The cluster-averaged trace is too smooth: even after Phase 2's per-slot
variation it peaks at only ~63 °C, so the rack never reaches the 75–80 °C regime
where cooling control must act, and the nodes barely differ. `src/burstiness.py`
fixes this by injecting deterministic, **independent** per-node load bursts on
top of the Phase 2 baseline.

- **Burst model.** A burst is a temporary *additive* elevation of one slot's
  utilization (`baseline + burst`, then clipped to [0, 1]). Bursts are laid down
  sequentially per slot and never overlap: each occupies `[start, start+duration)`
  and the next start follows after an exponentially-distributed idle gap (mean
  `3600 / arrival_per_hour` s).
- **Independence & determinism.** Each slot draws from its own RNG seeded with
  `seed + slot_index`. Independence is the whole point — it makes nodes diverge
  from one another — and the seeding makes runs byte-for-byte reproducible with
  zero drift. `enabled: false` returns the exact Phase 2 baseline.
- **Why these defaults (`arrival_per_hour: 2`, `magnitude: [0.5, 0.7]`,
  `duration_s: [180, 480]`).** The thermal model relaxes with τ ≈ 60 s, so a
  burst shorter than ~τ barely moves temperature — the node sheds heat as fast as
  it arrives. Durations are therefore **3–8 × τ** so heat genuinely accumulates,
  and the additive magnitude lifts the ~0.4 baseline into the **0.85–1.0** range
  where a sustained burst drives a node to ~80 °C. (This is verified against the
  *temperature*, not just the utilization — see the test below.)

All burst parameters live under the `burstiness:` block in
`config/sim_config.yaml`; nothing is hardcoded in Python. `BurstyTraceLoader`
composes a `TraceLoader` and applies the layer, exposing the same accessor
interface plus `baseline_utilization` for off-vs-on comparison.

### Validation results (`scripts/validate_burstiness.py`, 4 simulated hours)

| Quantity | Result |
|---|---|
| Bursty utilization range (after clip) | 0.141 – 1.000 (≤ 1.0 ✓) |
| Baseline (bursts off) peak temperature | 63.2 °C |
| **Bursty (bursts on) peak temperature** | **81.7 °C** |
| Slots crossing 75 °C (warning) | **15 / 15** |
| Slots crossing 80 °C (critical) | 8 / 15 |
| Divergence (hottest − coolest slot peak) | 5.5 °C |
| Max instantaneous slot-to-slot spread | 28.2 °C |

Every slot crosses the 75 °C warning line and 8 of 15 cross 80 °C, while at any
single moment the rack spans up to ~28 °C between its hottest and coolest node —
a clear, heterogeneous control problem for later routing/cooling agents to solve.

Plots written to `outputs/`:
- `burstiness_utilization.png` — per-slot utilization heatmap; bursts (bright
  cells) appear at independent times per slot, with the faint shared vertical
  band being the underlying trace.
- `burstiness_temperature.png` — per-slot temperature with the 75/80 °C
  thresholds drawn; nodes spike past both lines and visibly diverge.
- `burstiness_baseline_vs_bursty.png` — per-slot peak temperature, bursts off
  vs. on, against the threshold lines.

The `test_at_least_one_slot_crosses_temperature_threshold` test in
`tests/test_burstiness.py` asserts a slot exceeds 75 °C under the default config —
this is the test that proves the control problem now exists.

## Phase 3 — air/liquid cooling fabric

The substrate now models two cooling zones. A `Rack` belongs to a `CoolingZone`;
zones are keyed by name in the `zones:` config block, and **adding a zone is
purely config** — there is no branching on `zone.kind` anywhere in the Python
(the `kind` field is a descriptive discriminator only).

**The one refactor — `alpha` is now per-zone.** Recirculation strength used to
live in the global `thermal:` block and applied to every rack. That is wrong for
liquid: direct-to-chip / cold-plate cooling removes heat at the component, so
there is no hot exhaust-air column rising through the rack and no vertical
gradient. `alpha` was therefore moved onto `CoolingZone` and is passed into
`thermal_step` by the caller. `ThermalParams` now holds only the true node
properties (`K_heat`, `C`). The air zone keeps `alpha = 0.006`, so its behaviour
is **byte-for-byte unchanged** (verified — see the regression below).

### Liquid zone calibration

| Param | Air | Liquid | Rationale |
|---|---|---|---|
| `K_cool` | 7.0 | **10.0** | ~1.4× air — better heat removal, but *finite* capacity |
| `T_supply` | 22.0 | **18.0** | chilled liquid loop runs cooler than cold-aisle air |
| `alpha` | 0.006 | **0.0002** | near-zero — no exhaust column, so no vertical gradient |
| `omega_fan` | 1.0 | 1.0 | pump multiplier; identical actuation interface as the fan |

`K_cool` is sized for **finite headroom**, not maximal cooling. The liquid node
rise is `P/K_cool ≈ 400/10 ≈ 40 °C` at full load, so with the 18 °C supply
liquid peaks at ~58 °C under the same bursty load that drives air to 82 °C —
safely below the 75 °C line but with real spare capacity, so routing into liquid
is advantageous yet not unlimited. (An earlier 6× value sat liquid at ~28 °C,
which made "put everything in liquid" trivially optimal and removed the
finite-capacity constraint that makes routing interesting.) The time constant
`τ = C/K_cool = 420/10 = 42 s` is a physical consequence of the coupling, not a
separate knob. `alpha = 0.0002` makes the recirculation contribution ~30×
smaller than air's *and* it acts on a smaller temperature rise, so the vertical
gradient collapses to ~0.1 °C.

### Validation results (`scripts/validate_zones.py`, identical bursty load)

| Quantity | Air | Liquid |
|---|---|---|
| Peak temperature (4 h bursty) | **81.70 °C** | **58.02 °C** |
| Slots crossing 75 °C / 80 °C | 15 / 15, 8 / 15 | 0 / 15 |
| Steady-state vertical gradient (uniform 70%) | **+4.27 °C** (monotonic) | **+0.10 °C** (flat) |

**Air regression:** under identical bursty load the air peak is `81.70 °C` and
the uniform-70% steady state is `70.87 / 75.13 °C` (bottom/top) — matching the
Phase 2b golden values, confirming the `alpha` refactor did not change the air
zone. `tests/test_zones.py` enforces this, plus liquid safety, the near-flat
liquid gradient, and that a zone with air's numbers but a `liquid` label behaves
identically to air (proving no `kind`-based branching).

Plots written to `outputs/`:
- `zones_air_vs_liquid_temperature.png` — overlaid per-slot temperature, air
  (red) vs. liquid (blue) under the same load, with the 75/80 °C thresholds.
- `zones_gradient_profiles.png` — steady-state slot-vs-temperature for both
  zones: air monotonic, liquid near-flat.

## Phase 4 — Cooling Agent

`src/cooling_agent.py` regulates one air rack. It is **predictive**: each step it
rolls the existing thermal model forward `horizon_s` (15 s) under the current
load to forecast the hottest-slot temperature, and makes all decisions on that
forecast. It holds no rack reference and no "air"/single-rack assumptions — the
rack is passed in per call and the latch state lives in the instance, so a second
instance regulates another rack/zone unmodified.

**Zoned state machine (on predicted temperature):**

| Zone | Predicted temp | Action |
|---|---|---|
| Safe | < 75 °C | `omega_fan = 1.0` |
| Warning | 75–80 °C | `omega_fan` ramps linearly 1.0 → 2.5 across the band |
| Critical | > 80 °C | assert `THROTTLE_REQUEST`, hold `omega_fan = 2.5` |

**Hysteresis latch.** Once critical, the agent stays latched (throttle asserted,
fans maxed) until the *predicted* temperature falls below `release_c = 72 °C` —
not merely back below 80. This prevents fan/throttle chatter at the 80 °C edge
and is tested explicitly.

**The hard boundary.** `THROTTLE_REQUEST` is decided, exposed, and logged, but
has **no effect on load** this phase — acting on it is the Workload Agent's job
later. The agent's only real actuation is `omega_fan` (written via the existing
zone hook with `dataclasses.replace`, so no substrate internals change); the
throttle flag feeds nothing. The two outputs are kept cleanly separate, and a
regression test confirms the load is byte-identical whether or not the throttle
fires.

All thresholds, the horizon, and the fan ceiling come from the `cooling_agent:`
config block; nothing is hardcoded.

### Validation (`scripts/validate_cooling.py`) — two scenarios

**Nominal** (real bursty load, spec config): the predictive fan ramp *fully
contains* the load, so the throttle never fires — a correct finding, not a
failure. Fans alone suffice here.

| | Peak temperature | Throttle events |
|---|---|---|
| No agent (fans = 1.0) | 81.70 °C | — |
| **Cooling agent** | **74.95 °C** | **0** |

The fan ramp removes ~6.8 °C of peak and pins the rack at the 75 °C warning line.

**Degraded** (controlled CRAC-fault stress test — a validation artifact only; it
does not change default behaviour): a cooling fault reduces the air zone's
`K_cool` for a window on a smooth load, driving the agent into critical so the
latch is exercised and legible. One clean cycle:

- **assert** at predicted 80 °C (actual 77 °C — the forecast leads);
- **hold**: a partial recovery parks the rack at **~75 °C — below the 80 °C
  line — for ~24 min with the throttle still latched**, proving it does *not*
  release at 80;
- **release** only when full repair drops the predicted temperature below 72 °C;
  no rebound or chatter afterward.

Peak in the fault reaches 96 °C; exactly **1** throttle event fires; the load is
unchanged by the throttle throughout (boundary regression).

Plots written to `outputs/`:
- `cooling_nominal_fan_effect.png` — hottest-slot temp with vs. without the
  agent (+ `omega_fan`), showing the fan ramp removes heat.
- `cooling_degraded_throttle.png` — temperature (actual + predicted) and
  `omega_fan` with the 75/80/72 °C lines, and the `THROTTLE_REQUEST` timeline
  beneath, showing the single latched assert → hold → release@72 cycle.

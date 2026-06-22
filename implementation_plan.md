# Visualization WebApp for Datacenter Cooling Simulation

A real-time, three-zone dashboard backed by FastAPI (WebSocket) and a vanilla HTML/CSS/JS frontend. The simulation engine reuses the existing `src/` modules and adds a **Workload Agent** with live network-sniffing and a regression-based **Thermal Predictor**.

## User Review Required

> [!IMPORTANT]
> **Network Sniffing**: The Workload Agent will use `psutil.net_io_counters()` to read real-time bytes sent/received from your machine's network interface. This is a read-only OS call. On some systems it may require admin privileges (we'll handle the fallback gracefully).

> [!IMPORTANT]
> **ML Model**: The "Brain" is a lightweight `scikit-learn` Ridge Regression model trained on-the-fly from the simulation's own thermal history. No external model files or GPUs are needed. Is this approach acceptable, or do you have a pre-trained model you'd like to integrate?

> [!IMPORTANT]
> **Scope Clarification**: The reference images show a "Leaf-Spine Routing" network topology and a "Waiting Room" task queue. I plan to implement these as *visual representations* of the existing rack/agent logic (tasks are the trace utilization slices, routing is the agent's load-shedding decision), not a full network simulator. Please confirm this is acceptable.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Browser (Frontend)                  │
│  Zone 1: Control Panel  │  Zone 2: Rack Heatmap     │
│  Zone 3: Agent Logs     │  + Leaf-Spine Topology     │
│         ← WebSocket (JSON frames @ ~1 Hz) →         │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│              FastAPI Backend (Python)                 │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ Workload │  │ Thermal  │  │ Cooling Agent     │  │
│  │ Agent    │←→│ Agent    │←→│ (existing module) │  │
│  └──────────┘  └──────────┘  └───────────────────┘  │
│        ↑             ↑               ↑               │
│  psutil.net_io   src/thermal.py  src/cooling_agent   │
│                  src/rack.py     + ML Predictor      │
│                  src/power.py                        │
└─────────────────────────────────────────────────────┘
```

---

## Proposed Changes

### Backend — FastAPI Server

#### [NEW] [server.py](file:///c:/college%20stuff/SEM%204/EL/ServerManager/server.py)

The main FastAPI application. Responsibilities:
- **`/`** — Serves the static `web/` directory (HTML/CSS/JS).
- **`/ws`** — WebSocket endpoint. On connect, starts the simulation loop on a background task and streams JSON frames at ~1 Hz to the client.
- **`/api/config`** — GET endpoint returning [sim_config.yaml](file:///c:/college%20stuff/SEM%204/EL/ServerManager/config/sim_config.yaml) as JSON (for the control panel).
- **`/api/start`** — POST to start/restart the simulation with optional parameter overrides (speed, mode).
- **`/api/stop`** — POST to pause the simulation.

Each WebSocket frame contains the full state snapshot:

```json
{
  "tick": 1234,
  "phase": "B",
  "slots": [
    {"id": 0, "temp": 65.2, "inlet": 23.1, "power": 280, "util": 0.55, "state": "safe"}
  ],
  "agents": {
    "workload": {"load_pct": 72, "status": "HIGH_LOAD", "bytes_in": 1234567, "bytes_out": 987654},
    "thermal":  {"predicted_temp": 78.3, "current_max": 75.1, "state": "warning"},
    "cooling":  {"omega_fan": 1.8, "throttle_active": false, "state": "warning"}
  },
  "kpis": {
    "total_energy_kwh": 1.23,
    "thermal_violations_s": 45,
    "task_preservation_pct": 97.2
  },
  "log": ["[WORKLOAD] UDP spike detected: 85 MB/s", "[COOLING] Fan ramp: ω=1.8"]
}
```

---

#### [NEW] [src/workload_agent.py](file:///c:/college%20stuff/SEM%204/EL/ServerManager/src/workload_agent.py)

The "Pulse" — reads real-time network throughput from the OS and converts it to a utilization index.

- Uses `psutil.net_io_counters()` to sample `bytes_sent` and `bytes_recv` every second.
- Normalizes to a 0–1 utilization index: `u = min(1.0, throughput_mbps / max_throughput_mbps)`.
- `max_throughput_mbps` defaults to 100 MB/s (configurable) — represents the machine's "capacity".
- Detects "burst" conditions (e.g., UDP-heavy traffic from streaming) and flags `HIGH_LOAD`.
- Implements the **Greedy Packet Shedding** response: when the Cooling Agent sends a `THROTTLE_REQUEST`, the Workload Agent reduces its reported utilization by dropping low-priority fraction first (configurable `shed_fraction`, default 0.3).
- Exposes: `current_load`, `status` ("NORMAL" / "HIGH_LOAD" / "THROTTLED"), `bytes_in`, `bytes_out`.

---

#### [NEW] [src/thermal_agent.py](file:///c:/college%20stuff/SEM%204/EL/ServerManager/src/thermal_agent.py)

The "Calculator" — wraps the existing [thermal.py](file:///c:/college%20stuff/SEM%204/EL/ServerManager/src/thermal.py) + [rack.py](file:///c:/college%20stuff/SEM%204/EL/ServerManager/src/rack.py) and adds ML prediction.

- Orchestrates the per-tick thermal step using the existing `Rack.step()`.
- Maintains a rolling history buffer (last 300 ticks) of [(utilization, fan_speed, temperature)](file:///c:/college%20stuff/SEM%204/EL/ServerManager/src/rack.py#103-107).
- **ML Predictor**: After accumulating ≥60 samples, trains a `sklearn.linear_model.Ridge` regressor on the history to predict temperature 15s ahead. The features are `[util, omega_fan, current_temp]` per slot; the target is `temp_at_t+15`. This is lightweight and fits in <1ms.
- The prediction is compared with the existing `CoolingAgent.predict_temperature()` deterministic forecast; the ML value is used for display, but the deterministic one still drives control (safety-critical).

---

#### [MODIFY] [cooling_agent.py](file:///c:/college%20stuff/SEM%204/EL/ServerManager/src/cooling_agent.py)

Minimal change: add a `to_dict()` method to [CoolingDecision](file:///c:/college%20stuff/SEM%204/EL/ServerManager/src/cooling_agent.py#66-74) for JSON serialization. No logic changes.

---

#### [NEW] [src/simulation_runner.py](file:///c:/college%20stuff/SEM%204/EL/ServerManager/src/simulation_runner.py)

The orchestrator that ties the three agents together in the OODA loop:

1. **Observe**: Workload Agent reads network counters → `current_load`.
2. **Orient**: Thermal Agent normalizes load into per-slot utilization (blends real network load with trace baseline via a configurable `real_load_weight`).
3. **Decide**: Cooling Agent runs [control()](file:///c:/college%20stuff/SEM%204/EL/ServerManager/src/cooling_agent.py#139-152) → fan speed + throttle decision. ML predictor forecasts ahead.
4. **Act**: If `throttle_request`, signal Workload Agent to shed load. Rack advances one thermal step.

Supports two modes:
- **Live Mode** (default): Workload Agent sniffs real network traffic. Runs at wall-clock 1 Hz.
- **Trace Mode**: Uses the Alibaba trace + burstiness layer. Runs at configurable speed (1x–100x).

Tracks KPIs:
- `total_energy_kwh`: cumulative `sum(power) * dt / 3.6e6`.
- `thermal_violations_s`: ticks where any slot > 85°C.
- `task_preservation_pct`: [(1 - shed_fraction_applied) * 100](file:///c:/college%20stuff/SEM%204/EL/ServerManager/src/rack.py#103-107).

Emits structured log messages for the agent negotiation terminal.

---

### Frontend — Vanilla HTML/CSS/JS

#### [NEW] [web/index.html](file:///c:/college%20stuff/SEM%204/EL/ServerManager/web/index.html)

Single-page dashboard with three zones, dark theme, Inter font.

---

#### [NEW] [web/style.css](file:///c:/college%20stuff/SEM%204/EL/ServerManager/web/style.css)

Design system:
- **Dark theme**: `#0a0e17` background, glassmorphic panels with `backdrop-filter: blur`.
- **Color ramp for temps**: CSS custom properties mapping safe (green `#22c55e`), warning (orange `#f59e0b`), critical (red `#ef4444`, blinking animation for >85°C).
- **Typography**: Inter from Google Fonts.
- Responsive grid layout with three zones.

---

#### [NEW] [web/app.js](file:///c:/college%20stuff/SEM%204/EL/ServerManager/web/app.js)

Client-side logic:

**Zone 1 — Control Panel (left sidebar)**:
- Mode toggle: "Baseline (Reactive)" ↔ "Agentic (Proactive)" (maps to `trace_mode` vs. `live_mode`).
- Simulation speed slider (1x–100x, trace mode only).
- KPI scoreboard: Total Energy (kWh), Thermal Violations (s), Task Preservation (%).

**Zone 2 — Digital Twin 42U Rack (center)**:
- 15 rectangular slot elements stacked vertically.
- Each slot has a **thermodynamic color gradient** (green → orange → blinking red) driven by `slot.temp`.
- Internal **load bar** (horizontal progress bar, 0–100%) inside each slot.
- Smooth CSS transitions on color and bar width for micro-animation.

**Zone 3 — Leaf-Spine Routing & Agent Logs (bottom)**:
- **Waiting Room**: A container showing queued "tasks" (represented as small cards) that get visually routed to slots. Priority 0 tasks get a bypass arrow animation.
- **Agent Negotiation Log**: A scrolling terminal-style `<pre>` element showing timestamped log lines like `[COOLING] → [WORKLOAD]: THROTTLE_REQUEST — shed 30% low-priority`.
- CSS animation for routing arrows.

---

### Configuration

#### [MODIFY] [sim_config.yaml](file:///c:/college%20stuff/SEM%204/EL/ServerManager/config/sim_config.yaml)

Add new sections:

```yaml
workload_agent:
  max_throughput_mbps: 100.0
  shed_fraction: 0.3
  burst_threshold_mbps: 50.0

webapp:
  host: "0.0.0.0"
  port: 8000
  tick_rate_hz: 1.0
  real_load_weight: 0.5    # blend: 0 = pure trace, 1 = pure live network
```

---

### Dependencies

#### [MODIFY] [requirements.txt](file:///c:/college%20stuff/SEM%204/EL/ServerManager/requirements.txt)

Add:
```
fastapi>=0.111
uvicorn[standard]>=0.30
websockets>=12.0
psutil>=5.9
scikit-learn>=1.5
```

---

## Verification Plan

### Automated Tests

1. **Existing test suite** (must stay green):
   ```bash
   python -m pytest tests/ -q
   ```
   All 64 existing tests must still pass — we are only *adding* code, not modifying simulation logic.

2. **New unit tests** — `tests/test_workload_agent.py`:
   - Verify utilization normalization (0 bytes → 0.0, max_throughput → 1.0, over-max → clipped to 1.0).
   - Verify `THROTTLE_REQUEST` response: when `shed_load(0.3)` is called, reported utilization drops by 30%.
   - Verify status transitions: NORMAL → HIGH_LOAD when throughput > burst_threshold.

3. **New unit tests** — `tests/test_simulation_runner.py`:
   - Run 100 ticks in trace mode and verify KPI accumulation (energy monotonically increases, violations count ≥ 0).
   - Verify agent negotiation: inject a scenario where temp > 80°C and confirm the Workload Agent receives the throttle signal and reduces load.

4. **WebSocket integration test** — `tests/test_server.py`:
   ```bash
   python -m pytest tests/test_server.py -q
   ```
   - Connect to `/ws`, receive ≥ 3 frames, validate JSON schema (all expected keys present, temps are floats, tick is incrementing).

### Manual Verification (Browser)

1. Start the server:
   ```bash
   python server.py
   ```
2. Open `http://localhost:8000` in a browser.
3. **Phase A (Baseline)**: Observe the rack in a stable green state with low load. KPIs should show 0 violations.
4. **Phase B (Stress Test)**: Start a heavy download or video stream. The rack slots should turn orange/red. The "Brain" prediction should show an upcoming spike.
5. **Phase C (Intervention)**: Observe the Cooling Agent ramp fans. If temp exceeds 80°C, the negotiation log should show `THROTTLE_REQUEST` and the Workload Agent responding. Slots should gradually return to green/orange.
6. Toggle between "Baseline" and "Agentic" modes and confirm the visual difference.

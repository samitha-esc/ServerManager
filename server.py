"""THERMOS — FastAPI server entry point.

Run with:
    python server.py

Then open http://localhost:8000 in your browser.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
CONFIG_PATH = ROOT / "config" / "sim_config.yaml"

app = FastAPI(title="THERMOS — Datacenter Cooling Orchestrator")

# Serve static files from web/
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ── Config ────────────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── Simulation singleton (one per server process) ─────────────────────────────

from src.agents.simulation_runner import SimulationRunner

_runner: SimulationRunner | None = None


def get_runner() -> SimulationRunner:
    global _runner
    if _runner is None:
        _runner = SimulationRunner()
    return _runner


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = WEB_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/config")
async def get_config():
    return _load_cfg()


@app.post("/api/mode/{mode}")
async def set_mode(mode: str):
    if mode not in ("baseline", "multi_agent"):
        return {"error": "mode must be 'baseline' or 'multi_agent'"}
    get_runner().set_mode(mode)
    return {"mode": mode, "ok": True}


@app.post("/api/reset")
async def reset_simulation():
    global _runner
    _runner = SimulationRunner()
    return {"ok": True}


@app.get("/api/snapshot")
async def snapshot():
    runner = get_runner()
    if runner._last_frame:
        return runner._last_frame
    return runner.tick()


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    runner = get_runner()
    cfg = _load_cfg().get("webapp", {})
    tick_hz = float(cfg.get("tick_hz", 2.0))
    interval = 1.0 / tick_hz

    try:
        while True:
            frame = runner.tick()
            await ws.send_text(json.dumps(frame))
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = _load_cfg().get("webapp", {})
    host = str(cfg.get("host", "0.0.0.0"))
    port = int(cfg.get("port", 8000))
    print(f"\n  THERMOS server starting → http://localhost:{port}\n")
    uvicorn.run("server:app", host=host, port=port, reload=False)

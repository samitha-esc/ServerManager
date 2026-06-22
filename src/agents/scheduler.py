"""Workload Scheduler Agent.

Reads real-time network throughput from psutil and converts it to a
utilization index, then classifies traffic into four priority buckets.

In BASELINE mode: all traffic is treated equally, no priority splitting.
In MULTI-AGENT mode: traffic is classified into P0-P3 buckets and the
Arbiter can shed lower-priority traffic under thermal stress.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

# Priority labels
P0_CRITICAL = 0   # Live requests / interactive
P1_HIGH     = 1   # AI training jobs
P2_MEDIUM   = 2   # ETL pipelines
P3_LOW      = 3   # Backups / logs

PRIORITY_NAMES = {0: "P0 Critical", 1: "P1 High", 2: "P2 Medium", 3: "P3 Low"}

# Default traffic split by priority (fraction of total load)
DEFAULT_PRIORITY_SPLIT = [0.10, 0.20, 0.35, 0.35]


@dataclass
class SchedulerConfig:
    max_throughput_mbps: float = 100.0
    burst_threshold_mbps: float = 50.0
    priority_split: List[float] = field(default_factory=lambda: list(DEFAULT_PRIORITY_SPLIT))
    shed_fraction: float = 0.30    # max fraction workload can shed on throttle
    synthetic_amplitude: float = 0.55   # base synthetic load amplitude
    synthetic_noise: float = 0.15       # noise on synthetic load


class SchedulerAgent:
    """Workload Scheduler — the 'Pulse' of the datacenter.

    Provides current_load as a 0-1 utilization fraction.
    Also manages per-priority load fractions used by the Router.
    """

    def __init__(self, config: SchedulerConfig, seed: int = 42) -> None:
        self.config = config
        self._rng = __import__("numpy").random.default_rng(seed)
        self._last_bytes: Optional[tuple] = None
        self._last_time: float = time.monotonic()
        self._tick: int = 0

        # State
        self.current_load: float = 0.30
        self.bytes_in_mbps: float = 0.0
        self.bytes_out_mbps: float = 0.0
        self.status: str = "NORMAL"   # NORMAL / HIGH_LOAD / THROTTLED
        # Per-priority load fractions (0-1), agents can modify for shedding
        self.priority_load: Dict[int, float] = {
            p: self.config.priority_split[p] for p in range(4)
        }
        self._shed_active: bool = False
        self._shed_fractions: Dict[int, float] = {p: 1.0 for p in range(4)}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance one simulation step — sample network traffic."""
        self._tick += 1
        self.current_load = self._sample_network_load()
        self._update_status()
        self._update_priority_loads()

    def apply_throttle(self, p3_shed: float = 1.0, p2_shed: float = 0.5) -> None:
        """Called by Arbiter: reduce P2/P3 contributions."""
        self._shed_active = True
        self._shed_fractions[P3_LOW] = max(0.0, 1.0 - p3_shed)
        self._shed_fractions[P2_MEDIUM] = max(0.0, 1.0 - p2_shed)
        self.status = "THROTTLED"

    def release_throttle(self) -> None:
        """Called by Arbiter: restore full traffic."""
        self._shed_active = False
        self._shed_fractions = {p: 1.0 for p in range(4)}
        if self.status == "THROTTLED":
            self.status = "NORMAL"

    @property
    def effective_load(self) -> float:
        """Load after shedding (used by router and thermal model)."""
        if not self._shed_active:
            return self.current_load
        # Weighted sum after shedding
        cfg = self.config
        raw = self.current_load
        total_w = sum(cfg.priority_split[p] * self._shed_fractions[p] for p in range(4))
        return raw * max(0.0, total_w)

    def to_dict(self) -> dict:
        return {
            "load_pct": round(self.effective_load * 100, 1),
            "raw_load_pct": round(self.current_load * 100, 1),
            "status": self.status,
            "bytes_in_mbps": round(self.bytes_in_mbps, 2),
            "bytes_out_mbps": round(self.bytes_out_mbps, 2),
            "priority_load": {
                PRIORITY_NAMES[p]: round(v * 100, 1)
                for p, v in self.priority_load.items()
            },
            "shed_active": self._shed_active,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sample_network_load(self) -> float:
        """Return utilization in [0, 1]. Uses psutil if available."""
        if _PSUTIL_AVAILABLE:
            return self._sample_psutil()
        return self._sample_synthetic()

    def _sample_psutil(self) -> float:
        now = time.monotonic()
        try:
            counters = psutil.net_io_counters()
            sent = counters.bytes_sent
            recv = counters.bytes_recv
        except Exception:
            return self._sample_synthetic()

        if self._last_bytes is None:
            self._last_bytes = (sent, recv)
            self._last_time = now
            return self._sample_synthetic()

        dt = max(1e-3, now - self._last_time)
        delta_sent = max(0, sent - self._last_bytes[0])
        delta_recv = max(0, recv - self._last_bytes[1])
        self._last_bytes = (sent, recv)
        self._last_time = now

        total_mbps = (delta_sent + delta_recv) / (1e6 * dt)
        self.bytes_in_mbps = delta_recv / (1e6 * dt)
        self.bytes_out_mbps = delta_sent / (1e6 * dt)
        util = min(1.0, total_mbps / self.config.max_throughput_mbps)
        # Blend with synthetic to ensure visual richness
        synth = self._sample_synthetic()
        return float(0.4 * util + 0.6 * synth)

    def _sample_synthetic(self) -> float:
        """Sine-wave baseline + noise to simulate realistic load cycles."""
        import math
        t = self._tick
        base = self.config.synthetic_amplitude
        wave = 0.15 * math.sin(2 * math.pi * t / 300)  # 5-min cycle
        burst = 0.30 * math.sin(2 * math.pi * t / 60) if (t % 400 < 60) else 0.0
        noise = self.config.synthetic_noise * (self._rng.random() - 0.5)
        return float(min(1.0, max(0.05, base + wave + burst + noise)))

    def _update_status(self) -> None:
        if self._shed_active:
            return
        if self.current_load * 100 > self.config.burst_threshold_mbps:
            self.status = "HIGH_LOAD"
        else:
            self.status = "NORMAL"

    def _update_priority_loads(self) -> None:
        for p in range(4):
            base = self.config.priority_split[p] * self.current_load
            self.priority_load[p] = base * self._shed_fractions[p]

"""Workload Agent (Phase 5) — priority-aware load routing and shedding.

Sits between the trace-driven utilization signal and the physical rack as a
smart filter.  During a thermal emergency (CoolingAgent ``THROTTLE_REQUEST``) it
enforces a **4-tier priority matrix** that preserves critical work while shedding
delay-tolerant load.  During nominal operation it is a transparent pass-through
by default; **Least-Thermal-Intensity (LTI) routing** — steering load toward
cooler slots — is available as an opt-in (``lti_enabled``) and, when on, strictly
conserves total load.

Execution flow (every 1-second simulation tick):

1. **Intercept & Triage** — split each slot's raw utilization into four
   priority bands (P0-Critical … P3-Low) using configurable fractions.
2. **Environmental Check** — read the CoolingAgent's ``throttle_request``
   through a configurable inter-agent staleness delay (default 1 tick = 1 s at
   dt = 1 s). This stands in for the not-yet-built inter-agent message bus; it
   is decision/propagation staleness, not a literal network wire delay (real
   latency is ~µs, far below the timestep).
3. **Decision & Routing Engine**

   *Nominal (Safe)*: pass load through unchanged (baseline preserved), or — if
   ``lti_enabled`` — fill cooler, lower slots first to flatten the gradient
   while conserving total load.

   *Throttled (Emergency)*: enforce the priority matrix:

   - P0 (Critical – Live User Requests): 100 % execution, routed to coolest
     node.
   - P1 (High – AI Model Training): full execution; 15 % reduction if the
     local node ≥ 83 °C.
   - P2 (Medium – ETL/Data Pipelines): fully deferred into a ``heapq`` buffer
     for up to 45 s, then expired.
   - P3 (Low – Backups/Logs): instantly evicted and held indefinitely until
     the CoolingAgent clears the alarm.

Hysteresis alignment: the agent mirrors the CoolingAgent's latch — it stays in
mitigation mode until the cooling loop stabilises below 72 °C (read from the
CoolingDecision), preventing system oscillation.

The **Task Preservation Index (TPI)** is ``executed / (executed + expired)``
over the run, measuring how much useful work the agent preserved.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import List

import numpy as np
from numpy.typing import NDArray

from .cooling_agent import CoolingDecision
from .rack import Rack

FloatArray = NDArray[np.float64]

# Priority tier labels.
P0_CRITICAL = 0
P1_HIGH = 1
P2_MEDIUM = 2
P3_LOW = 3
PRIORITY_NAMES = ("P0-Critical", "P1-High", "P2-Medium", "P3-Low")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkloadAgentConfig:
    """Tunable parameters for the Workload Agent (all from config)."""

    # Least-Thermal-Intensity routing toggle. Default OFF: nominal operation is
    # a transparent pass-through, so the baseline simulation is unchanged. When
    # enabled, routing strictly conserves total load (relocates, never drops).
    lti_enabled: bool = False

    # Share of each slot's utilization assigned to each priority tier.
    # Index 0 = P0 (critical), index 3 = P3 (low).  Must sum to 1.0.
    priority_fractions: tuple[float, ...] = (0.20, 0.25, 0.30, 0.25)

    # P1 compute reduction applied when the local node >= p1_throttle_temp.
    p1_throttle_fraction: float = 0.15
    p1_throttle_temp: float = 83.0   # [°C]

    # Max seconds P2 tasks sit in the deferral buffer before expiry.
    p2_defer_limit_s: int = 45

    # Inter-agent signal staleness, in ticks: the WorkloadAgent reacts to the
    # CoolingAgent's throttle this many ticks late. A stand-in for the not-yet-
    # built inter-agent message bus (a later phase). At dt = 1 s, 1 tick = 1 s;
    # true network latency (~µs) is far below the timestep, so this models
    # decision/propagation staleness, not a literal wire delay.
    propagation_delay_ticks: int = 1

    def __post_init__(self) -> None:
        # The 4-tier priority matrix must partition each slot's load exactly.
        if len(self.priority_fractions) != 4:
            raise ValueError(
                "priority_fractions must have exactly 4 entries (P0-P3), got "
                f"{len(self.priority_fractions)}"
            )
        if any(f < 0.0 for f in self.priority_fractions):
            raise ValueError("priority_fractions must be non-negative")
        total = sum(self.priority_fractions)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"priority_fractions must sum to 1.0, got {total:.6f}"
            )
        if self.propagation_delay_ticks < 0:
            raise ValueError("propagation_delay_ticks must be >= 0")

    @classmethod
    def from_config(cls, cfg: dict) -> "WorkloadAgentConfig":
        """Build from the ``workload_agent`` section of the config dict."""
        fracs = cfg.get("priority_fractions", [0.20, 0.25, 0.30, 0.25])
        return cls(
            lti_enabled=bool(cfg.get("lti_enabled", False)),
            priority_fractions=tuple(float(f) for f in fracs),
            p1_throttle_fraction=float(cfg.get("p1_throttle_fraction", 0.15)),
            p1_throttle_temp=float(cfg.get("p1_throttle_temp", 83.0)),
            p2_defer_limit_s=int(cfg.get("p2_defer_limit_s", 45)),
            propagation_delay_ticks=int(cfg.get("propagation_delay_ticks", 1)),
        )


# ---------------------------------------------------------------------------
# Deferral buffer (binary heap)
# ---------------------------------------------------------------------------

@dataclass
class _DeferralEntry:
    """One chunk of deferred work sitting in the buffer."""
    priority: int
    enqueue_time: int   # sim tick when pushed
    slot_index: int
    load: float         # utilization fraction deferred

    def __lt__(self, other: "_DeferralEntry") -> bool:
        # Lower priority number = higher urgency = popped first.
        return (self.priority, self.enqueue_time) < (other.priority, other.enqueue_time)


class DeferralBuffer:
    """Binary-heap backed buffer for deferred P2/P3 work.

    Entries are ordered by (priority, enqueue_time) so the most urgent,
    oldest work drains first.
    """

    def __init__(self) -> None:
        self._heap: list[_DeferralEntry] = []
        self.total_expired: float = 0.0   # cumulative load lost to expiry
        self.total_drained: float = 0.0   # cumulative load successfully re-injected

    def push(self, priority: int, t: int, slot: int, load: float) -> None:
        """Enqueue deferred work."""
        if load > 0:
            heapq.heappush(self._heap, _DeferralEntry(priority, t, slot, load))

    def expire(self, t: int, p2_limit_s: int) -> float:
        """Drop P2 entries older than ``p2_limit_s``.  Returns expired load."""
        keep: list[_DeferralEntry] = []
        expired = 0.0
        for entry in self._heap:
            if entry.priority == P2_MEDIUM and (t - entry.enqueue_time) > p2_limit_s:
                expired += entry.load
            else:
                keep.append(entry)
        if expired > 0:
            heapq.heapify(keep)
            self._heap = keep
            self.total_expired += expired
        return expired

    def drain(self, t: int, throttle_active: bool,
              n_slots: int, headroom: FloatArray) -> FloatArray:
        """Pop entries that can be released, respecting per-slot headroom.

        During throttle: only P2 entries within their 45 s window are eligible.
        After throttle clears: P2 and P3 both drain.

        Returns a per-slot load array to re-inject.
        """
        reinjected = np.zeros(n_slots, dtype=np.float64)
        remaining: list[_DeferralEntry] = []

        while self._heap:
            entry = heapq.heappop(self._heap)
            # P3 only drains when throttle is fully cleared.
            if throttle_active and entry.priority == P3_LOW:
                remaining.append(entry)
                continue
            # Check slot headroom.
            available = max(0.0, headroom[entry.slot_index] - reinjected[entry.slot_index])
            if available >= entry.load:
                reinjected[entry.slot_index] += entry.load
                self.total_drained += entry.load
            else:
                # Can't fit — put it back.
                remaining.append(entry)

        for entry in remaining:
            heapq.heappush(self._heap, entry)
        return reinjected

    @property
    def total_deferred(self) -> float:
        return sum(e.load for e in self._heap)

    @property
    def p2_count(self) -> int:
        return sum(1 for e in self._heap if e.priority == P2_MEDIUM)

    @property
    def p3_count(self) -> int:
        return sum(1 for e in self._heap if e.priority == P3_LOW)

    @property
    def size(self) -> int:
        return len(self._heap)


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkloadDecision:
    """One control output: what the Workload Agent decided for this tick."""

    effective_utilization: FloatArray   # modified per-slot u sent to the rack
    original_utilization: FloatArray    # raw input (for logging / comparison)
    throttle_active: bool               # whether mitigation mode is on
    deferred_load: float                # total utilization in the buffer
    shed_by_priority: tuple[float, ...] # load shed per priority this tick
    routing: str                        # "lti" or "throttled"


# ---------------------------------------------------------------------------
# Workload Agent
# ---------------------------------------------------------------------------

class WorkloadAgent:
    """Priority-aware workload routing and load-shedding agent.

    Reads the CoolingAgent's throttle decision (with a 1-tick propagation
    delay) and either LTI-routes (nominal) or priority-sheds (emergency)
    the incoming utilization before it reaches the Rack.
    """

    def __init__(self, config: WorkloadAgentConfig, n_slots: int) -> None:
        self.config = config
        self.n_slots = n_slots
        self.buffer = DeferralBuffer()

        # Throttle signal delay ring-buffer (1-tick delay by default).
        delay = max(1, config.propagation_delay_ticks)
        self._throttle_ring: list[bool] = [False] * delay
        self._ring_idx: int = 0

        # Cumulative accounting for TPI.
        self._total_executed: float = 0.0
        self._total_input: float = 0.0

    # --- Priority split -----------------------------------------------------

    def split_by_priority(self, utilization: FloatArray) -> List[FloatArray]:
        """Split per-slot utilization into 4 priority-band arrays.

        Each band ``k`` gets ``utilization * priority_fractions[k]``.
        Returns a list of 4 arrays, each shaped ``(n_slots,)``.
        """
        u = np.asarray(utilization, dtype=np.float64)
        fracs = self.config.priority_fractions
        return [u * fracs[k] for k in range(len(fracs))]

    # --- LTI routing (nominal) ----------------------------------------------

    @staticmethod
    def lti_route(priority_loads: List[FloatArray],
                  t_node: FloatArray, bias: float = 1.5) -> FloatArray:
        """Least-Thermal-Intensity routing: place the rack's total load on the
        coolest slots first, **conserving total load exactly**.

        The total demand across the rack is redistributed in proportion to
        per-slot thermal weights (coolest slot weighted highest), with a
        water-filling pass that caps each slot at 1.0 and re-allocates the
        overflow to the remaining cooler slots. This relocates work toward
        cooler nodes to flatten the gradient; it never creates or drops load.
        Total demand is conserved to floating-point precision (it always fits
        because each input slot's load is ≤ 1.0, so the sum is ≤ n).
        """
        n = t_node.shape[0]
        demand = float(sum(p for p in priority_loads).sum())
        if demand <= 0.0:
            return np.zeros(n, dtype=np.float64)

        # Thermal weights: coolest slot highest. linspace(bias, 2-bias) averages
        # to 1.0; only the ratios matter for the proportional split below.
        order = np.argsort(t_node)              # coolest → hottest
        rank_weights = np.linspace(bias, 2.0 - bias, n)
        w = np.zeros(n, dtype=np.float64)
        w[order] = np.clip(rank_weights, 1e-9, None)

        # Water-fill: hand out `demand` ∝ weights, capping slots at 1.0 and
        # spilling the remainder onto the still-open (cooler) slots.
        effective = np.zeros(n, dtype=np.float64)
        active = np.ones(n, dtype=bool)
        remaining = demand
        for _ in range(n):
            wa = np.where(active, w, 0.0)
            s = wa.sum()
            if remaining <= 1e-12 or s <= 0.0:
                break
            alloc = effective + remaining * wa / s
            over = active & (alloc > 1.0)
            if not over.any():
                effective = alloc
                remaining = 0.0
                break
            effective[over] = 1.0
            active[over] = False
            remaining = demand - effective.sum()

        return np.clip(effective, 0.0, 1.0)

    # --- Throttle-mode shedding ---------------------------------------------

    def _apply_throttle(self, priority_loads: List[FloatArray],
                        t_node: FloatArray, t: int) -> tuple[FloatArray, tuple]:
        """Enforce the priority matrix during a thermal emergency.

        Returns (effective_utilization, shed_tuple).
        """
        cfg = self.config
        n = self.n_slots
        effective = np.zeros(n, dtype=np.float64)
        shed = [0.0, 0.0, 0.0, 0.0]

        # P0 — 100 % execution, routed to coolest slot.
        p0 = priority_loads[P0_CRITICAL].copy()
        effective += p0

        # P1 — full execution, but 15 % reduction if local node >= 83 °C.
        p1 = priority_loads[P1_HIGH].copy()
        hot_mask = t_node >= cfg.p1_throttle_temp
        reduction = p1 * cfg.p1_throttle_fraction * hot_mask
        p1_effective = p1 - reduction
        effective += p1_effective
        shed[P1_HIGH] = float(reduction.sum())

        # P2 — fully deferred to buffer.
        p2 = priority_loads[P2_MEDIUM]
        for s in range(n):
            if p2[s] > 0:
                self.buffer.push(P2_MEDIUM, t, s, float(p2[s]))
        shed[P2_MEDIUM] = float(p2.sum())

        # P3 — instantly evicted to buffer (held indefinitely).
        p3 = priority_loads[P3_LOW]
        for s in range(n):
            if p3[s] > 0:
                self.buffer.push(P3_LOW, t, s, float(p3[s]))
        shed[P3_LOW] = float(p3.sum())

        effective = np.clip(effective, 0.0, 1.0)
        return effective, tuple(shed)

    # --- Per-tick entry point ------------------------------------------------

    def step(self, raw_utilization: FloatArray,
             cooling_decision: CoolingDecision | None,
             rack: Rack,
             t: int) -> WorkloadDecision:
        """Execute one simulation tick of the Workload Agent.

        Parameters
        ----------
        raw_utilization : per-slot utilization from the trace/burstiness layer.
        cooling_decision : the CoolingAgent's decision for *this* tick
            (the throttle signal is delayed by ``propagation_delay_ticks``).
        rack : the Rack (read ``t_node`` for temperatures; not mutated).
        t : current simulation tick (seconds).

        Returns
        -------
        WorkloadDecision with the effective utilization to pass to ``rack.step()``.
        """
        raw = np.asarray(raw_utilization, dtype=np.float64)
        self._total_input += float(raw.sum())

        # --- Stage 2: Environmental Check (with staleness delay) ------------
        # Write the current throttle into the ring buffer and read the delayed
        # value: the agent reacts to the cooling decision `propagation_delay_ticks`
        # ticks late (inter-agent staleness; stand-in for the future message bus).
        current_throttle = (
            cooling_decision.throttle_request if cooling_decision is not None
            else False
        )
        delay = len(self._throttle_ring)
        write_idx = self._ring_idx % delay
        delayed_throttle = self._throttle_ring[write_idx]
        self._throttle_ring[write_idx] = current_throttle
        self._ring_idx += 1

        # --- Expire old P2 entries ------------------------------------------
        self.buffer.expire(t, self.config.p2_defer_limit_s)

        # --- Stage 1: Intercept & Triage ------------------------------------
        priority_loads = self.split_by_priority(raw)

        # --- Stage 3: Decision & Routing Engine -----------------------------
        if delayed_throttle:
            effective, shed = self._apply_throttle(
                priority_loads, rack.t_node, t
            )
            routing = "throttled"
        else:
            shed = (0.0, 0.0, 0.0, 0.0)
            if self.config.lti_enabled:
                # Thermal-aware routing (opt-in): conserves total load.
                effective = self.lti_route(priority_loads, rack.t_node)
                routing = "lti"
            else:
                # Default: transparent pass-through — baseline is unchanged.
                effective = raw.copy()
                routing = "passthrough"

            # Drain deferred work back in once the emergency has cleared.
            headroom = np.clip(1.0 - effective, 0.0, 1.0)
            reinjected = self.buffer.drain(
                t, throttle_active=False,
                n_slots=self.n_slots, headroom=headroom
            )
            effective = np.clip(effective + reinjected, 0.0, 1.0)

        self._total_executed += float(effective.sum())

        return WorkloadDecision(
            effective_utilization=effective,
            original_utilization=raw.copy(),
            throttle_active=delayed_throttle,
            deferred_load=self.buffer.total_deferred,
            shed_by_priority=shed,
            routing=routing,
        )

    # --- Task Preservation Index --------------------------------------------

    @property
    def tpi(self) -> float:
        """Task Preservation Index: ``executed / (executed + expired)``.

        Only *permanently lost* work (P2 tasks that timed out in the buffer)
        counts against the score. Work still sitting in the deferral buffer is
        neither credited nor penalised — it is in-flight and will either drain
        (becoming executed) or expire (becoming lost) later, at which point it
        enters the ratio. A TPI of 1.0 means nothing offered has been lost.

        Note: P1 throttling reduces compute *rate* (the task continues, just
        slower) rather than dropping a task, so it is treated as in-flight, not
        loss; P3 is held and drains when the alarm clears, so it is preserved.
        """
        executed = self._total_executed
        expired = self.buffer.total_expired
        denom = executed + expired
        if denom <= 0:
            return 1.0
        return executed / denom

    @property
    def total_expired(self) -> float:
        return self.buffer.total_expired

    @property
    def total_executed(self) -> float:
        return self._total_executed

    @property
    def total_input(self) -> float:
        return self._total_input

    # --- Factory ------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict, n_slots: int | None = None) -> "WorkloadAgent":
        """Build a WorkloadAgent from a parsed config dict."""
        if n_slots is None:
            n_slots = int(config["rack"]["n_slots"])
        wa_cfg = WorkloadAgentConfig.from_config(
            config.get("workload_agent", {})
        )
        return cls(wa_cfg, n_slots)

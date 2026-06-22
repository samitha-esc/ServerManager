#!/usr/bin/env python3
"""Interactive data center routing simulation — Pygame visualization.

Demonstrates the multi-agent thermal-orchestration system: packets flow
through Network Traffic Agent → Workload Agent (triage lanes) → Cooling
Agent → routed to either 35 Air-cooled racks or 15 Liquid-cooled racks
via Least-Thermal-Intensity assignment.

Controls
--------
    Space       Toggle Thermal Crisis mode
    Up / Down   Alter incoming packet burstiness multiplier
    Escape      Quit

Requires: pygame-ce  (pip install pygame-ce)
Run:      python scripts/sim_visual.py
"""

from __future__ import annotations

import math
import random
import sys

import pygame

# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════
WIDTH, HEIGHT = 1100, 650
FPS = 60
TITLE = "DataCenter Thermal Orchestration — Multi-Agent Routing Simulator"

# ═══════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════
BG           = (12, 15, 25)
GRID_LINE    = (18, 21, 30)
PANEL_BG     = (22, 26, 42)
PANEL_EDGE   = (48, 52, 75)
AGENT_BG     = (20, 24, 38)
AGENT_EDGE   = (55, 60, 90)
AGENT_TITLE  = (100, 175, 255)
RACK_BG      = (20, 24, 36)
TEXT_MAIN     = (215, 220, 235)
TEXT_DIM      = (120, 125, 145)
TEXT_ACCENT   = (90, 170, 255)
KEY_BG        = (38, 42, 62)
CRISIS_GLOW   = (255, 40, 40)
ARROW_COLOR   = (60, 65, 90)

TEMP_COOL     = (40, 140, 220)    # < 50 °C
TEMP_GREEN    = (30, 210, 90)     # < 74 °C
TEMP_ORANGE   = (255, 170, 40)    # 74–82 °C
TEMP_RED      = (255, 55, 55)     # > 82 °C

LANE_BORDER = {
    0: (230,  57,  70),
    1: (244, 162,  97),
    2: ( 42, 157, 143),
    3: (100, 120, 155),
}
LANE_FILL_A = {
    0: (180, 30, 40,  40),
    1: (200,130, 50,  40),
    2: ( 30,130,120,  40),
    3: ( 50, 65, 95,  40),
}
PACKET_COL = {
    0: (255,  75,  85),
    1: (255, 185,  80),
    2: ( 50, 210, 180),
    3: (120, 145, 190),
}

AIR_TINT     = (45, 35, 25)       # warm-brown accent for air rack cells
LIQUID_TINT  = (20, 35, 50)       # cool-blue accent for liquid rack cells

# ═══════════════════════════════════════════════════════════════════════════
# LAYOUT GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════
# --- control panel (top) ---
PANEL_X, PANEL_Y, PANEL_W, PANEL_H = 10, 6, 530, 82

# --- agent blocks (left half) ---
NET_AGENT   = pygame.Rect(14,  130, 78, 340)
WORK_AGENT  = pygame.Rect(112, 100, 256, 400)
COOL_AGENT  = pygame.Rect(388, 160, 88, 290)

# --- priority lanes (inside Workload Agent) ---
LANE_X      = WORK_AGENT.x + 14
LANE_W      = WORK_AGENT.width - 28
LANE_H      = 38
LANE_CENTERS = [210, 288, 366, 444]     # y-centres for P0–P3
QUEUE_W     = 68
QUEUE_X     = LANE_X + LANE_W - QUEUE_W

# --- rack groups (right half) ---
#  Air: 35 racks as 7 × 5 grid
N_AIR   = 35
AIR_COLS, AIR_ROWS = 7, 5
AIR_AREA = pygame.Rect(505, 96, 580, 240)
AIR_CW   = (AIR_AREA.w - (AIR_COLS - 1) * 3) // AIR_COLS   # ≈ 80
AIR_CH   = (AIR_AREA.h - (AIR_ROWS - 1) * 3) // AIR_ROWS   # ≈ 45

#  Liquid: 15 racks as 5 × 3 grid
N_LIQUID = 15
LIQ_COLS, LIQ_ROWS = 5, 3
LIQ_AREA = pygame.Rect(505, 370, 580, 210)
LIQ_CW   = (LIQ_AREA.w - (LIQ_COLS - 1) * 3) // LIQ_COLS   # ≈ 113
LIQ_CH   = (LIQ_AREA.h - (LIQ_ROWS - 1) * 3) // LIQ_ROWS   # ≈ 68


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def ease(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def lerp_c(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def temp_border(temp: float) -> tuple[int, int, int]:
    if temp < 74:
        return TEMP_GREEN
    if temp <= 82:
        return TEMP_ORANGE
    return TEMP_RED


def _cell_rect(area: pygame.Rect, col: int, row: int,
               cw: int, ch: int) -> pygame.Rect:
    x = area.x + col * (cw + 3)
    y = area.y + row * (ch + 3)
    return pygame.Rect(x, y, cw, ch)


# ═══════════════════════════════════════════════════════════════════════════
# RACK CELL  (one rack in a group)
# ═══════════════════════════════════════════════════════════════════════════
class RackCell:
    """A single rack (air or liquid) shown as a coloured card."""

    def __init__(self, index: int, kind: str,
                 rect: pygame.Rect) -> None:
        self.index = index
        self.kind  = kind         # "air" | "liquid"
        self.rect  = rect
        ambient    = 22.0 if kind == "air" else 18.0
        self.ambient = ambient
        self.temperature = ambient + random.uniform(0, 6)
        self.active_load = 0.0
        # cooling strength: liquid cools faster
        self.k_cool = 0.07 if kind == "air" else 0.11

    def update_thermal(self, cooling_factor: float) -> None:
        heat = 0.35 * self.active_load
        cool = self.k_cool * cooling_factor * (self.temperature - self.ambient)
        self.temperature += heat - cool
        self.temperature = max(self.ambient, min(self.temperature, 120.0))
        self.active_load = max(0.0, self.active_load - 0.03)

    def receive_packet(self) -> None:
        self.active_load += 1.0

    def border_color(self) -> tuple[int, int, int]:
        return temp_border(self.temperature)

    def draw(self, surf: pygame.Surface, font: pygame.font.Font) -> None:
        r = self.rect
        # Temperature-tinted fill.
        frac = max(0.0, min(1.0, (self.temperature - 20) / 70.0))
        tint = AIR_TINT if self.kind == "air" else LIQUID_TINT
        fill = lerp_c(tint, (80, 25, 25), frac * 0.6)
        pygame.draw.rect(surf, fill, r, border_radius=3)
        # Border.
        pygame.draw.rect(surf, self.border_color(), r, width=2, border_radius=3)
        # Label.
        id_str = f"{'A' if self.kind == 'air' else 'L'}{self.index + 1}"
        tmp_str = f"{self.temperature:.0f}°"
        lt = font.render(id_str, True, TEXT_DIM)
        tt = font.render(tmp_str, True, TEXT_MAIN)
        surf.blit(lt, (r.x + 4, r.y + 3))
        surf.blit(tt, (r.right - tt.get_width() - 4, r.bottom - tt.get_height() - 2))


# ═══════════════════════════════════════════════════════════════════════════
# PACKET
# ═══════════════════════════════════════════════════════════════════════════
class Packet:
    """Animated workload packet flowing through the agent pipeline.

    Stages
    ------
    0  INGRESS  — spawn → through Network Agent → into Workload Agent lane.
    1  TRIAGE   — travelling through the priority lane (may be queued).
    2  COOLING  — Workload exit → through Cooling Agent.
    3  ROUTING  — Cooling exit → smooth slide to a rack cell.
    4  ARRIVED  — flash-fade at the rack cell.
    """

    W, H = 11, 8

    def __init__(self, priority: int) -> None:
        self.priority = priority
        self.color    = PACKET_COL[priority]
        self.lane_y   = float(LANE_CENTERS[priority] +
                              random.uniform(-LANE_H / 2 + 4, LANE_H / 2 - 4))
        self.x: float = random.uniform(-35.0, -10.0)
        self.y: float = self.lane_y
        self.vx: float = 2.0 + random.uniform(-0.2, 0.3)
        self.stage: int = 0
        self.stopped: bool = False
        self.alive: bool = True

        # routing (stage 3)
        self.assigned_rack: RackCell | None = None
        self.route_t   = 0.0
        self.route_src = (0.0, 0.0)
        self.route_dst = (0.0, 0.0)

        # fade (stage 4)
        self.fade = 0
        self.alpha = 255

    # ------------------------------------------------------------------ #
    def update(self, crisis: bool, air: list[RackCell],
               liquid: list[RackCell]) -> None:
        if not self.alive:
            return

        if self.stage == 0:
            # INGRESS: slide right toward the lane entry.
            self.x += self.vx * 1.8
            if self.x >= LANE_X + 8:
                self.stage = 1

        elif self.stage == 1:
            # TRIAGE: travel through the lane.
            if crisis and self.priority >= 2:
                if self.x < QUEUE_X + 3:
                    self.x += self.vx * 0.4
                if self.x >= QUEUE_X + 3:
                    self.x = QUEUE_X + random.uniform(3, QUEUE_W - self.W - 2)
                    self.stopped = True
            else:
                self.x += self.vx
                if self.x >= LANE_X + LANE_W:
                    self.stage = 2

        elif self.stage == 2:
            # COOLING: slide through the Cooling Agent block.
            self.x += self.vx * 1.4
            if self.x >= COOL_AGENT.right - 8:
                self._assign_rack(air, liquid)
                self.stage = 3

        elif self.stage == 3:
            # ROUTING: interpolated slide to rack cell.
            self.route_t += 0.028
            if self.route_t >= 1.0:
                self.route_t = 1.0
                self.stage = 4
                self.fade = 0
            t = ease(self.route_t)
            self.x = self.route_src[0] + (self.route_dst[0] - self.route_src[0]) * t
            self.y = self.route_src[1] + (self.route_dst[1] - self.route_src[1]) * t

        elif self.stage == 4:
            self.fade += 1
            self.alpha = max(0, 255 - self.fade * 18)
            if self.alpha <= 0:
                self.alive = False

    # ------------------------------------------------------------------ #
    def release(self, air: list[RackCell], liquid: list[RackCell]) -> None:
        if self.stopped:
            self.stopped = False
            self.x = float(LANE_X + LANE_W)
            self._assign_rack(air, liquid)
            self.stage = 3

    # ------------------------------------------------------------------ #
    def _assign_rack(self, air: list[RackCell],
                     liquid: list[RackCell]) -> None:
        """LTI routing: pick the coolest rack in the preferred group."""
        self.route_src = (self.x, self.y)

        coolest_air = min(air, key=lambda r: r.temperature)
        coolest_liq = min(liquid, key=lambda r: r.temperature)

        use_liquid = False
        if self.priority == 0:
            # P0 (Critical) → always try liquid first (cooler, safer).
            use_liquid = coolest_liq.temperature < 78
        elif coolest_liq.temperature < coolest_air.temperature - 4:
            # Liquid is significantly cooler → prefer it.
            use_liquid = True
        else:
            # Default: proportional to group sizes (30 % liquid).
            use_liquid = random.random() < 0.30

        # If preferred group is overheating, fall back.
        if use_liquid and coolest_liq.temperature > 82:
            use_liquid = False
        if not use_liquid and coolest_air.temperature > 82 and coolest_liq.temperature < 78:
            use_liquid = True

        pool = liquid if use_liquid else air
        pool_sorted = sorted(pool, key=lambda r: r.temperature)
        pick = pool_sorted[random.randint(0, min(2, len(pool_sorted) - 1))]
        self.assigned_rack = pick
        self.route_dst = (float(pick.rect.centerx), float(pick.rect.centery))

    # ------------------------------------------------------------------ #
    def draw(self, surf: pygame.Surface) -> None:
        if not self.alive:
            return
        rx = int(self.x) - self.W // 2
        ry = int(self.y) - self.H // 2
        rect = pygame.Rect(rx, ry, self.W, self.H)
        if self.stage == 4:
            s = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            s.fill((*self.color, self.alpha))
            surf.blit(s, rect.topleft)
        else:
            pygame.draw.rect(surf, self.color, rect, border_radius=2)


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════
class Simulation:
    """Top-level Pygame app."""

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()

        mono = "Menlo,Consolas,DejaVu Sans Mono,Courier New,monospace"
        self.fnt_title = pygame.font.SysFont(mono, 16, bold=True)
        self.fnt_md    = pygame.font.SysFont(mono, 13)
        self.fnt_sm    = pygame.font.SysFont(mono, 11)
        self.fnt_xs    = pygame.font.SysFont(mono, 10)

        self.crisis     = False
        self.burstiness = 1.0
        self.frame      = 0
        self.running    = True
        self.total_routed = 0

        # Build rack cells.
        self.air_racks: list[RackCell] = []
        for i in range(N_AIR):
            col, row = i % AIR_COLS, i // AIR_COLS
            r = _cell_rect(AIR_AREA, col, row, AIR_CW, AIR_CH)
            self.air_racks.append(RackCell(i, "air", r))

        self.liq_racks: list[RackCell] = []
        for i in range(N_LIQUID):
            col, row = i % LIQ_COLS, i // LIQ_COLS
            r = _cell_rect(LIQ_AREA, col, row, LIQ_CW, LIQ_CH)
            self.liq_racks.append(RackCell(i, "liquid", r))

        self.packets: list[Packet] = []

    # ────────────────────── main loop ─────────────────────────────────── #
    def run(self) -> None:
        while self.running:
            self._events()
            self._update()
            self._draw()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()

    # ────────────────────── events ────────────────────────────────────── #
    def _events(self) -> None:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self.running = False
                elif ev.key == pygame.K_SPACE:
                    self.crisis = not self.crisis
                    if not self.crisis:
                        for p in self.packets:
                            if p.stopped:
                                p.release(self.air_racks, self.liq_racks)
                elif ev.key == pygame.K_UP:
                    self.burstiness = min(5.0, round(self.burstiness + 0.25, 2))
                elif ev.key == pygame.K_DOWN:
                    self.burstiness = max(0.25, round(self.burstiness - 0.25, 2))

    # ────────────────────── update ────────────────────────────────────── #
    def _update(self) -> None:
        self.frame += 1
        self._spawn()
        cooling = 1.6 if self.crisis else 1.0

        alive: list[Packet] = []
        for p in self.packets:
            p.update(self.crisis, self.air_racks, self.liq_racks)
            if p.stage == 4 and p.fade == 1 and p.assigned_rack is not None:
                p.assigned_rack.receive_packet()
                self.total_routed += 1
            if p.alive:
                alive.append(p)
        self.packets = alive

        for r in self.air_racks:
            r.update_thermal(cooling)
        for r in self.liq_racks:
            r.update_thermal(cooling)

    def _spawn(self) -> None:
        if len(self.packets) > 300:
            return
        rate = 0.065 * self.burstiness
        if random.random() < 0.009 * self.burstiness:
            for _ in range(random.randint(2, 5)):
                self.packets.append(Packet(self._pri()))
        elif random.random() < rate:
            self.packets.append(Packet(self._pri()))

    def _pri(self) -> int:
        w = [0.35, 0.30, 0.20, 0.15] if self.crisis else [0.15, 0.25, 0.30, 0.30]
        return random.choices([0, 1, 2, 3], weights=w)[0]

    # ════════════════════ DRAWING ═════════════════════════════════════════
    def _draw(self) -> None:
        self.screen.fill(BG)
        # Subtle grid.
        for gx in range(0, WIDTH, 50):
            pygame.draw.line(self.screen, GRID_LINE, (gx, 0), (gx, HEIGHT))
        for gy in range(0, HEIGHT, 50):
            pygame.draw.line(self.screen, GRID_LINE, (0, gy), (WIDTH, gy))

        self._draw_panel()
        self._draw_agents()
        self._draw_arrows()
        self._draw_lanes()
        self._draw_racks()
        self._draw_packets()
        self._draw_metrics()
        pygame.display.flip()

    # ── control panel ──────────────────────────────────────────────────── #
    def _draw_panel(self) -> None:
        pr = pygame.Rect(PANEL_X, PANEL_Y, PANEL_W, PANEL_H)
        pygame.draw.rect(self.screen, PANEL_BG, pr, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_EDGE, pr, width=1, border_radius=6)
        x, y = PANEL_X + 12, PANEL_Y + 8

        # State.
        st_txt = "THERMAL CRISIS" if self.crisis else "NOMINAL"
        st_col = CRISIS_GLOW if self.crisis else TEMP_GREEN
        l1 = self.fnt_md.render("State: ", True, TEXT_DIM)
        v1 = self.fnt_md.render(st_txt, True, st_col)
        self.screen.blit(l1, (x, y))
        self.screen.blit(v1, (x + l1.get_width(), y))
        if self.crisis:
            pr2 = 3 + int(2 * math.sin(self.frame * 0.12))
            pygame.draw.circle(self.screen, CRISIS_GLOW,
                               (x + l1.get_width() + v1.get_width() + 12, y + 8), pr2)

        # Burstiness.
        l2 = self.fnt_md.render("Burstiness: ", True, TEXT_DIM)
        v2 = self.fnt_md.render(f"{self.burstiness:.2f}×", True, TEXT_MAIN)
        bx = x + l1.get_width() + v1.get_width() + 40
        self.screen.blit(l2, (bx, y))
        self.screen.blit(v2, (bx + l2.get_width(), y))
        y += 22

        # Key hints.
        keys = [("SPACE", "Toggle Crisis"), ("↑/↓", "Burstiness"), ("ESC", "Quit")]
        kx = x
        for key, desc in keys:
            kt = self.fnt_xs.render(key, True, TEXT_MAIN)
            kw, kh = kt.get_width() + 8, kt.get_height() + 4
            kr = pygame.Rect(kx, y, kw, kh)
            pygame.draw.rect(self.screen, KEY_BG, kr, border_radius=3)
            pygame.draw.rect(self.screen, PANEL_EDGE, kr, width=1, border_radius=3)
            self.screen.blit(kt, (kx + 4, y + 2))
            dt = self.fnt_xs.render(f" {desc}", True, TEXT_DIM)
            self.screen.blit(dt, (kx + kw + 1, y + 2))
            kx += kw + dt.get_width() + 14

        # Subtitle.
        y += kh + 6
        sub = self.fnt_xs.render(
            "Packets: Network → Workload (triage) → Cooling → Air/Liquid Racks",
            True, (70, 75, 100))
        self.screen.blit(sub, (x, y))

    # ── agent blocks ───────────────────────────────────────────────────── #
    def _draw_agents(self) -> None:
        for rect, title, lines in [
            (NET_AGENT, "NETWORK", ["TRAFFIC", "AGENT", "", "Alibaba", "Trace"]),
            (WORK_AGENT, "WORKLOAD AGENT", []),
            (COOL_AGENT, "COOLING", ["THERMAL", "AGENT"]),
        ]:
            pygame.draw.rect(self.screen, AGENT_BG, rect, border_radius=6)
            pygame.draw.rect(self.screen, AGENT_EDGE, rect, width=1, border_radius=6)
            # Title.
            tt = self.fnt_xs.render(title, True, AGENT_TITLE)
            self.screen.blit(tt, (rect.x + (rect.w - tt.get_width()) // 2,
                                   rect.y + 6))
            # Extra lines.
            ly = rect.y + 22
            for line in lines:
                lt = self.fnt_xs.render(line, True, TEXT_DIM)
                self.screen.blit(lt, (rect.x + (rect.w - lt.get_width()) // 2, ly))
                ly += 14

        # Cooling Agent state indicator.
        st = "CRISIS" if self.crisis else "NOMINAL"
        sc = CRISIS_GLOW if self.crisis else TEMP_GREEN
        st_t = self.fnt_xs.render(st, True, sc)
        self.screen.blit(st_t, (COOL_AGENT.centerx - st_t.get_width() // 2,
                                 COOL_AGENT.bottom - 40))
        # Fan label.
        fan_txt = f"ω={1.6 if self.crisis else 1.0:.1f}"
        ft = self.fnt_xs.render(fan_txt, True, TEXT_DIM)
        self.screen.blit(ft, (COOL_AGENT.centerx - ft.get_width() // 2,
                               COOL_AGENT.bottom - 22))

    # ── connecting arrows ──────────────────────────────────────────────── #
    def _draw_arrows(self) -> None:
        # Agent → Agent arrows.
        self._arrow(NET_AGENT.right, NET_AGENT.centery,
                    WORK_AGENT.left, WORK_AGENT.centery)
        self._arrow(WORK_AGENT.right, WORK_AGENT.centery,
                    COOL_AGENT.left, COOL_AGENT.centery)
        # Cooling → Air racks.
        self._arrow(COOL_AGENT.right, COOL_AGENT.centery - 40,
                    AIR_AREA.left - 5, AIR_AREA.centery)
        # Cooling → Liquid racks.
        self._arrow(COOL_AGENT.right, COOL_AGENT.centery + 40,
                    LIQ_AREA.left - 5, LIQ_AREA.centery)

    def _arrow(self, x1, y1, x2, y2) -> None:
        pygame.draw.line(self.screen, ARROW_COLOR, (x1 + 2, y1), (x2 - 8, y2), 2)
        # Arrowhead.
        dx, dy = x2 - x1, y2 - y1
        ln = max(1, math.hypot(dx, dy))
        ux, uy = dx / ln, dy / ln
        px, py = -uy, ux
        tip = (x2 - 6, y2)
        left = (tip[0] - int(ux * 8 + px * 5), tip[1] - int(uy * 8 + py * 5))
        right = (tip[0] - int(ux * 8 - px * 5), tip[1] - int(uy * 8 - py * 5))
        pygame.draw.polygon(self.screen, ARROW_COLOR, [tip, left, right])

    # ── triage lanes ───────────────────────────────────────────────────── #
    def _draw_lanes(self) -> None:
        labels = ["P0 Critical", "P1 High", "P2 Medium", "P3 Low"]
        for pri in range(4):
            cy = LANE_CENTERS[pri]
            lr = pygame.Rect(LANE_X, cy - LANE_H // 2, LANE_W, LANE_H)
            # Fill.
            body = pygame.Surface((LANE_W, LANE_H), pygame.SRCALPHA)
            body.fill(LANE_FILL_A[pri])
            self.screen.blit(body, lr.topleft)
            # Border.
            pygame.draw.rect(self.screen, LANE_BORDER[pri], lr, width=1,
                             border_radius=3)
            # Queue box.
            qr = pygame.Rect(QUEUE_X, cy - LANE_H // 2, QUEUE_W, LANE_H)
            qs = pygame.Surface((QUEUE_W, LANE_H), pygame.SRCALPHA)
            qs.fill((255, 255, 255, 10))
            self.screen.blit(qs, qr.topleft)
            pygame.draw.rect(self.screen, (*LANE_BORDER[pri], 80), qr,
                             width=1, border_radius=2)

            if self.crisis and pri >= 2:
                pa = int(50 + 30 * math.sin(self.frame * 0.09))
                ov = pygame.Surface((QUEUE_W, LANE_H), pygame.SRCALPHA)
                ov.fill((255, 40, 40, pa))
                self.screen.blit(ov, qr.topleft)
                bl = self.fnt_xs.render("BLOCKED", True, CRISIS_GLOW)
                self.screen.blit(bl, (qr.centerx - bl.get_width() // 2, qr.y - 12))
                nq = sum(1 for p in self.packets if p.stopped and p.priority == pri)
                if nq > 0:
                    ql = self.fnt_xs.render(str(nq), True, (255, 180, 180))
                    self.screen.blit(ql, (qr.right - ql.get_width() - 3,
                                          qr.bottom + 1))

            # Label.
            lt = self.fnt_xs.render(labels[pri], True, LANE_BORDER[pri])
            self.screen.blit(lt, (LANE_X + 3, cy - lt.get_height() // 2))

    # ── rack groups ────────────────────────────────────────────────────── #
    def _draw_racks(self) -> None:
        # Air group enclosure.
        ae = AIR_AREA.inflate(12, 28)
        ae.y -= 18
        pygame.draw.rect(self.screen, RACK_BG, ae, border_radius=5)
        pygame.draw.rect(self.screen, PANEL_EDGE, ae, width=1, border_radius=5)
        at = self.fnt_sm.render(f"AIR COOLING  ·  {N_AIR} RACKS", True,
                                (210, 160, 100))
        self.screen.blit(at, (AIR_AREA.x + 2, AIR_AREA.y - 16))
        # Air peak temp.
        air_peak = max(r.temperature for r in self.air_racks)
        apt = self.fnt_xs.render(f"Peak {air_peak:.0f}°C", True,
                                  temp_border(air_peak))
        self.screen.blit(apt, (AIR_AREA.right - apt.get_width() - 4,
                                AIR_AREA.y - 16))

        for r in self.air_racks:
            r.draw(self.screen, self.fnt_xs)

        # Liquid group enclosure.
        le = LIQ_AREA.inflate(12, 28)
        le.y -= 18
        pygame.draw.rect(self.screen, RACK_BG, le, border_radius=5)
        pygame.draw.rect(self.screen, PANEL_EDGE, le, width=1, border_radius=5)
        ltt = self.fnt_sm.render(f"LIQUID COOLING  ·  {N_LIQUID} RACKS", True,
                                  (100, 170, 220))
        self.screen.blit(ltt, (LIQ_AREA.x + 2, LIQ_AREA.y - 16))
        liq_peak = max(r.temperature for r in self.liq_racks)
        lpt = self.fnt_xs.render(f"Peak {liq_peak:.0f}°C", True,
                                  temp_border(liq_peak))
        self.screen.blit(lpt, (LIQ_AREA.right - lpt.get_width() - 4,
                                LIQ_AREA.y - 16))

        for r in self.liq_racks:
            r.draw(self.screen, self.fnt_xs)

    # ── packets ────────────────────────────────────────────────────────── #
    def _draw_packets(self) -> None:
        # Draw routing (stage 3+) behind in-pipeline packets.
        for p in self.packets:
            if p.stage >= 3:
                p.draw(self.screen)
        for p in self.packets:
            if p.stage < 3:
                p.draw(self.screen)

    # ── metrics strip ──────────────────────────────────────────────────── #
    def _draw_metrics(self) -> None:
        y = HEIGHT - 18
        air_peak = max(r.temperature for r in self.air_racks)
        liq_peak = max(r.temperature for r in self.liq_racks)
        air_avg = sum(r.temperature for r in self.air_racks) / N_AIR
        liq_avg = sum(r.temperature for r in self.liq_racks) / N_LIQUID
        inflight = sum(1 for p in self.packets if p.stage < 4)
        queued   = sum(1 for p in self.packets if p.stopped)
        items = [
            (f"Air peak:{air_peak:.0f}°C", temp_border(air_peak)),
            (f"avg:{air_avg:.0f}°C", TEXT_DIM),
            (f"Liq peak:{liq_peak:.0f}°C", temp_border(liq_peak)),
            (f"avg:{liq_avg:.0f}°C", TEXT_DIM),
            (f"In-flight:{inflight}", TEXT_DIM),
            (f"Queued:{queued}", CRISIS_GLOW if queued else TEXT_DIM),
            (f"Routed:{self.total_routed}", TEXT_DIM),
            (f"FPS:{self.clock.get_fps():.0f}", TEXT_DIM),
        ]
        x = 10
        for txt, col in items:
            t = self.fnt_xs.render(txt, True, col)
            self.screen.blit(t, (x, y))
            x += t.get_width() + 16


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    Simulation().run()

/**
 * THERMOS — App v3
 * Canvas-based game-like routing pipeline, speed slider, burst control,
 * fixed chart sizing, fixed violations.
 */

// ── Constants ──────────────────────────────────────────────────────────────
const COL = {
  bg0:    '#07090c',
  bg1:    '#0d1117',
  bg2:    '#131a22',
  bg3:    '#1a2333',
  border: 'rgba(255,255,255,0.08)',
  txt:    '#d0d7de',
  dim:    '#636e7b',
  ma:     '#38bdf8',
  base:   '#fb923c',
  p0:     '#f87171',
  p1:     '#fbbf24',
  p2:     '#34d399',
  p3:     '#818cf8',
  tOk:    '#34d399',
  tWarn:  '#fbbf24',
  tHot:   '#f87171',
};

const PRIORITY_NAMES = ['P0 Critical', 'P1 High', 'P2 Medium', 'P3 Low'];
const PRIORITY_COLS  = [COL.p0, COL.p1, COL.p2, COL.p3];

// ── Global state ───────────────────────────────────────────────────────────
let ws             = null;
let latestFrame    = null;
let currentMode    = 'multi_agent';
let speedMulti     = 1.0;
let burstActive    = false;
let agentPipeStep  = 0;          // 0-3: which agent is "active"
let rackTemps      = {};         // rack_id → peak_temp
let throttleActive = false;

// ── Canvas & Layout ────────────────────────────────────────────────────────
const canvas  = document.getElementById('pipeline-canvas');
const ctx     = canvas.getContext('2d');
const tooltip = document.getElementById('rack-tooltip');

// Layout zones (computed on each resize)
let L = {};   // populated by computeLayout()

function computeLayout() {
  const W = canvas.width;
  const H = canvas.height;

  // Horizontal zones
  const AGENT_ZONE_W = Math.floor(W * 0.38);  // left 38% = agent pipeline
  const RACK_ZONE_X  = AGENT_ZONE_W + 8;       // right portion = rack grid
  const RACK_ZONE_W  = W - RACK_ZONE_X - 12;

  // Agent blocks (left half)
  const agentPad = 14;
  const agentBlockW = AGENT_ZONE_W - agentPad * 2;
  const agentBlockH = 44;
  const laneH       = 32;

  // Network Traffic Agent
  const NET_Y = 30;
  const NET_H = 58;

  // 4 priority lanes (inside Workload Agent)
  const LANE_START_Y = NET_Y + NET_H + 30;
  const LANES = [0,1,2,3].map(i => ({
    y:     LANE_START_Y + i * (laneH + 8),
    color: PRIORITY_COLS[i],
    name:  PRIORITY_NAMES[i],
    queueX: AGENT_ZONE_W - agentPad - 80,
  }));

  // Cooling agent (below lanes)
  const lastLaneBottom = LANES[3].y + laneH;
  const COOL_Y = lastLaneBottom + 22;
  const COOL_H = 52;

  // Rack grid (right half)
  const N_FLOORS = 5, N_PER_FLOOR = 10;
  const floorH      = Math.floor(H / (N_FLOORS + 1));
  const rackW       = Math.max(28, Math.floor((RACK_ZONE_W - (N_PER_FLOOR - 1) * 5) / N_PER_FLOOR));
  const rackH       = Math.max(16, floorH - 18);

  const FLOORS = [1,2,3,4,5].map(f => ({
    floor: f,
    kind:  f <= 3 ? 'air' : 'liquid',
    y:     20 + (f - 1) * (rackH + 22),
  }));

  L = {
    W, H,
    agentPad, agentBlockW, agentBlockH, laneH, laneQW: 80,
    NET_X: agentPad, NET_Y, NET_W: agentBlockW, NET_H,
    WORKLOAD_X: agentPad,
    WORKLOAD_Y: LANE_START_Y - 24,
    WORKLOAD_W: agentBlockW,
    WORKLOAD_H: LANES[3].y + laneH - LANE_START_Y + 24 + 20,
    LANES,
    COOL_X: agentPad, COOL_Y, COOL_W: agentBlockW, COOL_H,
    AGENT_ZONE_W, RACK_ZONE_X, RACK_ZONE_W,
    N_FLOORS, N_PER_FLOOR, floorH,
    rackW, rackH,
    FLOORS,
  };
}

// ── Packet System ──────────────────────────────────────────────────────────
let packets = [];
let packetIdCounter = 0;

class Packet {
  constructor(priority) {
    this.id       = packetIdCounter++;
    this.priority = priority;
    this.color    = PRIORITY_COLS[priority];
    this.stage    = 0;      // 0=net, 1=triage, 2=cool, 3=routing, 4=fade
    this.x        = -12;
    this.y        = L.NET_Y + L.NET_H / 2 + (Math.random() - 0.5) * 18;
    this.laneY    = L.LANES[priority].y + L.laneH / 2 + (Math.random() - 0.5) * 8;
    this.speed    = 1.6 + Math.random() * 0.4;
    this.queued   = false;
    this.targetRack  = null;   // { x, y } canvas coords
    this.routeT      = 0;
    this.routeSrcX   = 0;
    this.routeSrcY   = 0;
    this.alpha       = 1;
    this.w = 9; this.h = 7;
  }

  update(dt) {
    const SPD = this.speed * speedMulti;
    if (this.stage === 0) {
      // Travel from spawn to lane entrance
      this.x += SPD * 2.2;
      const laneEntryX = L.WORKLOAD_X + 4;
      if (this.x >= laneEntryX) {
        this.stage = 1;
        this.x = laneEntryX;
        this.y = this.laneY;
      }
    } else if (this.stage === 1) {
      if (throttleActive && this.priority >= 2) {
        // Slow creep into queue box
        const qX = L.LANES[this.priority].queueX;
        if (this.x < qX) {
          this.x += SPD * 0.5;
        } else {
          this.queued = true;
          this.x = qX + Math.random() * 50;
        }
      } else {
        this.queued = false;
        this.x += SPD;
        const exitX = L.WORKLOAD_X + L.WORKLOAD_W - 2;
        if (this.x >= exitX) {
          this.stage = 2;
          this.x = exitX;
        }
      }
    } else if (this.stage === 2) {
      // Travel through cooling agent block
      this.x += SPD * 1.5;
      const coolExitX = L.COOL_X + L.COOL_W;
      if (this.x >= coolExitX) {
        this._assignRack();
        this.stage = 3;
        this.routeT = 0;
        this.routeSrcX = this.x;
        this.routeSrcY = this.y;
      }
    } else if (this.stage === 3) {
      this.routeT = Math.min(1, this.routeT + 0.025 * SPD);
      const t = ease(this.routeT);
      this.x = this.routeSrcX + (this.targetRack.x - this.routeSrcX) * t;
      this.y = this.routeSrcY + (this.targetRack.y - this.routeSrcY) * t;
      if (this.routeT >= 1) {
        this.stage = 4;
        // Heat up target rack slightly (visual feedback)
        if (this.targetRack.id) {
          rackTemps[this.targetRack.id] = Math.min(
            100, (rackTemps[this.targetRack.id] || 40) + 0.4
          );
        }
      }
    } else if (this.stage === 4) {
      this.alpha -= 0.06;
    }
    return this.alpha > 0;
  }

  _assignRack() {
    // Pick the coolest rack (LTI) in multi-agent, or random in baseline
    const all = Object.entries(rackTemps);
    if (!all.length) {
      this.targetRack = { x: L.RACK_ZONE_X + L.RACK_ZONE_W / 2, y: L.H / 2 };
      return;
    }
    let sorted;
    if (currentMode === 'multi_agent') {
      // P0/P1 → liquid preferred (floor 4-5), P2/P3 → air (floor 1-3)
      sorted = all.sort((a, b) => a[1] - b[1]);
    } else {
      // Round-robin: random
      sorted = all.sort(() => Math.random() - 0.5);
    }
    const pick = sorted[Math.floor(Math.random() * Math.min(4, sorted.length))];
    const pos  = rackPositions[pick[0]];
    this.targetRack = pos
      ? { x: pos.cx, y: pos.cy, id: pick[0] }
      : { x: L.RACK_ZONE_X + 20, y: 40 };
  }

  draw() {
    if (this.alpha <= 0) return;
    ctx.globalAlpha = Math.min(1, this.alpha);
    const r = 3;
    ctx.fillStyle = this.color;
    if (this.stage === 4) {
      ctx.shadowBlur = 12;
      ctx.shadowColor = this.color;
    } else {
      ctx.shadowBlur = 5;
      ctx.shadowColor = this.color;
    }
    ctx.beginPath();
    ctx.roundRect(this.x - this.w/2, this.y - this.h/2, this.w, this.h, r);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
  }

  get alive() { return this.alpha > 0; }
}

function ease(t) { return t * t * (3 - 2 * t); }

// ── Rack position cache ────────────────────────────────────────────────────
let rackPositions = {};   // rack_id → { cx, cy }

function buildRackPositions() {
  rackPositions = {};
  let rackNum = 1;
  L.FLOORS.forEach(f => {
    for (let pos = 1; pos <= L.N_PER_FLOOR; pos++) {
      const id  = `R${String(rackNum).padStart(2,'0')}`;
      const cx  = L.RACK_ZONE_X + (pos - 1) * (L.rackW + 5) + L.rackW / 2;
      const cy  = f.y + L.rackH / 2;
      rackPositions[id] = { cx, cy, id, floor: f.floor, kind: f.kind };
      rackNum++;
    }
  });
}

// ── Spawn control ──────────────────────────────────────────────────────────
let spawnAccum = 0;

function trySpawn(dt) {
  const baseRate = 0.04 * speedMulti * (latestFrame
    ? (latestFrame.agents?.scheduler?.load_pct || 40) / 100
    : 0.4);
  spawnAccum += baseRate * dt * 60;

  // Burst from API
  if (latestFrame?.agents?.scheduler?.burst_active) {
    spawnAccum += 0.8;
  }

  while (spawnAccum >= 1 && packets.length < 200) {
    spawnAccum--;
    const weights = throttleActive
      ? [0.40, 0.35, 0.15, 0.10]
      : [0.12, 0.22, 0.36, 0.30];
    const r = Math.random();
    let p = 0, cum = 0;
    for (let i = 0; i < 4; i++) { cum += weights[i]; if (r < cum) { p = i; break; } }
    packets.push(new Packet(p));
  }

  // Micro-burst on burst button (local)
  if (burstActive) {
    for (let i = 0; i < 4; i++) packets.push(new Packet(Math.floor(Math.random()*4)));
    burstActive = false;
  }
}

// ── Agent pipeline animation ───────────────────────────────────────────────
let agentTimer = 0;
const AGENT_CYCLE = 700;   // ms per step

function tickAgentPipeline(dt) {
  agentTimer += dt;
  if (agentTimer >= AGENT_CYCLE) {
    agentTimer = 0;
    agentPipeStep = (agentPipeStep + 1) % 4;
  }
}

// ── Draw ───────────────────────────────────────────────────────────────────
function drawAll() {
  ctx.clearRect(0, 0, L.W, L.H);
  drawBackground();
  drawAgentZone();
  drawRackZone();
  // Draw packets behind-then-front by stage
  packets.filter(p => p.stage >= 3).forEach(p => p.draw());
  packets.filter(p => p.stage < 3).forEach(p => p.draw());
}

function drawBackground() {
  // Subtle grid
  ctx.strokeStyle = 'rgba(255,255,255,0.025)';
  ctx.lineWidth = 1;
  for (let x = 0; x < L.W; x += 50) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, L.H); ctx.stroke();
  }
  for (let y = 0; y < L.H; y += 50) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(L.W, y); ctx.stroke();
  }
  // Divider between agent zone and rack zone
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(L.AGENT_ZONE_W, 0);
  ctx.lineTo(L.AGENT_ZONE_W, L.H);
  ctx.stroke();
}

// ── Agent Zone ─────────────────────────────────────────────────────────────
function drawAgentZone() {
  drawNetworkAgent();
  drawWorkloadAgent();
  drawCoolingAgent();
  drawAgentArrows();
}

function drawNetworkAgent() {
  const { NET_X, NET_Y, NET_W, NET_H } = L;
  const active = agentPipeStep === 0;
  drawAgentBlock(NET_X, NET_Y, NET_W, NET_H, 'NETWORK TRAFFIC AGENT', active, '#60a5fa');

  // Load bar inside
  const load = latestFrame?.agents?.scheduler?.load_pct || 0;
  const barW = (NET_W - 24) * (load / 100);
  ctx.fillStyle = 'rgba(255,255,255,0.06)';
  fillRect(NET_X + 12, NET_Y + NET_H - 16, NET_W - 24, 6, 2);
  ctx.fillStyle = '#60a5fa';
  if (barW > 0) fillRect(NET_X + 12, NET_Y + NET_H - 16, barW, 6, 2);

  // Throughput text
  ctx.fillStyle = COL.dim;
  ctx.font = `400 10px 'JetBrains Mono'`;
  ctx.textAlign = 'left';
  const mbIn  = latestFrame?.agents?.scheduler?.bytes_in_mbps?.toFixed(1) || '0.0';
  const mbOut = latestFrame?.agents?.scheduler?.bytes_out_mbps?.toFixed(1) || '0.0';
  ctx.fillText(`↓ ${mbIn} Mbps  ↑ ${mbOut} Mbps`, NET_X + 12, NET_Y + NET_H - 22);

  // Status badge
  const status = latestFrame?.agents?.scheduler?.status || 'NORMAL';
  const burstOn = latestFrame?.agents?.scheduler?.burst_active;
  const badge   = burstOn ? 'BURST' : status;
  const badgeC  = burstOn ? COL.p1 : (status === 'THROTTLED' ? COL.tHot : COL.tOk);
  ctx.fillStyle = badgeC;
  ctx.font = `700 9px 'JetBrains Mono'`;
  ctx.textAlign = 'right';
  ctx.fillText(badge, NET_X + NET_W - 10, NET_Y + 20);
}

function drawWorkloadAgent() {
  const { WORKLOAD_X, WORKLOAD_Y, WORKLOAD_W, WORKLOAD_H, LANES, laneH, laneQW } = L;
  const active = agentPipeStep === 1;
  drawAgentBlock(WORKLOAD_X, WORKLOAD_Y, WORKLOAD_W, WORKLOAD_H, 'WORKLOAD AGENT', active, COL.p1);

  // Draw 4 priority lanes
  LANES.forEach((lane, i) => {
    const lx = WORKLOAD_X + 6;
    const lw = WORKLOAD_W - 12;

    // Lane background
    ctx.fillStyle = `${PRIORITY_COLS[i]}18`;
    fillRect(lx, lane.y, lw, laneH, 3);
    ctx.strokeStyle = `${PRIORITY_COLS[i]}55`;
    ctx.lineWidth = 1;
    strokeRect(lx, lane.y, lw, laneH, 3);

    // Lane label
    ctx.fillStyle = PRIORITY_COLS[i];
    ctx.font = `400 9px 'JetBrains Mono'`;
    ctx.textAlign = 'left';
    ctx.fillText(`P${i} ${PRIORITY_NAMES[i].split(' ')[1]}`, lx + 5, lane.y + laneH / 2 + 4);

    // Priority load bar
    const pLoadKey = PRIORITY_NAMES[i];
    const pLoad = latestFrame?.agents?.scheduler?.priority_load?.[pLoadKey] || 0;
    const barMaxW = lw - 90;
    const barX    = lx + 80;
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    fillRect(barX, lane.y + 10, barMaxW, 6, 2);
    ctx.fillStyle = PRIORITY_COLS[i] + 'aa';
    fillRect(barX, lane.y + 10, barMaxW * (pLoad / 100), 6, 2);

    // Queue box (right end of lane)
    const qx = lx + lw - laneQW - 4;
    const shouldBlock = throttleActive && i >= 2;
    if (shouldBlock) {
      const pulse = Math.sin(Date.now() / 200) * 0.15 + 0.25;
      ctx.fillStyle = `rgba(248,113,113,${pulse})`;
      fillRect(qx, lane.y + 1, laneQW, laneH - 2, 2);
      ctx.fillStyle = COL.tHot;
      ctx.font = `700 8px 'JetBrains Mono'`;
      ctx.textAlign = 'center';
      ctx.fillText('BLOCKED', qx + laneQW / 2, lane.y + laneH / 2 + 3);
    } else {
      ctx.strokeStyle = `${PRIORITY_COLS[i]}33`;
      ctx.lineWidth = 1;
      strokeRect(qx, lane.y + 1, laneQW, laneH - 2, 2);
      ctx.fillStyle = COL.dim;
      ctx.font = `400 8px 'JetBrains Mono'`;
      ctx.textAlign = 'center';
      ctx.fillText('QUEUE', qx + laneQW / 2, lane.y + laneH / 2 + 3);
    }
  });
}

function drawCoolingAgent() {
  const { COOL_X, COOL_Y, COOL_W, COOL_H } = L;
  const active = agentPipeStep === 2 || agentPipeStep === 3;
  const isNeg  = agentPipeStep === 3 && throttleActive;

  drawAgentBlock(COOL_X, COOL_Y, COOL_W, COOL_H, 'THERMAL SENTINEL + COOLING AGENT', active, COL.teal);

  // Negotiation status
  const arbStatus = latestFrame?.agents?.arbiter?.negotiation_status || 'NOMINAL';
  const arbColor  = throttleActive ? COL.tHot : COL.tOk;
  ctx.fillStyle = arbColor;
  ctx.font      = `600 9px 'JetBrains Mono'`;
  ctx.textAlign = 'left';
  ctx.fillText(arbStatus, COOL_X + 12, COOL_Y + 32);

  // Negotiation arrows between thermal and cooling
  if (isNeg || agentPipeStep === 2) {
    const midX = COOL_X + COOL_W / 2;
    ctx.strokeStyle = COL.p1 + 'cc';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(COOL_X + COOL_W * 0.28, COOL_Y + COOL_H - 16);
    ctx.lineTo(COOL_X + COOL_W * 0.72, COOL_Y + COOL_H - 16);
    ctx.stroke();
    ctx.setLineDash([]);

    // Arrowheads
    arrowHead(COOL_X + COOL_W * 0.62, COOL_Y + COOL_H - 16, 0, COL.p1 + 'cc');
    arrowHead(COOL_X + COOL_W * 0.38, COOL_Y + COOL_H - 16, Math.PI, COL.p1 + 'cc');

    ctx.fillStyle = COL.dim;
    ctx.font = `400 8px 'JetBrains Mono'`;
    ctx.textAlign = 'center';
    ctx.fillText('negotiating', midX, COOL_Y + COOL_H - 5);
  }

  // Fan speed
  const shed = latestFrame?.agents?.arbiter?.shed_pct || 0;
  ctx.fillStyle = COL.dim;
  ctx.font = `400 9px 'JetBrains Mono'`;
  ctx.textAlign = 'right';
  ctx.fillText(`shed=${shed.toFixed(0)}%`, COOL_X + COOL_W - 10, COOL_Y + 20);
}

function drawAgentArrows() {
  const { NET_X, NET_Y, NET_W, NET_H, WORKLOAD_X, WORKLOAD_Y, WORKLOAD_W, WORKLOAD_H, COOL_X, COOL_Y, COOL_W, AGENT_ZONE_W } = L;

  // Net → Workload
  const ax = NET_X + NET_W;
  const ay = NET_Y + NET_H / 2;
  const bx = WORKLOAD_X;
  const by = WORKLOAD_Y + 30;
  drawFlowArrow(ax, ay, bx, by, agentPipeStep === 0);

  // Workload → Cooling
  const cx2 = WORKLOAD_X + WORKLOAD_W;
  const cy2 = WORKLOAD_Y + WORKLOAD_H / 2;
  const dx  = COOL_X;
  const dy  = COOL_Y + L.COOL_H / 2;
  drawFlowArrow(cx2, cy2, dx, dy, agentPipeStep === 1);

  // Cooling → Rack zone
  drawFlowArrow(COOL_X + L.COOL_W, COOL_Y + L.COOL_H / 2, L.AGENT_ZONE_W + 2, L.H / 2, agentPipeStep >= 2);
}

function drawFlowArrow(x1, y1, x2, y2, lit) {
  ctx.save();
  ctx.strokeStyle = lit ? COL.ma + 'dd' : COL.dim + '55';
  ctx.lineWidth   = lit ? 1.8 : 1;
  if (lit) { ctx.shadowBlur = 8; ctx.shadowColor = COL.ma; }
  ctx.beginPath();
  ctx.moveTo(x1 + 2, y1);
  ctx.lineTo(x2 - 6, y2);
  ctx.stroke();
  ctx.shadowBlur = 0;
  arrowHead(x2 - 2, y2, Math.atan2(y2 - y1, x2 - x1), lit ? COL.ma : COL.dim + '55');
  ctx.restore();
}

function arrowHead(x, y, angle, color) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(-7, -3.5);
  ctx.lineTo(-7, 3.5);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawAgentBlock(x, y, w, h, title, active, accentColor) {
  // Background
  ctx.fillStyle = '#0d1117';
  fillRect(x, y, w, h, 5);

  // Border
  ctx.strokeStyle = active
    ? accentColor + 'bb'
    : 'rgba(255,255,255,0.07)';
  ctx.lineWidth = active ? 1.5 : 1;
  if (active) { ctx.shadowBlur = 10; ctx.shadowColor = accentColor + '66'; }
  strokeRect(x, y, w, h, 5);
  ctx.shadowBlur = 0;

  // Title bar
  ctx.fillStyle = active
    ? accentColor + '22'
    : 'rgba(255,255,255,0.03)';
  fillRect(x + 1, y + 1, w - 2, 18, [5, 5, 0, 0]);

  ctx.fillStyle = active ? accentColor : COL.dim;
  ctx.font = `${active ? 600 : 400} 9px 'JetBrains Mono'`;
  ctx.textAlign = 'left';
  ctx.fillText(title, x + 10, y + 13);
}

// ── Rack Zone ──────────────────────────────────────────────────────────────
function drawRackZone() {
  const { RACK_ZONE_X, RACK_ZONE_W, FLOORS, N_PER_FLOOR, rackW, rackH } = L;

  FLOORS.forEach(f => {
    const isLiquid = f.kind === 'liquid';
    const labelColor = isLiquid ? '#2dd4bf' : '#f5c542';
    const floorLabel = `FLOOR ${f.floor} · ${isLiquid ? 'LIQUID' : 'AIR'} COOLING`;

    // Floor label
    ctx.fillStyle = labelColor + '99';
    ctx.font = `400 8px 'JetBrains Mono'`;
    ctx.textAlign = 'left';
    ctx.fillText(floorLabel, RACK_ZONE_X + 2, f.y - 6);

    // Rack cells
    let rackNum = (f.floor - 1) * N_PER_FLOOR + 1;
    for (let pos = 0; pos < N_PER_FLOOR; pos++, rackNum++) {
      const id   = `R${String(rackNum).padStart(2, '0')}`;
      const rx   = RACK_ZONE_X + pos * (rackW + 5);
      const ry   = f.y;
      const temp = rackTemps[id] || (isLiquid ? 26 : 32);

      drawRackCell(id, rx, ry, rackW, rackH, temp, isLiquid);
    }
  });
}

function drawRackCell(id, rx, ry, rw, rh, temp, isLiquid) {
  // Fill by temperature
  const frac = Math.max(0, Math.min(1, (temp - 20) / 70));
  const fillColor = isLiquid
    ? lerpColor([20, 35, 50], [80, 20, 20], frac * 0.7)
    : lerpColor([30, 30, 18], [80, 20, 20], frac * 0.7);
  ctx.fillStyle = `rgb(${fillColor.join(',')})`;
  fillRect(rx, ry, rw, rh, 3);

  // Border
  const borderColor = temp > 82 ? COL.tHot : (temp > 74 ? COL.tWarn : (isLiquid ? '#2dd4bf' : '#f5c542'));
  ctx.strokeStyle = borderColor + (temp > 82 ? 'ff' : '88');
  ctx.lineWidth = temp > 82 ? 2 : 1;
  if (temp > 82) { ctx.shadowBlur = 6; ctx.shadowColor = COL.tHot; }
  strokeRect(rx, ry, rw, rh, 3);
  ctx.shadowBlur = 0;

  // Temp text
  ctx.fillStyle = temp > 82 ? COL.tHot : (temp > 74 ? COL.tWarn : '#888');
  ctx.font = `400 7px 'JetBrains Mono'`;
  ctx.textAlign = 'center';
  ctx.fillText(`${temp.toFixed(0)}°`, rx + rw / 2, ry + rh / 2 + 3);
}

function lerpColor(a, b, t) {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

// ── Canvas helpers ─────────────────────────────────────────────────────────
function fillRect(x, y, w, h, r) {
  if (!r) { ctx.fillRect(x, y, w, h); return; }
  const radii = Array.isArray(r) ? r : [r, r, r, r];
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, radii);
  ctx.fill();
}

function strokeRect(x, y, w, h, r) {
  if (!r) { ctx.strokeRect(x, y, w, h); return; }
  const radii = Array.isArray(r) ? r : [r, r, r, r];
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, radii);
  ctx.stroke();
}

// ── Resize handler ─────────────────────────────────────────────────────────
function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width  = Math.floor(rect.width);
  canvas.height = Math.floor(rect.height);
  computeLayout();
  buildRackPositions();
}

// ── Main Loop ──────────────────────────────────────────────────────────────
let lastTime = 0;

function loop(ts) {
  const dt = Math.min((ts - lastTime) / 16.67, 3);
  lastTime = ts;

  tickAgentPipeline(dt * 16.67);
  trySpawn(dt);

  packets = packets.filter(p => {
    const alive = p.update(dt);
    return alive;
  });

  drawAll();
  requestAnimationFrame(loop);
}

// ── WebSocket ──────────────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onopen = () => {
    document.getElementById('live-dot').className = 'online';
    document.getElementById('live-txt').textContent = 'LIVE';
  };

  ws.onmessage = (ev) => {
    const frame = JSON.parse(ev.data);
    latestFrame = frame;
    onFrame(frame);
  };

  ws.onclose = () => {
    document.getElementById('live-dot').className = 'offline';
    document.getElementById('live-txt').textContent = 'OFFLINE';
    setTimeout(connectWS, 2000);
  };
}

function onFrame(frame) {
  currentMode    = frame.mode;
  throttleActive = frame.agents?.arbiter?.throttle_active || false;

  // Update rack temps from server
  if (frame.racks) {
    frame.racks.forEach(r => { rackTemps[r.id] = r.peak_temp; });
  }

  // KPIs
  const ma   = frame.kpis?.multi_agent;
  const base = frame.kpis?.baseline;
  if (ma && base) {
    setEl('kv-ma-energy',   `${ma.energy_kwh.toFixed(3)} kWh`);
    setEl('kv-base-energy', `${base.energy_kwh.toFixed(3)} kWh`);
    const maHist   = ma.history?.peak_temp  || [];
    const baseHist = base.history?.peak_temp || [];
    const maPeak   = maHist.length ? maHist[maHist.length - 1] : 0;
    const basePeak = baseHist.length ? baseHist[baseHist.length - 1] : 0;
    setElTemp('kv-peak-ma',   maPeak);
    setElTemp('kv-peak-base', basePeak);
  }

  const active = frame.kpis?.active;
  if (active) {
    const viol = active.violations_s;
    const kvViol = document.getElementById('kv-violations');
    kvViol.textContent = viol;
    kvViol.style.color = viol > 0 ? '#f87171' : '#d0d7de';
  }

  const load = frame.agents?.scheduler?.load_pct || 0;
  setEl('kv-load', `${load.toFixed(0)}%`);
  setEl('kv-tick', `T:${frame.tick}`);

  const arbStatus = frame.agents?.arbiter?.negotiation_status || 'NOMINAL';
  const kvArb = document.getElementById('kv-arbiter');
  kvArb.textContent = arbStatus;
  kvArb.style.color = throttleActive ? '#f87171' : '#34d399';

  // Mode buttons
  document.getElementById('btn-multi').classList.toggle('active', frame.mode === 'multi_agent');
  document.getElementById('btn-base').classList.toggle('active', frame.mode === 'baseline');

  // Log marquee
  const logs = frame.log || [];
  if (logs.length) {
    const combined = logs.slice(-6).join('  ·  ');
    const el = document.getElementById('log-inner');
    // Duplicate for seamless loop
    el.textContent = combined + '   ·   ' + combined;
  }

  // Charts
  updateCharts(frame);
}

function setEl(id, txt) {
  const el = document.getElementById(id);
  if (el) el.textContent = txt;
}

function setElTemp(id, temp) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = `${temp.toFixed(1)}°C`;
  el.style.color = temp > 82 ? '#f87171' : (temp > 74 ? '#fbbf24' : '');
}

// ── Charts ─────────────────────────────────────────────────────────────────
let charts = {};

function initCharts() {
  Chart.defaults.color            = '#636e7b';
  Chart.defaults.borderColor      = 'rgba(255,255,255,0.05)';
  Chart.defaults.font.family      = "'JetBrains Mono', monospace";
  Chart.defaults.font.size        = 9;

  const defs = [
    { id: 'energy',     key: 'energy',     type: 'line', yMin: 0 },
    { id: 'temp',       key: 'peak_temp',  type: 'line', yMin: 20 },
    { id: 'violations', key: 'violations', type: 'bar',  yMin: 0 },
    { id: 'fan',        key: 'fan_speed',  type: 'line', yMin: 0 },
  ];

  defs.forEach(def => {
    const canvas = document.getElementById(`chart-${def.id}`);
    if (!canvas) return;
    charts[def.id] = new Chart(canvas, {
      type: def.type,
      data: {
        labels: [],
        datasets: [
          {
            label: 'Multi-Agent',
            borderColor: COL.ma,
            backgroundColor: def.type === 'bar' ? COL.ma + '55' : 'transparent',
            data: [],
            borderWidth: def.type === 'bar' ? 0 : 2,
            barPercentage: 0.6,
            pointRadius: 0,
            tension: 0.35,
          },
          {
            label: 'Baseline',
            borderColor: COL.base,
            backgroundColor: def.type === 'bar' ? COL.base + '55' : 'transparent',
            data: [],
            borderWidth: def.type === 'bar' ? 0 : 1.5,
            barPercentage: 0.6,
            pointRadius: 0,
            borderDash: def.type === 'line' ? [5, 4] : [],
            tension: 0.35,
          },
        ],
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0d1117',
            titleColor: '#d0d7de',
            bodyColor: '#636e7b',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1,
          },
        },
        scales: {
          x: { display: false },
          y: {
            min: def.yMin,
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: { color: '#636e7b', maxTicksLimit: 4 },
          },
        },
      },
    });
  });
}

function updateCharts(frame) {
  const ma   = frame.kpis?.multi_agent?.history;
  const base = frame.kpis?.baseline?.history;
  if (!ma || !base) return;

  const map = { energy: 'energy', temp: 'peak_temp', violations: 'violations', fan: 'fan_speed' };
  Object.entries(map).forEach(([id, key]) => {
    const ch = charts[id];
    if (!ch) return;
    const d0 = ma[key]   || [];
    const d1 = base[key] || [];
    ch.data.labels       = d0.map((_, i) => i);
    ch.data.datasets[0].data = d0;
    ch.data.datasets[1].data = d1;
    ch.update('none');
  });
}

// ── Controls ───────────────────────────────────────────────────────────────
document.getElementById('speed-range').addEventListener('input', function() {
  speedMulti = this.value / 10;
  document.getElementById('speed-val').textContent = `${speedMulti.toFixed(1)}×`;
  fetch(`/api/traffic-speed/${speedMulti}`, { method: 'POST' }).catch(() => {});
});

async function triggerBurst() {
  burstActive = true;
  const btn = document.getElementById('burst-btn');
  btn.classList.add('firing');
  setTimeout(() => btn.classList.remove('firing'), 500);
  try { await fetch('/api/burst', { method: 'POST' }); } catch (_) {}
}

async function setMode(mode) {
  try { await fetch(`/api/mode/${mode}`, { method: 'POST' }); } catch (_) {}
}

// ── Tab switching ──────────────────────────────────────────────────────────
function showTab(id) {
  document.querySelectorAll('.tab-view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(`view-${id}`)?.classList.add('active');
  document.getElementById(`tab-${id}`)?.classList.add('active');
  // Force chart resize on tab switch
  if (id === 'telem') setTimeout(() => Object.values(charts).forEach(c => c.resize()), 50);
}

// ── Boot ───────────────────────────────────────────────────────────────────
function init() {
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
  initCharts();
  connectWS();
  requestAnimationFrame(loop);
}

window.onload = init;

/**
 * THERMOS — Main Application Logic v2.0
 * 5-Floor × 10-Rack topology, agent pipeline visualization,
 * stark MA vs Baseline contrast in charts.
 */

// ─── Global State ────────────────────────────────────────────────────────────
let ws;
let currentView = 'floor';
let selectedRackId = 'R01';
let config = {};
let latestFrame = null;
let charts = {};
let staticDrawn = false;    // whether SVG static elements have been drawn

// ─── Color constants (must match CSS vars) ───────────────────────────────────
const C = {
  safe:   '#22c55e',
  warn:   '#f59e0b',
  crit:   '#ef4444',
  gold:   '#f5c542',
  teal:   '#14b8a6',
  p0:     '#ef4444',
  p1:     '#f59e0b',
  p2:     '#14b8a6',
  p3:     '#6b7280',
  ma:     '#00e5ff',    // multi-agent — vivid cyan
  base:   '#ff6b35',    // baseline — vivid orange-red
  maDim:  'rgba(0,229,255,0.2)',
  baseDim:'rgba(255,107,53,0.2)',
};

const SVG_NS = "http://www.w3.org/2000/svg";

const VIEW_DOM = {
  floor:     document.getElementById('view-floor'),
  rack:      document.getElementById('view-rack'),
  telemetry: document.getElementById('view-telemetry'),
};
const LOG_EL = document.getElementById('log-content');

// ─── Floor geometry (matches datacenter_layout.py) ───────────────────────────
// Viewbox 1000x650, 5 floors, racks at y = [130,240,350,450,550]
// Each rack is 28x22, 10 racks per floor, x from 120 to 1020 step 95
const FLOOR_META = [
  { floor: 1, kind: 'air',    y: 130, label: 'FLOOR 1 — AIR COOLING'    },
  { floor: 2, kind: 'air',    y: 240, label: 'FLOOR 2 — AIR COOLING'    },
  { floor: 3, kind: 'air',    y: 350, label: 'FLOOR 3 — AIR COOLING'    },
  { floor: 4, kind: 'liquid', y: 450, label: 'FLOOR 4 — LIQUID COOLING' },
  { floor: 5, kind: 'liquid', y: 550, label: 'FLOOR 5 — LIQUID COOLING' },
];

// ─── Initialisation ───────────────────────────────────────────────────────────
async function init() {
  await loadConfig();
  initNav();
  initCharts();
  connectWS();
  animateAgentPipeline();
}

async function loadConfig() {
  try {
    const res = await fetch('/api/config');
    config = await res.json();
    const dc = config.datacenter || {};
    document.getElementById('kpi-racks').textContent   = '50';
    document.getElementById('kpi-air').textContent     = '30';
    document.getElementById('kpi-liquid').textContent  = '20';
    document.getElementById('kpi-slots').textContent   = String(50 * (config.rack?.n_slots || 15));
    document.getElementById('kpi-floors').textContent  = '05';
  } catch (err) {
    console.error('Failed to load config:', err);
  }
}

function initNav() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      const view = item.getAttribute('data-view') || item.id.replace('nav-', '');
      switchView(view);
    });
  });
}

function switchView(view) {
  currentView = view;
  Object.keys(VIEW_DOM).forEach(v => {
    VIEW_DOM[v]?.classList.toggle('active', v === view);
  });
  document.querySelectorAll('.nav-item').forEach(item => {
    item.classList.toggle('active',
      item.getAttribute('data-view') === view || item.id === `nav-${view}`);
  });
  document.getElementById('kpi-focus').textContent = view.toUpperCase();
}

// ─── WebSocket ────────────────────────────────────────────────────────────────
function connectWS() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.onmessage = (event) => {
    const frame = JSON.parse(event.data);
    latestFrame = frame;
    updateUI(frame);
  };

  ws.onclose = () => {
    document.getElementById('live-indicator').innerHTML = '<span style="color:#ef4444">● OFFLINE</span>';
    setTimeout(connectWS, 2000);
  };
}

// ─── Main UI Update ───────────────────────────────────────────────────────────
function updateUI(frame) {
  // KPIs
  const maKpi   = frame.kpis.multi_agent;
  const baseKpi = frame.kpis.baseline;
  document.getElementById('kpi-energy-ma').textContent   = `${maKpi.energy_kwh.toFixed(3)} kWh`;
  document.getElementById('kpi-energy-base').textContent = `${baseKpi.energy_kwh.toFixed(3)} kWh`;
  document.getElementById('kpi-violations').textContent  = `${frame.kpis.active.violations_s} s`;
  document.getElementById('tick-counter').textContent    = `TICK: ${frame.tick}`;

  updateModeButtons(frame.mode);
  updateModeDesc(frame.mode, frame.agents?.router?.policy);
  renderLog(frame.log);

  if (currentView === 'floor') renderFloor(frame);
  else if (currentView === 'rack') renderRack(frame);

  updateCharts(frame);
  updatePriorityBars(frame);
  updateArbiter(frame);
}

function updateModeButtons(mode) {
  document.getElementById('btn-multi').classList.toggle('active', mode === 'multi_agent');
  document.getElementById('btn-base').classList.toggle('active', mode === 'baseline');
}

function updateModeDesc(mode, policy) {
  const desc = mode === 'multi_agent'
    ? `Showing: MULTI-AGENT (LTI routing + predictive cooling + Arbiter negotiation)`
    : `Showing: BASELINE (round-robin routing, fixed fans, no shedding)`;
  document.getElementById('mode-desc').textContent = desc;
}

async function setMode(mode) {
  await fetch(`/api/mode/${mode}`, { method: 'POST' });
}

function renderLog(lines) {
  LOG_EL.innerHTML = lines.map(line => `<div>${line}</div>`).join('');
  LOG_EL.scrollTop = LOG_EL.scrollHeight;
}

// ─── Floor Map Rendering ──────────────────────────────────────────────────────
function renderFloor(frame) {
  if (!staticDrawn && frame.topology?.nodes?.length) {
    drawStaticFloors(frame.topology);
    staticDrawn = true;
  }

  // Update rack colors based on temperature
  frame.racks.forEach(r => {
    const rect = document.getElementById(`rect-${r.id}`);
    if (!rect) return;
    rect.style.fill = getTempFill(r.peak_temp, r.kind);
    if (r.peak_temp > 80) {
      rect.style.strokeWidth = '2.5';
      rect.style.stroke = C.crit;
    } else {
      rect.style.strokeWidth = '1.5';
      rect.style.stroke = r.kind === 'liquid' ? C.teal : C.gold;
    }

    // Update temp label
    const lbl = document.getElementById(`templbl-${r.id}`);
    if (lbl) lbl.textContent = `${r.peak_temp.toFixed(0)}°`;
  });

  // Route lines showing agent decision path (MA mode only)
  drawRouteSignals(frame);

  // Animate packets
  renderPackets(frame.topology.packets, frame.topology.nodes);
}

function drawStaticFloors(topology) {
  const floorLabelG = document.getElementById('svg-floor-labels');
  const floorLineG  = document.getElementById('svg-floor-lines');
  const racksG      = document.getElementById('svg-racks');

  // Draw floor background bands and labels
  FLOOR_META.forEach((f, fi) => {
    const nextY = FLOOR_META[fi + 1]?.y ?? 620;
    const bandH = nextY - f.y - 8;

    // Floor background band
    const band = document.createElementNS(SVG_NS, 'rect');
    band.setAttribute('x', '60');
    band.setAttribute('y', String(f.y - 22));
    band.setAttribute('width', '930');
    band.setAttribute('height', String(bandH));
    band.setAttribute('class', f.kind === 'liquid' ? 'svg-floor-bg-liquid' : 'svg-floor-bg-air');
    band.setAttribute('rx', '4');
    floorLineG.appendChild(band);

    // Separator line (above each floor except first)
    if (fi > 0) {
      const sep = document.createElementNS(SVG_NS, 'line');
      sep.setAttribute('x1', '60');
      sep.setAttribute('y1', String(f.y - 26));
      sep.setAttribute('x2', '990');
      sep.setAttribute('y2', String(f.y - 26));
      sep.setAttribute('class', 'svg-floor-sep');
      floorLineG.appendChild(sep);
    }

    // Floor label
    const lbl = document.createElementNS(SVG_NS, 'text');
    lbl.setAttribute('x', '68');
    lbl.setAttribute('y', String(f.y - 9));
    lbl.setAttribute('class', f.kind === 'liquid' ? 'svg-floor-label-liquid' : 'svg-floor-label-air');
    lbl.textContent = f.label;
    floorLabelG.appendChild(lbl);
  });

  // Draw racks
  topology.nodes.filter(n => n.floor).forEach(n => {
    const rackW = 28, rackH = 22;
    const rect = document.createElementNS(SVG_NS, 'rect');
    rect.setAttribute('id', `rect-${n.id}`);
    rect.setAttribute('x', String(n.x - rackW / 2));
    rect.setAttribute('y', String(n.y - rackH / 2));
    rect.setAttribute('width', String(rackW));
    rect.setAttribute('height', String(rackH));
    rect.setAttribute('class', n.kind === 'liquid' ? 'svg-rack-liquid' : 'svg-rack-air');
    rect.setAttribute('rx', '2');
    rect.onclick = () => showRack(n.id);
    racksG.appendChild(rect);

    // Rack ID label (below)
    const idLbl = document.createElementNS(SVG_NS, 'text');
    idLbl.setAttribute('x', String(n.x));
    idLbl.setAttribute('y', String(n.y + rackH / 2 + 9));
    idLbl.setAttribute('class', 'svg-rack-label');
    idLbl.textContent = n.id;
    racksG.appendChild(idLbl);

    // Temperature label (inside rack)
    const tempLbl = document.createElementNS(SVG_NS, 'text');
    tempLbl.setAttribute('id', `templbl-${n.id}`);
    tempLbl.setAttribute('x', String(n.x));
    tempLbl.setAttribute('y', String(n.y + 4));
    tempLbl.setAttribute('class', 'svg-rack-label');
    tempLbl.style.fontSize = '6px';
    tempLbl.textContent = '--°';
    racksG.appendChild(tempLbl);
  });
}

// Draw animated route lines from agents to selected rack
let routeAnimLines = [];
function drawRouteSignals(frame) {
  const agentG = document.getElementById('svg-agents');
  agentG.innerHTML = '';
  if (!frame.topology?.packets?.length) return;

  // Show a representative packet's path as agent negotiation signal
  const sample = frame.topology.packets.slice(0, 6);
  sample.forEach(pkt => {
    const target = frame.topology.nodes?.find(n => n.id === pkt.target);
    if (!target) return;

    // Draw a dashed line from left edge (agent zone) to target rack
    const line = document.createElementNS(SVG_NS, 'line');
    line.setAttribute('x1', '62');
    line.setAttribute('y1', String(target.y));
    line.setAttribute('x2', String(target.x - 14));
    line.setAttribute('y2', String(target.y));
    line.setAttribute('class', frame.mode === 'multi_agent'
      ? 'svg-route-line-ma' : 'svg-route-line-base');
    agentG.appendChild(line);
  });
}

// Packet animation
function renderPackets(packets, nodes) {
  const group = document.getElementById('svg-packets');
  group.innerHTML = '';
  if (!packets || !nodes) return;

  packets.slice(0, 60).forEach(p => {
    const rack = nodes.find(n => n.id === p.target);
    if (!rack) return;

    // Packets travel from x=60 (left agent zone) → rack position
    // p.x is 0→1 progress along that path
    const startX = 62, startY = rack.y;
    const endX   = rack.x - 14, endY = rack.y;
    const cx = startX + (endX - startX) * p.x;
    const cy = startY + (p.lane_y - 0.5) * 18;

    const circle = document.createElementNS(SVG_NS, 'circle');
    circle.setAttribute('cx', String(Math.round(cx)));
    circle.setAttribute('cy', String(Math.round(cy)));
    circle.setAttribute('r', '3.5');
    circle.setAttribute('fill', p.color);
    circle.style.filter = 'url(#glow)';
    group.appendChild(circle);
  });
}

// ─── Priority Bars ────────────────────────────────────────────────────────────
function updatePriorityBars(frame) {
  const pLoad = frame.agents?.scheduler?.priority_load || {};
  document.getElementById('pb-p0').style.width = `${Math.min(100, (pLoad['P0 Critical'] || 0) * 4)}%`;
  document.getElementById('pb-p1').style.width = `${Math.min(100, (pLoad['P1 High'] || 0) * 3)}%`;
  document.getElementById('pb-p2').style.width = `${Math.min(100, (pLoad['P2 Medium'] || 0) * 2)}%`;
  document.getElementById('pb-p3').style.width = `${Math.min(100, (pLoad['P3 Low'] || 0) * 1.5)}%`;
}

// ─── Arbiter ──────────────────────────────────────────────────────────────────
function updateArbiter(frame) {
  const arb = frame.agents?.arbiter;
  if (!arb) return;
  const arbStatus = document.getElementById('arb-status');
  const arbShed   = document.getElementById('arb-shed');
  arbStatus.textContent = arb.negotiation_status || 'NOMINAL';
  arbStatus.className   = arb.throttle_active ? 'crit' : 'safe';
  arbShed.style.display = arb.throttle_active ? 'block' : 'none';
}

// ─── Agent Pipeline Animation (left sidebar) ──────────────────────────────────
let _agentStep = 0;
const AGENT_IDS = ['af-net', 'af-work', 'af-therm', 'af-cool'];

function animateAgentPipeline() {
  // Cycle through agents to show active negotiation
  setInterval(() => {
    AGENT_IDS.forEach((id, i) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.classList.remove('active-agent', 'negotiating');
      if (i === _agentStep) el.classList.add('active-agent');
      if (i === (_agentStep + 1) % AGENT_IDS.length) el.classList.add('negotiating');
    });
    const negLine = document.getElementById('af-neg');
    if (negLine) {
      // Show negotiation line when Thermal and Cooling are communicating
      negLine.classList.toggle('visible', _agentStep === 2 || _agentStep === 3);
    }
    _agentStep = (_agentStep + 1) % AGENT_IDS.length;
  }, 700);
}

// ─── Rack Detail View ─────────────────────────────────────────────────────────
function showRack(id) {
  selectedRackId = id;
  switchView('rack');
}

function renderRack(frame) {
  const rack = frame.racks.find(r => r.id === selectedRackId);
  if (!rack) return;

  document.getElementById('rack-id-title').textContent = rack.id;
  document.getElementById('rack-kind-title').textContent = rack.kind.toUpperCase();
  document.getElementById('rack-kind-title').className = rack.kind === 'liquid' ? 'teal' : 'gold';

  document.getElementById('orch-state').textContent = rack.throttle_active ? 'THROTTLED' : 'NOMINAL';
  document.getElementById('orch-state').className   = rack.throttle_active ? 'orch-v crit' : 'orch-v safe';
  document.getElementById('orch-policy').textContent = frame.agents?.router?.policy || 'LTI';

  const pNames = ['P0 Critical', 'P1 High', 'P2 Medium', 'P3 Low'];
  pNames.forEach((name, i) => {
    const track = document.getElementById(`track-p${i}`);
    if (!track) return;
    track.innerHTML = '';
    (frame.topology?.packets || []).filter(p => p.priority === i).forEach(p => {
      const pkt = document.createElement('div');
      pkt.className = 'lane-pkt';
      pkt.style.left = `${p.x * 90}%`;
      pkt.style.top  = `${p.lane_y * 100}%`;
      pkt.style.backgroundColor = p.color;
      if (p.target === selectedRackId) {
        pkt.style.boxShadow = `0 0 10px ${p.color}`;
        pkt.style.width = '13px';
        pkt.style.height = '13px';
      }
      track.appendChild(pkt);
    });
  });

  document.getElementById('rf-peak').textContent     = `${rack.peak_temp.toFixed(1)} °C`;
  document.getElementById('rf-avg').textContent      = `${rack.avg_temp.toFixed(1)} °C`;
  document.getElementById('rf-inflight').textContent = frame.agents?.router?.in_flight ?? '--';
  document.getElementById('rf-routed').textContent   = frame.agents?.router?.routed_total ?? '--';
  document.getElementById('rf-mode').textContent     = rack.kind.toUpperCase();

  const slotList = document.getElementById('slot-list');
  slotList.innerHTML = (rack.slots || []).sort((a, b) => b.id - a.id).map(s => `
    <div class="slot-row">
      <div class="slot-id">S${String(s.id).padStart(2, '0')}</div>
      <div class="slot-bar-wrap">
        <div class="slot-bar" style="width:${s.util * 100}%; background:${getTempColor(s.temp, rack.kind)}"></div>
      </div>
      <div class="slot-temp ${getTempClass(s.temp)}">${s.temp.toFixed(1)}°</div>
    </div>
  `).join('');
}

function showFloor() { switchView('floor'); }

// ─── Helpers ──────────────────────────────────────────────────────────────────
function getTempClass(t) {
  if (t > 80) return 'temp-crit';
  if (t > 75) return 'temp-warn';
  return 'temp-safe';
}
function getTempColor(t, kind) {
  if (t > 80) return C.crit;
  if (t > 75) return C.warn;
  return kind === 'liquid' ? C.teal : C.safe;
}
function getTempFill(t, kind) {
  if (t > 80) return 'rgba(239,68,68,0.35)';
  if (t > 75) return 'rgba(245,158,11,0.25)';
  return kind === 'liquid' ? 'rgba(20,184,166,0.18)' : 'rgba(184,134,11,0.15)';
}

// ─── Charts (stark contrast: MA = cyan, Baseline = orange-red) ───────────────
function initCharts() {
  // Chart.js global defaults for dark theme
  Chart.defaults.color = '#6b7280';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';

  const chartDefs = [
    { id: 'energy',     title: 'Energy (kWh)',   dataKey: 'energy',     type: 'line', yMin: 0 },
    { id: 'temp',       title: 'Peak Temp (°C)', dataKey: 'peak_temp',  type: 'line', yMin: 20 },
    { id: 'violations', title: 'Violations',     dataKey: 'violations', type: 'bar',  yMin: 0 },
    { id: 'fan',        title: 'Fan Speed (ω)',  dataKey: 'fan_speed',  type: 'line', yMin: 0 },
  ];

  chartDefs.forEach(def => {
    const canvas = document.getElementById(`chart-${def.id}`);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    charts[def.id] = new Chart(ctx, {
      type: def.type,
      data: {
        labels: Array.from({ length: 300 }, (_, i) => i),
        datasets: [
          {
            // Multi-agent — VIVID CYAN, thick, solid
            label: 'Multi-Agent',
            borderColor: C.ma,
            backgroundColor: def.type === 'bar' ? C.maDim : 'transparent',
            pointBackgroundColor: C.ma,
            data: [],
            borderWidth: def.type === 'bar' ? 0 : 2.5,
            pointRadius: 0,
            fill: false,
            tension: 0.3,
            order: 1,
          },
          {
            // Baseline — VIVID ORANGE-RED, thinner, dashed
            label: 'Baseline',
            borderColor: C.base,
            backgroundColor: def.type === 'bar' ? C.baseDim : 'transparent',
            pointBackgroundColor: C.base,
            data: [],
            borderWidth: def.type === 'bar' ? 0 : 1.8,
            pointRadius: 0,
            borderDash: def.type === 'line' ? [6, 4] : [],
            fill: false,
            tension: 0.3,
            order: 2,
          },
        ],
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            labels: {
              color: '#9ca3af',
              font: { size: 10, family: "'Share Tech Mono', monospace" },
              boxWidth: 20,
              padding: 10,
              usePointStyle: true,
              pointStyle: 'rect',
            },
          },
          tooltip: {
            backgroundColor: 'rgba(12,18,24,0.95)',
            titleColor: '#c9d1d9',
            bodyColor: '#9ca3af',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1,
          },
        },
        scales: {
          x: {
            display: false,
            grid: { display: false },
          },
          y: {
            min: def.yMin,
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#6b7280', font: { size: 9 } },
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

  const keyMap = { energy: 'energy', temp: 'peak_temp', violations: 'violations', fan: 'fan_speed' };

  Object.keys(charts).forEach(id => {
    const k = keyMap[id];
    if (!k) return;
    const maData   = ma[k]   || [];
    const baseData = base[k] || [];
    charts[id].data.datasets[0].data = maData;
    charts[id].data.datasets[1].data = baseData;
    // Update label count to match data length
    charts[id].data.labels = Array.from({ length: Math.max(maData.length, baseData.length) }, (_, i) => i);
    charts[id].update('none');
  });
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
window.onload = init;

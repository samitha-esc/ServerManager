/**
 * THERMOS — Main Application Logic
 * Handles WebSocket simulation data, SVG rendering, view toggling, and charts.
 */

// --- Global State ---
let ws;
let currentView = 'floor';
let selectedRackId = 'R07';
let config = {};
let latestFrame = null;
let charts = {};

// --- Constants & Config ---
const COLORS = {
  safe: '#22c55e',
  warn: '#f59e0b',
  crit: '#ef4444',
  gold: '#f5c542',
  teal: '#14b8a6',
  p0: '#ef4444',
  p1: '#f59e0b',
  p2: '#14b8a6',
  p3: '#6b7280'
};

const VIEW_DOM = {
  floor: document.getElementById('view-floor'),
  rack: document.getElementById('view-rack'),
  telemetry: document.getElementById('view-telemetry')
};

const LOG_EL = document.getElementById('log-content');

// --- Initialisation ---
async function init() {
  await loadConfig();
  initNav();
  initCharts();
  connectWS();
}

async function loadConfig() {
  try {
    const res = await fetch('/api/config');
    config = await res.json();
    document.getElementById('kpi-racks').textContent = config.datacenter.n_racks.toString().padStart(2, '0');
    document.getElementById('kpi-air').textContent = config.datacenter.n_air.toString().padStart(2, '0');
    document.getElementById('kpi-liquid').textContent = config.datacenter.n_liquid.toString().padStart(2, '0');
    document.getElementById('kpi-slots').textContent = (config.datacenter.n_racks * config.rack.n_slots).toString();
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
    VIEW_DOM[v].classList.toggle('active', v === view);
  });
  document.querySelectorAll('.nav-item').forEach(item => {
    item.classList.toggle('active', (item.getAttribute('data-view') === view || item.id === `nav-${view}`));
  });
  document.getElementById('kpi-focus').textContent = view.toUpperCase();
}

// --- WebSocket ---
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

// --- UI Updates ---
function updateUI(frame) {
  // 1. KPIs
  document.getElementById('kpi-energy').textContent = `${frame.kpis.active.energy_kwh.toFixed(3)} kWh`;
  document.getElementById('kpi-violations').textContent = `${frame.kpis.active.violations_s} s`;
  document.getElementById('tick-counter').textContent = `TICK: ${frame.tick}`;

  updateModeButtons(frame.mode);
  renderLog(frame.log);

  // 2. View-specific updates
  if (currentView === 'floor') {
    renderFloor(frame);
  } else if (currentView === 'rack') {
    renderRack(frame);
  }

  // 3. Update Charts
  updateCharts(frame);
}

function updateModeButtons(mode) {
  document.getElementById('btn-multi').classList.toggle('active', mode === 'multi_agent');
  document.getElementById('btn-base').classList.toggle('active', mode === 'baseline');
}

async function setMode(mode) {
  await fetch(`/api/mode/${mode}`, { method: 'POST' });
}

function renderLog(lines) {
  LOG_EL.innerHTML = lines.map(line => `<div>${line}</div>`).join('');
  LOG_EL.scrollTop = LOG_EL.scrollHeight;
}

// --- Floor Map Rendering ---
const SVG_NS = "http://www.w3.org/2000/svg";

function renderFloor(frame) {
  const svgRacks = document.getElementById('svg-racks');
  const svgSwitches = document.getElementById('svg-switches');
  const svgEdges = document.getElementById('svg-edges');
  const svgPackets = document.getElementById('svg-packets');

  // Initial draw of static elements if needed
  if (!svgRacks.children.length) {
    drawStaticTopology(frame.topology, svgEdges, svgSwitches);
  }

  // Update Racks
  frame.racks.forEach(r => {
    let rect = document.getElementById(`rect-${r.id}`);
    if (!rect) {
      const rackDef = frame.topology.nodes.find(n => n.id === r.id);
      rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("id", `rect-${r.id}`);
      rect.setAttribute("x", rackDef.x - 15);
      rect.setAttribute("y", rackDef.y - 15);
      rect.setAttribute("width", 30);
      rect.setAttribute("height", 30);
      rect.setAttribute("class", rackDef.kind === 'liquid' ? 'svg-rack-liquid' : 'svg-rack-air');
      rect.onclick = () => showRack(r.id);
      svgRacks.appendChild(rect);

      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("x", rackDef.x);
      label.setAttribute("y", rackDef.y + 25);
      label.setAttribute("class", "svg-rack-label");
      label.textContent = r.id;
      svgRacks.appendChild(label);
    }

    const tempClass = getTempClass(r.peak_temp);
    rect.style.fill = getTempColor(r.peak_temp, r.kind);
    if (tempClass === 'temp-crit') {
        rect.style.strokeWidth = "2.5";
        rect.style.stroke = COLORS.crit;
    } else {
        rect.style.strokeWidth = "1.5";
        rect.style.stroke = r.kind === 'liquid' ? COLORS.teal : COLORS.gold;
    }
  });

  // Update Priority bars
  const pLoad = frame.agents.scheduler.priority_load;
  document.getElementById('pb-p0').style.width = `${Math.min(100, (pLoad['P0 Critical'] || 0) * 4)}%`;
  document.getElementById('pb-p1').style.width = `${Math.min(100, (pLoad['P1 High'] || 0) * 3)}%`;
  document.getElementById('pb-p2').style.width = `${Math.min(100, (pLoad['P2 Medium'] || 0) * 2)}%`;
  document.getElementById('pb-p3').style.width = `${Math.min(100, (pLoad['P3 Low'] || 0) * 1.5)}%`;

  // Arbiter box
  const arbStatus = document.getElementById('arb-status');
  const arbShed = document.getElementById('arb-shed');
  arbStatus.textContent = frame.agents.arbiter.negotiation_status;
  arbStatus.className = frame.agents.arbiter.throttle_active ? 'crit' : 'safe';
  arbShed.style.display = frame.agents.arbiter.throttle_active ? 'block' : 'none';

  // Packets
  renderPackets(frame.topology.packets, svgPackets, frame.topology.nodes);
}

function drawStaticTopology(topology, edgeGroup, switchGroup) {
  // Edges
  topology.edges.forEach(([u, v]) => {
    const n1 = topology.nodes.find(n => n.id === u);
    const n2 = topology.nodes.find(n => n.id === v);
    if (n1 && n2) {
      const line = document.createElementNS(SVG_NS, "line");
      line.setAttribute("x1", n1.x);
      line.setAttribute("y1", n1.y);
      line.setAttribute("x2", n2.x);
      line.setAttribute("y2", n2.y);
      line.setAttribute("class", "svg-edge");
      edgeGroup.appendChild(line);
    }
  });

  // Switches
  topology.nodes.filter(n => n.kind === 'leaf' || n.kind === 'spine').forEach(s => {
    const rect = document.createElementNS(SVG_NS, "rect");
    const w = s.kind === 'spine' ? 60 : 40;
    const h = 18;
    rect.setAttribute("x", s.x - w/2);
    rect.setAttribute("y", s.y - h/2);
    rect.setAttribute("width", w);
    rect.setAttribute("height", h);
    rect.setAttribute("class", s.kind === 'spine' ? 'svg-spine' : 'svg-leaf');
    switchGroup.appendChild(rect);

    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", s.x);
    label.setAttribute("y", s.y + 18);
    label.setAttribute("class", "svg-switch-label");
    label.textContent = s.id;
    switchGroup.appendChild(label);
  });
}

function renderPackets(packets, group, nodes) {
  group.innerHTML = ''; // Fresh every tick for simple animation
  packets.forEach(p => {
    const rack = nodes.find(n => n.id === p.target);
    if (!rack) return;
    const leaf = nodes.find(n => n.id === rack.leaf);
    const spine = nodes.find(n => n.id === (p.id.charCodeAt(5) % 2 === 0 ? 'SPINE-01' : 'SPINE-02'));

    // Calc position along path: Spine -> Leaf -> Rack
    let x, y;
    if (p.x < 0.5) {
      const t = p.x * 2;
      x = spine.x + t * (leaf.x - spine.x);
      y = spine.y + t * (leaf.y - spine.y);
    } else {
      const t = (p.x - 0.5) * 2;
      x = leaf.x + t * (rack.x - leaf.x);
      y = leaf.y + t * (rack.y - leaf.y);
    }

    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", x);
    circle.setAttribute("cy", y);
    circle.setAttribute("r", 3);
    circle.setAttribute("fill", p.color);
    circle.style.filter = "url(#glow)";
    group.appendChild(circle);
  });
}

// --- Rack View Rendering ---
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

  // Orchestration box
  document.getElementById('orch-state').textContent = rack.throttle_active ? 'THROTTLED' : 'NOMINAL';
  document.getElementById('orch-state').className = rack.throttle_active ? 'orch-v crit' : 'orch-v safe';
  document.getElementById('orch-policy').textContent = frame.agents.router.policy;
  document.getElementById('orch-burst').textContent = rack.id === 'R02' ? '2.10x' : '1.00x'; // Synthetic visual flair

  // Lanes
  const pNames = ['P0 Critical', 'P1 High', 'P2 Medium', 'P3 Low'];
  pNames.forEach((name, i) => {
    const track = document.getElementById(`track-p${i}`);
    track.innerHTML = '';
    // Show packets targeting this rack in this lane
    frame.topology.packets.filter(p => p.priority === i).forEach(p => {
       const pkt = document.createElement('div');
       pkt.className = 'lane-pkt';
       pkt.style.left = `${(p.x * 90)}%`;
       pkt.style.top = `${p.lane_y * 100}%`;
       pkt.style.backgroundColor = p.color;
       if (p.target === selectedRackId) {
         pkt.style.boxShadow = `0 0 8px ${p.color}`;
         pkt.style.width = '12px';
         pkt.style.height = '12px';
       }
       track.appendChild(pkt);
    });
  });

  // Footer
  document.getElementById('rf-peak').textContent = `${rack.peak_temp.toFixed(1)} °C`;
  document.getElementById('rf-avg').textContent = `${rack.avg_temp.toFixed(1)} °C`;
  document.getElementById('rf-inflight').textContent = frame.agents.router.in_flight;
  document.getElementById('rf-routed').textContent = frame.agents.router.routed_total;
  document.getElementById('rf-mode').textContent = rack.kind.toUpperCase();

  // Slots
  const slotList = document.getElementById('slot-list');
  slotList.innerHTML = rack.slots.sort((a,b) => b.id - a.id).map(s => `
    <div class="slot-row">
      <div class="slot-id">S${s.id.toString().padStart(2, '0')}</div>
      <div class="slot-bar-wrap">
        <div class="slot-bar" style="width: ${s.util * 100}%; background: ${getTempColor(s.temp, rack.kind)}"></div>
      </div>
      <div class="slot-temp ${getTempClass(s.temp)}">${s.temp.toFixed(1)}°</div>
    </div>
  `).join('');
}

// --- Helpers ---
function getTempClass(t) {
  if (t > 80) return 'temp-crit';
  if (t > 75) return 'temp-warn';
  return 'temp-safe';
}

function getTempColor(t, kind) {
  if (t > 80) return COLORS.crit;
  if (t > 75) return COLORS.warn;
  return kind === 'liquid' ? COLORS.teal : COLORS.safe;
}

// --- Charts ---
function initCharts() {
  const chartConfigs = [
    { id: 'energy', label: 'Energy (kWh)', color: COLORS.gold, dataKey: 'energy' },
    { id: 'temp', label: 'Peak Temp (°C)', color: COLORS.crit, dataKey: 'peak_temp' },
    { id: 'violations', label: 'Violations', color: COLORS.warn, dataKey: 'violations', type: 'bar' },
    { id: 'fan', label: 'Fan Speed', color: COLORS.teal, dataKey: 'fan_speed' }
  ];

  chartConfigs.forEach(cfg => {
    const canvas = document.getElementById(`chart-${cfg.id}`);
    charts[cfg.id] = new Chart(canvas, {
      type: cfg.type || 'line',
      data: {
        labels: Array.from({length: 300}, (_, i) => i),
        datasets: [
          {
            label: 'Multi-Agent',
            borderColor: cfg.color,
            backgroundColor: `${cfg.color}33`,
            data: [],
            borderWidth: 1.5,
            pointRadius: 0,
            fill: true
          },
          {
            label: 'Baseline',
            borderColor: '#4b5563',
            backgroundColor: 'rgba(75, 85, 99, 0.1)',
            data: [],
            borderWidth: 1,
            pointRadius: 0,
            borderDash: [5, 5]
          }
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: true, labels: { color: '#6b7280', font: { size: 9 } } } },
        scales: {
          x: { display: false },
          y: {
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#4b5563', font: { size: 9 } }
          }
        }
      }
    });
  });
}

function updateCharts(frame) {
  const ma = frame.kpis.multi_agent.history;
  const base = frame.kpis.baseline.history;

  Object.keys(charts).forEach(id => {
    const dataKey = {
      energy: 'energy',
      temp: 'peak_temp',
      violations: 'violations',
      fan: 'fan_speed'
    }[id];

    charts[id].data.datasets[0].data = ma[dataKey];
    charts[id].data.datasets[1].data = base[dataKey];
    charts[id].update('none');
  });
}

function showFloor() {
  switchView('floor');
}

// Start
window.onload = init;

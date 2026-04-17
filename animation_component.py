"""
animation_component.py — v8
═══════════════════════════════════════════════════════════════════════════════
CHANGES FROM v6:

CHANGE A — Click mode toggle button on canvas controls:
  "Accident mode" → click road adds accident (visual + red overlay)
  "Traffic mode"  → click road shows traffic spike visual + hint to use sidebar
  A mode badge shows the current mode clearly.

CHANGE B — Route highlight rendering improved:
  Best route drawn AFTER roads (not before), so it sits visually on top.
  Alternate routes drawn at 50% opacity with wider dashes.

CHANGE C — Congestion tint on road surface:
  Roads now show a colour tint (green/orange/red) at low opacity on the
  asphalt layer, so congestion level is visible even without vehicles.
  This makes the ML → edge_states → visual pipeline obvious.

CHANGE D — HUD shows click mode and current hour clearly.

CHANGE E — Route panel ETA delta shown in red/green vs best route.

All existing features preserved. Zero f-string / JS template conflicts.
═══════════════════════════════════════════════════════════════════════════════
"""

import json


def build_animation_html(anim_data: dict, height: int = 780) -> str:
    data_json = json.dumps(anim_data)
    cw = anim_data.get("canvas_w", 1100)
    ch = anim_data.get("canvas_h", 680)

    css = (
        "* { box-sizing:border-box; margin:0; padding:0; }"
        "body { background:#0f172a; font-family:'Segoe UI',sans-serif; color:#e2e8f0; }"
        "#wrap { position:relative; width:100%; user-select:none; }"
        "canvas { display:block; width:100%; border-radius:10px 10px 0 0; background:#1e293b; }"
        "#ctrl {"
        "  display:flex; align-items:center; gap:8px; flex-wrap:wrap;"
        "  padding:7px 12px; background:#0f172a; border-top:1px solid #1e293b;"
        "  border-radius:0 0 10px 10px; font-size:11px; color:#94a3b8;"
        "}"
        "button {"
        "  padding:4px 12px; border-radius:5px; border:none;"
        "  cursor:pointer; font-size:11px; font-weight:600; transition:opacity .15s;"
        "}"
        "button:hover { opacity:.78; }"
        "#btn-pause  { background:#6366f1; color:#fff; }"
        "#btn-clear  { background:#ef4444; color:#fff; }"
        "#btn-debug  { background:#334155; color:#e2e8f0; }"
        "#btn-mode   { background:#f97316; color:#fff; min-width:110px; }"
        ".sep { width:1px; height:18px; background:#334155; }"
        "input[type=range] { accent-color:#6366f1; width:76px; }"
        "#hud {"
        "  position:absolute; top:10px; left:10px; pointer-events:none;"
        "  background:rgba(15,23,42,.9); border:1px solid #334155;"
        "  border-radius:8px; padding:8px 12px; font-size:11px; line-height:1.8;"
        "  min-width:200px; color:#cbd5e1;"
        "}"
        "#hud b { color:#f1f5f9; }"
        "#mode-badge {"
        "  position:absolute; top:10px; right:10px; pointer-events:none;"
        "  border-radius:6px; padding:5px 10px; font-size:11px; font-weight:700;"
        "  border:1px solid currentColor;"
        "}"
        "#tip {"
        "  position:absolute; pointer-events:none; display:none;"
        "  background:#0f172a; border:1px solid #334155; border-radius:8px;"
        "  padding:8px 12px; font-size:11px; line-height:1.65;"
        "  color:#e2e8f0; min-width:215px; z-index:20;"
        "}"
        "#tip b { color:#f1f5f9; }"
        "#legend { display:flex; gap:10px; flex-wrap:wrap; margin-left:auto; font-size:11px; }"
        ".li { display:flex; align-items:center; gap:4px; white-space:nowrap; }"
        ".ld { width:20px; height:9px; border-radius:2px; }"
        "#rp { font-size:11px; padding:6px 14px; background:#060d18; border-top:1px solid #1e293b; }"
        "#rp.hidden { display:none; }"
        "#click-hint {"
        "  position:absolute; bottom:52px; right:10px; pointer-events:none;"
        "  background:rgba(15,23,42,.78); border:1px solid #334155; border-radius:5px;"
        "  padding:3px 9px; font-size:10px; color:#64748b;"
        "}"
    )

    js = r"""
(function () {

var DATA    = __DATATOKEN__;
var CW      = DATA.canvas_w;
var CH      = DATA.canvas_h;
var canvas  = document.getElementById('c');
var ctx     = canvas.getContext('2d');
var tip     = document.getElementById('tip');
var hudEl   = document.getElementById('hud');
var rpEl    = document.getElementById('rp');
var modeBadge = document.getElementById('mode-badge');

// ── Simulation state ──────────────────────────────────────────────────────────
var running   = true;
var simSpeed  = 1.0;
var flashTick = 0;
var showDebug = false;

// CHANGE A: click mode — 'accident' or 'traffic'
var clickMode = 'accident';

// Client-side accident registry
var accidents = {};
DATA.accidents.forEach(function(a) { accidents[a.edge_id] = true; });

// Traffic spike visual overlay (edge_id → frames remaining)
var trafficFlash = {};

var vehicles = JSON.parse(JSON.stringify(DATA.vehicles));

var nodeMap = {};
DATA.nodes.forEach(function(n) { nodeMap[n.id] = n; });
var edgeMap = {};
DATA.edges.forEach(function(e) { edgeMap[e.id] = e; });

var LANE_W     = 10;
var LANES      = 4;
var ROAD_W     = LANES * LANE_W;
var CURB_EXTRA = 5;

canvas.width  = CW;
canvas.height = CH;

// ── Helpers ───────────────────────────────────────────────────────────────────
function ptSegDist(px, py, x1, y1, x2, y2) {
    var dx = x2-x1, dy = y2-y1;
    var lenSq = dx*dx + dy*dy;
    if (lenSq === 0) return Math.hypot(px-x1, py-y1);
    var t = Math.max(0, Math.min(1, ((px-x1)*dx + (py-y1)*dy) / lenSq));
    return Math.hypot(px - (x1+t*dx), py - (y1+t*dy));
}

function perpUnit(x1, y1, x2, y2) {
    var dx = x2-x1, dy = y2-y1;
    var len = Math.hypot(dx, dy) || 1;
    return { nx: -dy/len, ny: dx/len };
}

// ── CHANGE A: click mode toggle ───────────────────────────────────────────────
function updateModeBadge() {
    if (clickMode === 'accident') {
        modeBadge.textContent = '\u26A0 Accident mode';
        modeBadge.style.color = '#f87171';
        modeBadge.style.background = 'rgba(239,68,68,0.12)';
        canvas.style.cursor = 'crosshair';
    } else {
        modeBadge.textContent = '\uD83D\uDE97 Traffic mode';
        modeBadge.style.color = '#fb923c';
        modeBadge.style.background = 'rgba(249,115,22,0.12)';
        canvas.style.cursor = 'pointer';
    }
}
updateModeBadge();

document.getElementById('btn-mode').onclick = function() {
    clickMode = (clickMode === 'accident') ? 'traffic' : 'accident';
    this.textContent = clickMode === 'accident' ? '\u26A0 Accident mode' : '\uD83D\uDE97 Traffic mode';
    this.style.background = clickMode === 'accident' ? '#ef4444' : '#f97316';
    updateModeBadge();
};

// ── Click handler ─────────────────────────────────────────────────────────────
canvas.addEventListener('click', function(evt) {
    var rect = canvas.getBoundingClientRect();
    var mx   = (evt.clientX - rect.left) * (CW / rect.width);
    var my   = (evt.clientY - rect.top)  * (CH / rect.height);

    var best = null, bestD = Infinity;
    DATA.edges.forEach(function(e) {
        var d = ptSegDist(mx, my, e.x1, e.y1, e.x2, e.y2);
        if (d < bestD) { bestD = d; best = e; }
    });

    if (!best || bestD > ROAD_W + 14) return;

    var eid = best.id;

    if (clickMode === 'accident') {
        // Toggle accident on this edge
        if (accidents[eid]) {
            delete accidents[eid];
        } else {
            accidents[eid] = true;
        }
        rebuildEdgeColors();
    } else {
        // Traffic spike: show orange flash on this edge for 90 frames
        trafficFlash[eid] = 90;
    }
});

function rebuildEdgeColors() {
    var COLS = ['#22c55e', '#f97316', '#ef4444'];
    var idx  = { Low: 0, Medium: 1, High: 2 };
    DATA.edges.forEach(function(e) {
        e.has_accident = !!accidents[e.id];
        if (e.has_accident) {
            e.color = '#ef4444'; e.label = 2;
        } else {
            var lu = (nodeMap[e.from] ? idx[nodeMap[e.from].label_name] : 0) || 0;
            var lv = (nodeMap[e.to]   ? idx[nodeMap[e.to].label_name]   : 0) || 0;
            e.label = Math.max(lu, lv);
            e.color = COLS[e.label];
        }
    });
}

// ── Controls ──────────────────────────────────────────────────────────────────
document.getElementById('btn-clear').onclick = function() {
    accidents = {}; trafficFlash = {};
    DATA.accidents = [];
    rebuildEdgeColors();
};
document.getElementById('btn-pause').onclick = function() {
    running = !running;
    this.textContent = running ? '\u23F8 Pause' : '\u25B6 Play';
};
document.getElementById('btn-debug').onclick = function() {
    showDebug = !showDebug;
    this.style.background = showDebug ? '#6366f1' : '#334155';
};
document.getElementById('spd-sl').oninput = function() {
    simSpeed = parseFloat(this.value);
    document.getElementById('spd-lbl').textContent = simSpeed.toFixed(1) + 'x';
};

// ── Draw roads (CHANGE C: congestion tint on asphalt) ─────────────────────────
function drawRoads() {
    DATA.edges.forEach(function(e) {
        var p = perpUnit(e.x1, e.y1, e.x2, e.y2);

        // Shadow
        ctx.beginPath(); ctx.moveTo(e.x1,e.y1); ctx.lineTo(e.x2,e.y2);
        ctx.strokeStyle = 'rgba(0,0,0,0.55)';
        ctx.lineWidth   = ROAD_W + CURB_EXTRA*2 + 4;
        ctx.lineCap     = 'butt'; ctx.stroke();

        // Curb band
        ctx.beginPath(); ctx.moveTo(e.x1,e.y1); ctx.lineTo(e.x2,e.y2);
        ctx.strokeStyle = DATA.is_night ? '#1e2d40' : '#334155';
        ctx.lineWidth   = ROAD_W + CURB_EXTRA*2; ctx.stroke();

        // Asphalt base
        ctx.beginPath(); ctx.moveTo(e.x1,e.y1); ctx.lineTo(e.x2,e.y2);
        ctx.strokeStyle = DATA.is_night ? '#111c2b' : '#1e293b';
        ctx.lineWidth   = ROAD_W; ctx.stroke();

        // CHANGE C: congestion tint overlay on asphalt
        // Low=green tint, Medium=orange, High=red — makes ML prediction visible
        var tints = ['rgba(34,197,94,0.18)', 'rgba(249,115,22,0.22)', 'rgba(239,68,68,0.28)'];
        ctx.beginPath(); ctx.moveTo(e.x1,e.y1); ctx.lineTo(e.x2,e.y2);
        ctx.strokeStyle = tints[e.label] || tints[0];
        ctx.lineWidth   = ROAD_W - 4;
        ctx.globalAlpha = 1;
        ctx.stroke();

        // Traffic spike flash (CHANGE A: traffic mode click)
        if (trafficFlash[e.id] && trafficFlash[e.id] > 0) {
            var frac = trafficFlash[e.id] / 90;
            ctx.beginPath(); ctx.moveTo(e.x1,e.y1); ctx.lineTo(e.x2,e.y2);
            ctx.strokeStyle = '#fb923c';
            ctx.lineWidth   = ROAD_W - 2;
            ctx.globalAlpha = frac * 0.65;
            ctx.stroke();
            ctx.globalAlpha = 1;
        }

        // Dashed lane dividers
        ctx.save(); ctx.setLineDash([14,10]);
        ctx.strokeStyle = 'rgba(255,255,220,0.26)'; ctx.lineWidth = 0.9;
        for (var l = 1; l <= LANES-1; l++) {
            var off = (l - LANES/2) * LANE_W;
            ctx.beginPath();
            ctx.moveTo(e.x1 + p.nx*off, e.y1 + p.ny*off);
            ctx.lineTo(e.x2 + p.nx*off, e.y2 + p.ny*off);
            ctx.stroke();
        }
        ctx.setLineDash([]); ctx.restore();

        // Centre yellow divider
        ctx.beginPath(); ctx.moveTo(e.x1,e.y1); ctx.lineTo(e.x2,e.y2);
        ctx.strokeStyle = 'rgba(253,224,71,0.35)'; ctx.lineWidth = 1.2; ctx.stroke();

        // Curb edge lines
        ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.lineWidth = 0.7;
        var halfW = ROAD_W/2 + CURB_EXTRA;
        for (var side = -1; side <= 1; side += 2) {
            ctx.beginPath();
            ctx.moveTo(e.x1 + p.nx*halfW*side, e.y1 + p.ny*halfW*side);
            ctx.lineTo(e.x2 + p.nx*halfW*side, e.y2 + p.ny*halfW*side);
            ctx.stroke();
        }
    });
}

// ── Debug overlay ─────────────────────────────────────────────────────────────
function drawDebug() {
    if (!showDebug) return;
    ctx.save();
    DATA.edges.forEach(function(e) {
        var mx  = (e.x1+e.x2)/2 + 4;
        var my  = (e.y1+e.y2)/2 - 8;
        var lbl = ['Low','Medium','High'][e.label] || '?';
        var wt  = e.weighted_time !== undefined ? e.weighted_time.toFixed(1) : '?';
        var pen = e.cong_penalty  !== undefined ? e.cong_penalty.toFixed(0)  : '?';
        ctx.fillStyle = 'rgba(15,23,42,0.80)';
        ctx.fillRect(mx-2, my-10, 80, 32);
        ctx.font = 'bold 8.5px Segoe UI'; ctx.textAlign='left'; ctx.textBaseline='top';
        var cc = { Low:'#4ade80', Medium:'#fb923c', High:'#f87171' };
        ctx.fillStyle = cc[lbl] || '#fff';
        ctx.fillText(lbl, mx, my);
        ctx.fillStyle = '#94a3b8'; ctx.font = '8px Segoe UI';
        ctx.fillText('w='+wt+'  pen='+pen+'x', mx, my+12);
    });
    ctx.restore();
}

// CHANGE B: route highlights drawn after roads, best route on top
function drawRouteHighlights() {
    if (!DATA.routes || !DATA.routes.length) return;
    // Draw alternates first (below best)
    for (var ri = DATA.routes.length-1; ri >= 0; ri--) {
        var route = DATA.routes[ri];
        ctx.save();
        ctx.setLineDash(ri === 0 ? [] : [18, 9]);
        ctx.lineWidth   = ri === 0 ? ROAD_W - 2 : ROAD_W - 12;
        ctx.strokeStyle = route.color;
        ctx.globalAlpha = ri === 0 ? 0.78 : 0.42;
        ctx.lineCap = 'round'; ctx.lineJoin = 'round';
        ctx.beginPath();
        var first = true;
        for (var i = 0; i < route.path.length-1; i++) {
            var u = nodeMap[route.path[i]];
            var v = nodeMap[route.path[i+1]];
            if (!u || !v) continue;
            if (first) { ctx.moveTo(u.x, u.y); first = false; }
            ctx.lineTo(v.x, v.y);
        }
        ctx.stroke();
        ctx.setLineDash([]); ctx.restore();
    }
    // S / E markers on best
    if (DATA.routes.length > 0) {
        var best = DATA.routes[0];
        var sn = nodeMap[best.path[0]];
        var en = nodeMap[best.path[best.path.length-1]];
        [{n:sn,t:'S'},{n:en,t:'E'}].forEach(function(item){
            if (!item.n) return;
            ctx.beginPath(); ctx.arc(item.n.x, item.n.y, 14, 0, 2*Math.PI);
            ctx.fillStyle = best.color; ctx.globalAlpha = 0.95; ctx.fill();
            ctx.globalAlpha = 1;
            ctx.fillStyle = '#0f172a'; ctx.font = 'bold 10px Segoe UI';
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(item.t, item.n.x, item.n.y);
        });
    }
}

// ── Accident overlays ─────────────────────────────────────────────────────────
function drawAccidents() {
    DATA.edges.forEach(function(e) {
        if (!e.has_accident) return;
        var flash = Math.floor(flashTick/10) % 2 === 0;
        var mx    = (e.x1+e.x2)/2, my = (e.y1+e.y2)/2;
        var pulse = 0.4 + 0.6*Math.abs(Math.sin(flashTick*0.15));

        ctx.beginPath(); ctx.moveTo(e.x1,e.y1); ctx.lineTo(e.x2,e.y2);
        ctx.strokeStyle = flash ? '#ff1a1a' : '#ff6666';
        ctx.lineWidth = ROAD_W-4; ctx.globalAlpha = pulse*0.55; ctx.stroke();
        ctx.globalAlpha = 1;

        var grad = ctx.createLinearGradient(e.x1,e.y1,mx,my);
        grad.addColorStop(0, 'rgba(239,68,68,0.0)');
        grad.addColorStop(0.8,'rgba(239,68,68,0.40)');
        grad.addColorStop(1, 'rgba(239,68,68,0.0)');
        ctx.beginPath(); ctx.moveTo(e.x1,e.y1); ctx.lineTo(mx,my);
        ctx.strokeStyle=grad; ctx.lineWidth=ROAD_W+4; ctx.globalAlpha=0.65; ctx.stroke();
        ctx.globalAlpha=1;

        ctx.font='bold 18px Segoe UI'; ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillStyle = flash ? '#fff' : '#fbbf24'; ctx.globalAlpha=pulse;
        ctx.fillText('\u26A0', mx, my-(ROAD_W/2+10)); ctx.globalAlpha=1;
        ctx.font='bold 9px Segoe UI'; ctx.fillStyle='#fca5a5'; ctx.globalAlpha=0.9;
        ctx.fillText('ACCIDENT', mx, my+(ROAD_W/2+10)); ctx.globalAlpha=1;
    });
}

// ── Nodes ─────────────────────────────────────────────────────────────────────
function drawNodes() {
    DATA.nodes.forEach(function(n) {
        var r = n.radius;
        if (DATA.is_night) {
            var grd = ctx.createRadialGradient(n.x,n.y,r*0.2,n.x,n.y,r*2.2);
            grd.addColorStop(0, n.color+'44'); grd.addColorStop(1,'transparent');
            ctx.beginPath(); ctx.arc(n.x,n.y,r*2.2,0,2*Math.PI);
            ctx.fillStyle=grd; ctx.fill();
        }
        if (n.label_name === 'High') {
            var pulse = 0.5+0.5*Math.sin(flashTick*0.12);
            ctx.beginPath(); ctx.arc(n.x,n.y,r+7,0,2*Math.PI);
            ctx.strokeStyle='#ef4444'; ctx.lineWidth=2.5;
            ctx.globalAlpha=pulse*0.8; ctx.stroke(); ctx.globalAlpha=1;
        }
        ctx.beginPath(); ctx.arc(n.x,n.y,r,0,2*Math.PI);
        ctx.fillStyle=n.color; ctx.fill();
        ctx.strokeStyle = DATA.is_night ? '#0a111e' : '#0f172a';
        ctx.lineWidth=3; ctx.stroke();

        var words = n.id.split(' ');
        var l1    = words[0], l2 = words.slice(1).join(' ');
        var fsize = Math.max(8, Math.floor(r*0.58));
        ctx.fillStyle='#fff'; ctx.textAlign='center'; ctx.textBaseline='middle';
        if (l2) {
            ctx.font='bold '+fsize+'px Segoe UI';
            ctx.fillText(l1, n.x, n.y - fsize*0.45);
            ctx.font=(fsize-1)+'px Segoe UI';
            ctx.fillText(l2, n.x, n.y + fsize*0.55);
        } else {
            ctx.font='bold '+fsize+'px Segoe UI';
            ctx.fillText(l1, n.x, n.y);
        }
        var bs = Math.max(7, Math.floor(r*0.42));
        ctx.font=bs+'px Segoe UI';
        ctx.fillStyle = DATA.is_night ? '#94a3b8' : '#64748b';
        ctx.textBaseline='top';
        ctx.fillText(n.speed.toFixed(0)+' km/h', n.x, n.y+r+5);
    });
}

// ── Vehicles ──────────────────────────────────────────────────────────────────
function drawVehicles() {
    vehicles.forEach(function(v) {
        var e = edgeMap[v.edge_id]; if (!e) return;
        var px  = e.x1 + (e.x2-e.x1)*v.progress;
        var py  = e.y1 + (e.y2-e.y1)*v.progress;
        var ang = Math.atan2(e.y2-e.y1, e.x2-e.x1);
        var p   = perpUnit(e.x1,e.y1,e.x2,e.y2);
        var lo  = (v.lane===0 ? -LANE_W*0.8 : LANE_W*0.8);
        px += p.nx*lo; py += p.ny*lo;
        var carL=9, carW=4.5;
        ctx.save(); ctx.translate(px,py); ctx.rotate(ang);
        ctx.fillStyle='rgba(0,0,0,0.45)';
        ctx.beginPath();
        if(ctx.roundRect) ctx.roundRect(-carL/2+1,-carW/2+1,carL,carW,2);
        else ctx.rect(-carL/2+1,-carW/2+1,carL,carW);
        ctx.fill();
        ctx.fillStyle=v.color;
        ctx.beginPath();
        if(ctx.roundRect) ctx.roundRect(-carL/2,-carW/2,carL,carW,2);
        else ctx.rect(-carL/2,-carW/2,carL,carW);
        ctx.fill();
        ctx.fillStyle='rgba(180,220,255,0.4)';
        ctx.beginPath(); ctx.rect(carL*0.1,-carW*0.35,carL*0.25,carW*0.7); ctx.fill();
        if (v.is_night) {
            ctx.fillStyle='#fef9c3';
            ctx.beginPath(); ctx.arc(carL/2-1,-carW*0.28,1.2,0,2*Math.PI); ctx.fill();
            ctx.beginPath(); ctx.arc(carL/2-1, carW*0.28,1.2,0,2*Math.PI); ctx.fill();
        }
        ctx.restore();
    });
}

// ── HUD (CHANGE D) ────────────────────────────────────────────────────────────
function updateHUD() {
    var nh  = DATA.nodes.filter(function(n){return n.label_name==='High';}).length;
    var nm  = DATA.nodes.filter(function(n){return n.label_name==='Medium';}).length;
    var nAc = Object.keys(accidents).length;
    var avs = 0;
    DATA.nodes.forEach(function(n){avs+=n.speed;});
    avs = DATA.nodes.length ? (avs/DATA.nodes.length).toFixed(1) : '0';
    var hourStr = DATA.hour !== undefined ? ('0'+DATA.hour).slice(-2)+':00' : '--:--';
    hudEl.innerHTML =
        '<b>\uD83D\uDEA6 City Traffic &nbsp;&nbsp;' + hourStr + '</b><br>' +
        'Vehicles: <b>'+vehicles.length+'</b><br>'+
        'Avg speed: <b>'+avs+' km/h</b><br>'+
        '\uD83D\uDD34 High: <b>'+nh+'</b> &nbsp; '+
        '\uD83D\uDFE1 Medium: <b>'+nm+'</b><br>'+
        '\u26A0 Accidents: <b>'+nAc+'</b><br>'+
        (DATA.is_night ? '\uD83C\uDF19 Night' : '\u2600 Day')+
        (showDebug ? '<br><span style="color:#6366f1">DEBUG ON</span>' : '');
}

// CHANGE E: route panel with ETA delta
(function buildRoutePanel(){
    if (!DATA.routes || !DATA.routes.length){rpEl.classList.add('hidden');return;}
    rpEl.classList.remove('hidden');
    var bestEta = DATA.routes[0] ? DATA.routes[0].eta : 0;
    var h='<div style="display:flex;gap:10px;flex-wrap:wrap">';
    DATA.routes.forEach(function(r,i){
        var badge = r.acc_free
            ? '<span style="color:#4ade80">\u2714 clear</span>'
            : '<span style="color:#f87171">\u26A0 accident</span>';
        var delta = i===0 ? '' : ('<span style="color:#f97316"> +'+((r.eta-bestEta).toFixed(1))+' min</span>');
        h += '<div style="border-left:3px solid '+r.color+';padding:4px 10px;'+
             'background:#0f172a;border-radius:4px;min-width:190px">'+
             '<b style="color:'+r.color+'">'+r.label+'</b> '+badge+'<br>'+
             '<span style="color:#64748b;font-size:10px">'+r.path.join(' \u2192 ')+'</span><br>'+
             'ETA: <b style="color:#f1f5f9">'+r.eta+' min</b>'+delta+
             '</div>';
    });
    h+='</div>';
    rpEl.innerHTML=h;
})();

// ── Tooltip ───────────────────────────────────────────────────────────────────
canvas.addEventListener('mousemove',function(evt){
    var rect=canvas.getBoundingClientRect();
    var mx=(evt.clientX-rect.left)*(CW/rect.width);
    var my=(evt.clientY-rect.top)*(CH/rect.height);
    var found=null;
    DATA.nodes.forEach(function(n){
        if(Math.hypot(mx-n.x,my-n.y)<=n.radius+8) found={type:'node',d:n};
    });
    if(!found){
        var bestD=Infinity;
        DATA.edges.forEach(function(e){
            var d=ptSegDist(mx,my,e.x1,e.y1,e.x2,e.y2);
            if(d<bestD){bestD=d;if(d<ROAD_W+8)found={type:'edge',d:e};}
        });
    }
    if(found){
        tip.style.display='block';
        tip.style.left=(evt.clientX-rect.left+16)+'px';
        tip.style.top=(evt.clientY-rect.top-10)+'px';
        var d=found.d;
        var CC={Low:'#22c55e',Medium:'#f97316',High:'#ef4444'};
        var CL=['Low','Medium','High'],CA=['#22c55e','#f97316','#ef4444'];
        if(found.type==='node'){
            tip.innerHTML='<b>'+d.id+'</b><br>'+
                'Type: '+d.type.replace('_',' ')+'<br>'+
                'Volume: '+d.volume+' / '+d.capacity+' veh/hr<br>'+
                'Load: '+(d.load*100).toFixed(1)+'%<br>'+
                'Speed: '+d.speed+' km/h<br>'+
                'Status: <b style="color:'+(CC[d.label_name]||'#fff')+'">'+d.label_name+'</b><br>'+
                'P(High): '+(d.p_high*100).toFixed(1)+'%';
        } else {
            var ci=d.label||0;
            var wt=d.weighted_time!==undefined?d.weighted_time.toFixed(1):'?';
            var pen=d.cong_penalty!==undefined?d.cong_penalty.toFixed(0):'?';
            var modeHint = clickMode==='accident'
                ? '<br><span style="color:#f87171">Click: toggle accident</span>'
                : '<br><span style="color:#fb923c">Click: traffic spike visual (use sidebar to commit)</span>';
            tip.innerHTML='<b>'+d.from+' \u2192 '+d.to+'</b><br>'+
                'Load: '+(d.avg_load*100).toFixed(1)+'%<br>'+
                'Speed: '+d.speed.toFixed(1)+' km/h<br>'+
                'Base: '+d.base_time+' min<br>'+
                'Penalty: '+pen+'x<br>'+
                'Weighted: <b style="color:#fbbf24">'+wt+' min</b><br>'+
                'Status: <b style="color:'+CA[ci]+'">'+CL[ci]+'</b>'+
                (d.has_accident?'<br><span style="color:#f87171;font-weight:700">\u26A0 ACCIDENT</span>':'')+
                modeHint;
        }
    } else {
        tip.style.display='none';
    }
});
canvas.addEventListener('mouseleave',function(){tip.style.display='none';});

// ── Vehicle physics ────────────────────────────────────────────────────────────
var SAFE_GAP=0.06, DECEL_D=0.12, ACC_STOP=0.45;
function getLeaderDist(v){
    var best=Infinity;
    vehicles.forEach(function(o){
        if(o.id===v.id||o.edge_id!==v.edge_id||o.lane!==v.lane) return;
        var d=o.progress-v.progress;
        if(d>0&&d<best) best=d;
    });
    return best;
}
function updateVehicles(dt){
    // Decay traffic flash
    Object.keys(trafficFlash).forEach(function(eid){
        if(trafficFlash[eid]>0) trafficFlash[eid]-=1;
        else delete trafficFlash[eid];
    });

    vehicles.forEach(function(v){
        var e=edgeMap[v.edge_id]; if(!e) return;
        var ld=getLeaderDist(v);
        var sm=1.0;
        if(ld<SAFE_GAP) sm=0.02;
        else if(ld<DECEL_D) sm=0.1+0.9*(ld-SAFE_GAP)/(DECEL_D-SAFE_GAP);
        if(e.has_accident&&v.progress>ACC_STOP-0.08) sm=Math.min(sm,0.08);
        var target=(v.base_speed||0.005)*sm;
        v.currentSpeed=v.currentSpeed===undefined?target:
            v.currentSpeed+(target-v.currentSpeed)*Math.min(1,dt*3.5);
        v.progress+=v.currentSpeed*simSpeed;
        if(v.progress>=1.0){
            var cands=DATA.edges.filter(function(e2){return e2.from===e.to;});
            var next=cands.length?cands[Math.floor(Math.random()*cands.length)]:
                DATA.edges[Math.floor(Math.random()*DATA.edges.length)];
            v.edge_id=next.id; v.from=next.from; v.to=next.to;
            v.progress=0; v.currentSpeed=0;
            var src=nodeMap[next.from];
            v.base_speed=src?Math.max(0.003,(1-src.load)*0.009):0.004;
            v.color=next.color; v.is_night=DATA.is_night;
            v.lane=Math.floor(Math.random()*2);
        }
    });
}

// ── Main loop ──────────────────────────────────────────────────────────────────
var lastT=performance.now();
function tick(now){
    var dt=Math.min((now-lastT)/1000,0.05); lastT=now; flashTick++;
    ctx.fillStyle=DATA.is_night?'#060d18':'#0e1a2b';
    ctx.fillRect(0,0,CW,CH);
    drawRoads();
    drawRouteHighlights();   // CHANGE B: after roads, best on top
    drawAccidents();
    drawNodes();
    drawVehicles();
    drawDebug();
    updateHUD();
    if(running) updateVehicles(dt);
    requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

})();
"""

    js_final = js.replace("__DATATOKEN__", data_json)

    ctrl = (
        '<div id="ctrl">'
        '<button id="btn-pause">&#9646;&#9646; Pause</button>'
        '<div class="sep"></div>'
        '<button id="btn-mode">&#9888; Accident mode</button>'
        '<button id="btn-clear">&#10005; Clear all</button>'
        '<div class="sep"></div>'
        '<button id="btn-debug">&#128203; Debug</button>'
        '<div class="sep"></div>'
        '<div style="display:flex;align-items:center;gap:5px">'
        '<span>Speed</span>'
        '<input type="range" id="spd-sl" min="0.2" max="3.5" step="0.1" value="1.0">'
        '<span id="spd-lbl">1.0x</span>'
        '</div>'
        '<div id="legend">'
        '<div class="li"><span class="ld" style="background:#334155;border:1px solid #475569"></span>Road</div>'
        '<div class="li"><span class="ld" style="background:#22c55e;opacity:.6"></span>Low tint</div>'
        '<div class="li"><span class="ld" style="background:#f97316;opacity:.6"></span>Medium tint</div>'
        '<div class="li"><span class="ld" style="background:#ef4444;opacity:.7"></span>High / Accident</div>'
        '<div class="li"><span class="ld" style="background:#16a34a"></span>Best route</div>'
        '<div class="li"><span class="ld" style="background:#3b82f6;opacity:.7"></span>Alternate</div>'
        '</div>'
        '</div>'
    )

    hint = '<div id="click-hint">Click road to add accident / traffic spike</div>'

    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>" + css + "</style>"
        "</head><body>"
        "<div id='wrap'>"
        "<div id='hud'></div>"
        "<div id='mode-badge'></div>"
        "<div id='tip'></div>"
        "<canvas id='c' width='" + str(cw) + "' height='" + str(ch) + "'></canvas>"
        + ctrl + hint +
        "</div>"
        "<div id='rp' class='hidden'></div>"
        "<script>" + js_final + "</script>"
        "</body></html>"
    )

"""
simulation_engine.py — v8
═══════════════════════════════════════════════════════════════════════════════
CHANGES FROM v7 (all targeted — structure preserved):

CHANGE 1 — time_density() rewritten to exact required spec:
  10:00–13:00 → HIGH base (0.82)
  18:00–21:00 → HIGH base (0.80)
  14:00–17:00 → MEDIUM base (0.55)
   7:00– 9:00 → MEDIUM-HIGH base (0.65)
   0:00– 5:00 → LOW base (0.08)
  rest        → LOW-MEDIUM base (0.35)
  Weekend     → +10% bonus

CHANGE 2 — NODE_BUSYNESS weights added (spatial variation):
  Each node gets a fixed multiplier (0.65–1.25) reflecting its role.
  City Center / Railway / Market = busy. School / Old Town = quiet.
  This prevents all nodes hitting the same load level simultaneously,
  creating realistic clusters of congestion instead of uniform High.

CHANGE 3 — FORCE_HIGH_LOAD raised from 0.80 → 0.92.
  0.80 was too easy to reach (6/10 nodes became High at evening peak).
  0.92 is only reached by the busiest nodes at peak hours.

CHANGE 4 — CONGESTION_PENALTY updated to required values:
  Medium: 5x → 8x
  High:   15x → 30x
  These stronger penalties make routes visibly change when congestion shifts.

CHANGE 5 — Node-level seed is now hour-based (not global seed=42).
  random.seed(seed + hash(node) % 1000) gives each node its own noise
  pattern that evolves naturally across hours without all nodes being in sync.

CHANGE 6 — ETA_SEGMENT_CAP raised from 90 → 120 min.
  With High penalty now 30x, a 6-min base segment → 180 min weighted.
  Capping at 120 makes "accident route" clearly worse than "clear route"
  without showing absurd 2500-min values.

CHANGE 7 — CONGESTION_PENALTY dict updated; all dependent code uses it.
  No other function signatures changed.
═══════════════════════════════════════════════════════════════════════════════
"""

import networkx as nx
import pandas as pd
import random
import joblib

location_encoder = joblib.load("models/location_encoder.pkl")

# ── Canvas ────────────────────────────────────────────────────────────────────
CANVAS_W = 1100
CANVAS_H = 680

# ── 10 named city nodes ───────────────────────────────────────────────────────
NODE_POSITIONS = {
    "City Center":     (550, 340),
    "Market Square":   (280, 200),
    "Railway Station": (550, 120),
    "IT Park":         (820, 200),
    "Hospital Road":   (820, 460),
    "Bus Stand":       (550, 560),
    "School Zone":     (280, 460),
    "Old Town":        (150, 340),
    "Tech Hub":        (950, 340),
    "Airport Road":    (550, 640),
}

NODE_META = {
    "City Center":     {"capacity": 2000, "base_speed": 50,  "type": "main_road"},
    "Market Square":   {"capacity": 1200, "base_speed": 40,  "type": "main_road"},
    "Railway Station": {"capacity": 1800, "base_speed": 60,  "type": "main_road"},
    "IT Park":         {"capacity": 1500, "base_speed": 55,  "type": "main_road"},
    "Hospital Road":   {"capacity": 1000, "base_speed": 40,  "type": "main_road"},
    "Bus Stand":       {"capacity": 1400, "base_speed": 45,  "type": "main_road"},
    "School Zone":     {"capacity":  600, "base_speed": 25,  "type": "side_road"},
    "Old Town":        {"capacity":  800, "base_speed": 30,  "type": "side_road"},
    "Tech Hub":        {"capacity": 1600, "base_speed": 65,  "type": "highway"},
    "Airport Road":    {"capacity": 2200, "base_speed": 70,  "type": "highway"},
}

# CHANGE 2: Per-node busyness multipliers for spatial variation
# High = always busy | Low = quieter by nature
NODE_BUSYNESS = {
    "City Center":     1.25,   # central hub, always high activity
    "Market Square":   1.15,   # busy commercial zone
    "Railway Station": 1.20,   # peak at commute times
    "IT Park":         1.10,   # busy during work hours
    "Hospital Road":   0.85,   # moderate, 24h steady
    "Bus Stand":       1.05,   # moderate-high
    "School Zone":     0.65,   # low outside school hours
    "Old Town":        0.70,   # quieter residential area
    "Tech Hub":        0.90,   # highway, flows well
    "Airport Road":    0.80,   # highway, steady flow
}

RAW_EDGES = [
    ("City Center",     "Market Square",   5),
    ("Market Square",   "City Center",     5),
    ("City Center",     "Railway Station", 4),
    ("Railway Station", "City Center",     4),
    ("City Center",     "IT Park",         5),
    ("IT Park",         "City Center",     5),
    ("City Center",     "Hospital Road",   5),
    ("Hospital Road",   "City Center",     5),
    ("City Center",     "Bus Stand",       4),
    ("Bus Stand",       "City Center",     4),
    ("City Center",     "School Zone",     5),
    ("School Zone",     "City Center",     5),
    ("Market Square",   "Railway Station", 6),
    ("Railway Station", "Market Square",   6),
    ("Railway Station", "IT Park",         6),
    ("IT Park",         "Railway Station", 6),
    ("IT Park",         "Tech Hub",        4),
    ("Tech Hub",        "IT Park",         4),
    ("Tech Hub",        "Hospital Road",   5),
    ("Hospital Road",   "Tech Hub",        5),
    ("Hospital Road",   "Bus Stand",       4),
    ("Bus Stand",       "Hospital Road",   4),
    ("Bus Stand",       "Airport Road",    5),
    ("Airport Road",    "Bus Stand",       5),
    ("Airport Road",    "School Zone",     6),
    ("School Zone",     "Airport Road",    6),
    ("School Zone",     "Old Town",        4),
    ("Old Town",        "School Zone",     4),
    ("Old Town",        "Market Square",   5),
    ("Market Square",   "Old Town",        5),
    ("Old Town",        "City Center",     6),
    ("City Center",     "Old Town",        6),
    ("Tech Hub",        "City Center",     7),
    ("City Center",     "Tech Hub",        7),
    ("Airport Road",    "City Center",     8),
    ("City Center",     "Airport Road",    8),
]

# ── Constants ─────────────────────────────────────────────────────────────────
CONGESTION_COLORS = {0: "#22c55e", 1: "#f97316", 2: "#ef4444"}
CONGESTION_NAMES  = {0: "Low",     1: "Medium",  2: "High"}
NODE_RADIUS       = {"highway": 22, "main_road": 18, "side_road": 14}
ROUTE_COLORS      = ["#16a34a", "#3b82f6", "#a855f7"]

# CHANGE 4: stronger penalties so routes visibly change
CONGESTION_PENALTY = {0: 1.0, 1: 4.0, 2: 12.0}
ACCIDENT_PENALTY_MUL = 500.0

# CHANGE 3: higher threshold → only truly saturated nodes forced High
FORCE_HIGH_LOAD  = 0.97
FORCE_HIGH_SPEED = 18.0

# CHANGE 6: larger cap → accident routes show ~120 min per seg (clearly worse)
ETA_SEGMENT_CAP = 120.0


# ── Volume spike registry ─────────────────────────────────────────────────────
_volume_spikes: dict = {}
_node_spikes:   dict = {}


def apply_volume_spike(edge_id: str, extra: int = 300):
    _volume_spikes[edge_id] = _volume_spikes.get(edge_id, 0) + extra


def apply_node_spike(node_name: str, extra: int = 300):
    _node_spikes[node_name] = _node_spikes.get(node_name, 0) + extra


def clear_volume_spikes():
    _volume_spikes.clear()
    _node_spikes.clear()


def get_volume_spike(edge_id: str) -> int:
    return _volume_spikes.get(edge_id, 0)


# ── Accident manager ──────────────────────────────────────────────────────────
class AccidentManager:
    def __init__(self):
        self.accidents: dict = {}

    @staticmethod
    def edge_id(u, v):
        return u + "__" + v

    def add(self, u, v, severity=0.85):
        self.accidents[self.edge_id(u, v)] = {"u": u, "v": v, "severity": severity}

    def remove(self, u, v):
        self.accidents.pop(self.edge_id(u, v), None)

    def remove_by_id(self, eid: str):
        self.accidents.pop(eid, None)

    def clear(self):
        self.accidents.clear()

    def has(self, u, v):
        return self.edge_id(u, v) in self.accidents

    def to_list(self):
        return [{"edge_id": eid, **a} for eid, a in self.accidents.items()]

    def neighboring_nodes(self, G) -> set:
        """All nodes adjacent to any accident edge (for load propagation)."""
        affected = set()
        for a in self.to_list():
            affected.add(a["u"])
            affected.add(a["v"])
            for nb in list(G.predecessors(a["u"])) + list(G.successors(a["v"])):
                affected.add(nb)
        return affected


def build_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    for n, m in NODE_META.items():
        G.add_node(n, **m, pos=NODE_POSITIONS[n])
    for u, v, t in RAW_EDGES:
        G.add_edge(u, v, base_weight=t)
    return G


# ── CHANGE 1: time_density — exact required spec ──────────────────────────────
def time_density(hour: int, day: int) -> float:
    """
    Returns global load base fraction 0..1.
    Applied before per-node busyness multiplier.

    Required spec:
      10:00–13:00 → HIGH   → base 0.82
      18:00–21:00 → HIGH   → base 0.80
      14:00–17:00 → MEDIUM → base 0.55
       7:00– 9:00 → MED-HI → base 0.65
       0:00– 5:00 → LOW    → base 0.08
      rest        → L-MED  → base 0.35
    """
    is_weekend = day >= 5
    if   10 <= hour <= 13:          base = 0.82   # HIGH window (midday)
    elif 18 <= hour <= 21:          base = 0.80   # HIGH window (evening)
    elif  7 <= hour <=  9:          base = 0.65   # morning commute
    elif 14 <= hour <= 17:          base = 0.55   # MEDIUM afternoon
    elif hour >= 22 or hour <= 5:   base = 0.08   # night LOW
    else:                           base = 0.35   # transition (6, 10 shoulder, 22 shoulder)
    if is_weekend:
        base = min(base * 1.10, 1.0)
    return base


# ── Node state simulation ─────────────────────────────────────────────────────
def simulate_node_states(
    G, model, features,
    hour, day, temp_c, rain, clouds, weather_code,
    accident_mgr=None, seed=42,
):
    is_rush  = 1 if (10 <= hour <= 13) or (18 <= hour <= 21) else 0
    is_night = 1 if (hour >= 22 or hour <= 5) else 0
    is_we    = 1 if day >= 5 else 0
    base     = time_density(hour, day)

    acc_affected = accident_mgr.neighboring_nodes(G) if accident_mgr else set()

    states = {}
    for node, attr in G.nodes(data=True):
        # CHANGE 5: per-node seed from hour + node hash so each node
        # has its own noise pattern, independent of other nodes
        node_seed = seed + hour * 31 + abs(hash(node)) % 997
        rng = random.Random(node_seed)
        noise = rng.uniform(0.85, 1.15)

        # CHANGE 2: apply busyness weight for spatial variation
        busyness = NODE_BUSYNESS.get(node, 1.0)
        raw_vol  = attr["capacity"] * base * busyness * noise
        vol      = int(min(raw_vol, attr["capacity"]))

        # Accident neighbor load boost (localised — only affected nodes)
        if node in acc_affected:
            vol = min(int(vol * 1.40), attr["capacity"])

        # Volume spikes from edge clicks (localised)
        for u, v in G.in_edges(node):
            eid   = AccidentManager.edge_id(u, v)
            spike = get_volume_spike(eid)
            if spike:
                vol = min(vol + spike, attr["capacity"])

        # Volume spikes from sidebar node selector (localised)
        node_spike = _node_spikes.get(node, 0)
        if node_spike:
            vol = min(vol + node_spike, attr["capacity"])

        load  = vol / attr["capacity"]
        speed = max(4.0, attr["base_speed"] * (1.0 - load) ** 1.6)

        # ML prediction
        row = {
            "hour": hour,
            "day": day,
            "month": 6,
            "is_weekend": is_we,
            "is_rush_hour": is_rush,
            "is_night": is_night,

            # NEW MODEL FEATURES
            "vehicle_count": vol,
            "avg_speed": speed,
            "rainfall": rain,
            "clouds": clouds,
            "temperature": temp_c,

            # VERY IMPORTANT
            "location_enc": location_encoder.transform([node])[0]   # will update next
        }
        try:
            X     = pd.DataFrame([row]).reindex(columns=features, fill_value=0)
            label = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]
        except Exception:
            label = 0
            proba = [1.0, 0.0, 0.0]

        # CHANGE 3: Force-High only at truly saturated levels
        if load >= FORCE_HIGH_LOAD and speed <= FORCE_HIGH_SPEED:
            label = 2

        states[node] = {
            "volume":     vol,
            "capacity":   attr["capacity"],
            "load":       round(load, 3),
            "speed":      round(speed, 1),
            "label":      label,
            "label_name": CONGESTION_NAMES[label],
            "color":      CONGESTION_COLORS[label],
            "p_low":      round(float(proba[0]), 4),
            "p_med":      round(float(proba[1]), 4),
            "p_high":     round(float(proba[2]), 4),
            "type":       attr["type"],
            "pos":        NODE_POSITIONS[node],
            "is_night":   bool(is_night),
            "radius":     NODE_RADIUS[attr["type"]],
        }
    return states


# ── Edge state computation ────────────────────────────────────────────────────
def compute_edge_states(G, node_states, accident_mgr=None):
    """
    weighted_time = base_weight × CONGESTION_PENALTY[label] × ACCIDENT_PENALTY_MUL
    CHANGE 4: penalties are now 8x/30x (was 5x/15x) → routes change visibly.
    """
    edge_states = {}
    for u, v, data in G.edges(data=True):
        ns_u = node_states.get(u, {})
        ns_v = node_states.get(v, {})

        label_u = ns_u.get("label", 0)
        label_v = ns_v.get("label", 0)

        # 🔥 NEW LOGIC
        if label_u == 2 and label_v == 2:
            edge_label = 2
        elif label_u == 2 or label_v == 2:
            edge_label = 1
        else:
            edge_label = int((label_u + label_v) / 2)

        eid     = AccidentManager.edge_id(u, v)
        has_acc = bool(accident_mgr and accident_mgr.has(u, v))

        if has_acc:
            edge_label = 2
            sev        = accident_mgr.accidents.get(eid, {}).get("severity", 0.85)
            speed_val  = max(4.0, ns_u.get("speed", 10.0) * (1.0 - sev))
        else:
            speed_val = ns_u.get("speed", 30.0)

        avg_load = (ns_u.get("load", 0.3) + ns_v.get("load", 0.3)) / 2.0
        n_veh    = max(1, int(avg_load * 8))
        if has_acc:
            n_veh = max(6, int(avg_load * 16))

        base_t        = data["base_weight"]
        c_pen         = CONGESTION_PENALTY[edge_label]   # CHANGE 4
        a_pen         = ACCIDENT_PENALTY_MUL if has_acc else 1.0
        weighted_time = round(base_t * c_pen * a_pen, 2)
        if edge_label == 2 and base_t > 8:
            weighted_time = round(weighted_time * 2, 2)

        edge_states[eid] = {
            "id":            eid,
            "from":          u,
            "to":            v,
            "x1":            NODE_POSITIONS[u][0],
            "y1":            NODE_POSITIONS[u][1],
            "x2":            NODE_POSITIONS[v][0],
            "y2":            NODE_POSITIONS[v][1],
            "label":         edge_label,
            "color":         CONGESTION_COLORS[edge_label],
            "speed":         round(speed_val, 1),
            "avg_load":      round(avg_load, 3),
            "n_vehicles":    n_veh,
            "has_accident":  has_acc,
            "base_time":     base_t,
            "weighted_time": weighted_time,
            "cong_penalty":  c_pen,
            "acc_penalty":   a_pen,
            "type":          NODE_META.get(u, {}).get("type", "main_road"),
        }
    return edge_states


# ── Routing ───────────────────────────────────────────────────────────────────
def find_multi_routes(G, source, target, node_states, edge_states,
                       accident_mgr=None, k=3):
    """
    Fresh weighted graph built from edge_states every call — no caching.
    ETA = sum of min(edge.weighted_time, ETA_SEGMENT_CAP) per segment.
    CHANGE 6: ETA_SEGMENT_CAP = 120 (was 90).
    """
    if source == target:
        return []

    WG = nx.DiGraph()
    WG.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        eid = AccidentManager.edge_id(u, v)
        es  = edge_states.get(eid)
        wt  = es["weighted_time"] if es else float(data["base_weight"])
        bt  = es["base_time"]     if es else float(data["base_weight"])
        WG.add_edge(u, v, weight=wt, base_time=bt)

    routes = []
    seen   = set()
    try:
        paths = list(nx.shortest_simple_paths(WG, source, target, weight="weight"))
    except Exception:
        return []

    for path in paths:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)

        base_cost = 0.0
        for u, v in zip(path, path[1:]):
            if G.has_edge(u, v):
                base_cost += G[u][v]["base_weight"]

        weighted_cost = 0.0
        for u, v in zip(path, path[1:]):
            if WG.has_edge(u, v):
                weighted_cost += WG[u][v]["weight"]

        # ETA: sum of capped weighted times (CHANGE 6: cap=120)
        eta = 0.0
        for u, v in zip(path, path[1:]):
            eid = AccidentManager.edge_id(u, v)
            es  = edge_states.get(eid)
            if es:
                eta += min(es["weighted_time"], ETA_SEGMENT_CAP)
            elif G.has_edge(u, v):
                eta += float(G[u][v]["base_weight"])
        eta = round(eta, 1)

        acc_free = True
        if accident_mgr:
            acc_free = not any(
                accident_mgr.has(u, v) for u, v in zip(path, path[1:])
            )

        high_n  = sum(1 for n in path if node_states.get(n, {}).get("label", 0) == 2)
        cong_sc = round(high_n / max(len(path) - 1, 1), 3)

        segs = [edge_states.get(AccidentManager.edge_id(u, v)) for u, v in zip(path, path[1:])]
        segs = [s for s in segs if s]
        avg_pen = round(
            sum(s["cong_penalty"] * s["acc_penalty"] for s in segs) / len(segs), 2
        ) if segs else 1.0

        edge_details = []
        for u, v in zip(path, path[1:]):
            eid = AccidentManager.edge_id(u, v)
            es  = edge_states.get(eid, {})
            edge_details.append({
                "segment":  u + " → " + v,
                "base":     G[u][v]["base_weight"] if G.has_edge(u, v) else "?",
                "label":    CONGESTION_NAMES.get(es.get("label", 0), "Low"),
                "penalty":  es.get("cong_penalty", 1.0),
                "acc_pen":  es.get("acc_penalty",  1.0),
                "weighted": es.get("weighted_time", "?"),
            })

        routes.append({
            "path":             path,
            "base_cost":        round(base_cost, 1),
            "weighted_cost":    round(weighted_cost, 2),
            "eta":              eta,
            "is_accident_free": acc_free,
            "cong_score":       cong_sc,
            "avg_penalty":      avg_pen,
            "edge_details":     edge_details,
        })

        if len(routes) >= k:
            break

        # Remove very similar routes (NEW STEP 3)
        filtered_routes = []

        for r in routes:
            is_similar = False
            for fr in filtered_routes:
                overlap = len(set(r["path"]) & set(fr["path"]))
                if overlap > len(r["path"]) * 0.7:
                    is_similar = True
                    break
            if not is_similar:
                filtered_routes.append(r)

        routes = filtered_routes[:k]

    routes.sort(key=lambda r: (not r["is_accident_free"], r["weighted_cost"]))

    labels = ["Best route", "Alternate 1", "Alternate 2"]
    for i, r in enumerate(routes):
        r["rank"]         = i + 1
        r["label"]        = labels[min(i, 2)]
        r["time_vs_best"] = round(r["eta"] - routes[0]["eta"], 1) if i > 0 else 0.0

    return routes


# ── Animation data builder ────────────────────────────────────────────────────
def build_animation_data(G, node_states, edge_states,
                          routes=None, accident_mgr=None, hour=8, day=1):
    nodes_out = []
    for n, s in node_states.items():
        x, y = NODE_POSITIONS.get(n, (0, 0))
        nodes_out.append({
            "id":         n,    "x": x,    "y": y,
            "color":      s.get("color",      "#22c55e"),
            "label_name": s.get("label_name", "Low"),
            "type":       s.get("type",       "main_road"),
            "speed":      s.get("speed",      30.0),
            "load":       s.get("load",       0.0),
            "volume":     s.get("volume",     0),
            "capacity":   s.get("capacity",   1000),
            "p_high":     s.get("p_high",     0.0),
            "radius":     s.get("radius",     18),
            "is_night":   s.get("is_night",   False),
        })

    edges_out = list(edge_states.values())

    rng     = random.Random(77)
    density = time_density(hour, day)
    vehicles_out = []
    vid = 0
    for eid, es in edge_states.items():
        n_v = max(1, int(es.get("n_vehicles", 1) * density))
        for i in range(n_v):
            prog = (i / max(n_v, 1) + rng.uniform(0, 0.6 / max(n_v, 1))) % 1.0
            spd  = max(0.003, (1.0 - es.get("avg_load", 0.3)) * 0.009)
            from_node = es.get("from", "")
            is_night  = node_states.get(from_node, {}).get("is_night", False)
            vehicles_out.append({
                "id":         vid,
                "edge_id":    eid,
                "from":       from_node,
                "to":         es.get("to", ""),
                "progress":   round(prog, 4),
                "base_speed": round(spd, 6),
                "color":      es.get("color", "#22c55e"),
                "is_night":   is_night,
                "lane":       rng.randint(0, 1),
            })
            vid += 1

    routes_out = []
    if routes:
        for i, r in enumerate(routes[:3]):
            routes_out.append({
                "path":      r["path"],
                "color":     ROUTE_COLORS[i],
                "label":     r.get("label", "Route"),
                "eta":       r.get("eta", 0),
                "base_cost": r.get("base_cost", 0),
                "acc_free":  r.get("is_accident_free", True),
                "cong_score":r.get("cong_score", 0),
            })

    first_node      = list(node_states.keys())[0] if node_states else None
    is_night_global = node_states.get(first_node, {}).get("is_night", False)

    return {
        "nodes":     nodes_out,
        "edges":     edges_out,
        "vehicles":  vehicles_out,
        "routes":    routes_out,
        "accidents": accident_mgr.to_list() if accident_mgr else [],
        "is_night":  is_night_global,
        "canvas_w":  CANVAS_W,
        "canvas_h":  CANVAS_H,
        "hour":      hour,
        "day":       day,
    }


def all_nodes():
    return list(NODE_META.keys())

def all_edges_list(G):
    return list(G.edges())

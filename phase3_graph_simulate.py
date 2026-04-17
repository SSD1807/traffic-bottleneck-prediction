"""
Phase 3 — Graph Road Network + Congestion Simulation (FINAL)
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import random
import os
import warnings

warnings.filterwarnings("ignore")  # suppress sklearn warnings

os.makedirs("outputs", exist_ok=True)

# ── 1. Build graph ──────────────────────────────────────────
print("[1/5] Building road network graph...")

G = nx.DiGraph()

nodes = {
    "A_Highway_N":  {"capacity": 3000, "base_speed": 100, "type": "highway"},
    "B_Highway_S":  {"capacity": 3000, "base_speed": 100, "type": "highway"},
    "C_Main_NW":    {"capacity": 1800, "base_speed": 60,  "type": "main_road"},
    "D_Main_NE":    {"capacity": 1800, "base_speed": 60,  "type": "main_road"},
    "E_Main_SW":    {"capacity": 1800, "base_speed": 60,  "type": "main_road"},
    "F_Main_SE":    {"capacity": 1800, "base_speed": 60,  "type": "main_road"},
    "G_Central":    {"capacity": 800,  "base_speed": 40,  "type": "main_road"},
    "H_Side_N":     {"capacity": 400,  "base_speed": 30,  "type": "side_road"},
    "I_Side_E":     {"capacity": 400,  "base_speed": 30,  "type": "side_road"},
    "J_Side_W":     {"capacity": 400,  "base_speed": 30,  "type": "side_road"},
    "K_School_Zone":{"capacity": 200,  "base_speed": 20,  "type": "side_road"},
    "L_Market":     {"capacity": 600,  "base_speed": 30,  "type": "side_road"},
}

for node, attr in nodes.items():
    G.add_node(node, **attr)

edges = [
    ("A_Highway_N", "C_Main_NW", 5),
    ("A_Highway_N", "D_Main_NE", 5),
    ("B_Highway_S", "E_Main_SW", 5),
    ("B_Highway_S", "F_Main_SE", 5),
    ("C_Main_NW", "G_Central", 8),
    ("D_Main_NE", "G_Central", 8),
    ("E_Main_SW", "G_Central", 8),
    ("F_Main_SE", "G_Central", 8),
    ("G_Central", "H_Side_N", 10),
    ("G_Central", "I_Side_E", 10),
    ("G_Central", "J_Side_W", 10),
    ("G_Central", "K_School_Zone", 12),
    ("H_Side_N", "L_Market", 6),
    ("I_Side_E", "L_Market", 6),
    ("C_Main_NW", "H_Side_N", 7),
    ("D_Main_NE", "I_Side_E", 7),
]

for u, v, t in edges:
    G.add_edge(u, v, travel_time=t)

print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── 2. Load model ───────────────────────────────────────────
print("[2/5] Loading trained model...")
model = joblib.load("models/best_model.pkl")
features = joblib.load("models/feature_cols.pkl")

# ── 3. Simulation ───────────────────────────────────────────
print("[3/5] Simulating traffic...")

def simulate_traffic_state(hour, day, temp_c=20, rain=0, snow=0, clouds=30, weather_code=0):

    is_rush = 1 if (7 <= hour <= 9 or 16 <= hour <= 18) else 0
    is_night = 1 if (hour >= 22 or hour <= 5) else 0
    is_we = 1 if day >= 5 else 0

    base_load = 0.5
    if is_rush: base_load = 0.85
    if is_night: base_load = 0.2
    if is_we: base_load *= 0.7

    state = {}

    for node, attr in G.nodes(data=True):
        try:
            noise = random.uniform(0.7, 1.3)
            vol = int(attr["capacity"] * base_load * noise)
            vol = max(10, min(vol, attr["capacity"]))

            load = vol / max(attr["capacity"], 1)
            speed = max(5, attr["base_speed"] * (1 - load) ** 1.5)

            row = {
                "hour": hour,
                "day": day,
                "month": 6,
                "is_weekend": is_we,
                "is_rush_hour": is_rush,
                "is_night": is_night,
                "temp_celsius": temp_c,
                "rain_1h": rain,
                "snow_1h": snow,
                "clouds_all": clouds,
                "weather_code": weather_code,
                "traffic_volume": vol   # 🔥 CRITICAL FIX
            }

            X_row = pd.DataFrame([row]).reindex(columns=features, fill_value=0)

            pred = model.predict(X_row)[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_row)[0]
            else:
                proba = [0, 0, 0]

        except Exception as e:
            print(f"⚠️ Error at node {node}: {e}")
            pred = 1
            proba = [0.3, 0.4, 0.3]

        state[node] = {
            "volume": vol,
            "capacity": attr["capacity"],
            "speed": round(speed, 1),
            "label": int(pred),
            "label_name": ["Low","Medium","High"][int(pred)],
            "p_high": round(proba[2], 3),
            "type": attr["type"]
        }

    return state


# Run simulation
state = simulate_traffic_state(hour=8, day=1)

print("\nNode predictions:")
for n, s in state.items():
    print(f"{n}: {s}")

bottlenecks = [n for n, s in state.items() if s["label"] == 2]
print("\n🔴 Bottlenecks:", bottlenecks)

# ── 4. Visualization ────────────────────────────────────────
print("[4/5] Drawing graph...")

color_map = {0: "green", 1: "orange", 2: "red"}
node_colors = [color_map[state[n]["label"]] for n in G.nodes()]

pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(10,6))
nx.draw(G, pos, with_labels=True, node_color=node_colors)
plt.savefig("outputs/network.png")
plt.close()

# ── 5. Hourly sweep (optimized) ─────────────────────────────
print("[5/5] Hourly congestion sweep...")

hours = range(0, 24, 2)  # 🔥 faster
high_counts = []

for h in hours:
    st = simulate_traffic_state(hour=h, day=1)
    high_counts.append(sum(1 for n in st if st[n]["label"] == 2))

plt.figure(figsize=(10,4))
plt.plot(list(hours), high_counts, marker='o')
plt.xlabel("Hour")
plt.ylabel("High congestion nodes")
plt.title("Congestion trend")
plt.savefig("outputs/hourly_congestion_sweep.png")
plt.close()

print("\n✅ Phase 3 COMPLETE")
print("Next → streamlit run dashboard.py")
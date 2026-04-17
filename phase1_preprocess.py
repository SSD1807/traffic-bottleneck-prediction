"""
Phase 1 — Data Preprocessing & Feature Engineering
Traffic Bottleneck / Congestion Prediction Project
Run: python phase1_preprocess.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

# ── 1. Load raw data ──────────────────────────────────────────────────────────
print("[1/6] Loading dataset...")
df = pd.read_csv("dataset/Metro_Interstate_Traffic_Volume.csv")
print(f"  Shape: {df.shape}")
print(df.head(3))

# ── 2. Parse datetime ─────────────────────────────────────────────────────────
print("[2/6] Parsing datetime...")
df["date_time"] = pd.to_datetime(df["date_time"], dayfirst=True)
df = df.sort_values("date_time").reset_index(drop=True)

# ── 3. Feature engineering ────────────────────────────────────────────────────
print("[3/6] Engineering features...")
df["hour"]       = df["date_time"].dt.hour
df["day"]        = df["date_time"].dt.dayofweek          # 0=Mon … 6=Sun
df["month"]      = df["date_time"].dt.month
df["is_weekend"] = (df["day"] >= 5).astype(int)

# Rush hours: 7-9 AM and 4-6 PM
df["is_rush_hour"] = df["hour"].apply(
    lambda h: 1 if (7 <= h <= 9) or (16 <= h <= 18) else 0
)

# Night (10 PM – 5 AM)
df["is_night"] = df["hour"].apply(lambda h: 1 if (h >= 22 or h <= 5) else 0)

# Temperature: Kelvin → Celsius
df["temp_celsius"] = df["temp"] - 273.15
df = df[df["temp_celsius"].between(-40, 60)]

# Rain: clip extreme outlier
df["rain_1h"] = df["rain_1h"].clip(upper=100)

# Weather encoding
weather_map = {w: i for i, w in enumerate(df["weather_main"].unique())}
df["weather_code"] = df["weather_main"].map(weather_map)

# ── 4. Create congestion label ────────────────────────────────────────────────
print("[4/6] Creating congestion labels...")
low_cut  = df["traffic_volume"].quantile(0.33)
high_cut = df["traffic_volume"].quantile(0.67)

def label_congestion(v):
    if v <= low_cut:
        return 0
    elif v <= high_cut:
        return 1
    else:
        return 2

df["congestion_label"] = df["traffic_volume"].apply(label_congestion)
label_names = {0: "Low", 1: "Medium", 2: "High"}
df["congestion_name"] = df["congestion_label"].map(label_names)

print(f"  Low cut:  {low_cut:.0f} vehicles/hr")
print(f"  High cut: {high_cut:.0f} vehicles/hr")
print(df["congestion_name"].value_counts())

# ── 5. Handle missing & save ──────────────────────────────────────────────────
print("[5/6] Saving clean dataset...")
df.fillna(0, inplace=True)   # FIXED (was dropna)
df.to_csv("clean_traffic.csv", index=False)
print(f"  Saved clean_traffic.csv  ({len(df)} rows)")

# ── 6. EDA plots ──────────────────────────────────────────────────────────────
print("[6/6] Generating EDA charts...")

# Traffic volume by hour
hourly = df.groupby("hour")["traffic_volume"].mean()
plt.figure(figsize=(10, 4))
plt.bar(hourly.index, hourly.values, color="steelblue", edgecolor="white")
plt.axvspan(7, 9, alpha=0.15, color="red", label="Morning rush")
plt.axvspan(16, 18, alpha=0.15, color="orange", label="Evening rush")
plt.xlabel("Hour of day"); plt.ylabel("Avg vehicles/hr")
plt.title("Average Traffic Volume by Hour")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/traffic_by_hour.png", dpi=120)
plt.close()

# Traffic volume by day of week
day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
daily = df.groupby("day")["traffic_volume"].mean()
plt.figure(figsize=(7, 4))
colors = ["#e74c3c" if d >= 5 else "steelblue" for d in daily.index]
plt.bar([day_labels[d] for d in daily.index], daily.values, color=colors)
plt.xlabel("Day of week"); plt.ylabel("Avg vehicles/hr")
plt.title("Traffic Volume by Day (red = weekend)")
plt.tight_layout()
plt.savefig("outputs/traffic_by_day.png", dpi=120)
plt.close()

# Congestion distribution (FIXED)
plt.figure(figsize=(5, 4))
counts = df["congestion_name"].value_counts().reindex(["Low","Medium","High"], fill_value=0)
counts.plot(
    kind="bar",
    color=["#2ecc71","#f39c12","#e74c3c"],
    edgecolor="white"
)
plt.title("Congestion Class Distribution")
plt.ylabel("Count"); plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/congestion_distribution.png", dpi=120)
plt.close()

# Heatmap
pivot = df.pivot_table(
    values="traffic_volume", index="hour", columns="day", aggfunc="mean"
)
pivot.columns = day_labels
plt.figure(figsize=(9, 6))
sns.heatmap(pivot, cmap="RdYlGn_r", linewidths=0.3, annot=False)
plt.title("Traffic Volume Heatmap (Hour vs Day)")
plt.tight_layout()
plt.savefig("outputs/traffic_heatmap.png", dpi=120)
plt.close()

print("\n✅ Phase 1 complete. Charts saved to outputs/")
print("   Next → python phase2_train.py")
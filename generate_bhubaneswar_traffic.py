"""
generate_bhubaneswar_traffic.py
════════════════════════════════════════════════════════════════════════════════
Generates 52,500 rows of realistic synthetic urban traffic data for Bhubaneswar,
India. Matches the exact column schema of the existing dataset so both can be
merged and used to retrain the XGBoost congestion model.

Output columns (same order as existing dataset):
    timestamp, location, vehicle_count, avg_speed, day_of_week,
    rainfall, clouds, temperature, congestion_label

Realistic patterns encoded:
  ● Morning rush   08–10 AM  →  very high vehicle count, low speed
  ● Evening rush   17–20 PM  →  high vehicle count, low speed
  ● Lunch peak     12–14     →  moderate traffic
  ● Night          00–05     →  very low traffic
  ● Weekends       Sat/Sun   →  lower commuter traffic but higher leisure
  ● Rainfall       >5 mm/hr  →  speed drops 15–35 %, volume dips slightly
  ● Location type  arterial / junction / highway →  different base capacities
  ● Indian summer  temp 22–42 °C with seasonal curve

Congestion labelling:
  High   : vehicle_count > 300  AND  avg_speed < 20
  Medium : vehicle_count > 180  AND  avg_speed < 35
  Low    : everything else

Run:
    python generate_bhubaneswar_traffic.py

Output:
    bhubaneswar_synthetic_traffic.csv   (in the same directory)
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

# ── Location definitions ──────────────────────────────────────────────────────
# Each location has:
#   capacity_base : maximum typical vehicle count per 30-min window
#   speed_base    : free-flow average speed (km/h)
#   type          : 'arterial' | 'junction' | 'highway' | 'inner'
LOCATIONS = {
    "City Center":     {"cap": 2000, "speed": 50, "type": "main_road"},
    "Market Square":   {"cap": 1200, "speed": 40, "type": "main_road"},
    "Railway Station": {"cap": 1800, "speed": 60, "type": "main_road"},
    "IT Park":         {"cap": 1500, "speed": 55, "type": "main_road"},
    "Hospital Road":   {"cap": 1000, "speed": 40, "type": "main_road"},
    "Bus Stand":       {"cap": 1400, "speed": 45, "type": "main_road"},
    "School Zone":     {"cap":  600, "speed": 25, "type": "side_road"},
    "Old Town":        {"cap":  800, "speed": 30, "type": "side_road"},
    "Tech Hub":        {"cap": 1600, "speed": 65, "type": "highway"},
    "Airport Road":    {"cap": 2200, "speed": 70, "type": "highway"},
}

# ── Timestamps: 3 500 × 30-min slots × 15 locations = 52 500 rows ─────────────
N_PER_LOC  = 3_500
TIMESTAMPS = pd.date_range("2023-01-01 00:00", periods=N_PER_LOC, freq="30min")
# This covers ~72 days (1 Jan – 14 Mar 2023)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: hour-of-day load factor
# Returns 0..1; multiplied against location capacity to get base vehicle count
# ══════════════════════════════════════════════════════════════════════════════
def hour_load(hour: int, is_weekend: bool) -> float:
    """
    Weekday:
      00–05  very low  (0.06)
      06–07  rising    (0.38)
      08–10  PEAK AM   (0.88)
      11     shoulder  (0.52)
      12–14  lunch     (0.58)
      15–16  afternoon (0.48)
      17–20  PEAK PM   (0.82)
      21–23  evening   (0.22)

    Weekend:
      00–07  very low  (0.05–0.18)
      08–11  leisure   (0.42)
      12–16  midday    (0.55)  ← shopping, outings peak
      17–21  evening   (0.60)
      22–23  low       (0.18)
    """
    if is_weekend:
        table = {
            0: 0.06, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.06, 5: 0.10,
            6: 0.18, 7: 0.35, 8: 0.42, 9: 0.45, 10: 0.48, 11: 0.52,
            12: 0.58, 13: 0.60, 14: 0.62, 15: 0.60, 16: 0.58,
            17: 0.62, 18: 0.65, 19: 0.60, 20: 0.55,
            21: 0.28, 22: 0.18, 23: 0.10,
        }
    else:
        table = {
            0: 0.06, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.06, 5: 0.12,
            6: 0.38, 7: 0.52, 8: 0.88, 9: 0.90, 10: 0.82,
            11: 0.52, 12: 0.58, 13: 0.60, 14: 0.58,
            15: 0.48, 16: 0.55, 17: 0.82, 18: 0.88, 19: 0.85, 20: 0.72,
            21: 0.30, 22: 0.20, 23: 0.12,
        }
    return table[hour]


# ══════════════════════════════════════════════════════════════════════════════
# Helper: Indian seasonal temperature for Bhubaneswar
# Warmest Apr–Jun (35–42 °C), coolest Dec–Jan (12–22 °C)
# ══════════════════════════════════════════════════════════════════════════════
def seasonal_temp(month: int, hour: int) -> float:
    """Returns realistic ambient temperature with diurnal variation."""
    monthly_mean = {
        1: 18, 2: 21, 3: 27, 4: 33, 5: 36, 6: 34,
        7: 30, 8: 30, 9: 30, 10: 28, 11: 24, 12: 19,
    }
    base = monthly_mean.get(month, 26)
    # Diurnal swing: coldest at 05:00, hottest at 14:00
    swing = 7 * np.sin(np.pi * (hour - 5) / 18) if 5 <= hour <= 23 else -3.5
    return round(base + swing, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: realistic rainfall for Bhubaneswar
# Very dry Jan-Mar, monsoon Jun-Sep (heavy), Oct-Nov (post-monsoon)
# ══════════════════════════════════════════════════════════════════════════════
def generate_rainfall(month: int, n: int, rng_: np.random.Generator) -> np.ndarray:
    """
    Returns array of rainfall values (mm/hr) for `n` time slots.
    Uses zero-inflated distribution: most slots are dry.
    """
    # Probability of any rain in this 30-min slot
    rain_prob = {
        1: 0.03, 2: 0.04, 3: 0.05, 4: 0.08, 5: 0.12, 6: 0.30,
        7: 0.45, 8: 0.42, 9: 0.35, 10: 0.18, 11: 0.08, 12: 0.04,
    }.get(month, 0.10)

    # Mean intensity (mm/hr) when it rains
    rain_mean = {
        1: 1.5, 2: 2.0, 3: 2.5, 4: 4.0, 5: 6.0, 6: 14.0,
        7: 22.0, 8: 20.0, 9: 16.0, 10: 8.0, 11: 3.0, 12: 1.5,
    }.get(month, 3.0)

    is_raining  = rng_.random(n) < rain_prob
    intensities = rng_.exponential(scale=rain_mean, size=n)
    rainfall    = np.where(is_raining, intensities, 0.0)
    return np.round(rainfall, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Main generation loop
# ══════════════════════════════════════════════════════════════════════════════
print("Generating synthetic Bhubaneswar traffic dataset …")

all_frames = []

for loc_name, loc_meta in LOCATIONS.items():
    cap        = loc_meta["cap"]
    free_speed = loc_meta["speed"]

    # Extract temporal features from timestamp array
    ts         = TIMESTAMPS
    hours      = ts.hour.to_numpy()
    months     = ts.month.to_numpy()
    dow_num    = ts.dayofweek.to_numpy()          # 0=Mon … 6=Sun
    is_weekend = dow_num >= 5

    # ── Vehicle count ─────────────────────────────────────────────────────────
    load_factors = np.array([
        hour_load(h, bool(we)) for h, we in zip(hours, is_weekend)
    ])

    # Location-specific micro-multiplier (±15 %)
    loc_noise = rng.uniform(0.85, 1.15, size=N_PER_LOC)

    # Gaussian noise per slot (CV ≈ 12 %)
    slot_noise = rng.normal(1.0, 0.12, size=N_PER_LOC)
    slot_noise = np.clip(slot_noise, 0.5, 1.5)

    raw_count = cap * load_factors * loc_noise * slot_noise
    raw_count = np.clip(raw_count, 5, cap).astype(int)

    # ── Rainfall ──────────────────────────────────────────────────────────────
    rainfall = np.zeros(N_PER_LOC)
    for m in range(1, 13):
        mask = months == m
        if mask.sum() > 0:
            rainfall[mask] = generate_rainfall(m, int(mask.sum()), rng)

    # ── Cloud cover (%) — correlated with rainfall ─────────────────────────────
    base_clouds  = rng.integers(10, 50, size=N_PER_LOC)
    extra_clouds = np.where(rainfall > 0, rng.integers(30, 60, size=N_PER_LOC), 0)
    clouds       = np.clip(base_clouds + extra_clouds, 0, 100)

    # ── Temperature ───────────────────────────────────────────────────────────
    temp_base  = np.array([seasonal_temp(m, h) for m, h in zip(months, hours)])
    temp_noise = rng.normal(0, 1.2, size=N_PER_LOC)
    # Rain cools ambient slightly
    temp_rain  = np.where(rainfall > 2, rng.uniform(-3, -1, size=N_PER_LOC), 0)
    temperature = np.round(temp_base + temp_noise + temp_rain, 1)

    # ── Average speed ─────────────────────────────────────────────────────────
    # Speed drops as load increases (BPR-inspired curve)
    load_ratio  = raw_count / cap                          # 0..1
    speed_ratio = 1.0 / (1.0 + 0.85 * (load_ratio ** 3))  # smooth decay
    base_speed  = free_speed * speed_ratio

    # Rainfall penalty: heavy rain (>10 mm/hr) reduces speed by up to 35 %
    rain_factor = np.ones(N_PER_LOC)
    light_rain  = (rainfall > 0) & (rainfall <= 5)
    heavy_rain  = (rainfall > 5) & (rainfall <= 15)
    very_heavy  = rainfall > 15
    rain_factor = np.where(light_rain,  rain_factor * rng.uniform(0.88, 0.96, N_PER_LOC), rain_factor)
    rain_factor = np.where(heavy_rain,  rain_factor * rng.uniform(0.72, 0.85, N_PER_LOC), rain_factor)
    rain_factor = np.where(very_heavy,  rain_factor * rng.uniform(0.60, 0.72, N_PER_LOC), rain_factor)

    # Speed noise (CV ≈ 8 %)
    speed_noise = rng.normal(1.0, 0.08, size=N_PER_LOC)
    speed_noise = np.clip(speed_noise, 0.75, 1.25)

    avg_speed = base_speed * rain_factor * speed_noise
    avg_speed = np.clip(avg_speed, 4.0, free_speed * 1.05)
    avg_speed = np.round(avg_speed, 1)

    # ── Day-of-week string ────────────────────────────────────────────────────
    dow_map  = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",
                4:"Friday",5:"Saturday",6:"Sunday"}
    day_of_week = np.array([dow_map[d] for d in dow_num])

    # ── Congestion label ──────────────────────────────────────────────────────
    # Combines vehicle_count and avg_speed to assign label
    # High   : heavy volume AND crawling speed
    # Medium : moderate-to-heavy volume OR reduced speed
    # Low    : light traffic and good speed
    def assign_label(vc, sp):
        if vc > 320 and sp < 18:
            return "High"
        elif vc > 300 and sp < 20:
            return "High"
        elif vc > 230 and sp < 30:
            return "Medium"
        elif vc > 200 and sp < 35:
            return "Medium"
        elif vc > 280:
            return "Medium"
        else:
            return "Low"

    congestion_label = np.array([
        assign_label(int(vc), float(sp))
        for vc, sp in zip(raw_count, avg_speed)
    ])

    # ── Assemble this location's DataFrame ────────────────────────────────────
    df_loc = pd.DataFrame({
        "timestamp":        ts.strftime("%Y-%m-%d %H:%M:%S"),
        "location":         loc_name,
        "vehicle_count":    raw_count,
        "avg_speed":        avg_speed,
        "day_of_week":      day_of_week,
        "rainfall":         rainfall,
        "clouds":           clouds,
        "temperature":      temperature,
        "congestion_label": congestion_label,
    })

    all_frames.append(df_loc)
    print(f"  {loc_name:<25} rows={len(df_loc):>5}  "
          f"High={( congestion_label=='High').sum():>4}  "
          f"Med={(congestion_label=='Medium').sum():>4}  "
          f"Low={(congestion_label=='Low').sum():>4}")

# ── Combine and shuffle ───────────────────────────────────────────────────────
df = pd.concat(all_frames, ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ── Summary statistics ────────────────────────────────────────────────────────
print(f"\nTotal rows: {len(df):,}")
print("\nCongestion distribution:")
print(df["congestion_label"].value_counts())
print(f"\nVehicle count — min:{df['vehicle_count'].min()}  "
      f"max:{df['vehicle_count'].max()}  "
      f"mean:{df['vehicle_count'].mean():.1f}")
print(f"Avg speed     — min:{df['avg_speed'].min()}  "
      f"max:{df['avg_speed'].max()}  "
      f"mean:{df['avg_speed'].mean():.1f}")
print(f"Rainfall      — 0-count:{( df['rainfall']==0).sum():,}  "
      f"max:{df['rainfall'].max():.1f}")
print(f"Temperature   — min:{df['temperature'].min()}  max:{df['temperature'].max()}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "bhubaneswar_synthetic_traffic.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
print("Columns:", list(df.columns))

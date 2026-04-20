"""
merge_and_retrain.py
════════════════════════════════════════════════════════════════════════════════
Step 2 of the workflow:
  1. Load your existing dataset (clean_traffic.csv or raw CSV)
  2. Load the new synthetic Bhubaneswar dataset
  3. Align columns, resolve conflicts, merge
  4. Engineer features
  5. Retrain XGBoost (and compare with Random Forest)
  6. Save updated model to models/best_model.pkl

Usage:
    python merge_and_retrain.py \
        --existing  clean_traffic.csv \
        --synthetic bhubaneswar_synthetic_traffic.csv \
        --out       merged_traffic.csv

All arguments are optional — defaults match the project folder structure.
════════════════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Merge + Retrain traffic model")
parser.add_argument("--existing",  default="clean_traffic.csv",
                    help="Path to existing clean dataset")
parser.add_argument("--synthetic", default="bhubaneswar_synthetic_traffic.csv",
                    help="Path to synthetic Bhubaneswar dataset")
parser.add_argument("--out",       default="merged_traffic.csv",
                    help="Output path for merged CSV")
parser.add_argument("--models-dir", default="models",
                    help="Folder for saving model artefacts")
parser.add_argument("--outputs-dir", default="outputs",
                    help="Folder for saving charts")
args = parser.parse_args()

os.makedirs(args.models_dir,  exist_ok=True)
os.makedirs(args.outputs_dir, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load datasets
# ══════════════════════════════════════════════════════════════════════════════
print("[1/7] Loading datasets …")

# Load synthetic (always present when this script runs)
df_syn = pd.read_csv(args.synthetic)
print(f"  Synthetic  : {len(df_syn):>7,} rows  cols={list(df_syn.columns)}")

# Load existing (skip gracefully if missing)
if os.path.exists(args.existing):
    df_old = pd.read_csv(args.existing)
    print(f"  Existing   : {len(df_old):>7,} rows  cols={list(df_old.columns)}")
    have_existing = True
else:
    print(f"  Existing dataset not found at '{args.existing}' — using synthetic only.")
    df_old = pd.DataFrame()
    have_existing = False

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Harmonise columns
# ══════════════════════════════════════════════════════════════════════════════
print("[2/7] Harmonising columns …")

# Required output columns (superset that covers both schemas)
REQUIRED = [
    "timestamp", "location", "vehicle_count", "avg_speed",
    "day_of_week", "rainfall", "clouds", "temperature", "congestion_label",
]

def harmonise(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Align any incoming DataFrame to REQUIRED columns."""
    df = df.copy()

    # Rename common variants
    rename_map = {
        "traffic_volume": "vehicle_count",
        "speed":          "avg_speed",
        "rain_1h":        "rainfall",
        "clouds_all":     "clouds",
        "temp_celsius":   "temperature",
        "date_time":      "timestamp",
        "congestion_name":"congestion_label",
    }
    df.rename(columns=rename_map, inplace=True)

    # Derive congestion_label if absent (from existing Metro dataset)
    if "congestion_label" not in df.columns:
        if "vehicle_count" in df.columns and "avg_speed" in df.columns:
            vc_lo  = df["vehicle_count"].quantile(0.33)
            vc_hi  = df["vehicle_count"].quantile(0.67)
            def label(row):
                if row["vehicle_count"] >= vc_hi and row["avg_speed"] < 25:
                    return "High"
                elif row["vehicle_count"] >= vc_lo:
                    return "Medium"
                return "Low"
            df["congestion_label"] = df.apply(label, axis=1)
            print(f"    ({tag}) derived congestion_label from vehicle_count + avg_speed")
        else:
            df["congestion_label"] = "Low"

    # Derive location if absent
    if "location" not in df.columns:
        df["location"] = "Unknown"

    # Ensure all required columns exist (fill with NaN if truly missing)
    for col in REQUIRED:
        if col not in df.columns:
            df[col] = np.nan

    df["_source"] = tag
    return df[REQUIRED + ["_source"]]


df_syn_h = harmonise(df_syn, "synthetic")
print(f"    Synthetic harmonised: {len(df_syn_h):,} rows")

if have_existing:
    df_old_h = harmonise(df_old, "existing")
    print(f"    Existing harmonised: {len(df_old_h):,} rows")
    #  FIX duplicate columns before merging
    df_old_h = df_old_h.loc[:, ~df_old_h.columns.duplicated()]
    df_syn_h = df_syn_h.loc[:, ~df_syn_h.columns.duplicated()]

    #  Merge safely
    df_merged = pd.concat([df_old_h, df_syn_h], ignore_index=True)
else:
    df_merged = df_syn_h.copy()

print(f"  Merged total: {len(df_merged):,} rows")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Parse timestamp + feature engineering
# ══════════════════════════════════════════════════════════════════════════════
print("[3/7] Feature engineering …")

df_merged["timestamp"] = pd.to_datetime(df_merged["timestamp"], errors="coerce")
df_merged.dropna(subset=["timestamp"], inplace=True)

df_merged["hour"]         = df_merged["timestamp"].dt.hour
df_merged["day"]          = df_merged["timestamp"].dt.dayofweek   # 0=Mon
df_merged["month"]        = df_merged["timestamp"].dt.month
df_merged["is_weekend"]   = (df_merged["day"] >= 5).astype(int)
df_merged["is_rush_hour"] = df_merged["hour"].apply(
    lambda h: 1 if (8 <= h <= 10) or (17 <= h <= 20) else 0
)
df_merged["is_night"]     = df_merged["hour"].apply(
    lambda h: 1 if (h >= 22 or h <= 5) else 0
)

# Encode location as integer category
loc_le = LabelEncoder()
df_merged["location_enc"] = loc_le.fit_transform(
    df_merged["location"].fillna("Unknown")
)

# Numeric coercions (safe)
for col in ["vehicle_count", "avg_speed", "rainfall", "clouds", "temperature"]:
    df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce")

df_merged.dropna(subset=["vehicle_count", "avg_speed", "congestion_label"], inplace=True)
df_merged.reset_index(drop=True, inplace=True)

# ── Label encoding for XGBoost ────────────────────────────────────────────────
LABEL_MAP  = {"Low": 0, "Medium": 1, "High": 2}
LABEL_RMAP = {0: "Low", 1: "Medium", 2: "High"}
df_merged["label_int"] = df_merged["congestion_label"].map(LABEL_MAP)
df_merged.dropna(subset=["label_int"], inplace=True)
df_merged["label_int"] = df_merged["label_int"].astype(int)

print(f"  Final rows after cleaning: {len(df_merged):,}")
print("  Label distribution:")
print(df_merged["congestion_label"].value_counts().to_string())

# Save merged CSV
df_merged.drop(columns=["_source", "label_int"], errors="ignore").to_csv(
    args.out, index=False
)
print(f"  Merged CSV saved → {args.out}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Train / test split
# ══════════════════════════════════════════════════════════════════════════════
print("[4/7] Splitting data (80/20 stratified) …")

FEATURES = [
    "hour", "day", "month", "is_weekend", "is_rush_hour", "is_night",
    "vehicle_count", "avg_speed", "rainfall", "clouds", "temperature",
    "location_enc",
]
# Keep only features that exist and are fully numeric
available = [f for f in FEATURES if f in df_merged.columns]
X = df_merged[available].fillna(0)
y = df_merged["label_int"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}  Features: {available}")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Train models
# ══════════════════════════════════════════════════════════════════════════════
print("[5/7] Training models …")

# ── Random Forest ─────────────────────────────────────────────────────────────
print("  Random Forest …")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=16, min_samples_leaf=4,
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
rf_f1   = f1_score(y_test, rf_pred, average="weighted")
print(f"    RF  → Accuracy {rf_acc:.4f}  F1 {rf_f1:.4f}")

# ── XGBoost ───────────────────────────────────────────────────────────────────
print("  XGBoost …")
xgb = XGBClassifier(
    n_estimators=400, max_depth=7, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85,
    eval_metric="mlogloss", random_state=42, n_jobs=-1,
    verbosity=0,
)
xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
xgb_pred = xgb.predict(X_test)
xgb_acc  = accuracy_score(y_test, xgb_pred)
xgb_f1   = f1_score(y_test, xgb_pred, average="weighted")
print(f"    XGB → Accuracy {xgb_acc:.4f}  F1 {xgb_f1:.4f}")

# Pick best
best_model  = xgb if xgb_acc >= rf_acc else rf
best_name   = "XGBoost" if xgb_acc >= rf_acc else "RandomForest"
best_pred   = xgb_pred if xgb_acc >= rf_acc else rf_pred
print(f"  Best model: {best_name}")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  Evaluate + save artefacts
# ══════════════════════════════════════════════════════════════════════════════
print("[6/7] Saving artefacts …")

# Classification report
target_names = [LABEL_RMAP[i] for i in sorted(LABEL_RMAP)]
print("\n  Classification report ({best_name}):".format(best_name=best_name))
print(classification_report(y_test, best_pred, target_names=target_names))

# Confusion matrix plot
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

cm = confusion_matrix(y_test, best_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_name} (merged dataset)")
plt.tight_layout()
cm_path = os.path.join(args.outputs_dir, "confusion_matrix_merged.png")
plt.savefig(cm_path, dpi=120); plt.close()
print(f"  Confusion matrix → {cm_path}")

# Feature importance plot
fi_vals = (best_model.feature_importances_
           if hasattr(best_model, "feature_importances_") else None)
if fi_vals is not None:
    fi = pd.Series(fi_vals, index=available).sort_values(ascending=True)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if v == fi.max() else "steelblue" for v in fi.values]
    fi.plot(kind="barh", color=colors, edgecolor="white", ax=ax2)
    ax2.set_title(f"Feature Importance — {best_name} (merged dataset)")
    ax2.set_xlabel("Importance score")
    plt.tight_layout()
    fi_path = os.path.join(args.outputs_dir, "feature_importance_merged.png")
    plt.savefig(fi_path, dpi=120); plt.close()
    print(f"  Feature importance → {fi_path}")

# Model comparison bar chart
fig3, ax3 = plt.subplots(figsize=(5, 4))
models = ["Random Forest", "XGBoost"]
accs   = [rf_acc, xgb_acc]
f1s    = [rf_f1,  xgb_f1]
x = np.arange(2)
ax3.bar(x - 0.18, accs, 0.32, label="Accuracy", color="steelblue")
ax3.bar(x + 0.18, f1s,  0.32, label="F1 Score",  color="coral")
ax3.set_xticks(x); ax3.set_xticklabels(models)
ax3.set_ylim(0.70, 1.0); ax3.set_title("Model Comparison (merged dataset)")
ax3.legend(); plt.tight_layout()
mc_path = os.path.join(args.outputs_dir, "model_comparison_merged.png")
plt.savefig(mc_path, dpi=120); plt.close()
print(f"  Model comparison  → {mc_path}")

# Save model + feature list
model_path   = os.path.join(args.models_dir, "best_model.pkl")
feature_path = os.path.join(args.models_dir, "feature_cols.pkl")
joblib.dump(best_model, model_path)
joblib.dump(available,  feature_path)
print(f"  Model saved       → {model_path}")
print(f"  Feature list      → {feature_path}")

# Save location encoder for dashboard use
enc_path = os.path.join(args.models_dir, "location_encoder.pkl")
joblib.dump(loc_le, enc_path)
print(f"  Location encoder  → {enc_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/7] Summary")
print("=" * 55)
print(f"  Total training rows : {len(X_train):,}")
print(f"  Test rows           : {len(X_test):,}")
print(f"  Features used       : {available}")
print(f"  Random Forest       : Acc={rf_acc:.4f}  F1={rf_f1:.4f}")
print(f"  XGBoost             : Acc={xgb_acc:.4f}  F1={xgb_f1:.4f}")
print(f"  Best model          : {best_name}")
print(f"  Merged CSV          : {args.out}")
print(f"  Model artefact      : {model_path}")
print("=" * 55)
print("\nDone. Restart the Streamlit dashboard to use the new model.")

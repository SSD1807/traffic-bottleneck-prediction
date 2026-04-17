"""
Phase 2 — Model Training & Evaluation
Traffic Bottleneck / Congestion Prediction Project
Run: python phase2_train.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── 1. Load clean data ────────────────────────────────────────────────────────
print("[1/7] Loading clean_traffic.csv...")
df = pd.read_csv("clean_traffic.csv")

FEATURES = [
    "hour", "day", "month", "is_weekend", "is_rush_hour", "is_night",
    "temp_celsius", "rain_1h", "snow_1h", "clouds_all", "weather_code",
    "traffic_volume"   # 🔥 ADDED (IMPORTANT)
]

TARGET = "congestion_label"

X = df[FEATURES]
y = df[TARGET]
print(f"  X shape: {X.shape}  |  Class counts:\n{y.value_counts().sort_index()}")

# ── 2. Train / test split ─────────────────────────────────────────────────────
print("[2/7] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Random Forest ──────────────────────────────────────────────────────────
print("[3/7] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
rf_f1   = f1_score(y_test, rf_pred, average="weighted")
print(f"  Random Forest  → Accuracy: {rf_acc:.4f}  F1: {rf_f1:.4f}")

# ── 4. XGBoost ────────────────────────────────────────────────────────────────
print("[4/7] Training XGBoost...")

xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    eval_metric="mlogloss",   # 🔥 FIXED (removed use_label_encoder)
    random_state=42, n_jobs=-1
)

xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_pred = xgb.predict(X_test)
xgb_acc  = accuracy_score(y_test, xgb_pred)
xgb_f1   = f1_score(y_test, xgb_pred, average="weighted")
print(f"  XGBoost        → Accuracy: {xgb_acc:.4f}  F1: {xgb_f1:.4f}")

# ── 5. Cross-validation (NEW, small addition) ─────────────────────────────────
print("[5/7] Cross-validation (XGBoost)...")
cv_scores = cross_val_score(xgb, X, y, cv=5, scoring="accuracy")
print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 6. Best model → save ──────────────────────────────────────────────────────
print("[6/7] Saving best model...")

best = xgb if xgb_acc >= rf_acc else rf   # 🔥 IMPROVED
model_name = "XGBoost" if xgb_acc >= rf_acc else "RandomForest"

joblib.dump(best, "models/best_model.pkl")
joblib.dump(FEATURES, "models/feature_cols.pkl")
print(f"  Saved → models/best_model.pkl ({model_name})")

# ── 7. Classification report ──────────────────────────────────────────────────
print("\n[7/7] Classification report:")

label_names = {0: "Low", 1: "Medium", 2: "High"}
print(classification_report(
    y_test, xgb_pred,
    target_names=[label_names[i] for i in sorted(label_names)]
))

# Confusion matrix
cm = confusion_matrix(y_test, xgb_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Low","Medium","High"],
    yticklabels=["Low","Medium","High"]
)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix — XGBoost")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=120)
plt.close()

# ── Feature importance ─────────────────────────────────────────────────────────
print("Generating feature importance chart...")

importances = best.feature_importances_
fi = pd.Series(importances, index=FEATURES).sort_values(ascending=True)

max_feature = fi.idxmax()
colors = ["#e74c3c" if f == max_feature else "steelblue" for f in fi.index]

plt.figure(figsize=(8, 5))
fi.plot(kind="barh", color=colors, edgecolor="white")
plt.xlabel("Feature importance score")
plt.title(f"{model_name} Feature Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=120)
plt.close()

# Model comparison chart (UNCHANGED)
fig, ax = plt.subplots(figsize=(6, 4))
models = ["Random Forest", "XGBoost"]
accs   = [rf_acc, xgb_acc]
f1s    = [rf_f1,  xgb_f1]

x = np.arange(2)
ax.bar(x - 0.2, accs, 0.35, label="Accuracy", color="steelblue")
ax.bar(x + 0.2, f1s,  0.35, label="F1 Score", color="coral")
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylim(0.75, 1.0)
ax.set_title("Model Comparison")
ax.legend(); plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=120)
plt.close()

print("\n✅ Phase 2 complete.")
print(f"   Best model: {model_name}  Accuracy={xgb_acc:.4f}  F1={xgb_f1:.4f}")
print("   Next → python phase3_graph_simulate.py")
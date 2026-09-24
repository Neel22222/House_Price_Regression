"""
train_model.py
--------------
Trains a GradientBoostingRegressor on the house price dataset,
evaluates it, and saves the model + feature metadata to ./model/.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "house_price_regression_dataset.csv"
MODEL_DIR  = "model"
PLOTS_DIR  = "plots"
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(df.describe())

FEATURES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]
TARGET = "House_Price"

X = df[FEATURES]
y = df[TARGET]

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ── Compare three models (no scaling needed for tree models) ──────────────────
models = {
    "LinearRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ]),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=5, random_state=RANDOM_STATE
    ),
}

results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"{name:25s}  MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.4f}")

# ── Pick best model (highest R²) ──────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["R2"])
best_model = models[best_name]
print(f"\nBest model: {best_name}")

# ── Save artefacts ────────────────────────────────────────────────────────────
joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))

meta = {
    "model_name": best_name,
    "features": FEATURES,
    "metrics": results[best_name],
    "feature_stats": {
        col: {
            "min":  float(df[col].min()),
            "max":  float(df[col].max()),
            "mean": float(df[col].mean()),
        }
        for col in FEATURES
    },
    "all_model_metrics": results,
}
with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"Model saved -> {MODEL_DIR}/model.pkl")
print(f"Meta  saved -> {MODEL_DIR}/meta.json")

# ── Plot: Feature Importance ──────────────────────────────────────────────────
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
else:
    importances = best_model.named_steps["model"].coef_

fig, ax = plt.subplots(figsize=(8, 5))
sorted_idx = np.argsort(importances)[::-1]
sns.barplot(x=[importances[i] for i in sorted_idx],
            y=[FEATURES[i] for i in sorted_idx], ax=ax, palette="viridis")
ax.set_title(f"Feature Importances — {best_name}")
ax.set_xlabel("Importance")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
plt.close(fig)
print(f"Plot saved -> {PLOTS_DIR}/feature_importance.png")

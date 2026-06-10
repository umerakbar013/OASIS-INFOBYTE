# ============================================================
#  Car Price Prediction with Machine Learning
#  OASIS INFOBYTE — Task 3
#  Author : Umer Akbar
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "car_data.csv")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Style ────────────────────────────────────────────────
ORANGE = "#F5A623"
DARK   = "#1C1C2E"
LIGHT  = "#F4F4F4"
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor":   "#2A2A3E",
    "axes.labelcolor":  LIGHT,
    "xtick.color":      LIGHT,
    "ytick.color":      LIGHT,
    "text.color":       LIGHT,
    "grid.color":       "#3A3A5C",
})

# ════════════════════════════════════════════════════════════
#  1 — LOAD & INSPECT
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("  CAR PRICE PREDICTION — OASIS INFOBYTE Task 3")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"\n[1] Dataset loaded  →  {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nBasic Statistics:\n", df.describe())

# ════════════════════════════════════════════════════════════
#  2 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════
print("\n[2] Feature Engineering …")

# Car age is more informative than raw year
df["Car_Age"] = 2024 - df["Year"]

# Drop raw Year and Car_Name (too many unique values, not useful as-is)
df.drop(columns=["Car_Name", "Year"], inplace=True)

# Encode categoricals
le = LabelEncoder()
for col in ["Fuel_Type", "Selling_type", "Transmission"]:
    df[col] = le.fit_transform(df[col])

print("Engineered features:", df.columns.tolist())
print(df.head())

# ════════════════════════════════════════════════════════════
#  3 — EDA VISUALIZATIONS
# ════════════════════════════════════════════════════════════
print("\n[3] Generating EDA plots …")

# ── 3a: Distribution of Selling Price ──────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["Selling_Price"], bins=30, color=ORANGE, edgecolor=DARK, alpha=0.9)
ax.set_title("Distribution of Selling Price", fontsize=14, fontweight="bold", color=ORANGE)
ax.set_xlabel("Selling Price (Lakhs ₹)")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_selling_price_distribution.png"), dpi=150)
plt.close()
print("   ✔ 01_selling_price_distribution.png")

# ── 3b: Correlation Heatmap ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="YlOrBr",
            ax=ax, linewidths=0.5, linecolor=DARK,
            annot_kws={"size": 9, "color": DARK})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", color=ORANGE)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_correlation_heatmap.png"), dpi=150)
plt.close()
print("   ✔ 02_correlation_heatmap.png")

# ── 3c: Selling Price vs Present Price ──────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(df["Present_Price"], df["Selling_Price"],
                     c=df["Car_Age"], cmap="YlOrBr", alpha=0.8,
                     edgecolors=DARK, linewidths=0.4, s=60)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Car Age (years)", color=LIGHT)
cbar.ax.yaxis.set_tick_params(color=LIGHT)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=LIGHT)
ax.set_title("Present Price vs Selling Price (coloured by Age)",
             fontsize=13, fontweight="bold", color=ORANGE)
ax.set_xlabel("Present Price (Lakhs ₹)")
ax.set_ylabel("Selling Price (Lakhs ₹)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_price_scatter.png"), dpi=150)
plt.close()
print("   ✔ 03_price_scatter.png")

# ── 3d: Fuel Type vs Selling Price ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
# Re-read original for readable labels
df_orig = pd.read_csv(DATA_PATH)
sns.boxplot(data=df_orig, x="Fuel_Type", y="Selling_Price",
            palette={"Petrol": ORANGE, "Diesel": "#4FC3F7", "CNG": "#A5D6A7"},
            linewidth=1.2, ax=ax)
ax.set_title("Selling Price by Fuel Type", fontsize=14, fontweight="bold", color=ORANGE)
ax.set_xlabel("Fuel Type")
ax.set_ylabel("Selling Price (Lakhs ₹)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_fuel_type_boxplot.png"), dpi=150)
plt.close()
print("   ✔ 04_fuel_type_boxplot.png")

# ── 3e: Car Age vs Selling Price ────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(df["Car_Age"], df["Selling_Price"],
           color=ORANGE, alpha=0.6, edgecolors=DARK, linewidths=0.4)
ax.set_title("Car Age vs Selling Price", fontsize=14, fontweight="bold", color=ORANGE)
ax.set_xlabel("Car Age (years)")
ax.set_ylabel("Selling Price (Lakhs ₹)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_age_vs_price.png"), dpi=150)
plt.close()
print("   ✔ 05_age_vs_price.png")

# ════════════════════════════════════════════════════════════
#  4 — TRAIN / TEST SPLIT
# ════════════════════════════════════════════════════════════
print("\n[4] Splitting dataset …")
X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

print(f"   Train samples : {X_train.shape[0]}")
print(f"   Test  samples : {X_test.shape[0]}")

# ════════════════════════════════════════════════════════════
#  5 — MODEL TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════
print("\n[5] Training models …")

models = {
    "Linear Regression":        LinearRegression(),
    "Random Forest":             RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting":         GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv, "model": model, "y_pred": y_pred}

    print(f"\n  ── {name}")
    print(f"     MAE  : {mae:.4f}")
    print(f"     RMSE : {rmse:.4f}")
    print(f"     R²   : {r2:.4f}")
    print(f"     CV R²: {cv:.4f}")

# ════════════════════════════════════════════════════════════
#  6 — MODEL COMPARISON PLOT
# ════════════════════════════════════════════════════════════
print("\n[6] Generating model comparison plots …")

metrics = ["MAE", "RMSE", "R2"]
model_names = list(results.keys())
colors_bar = [ORANGE, "#4FC3F7", "#A5D6A7"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold", color=ORANGE)

for i, metric in enumerate(metrics):
    vals = [results[m][metric] for m in model_names]
    bars = axes[i].bar(model_names, vals, color=colors_bar, edgecolor=DARK, linewidth=0.8)
    axes[i].set_title(metric, fontsize=12, color=LIGHT)
    axes[i].set_xticklabels(model_names, rotation=15, ha="right", fontsize=8)
    for bar, val in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(vals) * 0.02,
                     f"{val:.3f}", ha="center", va="bottom",
                     fontsize=8, color=LIGHT)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_model_comparison.png"), dpi=150)
plt.close()
print("   ✔ 06_model_comparison.png")

# ════════════════════════════════════════════════════════════
#  7 — BEST MODEL: ACTUAL vs PREDICTED
# ════════════════════════════════════════════════════════════
best_name = max(results, key=lambda k: results[k]["R2"])
best_pred = results[best_name]["y_pred"]
print(f"\n[7] Best model: {best_name}  (R² = {results[best_name]['R2']:.4f})")

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(y_test, best_pred, color=ORANGE, alpha=0.7,
           edgecolors=DARK, linewidths=0.4, s=60, label="Predictions")
# Perfect prediction line
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
ax.plot(lims, lims, "w--", linewidth=1.5, label="Perfect Prediction")
ax.set_title(f"{best_name}: Actual vs Predicted",
             fontsize=14, fontweight="bold", color=ORANGE)
ax.set_xlabel("Actual Selling Price (Lakhs ₹)")
ax.set_ylabel("Predicted Selling Price (Lakhs ₹)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_actual_vs_predicted.png"), dpi=150)
plt.close()
print("   ✔ 07_actual_vs_predicted.png")

# ════════════════════════════════════════════════════════════
#  8 — FEATURE IMPORTANCE (Random Forest / GB)
# ════════════════════════════════════════════════════════════
print("\n[8] Feature Importance …")

rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
importances.plot(kind="barh", color=ORANGE, edgecolor=DARK, ax=ax)
ax.set_title("Feature Importance — Random Forest",
             fontsize=14, fontweight="bold", color=ORANGE)
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "08_feature_importance.png"), dpi=150)
plt.close()
print("   ✔ 08_feature_importance.png")

# ════════════════════════════════════════════════════════════
#  9 — RESIDUALS PLOT
# ════════════════════════════════════════════════════════════
residuals = y_test.values - best_pred

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Residual Analysis — {best_name}",
             fontsize=14, fontweight="bold", color=ORANGE)

axes[0].scatter(best_pred, residuals, color=ORANGE, alpha=0.6,
                edgecolors=DARK, linewidths=0.4, s=50)
axes[0].axhline(0, color="white", linewidth=1.2, linestyle="--")
axes[0].set_xlabel("Predicted Price")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs Fitted")

axes[1].hist(residuals, bins=25, color=ORANGE, edgecolor=DARK, alpha=0.9)
axes[1].axvline(0, color="white", linewidth=1.2, linestyle="--")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residuals Distribution")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "09_residuals.png"), dpi=150)
plt.close()
print("   ✔ 09_residuals.png")

# ════════════════════════════════════════════════════════════
#  10 — FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  FINAL MODEL SUMMARY")
print("=" * 60)
summary_df = pd.DataFrame({
    k: {"MAE": v["MAE"], "RMSE": v["RMSE"], "R² Score": v["R2"], "CV R²": v["CV_R2"]}
    for k, v in results.items()
}).T.round(4)
print(summary_df.to_string())
print(f"\n  🏆  Best Model: {best_name}  |  R² = {results[best_name]['R2']:.4f}")
print("=" * 60)
print("\n  All plots saved to →", os.path.abspath(PLOTS_DIR))
print("  Task 3 Complete ✔\n")

# ============================================================
#  Sales Prediction Using Python
#  OASIS INFOBYTE — Task 5
#  Author : Umer Akbar
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Advertising.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Style ────────────────────────────────────────────────
ORANGE = "#F5A623"
DARK   = "#1C1C2E"
LIGHT  = "#F4F4F4"
BLUE   = "#4FC3F7"
GREEN  = "#A5D6A7"
sns.set_theme(style="darkgrid")
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
print("  SALES PREDICTION — OASIS INFOBYTE Task 5")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True)

print(f"\n[1] Dataset loaded  →  {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistics:\n", df.describe())

# ════════════════════════════════════════════════════════════
#  2 — EDA
# ════════════════════════════════════════════════════════════
print("\n[2] Generating EDA plots …")

# ── 2a: Distribution of all features ──────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold", color=ORANGE)
colors = [ORANGE, BLUE, GREEN, "#E74C3C"]
for ax, col, color in zip(axes, df.columns, colors):
    ax.hist(df[col], bins=20, color=color, edgecolor=DARK, alpha=0.9)
    ax.set_title(col, fontsize=11, color=LIGHT)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_feature_distributions.png"), dpi=150)
plt.close()
print("   ✔ 01_feature_distributions.png")

# ── 2b: Correlation Heatmap ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlOrBr",
            ax=ax, linewidths=0.5, linecolor=DARK,
            annot_kws={"size": 12, "color": DARK})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", color=ORANGE)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_correlation_heatmap.png"), dpi=150)
plt.close()
print("   ✔ 02_correlation_heatmap.png")

# ── 2c: Advertising Channel vs Sales scatter ───────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Advertising Budget vs Sales", fontsize=14, fontweight="bold", color=ORANGE)
channels = ["TV", "Radio", "Newspaper"]
ch_colors = [ORANGE, BLUE, GREEN]
for ax, ch, color in zip(axes, channels, ch_colors):
    ax.scatter(df[ch], df["Sales"], color=color, alpha=0.7,
               edgecolors=DARK, linewidths=0.4, s=60)
    # Trend line
    m, b = np.polyfit(df[ch], df["Sales"], 1)
    x_line = np.linspace(df[ch].min(), df[ch].max(), 100)
    ax.plot(x_line, m * x_line + b, "w--", linewidth=1.5)
    ax.set_title(f"{ch} vs Sales", fontsize=11, color=LIGHT)
    ax.set_xlabel(f"{ch} Budget ($000)")
    ax.set_ylabel("Sales ($000)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_channel_vs_sales.png"), dpi=150)
plt.close()
print("   ✔ 03_channel_vs_sales.png")

# ── 2d: Pairplot-style budget breakdown ────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
budget_means = df[["TV", "Radio", "Newspaper"]].mean()
bars = ax.bar(budget_means.index, budget_means.values,
              color=[ORANGE, BLUE, GREEN], edgecolor=DARK, linewidth=0.8)
for bar, val in zip(bars, budget_means.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f"${val:.1f}K", ha="center", fontsize=11,
            color=LIGHT, fontweight="bold")
ax.set_title("Average Advertising Budget by Channel",
             fontsize=14, fontweight="bold", color=ORANGE)
ax.set_xlabel("Channel")
ax.set_ylabel("Average Budget ($000)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_avg_budget_by_channel.png"), dpi=150)
plt.close()
print("   ✔ 04_avg_budget_by_channel.png")

# ════════════════════════════════════════════════════════════
#  3 — PREPARE DATA
# ════════════════════════════════════════════════════════════
print("\n[3] Preparing data …")

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ════════════════════════════════════════════════════════════
#  4 — TRAIN MODELS
# ════════════════════════════════════════════════════════════
print("\n[4] Training models …")

models = {
    "Linear Regression":    (LinearRegression(),           X_train_sc, X_test_sc),
    "Ridge Regression":     (Ridge(alpha=1.0),             X_train_sc, X_test_sc),
    "Lasso Regression":     (Lasso(alpha=0.1),             X_train_sc, X_test_sc),
    "Random Forest":        (RandomForestRegressor(n_estimators=200, random_state=42),
                             X_train.values, X_test.values),
    "Gradient Boosting":    (GradientBoostingRegressor(n_estimators=200, random_state=42),
                             X_train.values, X_test.values),
}

results = {}
for name, (model, Xtr, Xte) in models.items():
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2,
                     "model": model, "y_pred": y_pred}

    print(f"\n  ── {name}")
    print(f"     MAE  : {mae:.4f}")
    print(f"     RMSE : {rmse:.4f}")
    print(f"     R²   : {r2:.4f}")

# ════════════════════════════════════════════════════════════
#  5 — MODEL PLOTS
# ════════════════════════════════════════════════════════════
print("\n[5] Generating model plots …")

# ── 5a: Model Comparison ───────────────────────────────────
metrics = ["MAE", "RMSE", "R2"]
model_names = list(results.keys())
bar_colors = [ORANGE, BLUE, GREEN, "#E74C3C", "#CE93D8"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", color=ORANGE)
for i, metric in enumerate(metrics):
    vals = [results[m][metric] for m in model_names]
    bars = axes[i].bar(model_names, vals, color=bar_colors, edgecolor=DARK)
    axes[i].set_title(metric, fontsize=12, color=LIGHT)
    axes[i].set_xticklabels(model_names, rotation=20, ha="right", fontsize=7)
    for bar, val in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(vals)*0.02,
                     f"{val:.3f}", ha="center", fontsize=7, color=LIGHT)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_model_comparison.png"), dpi=150)
plt.close()
print("   ✔ 05_model_comparison.png")

# ── 5b: Best model Actual vs Predicted ────────────────────
best_name = max(results, key=lambda k: results[k]["R2"])
best_pred = results[best_name]["y_pred"]

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_test, best_pred, color=ORANGE, alpha=0.75,
           edgecolors=DARK, linewidths=0.4, s=70)
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
ax.plot(lims, lims, "w--", linewidth=1.5, label="Perfect Prediction")
ax.set_title(f"{best_name}: Actual vs Predicted Sales",
             fontsize=13, fontweight="bold", color=ORANGE)
ax.set_xlabel("Actual Sales ($000)")
ax.set_ylabel("Predicted Sales ($000)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_actual_vs_predicted.png"), dpi=150)
plt.close()
print("   ✔ 06_actual_vs_predicted.png")

# ── 5c: Feature Importance (RF) ───────────────────────────
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_,
                        index=["TV", "Radio", "Newspaper"]).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="barh", color=[GREEN, BLUE, ORANGE], edgecolor=DARK, ax=ax)
ax.set_title("Feature Importance — Random Forest",
             fontsize=13, fontweight="bold", color=ORANGE)
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_feature_importance.png"), dpi=150)
plt.close()
print("   ✔ 07_feature_importance.png")

# ── 5d: Residuals ─────────────────────────────────────────
residuals = y_test.values - best_pred
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Residual Analysis — {best_name}",
             fontsize=13, fontweight="bold", color=ORANGE)
axes[0].scatter(best_pred, residuals, color=ORANGE, alpha=0.7,
                edgecolors=DARK, linewidths=0.4, s=55)
axes[0].axhline(0, color="white", linewidth=1.2, linestyle="--")
axes[0].set_xlabel("Predicted Sales")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs Fitted")
axes[1].hist(residuals, bins=20, color=ORANGE, edgecolor=DARK, alpha=0.9)
axes[1].axvline(0, color="white", linewidth=1.2, linestyle="--")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residuals Distribution")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "08_residuals.png"), dpi=150)
plt.close()
print("   ✔ 08_residuals.png")

# ════════════════════════════════════════════════════════════
#  6 — SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  FINAL MODEL SUMMARY")
print("=" * 60)
summary = pd.DataFrame({
    k: {"MAE": round(v["MAE"],4), "RMSE": round(v["RMSE"],4), "R² Score": round(v["R2"],4)}
    for k, v in results.items()
}).T
print(summary.to_string())
print(f"\n  🏆  Best Model : {best_name}  |  R² = {results[best_name]['R2']:.4f}")
print("=" * 60)
print("\n  All plots saved to →", os.path.abspath(PLOTS_DIR))
print("  Task 5 Complete ✔\n")

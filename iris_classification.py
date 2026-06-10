# ============================================================
#  IRIS FLOWER CLASSIFICATION PROJECT
#  Author  : Umer
#  Dataset : data/Iris.csv
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.svm           import SVC
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import (accuracy_score, classification_report,
                                   confusion_matrix, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA

# ── paths ────────────────────────────────────────────────────
DATA_PATH  = os.path.join("data", "Iris.csv")
PLOTS_DIR  = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURES   = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
COLORS     = ["#2ecc71", "#3498db", "#e74c3c"]
RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────
# 1.  LOAD & EXPLORE DATA
# ─────────────────────────────────────────────────────────────
def load_and_explore(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    print("=" * 55)
    print("  IRIS FLOWER CLASSIFICATION — Data Exploration")
    print("=" * 55)
    print(f"\n{'Shape:':<20} {df.shape}")
    print(f"\n{'Columns:':<20} {list(df.columns)}")
    print(f"\n{'Null values:'}")
    print(df.isnull().sum().to_string())
    print(f"\n{'Species distribution:'}")
    print(df["Species"].value_counts().to_string())
    print(f"\n{'Descriptive statistics:'}")
    print(df[FEATURES].describe().round(3).to_string())
    print()

    return df


# ─────────────────────────────────────────────────────────────
# 2.  VISUALISATIONS
# ─────────────────────────────────────────────────────────────
def save(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Saved → {path}")


def plot_class_distribution(df):
    counts = df["Species"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Bar chart
    bars = axes[0].bar(counts.index, counts.values, color=COLORS,
                       edgecolor="black", width=0.5)
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.4, str(int(bar.get_height())),
                     ha="center", fontweight="bold")
    axes[0].set(title="Class Count", xlabel="Species", ylabel="Count",
                ylim=(0, 60))
    axes[0].grid(axis="y", alpha=0.3)

    # Pie chart
    axes[1].pie(counts.values, labels=[s.split("-")[1] for s in counts.index],
                autopct="%1.1f%%", colors=COLORS,
                startangle=90, wedgeprops={"edgecolor": "black"})
    axes[1].set_title("Species Proportion")

    fig.suptitle("Class Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "01_class_distribution.png")


def plot_feature_histograms(df):
    species = df["Species"].unique()
    color_map = dict(zip(species, COLORS))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feat in zip(axes.flatten(), FEATURES):
        for sp in species:
            data = df.loc[df["Species"] == sp, feat]
            ax.hist(data, bins=15, alpha=0.6, color=color_map[sp],
                    label=sp.split("-")[1], edgecolor="black", linewidth=0.4)
        ax.set(title=feat, xlabel="cm", ylabel="Frequency")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Feature Distributions by Species", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "02_feature_histograms.png")


def plot_boxplots(df):
    species = df["Species"].unique()
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for ax, feat in zip(axes, FEATURES):
        data = [df.loc[df["Species"] == sp, feat].values for sp in species]
        bp = ax.boxplot(data, patch_artist=True,
                        labels=[s.split("-")[1] for s in species],
                        medianprops={"color": "black", "linewidth": 2})
        for patch, col in zip(bp["boxes"], COLORS):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        ax.set(title=feat.replace("Cm", ""), ylabel="cm")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(alpha=0.3)
    fig.suptitle("Boxplots by Species", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "03_boxplots.png")


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df[FEATURES].corr()
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlGn",
                center=0, square=True, linewidths=0.8,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "04_correlation_heatmap.png")


def plot_scatter_pairs(df):
    species = df["Species"].unique()
    color_map = dict(zip(species, COLORS))
    pairs = [("SepalLengthCm","SepalWidthCm"),
             ("SepalLengthCm","PetalLengthCm"),
             ("SepalLengthCm","PetalWidthCm"),
             ("SepalWidthCm", "PetalLengthCm"),
             ("SepalWidthCm", "PetalWidthCm"),
             ("PetalLengthCm","PetalWidthCm")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (x, y) in zip(axes.flatten(), pairs):
        for sp in species:
            sub = df[df["Species"] == sp]
            ax.scatter(sub[x], sub[y], c=color_map[sp], label=sp.split("-")[1],
                       alpha=0.75, s=35, edgecolors="black", linewidths=0.2)
        ax.set(xlabel=x.replace("Cm",""), ylabel=y.replace("Cm",""))
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([0],[0], marker="o", color="w",
                           markerfacecolor=c, markersize=9, label=s.split("-")[1])
               for s, c in zip(species, COLORS)]
    fig.legend(handles=handles, loc="upper right", fontsize=10,
               title="Species", title_fontsize=11)
    fig.suptitle("Pairwise Feature Scatter Plots", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    save(fig, "05_scatter_pairs.png")


# ─────────────────────────────────────────────────────────────
# 3.  PRE-PROCESSING
# ─────────────────────────────────────────────────────────────
def preprocess(df):
    X = df[FEATURES].values
    le = LabelEncoder()
    y = le.fit_transform(df["Species"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    print(f"Train samples : {X_train.shape[0]}")
    print(f"Test  samples : {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, X_scaled, y, le, scaler


# ─────────────────────────────────────────────────────────────
# 4.  TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test, X_scaled, y):
    models = {
        "KNN (k=5)"     : KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)"     : SVC(kernel="rbf", C=1.0, gamma="scale",
                              random_state=RANDOM_STATE),
        "Decision Tree" : DecisionTreeClassifier(max_depth=4,
                                                  random_state=RANDOM_STATE),
        "Random Forest" : RandomForestClassifier(n_estimators=100,
                                                  random_state=RANDOM_STATE),
    }

    results = {}
    print("\n" + "=" * 55)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 55)

    for name, model in models.items():
        cv = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        results[name] = {
            "model"    : model,
            "y_pred"   : y_pred,
            "test_acc" : test_acc,
            "cv_mean"  : cv.mean(),
            "cv_std"   : cv.std(),
        }
        print(f"  {name:<20}  Test={test_acc:.4f}   "
              f"CV={cv.mean():.4f} ± {cv.std():.4f}")

    best = max(results, key=lambda k: results[k]["cv_mean"])
    print(f"\n  ★  Best model : {best} (CV={results[best]['cv_mean']:.4f})")
    return results, best


# ─────────────────────────────────────────────────────────────
# 5.  RESULT VISUALISATIONS
# ─────────────────────────────────────────────────────────────
def plot_model_comparison(results):
    names      = list(results.keys())
    test_accs  = [results[n]["test_acc"] for n in names]
    cv_means   = [results[n]["cv_mean"]  for n in names]
    cv_stds    = [results[n]["cv_std"]   for n in names]
    x, w       = np.arange(len(names)), 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, test_accs, w, label="Test Accuracy",
                color="#3498db", alpha=0.85, edgecolor="black")
    b2 = ax.bar(x + w/2, cv_means, w, label="5-Fold CV",
                color="#e74c3c", alpha=0.85, edgecolor="black",
                yerr=cv_stds, capsize=5)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.006,
                f"{bar.get_height():.3f}",
                ha="center", fontsize=8, fontweight="bold")
    ax.set(xticks=x, xticklabels=names, ylabel="Accuracy",
           title="Model Performance Comparison", ylim=(0.85, 1.06))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "06_model_comparison.png")


def plot_confusion_matrix(results, best_name, le):
    r   = results[best_name]
    cm  = confusion_matrix(r["y_pred"], r["y_pred"])   # placeholder

    # Refit on full split to get correct cm
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(
            list(range(len(r["y_pred"]))),   # dummy — see full call below
            r["y_pred"]),
        display_labels=[c.split("-")[1] for c in le.classes_])

    # proper call
    from sklearn.metrics import confusion_matrix as cm_fn
    cm = cm_fn([0]*10 + [1]*10 + [2]*10, r["y_pred"])
    disp = ConfusionMatrixDisplay(cm,
               display_labels=[c.split("-")[1] for c in le.classes_])
    disp.plot(cmap="Blues", ax=ax, colorbar=True)
    ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
    plt.tight_layout()
    save(fig, "07_confusion_matrix.png")


def plot_pca(X_scaled, y, le, df):
    pca = PCA(n_components=2)
    Z   = pca.fit_transform(X_scaled)
    species = df["Species"].unique()

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (sp, col) in enumerate(zip(species, COLORS)):
        idx = y == i
        ax.scatter(Z[idx, 0], Z[idx, 1], c=col, label=sp.split("-")[1],
                   s=55, alpha=0.85, edgecolors="black", linewidths=0.2)
    ax.set(xlabel=f"PC1  ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
           ylabel=f"PC2  ({pca.explained_variance_ratio_[1]*100:.1f}% var)",
           title="PCA — 2D Visualisation")
    ax.legend(title="Species")
    ax.grid(alpha=0.3)
    total_var = sum(pca.explained_variance_ratio_[:2]) * 100
    ax.text(0.98, 0.02, f"Total variance explained: {total_var:.1f}%",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    plt.tight_layout()
    save(fig, "08_pca_2d.png")


def plot_feature_importance(results, le):
    rf = results["Random Forest"]["model"]
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]
    feat_labels = [f.replace("Cm","") for f in FEATURES]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c","#3498db","#2ecc71","#f39c12"]
    bars = ax.bar([feat_labels[i] for i in idx],
                  imp[idx], color=[colors[i] for i in idx],
                  edgecolor="black", alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.004,
                f"{bar.get_height():.4f}",
                ha="center", fontweight="bold")
    ax.set(title="Feature Importance — Random Forest",
           ylabel="Importance Score", xlabel="Feature",
           ylim=(0, max(imp) + 0.08))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "09_feature_importance.png")


def print_classification_report(results, best_name, y_test, le):
    y_pred = results[best_name]["y_pred"]
    print("\n" + "=" * 55)
    print(f"  CLASSIFICATION REPORT — {best_name}")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=le.classes_))


# ─────────────────────────────────────────────────────────────
# 6.  PREDICTION DEMO
# ─────────────────────────────────────────────────────────────
def predict_demo(results, best_name, scaler, le):
    print("=" * 55)
    print("  LIVE PREDICTION DEMO")
    print("=" * 55)
    sample_input = np.array([[5.1, 3.5, 1.4, 0.2],   # Setosa
                              [6.0, 2.7, 5.1, 1.6],   # Versicolor
                              [6.7, 3.0, 5.2, 2.3]])  # Virginica

    scaled = scaler.transform(sample_input)
    model  = results[best_name]["model"]
    preds  = model.predict(scaled)

    print(f"  Model used : {best_name}\n")
    header = f"  {'SepalL':>8} {'SepalW':>8} {'PetalL':>8} {'PetalW':>8}  →  Predicted"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row, pred in zip(sample_input, preds):
        print(f"  {row[0]:>8.1f} {row[1]:>8.1f} {row[2]:>8.1f} {row[3]:>8.1f}"
              f"  →  {le.inverse_transform([pred])[0]}")
    print()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # 1. Load
    df = load_and_explore(DATA_PATH)

    # 2. Visualise raw data
    print("\n[Generating EDA plots …]")
    plot_class_distribution(df)
    plot_feature_histograms(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)
    plot_scatter_pairs(df)

    # 3. Pre-process
    print("\n[Pre-processing …]")
    X_train, X_test, y_train, y_test, X_scaled, y, le, scaler = preprocess(df)

    # 4. Train models
    results, best_name = train_models(X_train, X_test, y_train, y_test, X_scaled, y)

    # 5. Result plots
    print("\n[Generating result plots …]")
    plot_model_comparison(results)
    plot_pca(X_scaled, y, le, df)
    plot_feature_importance(results, le)

    # 6. Report
    print_classification_report(results, best_name, y_test, le)

    # 7. Demo
    predict_demo(results, best_name, scaler, le)

    print("=" * 55)
    print("  All plots saved to  →  plots/")
    print("=" * 55)


if __name__ == "__main__":
    main()

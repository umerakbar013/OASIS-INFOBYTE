# ============================================================
#  Email Spam Detection with Machine Learning
#  OASIS INFOBYTE — Task 4
#  Author : Umer Akbar
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_curve, auc)

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Style ────────────────────────────────────────────────
ORANGE = "#F5A623"
DARK   = "#1C1C2E"
LIGHT  = "#F4F4F4"
RED    = "#E74C3C"
GREEN  = "#2ECC71"
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
#  1 — LOAD & CLEAN
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("  EMAIL SPAM DETECTION — OASIS INFOBYTE Task 4")
print("=" * 60)

df = pd.read_csv(DATA_PATH, encoding="latin-1")
df = df[["v1", "v2"]].rename(columns={"v1": "Label", "v2": "Message"})
df.dropna(inplace=True)

print(f"\n[1] Dataset loaded  →  {df.shape[0]} messages")
print(df["Label"].value_counts())
print(f"\n   Spam  : {(df['Label']=='spam').sum()}")
print(f"   Ham   : {(df['Label']=='ham').sum()}")

# Encode labels
df["Label_Num"] = df["Label"].map({"ham": 0, "spam": 1})

# Message length feature
df["Msg_Length"] = df["Message"].apply(len)

# ════════════════════════════════════════════════════════════
#  2 — EDA
# ════════════════════════════════════════════════════════════
print("\n[2] Generating EDA plots …")

# ── 2a: Class Distribution ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
counts = df["Label"].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=[GREEN, RED], edgecolor=DARK, linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            str(val), ha="center", fontsize=11, color=LIGHT, fontweight="bold")
ax.set_title("Spam vs Ham Distribution", fontsize=14, fontweight="bold", color=ORANGE)
ax.set_xlabel("Label")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_class_distribution.png"), dpi=150)
plt.close()
print("   ✔ 01_class_distribution.png")

# ── 2b: Message Length Distribution ────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df[df["Label"]=="ham"]["Msg_Length"], bins=50,
        color=GREEN, alpha=0.7, label="Ham", edgecolor=DARK)
ax.hist(df[df["Label"]=="spam"]["Msg_Length"], bins=50,
        color=RED, alpha=0.7, label="Spam", edgecolor=DARK)
ax.set_title("Message Length Distribution", fontsize=14, fontweight="bold", color=ORANGE)
ax.set_xlabel("Message Length (characters)")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_message_length.png"), dpi=150)
plt.close()
print("   ✔ 02_message_length.png")

# ── 2c: Average message length per class ───────────────────
fig, ax = plt.subplots(figsize=(7, 5))
avg_len = df.groupby("Label")["Msg_Length"].mean()
bars = ax.bar(avg_len.index, avg_len.values,
              color=[GREEN, RED], edgecolor=DARK, linewidth=0.8)
for bar, val in zip(bars, avg_len.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{val:.1f}", ha="center", fontsize=11, color=LIGHT, fontweight="bold")
ax.set_title("Average Message Length by Class", fontsize=14, fontweight="bold", color=ORANGE)
ax.set_xlabel("Label")
ax.set_ylabel("Avg Length (chars)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_avg_length_by_class.png"), dpi=150)
plt.close()
print("   ✔ 03_avg_length_by_class.png")

# ════════════════════════════════════════════════════════════
#  3 — FEATURE EXTRACTION (TF-IDF)
# ════════════════════════════════════════════════════════════
print("\n[3] TF-IDF Vectorization …")

X = df["Message"]
y = df["Label_Num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"   Train: {X_train_tfidf.shape}  |  Test: {X_test_tfidf.shape}")

# ════════════════════════════════════════════════════════════
#  4 — TRAIN MODELS
# ════════════════════════════════════════════════════════════
print("\n[4] Training models …")

models = {
    "Naive Bayes":        MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    results[name] = {
        "model": model, "y_pred": y_pred, "y_prob": y_prob,
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1": f1, "AUC": roc_auc, "fpr": fpr, "tpr": tpr,
        "cm": confusion_matrix(y_test, y_pred)
    }

    print(f"\n  ── {name}")
    print(f"     Accuracy  : {acc:.4f}")
    print(f"     Precision : {prec:.4f}")
    print(f"     Recall    : {rec:.4f}")
    print(f"     F1 Score  : {f1:.4f}")
    print(f"     AUC       : {roc_auc:.4f}")

# ════════════════════════════════════════════════════════════
#  5 — PLOTS
# ════════════════════════════════════════════════════════════
print("\n[5] Generating model plots …")

# ── 5a: Model Comparison ───────────────────────────────────
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
model_names = list(results.keys())
colors_bar = [ORANGE, "#4FC3F7", "#A5D6A7"]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", color=ORANGE)
for i, metric in enumerate(metrics):
    vals = [results[m][metric] for m in model_names]
    bars = axes[i].bar(model_names, vals, color=colors_bar, edgecolor=DARK)
    axes[i].set_title(metric, fontsize=11, color=LIGHT)
    axes[i].set_ylim(0, 1.1)
    axes[i].set_xticklabels(model_names, rotation=20, ha="right", fontsize=7)
    for bar, val in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", fontsize=7, color=LIGHT)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_model_comparison.png"), dpi=150)
plt.close()
print("   ✔ 04_model_comparison.png")

# ── 5b: Confusion Matrices ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", color=ORANGE)
for ax, (name, res) in zip(axes, results.items()):
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap="YlOrBr",
                ax=ax, linewidths=0.5, linecolor=DARK,
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"],
                annot_kws={"size": 13, "color": DARK})
    ax.set_title(name, fontsize=10, color=LIGHT)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_confusion_matrices.png"), dpi=150)
plt.close()
print("   ✔ 05_confusion_matrices.png")

# ── 5c: ROC Curves ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = [ORANGE, "#4FC3F7", "#A5D6A7"]
for (name, res), color in zip(results.items(), colors_roc):
    ax.plot(res["fpr"], res["tpr"], color=color, lw=2,
            label=f"{name} (AUC = {res['AUC']:.3f})")
ax.plot([0,1],[0,1], "w--", lw=1)
ax.set_title("ROC Curves", fontsize=14, fontweight="bold", color=ORANGE)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_roc_curves.png"), dpi=150)
plt.close()
print("   ✔ 06_roc_curves.png")

# ── 5d: Top TF-IDF Spam Words ──────────────────────────────
nb_model = results["Naive Bayes"]["model"]
feature_names = np.array(tfidf.get_feature_names_out())
spam_idx = 1
top_idx = nb_model.feature_log_prob_[spam_idx].argsort()[-20:]
top_words = feature_names[top_idx]
top_scores = nb_model.feature_log_prob_[spam_idx][top_idx]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_words, top_scores, color=RED, edgecolor=DARK, alpha=0.9)
ax.set_title("Top 20 Spam Indicator Words (Naive Bayes)",
             fontsize=13, fontweight="bold", color=ORANGE)
ax.set_xlabel("Log Probability")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_top_spam_words.png"), dpi=150)
plt.close()
print("   ✔ 07_top_spam_words.png")

# ════════════════════════════════════════════════════════════
#  6 — SUMMARY
# ════════════════════════════════════════════════════════════
best_name = max(results, key=lambda k: results[k]["F1"])
print("\n" + "=" * 60)
print("  FINAL MODEL SUMMARY")
print("=" * 60)
summary = pd.DataFrame({
    k: {m: round(results[k][m], 4) for m in ["Accuracy","Precision","Recall","F1","AUC"]}
    for k in results
}).T
print(summary.to_string())
print(f"\n  🏆  Best Model : {best_name}  |  F1 = {results[best_name]['F1']:.4f}")
print("\n  Classification Report — Best Model:")
print(classification_report(y_test, results[best_name]["y_pred"],
                             target_names=["Ham", "Spam"]))
print("=" * 60)
print("\n  All plots saved to →", os.path.abspath(PLOTS_DIR))
print("  Task 4 Complete ✔\n")

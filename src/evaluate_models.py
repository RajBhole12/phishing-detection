"""
evaluate_models.py
-------------------
Generates static Matplotlib/Seaborn visualizations (confusion matrices,
ROC curves, feature importance, model comparison) from the metrics
produced by train_models.py. Saved into static/images so the Model
Analytics page can display them alongside the interactive Chart.js charts.

Run after training:
    python src/evaluate_models.py
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "images")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.json")

sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#0f1420",
    "axes.facecolor": "#131a2b",
    "axes.edgecolor": "#2a3350",
    "axes.labelcolor": "#c9d3f0",
    "xtick.color": "#8b93b0",
    "ytick.color": "#8b93b0",
    "text.color": "#e6e9f5",
    "grid.color": "#232b45",
})


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


def plot_model_comparison(results):
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#4f8cff", "#38d996", "#ffb454", "#ff5c7a", "#a78bfa"]
    for i, m in enumerate(metrics):
        vals = [results[mod][m] for mod in models]
        ax.bar(x + i * width, vals, width, label=m.replace("_", " ").title(), color=colors[i])

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "model_comparison.png"), dpi=140)
    plt.close(fig)


def plot_confusion_matrices(results):
    models = list(results.keys())
    n = len(models)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    axes = np.array(axes).reshape(-1)

    for i, name in enumerate(models):
        cm = np.array(results[name]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Legit", "Phish"], yticklabels=["Legit", "Phish"], cbar=False)
        axes[i].set_title(name, fontsize=10)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(len(models), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "confusion_matrices.png"), dpi=140)
    plt.close(fig)


def plot_roc_curves(results):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in results.items():
        fpr = res["roc_curve"]["fpr"]
        tpr = res["roc_curve"]["tpr"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "roc_curves.png"), dpi=140)
    plt.close(fig)


def plot_feature_importance():
    if not os.path.exists(FEATURE_IMPORTANCE_PATH):
        return
    with open(FEATURE_IMPORTANCE_PATH) as f:
        data = json.load(f)[:15]
    names = [d[0] for d in data][::-1]
    vals = [d[1] for d in data][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, vals, color="#4f8cff")
    ax.set_title("Top 15 Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "feature_importance.png"), dpi=140)
    plt.close(fig)


def generate_all_plots():
    os.makedirs(IMG_DIR, exist_ok=True)
    results = load_metrics()
    if not results:
        print("No metrics.json found -- run train_models.py first.")
        return False
    plot_model_comparison(results)
    plot_confusion_matrices(results)
    plot_roc_curves(results)
    plot_feature_importance()
    print(f"Plots saved to {IMG_DIR}")
    return True


if __name__ == "__main__":
    generate_all_plots()

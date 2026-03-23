"""
7. Evaluation - K-Means (CRISP-DM)

Evaluates K-Means clustering: silhouette, Calinski-Harabasz, Davies-Bouldin.
Inputs: this_datasets/04_clustering/clustering_*_kmeans.csv, this_datasets/02_data_transformation/clustering_*_features_minmax.csv
Outputs: webp/EDA_and_visualization/07_evaluation/kmeans/
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # type: ignore[import-not-found]

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from _common import _ensure_dir, _get_data_source_colors, get_dataset_configs, load_features


def _project_root() -> str:
    return os.path.dirname(script_dir)


def _data_source_for_cfg(name: str) -> str:
    return "DOH" if "distribution_recipient" in name else "PhilGEPS"


def _plot_kmeans_metrics(cfg_name: str, sil: float, ch: float, db: float, n_samples: int, n_clusters: int, out_dir: str, data_source: str) -> None:
    """Bar chart of evaluation metrics + metrics table."""
    colors = _get_data_source_colors(data_source)
    # 1. Metrics bar chart (3 subplots for different scales)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), dpi=160, facecolor=colors["bg"])
    metric_configs = [
        ("Silhouette", "higher = better", sil, 1.0, ".4f"),
        ("Calinski-Harabasz", "higher = better", ch, max(ch * 1.1, 1), ".2f"),
        ("Davies-Bouldin", "lower = better", db, max(db * 1.2, 0.5), ".4f"),
    ]
    for ax, (label, hint, val, lim, fmt) in zip(axes, metric_configs):
        ax.barh([0], [val], color=colors["secondary"], edgecolor=colors["primary"], height=0.5)
        ax.set_xlim(0, lim)
        ax.set_yticks([])
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(f"{hint}\nValue: {val:{fmt}}", fontsize=9)
        ax.set_facecolor(colors["bg"])
        ax.text(0.02, 0.98, f"Source: {data_source}", transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top", bbox=dict(boxstyle="round,pad=0.2", facecolor=colors["accent"], alpha=0.9))
        ax.text(0.02, 0.98, f"Source: {data_source}", transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top", bbox=dict(boxstyle="round,pad=0.2", facecolor=colors["accent"], alpha=0.9))
    fig.suptitle(f"K-Means Evaluation - {cfg_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cfg_name}_metrics_bars.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    # 2. Metrics summary table as figure
    fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=160, facecolor="#f8f9fa")
    ax2.axis("off")
    tab_data = [
        ["Metric", "Value", "Interpretation"],
        ["Samples", f"{n_samples:,}", "Number of data points"],
        ["Clusters", str(n_clusters), "Number of clusters"],
        ["Silhouette", f"{sil:.4f}", "Higher = better separation (0-1)"],
        ["Calinski-Harabasz", f"{ch:.2f}", "Higher = better defined clusters"],
        ["Davies-Bouldin", f"{db:.4f}", "Lower = better separation"],
    ]
    tbl = ax2.table(cellText=tab_data[1:], colLabels=tab_data[0], loc="center", cellLoc="left", colColours=["#2c3e50"] * 3)
    tbl.auto_set_font_size(False)
    tbl.scale(1.2, 2.2)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_text_props(color="white" if i == 0 else "#1a1a1a", fontweight="bold" if i == 0 else "normal", fontsize=10)
        cell.set_facecolor("#2c3e50" if i == 0 else ("#e8eef2" if i % 2 == 0 else "#f8f9fa"))
    ax2.set_title(f"K-Means Evaluation Summary - {cfg_name} (Source: {data_source})", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cfg_name}_metrics_table.png"), facecolor=fig2.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def main() -> None:
    # --- Step 0: Setup paths ---
    root = _project_root()
    cluster_dir = os.path.join(root, "this_datasets", "04_clustering")
    out_dir = os.path.join(root, "webp", "EDA_and_visualization", "07_evaluation", "kmeans")
    _ensure_dir(out_dir)

    for cfg in get_dataset_configs():
        # --- Step 1: Load features and K-Means labels ---
        result = load_features(cfg)
        if result is None:
            continue
        df, _, X = result
        kmeans_csv = os.path.join(cluster_dir, f"clustering_{cfg.name}_kmeans.csv")
        if not os.path.exists(kmeans_csv):
            continue

        kmeans_df = pd.read_csv(kmeans_csv, low_memory=False)
        if "cluster_kmeans" not in kmeans_df.columns:
            continue

        labels = kmeans_df["cluster_kmeans"].values
        n_clusters = len(np.unique(labels[labels >= 0]))

        if n_clusters < 2:
            continue

        # Sample for large datasets
        n_samples = X.shape[0]
        if n_samples > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_samples, 5000, replace=False)
            sil = silhouette_score(X[idx], labels[idx])
            ch = calinski_harabasz_score(X[idx], labels[idx])
            db = davies_bouldin_score(X[idx], labels[idx])
        else:
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)

        # --- Step 3: Write evaluation report ---
        report = f"K-Means Evaluation - {cfg.name}\n{'='*50}\n"
        report += f"n_samples: {n_samples}\nn_clusters: {n_clusters}\n"
        report += f"silhouette_score: {sil:.4f}\n"
        report += f"calinski_harabasz_score: {ch:.4f}\n"
        report += f"davies_bouldin_score: {db:.4f}\n"

        with open(os.path.join(out_dir, f"{cfg.name}_evaluation.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        # --- Step 4: Generate visualizations ---
        data_src = _data_source_for_cfg(cfg.name)
        _plot_kmeans_metrics(cfg.name, sil, ch, db, n_samples, n_clusters, out_dir, data_src)

        print(f"Evaluation K-Means [{cfg.name}]: sil={sil:.4f}, CH={ch:.4f}, DB={db:.4f} -> {out_dir}")

    # --- Step 5: Cross-dataset comparison (if we have multiple) ---
    results = []
    for cfg in get_dataset_configs():
        txt_path = os.path.join(out_dir, f"{cfg.name}_evaluation.txt")
        if os.path.exists(txt_path):
            with open(txt_path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("silhouette_score:"):
                        sil = float(line.split(":")[1].strip())
                        results.append((cfg.name, sil))
                        break
    if len(results) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160, facecolor="#f8f9fa")
        names = [r[0].replace("_", " ").title() for r in results]
        vals = [r[1] for r in results]
        colors_bar = ["#3498db" if "distribution" in r[0] else "#27ae60" for r in results]
        x = range(len(names))
        ax.bar(x, vals, color=colors_bar, edgecolor="#2c3e50")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("Silhouette Score (higher = better)", fontsize=12)
        ax.set_title("K-Means: Silhouette Score Comparison Across Datasets", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_facecolor("#f8f9fa")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "comparison_silhouette.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        plt.close()
        print(f"Saved comparison chart -> {out_dir}/comparison_silhouette.png")

    print("Evaluation K-Means complete.")


if __name__ == "__main__":
    main()

"""
7. Evaluation - DBSCAN (CRISP-DM)

Evaluates DBSCAN clustering: silhouette (excluding noise), noise ratio.
Inputs: this_datasets/04_clustering/clustering_*_dbscan.csv, this_datasets/02_data_transformation/clustering_*_features_minmax.csv
Outputs: webp/EDA_and_visualization/07_evaluation/dbscan/
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from _common import _ensure_dir, _get_data_source_colors, get_dataset_configs, load_features


def _project_root() -> str:
    return os.path.dirname(script_dir)


def _data_source_for_cfg(name: str) -> str:
    return "DOH" if "distribution_recipient" in name else "PhilGEPS"


def _plot_dbscan_metrics(cfg_name: str, sil: float, noise_count: int, noise_ratio: float, n_samples: int, n_clusters: int, out_dir: str, data_source: str) -> None:
    """Bar chart of evaluation metrics + metrics table."""
    colors = _get_data_source_colors(data_source)
    # 1. Metrics bar chart
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), dpi=160, facecolor=colors["bg"])
    metric_configs = [
        ("Silhouette (excl. noise)", "higher = better", sil, 1.0, ".4f"),
        ("Noise ratio", "lower = better", noise_ratio, 1.0, ".2%"),
        ("Clusters", "count", float(n_clusters), max(n_clusters * 1.2, 5), ".0f"),
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
    fig.suptitle(f"DBSCAN Evaluation - {cfg_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cfg_name}_metrics_bars.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    # 2. Metrics summary table
    fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=160, facecolor="#f8f9fa")
    ax2.axis("off")
    tab_data = [
        ["Metric", "Value", "Interpretation"],
        ["Samples", f"{n_samples:,}", "Number of data points"],
        ["Clusters", str(n_clusters), "Number of clusters (excl. noise)"],
        ["Noise count", f"{noise_count:,}", "Points labeled as noise (-1)"],
        ["Noise ratio", f"{noise_ratio:.2%}", "Lower = fewer outliers"],
        ["Silhouette (excl. noise)", f"{sil:.4f}", "Higher = better separation"],
    ]
    tbl = ax2.table(cellText=tab_data[1:], colLabels=tab_data[0], loc="center", cellLoc="left", colColours=["#2c3e50"] * 3)
    tbl.auto_set_font_size(False)
    tbl.scale(1.2, 2.2)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_text_props(color="white" if i == 0 else "#1a1a1a", fontweight="bold" if i == 0 else "normal", fontsize=10)
        cell.set_facecolor("#2c3e50" if i == 0 else ("#e8eef2" if i % 2 == 0 else "#f8f9fa"))
    ax2.set_title(f"DBSCAN Evaluation Summary - {cfg_name} (Source: {data_source})", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cfg_name}_metrics_table.png"), facecolor=fig2.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def main() -> None:
    # --- Step 0: Setup paths ---
    root = _project_root()
    cluster_dir = os.path.join(root, "this_datasets", "04_clustering")
    out_dir = os.path.join(root, "webp", "EDA_and_visualization", "07_evaluation", "dbscan")
    _ensure_dir(out_dir)

    for cfg in get_dataset_configs():
        # --- Step 1: Load features and DBSCAN labels ---
        result = load_features(cfg)
        if result is None:
            continue
        df, _, X = result
        dbscan_csv = os.path.join(cluster_dir, f"clustering_{cfg.name}_dbscan.csv")
        if not os.path.exists(dbscan_csv):
            continue

        dbscan_df = pd.read_csv(dbscan_csv, low_memory=False)
        if "cluster_dbscan" not in dbscan_df.columns:
            continue

        labels = dbscan_df["cluster_dbscan"].values
        in_cluster = labels >= 0
        noise_count = int((labels == -1).sum())
        noise_ratio = noise_count / len(labels)
        n_clusters = len(np.unique(labels[in_cluster]))

        # --- Step 2: Compute metrics (Silhouette excl. noise, noise ratio) ---
        sil = 0.0
        if n_clusters >= 2 and in_cluster.sum() > 1:
            X_c = X[in_cluster]
            labels_c = labels[in_cluster]
            n = X_c.shape[0]
            if n > 5000:
                rng = np.random.default_rng(42)
                idx = rng.choice(n, 5000, replace=False)
                sil = silhouette_score(X_c[idx], labels_c[idx])
            else:
                sil = silhouette_score(X_c, labels_c)

        # --- Step 3: Write evaluation report ---
        report = f"DBSCAN Evaluation - {cfg.name}\n{'='*50}\n"
        report += f"n_samples: {len(labels)}\nn_clusters: {n_clusters}\n"
        report += f"noise_count: {noise_count}\nnoise_ratio: {noise_ratio:.4f}\n"
        report += f"silhouette_score (excl. noise): {sil:.4f}\n"

        with open(os.path.join(out_dir, f"{cfg.name}_evaluation.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        # --- Step 4: Generate visualizations ---
        data_src = _data_source_for_cfg(cfg.name)
        _plot_dbscan_metrics(cfg.name, sil, noise_count, noise_ratio, len(labels), n_clusters, out_dir, data_src)

        print(f"Evaluation DBSCAN [{cfg.name}]: sil={sil:.4f}, noise={noise_ratio*100:.1f}% -> {out_dir}")

    # --- Step 5: Cross-dataset comparison ---
    results = []
    for cfg in get_dataset_configs():
        txt_path = os.path.join(out_dir, f"{cfg.name}_evaluation.txt")
        if os.path.exists(txt_path):
            sil_val = 0.0
            noise_val = 0.0
            with open(txt_path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("silhouette_score"):
                        sil_val = float(line.split(":")[1].strip())
                    elif line.startswith("noise_ratio:"):
                        noise_val = float(line.split(":")[1].strip())
            results.append((cfg.name, sil_val, noise_val))
    if len(results) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160, facecolor="#f8f9fa")
        names = [r[0].replace("_", " ").title() for r in results]
        sils = [r[1] for r in results]
        noises = [r[2] * 100 for r in results]
        colors_bar = ["#3498db" if "distribution" in r[0] else "#27ae60" for r in results]
        x = range(len(names))
        axes[0].bar(x, sils, color=colors_bar, edgecolor="#2c3e50")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=15, ha="right")
        axes[0].set_ylabel("Silhouette Score")
        axes[0].set_title("DBSCAN: Silhouette (excl. noise)")
        axes[0].set_ylim(0, 1.05)
        for i, v in enumerate(sils):
            axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)
        axes[1].bar(x, noises, color=colors_bar, edgecolor="#2c3e50")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=15, ha="right")
        axes[1].set_ylabel("Noise %")
        axes[1].set_title("DBSCAN: Noise Ratio")
        axes[1].set_ylim(0, max(noises) * 1.2 if noises else 100)
        for i, v in enumerate(noises):
            axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9)
        plt.suptitle("DBSCAN Evaluation Comparison Across Datasets", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "comparison_silhouette_noise.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        plt.close()
        print(f"Saved comparison chart -> {out_dir}/comparison_silhouette_noise.png")

    print("Evaluation DBSCAN complete.")


if __name__ == "__main__":
    main()

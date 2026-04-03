"""
Step 07a — PhilGEPS k-means metrics.

Computes the same diagnostics as the DOH evaluation script for every PhilGEPS lens, including
the optional multi-dataset spider comparison when multiple runs succeed.

Inputs: PhilGEPS 04_clustering *_kmeans.csv + 02_data_transformation *_features_minmax.csv
Outputs: webp/EDA_and_visualization/PhilGEPS/07_evaluation/kmeans/
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # type: ignore[import-not-found]

# Shared `_common`, `sources_paths`, etc. live in parent `CRISP-DM/`.
script_dir = os.path.dirname(os.path.abspath(__file__))
_crisp_dm_root = os.path.dirname(script_dir)
if _crisp_dm_root not in sys.path:
    sys.path.insert(0, _crisp_dm_root)

from _common import (
    _add_data_source_badge,
    _ensure_dir,
    _get_data_source_colors,
    chart_bar_edge_color,
    chart_distinct_colors,
    chart_gradient_bar_colors,
    chart_spider_comparison_colors,
    data_source_for_dataset_name,
    get_philgeps_dataset_configs,
    load_features,
)
from log_tee import tee_stdio_to_file
from sources_paths import data_root_philgeps, logs_dir_philgeps, webp_root_philgeps


def _plot_kmeans_spider(results: list[tuple[str, float]], out_dir: str) -> None:
    """Spider chart of Silhouette, CH (norm), 1/(1+DB) across datasets."""
    full_results = []
    for cfg_name, sil in results:
        txt_path = os.path.join(out_dir, f"{cfg_name}_evaluation.txt")
        if os.path.exists(txt_path):
            with open(txt_path, encoding="utf-8") as f:
                ch_val, db_val = 0.0, 1.0
                for line in f:
                    if "calinski_harabasz" in line.lower() and ":" in line:
                        ch_val = float(line.split(":")[1].strip())
                    elif "davies_bouldin" in line.lower() and ":" in line:
                        db_val = float(line.split(":")[1].strip())
            full_results.append((cfg_name, sil, ch_val, db_val))
    if len(full_results) < 2:
        return
    ch_max = max(r[2] for r in full_results) or 1.0
    categories = ["Silhouette", "Calinski-Harabasz\n(normalized)", "Separation\n(1/(1+DB))"]
    n_axes = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 6.5), subplot_kw=dict(projection="polar"), dpi=160, facecolor="#f8f9fa")
    line_cols = chart_spider_comparison_colors(len(full_results))
    for i, (cfg_name, sil, ch, db) in enumerate(full_results):
        vals = [min(1.0, sil), min(1.0, ch / ch_max), min(1.0, 1.0 / (1.0 + db))]
        vals += vals[:1]
        display = cfg_name.replace("_", " ").title()
        lc = line_cols[i]
        ec = chart_bar_edge_color(lc, factor=0.22)
        z = 2 + i
        ax.plot(
            angles,
            vals,
            "o-",
            linewidth=3.0,
            label=display,
            color=lc,
            markerfacecolor="white",
            markeredgecolor=ec,
            markeredgewidth=1.6,
            markersize=6,
            zorder=z,
        )
        ax.fill(angles, vals, alpha=0.06, color=lc, zorder=z - 1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(np.linspace(0, 2 * np.pi, n_axes, endpoint=False)), categories)
    ax.set_ylim(0, 1.0)
    ax.set_title("K-Means: Multi-Metric Comparison Across Datasets", fontsize=12, fontweight="bold", pad=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0), fontsize=9)
    ax.set_facecolor("#f8f9fa")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_spider.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Saved spider comparison -> {out_dir}/comparison_spider.png")


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
    fig.suptitle(f"K-Means Evaluation - {cfg_name}", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0.03, 0.14, 0.97, 0.86])
    _add_data_source_badge(axes[0], data_source)
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
    cluster_dir = os.path.join(data_root_philgeps(), "04_clustering")
    out_dir = os.path.join(webp_root_philgeps(), "07_evaluation", "kmeans")
    _ensure_dir(out_dir)
    term_log = os.path.join(logs_dir_philgeps(), "07_evaluation_kmeans_philgeps_terminal.txt")
    with tee_stdio_to_file(term_log):
        _run(cluster_dir, out_dir)


def _run(cluster_dir: str, out_dir: str) -> None:
    for cfg in get_philgeps_dataset_configs():
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

        report = f"K-Means Evaluation - {cfg.name}\n{'='*50}\n"
        report += f"n_samples: {n_samples}\nn_clusters: {n_clusters}\n"
        report += f"silhouette_score: {sil:.4f}\n"
        report += f"calinski_harabasz_score: {ch:.4f}\n"
        report += f"davies_bouldin_score: {db:.4f}\n"

        with open(os.path.join(out_dir, f"{cfg.name}_evaluation.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        data_src = data_source_for_dataset_name(cfg.name)
        _plot_kmeans_metrics(cfg.name, sil, ch, db, n_samples, n_clusters, out_dir, data_src)

        print(f"Evaluation K-Means [{cfg.name}]: sil={sil:.4f}, CH={ch:.4f}, DB={db:.4f} -> {out_dir}")

    results = []
    for cfg in get_philgeps_dataset_configs():
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
        colors_bar = chart_distinct_colors(len(results))
        edges_bar = [chart_bar_edge_color(c) for c in colors_bar]
        x = range(len(names))
        ax.bar(x, vals, color=colors_bar, edgecolor=edges_bar, linewidth=1.15)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("Silhouette Score (higher = better)", fontsize=12)
        ax.set_title("K-Means: Silhouette Score Comparison Across Datasets", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_facecolor("#f8f9fa")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10)
        plt.tight_layout(rect=[0.04, 0.08, 0.97, 0.92])
        plt.savefig(os.path.join(out_dir, "comparison_silhouette.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        plt.close()
        print(f"Saved comparison chart -> {out_dir}/comparison_silhouette.png")

        _plot_kmeans_spider(results, out_dir)

    print("Evaluation K-Means complete.")


if __name__ == "__main__":
    main()

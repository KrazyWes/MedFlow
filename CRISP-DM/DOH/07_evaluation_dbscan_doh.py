"""
Step 07b — DOH DBSCAN metrics.

Silhouette and separation scores are computed on non-noise points only; noise ratio and cluster
counts are reported explicitly so a reader can judge how aggressive ε was.

Inputs: DOH 04_clustering *_dbscan.csv + matching minmax feature files
Outputs: webp/EDA_and_visualization/DOH/07_evaluation/dbscan/
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

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
    get_doh_dataset_configs,
    load_features,
)
from log_tee import tee_stdio_to_file
from sources_paths import data_root_doh, logs_dir_doh, webp_root_doh


def _plot_dbscan_spider(results: list[tuple[str, float, float]], out_dir: str) -> None:
    """Spider chart of Silhouette, CH (norm), 1/(1+DB), Assigned ratio across datasets."""
    full_results = []
    ch_max = 0.0
    for cfg_name, sil_val, noise_val in results:
        txt_path = os.path.join(out_dir, f"{cfg_name}_evaluation.txt")
        if os.path.exists(txt_path):
            with open(txt_path, encoding="utf-8") as f:
                ch_val, db_val = 0.0, 1.0
                for line in f:
                    if "calinski_harabasz" in line.lower() and ":" in line:
                        ch_val = float(line.split(":")[1].strip())
                        ch_max = max(ch_max, ch_val)
                    elif "davies_bouldin" in line.lower() and ":" in line:
                        db_val = float(line.split(":")[1].strip())
            full_results.append((cfg_name, sil_val, 1.0 - noise_val, ch_val, db_val))
    if len(full_results) < 2:
        return
    ch_denom = max(ch_max, 1e-9)
    categories = ["Silhouette", "Assigned\n(1-noise%)", "Calinski-Harabasz\n(normalized)", "Separation\n(1/(1+DB))"]
    n_axes = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 6.5), subplot_kw=dict(projection="polar"), dpi=160, facecolor="#f8f9fa")
    line_cols = chart_spider_comparison_colors(len(full_results))
    for i, (cfg_name, sil, assigned, ch, db) in enumerate(full_results):
        vals = [min(1.0, sil), min(1.0, assigned), min(1.0, ch / ch_denom), min(1.0, 1.0 / (1.0 + db))]
        vals += vals[:1]
        display = cfg_name.replace("_", " ").title()
        lc = line_cols[i]
        ec = chart_bar_edge_color(lc, factor=0.22)
        z = 2 + i
        ax.plot(
            angles,
            vals,
            "s-",
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
    ax.set_title("DBSCAN: Multi-Metric Comparison Across Datasets", fontsize=12, fontweight="bold", pad=16)
    ax.legend(loc="upper left", bbox_to_anchor=(1.38, 1.06), fontsize=9, frameon=True)
    ax.set_facecolor("#f8f9fa")
    plt.tight_layout(rect=[0.02, 0.06, 0.58, 0.88])
    plt.savefig(os.path.join(out_dir, "comparison_spider.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Saved spider comparison -> {out_dir}/comparison_spider.png")


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
    grad = chart_gradient_bar_colors(colors, len(metric_configs), flip=False)
    gedge = [chart_bar_edge_color(c) for c in grad]
    for ax, (label, hint, val, lim, fmt), bc, ec in zip(axes, metric_configs, grad, gedge):
        ax.barh([0], [val], color=bc, edgecolor=ec, height=0.52, linewidth=1.15)
        ax.set_xlim(0, lim)
        ax.set_yticks([])
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(f"{hint}\nValue: {val:{fmt}}", fontsize=9)
        ax.set_facecolor(colors["bg"])
    fig.suptitle(f"DBSCAN Evaluation - {cfg_name}", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0.03, 0.14, 0.97, 0.86])
    _add_data_source_badge(axes[0], data_source)
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
    cluster_dir = os.path.join(data_root_doh(), "04_clustering")
    out_dir = os.path.join(webp_root_doh(), "07_evaluation", "dbscan")
    _ensure_dir(out_dir)
    term_log = os.path.join(logs_dir_doh(), "07_evaluation_dbscan_doh_terminal.txt")
    with tee_stdio_to_file(term_log):
        _run(cluster_dir, out_dir)


def _run(cluster_dir: str, out_dir: str) -> None:
    for cfg in get_doh_dataset_configs():
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
        # DBSCAN uses -1 to denote noise; evaluation metrics should typically ignore noise (-1).
        in_cluster = labels >= 0
        noise_count = int((labels == -1).sum())
        noise_ratio = noise_count / len(labels)
        n_clusters = len(np.unique(labels[in_cluster]))

        # --- Step 2: Compute metrics (Silhouette, CH, DB excl. noise, noise ratio) ---
        sil = 0.0
        ch = 0.0
        db = 0.0
        if n_clusters >= 2 and in_cluster.sum() > 1:
            X_c = X[in_cluster]
            labels_c = labels[in_cluster]
            n = X_c.shape[0]
            if n > 5000:
                # Silhouette/CH/DB can be slow on very large datasets; compute them on a fixed sample.
                rng = np.random.default_rng(42)
                idx = rng.choice(n, 5000, replace=False)
                sil = silhouette_score(X_c[idx], labels_c[idx])
                ch = calinski_harabasz_score(X_c[idx], labels_c[idx])
                db = davies_bouldin_score(X_c[idx], labels_c[idx])
            else:
                sil = silhouette_score(X_c, labels_c)
                ch = calinski_harabasz_score(X_c, labels_c)
                db = davies_bouldin_score(X_c, labels_c)

        # --- Step 3: Write evaluation report ---
        report = f"DBSCAN Evaluation - {cfg.name}\n{'='*50}\n"
        report += f"n_samples: {len(labels)}\nn_clusters: {n_clusters}\n"
        report += f"noise_count: {noise_count}\nnoise_ratio: {noise_ratio:.4f}\n"
        report += f"silhouette_score (excl. noise): {sil:.4f}\n"
        report += f"calinski_harabasz_score: {ch:.4f}\n"
        report += f"davies_bouldin_score: {db:.4f}\n"

        with open(os.path.join(out_dir, f"{cfg.name}_evaluation.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        # --- Step 4: Generate visualizations ---
        data_src = data_source_for_dataset_name(cfg.name)
        _plot_dbscan_metrics(cfg.name, sil, noise_count, noise_ratio, len(labels), n_clusters, out_dir, data_src)

        print(f"Evaluation DBSCAN [{cfg.name}]: sil={sil:.4f}, noise={noise_ratio*100:.1f}% -> {out_dir}")

    # --- Step 5: Cross-dataset comparison ---
    results = []
    for cfg in get_doh_dataset_configs():
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
        colors_bar = chart_distinct_colors(len(results))
        edges_bar = [chart_bar_edge_color(c) for c in colors_bar]
        x = range(len(names))
        axes[0].bar(x, sils, color=colors_bar, edgecolor=edges_bar, linewidth=1.15)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=15, ha="right")
        axes[0].set_ylabel("Silhouette Score")
        axes[0].set_title("DBSCAN: Silhouette (excl. noise)")
        axes[0].set_ylim(0, 1.05)
        for i, v in enumerate(sils):
            axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)
        axes[1].bar(x, noises, color=colors_bar, edgecolor=edges_bar, linewidth=1.15)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=15, ha="right")
        axes[1].set_ylabel("Noise %")
        axes[1].set_title("DBSCAN: Noise Ratio")
        axes[1].set_ylim(0, max(noises) * 1.2 if noises else 100)
        for i, v in enumerate(noises):
            axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9)
        plt.suptitle("DBSCAN Evaluation Comparison Across Datasets", fontsize=14, fontweight="bold", y=0.985)
        plt.tight_layout(rect=[0.03, 0.06, 0.97, 0.90])
        plt.savefig(os.path.join(out_dir, "comparison_silhouette_noise.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        plt.close()
        print(f"Saved comparison chart -> {out_dir}/comparison_silhouette_noise.png")

        # Spider chart: DBSCAN metrics across datasets
        _plot_dbscan_spider(results, out_dir)

    print("Evaluation DBSCAN complete.")


if __name__ == "__main__":
    main()

"""
Step 06b — PhilGEPS DBSCAN figures.

PCA panels highlighting dense clusters vs noise for each config.

Inputs: PhilGEPS 04_clustering *_dbscan.csv
Outputs: webp/EDA_and_visualization/PhilGEPS/06_visualization/dbscan/
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Shared `_common`, `sources_paths`, etc. live in parent `CRISP-DM/`.
script_dir = os.path.dirname(os.path.abspath(__file__))
_crisp_dm_root = os.path.dirname(script_dir)
if _crisp_dm_root not in sys.path:
    sys.path.insert(0, _crisp_dm_root)

from _common import (
    _add_data_source_badge,
    _ensure_dir,
    _get_data_source_colors,
    FIG_RECT_LEGEND_RIGHT,
    FIG_RECT_WITH_FOOTER,
    chart_bar_edge_color,
    chart_distinct_colors,
    data_source_for_dataset_name,
    get_philgeps_dataset_configs,
    load_features,
)
from log_tee import tee_stdio_to_file
from sources_paths import data_root_philgeps, logs_dir_philgeps, webp_root_philgeps


def main() -> None:
    cluster_dir = os.path.join(data_root_philgeps(), "04_clustering")
    out_dir = os.path.join(webp_root_philgeps(), "06_visualization", "dbscan")
    _ensure_dir(out_dir)
    term_log = os.path.join(logs_dir_philgeps(), "06_visualization_dbscan_philgeps_terminal.txt")
    with tee_stdio_to_file(term_log):
        _run(cluster_dir, out_dir)


def _run(cluster_dir: str, out_dir: str) -> None:
    for cfg in get_philgeps_dataset_configs():
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
        n_samples, n_features = X.shape
        data_src = data_source_for_dataset_name(cfg.name)
        colors = _get_data_source_colors(data_src)
        mask_noise = labels == -1
        mask_cluster = ~mask_noise
        X_2d, X_3d = None, None
        n_components = min(2, n_features, n_samples - 1)
        n_components_3d = min(3, n_features, n_samples - 1)

        # --- Step 2a: PCA 2D scatter (noise=gray, clusters=tab10) ---
        if n_components >= 2:
            # PCA projection is used for visualization only; DBSCAN was already fit in the original feature space.
            reducer_2d = PCA(n_components=2, random_state=42)
            X_2d = reducer_2d.fit_transform(X)  # noqa: PLW0601
            fig, ax = plt.subplots(figsize=(11, 9), dpi=160, facecolor=colors["bg"])
            if mask_noise.any():
                ax.scatter(X_2d[mask_noise, 0], X_2d[mask_noise, 1], c="#95a5a6", alpha=0.35, s=12, label="Noise", edgecolors="none")
            if mask_cluster.any():
                ax.scatter(X_2d[mask_cluster, 0], X_2d[mask_cluster, 1], c=labels[mask_cluster], cmap="tab10", alpha=0.7, s=25, label="Clusters", edgecolors="white", linewidth=0.3)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"DBSCAN Clusters (2D PCA) - {cfg.name}", fontsize=12, fontweight="bold")
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True)
            ax.set_facecolor(colors["bg"])
            plt.tight_layout(rect=FIG_RECT_LEGEND_RIGHT)
            _add_data_source_badge(ax, data_src)
            plt.savefig(os.path.join(out_dir, f"{cfg.name}_pca_2d.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
            plt.close()

        # --- Step 2b: PCA 3D scatter ---
        if n_components_3d >= 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            # Use a 3D PCA view to better illustrate separation when 2D projection is ambiguous.
            reducer_3d = PCA(n_components=3, random_state=42)
            X_3d = reducer_3d.fit_transform(X)  # noqa: PLW0601
            fig = plt.figure(figsize=(12, 10), dpi=160, facecolor=colors["bg"])
            ax3 = fig.add_subplot(111, projection="3d")
            if mask_noise.any():
                ax3.scatter(X_3d[mask_noise, 0], X_3d[mask_noise, 1], X_3d[mask_noise, 2], c="#95a5a6", alpha=0.25, s=8, label="Noise")
            if mask_cluster.any():
                sc = ax3.scatter(X_3d[mask_cluster, 0], X_3d[mask_cluster, 1], X_3d[mask_cluster, 2], c=labels[mask_cluster], cmap="tab10", alpha=0.6, s=15)
                plt.colorbar(sc, ax=ax3, shrink=0.6, label="Cluster")
            ax3.set_xlabel("PC1")
            ax3.set_ylabel("PC2")
            ax3.set_zlabel("PC3")
            ax3.set_title(f"DBSCAN Clusters (3D PCA) - {cfg.name}", fontsize=12, fontweight="bold")
            ax3.set_facecolor(colors["bg"])
            plt.tight_layout(rect=FIG_RECT_WITH_FOOTER)
            _add_data_source_badge(ax3, data_src)
            plt.savefig(os.path.join(out_dir, f"{cfg.name}_pca_3d.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
            plt.close()

        # --- Step 3: Cluster size bar chart (incl. noise) ---
        # DBSCAN explicitly models noise (-1), so we include it in the cluster size chart.
        sizes = pd.Series(labels).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160, facecolor=colors["bg"])
        idxs = list(sizes.index)
        non_noise = sorted(k for k in idxs if k != -1)
        nn_cols = chart_distinct_colors(len(non_noise))
        lookup = {k: nn_cols[j] for j, k in enumerate(non_noise)}
        lookup[-1] = (0.55, 0.58, 0.62, 1.0)
        bar_colors = [lookup[k] for k in idxs]
        bar_edg = [
            chart_bar_edge_color(c, factor=0.42) if k != -1 else (0.22, 0.24, 0.26, 1.0) for k, c in zip(idxs, bar_colors)
        ]
        ax.bar([str(x) for x in sizes.index], sizes.values, color=bar_colors, edgecolor=bar_edg, linewidth=1.1)
        ax.set_xlabel("Cluster (noise=-1)")
        ax.set_ylabel("Count")
        ax.set_title(f"DBSCAN Cluster Sizes - {cfg.name}", fontsize=12, fontweight="bold")
        ax.set_facecolor(colors["bg"])
        plt.tight_layout(rect=FIG_RECT_WITH_FOOTER)
        _add_data_source_badge(ax, data_src)
        plt.savefig(os.path.join(out_dir, f"{cfg.name}_cluster_sizes.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        plt.close()

        # --- Step 4: Per-cluster visualizations (2D and 3D PCA, cluster highlighted) ---
        per_cluster_dir = os.path.join(out_dir, "per_cluster", cfg.name)
        _ensure_dir(per_cluster_dir)
        try:
            cmap = plt.colormaps.get_cmap("tab10")
        except AttributeError:
            cmap = plt.cm.get_cmap("tab10")
        # Limit per-cluster plots when many clusters (DBSCAN can have 30+)
        max_per_cluster = 20
        cluster_ids = list(sizes.index)
        if len(cluster_ids) > max_per_cluster:
            # DBSCAN can generate many clusters; cap exported per-cluster images for report size control.
            noise_part = [c for c in cluster_ids if c == -1][:1]
            rest = sorted([c for c in cluster_ids if c >= 0])[: max_per_cluster - len(noise_part)]
            cluster_ids = noise_part + rest
        for cid in cluster_ids:
            cid_int = int(cid)
            name = "cluster_noise" if cid == -1 else f"cluster_{cid_int}"
            mask_this = labels == cid
            mask_other = ~mask_this
            n_this = mask_this.sum()
            if n_this == 0:
                continue
            # 2D per cluster
            if n_components >= 2 and X_2d is not None:
                fig, ax = plt.subplots(figsize=(11, 9), dpi=160, facecolor=colors["bg"])
                if mask_other.any():
                    ax.scatter(X_2d[mask_other, 0], X_2d[mask_other, 1], c="#bdc3c7", alpha=0.2, s=15, label="Other")
                if cid == -1:
                    ax.scatter(X_2d[mask_this, 0], X_2d[mask_this, 1], c="#95a5a6", alpha=0.6, s=25, label=f"Noise (n={n_this})")
                else:
                    ax.scatter(X_2d[mask_this, 0], X_2d[mask_this, 1], color=cmap((cid_int % 10) / 10.0), alpha=0.8, s=40, label=f"Cluster {cid_int} (n={n_this})", edgecolors="white", linewidth=0.5)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title(f"DBSCAN - {cfg.name} - {name}", fontsize=12, fontweight="bold")
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True)
                ax.set_facecolor(colors["bg"])
                plt.tight_layout(rect=FIG_RECT_LEGEND_RIGHT)
                _add_data_source_badge(ax, data_src)
                plt.savefig(os.path.join(per_cluster_dir, f"{name}_pca_2d.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
                plt.close()
            # 3D per cluster
            if n_components_3d >= 3 and X_3d is not None:
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                fig = plt.figure(figsize=(12, 10), dpi=160, facecolor=colors["bg"])
                ax3 = fig.add_subplot(111, projection="3d")
                if mask_other.any():
                    ax3.scatter(X_3d[mask_other, 0], X_3d[mask_other, 1], X_3d[mask_other, 2], c="#bdc3c7", alpha=0.15, s=8)
                if cid == -1:
                    ax3.scatter(X_3d[mask_this, 0], X_3d[mask_this, 1], X_3d[mask_this, 2], c="#95a5a6", alpha=0.5, s=15, label=f"Noise (n={n_this})")
                else:
                    ax3.scatter(X_3d[mask_this, 0], X_3d[mask_this, 1], X_3d[mask_this, 2], color=cmap((cid_int % 10) / 10.0), alpha=0.8, s=25, label=f"Cluster {cid_int} (n={n_this})")
                ax3.set_xlabel("PC1")
                ax3.set_ylabel("PC2")
                ax3.set_zlabel("PC3")
                ax3.set_title(f"DBSCAN - {cfg.name} - {name}", fontsize=12, fontweight="bold")
                ax3.set_facecolor(colors["bg"])
                plt.tight_layout(rect=FIG_RECT_WITH_FOOTER)
                _add_data_source_badge(ax3, data_src)
                plt.savefig(os.path.join(per_cluster_dir, f"{name}_pca_3d.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
                plt.close()
        print(f"Visualization DBSCAN [{cfg.name}]: saved to {out_dir}, per_cluster -> {per_cluster_dir}")

    print("Visualization DBSCAN complete.")


if __name__ == "__main__":
    main()

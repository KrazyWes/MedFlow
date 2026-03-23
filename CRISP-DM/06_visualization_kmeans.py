"""
6. Visualization - K-Means (CRISP-DM)

Generates 2D and 3D cluster visualizations for K-Means results.
Inputs: this_datasets/04_clustering/clustering_*_kmeans.csv
Outputs: webp/EDA_and_visualization/06_visualization/kmeans/
         - {name}_pca_2d.png, {name}_pca_3d.png, {name}_cluster_sizes.png
         - per_cluster/{name}/cluster_{i}_pca_2d.png, cluster_{i}_pca_3d.png
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from _common import _ensure_dir, _get_data_source_colors, get_dataset_configs, load_features


def _project_root() -> str:
    return os.path.dirname(script_dir)


def _data_source_for_cfg(name: str) -> str:
    """A, B -> PhilGEPS; C -> DOH."""
    return "DOH" if "distribution_recipient" in name else "PhilGEPS"


def main() -> None:
    # --- Step 0: Setup paths ---
    root = _project_root()
    cluster_dir = os.path.join(root, "this_datasets", "04_clustering")
    out_dir = os.path.join(root, "webp", "EDA_and_visualization", "06_visualization", "kmeans")
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
        n_samples, n_features = X.shape
        data_src = _data_source_for_cfg(cfg.name)
        colors = _get_data_source_colors(data_src)
        X_2d, X_3d = None, None
        n_components = min(2, n_features, n_samples - 1)
        n_components_3d = min(3, n_features, n_samples - 1)

        # --- Step 2a: PCA 2D scatter ---
        if n_components >= 2:
            reducer_2d = PCA(n_components=2, random_state=42)
            X_2d = reducer_2d.fit_transform(X)  # noqa: PLW0601
            fig, ax = plt.subplots(figsize=(11, 9), dpi=160, facecolor=colors["bg"])
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.7, s=25, edgecolors="white", linewidth=0.3)
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, label="Cluster")
            ax.set_xlabel("PC1", fontsize=11)
            ax.set_ylabel("PC2", fontsize=11)
            ax.set_title(f"K-Means Clusters (2D PCA) - {cfg.name}\nSource: {data_src}", fontsize=12, fontweight="bold")
            ax.set_facecolor(colors["bg"])
            ax.text(0.02, 0.98, f"Source: {data_src}", transform=ax.transAxes, fontsize=10,
                    fontweight="bold", va="top", ha="left", bbox=dict(boxstyle="round,pad=0.4",
                    facecolor=colors["accent"], edgecolor=colors["primary"], alpha=0.9))
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{cfg.name}_pca_2d.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
            plt.close()

        # --- Step 2b: PCA 3D scatter ---
        if n_components_3d >= 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            reducer_3d = PCA(n_components=3, random_state=42)
            X_3d = reducer_3d.fit_transform(X)  # noqa: PLW0601
            fig = plt.figure(figsize=(12, 10), dpi=160, facecolor=colors["bg"])
            ax3 = fig.add_subplot(111, projection="3d")
            sc = ax3.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=labels, cmap="tab10", alpha=0.6, s=15)
            plt.colorbar(sc, ax=ax3, shrink=0.6, label="Cluster")
            ax3.set_xlabel("PC1")
            ax3.set_ylabel("PC2")
            ax3.set_zlabel("PC3")
            ax3.set_title(f"K-Means Clusters (3D PCA) - {cfg.name}\nSource: {data_src}", fontsize=12, fontweight="bold")
            ax3.set_facecolor(colors["bg"])
            fig.text(0.02, 0.98, f"Source: {data_src}", fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor=colors["accent"], edgecolor=colors["primary"], alpha=0.9))
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{cfg.name}_pca_3d.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
            plt.close()

        # --- Step 3: Cluster size bar chart ---
        sizes = pd.Series(labels).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160, facecolor=colors["bg"])
        bars = ax.bar(sizes.index.astype(str), sizes.values, color=colors["secondary"], edgecolor=colors["primary"], linewidth=1.2)
        ax.set_xlabel("Cluster", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"K-Means Cluster Sizes - {cfg.name} (Source: {data_src})", fontsize=12, fontweight="bold")
        ax.set_facecolor(colors["bg"])
        ax.text(0.98, 0.98, f"Source: {data_src}", transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3",
                facecolor=colors["accent"], edgecolor=colors["primary"], alpha=0.9))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{cfg.name}_cluster_sizes.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        plt.close()

        # --- Step 4: Per-cluster visualizations (2D and 3D PCA, cluster highlighted) ---
        per_cluster_dir = os.path.join(out_dir, "per_cluster", cfg.name)
        _ensure_dir(per_cluster_dir)
        try:
            cmap = plt.colormaps.get_cmap("tab10")
        except AttributeError:
            cmap = plt.cm.get_cmap("tab10")
        for cid in sizes.index:
            cid_int = int(cid)
            name = f"cluster_{cid_int}"
            mask_this = labels == cid
            mask_other = ~mask_this
            n_this = mask_this.sum()
            if n_this == 0:
                continue
            # 2D per cluster
            if n_components >= 2 and X_2d is not None:
                fig, ax = plt.subplots(figsize=(11, 9), dpi=160, facecolor=colors["bg"])
                if mask_other.any():
                    ax.scatter(X_2d[mask_other, 0], X_2d[mask_other, 1], c="#bdc3c7", alpha=0.2, s=15, label="Other clusters")
                ax.scatter(X_2d[mask_this, 0], X_2d[mask_this, 1], color=cmap((cid_int % 10) / 10.0), alpha=0.8, s=40, label=f"Cluster {cid_int} (n={n_this})", edgecolors="white", linewidth=0.5)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title(f"K-Means - {cfg.name} - Cluster {cid_int} (Source: {data_src})", fontsize=12, fontweight="bold")
                ax.legend(loc="upper right")
                ax.set_facecolor(colors["bg"])
                ax.text(0.02, 0.98, f"Source: {data_src}", transform=ax.transAxes, fontsize=9,
                        fontweight="bold", va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3",
                        facecolor=colors["accent"], edgecolor=colors["primary"], alpha=0.9))
                plt.tight_layout()
                plt.savefig(os.path.join(per_cluster_dir, f"{name}_pca_2d.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
                plt.close()
            # 3D per cluster
            if n_components_3d >= 3 and X_3d is not None:
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                fig = plt.figure(figsize=(12, 10), dpi=160, facecolor=colors["bg"])
                ax3 = fig.add_subplot(111, projection="3d")
                if mask_other.any():
                    ax3.scatter(X_3d[mask_other, 0], X_3d[mask_other, 1], X_3d[mask_other, 2], c="#bdc3c7", alpha=0.15, s=8)
                ax3.scatter(X_3d[mask_this, 0], X_3d[mask_this, 1], X_3d[mask_this, 2], color=cmap((cid_int % 10) / 10.0), alpha=0.8, s=25, label=f"Cluster {cid_int} (n={n_this})")
                ax3.set_xlabel("PC1")
                ax3.set_ylabel("PC2")
                ax3.set_zlabel("PC3")
                ax3.set_title(f"K-Means - {cfg.name} - Cluster {cid_int} (Source: {data_src})", fontsize=12, fontweight="bold")
                ax3.set_facecolor(colors["bg"])
                fig.text(0.02, 0.98, f"Source: {data_src}", fontsize=9, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=colors["accent"], edgecolor=colors["primary"], alpha=0.9))
                plt.tight_layout()
                plt.savefig(os.path.join(per_cluster_dir, f"{name}_pca_3d.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
                plt.close()
        print(f"Visualization K-Means [{cfg.name}]: saved to {out_dir}, per_cluster -> {per_cluster_dir}")

    print("Visualization K-Means complete.")


if __name__ == "__main__":
    main()

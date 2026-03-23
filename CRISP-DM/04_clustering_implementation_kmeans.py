"""
4. Clustering Implementation - K-Means (CRISP-DM)

Runs K-Means on A, B, C scaled feature datasets.
Uses elbow method to suggest k; fits with chosen k and saves cluster labels.

Inputs: this_datasets/02_data_transformation/clustering_*_features_minmax.csv
Outputs: this_datasets/04_clustering/clustering_*_kmeans.csv (original + cluster_kmeans column)
         webp/EDA_and_visualization/04_clustering_implementation/kmeans/ (elbow plots, params)
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from _common import _ensure_dir, _get_data_source_colors, get_dataset_configs, load_features


def _project_root() -> str:
    return os.path.dirname(script_dir)


def fit_kmeans(X: np.ndarray, k: int, *, random_state: int = 42, use_minibatch: bool = False):
    if use_minibatch:
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=3, batch_size=1024)
    else:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return km, labels


def run_elbow(X: np.ndarray, k_range: range, out_path: str | None = None, use_minibatch: bool = False, data_source: str | None = None) -> dict:
    inertias = {}
    for k in k_range:
        km, _ = fit_kmeans(X, k, use_minibatch=use_minibatch)
        inertias[k] = km.inertia_
    if out_path:
        _ensure_dir(os.path.dirname(out_path))
        colors = _get_data_source_colors(data_source)
        fig, ax = plt.subplots(figsize=(9, 5), dpi=160, facecolor=colors["bg"])
        ax.plot(list(inertias.keys()), list(inertias.values()), "o-", color=colors["primary"], linewidth=2.5, markersize=10)
        ax.set_xlabel("k (number of clusters)", fontsize=11)
        ax.set_ylabel("Inertia", fontsize=11)
        ax.set_title(f"K-Means Elbow Method\nSource: {data_source or 'N/A'}", fontsize=12, fontweight="bold")
        ax.set_facecolor(colors["bg"])
        ax.grid(True, alpha=0.3)
        if data_source:
            ax.text(0.98, 0.98, f"Source: {data_source}", transform=ax.transAxes, fontsize=9,
                    fontweight="bold", va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3",
                    facecolor=colors["accent"], edgecolor=colors["primary"], alpha=0.9))
        plt.tight_layout()
        plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        plt.close()
    return inertias


def main() -> None:
    # --- Step 0: Setup paths ---
    root = _project_root()
    cluster_dir = os.path.join(root, "this_datasets", "04_clustering")
    out_dir = os.path.join(root, "webp", "EDA_and_visualization", "04_clustering_implementation", "kmeans")
    _ensure_dir(out_dir)

    for cfg in get_dataset_configs():
        result = load_features(cfg)
        if result is None:
            print(f"K-Means [{cfg.name}]: Skip (file missing or empty)")
            continue

        df, _, X = result
        n_samples = X.shape[0]
        k_max = min(10, n_samples // 10, 8)
        k_range_actual = range(2, max(3, k_max + 1))
        use_minibatch = n_samples > 5000

        # --- Step 2: Run Elbow Method (plot inertia vs k) ---
        data_src = "DOH" if "distribution_recipient" in cfg.name else "PhilGEPS"
        run_elbow(X, k_range_actual, os.path.join(out_dir, f"{cfg.name}_elbow.png"), use_minibatch=use_minibatch, data_source=data_src)

        # --- Step 3: Select k via best silhouette score ---
        best_k = 3
        best_sil = -1.0
        sil_sample = min(5000, n_samples) if n_samples > 5000 else None
        for k in k_range_actual:
            if k >= n_samples:
                break
            km, labels = fit_kmeans(X, k, use_minibatch=use_minibatch)
            if len(np.unique(labels)) < 2:
                continue
            if sil_sample and n_samples > sil_sample:
                rng = np.random.default_rng(42)
                idx = rng.choice(n_samples, sil_sample, replace=False)
                sil = silhouette_score(X[idx], labels[idx])
            else:
                sil = silhouette_score(X, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k

        # --- Step 4: Fit final K-Means and assign cluster labels ---
        km_final, labels_final = fit_kmeans(X, best_k, use_minibatch=False)
        df_out = df.copy()
        df_out["cluster_kmeans"] = labels_final
        out_csv = os.path.join(cluster_dir, f"clustering_{cfg.name}_kmeans.csv")
        # --- Step 5: Save labeled data and params ---
        df_out.to_csv(out_csv, index=False)

        n_clusters = len(np.unique(labels_final))
        if n_clusters >= 2:
            if n_samples > 5000:
                rng = np.random.default_rng(42)
                idx = rng.choice(n_samples, min(5000, n_samples), replace=False)
                sil_final = silhouette_score(X[idx], labels_final[idx])
            else:
                sil_final = silhouette_score(X, labels_final)
        else:
            sil_final = 0.0

        print(f"K-Means [{cfg.name}]: k={best_k}, clusters={n_clusters}, silhouette={sil_final:.4f} -> {out_csv}")
        with open(os.path.join(out_dir, f"{cfg.name}_params.txt"), "w", encoding="utf-8") as f:
            f.write(f"k={best_k}\ninertia={km_final.inertia_:.4f}\nsilhouette={sil_final:.4f}\noutput={out_csv}\n")


if __name__ == "__main__":
    main()

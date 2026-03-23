"""
4. Clustering Implementation - DBSCAN (CRISP-DM)

Runs DBSCAN on A, B, C scaled feature datasets.
Uses heuristic for eps/min_samples; fits and saves cluster labels (-1 = noise).

Inputs: this_datasets/02_data_transformation/clustering_*_features_minmax.csv
Outputs: this_datasets/04_clustering/clustering_*_dbscan.csv (original + cluster_dbscan column)
         webp/EDA_and_visualization/04_clustering_implementation/dbscan/ (params)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from _common import _ensure_dir, get_dataset_configs, load_features


def _project_root() -> str:
    return os.path.dirname(script_dir)


def _suggest_eps(X: np.ndarray, k: int = 5) -> float:
    n_samples = X.shape[0]
    k = min(k, n_samples - 1)
    if k < 1:
        return 0.5
    n = min(n_samples, 1000)
    rng = np.random.default_rng(42)
    idx = rng.choice(n_samples, n, replace=False) if n_samples > n else np.arange(n_samples)
    X_sample = X[idx]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_sample)
    dists, _ = nn.kneighbors(X_sample)
    return float(np.median(dists[:, k]))


def fit_dbscan(X: np.ndarray, eps: float, min_samples: int):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(X)
    return db, labels


def main() -> None:
    # --- Step 0: Setup paths ---
    root = _project_root()
    cluster_dir = os.path.join(root, "this_datasets", "04_clustering")
    out_dir = os.path.join(root, "webp", "EDA_and_visualization", "04_clustering_implementation", "dbscan")
    _ensure_dir(out_dir)

    for cfg in get_dataset_configs():
        # --- Step 1: Load minmax-scaled features ---
        result = load_features(cfg)
        if result is None:
            print(f"DBSCAN [{cfg.name}]: Skip (file missing or empty)")
            continue

        df, _, X = result
        n_samples, n_features = X.shape
        # --- Step 2: Parameter Tuning (eps heuristic, min_samples) ---
        min_samples = max(2, min(10, 2 * n_features))
        eps_suggested = _suggest_eps(X, k=min(5, n_samples - 1))
        eps_candidates = [round(e, 4) for e in [eps_suggested * 0.8, eps_suggested, eps_suggested * 1.2, eps_suggested * 1.5]]

        best_eps = eps_suggested
        best_labels = None
        best_n_clusters = 0
        best_noise_ratio = 1.0

        for eps in eps_candidates:
            if eps <= 0:
                continue
            _, labels = fit_dbscan(X, eps, min_samples)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = (labels == -1).sum() / n_samples
            if n_clusters >= 2 and noise_ratio < 0.7:
                best_eps = eps
                best_labels = labels
                best_n_clusters = n_clusters
                best_noise_ratio = noise_ratio
                break
            elif best_labels is None and n_clusters >= 1:
                best_eps = eps
                best_labels = labels
                best_n_clusters = n_clusters
                best_noise_ratio = noise_ratio

        if best_labels is None:
            best_eps = eps_candidates[0]
            _, best_labels = fit_dbscan(X, best_eps, min_samples)
            best_n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
            best_noise_ratio = (best_labels == -1).sum() / n_samples

        # --- Step 4: Fit DBSCAN, assign labels, save (-1 = noise) ---
        df_out = df.copy()
        df_out["cluster_dbscan"] = best_labels
        out_csv = os.path.join(cluster_dir, f"clustering_{cfg.name}_dbscan.csv")
        df_out.to_csv(out_csv, index=False)

        noise_count = int((best_labels == -1).sum())
        sil = silhouette_score(X[best_labels >= 0], best_labels[best_labels >= 0]) if best_n_clusters >= 2 else 0.0

        print(f"DBSCAN [{cfg.name}]: eps={best_eps:.4f}, min_samples={min_samples}, clusters={best_n_clusters}, noise={noise_count} ({best_noise_ratio*100:.1f}%), sil={sil:.4f} -> {out_csv}")
        with open(os.path.join(out_dir, f"{cfg.name}_params.txt"), "w", encoding="utf-8") as f:
            f.write(f"eps={best_eps}\nmin_samples={min_samples}\nn_clusters={best_n_clusters}\nnoise_count={noise_count}\nnoise_ratio={best_noise_ratio:.4f}\nsilhouette={sil:.4f}\noutput={out_csv}\n")


if __name__ == "__main__":
    main()

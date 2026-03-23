"""
5. Cluster Analysis - DBSCAN (CRISP-DM)

Analyzes DBSCAN clustering results: cluster sizes, noise count, feature profiles per cluster.
Inputs: this_datasets/04_clustering/clustering_*_dbscan.csv
Outputs: webp/EDA_and_visualization/05_cluster_analysis/dbscan/
         - {name}_cluster_sizes.csv, {name}_feature_profiles.csv
         - per_cluster/{name}/cluster_{i}_data.csv, cluster_{i}_profile.csv, cluster_noise_*
"""

from __future__ import annotations

import os
import sys

import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from _common import _ensure_dir, get_dataset_configs


def _project_root() -> str:
    return os.path.dirname(script_dir)


def main() -> None:
    # --- Step 0: Setup paths ---
    root = _project_root()
    cluster_dir = os.path.join(root, "this_datasets", "04_clustering")
    out_dir = os.path.join(root, "webp", "EDA_and_visualization", "05_cluster_analysis", "dbscan")
    _ensure_dir(out_dir)

    for cfg in get_dataset_configs():
        # --- Step 1: Load DBSCAN labeled data ---
        dbscan_csv = os.path.join(cluster_dir, f"clustering_{cfg.name}_dbscan.csv")
        if not os.path.exists(dbscan_csv) or os.path.getsize(dbscan_csv) == 0:
            print(f"Cluster Analysis DBSCAN [{cfg.name}]: Skip (no dbscan output)")
            continue

        df = pd.read_csv(dbscan_csv, low_memory=False)
        if "cluster_dbscan" not in df.columns:
            print(f"Cluster Analysis DBSCAN [{cfg.name}]: Skip (no cluster_dbscan column)")
            continue

        cluster_col = "cluster_dbscan"
        num_cols = [c for c in df.columns if c != cluster_col and pd.api.types.is_numeric_dtype(df[c])]

        # Cluster sizes (including noise = -1)
        sizes = df[cluster_col].value_counts().sort_index()
        summary = pd.DataFrame({"cluster": sizes.index, "count": sizes.values, "pct": (sizes.values / len(df) * 100).round(2)})
        summary.to_csv(os.path.join(out_dir, f"{cfg.name}_cluster_sizes.csv"), index=False)

        noise_count = int((df[cluster_col] == -1).sum())
        n_clusters = len(sizes) - (1 if -1 in sizes.index else 0)

        # --- Step 3: Feature profiles (mean per cluster, exclude noise) ---
        in_cluster = df[cluster_col] >= 0
        if num_cols and in_cluster.any():
            profile = df.loc[in_cluster].groupby(cluster_col)[num_cols].mean().round(4)
            profile.to_csv(os.path.join(out_dir, f"{cfg.name}_feature_profiles.csv"))

        # --- Step 4: Per-cluster outputs (data + profile each; noise as cluster_noise) ---
        dataset_out = os.path.join(out_dir, "per_cluster", cfg.name)
        _ensure_dir(dataset_out)
        n_total = len(df)
        for cid in sizes.index:
            sub = df[df[cluster_col] == cid]
            name = "cluster_noise" if cid == -1 else f"cluster_{int(cid)}"
            sub.to_csv(os.path.join(dataset_out, f"{name}_data.csv"), index=False)
            prof_row = {"count": len(sub), "pct": round(len(sub) / n_total * 100, 2)}
            if num_cols:
                for col in num_cols:
                    prof_row[f"{col}_mean"] = round(sub[col].mean(), 4)
            pd.DataFrame([prof_row]).to_csv(os.path.join(dataset_out, f"{name}_profile.csv"), index=False)
        print(f"Cluster Analysis DBSCAN [{cfg.name}]: n_clusters={n_clusters}, noise={noise_count}, per-cluster outputs -> {dataset_out}")

    print("Cluster Analysis DBSCAN complete.")


if __name__ == "__main__":
    main()

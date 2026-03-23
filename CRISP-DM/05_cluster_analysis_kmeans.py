"""
5. Cluster Analysis - K-Means (CRISP-DM)

Analyzes K-Means clustering results: cluster sizes, feature profiles per cluster.
Inputs: this_datasets/04_clustering/clustering_*_kmeans.csv
Outputs: webp/EDA_and_visualization/05_cluster_analysis/kmeans/
         - {name}_cluster_sizes.csv, {name}_feature_profiles.csv
         - per_cluster/{name}/cluster_{i}_data.csv, cluster_{i}_profile.csv
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
    out_dir = os.path.join(root, "webp", "EDA_and_visualization", "05_cluster_analysis", "kmeans")
    _ensure_dir(out_dir)

    for cfg in get_dataset_configs():
        # --- Step 1: Load K-Means labeled data ---
        kmeans_csv = os.path.join(cluster_dir, f"clustering_{cfg.name}_kmeans.csv")
        if not os.path.exists(kmeans_csv) or os.path.getsize(kmeans_csv) == 0:
            print(f"Cluster Analysis K-Means [{cfg.name}]: Skip (no kmeans output)")
            continue

        df = pd.read_csv(kmeans_csv, low_memory=False)
        if "cluster_kmeans" not in df.columns:
            print(f"Cluster Analysis K-Means [{cfg.name}]: Skip (no cluster_kmeans column)")
            continue

        num_cols = [c for c in df.columns if c != "cluster_kmeans" and pd.api.types.is_numeric_dtype(df[c])]
        cluster_col = "cluster_kmeans"

        # --- Step 2: Compute cluster sizes summary ---
        sizes = df[cluster_col].value_counts().sort_index()
        summary = pd.DataFrame({"cluster": sizes.index, "count": sizes.values, "pct": (sizes.values / len(df) * 100).round(2)})
        summary.to_csv(os.path.join(out_dir, f"{cfg.name}_cluster_sizes.csv"), index=False)

        # --- Step 3: Feature profiles (mean per cluster) ---
        if num_cols:
            profile = df.groupby(cluster_col)[num_cols].mean().round(4)
            profile.to_csv(os.path.join(out_dir, f"{cfg.name}_feature_profiles.csv"))

        # --- Step 4: Per-cluster outputs (data CSV + profile CSV each) ---
        dataset_out = os.path.join(out_dir, "per_cluster", cfg.name)
        _ensure_dir(dataset_out)
        n_total = len(df)
        for cid in sizes.index:
            sub = df[df[cluster_col] == cid]
            name = f"cluster_{int(cid)}"
            sub.to_csv(os.path.join(dataset_out, f"{name}_data.csv"), index=False)
            prof_row = {"count": len(sub), "pct": round(len(sub) / n_total * 100, 2)}
            if num_cols:
                for col in num_cols:
                    prof_row[f"{col}_mean"] = round(sub[col].mean(), 4)
            pd.DataFrame([prof_row]).to_csv(os.path.join(dataset_out, f"{name}_profile.csv"), index=False)
        print(f"Cluster Analysis K-Means [{cfg.name}]: n_clusters={len(sizes)}, per-cluster outputs -> {dataset_out}")

    print("Cluster Analysis K-Means complete.")


if __name__ == "__main__":
    main()

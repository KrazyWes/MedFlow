"""
Step 05a — PhilGEPS k-means cluster profiles.

Mirrors the DOH cluster analysis script: descriptive tables + per-cluster CSV slices for each
PhilGEPS lens that finished step 04.

Inputs: this_datasets/PhilGEPS/04_clustering/*_kmeans.csv
Outputs: webp/EDA_and_visualization/PhilGEPS/05_cluster_analysis/kmeans/
"""

from __future__ import annotations

import os
import sys

import pandas as pd

# Shared `_common`, `sources_paths`, etc. live in parent `CRISP-DM/`.
script_dir = os.path.dirname(os.path.abspath(__file__))
_crisp_dm_root = os.path.dirname(script_dir)
if _crisp_dm_root not in sys.path:
    sys.path.insert(0, _crisp_dm_root)

from _common import _ensure_dir, get_philgeps_dataset_configs
from log_tee import tee_stdio_to_file
from sources_paths import data_root_philgeps, logs_dir_philgeps, webp_root_philgeps


def main() -> None:
    cluster_dir = os.path.join(data_root_philgeps(), "04_clustering")
    out_dir = os.path.join(webp_root_philgeps(), "05_cluster_analysis", "kmeans")
    _ensure_dir(out_dir)
    term_log = os.path.join(logs_dir_philgeps(), "05_cluster_analysis_kmeans_philgeps_terminal.txt")
    with tee_stdio_to_file(term_log):
        _run(cluster_dir, out_dir)


def _run(cluster_dir: str, out_dir: str) -> None:
    for cfg in get_philgeps_dataset_configs():
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
        # `value_counts()` counts how many rows were assigned to each cluster id.
        sizes = df[cluster_col].value_counts().sort_index()
        summary = pd.DataFrame({"cluster": sizes.index, "count": sizes.values, "pct": (sizes.values / len(df) * 100).round(2)})
        summary.to_csv(os.path.join(out_dir, f"{cfg.name}_cluster_sizes.csv"), index=False)

        # --- Step 3: Feature profiles (mean per cluster) ---
        if num_cols:
            # Profile uses mean values of each numeric feature, per cluster.
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
            # These per-cluster exports are used for deep-dive interpretation in the report.
        print(f"Cluster Analysis K-Means [{cfg.name}]: n_clusters={len(sizes)}, per-cluster outputs -> {dataset_out}")

    print("Cluster Analysis K-Means complete.")


if __name__ == "__main__":
    main()

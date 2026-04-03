"""
Step 05b — PhilGEPS DBSCAN cluster profiles.

Noise-aware summaries for every PhilGEPS config with a DBSCAN labeling CSV.

Inputs: this_datasets/PhilGEPS/04_clustering/*_dbscan.csv
Outputs: webp/EDA_and_visualization/PhilGEPS/05_cluster_analysis/dbscan/
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
    out_dir = os.path.join(webp_root_philgeps(), "05_cluster_analysis", "dbscan")
    _ensure_dir(out_dir)
    term_log = os.path.join(logs_dir_philgeps(), "05_cluster_analysis_dbscan_philgeps_terminal.txt")
    with tee_stdio_to_file(term_log):
        _run(cluster_dir, out_dir)


def _run(cluster_dir: str, out_dir: str) -> None:
    for cfg in get_philgeps_dataset_configs():
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
        # Create an easy summary table for thesis tables/figures.
        summary = pd.DataFrame({"cluster": sizes.index, "count": sizes.values, "pct": (sizes.values / len(df) * 100).round(2)})
        summary.to_csv(os.path.join(out_dir, f"{cfg.name}_cluster_sizes.csv"), index=False)

        noise_count = int((df[cluster_col] == -1).sum())
        n_clusters = len(sizes) - (1 if -1 in sizes.index else 0)

        # --- Step 3: Feature profiles (mean per cluster, exclude noise) ---
        in_cluster = df[cluster_col] >= 0
        if num_cols and in_cluster.any():
            # Compute cluster “profiles” from assigned points only; noise is excluded (-1).
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
            # Each cluster gets a small “profile row” (mean values + size) to support qualitative interpretation.
            pd.DataFrame([prof_row]).to_csv(os.path.join(dataset_out, f"{name}_profile.csv"), index=False)
        print(f"Cluster Analysis DBSCAN [{cfg.name}]: n_clusters={n_clusters}, noise={noise_count}, per-cluster outputs -> {dataset_out}")

    print("Cluster Analysis DBSCAN complete.")


if __name__ == "__main__":
    main()

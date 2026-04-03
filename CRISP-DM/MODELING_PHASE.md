# Modeling phase (clustering)

This file is the **methodology cheat sheet**: what algorithms run, what files they read, and what metrics appear in step 7. For disk paths and filenames, use `OUTPUT_LAYOUT.md`. For commands and script locations, use `RUN_PIPELINE.md`.

---

## Scope

Unsupervised clustering on **MinMax-scaled** feature matrices produced in step 2. Each **lens** (DOH A–E, PhilGEPS A–G) is a separate table with its own engineered columns — see `get_doh_dataset_configs` / `get_philgeps_dataset_configs` in `_common.py`.

**End-to-end:** `python main.py` → `CRISP-DM/run_all.py` → all DOH steps 01–08, then PhilGEPS 01–08.

---

## Algorithms

| Algorithm | Scripts | What it does |
|-----------|---------|----------------|
| **K-means** | `DOH/04_clustering_implementation_kmeans_doh.py`, `PhilGEPS/04_clustering_implementation_kmeans_philgeps.py` | Partitions into k clusters; k chosen by silhouette over a bounded grid (with safeguards for small n). |
| **DBSCAN** | `DOH/04_clustering_implementation_dbscan_doh.py`, `PhilGEPS/04_clustering_implementation_dbscan_philgeps.py` | Density clusters; label `-1` = noise/outliers. ε from k-distance style heuristics; `min_samples` scales with feature dimensionality. |

Scripts live under `CRISP-DM/DOH/` and `CRISP-DM/PhilGEPS/`; run with working directory `CRISP-DM/` (as `run_all.py` does).

---

## Inputs and outputs (modeling-relevant)

| Artifact | Path pattern |
|----------|----------------|
| Scaled features (model input) | `this_datasets/{DOH|PhilGEPS}/02_data_transformation/clustering_*_features_minmax.csv` |
| K-means labels | `…/04_clustering/clustering_*_kmeans.csv` (`cluster_kmeans`) |
| DBSCAN labels | `…/04_clustering/clustering_*_dbscan.csv` (`cluster_dbscan`) |

Downstream steps only consume these labeled CSVs plus the same feature columns for silhouette / PCA plots.

---

## Parameter tuning (short)

**K-means:** sweep k, pick k with best silhouette (subsample if n is huge so the metric is affordable).

**DBSCAN:** ε multipliers around a k-NN suggestion; prefer solutions with at least two clusters and a sane noise rate. Large-N PhilGEPS uses subsampling **only** during the ε search, not for the final assignment.

Artifacts (plots, `*_params.txt`) sit under `webp/EDA_and_visualization/.../04_clustering_implementation/`.

---

## Steps after clustering

| Step | Role |
|------|------|
| 05 `*_cluster_analysis_*` | Cluster sizes, per-cluster CSV exports, feature profiles |
| 06 `*_visualization_*` | PCA 2D/3D and per-cluster views |
| 07 `*_evaluation_*` | Metric figures + text summaries |
| 08 `*_final_output_bundle_*` | Condensed “thesis pack” per lens × algorithm (`output_bundle.py`) |

---

## Evaluation metrics

| Metric | K-means | DBSCAN |
|--------|---------|--------|
| Silhouette | yes | yes (excluding noise `-1`) |
| Calinski–Harabasz | yes | yes when ≥2 non-noise clusters |
| Davies–Bouldin | yes | yes when ≥2 non-noise clusters |
| Noise share | — | yes (DBSCAN-specific) |

Figures: `webp/EDA_and_visualization/{DOH|PhilGEPS}/07_evaluation/{kmeans|dbscan}/`.

---

## Mapping to research goals

| Goal | Where it shows up |
|------|-------------------|
| Separate behavioral groups per lens | One clustering run per `DatasetConfig` |
| Compare k-means vs DBSCAN | Parallel outputs + step 7 + step 8 bundles |
| Explain clusters to a reader | Step 6–7 plots + step 8 top-10 / radar / CSV |

---

## Manual rerun order

From `CRISP-DM/`:

1. `DOH/01_*` → `02_*` → `03_*`
2. `DOH/04_kmeans_*` → `DOH/04_dbscan_*`
3. `DOH/05_*` … `DOH/08_*`

Repeat with `PhilGEPS/` and `*_philgeps.py`.

Or run `python main.py` once from the repo root.

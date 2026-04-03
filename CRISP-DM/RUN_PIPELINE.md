# MedFlow CRISP-DM pipeline

This document is the **operator’s view**: how to run the code and where scripts live. For the folder tree after a successful run, use `OUTPUT_LAYOUT.md`. For clustering methodology, use `MODELING_PHASE.md`.

## Data flow (one sentence)

Raw files in `raw_datasets/{DOH|PhilGEPS}/` → cleaned CSVs → engineered feature matrices (MinMax for clustering) → cluster labels (k-means + DBSCAN) → figures and evaluation under `webp/` → step 8 “bundle” packs per lens.

Everything downstream of raw data is **split by source**: parallel trees under `this_datasets/`, `webp/EDA_and_visualization/`, and `webp/logs/`.

---

## Run commands

**Full clean rebuild** (typical before archiving a thesis run):

```bash
python main.py --fresh
```

That runs `CRISP-DM/00_clear_pipeline_outputs.py` (wipes regenerated assets; **never** deletes `raw_datasets/`), then `CRISP-DM/run_all.py`.

**Clear only** (no clustering rerun):

```bash
python CRISP-DM/00_clear_pipeline_outputs.py
```

**Incremental run** (reuse existing CSVs/figures where steps succeed):

```bash
python main.py
```

---

## Where the Python files are

| Location | Contents |
|----------|----------|
| `CRISP-DM/DOH/` | DOH steps `01`–`08` (`*_doh.py`) |
| `CRISP-DM/PhilGEPS/` | PhilGEPS steps `01`–`08` (`*_philgeps.py`) |
| `CRISP-DM/` (root of that folder) | `run_all.py`, `00_clear_pipeline_outputs.py`, `_common.py`, `sources_paths.py`, `output_bundle.py`, `log_tee.py` |

Each step script adds `CRISP-DM/` to `sys.path` so imports resolve no matter which subfolder the file sits in.

---

## Step order (must stay in this sequence)

| Step | DOH (`CRISP-DM/DOH/`) | PhilGEPS (`CRISP-DM/PhilGEPS/`) |
|------|----------------------|----------------------------------|
| 1 | `01_data_cleaning_doh.py` | `01_data_cleaning_philgeps.py` |
| 2 | `02_data_transformation_doh.py` | `02_data_transformation_philgeps.py` |
| 3 | `03_exploratory_data_analysis_doh.py` | `03_exploratory_data_analysis_philgeps.py` |
| 4 | `04_clustering_implementation_kmeans_*`, then `04_clustering_implementation_dbscan_*` | same file names |
| 5 | `05_cluster_analysis_kmeans_*`, `05_cluster_analysis_dbscan_*` | same |
| 6 | `06_visualization_kmeans_*`, `06_visualization_dbscan_*` | same |
| 7 | `07_evaluation_kmeans_*`, `07_evaluation_dbscan_*` | same |
| 8 | `08_final_output_bundle_doh.py` | `08_final_output_bundle_philgeps.py` |

K-means is always run **before** DBSCAN at step 4 so both algorithms read the same feature files. Steps 5–7 repeat k-means then DBSCAN so logs and folders stay predictable.

---

## Datasets (analysis lenses)

- **DOH (A–E)** — `DOH_A_distribution_recipient` … `DOH_E_unequal_supply_regions`: recipient-centric views (volume, shortage risk, overstocking, inefficient delivery patterns, regional inequality).
- **PhilGEPS (A–G)** — `PhilGEPS_A_supplier_awardee` … `PhilGEPS_G_unequal_supply_regions`: supplier totals, line-level procurement, then awardee×region unit plus the same thematic lenses as DOH B–E.

Config objects (paths + ID columns) live in `_common.py`: `get_doh_dataset_configs()` and `get_philgeps_dataset_configs()`.

---

## Step 8 output bundle

Under each source: `webp/EDA_and_visualization/{DOH|PhilGEPS}/08_output_to_use/{kmeans|dbscan}/{slug}/`.

Per folder you get PCA scatter (+ flagged overlay), top-10 bar, silhouette figure, centroid heatmap, radar chart, and `export_labeled.csv`. See the table in `OUTPUT_LAYOUT.md` for filenames.

The **slug** is the config name without the `DOH_` / `PhilGEPS_` prefix (e.g. `B_high_risk_shortage`).

---

## Logs

Every step tees stdout/stderr to `webp/logs/{DOH|PhilGEPS}/<step>_*_terminal.txt` while also printing to the console. When something fails halfway through a long run, open the last log file first.

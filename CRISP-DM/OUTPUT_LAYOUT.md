# Expected outputs after a full run

Use this file when you need to **find a specific CSV or figure** after `python main.py` (or `--fresh`). It does not replace reading the scripts for *how* numbers were computed — pair it with `MODELING_PHASE.md` for methodology.

---

## Prerequisites (you supply)

- `raw_datasets/DOH/` — spreadsheets/CSVs consumed by `CRISP-DM/DOH/01_data_cleaning_doh.py`
- `raw_datasets/PhilGEPS/` — source file(s) for `CRISP-DM/PhilGEPS/01_data_cleaning_philgeps.py`

---

## Run command

```bash
python main.py --fresh
```

`--fresh` deletes regenerated trees first; omit it to keep existing `this_datasets/` and `webp/` content. `raw_datasets/` is never removed by the clear script.

A full run can take **tens of minutes** (PhilGEPS B is large). Success ends with:

```text
All pipeline steps completed successfully.
```

---

## 1. Processed tables (`this_datasets/`)

```
this_datasets/
├── DOH/
│   ├── 01_data_cleaning/
│   │   └── doh_medicine_distribution_2022_2025.csv
│   ├── 02_data_transformation/
│   │   ├── clustering_DOH_*_features.csv
│   │   ├── clustering_DOH_*_features_zscore.csv
│   │   ├── clustering_DOH_*_features_minmax.csv   ← input to step 04
│   │   └── …
│   └── 04_clustering/
│       ├── clustering_DOH_*_kmeans.csv          ← + cluster_kmeans column
│       └── clustering_DOH_*_dbscan.csv           ← + cluster_dbscan (-1 = noise)
└── PhilGEPS/
    ├── 01_data_cleaning/
    ├── 02_data_transformation/
    └── 04_clustering/
        └── (same pattern, PhilGEPS_A … G)
```

Clustering CSVs repeat the **same rows** as the corresponding `*_features_minmax.csv`, with one extra label column.

---

## 2. Figures (`webp/EDA_and_visualization/`)

Under each source, folder names mirror the CRISP-DM step:

```
webp/EDA_and_visualization/
├── DOH/
│   ├── 01_data_cleaning/
│   ├── 02_data_transformation/
│   ├── 03_exploratory_data_analysis/
│   ├── 04_clustering_implementation/{kmeans,dbscan}/
│   ├── 05_cluster_analysis/{kmeans,dbscan}/
│   ├── 06_visualization/{kmeans,dbscan}/
│   ├── 07_evaluation/{kmeans,dbscan}/
│   └── 08_output_to_use/{kmeans,dbscan}/{slug}/
└── PhilGEPS/
    └── (same step folders; step 08 has slugs A … G)
```

### Step 08 bundle (per `slug` × algorithm)

| File | What it is |
|------|------------|
| `cluster_scatter_pca2d.png` | Rows in PCA space, colored by cluster |
| `cluster_scatter_pca2d_flagged.png` | Same + top 10 “extreme” units (same rule as top-10 bar) |
| `top10_bar.png` | Highest thematic scores for that lens |
| `silhouette_plot.png` | Silhouette diagnostic (or placeholder if not applicable) |
| `clustermap_centroids.png` | Heatmap of cluster centroids |
| `web_chart_radar.png` | Normalized centroid shape (radar) |
| `export_labeled.csv` | IDs + features + cluster column for reuse in other tools |

**DOH slugs:** `A_distribution_recipient` … `E_unequal_supply_regions`  
**PhilGEPS slugs:** `A_supplier_awardee` … `G_unequal_supply_regions`

Scoring logic for top-10 / flagged plots: `CRISP-DM/output_bundle.py` (`_thematic_top10_scores`).

---

## 3. Terminal logs (`webp/logs/`)

```
webp/logs/
├── DOH/
│   ├── 01_data_cleaning_doh_terminal.txt
│   ├── …
│   └── 08_final_output_bundle_doh_terminal.txt
└── PhilGEPS/
    └── … *_philgeps_terminal.txt
```

Each file is a full copy of that step’s console output (`log_tee.tee_stdio_to_file`).

---

## 4. Not produced here

- Anything under `raw_datasets/**` — your responsibility; never auto-deleted.

---

## 5. PhilGEPS DBSCAN stuck or partial

**Symptom:** Only some `PhilGEPS/04_clustering/*_dbscan.csv` exist, or step 08 is incomplete.

**Cause:** Dataset B has ~35k rows. Older code ran expensive full-matrix sweeps repeatedly; runs often hit IDE timeouts.

**What the current scripts do:** For large `n`, the ε sweep uses a **fixed subsample**; the final DBSCAN fit still uses **all** rows. Very large-N figures may use a single ε panel to avoid redundant full fits. Details: `PhilGEPS/04_clustering_implementation_dbscan_philgeps.py`.

**Resume from DBSCAN** (from `CRISP-DM/`):

```bash
python PhilGEPS/04_clustering_implementation_dbscan_philgeps.py
python PhilGEPS/05_cluster_analysis_kmeans_philgeps.py
python PhilGEPS/05_cluster_analysis_dbscan_philgeps.py
python PhilGEPS/06_visualization_kmeans_philgeps.py
python PhilGEPS/06_visualization_dbscan_philgeps.py
python PhilGEPS/07_evaluation_kmeans_philgeps.py
python PhilGEPS/07_evaluation_dbscan_philgeps.py
python PhilGEPS/08_final_output_bundle_philgeps.py
```

Budget **~10–15+ minutes** for PhilGEPS DBSCAN on a typical laptop.

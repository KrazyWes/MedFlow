# CRISP-DM Modeling Phase

The Modeling phase applies unsupervised clustering algorithms to prepared data to identify patterns and group similar facilities/medicines by procurement and distribution behavior.

---

## Algorithm Selection ✓

| Algorithm | Script | Purpose |
|-----------|--------|---------|
| **K-Means** | `04_clustering_implementation_kmeans.py` | Partitions data into k clusters (pre-defined number) |
| **DBSCAN** | `04_clustering_implementation_dbscan.py` | Detects clusters of varying shapes, handles noise/outliers (-1) |

---

## Parameter Tuning ✓

### K-Means
- **Elbow Method**: Generates elbow plot (`*_elbow.png`) to suggest k
- **Silhouette Score**: Automatically selects k ∈ [2, k_max] that maximizes silhouette
- **Output**: `*_params.txt` with k, inertia, silhouette

### DBSCAN
- **epsilon (ε)**: Heuristic via k-NN distance median; candidates = 0.8×, 1×, 1.2×, 1.5× suggested ε
- **min_samples**: `max(2, min(10, 2×n_features))`
- **Selection**: Prefers ≥2 clusters and &lt;70% noise; falls back to best available
- **Output**: `*_params.txt` with eps, min_samples, n_clusters, noise_count, silhouette

---

## Model Training & Cluster Assignment ✓

| Dataset | Entities Clustered | Features |
|---------|--------------------|----------|
| **A_supplier_awardee** | PhilGEPS suppliers/awardees | contract_amount_total, num_awards, regions_served, etc. |
| **B_medicine_procurement_pattern** | Procurement patterns (awardee × item × region × mode × funding) | item_budget, quantity, procurement mode, funding source |
| **C_distribution_recipient** | DOH recipients (facilities) | medicines_received, quantity_total, delivery_frequency |

- **Input**: MinMax-scaled features from `02_data_transformation`
- **Output**: Each row gets `cluster_kmeans` or `cluster_dbscan` (DBSCAN: -1 = noise)

---

## Evaluation Metrics ✓

| Metric | K-Means | DBSCAN | Interpretation |
|--------|---------|--------|----------------|
| **Silhouette** | ✓ | ✓ (excl. noise) | Cohesion vs separation |
| **Calinski–Harabasz** | ✓ | — | Cluster separation |
| **Davies–Bouldin** | ✓ | — | Cohesion + separation |
| **Noise ratio** | — | ✓ | % of points labeled as noise |

Output: `07_evaluation/*/*_evaluation.txt`

---

## Mapping to Study Objectives

| Objective | Implementation |
|-----------|----------------|
| *Group public health facilities based on procurement and distribution behavior* | A (suppliers), B (procurement patterns), C (distribution recipients) clustering |
| *Cluster summaries, visualizations, comparative profiles* | `05_cluster_analysis` (sizes, feature profiles), `06_visualization` (PCA plots, bar charts) |
| *Evaluate K-means vs DBSCAN with silhouette, cohesion, separation* | `07_evaluation` (silhouette, CH, Davies–Bouldin) |

---

## Run Order

1. `01_data_cleaning_doh.py` + `01_data_cleaning_philgeps.py`
2. `02_data_transformation_doh.py` + `02_data_transformation_philgeps.py`
3. `04_clustering_implementation_kmeans.py`
4. `04_clustering_implementation_dbscan.py`
5. `05_cluster_analysis_kmeans.py` + `05_cluster_analysis_dbscan.py`
6. `06_visualization_kmeans.py` + `06_visualization_dbscan.py`
7. `07_evaluation_kmeans.py` + `07_evaluation_dbscan.py`

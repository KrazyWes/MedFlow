# MedFlow CRISP-DM Pipeline (Clustering)

Run scripts in order. Each step depends on outputs from previous steps.

## Pipeline Structure

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_data_cleaning_doh.py` | Clean DOH medicine distribution data |
| 1 | `01_data_cleaning_philgeps.py` | Clean PhilGEPS medical procurement data |
| 2 | `02_data_transformation_doh.py` | Transform DOH → C (distribution features) |
| 2 | `02_data_transformation_philgeps.py` | Transform PhilGEPS → A, B (supplier + procurement features) |
| 3 | `03_exploratory_data_analysis_doh.py` | EDA on DOH data |
| 3 | `03_exploratory_data_analysis_philgeps.py` | EDA on PhilGEPS data |
| 4 | `04_clustering_implementation_kmeans.py` | K-Means on A, B, C |
| 4 | `04_clustering_implementation_dbscan.py` | DBSCAN on A, B, C |
| 5 | `05_cluster_analysis_kmeans.py` | Cluster profiles (K-Means) |
| 5 | `05_cluster_analysis_dbscan.py` | Cluster profiles (DBSCAN) |
| 6 | `06_visualization_kmeans.py` | PCA scatter plots (K-Means) |
| 6 | `06_visualization_dbscan.py` | PCA scatter plots (DBSCAN) |
| 7 | `07_evaluation_kmeans.py` | Silhouette, CH, DB scores (K-Means) |
| 7 | `07_evaluation_dbscan.py` | Silhouette, noise ratio (DBSCAN) |

## Quick Run (full pipeline)

```bash
cd CRISP-DM
python 01_data_cleaning_doh.py
python 01_data_cleaning_philgeps.py
python 02_data_transformation_doh.py
python 02_data_transformation_philgeps.py
python 03_exploratory_data_analysis_doh.py
python 03_exploratory_data_analysis_philgeps.py
python 04_clustering_implementation_kmeans.py
python 04_clustering_implementation_dbscan.py
python 05_cluster_analysis_kmeans.py
python 05_cluster_analysis_dbscan.py
python 06_visualization_kmeans.py
python 06_visualization_dbscan.py
python 07_evaluation_kmeans.py
python 07_evaluation_dbscan.py
```

## Output Paths

All EDA and visualization outputs are under `webp/EDA_and_visualization/`, matching pipeline structure:

| Step | Script | Output folder |
|------|--------|---------------|
| 1 | 01_data_cleaning_* | `01_data_cleaning/steps/`, `01_data_cleaning/data_understanding/` |
| 2 | 02_data_transformation_* | `02_data_transformation/steps/` |
| 3 | 03_exploratory_data_analysis_* | `03_exploratory_data_analysis/DOH/`, `03_exploratory_data_analysis/PhilGEPS/` |
| 4 | 04_clustering_implementation_* | `04_clustering_implementation/kmeans/`, `04_clustering_implementation/dbscan/` |
| 5 | 05_cluster_analysis_* | `05_cluster_analysis/kmeans/`, `05_cluster_analysis/dbscan/` |
| 6 | 06_visualization_* | `06_visualization/kmeans/`, `06_visualization/dbscan/` |
| 7 | 07_evaluation_* | `07_evaluation/kmeans/`, `07_evaluation/dbscan/` |

Logs: `webp/logs/`

## Datasets (A, B, C)

- **A**: Supplier/Awardee clustering (PhilGEPS)
- **B**: Medicine procurement pattern (PhilGEPS)
- **C**: Distribution recipient clustering (DOH)

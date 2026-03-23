# EDA and Visualization Outputs

Output folder structure mirrors the CRISP-DM pipeline. Run scripts from `CRISP-DM/` to populate.

## Folder Structure

| Step | Folder | Scripts | Contents |
|------|--------|---------|----------|
| 1 | `01_data_cleaning/` | `01_data_cleaning_doh.py`, `01_data_cleaning_philgeps.py` | `steps/` (01_load_raw…06_final_clean), `data_understanding/` |
| 2 | `02_data_transformation/` | `02_data_transformation_doh.py`, `02_data_transformation_philgeps.py` | `steps/` (07_feature_engineering…11_final_dataset_structure) |
| 3 | `03_exploratory_data_analysis/` | `03_exploratory_data_analysis_doh.py`, `03_exploratory_data_analysis_philgeps.py` | `DOH/`, `PhilGEPS/` |
| 4 | `04_clustering_implementation/` | `04_clustering_implementation_kmeans.py`, `04_clustering_implementation_dbscan.py` | `kmeans/` (elbow, params), `dbscan/` (params) |
| 5 | `05_cluster_analysis/` | `05_cluster_analysis_kmeans.py`, `05_cluster_analysis_dbscan.py` | `kmeans/`, `dbscan/` (sizes, profiles) |
| 6 | `06_visualization/` | `06_visualization_kmeans.py`, `06_visualization_dbscan.py` | `kmeans/`, `dbscan/` (PCA plots, cluster charts) |
| 7 | `07_evaluation/` | `07_evaluation_kmeans.py`, `07_evaluation_dbscan.py` | `kmeans/`, `dbscan/` (silhouette, CH, DB scores) |

## Regenerate

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

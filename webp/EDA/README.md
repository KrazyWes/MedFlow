# Exploratory Data Analysis (EDA) Outputs

Step-based EDA visualizations produced during data preparation and preprocessing.

## Folder Structure

### `data_preparation/steps/`
EDA at each data preparation step:

| Step | Folder | Description |
|------|--------|-------------|
| 01 | `01_load_raw/` | Raw merged data (DOH, PhilGEPS) – missingness, histograms, categorical top, correlation |
| 02 | `02_fix_data_types/` | After currency/header cleaning (DOH) |
| 03 | `03_filter_relevant_data/` | After medical filter (PhilGEPS only) |
| 04 | `04_remove_duplicates/` | step_log.txt (dedup counts) |
| 05 | `05_handle_missing_values/` | Post-imputation |
| 06 | `06_final_clean/` | step_log.txt (export paths) |

### `data_preprocessing/steps/`
EDA for clustering feature tables (A/B/C):

| Step | Folder | Description |
|------|--------|-------------|
| 07 | `07_feature_engineering/` | Feature distributions, correlation (A, B, C) |
| 08 | `08_encode_categorical/` | step_log.txt (B one-hot encoding) |
| 09 | `09_handle_outliers/` | IQR boxplots, anomaly subsets (A, B, C) |
| 10 | `10_standardize_normalize/` | Scaled feature distributions (zscore) |
| 11 | `11_final_dataset_structure/` | Interpretation table EDA + export_summary.txt |

### `data_preparation/data_understanding/`
Data understanding snapshots (auto-generated on each run): `data_understanding_before.txt`, `data_understanding_after.txt`

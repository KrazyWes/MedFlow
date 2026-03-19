"""
Data Preparation

Loads and prepares data from raw_datasets/ for downstream clustering analysis.
Supports both CSV and Excel (.xlsx, .xls) inputs. Skips processing if a source
folder is empty or has no supported files.

Missing value handling:
- Computable: DOH TOTAL AMOUNT = QUANTITY * UNIT COST (and inverse) where possible
- Numeric: impute with median (0 if all missing)
- Categorical: impute with mode, or 'N/A' if no mode; empty strings -> 'N/A'

1. DOH Medicine Procurement and Distribution
   Input: raw_datasets/DOH/  (CSV or Excel)
   Output: this_datasets/doh_medicine_distribution_2022_2025.csv
   Use case: Distribution Pattern Clustering (C)

2. PhilGEPS Medical Procurement
   Input: raw_datasets/PhilGEPS/  (CSV or Excel)
   Output: this_datasets/philgeps_2025_medical_procurement.csv
   Use cases: Supplier/Awardee Clustering (A), Medicine Procurement Pattern (B)

================================================================================
CLUSTERING OBJECTIVES (informs required columns for cleaning)
================================================================================

A. Supplier / Awardee Clustering (PhilGEPS)
   Cluster suppliers based on:
   - Contract Amount
   - Number of awards (aggregate)
   - Regions served (Region of Awardee, Province of Awardee, City/Municipality)
   - Procurement Mode
   Key columns: Awardee Organization Name, Contract Amount, Procurement Mode,
                Region of Awardee, Province of Awardee, City/Municipality of Awardee

B. Medicine Procurement Pattern Clustering (PhilGEPS)
   Cluster procurement records based on:
   - Item Budget
   - Quantity
   - Procurement Mode
   - Funding Source
   Key columns: Item Budget, Quantity, Procurement Mode, Funding Source,
                Item Name, Item Description, UNSPSC Description

C. Distribution Pattern Clustering (DOH)
   Cluster recipients based on:
   - Medicines received (ITEM DESCRIPTION)
   - Quantity
   - Delivery frequency (derive from delivery dates per recipient)
   - Total amount
   Key columns: RECIPIENT, ITEM DESCRIPTION, QUANTITY, TOTAL AMOUNT,
                DATE DELIVERED (or equivalent)
================================================================================
"""

import os
import sys
import pandas as pd

# -----------------------------------------------------------------------------
# Step 0) Import helpers (keep this script as the orchestrator)
# -----------------------------------------------------------------------------
# This file focuses on sequencing the steps (CRISP-DM flow).

# Allow importing local modules inside `CRISP-DM/`
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# -----------------------------------------------------------------------------
# Loader utilities (replacing CRISP-DM/prep/loaders.py)
# -----------------------------------------------------------------------------
CSV_EXTENSIONS = (".csv",)
EXCEL_EXTENSIONS = (".xlsx", ".xls")
ALL_DATA_EXTENSIONS = CSV_EXTENSIONS + EXCEL_EXTENSIONS


def get_supported_files(directory: str):
    """Return list of (filepath, extension) for CSV/Excel files in directory."""
    if not os.path.isdir(directory):
        return []
    files = []
    for f in os.listdir(directory):
        fp = os.path.join(directory, f)
        if os.path.isfile(fp):
            ext = os.path.splitext(f)[1].lower()
            if ext in ALL_DATA_EXTENSIONS:
                files.append((fp, ext))
    return sorted(files, key=lambda x: x[0])


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file. Uses low_memory=False to avoid mixed-type warnings."""
    return pd.read_csv(path, low_memory=False)


def load_excel(path: str, dtype=str) -> list[pd.DataFrame]:
    """Load an Excel file. Returns list of DataFrames (one per sheet)."""
    xl = pd.ExcelFile(path)
    kwargs = {"dtype": dtype} if dtype is not None else {}
    return [pd.read_excel(path, sheet_name=name, **kwargs) for name in xl.sheet_names]


# -----------------------------------------------------------------------------
# Imports (replacing CRISP-DM/prep/* modules)
# -----------------------------------------------------------------------------
from data_cleaning import clean_doh_dataframe, handle_missing_values_doh, handle_missing_values_philgeps  # noqa: E402
from data_understanding import (  # noqa: E402
    build_visualization_paths,
    open_log_sinks,
    generate_eda_visualizations,
    infer_time_label,
    profile_frame,
    rank_attributes_for_clustering,
    write_snapshots_block,
    write_snapshot,
)

output_dir = os.path.join(project_root, "this_datasets")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Step 1) Configure visualization/report output paths (no terminal output)
# -----------------------------------------------------------------------------
# Outputs are written under `webp/visualizations/`:
# - Data Understanding TXT: `data_understanding/reports_txt/`
# - Data Understanding EDA PNGs: `data_understanding/eda_png/`
# - Data Preparation run log: `data_preparation/`
vis_paths = build_visualization_paths(project_root)

# -----------------------------------------------------------------------------
# Step 2) Open log sinks (overwrite per run)
# -----------------------------------------------------------------------------
# We write three text logs per run:
# - data_preparation_run.txt: progress + saved output locations
# - data_understanding_before.txt: consolidated BEFORE snapshots + attribute notes
# - data_understanding_after.txt: consolidated AFTER snapshots + attribute notes
sinks = open_log_sinks(
    prep_log_path=os.path.join(vis_paths["vis_prep_dir"], "data_preparation_run.txt"),
    du_before_log_path=os.path.join(vis_paths["vis_du_txt_dir"], "data_understanding_before.txt"),
    du_after_log_path=os.path.join(vis_paths["vis_du_txt_dir"], "data_understanding_after.txt"),
)

log_prep = sinks.log_prep
log_du_before = sinks.log_du_before
log_du_after = sinks.log_du_after


# =============================================================================
# Data Preparation - DOH Medicine Procurement and Distribution
# For: Distribution Pattern Clustering (C)
# =============================================================================
doh_input_dir = os.path.join(project_root, "raw_datasets", "DOH")
doh_output_filename = "doh_medicine_distribution_2022_2025.csv"
doh_sheets_years = ["2022", "2023", "2024", "2025"]  # Excel sheets to use if present

# -----------------------------------------------------------------------------
# Step 3) DOH pipeline (Objective C: Distribution Pattern Clustering)
# -----------------------------------------------------------------------------
# Goal: create a clustering-ready dataset by:
# - loading all years/sheets first (so BEFORE snapshots are reported once)
# - producing EDA visuals (missingness/distributions)
# - cleaning headers/currency and adding Year
# - imputing missing values and exporting to `this_datasets/`
doh_files = get_supported_files(doh_input_dir)
if doh_files:
    # Step 3.1) Validate presence of key columns (for clustering & data quality checks)
    doh_key_cols = ["RECIPIENT", "ITEM DESCRIPTION", "QUANTITY", "UNIT COST", "TOTAL AMOUNT", "DATE DELIVERED"]

    # Step 3.2) Load ALL raw sources first (all years/months), then proceed to cleaning.
    doh_raw_snapshots = []
    doh_loaded_raw_frames = []  # list of dicts: {"label":..., "df":...}

    # Step 3.3) Load each file/sheet and collect per-year snapshot metadata
    for filepath, ext in doh_files:
        try:
            if ext in CSV_EXTENSIONS:
                df = load_csv(filepath)
                label = f"DOH raw (CSV) - {os.path.basename(filepath)}"
                doh_loaded_raw_frames.append({"label": label, "df": df})
                doh_raw_snapshots.append({"label": label, "profile": profile_frame(df, key_columns=doh_key_cols, top_n=10)})
                log_prep(f"DOH: Loaded RAW {os.path.basename(filepath)} ({len(df)} rows)")
            elif ext in EXCEL_EXTENSIONS:
                sheets = load_excel(filepath)
                xl = pd.ExcelFile(filepath)
                for i, sheet_name in enumerate(xl.sheet_names):
                    if sheet_name in doh_sheets_years or not doh_sheets_years:
                        df = sheets[i]
                        label = f"DOH raw (Excel) - {os.path.basename(filepath)} / sheet '{sheet_name}'"
                        doh_loaded_raw_frames.append({"label": label, "df": df, "sheet_name": sheet_name})
                        doh_raw_snapshots.append({"label": label, "profile": profile_frame(df, key_columns=doh_key_cols, top_n=10)})
                        log_prep(f"DOH: Loaded RAW {os.path.basename(filepath)} sheet '{sheet_name}' ({len(df)} rows)")
        except Exception as e:
            log_prep(f"DOH: Error loading {filepath}: {e}")

    if doh_loaded_raw_frames:
        # Step 3.4) Consolidate all raw years/sheets and write BEFORE report + EDA
        raw_merged = pd.concat([x["df"] for x in doh_loaded_raw_frames], ignore_index=True)
        generate_eda_visualizations(
            raw_merged,
            out_dir=os.path.join(vis_paths["vis_du_eda_dir"], "before", "DOH_raw_merged"),
            dataset_label="DOH",
            stage_label="BEFORE (RAW merged)",
            numeric_focus=["UNIT COST", "TOTAL AMOUNT", "QUANTITY"],
        )
        raw_ranking = rank_attributes_for_clustering(
            raw_merged,
            objective="C. Distribution Pattern Clustering (DOH)",
            must_have=["RECIPIENT", "ITEM DESCRIPTION"],
            exclude_contains=["PTR", "NUMBER", "CONTRACT NO", "SOURCE"],
        )
        write_snapshots_block(
            dataset_name="DOH",
            stage="BEFORE CLEANING / PREPROCESSING (RAW)",
            snapshots=doh_raw_snapshots,
            key_columns=doh_key_cols,
            objective="Cluster recipients/items by quantities, totals, and delivery patterns (DBSCAN for anomalies, K-Means for similarity).",
            merged_df=raw_merged,
            ranking=raw_ranking,
            log_fn=log_du_before,
        )

        # Step 3.5) Apply basic cleaning to each year/sheet, add Year, then write AFTER report + EDA
        doh_clean_snapshots = []
        doh_clean_frames = []
        for item in doh_loaded_raw_frames:
            df = item["df"]
            cleaned = clean_doh_dataframe(df)

            # Add Year if available (sheet name) else infer from filename/label.
            year_val = None
            if "sheet_name" in item:
                sn = str(item["sheet_name"])
                year_val = int(sn) if sn.isdigit() else infer_time_label(sn)
            else:
                year_val = infer_time_label(os.path.basename(item["label"]))

            cleaned = cleaned.copy()
            cleaned["Year"] = year_val

            label = f"DOH after header/currency cleaning - {item['label']}"
            doh_clean_frames.append(cleaned)
            doh_clean_snapshots.append({"label": label, "profile": profile_frame(cleaned, key_columns=doh_key_cols + ["Year"], top_n=10)})

        # Consolidated AFTER cleaning snapshots — written once.
        clean_merged = pd.concat(doh_clean_frames, ignore_index=True)
        generate_eda_visualizations(
            clean_merged,
            out_dir=os.path.join(vis_paths["vis_du_eda_dir"], "after", "DOH_after_basic_cleaning"),
            dataset_label="DOH",
            stage_label="AFTER (basic cleaning, pre-imputation)",
            numeric_focus=["UNIT COST", "TOTAL AMOUNT", "QUANTITY", "Year"],
        )
        clean_ranking = rank_attributes_for_clustering(
            clean_merged,
            objective="C. Distribution Pattern Clustering (DOH)",
            must_have=["RECIPIENT", "ITEM DESCRIPTION", "Year"],
            exclude_contains=["PTR", "NUMBER", "CONTRACT NO", "SOURCE"],
        )
        write_snapshots_block(
            dataset_name="DOH",
            stage="AFTER BASIC CLEANING (headers/currency) — BEFORE IMPUTATION",
            snapshots=doh_clean_snapshots,
            key_columns=doh_key_cols + ["Year"],
            objective="Prepare numeric amounts/costs and a time indicator for later feature engineering (frequency, totals) and clustering.",
            merged_df=clean_merged,
            ranking=clean_ranking,
            log_fn=log_du_after,
        )

        # Deduplication (exact full-row duplicates) BEFORE imputation
        doh_before = len(clean_merged)
        clean_merged = clean_merged.drop_duplicates()
        doh_dropped = doh_before - len(clean_merged)
        if doh_dropped > 0:
            log_prep(f"DOH: Dropped {doh_dropped} exact duplicate rows before imputation.")

        # Existing logic: missing value handling + export (kept for now)
        doh_merged = handle_missing_values_doh(clean_merged)
        generate_eda_visualizations(
            doh_merged,
            out_dir=os.path.join(vis_paths["vis_du_eda_dir"], "after", "DOH_after_imputation"),
            dataset_label="DOH",
            stage_label="AFTER (post-imputation)",
            numeric_focus=["UNIT COST", "TOTAL AMOUNT", "QUANTITY", "Year"],
        )
        # Single consolidated snapshot after imputation
        write_snapshot(
            doh_merged,
            "DOH merged (after missing-value handling)",
            key_columns=doh_key_cols + ["Year"],
            top_n=10,
            log_fn=log_du_after,
        )

        doh_output_path = os.path.join(output_dir, doh_output_filename)
        doh_merged.to_csv(doh_output_path, index=False)
        log_prep(f"DOH: Merged {len(doh_merged)} records. Saved to {doh_output_path}")
    else:
        log_prep("DOH: No valid data loaded from files.")
else:
    log_prep(f"DOH: Folder empty or no CSV/Excel files in {doh_input_dir}. Skipping.")


# =============================================================================
# Data Preparation - PhilGEPS Medical Procurement Filter
# For: Supplier/Awardee Clustering (A), Medicine Procurement Pattern (B)
# =============================================================================
philgeps_input_dir = os.path.join(project_root, "raw_datasets", "PhilGEPS")
philgeps_output_filename = "philgeps_2025_medical_procurement.csv"

medical_keywords = [
    "medical", "medicine", "pharmaceutical", "drug", "vaccine",
    "hospital", "laboratory", "diagnostic", "surgical",
    "clinic", "health", "therapeutic", "antibiotic",
    "syringe", "test kit", "reagent", "biomedical"
]
pattern = "|".join(medical_keywords)

# -----------------------------------------------------------------------------
# Step 4) PhilGEPS pipeline (Objectives A/B: Supplier/Awardee + Procurement Pattern)
# -----------------------------------------------------------------------------
# Goal: build a medical-procurement subset suitable for clustering by:
# - loading all year/month files first (so BEFORE snapshots are reported once)
# - producing analyst-style EDA visuals (missingness, distributions)
# - filtering by UNSPSC medical keywords
# - imputing missing values and exporting the filtered dataset
philgeps_files = get_supported_files(philgeps_input_dir)
if philgeps_files:
    # Step 4.1) Key PhilGEPS columns used to judge usability for clustering
    pg_key_cols = [
        "UNSPSC Description",
        "Awardee Organization Name",
        "Contract Amount",
        "Procurement Mode",
        "Item Budget",
        "Quantity",
        "Funding Source",
    ]

    pg_raw_snapshots = []
    pg_loaded_raw_frames = []  # list of dicts: {"label":..., "df":...}

    # Step 4.2) Load all CSV files / Excel sheets before any filtering or imputation
    for filepath, ext in philgeps_files:
        try:
            if ext in CSV_EXTENSIONS:
                df = load_csv(filepath)
                label = f"PhilGEPS raw (CSV) - {os.path.basename(filepath)}"
                pg_loaded_raw_frames.append({"label": label, "df": df})
                pg_raw_snapshots.append({"label": label, "profile": profile_frame(df, key_columns=pg_key_cols, top_n=10)})
                log_prep(f"PhilGEPS: Loaded RAW {os.path.basename(filepath)} ({len(df)} rows)")
            elif ext in EXCEL_EXTENSIONS:
                sheets = load_excel(filepath, dtype=None)  # Infer types for PhilGEPS
                for i, df in enumerate(sheets):
                    label = f"PhilGEPS raw (Excel) - {os.path.basename(filepath)} / sheet_index {i}"
                    pg_loaded_raw_frames.append({"label": label, "df": df})
                    pg_raw_snapshots.append({"label": label, "profile": profile_frame(df, key_columns=pg_key_cols, top_n=10)})
                log_prep(
                    f"PhilGEPS: Loaded RAW {os.path.basename(filepath)} ({sum(len(s) for s in sheets)} rows from {len(sheets)} sheet(s))"
                )
        except Exception as e:
            log_prep(f"PhilGEPS: Error loading {filepath}: {e}")

    if pg_loaded_raw_frames:
        # Step 4.3) Merge all raw PhilGEPS sources and write BEFORE report + EDA
        pg_raw_merged = pd.concat([x["df"] for x in pg_loaded_raw_frames], ignore_index=True)
        generate_eda_visualizations(
            pg_raw_merged,
            out_dir=os.path.join(vis_paths["vis_du_eda_dir"], "before", "PhilGEPS_raw_merged"),
            dataset_label="PhilGEPS",
            stage_label="BEFORE (RAW merged)",
            numeric_focus=["Contract Amount", "Item Budget", "Quantity", "Approved Budget of the Contract"],
        )
        pg_raw_ranking = rank_attributes_for_clustering(
            pg_raw_merged,
            objective="A/B. Supplier/Awardee + Procurement Pattern (PhilGEPS)",
            must_have=["UNSPSC Description", "Procurement Mode"],
            exclude_contains=["Reference No", "Bid Reference", "Award Reference"],
        )
        write_snapshots_block(
            dataset_name="PhilGEPS",
            stage="BEFORE CLEANING / PREPROCESSING (RAW)",
            snapshots=pg_raw_snapshots,
            key_columns=pg_key_cols,
            objective="Select attributes that represent supplier behavior, contract scale, procurement patterns, and anomalies.",
            merged_df=pg_raw_merged,
            ranking=pg_raw_ranking,
            log_fn=log_du_before,
        )

        # Step 4.4) Filter to medical records using UNSPSC Description keywords
        df = pg_raw_merged
        if "UNSPSC Description" in df.columns:
            medical_df = df[df["UNSPSC Description"].str.contains(pattern, case=False, na=False)]
            # Step 4.5) EDA AFTER filtering (pre-imputation) + DU snapshot
            generate_eda_visualizations(
                medical_df,
                out_dir=os.path.join(vis_paths["vis_du_eda_dir"], "after", "PhilGEPS_filtered_medical_pre_impute"),
                dataset_label="PhilGEPS",
                stage_label="AFTER (filtered medical, pre-imputation)",
                numeric_focus=["Contract Amount", "Item Budget", "Quantity", "Approved Budget of the Contract"],
            )
            # Single consolidated snapshot (filtered) BEFORE imputation.
            write_snapshot(
                medical_df,
                "PhilGEPS filtered to medical records (before missing-value handling)",
                key_columns=pg_key_cols,
                top_n=10,
                log_fn=log_du_after,
            )
            # Deduplication (exact full-row duplicates) BEFORE imputation
            pg_before = len(medical_df)
            medical_df = medical_df.drop_duplicates()
            pg_dropped = pg_before - len(medical_df)
            if pg_dropped > 0:
                log_prep(f"PhilGEPS: Dropped {pg_dropped} exact duplicate rows before imputation.")

            # Step 4.6) Impute missing values (numeric medians; categorical mode/N/A)
            medical_df = handle_missing_values_philgeps(medical_df)
            # Step 4.7) EDA AFTER imputation + DU snapshot
            generate_eda_visualizations(
                medical_df,
                out_dir=os.path.join(vis_paths["vis_du_eda_dir"], "after", "PhilGEPS_filtered_medical_post_impute"),
                dataset_label="PhilGEPS",
                stage_label="AFTER (filtered medical, post-imputation)",
                numeric_focus=["Contract Amount", "Item Budget", "Quantity", "Approved Budget of the Contract"],
            )
            write_snapshot(
                medical_df,
                "PhilGEPS filtered to medical records (after missing-value handling)",
                key_columns=pg_key_cols,
                top_n=10,
                log_fn=log_du_after,
            )
            # Step 4.8) Export prepared PhilGEPS medical subset
            output_file = os.path.join(output_dir, philgeps_output_filename)
            medical_df.to_csv(output_file, index=False)
            log_prep(f"PhilGEPS: Filtered {len(medical_df)} medical records. Saved to {output_file}")
        else:
            log_prep("PhilGEPS: 'UNSPSC Description' column not found. Saving unfiltered.")
            # Step 4.9) Fallback: dedup + impute and export unfiltered dataset if UNSPSC Description is absent
            df = df.drop_duplicates()
            df = handle_missing_values_philgeps(df)
            write_snapshot(
                df,
                "PhilGEPS unfiltered (after missing-value handling)",
                key_columns=None,
                top_n=10,
                log_fn=log_du_after,
            )
            output_file = os.path.join(output_dir, philgeps_output_filename.replace(".csv", "_unfiltered.csv"))
            df.to_csv(output_file, index=False)
    else:
        log_prep("PhilGEPS: No valid data loaded from files.")
else:
    log_prep(f"PhilGEPS: Folder empty or no CSV/Excel files in {philgeps_input_dir}. Skipping.")

# -----------------------------------------------------------------------------
# Step 5) End-of-run summary (where outputs were written)
# -----------------------------------------------------------------------------

log_prep("Data preparation complete.")
log_prep(f"Visualization logs saved: {sinks.prep_log_path}")
log_prep(f"Data understanding BEFORE saved: {sinks.du_before_log_path}")
log_prep(f"Data understanding AFTER saved: {sinks.du_after_log_path}")

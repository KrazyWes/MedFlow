"""
Step 01 — PhilGEPS cleaning.

Loads procurement rows, keeps lines whose UNSPSC / item text matches a medical keyword list,
drops duplicate award lines, fills missing fields, and exports a single wide CSV for 2025.

That CSV is the sole input to 02_data_transformation_philgeps.py. Figures and text logs follow
the same pattern as DOH step 01 but under webp/.../PhilGEPS/01_data_cleaning/.
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

from _common import (
    CSV_EXTENSIONS,
    EXCEL_EXTENSIONS,
    build_source_visualization_paths,
    generate_eda_visualizations,
    get_supported_files,
    load_csv,
    load_excel,
    open_log_sinks,
    profile_frame,
    rank_attributes_for_clustering,
    write_snapshots_block,
    write_snapshot,
)
from log_tee import tee_stdio_to_file
from sources_paths import data_root_philgeps, ensure_all_base_dirs, logs_dir_philgeps, project_root

MEDICAL_KEYWORDS = [
    "medical", "medicine", "pharmaceutical", "drug", "vaccine",
    "hospital", "laboratory", "diagnostic", "surgical",
    "clinic", "health", "therapeutic", "antibiotic",
    "syringe", "test kit", "reagent", "biomedical",
]
PATTERN = "|".join(MEDICAL_KEYWORDS)
# Build a single case-insensitive regex pattern for identifying medical-related UNSPSC entries.


def handle_missing_values_philgeps(df: pd.DataFrame) -> pd.DataFrame:
    """Handle Missing Values: numeric->median, categorical->mode or 'N/A'."""
    df = df.copy()
    # Only impute numeric columns we explicitly trust for quantitative downstream clustering.
    numeric_cols = ["Contract Amount", "Item Budget", "Quantity", "Approved Budget of the Contract", "Line Item No"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    for col in numeric_cols:
        if df[col].isna().any():
            # Median is robust for procurement amounts that can be heavy-tailed.
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    for col in cat_cols:
        # Mode preserves the most common category without inventing new categories.
        mode_val = df[col].mode()
        fill_val = mode_val.iloc[0] if len(mode_val) > 0 and pd.notna(mode_val.iloc[0]) else "N/A"
        df[col] = df[col].fillna(fill_val).replace("", fill_val).replace("nan", "N/A").replace("None", "N/A")
    return df


def main() -> None:
    root = project_root()
    ensure_all_base_dirs()
    output_dir = os.path.join(data_root_philgeps(), "01_data_cleaning")
    input_dir = os.path.join(root, "raw_datasets", "PhilGEPS")
    output_file = os.path.join(output_dir, "philgeps_2025_medical_procurement.csv")
    os.makedirs(output_dir, exist_ok=True)

    vis_paths = build_source_visualization_paths(root, "PhilGEPS")
    term_log = os.path.join(logs_dir_philgeps(), "01_data_cleaning_philgeps_terminal.txt")
    with tee_stdio_to_file(term_log):
        _run_cleaning_philgeps(root, output_file, input_dir, vis_paths)


def _run_cleaning_philgeps(root: str, output_file: str, input_dir: str, vis_paths: dict) -> None:
    sinks = open_log_sinks(
        prep_log_path=os.path.join(vis_paths["vis_prep_dir"], "01_data_cleaning_philgeps_run.txt"),
        du_before_log_path=os.path.join(vis_paths["vis_du_txt_dir"], "data_understanding_philgeps_before.txt"),
        du_after_log_path=os.path.join(vis_paths["vis_du_txt_dir"], "data_understanding_philgeps_after.txt"),
    )
    log_prep, log_du_before, log_du_after = sinks.log_prep, sinks.log_du_before, sinks.log_du_after
    # Columns used to profile and rank attributes for clustering decisions.
    pg_key_cols = ["UNSPSC Description", "Awardee Organization Name", "Contract Amount", "Procurement Mode", "Item Budget", "Quantity", "Funding Source"]

    # --- Step 1: Load Raw Data ---
    files = get_supported_files(input_dir)
    if not files:
        log_prep(f"PhilGEPS: No CSV/Excel files in {input_dir}. Skipping.")
        return

    loaded = []
    raw_snapshots = []
    for filepath, ext in files:
        try:
            if ext in CSV_EXTENSIONS:
                # For CSV inputs, load as a single dataframe.
                df = load_csv(filepath)
                label = f"PhilGEPS raw (CSV) - {os.path.basename(filepath)}"
                loaded.append({"label": label, "df": df})
                raw_snapshots.append({"label": label, "profile": profile_frame(df, key_columns=pg_key_cols, top_n=10)})
                log_prep(f"PhilGEPS: Loaded {os.path.basename(filepath)} ({len(df)} rows)")
            elif ext in EXCEL_EXTENSIONS:
                # For Excel inputs, iterate all worksheets (PhilGEPS exports are often multi-tab).
                sheets = load_excel(filepath, dtype=None)
                for i, df in enumerate(sheets):
                    label = f"PhilGEPS raw (Excel) - {os.path.basename(filepath)} / sheet_index {i}"
                    loaded.append({"label": label, "df": df})
                    raw_snapshots.append({"label": label, "profile": profile_frame(df, key_columns=pg_key_cols, top_n=10)})
                log_prep(f"PhilGEPS: Loaded {os.path.basename(filepath)} ({sum(len(s) for s in sheets)} rows)")
        except Exception as e:
            log_prep(f"PhilGEPS: Error loading {filepath}: {e}")

    if not loaded:
        log_prep("PhilGEPS: No data loaded.")
        return

    # --- Step 2: Merge raw data and generate EDA (before cleaning) ---
    # Merge all loaded raw frames so the "before cleaning" visuals represent the whole dataset.
    pg_raw_merged = pd.concat([x["df"] for x in loaded], ignore_index=True)
    generate_eda_visualizations(
        pg_raw_merged,
        out_dir=os.path.join(vis_paths["vis_eda_steps_dir"], "01_load_raw", "PhilGEPS"),
        dataset_label="PhilGEPS", stage_label="BEFORE (RAW merged)",
        numeric_focus=["Contract Amount", "Item Budget", "Quantity", "Approved Budget of the Contract"], include_correlation=True,
        data_source="PhilGEPS",
    )
    raw_ranking = rank_attributes_for_clustering(
        pg_raw_merged,
        objective="A/B. Supplier + Procurement Pattern (PhilGEPS)",
        must_have=["UNSPSC Description", "Procurement Mode"],
        exclude_contains=["Reference No", "Bid Reference", "Award Reference"],
    )
    write_snapshots_block(
        dataset_name="PhilGEPS", stage="BEFORE CLEANING (RAW)",
        snapshots=raw_snapshots, key_columns=pg_key_cols,
        objective="Supplier/awardee and procurement pattern clustering (K-Means + DBSCAN).",
        merged_df=pg_raw_merged, ranking=raw_ranking, log_fn=log_du_before,
    )

    # --- Step 3: Filter Relevant Data (medical keywords in UNSPSC) ---
    if "UNSPSC Description" in pg_raw_merged.columns:
        # Keep only rows where UNSPSC Description matches at least one medical keyword.
        medical_df = pg_raw_merged[pg_raw_merged["UNSPSC Description"].str.contains(PATTERN, case=False, na=False)]
    else:
        medical_df = pg_raw_merged
        # If the expected column is missing, do not fail the pipeline; just continue unfiltered.
        log_prep("PhilGEPS: 'UNSPSC Description' not found. Using unfiltered data.")

    generate_eda_visualizations(
        medical_df,
        out_dir=os.path.join(vis_paths["vis_eda_steps_dir"], "03_filter_relevant_data", "PhilGEPS"),
        dataset_label="PhilGEPS", stage_label="AFTER (filtered medical)",
        numeric_focus=["Contract Amount", "Item Budget", "Quantity", "Approved Budget of the Contract"], include_correlation=True,
    )
    write_snapshot(medical_df, "PhilGEPS filtered to medical (before imputation)", key_columns=pg_key_cols, top_n=10, log_fn=log_du_after)

    # --- Step 4: Remove Duplicates ---
    # Remove exact duplicates before imputation to prevent skewing medians/modes.
    medical_df = medical_df.drop_duplicates()
    step04_dir = os.path.join(vis_paths["vis_eda_steps_dir"], "04_remove_duplicates")
    os.makedirs(step04_dir, exist_ok=True)
    with open(os.path.join(step04_dir, "step_log_philgeps.txt"), "w", encoding="utf-8") as f:
        f.write("PhilGEPS: Duplicates removed before imputation.\n")

    # --- Step 5: Handle Missing Values (numeric->median, categorical->mode) ---
    # Impute missing fields to avoid losing rows later during feature engineering/encoding.
    medical_df = handle_missing_values_philgeps(medical_df)
    generate_eda_visualizations(
        medical_df,
        out_dir=os.path.join(vis_paths["vis_eda_steps_dir"], "05_handle_missing_values", "PhilGEPS"),
        dataset_label="PhilGEPS", stage_label="AFTER (post-imputation)",
        numeric_focus=["Contract Amount", "Item Budget", "Quantity", "Approved Budget of the Contract"], include_correlation=True,
        data_source="PhilGEPS",
    )
    write_snapshot(medical_df, "PhilGEPS filtered to medical (after missing-value handling)", key_columns=pg_key_cols, top_n=10, log_fn=log_du_after)

    # --- Step 6: Export cleaned data to CSV ---
    # Export the final cleaned, filtered, and imputed dataset.
    medical_df.to_csv(output_file, index=False)
    log_prep(f"PhilGEPS: Saved {len(medical_df)} records -> {output_file}")


if __name__ == "__main__":
    main()

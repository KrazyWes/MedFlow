"""
Step 01 — DOH cleaning.

Reads every CSV/Excel under raw_datasets/DOH/, normalizes messy currency strings on money
columns, deduplicates, imputes missing numerics (including Q * unit_cost = total when the row
allows it), and writes one stacked CSV for 2022–2025.

Downstream: 02_data_transformation_doh.py expects this file path; EDA snapshots go to
webp/EDA_and_visualization/DOH/01_data_cleaning/ via generate_eda_visualizations.

Console + file logging: tee to webp/logs/DOH/01_data_cleaning_doh_terminal.txt; prep/DU text
files live next to the step-01 figures (see build_source_visualization_paths).
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
    infer_time_label,
    load_csv,
    load_excel,
    open_log_sinks,
    profile_frame,
    rank_attributes_for_clustering,
    write_snapshots_block,
    write_snapshot,
)
from log_tee import tee_stdio_to_file
from sources_paths import data_root_doh, ensure_all_base_dirs, logs_dir_doh, project_root


def remove_peso_and_currency_chars(value):
    if pd.isna(value):
        return value
    # Some numeric columns arrive as strings with currency markers (e.g. "₱", "PHP", "Php")
    # and thousands separators; strip them before numeric conversion.
    s = str(value).strip()
    for char in ("\u20b1", "₱", "PHP", "Php", "php"):
        s = s.replace(char, "")
    s = s.replace(",", "").strip()
    return s if s else value


def clean_doh_dataframe(df: pd.DataFrame, currency_cols: list[str] | None = None) -> pd.DataFrame:
    """Fix Data Types: strip peso/currency, convert UNIT COST/TOTAL AMOUNT to numeric, normalize headers."""
    if currency_cols is None:
        currency_cols = ["UNIT COST", "TOTAL AMOUNT"]
    df = df.copy()
    # Normalize headers so downstream code can find columns even if raw spreadsheets use line breaks.
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]
    currency_cols_clean = [c.replace("\n", " ").strip() for c in currency_cols]
    for col in df.columns:
        if df[col].dtype == object:
            # Convert object columns that may contain currency text into cleaner string values.
            df[col] = df[col].apply(
                lambda x: str(x).replace("\u20b1", "").replace("₱", "").strip() if pd.notna(x) else x
            )
    for col in currency_cols_clean:
        if col in df.columns:
            # Apply consistent currency stripping and then coerce failures to NaN.
            df[col] = df[col].apply(remove_peso_and_currency_chars)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def handle_missing_values_doh(df: pd.DataFrame) -> pd.DataFrame:
    """Handle Missing Values: TOTAL=Q*UNIT_COST where possible; numeric->median, categorical->'N/A'."""
    df = df.copy()
    q_col, uc_col, ta_col = "QUANTITY", "UNIT COST", "TOTAL AMOUNT"
    for col in [q_col, uc_col, ta_col]:
        if col in df.columns:
            # Ensure math-based imputation works with numeric dtypes.
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if all(c in df.columns for c in [q_col, uc_col, ta_col]):
        # If TOTAL is missing but QUANTITY and UNIT COST exist, reconstruct TOTAL = Q * UNIT_COST.
        mask = df[ta_col].isna() & df[q_col].notna() & df[uc_col].notna()
        df.loc[mask, ta_col] = (df.loc[mask, q_col] * df.loc[mask, uc_col]).astype(float)
        # If UNIT COST is missing but TOTAL and QUANTITY exist (and Q > 0), reconstruct UNIT_COST = TOTAL / Q.
        mask = df[uc_col].isna() & df[ta_col].notna() & (df[q_col] > 0)
        df.loc[mask, uc_col] = (df.loc[mask, ta_col] / df.loc[mask, q_col]).astype(float)
        # If QUANTITY is missing but TOTAL and UNIT COST exist (and UC > 0), reconstruct QUANTITY = TOTAL / UC.
        mask = df[q_col].isna() & df[ta_col].notna() & (df[uc_col] > 0)
        df.loc[mask, q_col] = (df.loc[mask, ta_col] / df.loc[mask, uc_col]).astype(float)
    numeric_cols = [c for c in [q_col, uc_col, ta_col] if c in df.columns]
    for col in numeric_cols:
        if df[col].isna().any():
            # After deterministic reconstruction, fill any remaining gaps using median (robust to outliers).
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
    cat_cols = [c for c in df.columns if c not in numeric_cols and df[c].dtype == object]
    for col in cat_cols:
        # For categorical fields, use a stable placeholder so clustering/encoding doesn't see empty strings.
        df[col] = df[col].fillna("N/A").replace("", "N/A").replace("nan", "N/A")
    return df


def main() -> None:
    # Wire paths, ensure DOH/PhilGEPS roots exist, then delegate to _run_cleaning inside tee.
    # --- Step 0: Setup paths and log sinks ---
    root = project_root()
    ensure_all_base_dirs()
    output_dir = os.path.join(data_root_doh(), "01_data_cleaning")
    input_dir = os.path.join(root, "raw_datasets", "DOH")
    output_file = os.path.join(output_dir, "doh_medicine_distribution_2022_2025.csv")
    os.makedirs(output_dir, exist_ok=True)

    vis_paths = build_source_visualization_paths(root, "DOH")
    term_log = os.path.join(logs_dir_doh(), "01_data_cleaning_doh_terminal.txt")
    with tee_stdio_to_file(term_log):
        _run_cleaning(root, output_file, input_dir, vis_paths)


def _run_cleaning(root: str, output_file: str, input_dir: str, vis_paths: dict) -> None:
    # open_log_sinks = three plain-text trails (prep + data understanding before/after).
    sinks = open_log_sinks(
        prep_log_path=os.path.join(vis_paths["vis_prep_dir"], "01_data_cleaning_doh_run.txt"),
        du_before_log_path=os.path.join(vis_paths["vis_du_txt_dir"], "data_understanding_doh_before.txt"),
        du_after_log_path=os.path.join(vis_paths["vis_du_txt_dir"], "data_understanding_doh_after.txt"),
    )
    log_prep, log_du_before, log_du_after = sinks.log_prep, sinks.log_du_before, sinks.log_du_after
    # Columns used to profile and snapshot data quality before/after cleaning.
    doh_key_cols = ["RECIPIENT", "ITEM DESCRIPTION", "QUANTITY", "UNIT COST", "TOTAL AMOUNT", "DATE DELIVERED"]
    doh_sheets_years = ["2022", "2023", "2024", "2025"]

    # --- Step 1: Load Raw Data ---
    files = get_supported_files(input_dir)
    if not files:
        log_prep(f"DOH: No CSV/Excel files in {input_dir}. Skipping.")
        return

    loaded = []
    raw_snapshots = []
    for filepath, ext in files:
        try:
            if ext in CSV_EXTENSIONS:
                # For CSV inputs, load as a single dataframe.
                df = load_csv(filepath)
                label = f"DOH raw (CSV) - {os.path.basename(filepath)}"
                loaded.append({"label": label, "df": df})
                raw_snapshots.append({"label": label, "profile": profile_frame(df, key_columns=doh_key_cols, top_n=10)})
                log_prep(f"DOH: Loaded {os.path.basename(filepath)} ({len(df)} rows)")
            elif ext in EXCEL_EXTENSIONS:
                # For Excel inputs, iterate through sheets and keep only year sheets of interest.
                xl = pd.ExcelFile(filepath)
                sheets = load_excel(filepath)
                for i, sheet_name in enumerate(xl.sheet_names):
                    if sheet_name in doh_sheets_years or not doh_sheets_years:
                        df = sheets[i]
                        label = f"DOH raw (Excel) - {os.path.basename(filepath)} / sheet '{sheet_name}'"
                        loaded.append({"label": label, "df": df, "sheet_name": sheet_name})
                        raw_snapshots.append({"label": label, "profile": profile_frame(df, key_columns=doh_key_cols, top_n=10)})
                        log_prep(f"DOH: Loaded {os.path.basename(filepath)} sheet '{sheet_name}' ({len(df)} rows)")
        except Exception as e:
            log_prep(f"DOH: Error loading {filepath}: {e}")

    if not loaded:
        log_prep("DOH: No data loaded.")
        return

    # --- Step 2: Merge raw data and generate EDA (before cleaning) ---
    # Merge all loaded raw frames to generate dataset-wide "before" visuals and clustering rankings.
    raw_merged = pd.concat([x["df"] for x in loaded], ignore_index=True)
    generate_eda_visualizations(
        raw_merged,
        out_dir=os.path.join(vis_paths["vis_eda_steps_dir"], "01_load_raw", "DOH"),
        dataset_label="DOH", stage_label="BEFORE (RAW merged)",
        numeric_focus=["UNIT COST", "TOTAL AMOUNT", "QUANTITY"], include_correlation=True,
        data_source="DOH",
    )
    raw_ranking = rank_attributes_for_clustering(
        raw_merged,
        objective="C. Distribution Pattern Clustering (DOH)",
        must_have=["RECIPIENT", "ITEM DESCRIPTION"],
        exclude_contains=["PTR", "NUMBER", "CONTRACT NO", "SOURCE"],
    )
    # Save "before cleaning" snapshots to support the report/thesis narrative.
    write_snapshots_block(
        dataset_name="DOH", stage="BEFORE CLEANING (RAW)",
        snapshots=raw_snapshots, key_columns=doh_key_cols,
        objective="Cluster recipients by quantities, totals, delivery patterns (K-Means + DBSCAN).",
        merged_df=raw_merged, ranking=raw_ranking, log_fn=log_du_before,
    )

    # --- Step 3: Fix Data Types (strip currency, convert numeric) ---
    clean_snapshots = []
    clean_frames = []
    for item in loaded:
        df = item["df"]
        # Apply schema normalization: header cleanup + currency stripping + numeric coercion.
        cleaned = clean_doh_dataframe(df)
        if "sheet_name" in item:
            sn = str(item["sheet_name"])
            # Derive `Year` primarily from sheet name when available.
            year_val = int(sn) if sn.isdigit() else infer_time_label(sn)
        else:
            # Otherwise infer year from the source filename label.
            year_val = infer_time_label(os.path.basename(item["label"]))
        try:
            # Ensure we always store an integer year (or 0 if inference fails).
            year_val = int(year_val) if year_val is not None else 0
        except (TypeError, ValueError):
            year_val = 0
        cleaned = cleaned.copy()
        cleaned["Year"] = year_val
        # Store per-input snapshot + cleaned frame for final concatenation.
        clean_frames.append(cleaned)
        clean_snapshots.append({"label": f"DOH cleaned - {item['label']}", "profile": profile_frame(cleaned, key_columns=doh_key_cols + ["Year"], top_n=10)})

    # --- Step 4: Merge cleaned frames, generate EDA after type fixes ---
    # Merge cleaned frames to visualize distributions after currency/numeric fixes.
    clean_merged = pd.concat(clean_frames, ignore_index=True)
    generate_eda_visualizations(
        clean_merged,
        out_dir=os.path.join(vis_paths["vis_eda_steps_dir"], "02_fix_data_types", "DOH"),
        dataset_label="DOH", stage_label="AFTER (basic cleaning)",
        numeric_focus=["UNIT COST", "TOTAL AMOUNT", "QUANTITY", "Year"], include_correlation=True,
        data_source="DOH",
    )
    # --- Step 5: Remove Duplicates ---
    # Drop exact duplicates before imputing so duplicated rows don't bias medians/modes.
    clean_merged = clean_merged.drop_duplicates()
    step04_dir = os.path.join(vis_paths["vis_eda_steps_dir"], "04_remove_duplicates")
    os.makedirs(step04_dir, exist_ok=True)
    with open(os.path.join(step04_dir, "step_log_doh.txt"), "w", encoding="utf-8") as f:
        f.write("DOH: Duplicates removed before imputation.\n")

    # --- Step 6: Handle Missing Values (impute TOTAL, median, mode) ---
    # Impute based on the deterministic relationship TOTAL = QUANTITY * UNIT_COST when possible.
    doh_merged = handle_missing_values_doh(clean_merged)
    generate_eda_visualizations(
        doh_merged,
        out_dir=os.path.join(vis_paths["vis_eda_steps_dir"], "05_handle_missing_values", "DOH"),
        dataset_label="DOH", stage_label="AFTER (post-imputation)",
        numeric_focus=["UNIT COST", "TOTAL AMOUNT", "QUANTITY", "Year"], include_correlation=True,
        data_source="DOH",
    )
    write_snapshot(doh_merged, "DOH merged (after missing-value handling)", key_columns=doh_key_cols + ["Year"], top_n=10, log_fn=log_du_after)

    # --- Step 7: Export cleaned data to CSV ---
    # Export the final cleaned dataset for downstream transformation/clustering steps.
    doh_merged.to_csv(output_file, index=False)
    log_prep(f"DOH: Saved {len(doh_merged)} records -> {output_file}")


if __name__ == "__main__":
    main()

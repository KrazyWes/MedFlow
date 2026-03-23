"""
1. Data Cleaning - DOH (CRISP-DM)

Cleans DOH Medicine Procurement and Distribution data.
Input: raw_datasets/DOH/ (CSV or Excel)
Output: this_datasets/01_data_cleaning/doh_medicine_distribution_2022_2025.csv

Steps: Load Raw -> Fix Data Types -> Remove Duplicates -> Handle Missing Values -> Export
"""

from __future__ import annotations

import os
import sys

import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from _common import (
    CSV_EXTENSIONS,
    EXCEL_EXTENSIONS,
    build_visualization_paths,
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


def _project_root() -> str:
    return os.path.dirname(script_dir)


def remove_peso_and_currency_chars(value):
    if pd.isna(value):
        return value
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
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]
    currency_cols_clean = [c.replace("\n", " ").strip() for c in currency_cols]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: str(x).replace("\u20b1", "").replace("₱", "").strip() if pd.notna(x) else x
            )
    for col in currency_cols_clean:
        if col in df.columns:
            df[col] = df[col].apply(remove_peso_and_currency_chars)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def handle_missing_values_doh(df: pd.DataFrame) -> pd.DataFrame:
    """Handle Missing Values: TOTAL=Q*UNIT_COST where possible; numeric->median, categorical->'N/A'."""
    df = df.copy()
    q_col, uc_col, ta_col = "QUANTITY", "UNIT COST", "TOTAL AMOUNT"
    for col in [q_col, uc_col, ta_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if all(c in df.columns for c in [q_col, uc_col, ta_col]):
        mask = df[ta_col].isna() & df[q_col].notna() & df[uc_col].notna()
        df.loc[mask, ta_col] = (df.loc[mask, q_col] * df.loc[mask, uc_col]).astype(float)
        mask = df[uc_col].isna() & df[ta_col].notna() & (df[q_col] > 0)
        df.loc[mask, uc_col] = (df.loc[mask, ta_col] / df.loc[mask, q_col]).astype(float)
        mask = df[q_col].isna() & df[ta_col].notna() & (df[uc_col] > 0)
        df.loc[mask, q_col] = (df.loc[mask, ta_col] / df.loc[mask, uc_col]).astype(float)
    numeric_cols = [c for c in [q_col, uc_col, ta_col] if c in df.columns]
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
    cat_cols = [c for c in df.columns if c not in numeric_cols and df[c].dtype == object]
    for col in cat_cols:
        df[col] = df[col].fillna("N/A").replace("", "N/A").replace("nan", "N/A")
    return df


def main() -> None:
    # --- Step 0: Setup paths and log sinks ---
    root = _project_root()
    output_dir = os.path.join(root, "this_datasets", "01_data_cleaning")
    input_dir = os.path.join(root, "raw_datasets", "DOH")
    output_file = os.path.join(output_dir, "doh_medicine_distribution_2022_2025.csv")
    os.makedirs(output_dir, exist_ok=True)

    vis_paths = build_visualization_paths(root)
    sinks = open_log_sinks(
        prep_log_path=os.path.join(vis_paths["vis_prep_dir"], "data_cleaning_doh_run.txt"),
        du_before_log_path=os.path.join(vis_paths["vis_du_txt_dir"], "data_understanding_doh_before.txt"),
        du_after_log_path=os.path.join(vis_paths["vis_du_txt_dir"], "data_understanding_doh_after.txt"),
    )
    log_prep, log_du_before, log_du_after = sinks.log_prep, sinks.log_du_before, sinks.log_du_after
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
                df = load_csv(filepath)
                label = f"DOH raw (CSV) - {os.path.basename(filepath)}"
                loaded.append({"label": label, "df": df})
                raw_snapshots.append({"label": label, "profile": profile_frame(df, key_columns=doh_key_cols, top_n=10)})
                log_prep(f"DOH: Loaded {os.path.basename(filepath)} ({len(df)} rows)")
            elif ext in EXCEL_EXTENSIONS:
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
        cleaned = clean_doh_dataframe(df)
        if "sheet_name" in item:
            sn = str(item["sheet_name"])
            year_val = int(sn) if sn.isdigit() else infer_time_label(sn)
        else:
            year_val = infer_time_label(os.path.basename(item["label"]))
        try:
            year_val = int(year_val) if year_val is not None else 0
        except (TypeError, ValueError):
            year_val = 0
        cleaned = cleaned.copy()
        cleaned["Year"] = year_val
        clean_frames.append(cleaned)
        clean_snapshots.append({"label": f"DOH cleaned - {item['label']}", "profile": profile_frame(cleaned, key_columns=doh_key_cols + ["Year"], top_n=10)})

    # --- Step 4: Merge cleaned frames, generate EDA after type fixes ---
    clean_merged = pd.concat(clean_frames, ignore_index=True)
    generate_eda_visualizations(
        clean_merged,
        out_dir=os.path.join(vis_paths["vis_eda_steps_dir"], "02_fix_data_types", "DOH"),
        dataset_label="DOH", stage_label="AFTER (basic cleaning)",
        numeric_focus=["UNIT COST", "TOTAL AMOUNT", "QUANTITY", "Year"], include_correlation=True,
        data_source="DOH",
    )
    # --- Step 5: Remove Duplicates ---
    clean_merged = clean_merged.drop_duplicates()
    step04_dir = os.path.join(vis_paths["vis_eda_steps_dir"], "04_remove_duplicates")
    os.makedirs(step04_dir, exist_ok=True)
    with open(os.path.join(step04_dir, "step_log_doh.txt"), "w", encoding="utf-8") as f:
        f.write("DOH: Duplicates removed before imputation.\n")

    # --- Step 6: Handle Missing Values (impute TOTAL, median, mode) ---
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
    doh_merged.to_csv(output_file, index=False)
    log_prep(f"DOH: Saved {len(doh_merged)} records -> {output_file}")


if __name__ == "__main__":
    main()

"""
2. Data Transformation - DOH (CRISP-DM)

Transforms cleaned DOH data into clustering-ready features for C (Distribution Pattern).
Input: this_datasets/01_data_cleaning/doh_medicine_distribution_2022_2025.csv
Output: clustering_C_distribution_recipient_features*.csv, clustering_C_*_interpretation.csv

Steps: Feature Engineering -> Outlier Analysis (retain) -> Standardize/Normalize -> Export
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from _common import (
    _coerce_numeric_and_impute,
    _ensure_dir,
    _fill_categorical,
    _project_root,
    analyze_outliers_iqr_and_boxplots,
    generate_eda_visualizations,
    scale_numeric_features,
)


def _safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _safe_write_csv(df: pd.DataFrame, path: str) -> str:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root, ext = os.path.splitext(path)
        alt = f"{root}_{ts}{ext}"
        df.to_csv(alt, index=False)
        return alt


def preprocess_C_distribution_pattern(doh_df: pd.DataFrame) -> pd.DataFrame:
    """Feature Engineering - C: aggregate per RECIPIENT."""
    required = ["RECIPIENT", "ITEM DESCRIPTION", "QUANTITY", "TOTAL AMOUNT"]
    missing = [c for c in required if c not in doh_df.columns]
    if missing:
        raise ValueError(f"DOH missing columns for C: {missing}")
    df = doh_df.copy()
    df = _coerce_numeric_and_impute(df, ["QUANTITY", "TOTAL AMOUNT"])
    df = _fill_categorical(df, ["RECIPIENT", "ITEM DESCRIPTION"])
    date_cols = [c for c in df.columns if "DATE" in str(c).upper() and ("DELIVER" in str(c).upper() or "PICK" in str(c).upper())]
    delivery_col = date_cols[0] if date_cols else None
    if delivery_col:
        df[delivery_col] = pd.to_datetime(df[delivery_col], errors="coerce")
        df["delivery_date_only"] = df[delivery_col].dt.date
    else:
        df["delivery_date_only"] = pd.NaT
    program_cols = [c for c in doh_df.columns if "PROGRAM" in str(c).upper()]
    program_col = program_cols[0] if program_cols else None
    if program_col and program_col in doh_df.columns:
        df[program_col] = doh_df[program_col]
        df = _fill_categorical(df, [program_col])

    def _mode_or_na(s: pd.Series) -> str:
        m = s.mode(dropna=True)
        return str(m.iloc[0]) if len(m) > 0 else "N/A"

    out = (
        df.groupby("RECIPIENT")
        .agg(
            medicines_received_lines=("ITEM DESCRIPTION", "count"),
            medicines_received_unique=("ITEM DESCRIPTION", "nunique"),
            quantity_total=("QUANTITY", "sum"),
            total_amount_total=("TOTAL AMOUNT", "sum"),
            delivery_frequency_unique_dates=("delivery_date_only", "nunique"),
            top_program=(program_col, _mode_or_na) if program_col else ("ITEM DESCRIPTION", _mode_or_na),
            top_item_description=("ITEM DESCRIPTION", _mode_or_na),
        )
        .reset_index()
    )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def main() -> None:
    # --- Step 0: Setup paths and ensure output dirs exist ---
    root = _project_root()
    clean_dir = os.path.join(root, "this_datasets", "01_data_cleaning")
    trans_dir = os.path.join(root, "this_datasets", "02_data_transformation")
    doh_in = os.path.join(clean_dir, "doh_medicine_distribution_2022_2025.csv")
    C_out = os.path.join(trans_dir, "clustering_C_distribution_recipient_features.csv")
    C_interp_out = os.path.join(trans_dir, "clustering_C_distribution_recipient_interpretation.csv")
    eda_steps = os.path.join(root, "webp", "EDA_and_visualization", "02_data_transformation", "steps")
    for d in ("07_feature_engineering", "09_handle_outliers", "10_standardize_normalize", "11_final_dataset_structure"):
        _ensure_dir(os.path.join(eda_steps, d))

    # --- Step 1: Load cleaned DOH data ---
    if not os.path.exists(doh_in) or os.path.getsize(doh_in) == 0:
        print("DOH: Input missing/empty. Run 01_data_cleaning_doh.py first.")
        return

    doh_df = _safe_read_csv(doh_in)
    if doh_df.empty:
        print("DOH: Input empty. Skipping.")
        return

    # --- Step 2: Feature Engineering (aggregate per RECIPIENT) ---
    C_features = preprocess_C_distribution_pattern(doh_df)
    _safe_write_csv(C_features, C_out)
    print(f"C: Saved distribution features -> {C_out} (rows={len(C_features)})")

    generate_eda_visualizations(
        C_features,
        out_dir=os.path.join(eda_steps, "07_feature_engineering", "C_distribution_recipient"),
        dataset_label="C", stage_label="Feature Engineering (distribution)",
        numeric_focus=["medicines_received_lines", "medicines_received_unique", "quantity_total", "total_amount_total", "delivery_frequency_unique_dates"],
        include_correlation=True, data_source="DOH",
    )

    keep_cols = [c for c in ["RECIPIENT", "top_program", "top_item_description", "quantity_total", "total_amount_total", "delivery_frequency_unique_dates"] if c in C_features.columns]
    C_interp = C_features[keep_cols].copy()
    _safe_write_csv(C_interp, C_interp_out)
    generate_eda_visualizations(
        C_interp,
        out_dir=os.path.join(eda_steps, "11_final_dataset_structure", "C_distribution_recipient_interpretation"),
        dataset_label="C", stage_label="Final (interpretation)",
        numeric_focus=["quantity_total", "total_amount_total", "delivery_frequency_unique_dates"],
        include_correlation=True, data_source="DOH",
    )

    # --- Step 4: Outlier Analysis (IQR, boxplots; retain all data) ---
    analyze_outliers_iqr_and_boxplots(
        C_features,
        feature_cols=["medicines_received_lines", "medicines_received_unique", "quantity_total", "total_amount_total", "delivery_frequency_unique_dates"],
        dataset_label="C",
        out_dir=os.path.join(eda_steps, "09_handle_outliers", "C_distribution_recipient"),
        anomalies_out_csv="clustering_C_distribution_recipient_features_anomalies.csv",
        summary_out_csv="clustering_C_distribution_recipient_outlier_summary.csv",
        data_source="DOH",
    )

    # --- Step 5: Standardize/Normalize (zscore and minmax scaling) ---
    C_scale_cols = [c for c in C_features.columns if c != "RECIPIENT" and pd.api.types.is_numeric_dtype(C_features[c])]
    C_z = scale_numeric_features(C_features, feature_cols=C_scale_cols, method="zscore")
    C_m = scale_numeric_features(C_features, feature_cols=C_scale_cols, method="minmax")
    _safe_write_csv(C_z, C_out.replace(".csv", "_zscore.csv"))
    _safe_write_csv(C_m, C_out.replace(".csv", "_minmax.csv"))
    print(f"C: Saved zscore and minmax scaled features.")


if __name__ == "__main__":
    main()

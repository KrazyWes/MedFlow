"""
2. Data Transformation - PhilGEPS (CRISP-DM)

Transforms cleaned PhilGEPS data into clustering-ready features for A (Supplier) and B (Procurement Pattern).
Input: this_datasets/01_data_cleaning/philgeps_2025_medical_procurement.csv
Output: clustering_A_*, clustering_B_* (features, zscore, minmax, interpretation)

Steps: Feature Engineering -> Encode Categorical (B) -> Outlier Analysis -> Standardize/Normalize -> Export
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


def _top_categories(series: pd.Series, top_k: int) -> list[str]:
    vc = series.value_counts(dropna=True)
    return [str(x) for x in vc.head(top_k).index.tolist()]


def preprocess_A_supplier_awardee(philgeps_df: pd.DataFrame, *, top_k_modes: int = 10, top_k_regions: int = 8) -> pd.DataFrame:
    """Feature Engineering - A: aggregate per Awardee."""
    required = ["Awardee Organization Name", "Procurement Mode", "Region of Awardee", "Contract Amount", "Award Reference No."]
    missing = [c for c in required if c not in philgeps_df.columns]
    if missing:
        raise ValueError(f"PhilGEPS missing columns for A: {missing}")
    df = philgeps_df.copy()
    df = _coerce_numeric_and_impute(df, ["Contract Amount"])
    df = _fill_categorical(df, ["Procurement Mode", "Region of Awardee", "Province of Awardee", "City/Municipality of Awardee"])
    group_key = "Awardee Organization Name"
    agg_base = (
        df.groupby(group_key)
        .agg(
            contract_amount_total=("Contract Amount", "sum"),
            contract_amount_mean=("Contract Amount", "mean"),
            num_awards=("Award Reference No.", "nunique"),
            regions_served_nunique=("Region of Awardee", "nunique"),
            rows_count=("Award Reference No.", "count"),
        )
        .reset_index()
    )
    top_modes = _top_categories(df["Procurement Mode"], top_k_modes)
    mode_counts = (
        df[df["Procurement Mode"].isin(top_modes)]
        .pivot_table(index=group_key, columns="Procurement Mode", values="Award Reference No.", aggfunc="count", fill_value=0)
        .reindex(columns=top_modes, fill_value=0)
        .reset_index()
    )
    mode_counts.columns = [group_key] + [f"mode_count__{c}" for c in top_modes]
    mode_cols = [c for c in mode_counts.columns if c.startswith("mode_count__")]
    for c in mode_cols:
        denom = agg_base.set_index(group_key)["rows_count"].replace(0, np.nan)
        mode_counts[c] = mode_counts[c] / denom[mode_counts[group_key]].values
    top_regions = _top_categories(df["Region of Awardee"], top_k_regions)
    region_counts = (
        df[df["Region of Awardee"].isin(top_regions)]
        .pivot_table(index=group_key, columns="Region of Awardee", values="Award Reference No.", aggfunc="count", fill_value=0)
        .reindex(columns=top_regions, fill_value=0)
        .reset_index()
    )
    region_counts.columns = [group_key] + [f"region_count__{c}" for c in top_regions]
    region_cols = [c for c in region_counts.columns if c.startswith("region_count__")]
    for c in region_cols:
        denom = agg_base.set_index(group_key)["rows_count"].replace(0, np.nan)
        region_counts[c] = region_counts[c] / denom[region_counts[group_key]].values
    out = agg_base.merge(mode_counts, on=group_key, how="left").merge(region_counts, on=group_key, how="left")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def build_A_interpretation_table(philgeps_df: pd.DataFrame) -> pd.DataFrame:
    required = ["Awardee Organization Name", "Procurement Mode", "Region of Awardee", "Province of Awardee", "City/Municipality of Awardee"]
    missing = [c for c in required if c not in philgeps_df.columns]
    if missing:
        return philgeps_df[["Awardee Organization Name"]].drop_duplicates().copy()
    df = philgeps_df.copy()
    df = _fill_categorical(df, required)

    def _mode_or_na(s: pd.Series) -> str:
        m = s.mode(dropna=True)
        return str(m.iloc[0]) if len(m) > 0 else "N/A"

    return (
        df.groupby("Awardee Organization Name")
        .agg(
            top_procurement_mode=("Procurement Mode", _mode_or_na),
            top_region=("Region of Awardee", _mode_or_na),
            top_province=("Province of Awardee", _mode_or_na),
            top_city=("City/Municipality of Awardee", _mode_or_na),
        )
        .reset_index()
    )


def preprocess_B_medicine_procurement_pattern(philgeps_df: pd.DataFrame, *, top_k_modes: int = 10, top_k_funding: int = 10) -> pd.DataFrame:
    """Feature Engineering + Encode Categorical - B: row-level with one-hot."""
    required = ["Item Budget", "Quantity", "Procurement Mode", "Funding Source"]
    missing = [c for c in required if c not in philgeps_df.columns]
    if missing:
        raise ValueError(f"PhilGEPS missing columns for B: {missing}")
    df = philgeps_df.copy()
    df = _coerce_numeric_and_impute(df, ["Item Budget", "Quantity"])
    df = _fill_categorical(df, ["Procurement Mode", "Funding Source"])
    top_modes = _top_categories(df["Procurement Mode"], top_k_modes)
    top_funding = _top_categories(df["Funding Source"], top_k_funding)
    meta_cols = [c for c in ["Awardee Organization Name", "Item Name", "UNSPSC Description", "Region of Awardee", "Procurement Mode", "Funding Source"] if c in df.columns]
    trace_cols = [c for c in ["Award Reference No.", "UNSPSC Code"] if c in df.columns]
    df["Procurement Mode bucket"] = np.where(df["Procurement Mode"].isin(top_modes), df["Procurement Mode"], "Other")
    df["Funding Source bucket"] = np.where(df["Funding Source"].isin(top_funding), df["Funding Source"], "Other")
    ohe_mode = pd.get_dummies(df["Procurement Mode bucket"], prefix="mode", dtype=int)
    ohe_funding = pd.get_dummies(df["Funding Source bucket"], prefix="funding", dtype=int)
    numeric = df[["Item Budget", "Quantity"]].copy()
    numeric = numeric.rename(columns={"Item Budget": "item_budget", "Quantity": "quantity"})
    numeric["item_budget"] = numeric["item_budget"].clip(lower=0)
    numeric["quantity"] = numeric["quantity"].clip(lower=0)
    numeric["log1p_item_budget"] = np.log1p(numeric["item_budget"])
    numeric["log1p_quantity"] = np.log1p(numeric["quantity"])
    features = pd.concat([df[meta_cols + trace_cols].reset_index(drop=True), numeric.reset_index(drop=True), ohe_mode.reset_index(drop=True), ohe_funding.reset_index(drop=True)], axis=1)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0)


def main() -> None:
    # --- Step 0: Setup paths and ensure output dirs exist ---
    root = _project_root()
    clean_dir = os.path.join(root, "this_datasets", "01_data_cleaning")
    trans_dir = os.path.join(root, "this_datasets", "02_data_transformation")
    phil_in = os.path.join(clean_dir, "philgeps_2025_medical_procurement.csv")
    A_out = os.path.join(trans_dir, "clustering_A_supplier_awardee_features.csv")
    B_out = os.path.join(trans_dir, "clustering_B_medicine_procurement_pattern_features.csv")
    A_interp_out = os.path.join(trans_dir, "clustering_A_supplier_awardee_interpretation.csv")
    B_interp_out = os.path.join(trans_dir, "clustering_B_medicine_procurement_pattern_interpretation.csv")
    eda_steps = os.path.join(root, "webp", "EDA_and_visualization", "02_data_transformation", "steps")
    for d in ("07_feature_engineering", "08_encode_categorical", "09_handle_outliers", "10_standardize_normalize", "11_final_dataset_structure"):
        _ensure_dir(os.path.join(eda_steps, d))

    # --- Step 1: Load cleaned PhilGEPS data ---
    if not os.path.exists(phil_in) or os.path.getsize(phil_in) == 0:
        print("PhilGEPS: Input missing/empty. Run 01_data_cleaning_philgeps.py first.")
        return

    phil_df = _safe_read_csv(phil_in)
    if phil_df.empty:
        print("PhilGEPS: Input empty. Skipping.")
        return

    # --- Step 2: A - Feature Engineering (aggregate per Awardee) ---
    A_features = preprocess_A_supplier_awardee(phil_df)
    _safe_write_csv(A_features, A_out)
    print(f"A: Saved supplier features -> {A_out} (rows={len(A_features)})")
    generate_eda_visualizations(
        A_features,
        out_dir=os.path.join(eda_steps, "07_feature_engineering", "A_supplier_awardee"),
        dataset_label="A", stage_label="Feature Engineering (supplier)",
        numeric_focus=["contract_amount_total", "contract_amount_mean", "num_awards", "regions_served_nunique", "rows_count"],
        include_correlation=True, data_source="PhilGEPS",
    )
    A_interp = build_A_interpretation_table(phil_df)
    _safe_write_csv(A_interp, A_interp_out)
    generate_eda_visualizations(
        A_interp,
        out_dir=os.path.join(eda_steps, "11_final_dataset_structure", "A_supplier_awardee_interpretation"),
        dataset_label="A", stage_label="Final (interpretation)",
        numeric_focus=None, include_correlation=False, data_source="PhilGEPS",
    )
    # --- Step 3: A - Outlier Analysis ---
    analyze_outliers_iqr_and_boxplots(
        A_features,
        feature_cols=["contract_amount_total", "contract_amount_mean", "num_awards", "regions_served_nunique", "rows_count"],
        dataset_label="A",
        out_dir=os.path.join(eda_steps, "09_handle_outliers", "A_supplier_awardee"),
        anomalies_out_csv="clustering_A_supplier_awardee_features_anomalies.csv",
        summary_out_csv="clustering_A_supplier_awardee_outlier_summary.csv",
        data_source="PhilGEPS",
    )
    # --- Step 4: A - Scale features (zscore, minmax) and save ---
    A_scale_cols = ["contract_amount_total", "contract_amount_mean", "num_awards", "regions_served_nunique", "rows_count"] + [c for c in A_features.columns if c.startswith("mode_count__") or c.startswith("region_count__")]
    A_z = scale_numeric_features(A_features, feature_cols=A_scale_cols, method="zscore")
    A_m = scale_numeric_features(A_features, feature_cols=A_scale_cols, method="minmax")
    _safe_write_csv(A_z, A_out.replace(".csv", "_zscore.csv"))
    _safe_write_csv(A_m, A_out.replace(".csv", "_minmax.csv"))

    # --- Step 5: B - Feature Engineering + Encode Categorical (one-hot) ---
    B_features = preprocess_B_medicine_procurement_pattern(phil_df)
    _safe_write_csv(B_features, B_out)
    print(f"B: Saved medicine pattern features -> {B_out} (rows={len(B_features)})")
    with open(os.path.join(eda_steps, "08_encode_categorical", "step_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"B: Saved medicine pattern features -> {B_out} (rows={len(B_features)})\n")
    generate_eda_visualizations(
        B_features,
        out_dir=os.path.join(eda_steps, "07_feature_engineering", "B_medicine_procurement_pattern"),
        dataset_label="B", stage_label="Feature Engineering + Encode Categorical",
        numeric_focus=["item_budget", "quantity", "log1p_item_budget", "log1p_quantity"],
        include_correlation=True, data_source="PhilGEPS",
    )
    keep_cols = [c for c in ["Awardee Organization Name", "Item Name", "UNSPSC Description", "Region of Awardee", "Procurement Mode", "Funding Source", "item_budget", "quantity"] if c in B_features.columns]
    B_interp = B_features[keep_cols].copy()
    _safe_write_csv(B_interp, B_interp_out)
    generate_eda_visualizations(
        B_interp,
        out_dir=os.path.join(eda_steps, "11_final_dataset_structure", "B_medicine_procurement_pattern_interpretation"),
        dataset_label="B", stage_label="Final (interpretation)",
        numeric_focus=["item_budget", "quantity"],
        include_correlation=True, data_source="PhilGEPS",
    )
    # --- Step 6: B - Outlier Analysis ---
    analyze_outliers_iqr_and_boxplots(
        B_features,
        feature_cols=["item_budget", "quantity", "log1p_item_budget", "log1p_quantity"],
        dataset_label="B",
        out_dir=os.path.join(eda_steps, "09_handle_outliers", "B_medicine_pattern"),
        anomalies_out_csv="clustering_B_medicine_procurement_pattern_features_anomalies.csv",
        summary_out_csv="clustering_B_medicine_procurement_pattern_outlier_summary.csv",
        data_source="PhilGEPS",
    )
    # --- Step 7: B - Scale features (zscore, minmax) and save ---
    exclude = {"Award Reference No.", "UNSPSC Code"}
    B_scale_cols = [c for c in B_features.columns if c not in exclude and pd.api.types.is_numeric_dtype(B_features[c])]
    B_z = scale_numeric_features(B_features, feature_cols=B_scale_cols, method="zscore")
    B_m = scale_numeric_features(B_features, feature_cols=B_scale_cols, method="minmax")
    _safe_write_csv(B_z, B_out.replace(".csv", "_zscore.csv"))
    _safe_write_csv(B_m, B_out.replace(".csv", "_minmax.csv"))
    print("PhilGEPS: Saved A and B features (zscore, minmax).")


if __name__ == "__main__":
    main()

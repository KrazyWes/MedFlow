"""
Step 02 — PhilGEPS feature engineering.

Starts from the cleaned 2025 procurement CSV, then materializes seven lenses (PhilGEPS_A … G):
supplier-level rollups, dense line-item + one-hot encodings for B, an awardee×region grain for
C–G with thematic feature sets analogous to DOH B–E.

Outputs the same trio of CSVs per lens (raw / z-score / minmax) plus step-wise EDA figures.
Step 04 reads only the minmax files referenced in get_philgeps_dataset_configs().
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Shared `_common`, `sources_paths`, etc. live in parent `CRISP-DM/`.
script_dir = os.path.dirname(os.path.abspath(__file__))
_crisp_dm_root = os.path.dirname(script_dir)
if _crisp_dm_root not in sys.path:
    sys.path.insert(0, _crisp_dm_root)

from _common import (
    _coerce_numeric_and_impute,
    _ensure_dir,
    _fill_categorical,
    _project_root,
    analyze_outliers_iqr_and_boxplots,
    build_source_visualization_paths,
    generate_eda_visualizations,
    scale_numeric_features,
)
from log_tee import tee_stdio_to_file
from sources_paths import data_root_philgeps, ensure_all_base_dirs, logs_dir_philgeps


def _safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _safe_write_csv(df: pd.DataFrame, path: str) -> str:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
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


def preprocess_philgeps_distribution_unit(philgeps_df: pd.DataFrame) -> pd.DataFrame:
    """Unit = Awardee × Region (distribution recipient proxy for PhilGEPS)."""
    req = ["Awardee Organization Name", "Region of Awardee", "Contract Amount", "Quantity", "Item Budget", "Award Reference No."]
    miss = [c for c in req if c not in philgeps_df.columns]
    if miss:
        raise ValueError(f"PhilGEPS missing columns for C–G: {miss}")
    df = philgeps_df.copy()
    df = _coerce_numeric_and_impute(df, ["Contract Amount", "Quantity", "Item Budget"])
    df = _fill_categorical(df, ["Awardee Organization Name", "Region of Awardee"])
    g = (
        df.groupby(["Awardee Organization Name", "Region of Awardee"], dropna=False)
        .agg(
            procurement_lines=("Award Reference No.", "count"),
            unique_awards=("Award Reference No.", "nunique"),
            contract_amount_total=("Contract Amount", "sum"),
            quantity_total=("Quantity", "sum"),
            item_budget_total=("Item Budget", "sum"),
        )
        .reset_index()
    )
    g["DISTRIBUTION_UNIT_KEY"] = g["Awardee Organization Name"].astype(str) + " :: " + g["Region of Awardee"].astype(str)
    g["avg_amount_per_line"] = g["contract_amount_total"] / np.maximum(g["procurement_lines"], 1)
    g["avg_qty_per_line"] = g["quantity_total"] / np.maximum(g["procurement_lines"], 1)
    return g.replace([np.inf, -np.inf], np.nan).fillna(0)


def _unit_ids(df: pd.DataFrame) -> list[str]:
    return [c for c in ["DISTRIBUTION_UNIT_KEY", "Awardee Organization Name", "Region of Awardee"] if c in df.columns]


def frame_philgeps_c_unit(base: pd.DataFrame) -> pd.DataFrame:
    return base.copy()


def frame_philgeps_d_shortage(base: pd.DataFrame) -> pd.DataFrame:
    d = base.copy()
    d["feature_supply_shortage_proxy"] = -(
        np.log1p(d["quantity_total"].clip(lower=0)) + 0.5 * np.log1p(d["procurement_lines"].clip(lower=0))
    )
    cols = _unit_ids(d) + ["quantity_total", "procurement_lines", "item_budget_total", "feature_supply_shortage_proxy"]
    return d[[c for c in cols if c in d.columns]]


def frame_philgeps_e_overstock(base: pd.DataFrame) -> pd.DataFrame:
    d = base.copy()
    d["feature_procurement_intensity"] = (
        np.log1p(d["contract_amount_total"].clip(lower=0))
        + np.log1p(d["item_budget_total"].clip(lower=0))
        + np.log1p(d["quantity_total"].clip(lower=0))
    )
    cols = _unit_ids(d) + [
        "contract_amount_total",
        "item_budget_total",
        "quantity_total",
        "avg_amount_per_line",
        "feature_procurement_intensity",
    ]
    return d[[c for c in cols if c in d.columns]]


def frame_philgeps_f_inefficient(base: pd.DataFrame) -> pd.DataFrame:
    d = base.copy()
    d["feature_lines_per_unit_qty"] = d["procurement_lines"] / np.maximum(d["quantity_total"], 1e-6)
    cols = _unit_ids(d) + ["procurement_lines", "quantity_total", "unique_awards", "feature_lines_per_unit_qty"]
    return d[[c for c in cols if c in d.columns]]


def frame_philgeps_g_unequal(base: pd.DataFrame) -> pd.DataFrame:
    d = base.copy()
    ua = np.maximum(d["unique_awards"], 1)
    d["feature_amount_per_award"] = d["contract_amount_total"] / ua
    d["feature_lines_per_award"] = d["procurement_lines"] / ua
    d["feature_budget_per_award"] = d["item_budget_total"] / ua
    cols = _unit_ids(d) + [
        "contract_amount_total",
        "item_budget_total",
        "procurement_lines",
        "unique_awards",
        "feature_amount_per_award",
        "feature_lines_per_award",
        "feature_budget_per_award",
    ]
    return d[[c for c in cols if c in d.columns]]


def _scale_export(dfin: pd.DataFrame, base_path: str) -> None:
    id_keep = [c for c in dfin.columns if not pd.api.types.is_numeric_dtype(dfin[c])]
    num_cols = [c for c in dfin.columns if c not in id_keep and pd.api.types.is_numeric_dtype(dfin[c])]
    z = scale_numeric_features(dfin, feature_cols=num_cols, method="zscore")
    m = scale_numeric_features(dfin, feature_cols=num_cols, method="minmax")
    _safe_write_csv(z, base_path.replace(".csv", "_zscore.csv"))
    _safe_write_csv(m, base_path.replace(".csv", "_minmax.csv"))


def main() -> None:
    # _run_main builds all seven lenses; heavy lifting is in helper functions per lens.
    root = _project_root()
    ensure_all_base_dirs()
    term_log = os.path.join(logs_dir_philgeps(), "02_data_transformation_philgeps_terminal.txt")
    with tee_stdio_to_file(term_log):
        _run_main(root)


def _run_bundle(
    frame: pd.DataFrame,
    slug: str,
    human: str,
    trans_dir: str,
    eda_steps: str,
) -> None:
    out_csv = os.path.join(trans_dir, f"clustering_{slug}_features.csv")
    _safe_write_csv(frame, out_csv)
    print(f"PhilGEPS: Saved {slug} -> {out_csv} (rows={len(frame)})")
    num_for_eda = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    generate_eda_visualizations(
        frame,
        out_dir=os.path.join(eda_steps, "07_feature_engineering", slug),
        dataset_label=human,
        stage_label="Feature engineering",
        numeric_focus=num_for_eda[:12] if num_for_eda else None,
        include_correlation=bool(num_for_eda),
        data_source="PhilGEPS",
    )
    if num_for_eda:
        analyze_outliers_iqr_and_boxplots(
            frame,
            feature_cols=num_for_eda,
            dataset_label=slug,
            out_dir=os.path.join(eda_steps, "09_handle_outliers", slug),
            anomalies_out_csv=f"{slug}_anomalies.csv",
            summary_out_csv=f"{slug}_outlier_summary.csv",
            data_source="PhilGEPS",
        )
    _scale_export(frame, out_csv)


def _run_main(root: str) -> None:
    clean_dir = os.path.join(data_root_philgeps(), "01_data_cleaning")
    trans_dir = os.path.join(data_root_philgeps(), "02_data_transformation")
    phil_in = os.path.join(clean_dir, "philgeps_2025_medical_procurement.csv")
    os.makedirs(trans_dir, exist_ok=True)

    vis_paths = build_source_visualization_paths(root, "PhilGEPS")
    eda_steps = os.path.join(vis_paths["vis_eda_preprocessing_dir"], "steps")
    for d in ("07_feature_engineering", "08_encode_categorical", "09_handle_outliers", "11_final_dataset_structure"):
        _ensure_dir(os.path.join(eda_steps, d))

    if not os.path.exists(phil_in) or os.path.getsize(phil_in) == 0:
        print("PhilGEPS: Input missing/empty. Run PhilGEPS/01_data_cleaning_philgeps.py first.")
        return

    phil_df = _safe_read_csv(phil_in)
    if phil_df.empty:
        print("PhilGEPS: Input empty. Skipping.")
        return

    # --- A ---
    A_features = preprocess_A_supplier_awardee(phil_df)
    _run_bundle(A_features, "PhilGEPS_A_supplier_awardee", "PhilGEPS A - supplier awardee (unit: organization)", trans_dir, eda_steps)
    A_interp = build_A_interpretation_table(phil_df)
    _safe_write_csv(A_interp, os.path.join(trans_dir, "clustering_PhilGEPS_A_supplier_awardee_interpretation.csv"))
    generate_eda_visualizations(
        A_interp,
        out_dir=os.path.join(eda_steps, "11_final_dataset_structure", "PhilGEPS_A_interpretation"),
        dataset_label="A",
        stage_label="Interpretation",
        numeric_focus=None,
        include_correlation=False,
        data_source="PhilGEPS",
    )

    # --- B ---
    B_features = preprocess_B_medicine_procurement_pattern(phil_df)
    _run_bundle(B_features, "PhilGEPS_B_medicine_procurement_pattern", "PhilGEPS B - medicine procurement (unit: line item)", trans_dir, eda_steps)
    keep_b = [c for c in ["Awardee Organization Name", "Item Name", "UNSPSC Description", "Region of Awardee", "Procurement Mode", "Funding Source", "item_budget", "quantity"] if c in B_features.columns]
    _safe_write_csv(B_features[keep_b].copy(), os.path.join(trans_dir, "clustering_PhilGEPS_B_medicine_procurement_pattern_interpretation.csv"))

    with open(os.path.join(eda_steps, "08_encode_categorical", "step_log.txt"), "w", encoding="utf-8") as f:
        f.write("PhilGEPS B: encoded categoricals + numeric.\n")

    # --- C–G (distribution unit) ---
    unit_base = preprocess_philgeps_distribution_unit(phil_df)
    unit_bundles = [
        ("PhilGEPS_C_distribution_recipient", frame_philgeps_c_unit(unit_base), "PhilGEPS C - distribution unit (Awardee x Region)"),
        ("PhilGEPS_D_high_risk_shortage", frame_philgeps_d_shortage(unit_base), "PhilGEPS D - shortage-risk proxy (same unit, dedicated features)"),
        ("PhilGEPS_E_overstocking", frame_philgeps_e_overstock(unit_base), "PhilGEPS E - overstocking proxy (dedicated features)"),
        ("PhilGEPS_F_inefficient_distribution", frame_philgeps_f_inefficient(unit_base), "PhilGEPS F - inefficient distribution proxy (dedicated features)"),
        ("PhilGEPS_G_unequal_supply_regions", frame_philgeps_g_unequal(unit_base), "PhilGEPS G - unequal / fragmented spend (dedicated features)"),
    ]
    for slug, fr, hum in unit_bundles:
        _run_bundle(fr, slug, hum, trans_dir, eda_steps)

    print("PhilGEPS: Transformation complete (A-G).")


if __name__ == "__main__":
    main()

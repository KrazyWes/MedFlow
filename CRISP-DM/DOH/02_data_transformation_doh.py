"""
Step 02 — DOH feature engineering.

One cleaned delivery table -> five analysis lenses (DOH_A … DOH_E). Each lens aggregates or
filters to a unit (recipient, etc.), builds interpretable numeric features, writes raw + z-score
+ MinMax CSVs, and emits EDA under webp/.../DOH/02_data_transformation/steps/.

Clustering (step 04) only needs the *_features_minmax.csv files; z-score copies exist for
optional standard-scaled comparisons in EDA.
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
    _project_root,
    analyze_outliers_iqr_and_boxplots,
    build_source_visualization_paths,
    generate_eda_visualizations,
    scale_numeric_features,
)
from log_tee import tee_stdio_to_file
from sources_paths import data_root_doh, ensure_all_base_dirs, logs_dir_doh


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


def _fill_categorical(df: pd.DataFrame, cols: list[str], placeholder: str = "N/A") -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = df[c].replace({None: np.nan, "nan": np.nan, "None": np.nan})
        df[c] = df[c].fillna(placeholder).astype(str)
    return df


def preprocess_doh_base_recipient_features(doh_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate line-level DOH data to one row per RECIPIENT (distribution unit)."""
    required = ["RECIPIENT", "ITEM DESCRIPTION", "QUANTITY", "TOTAL AMOUNT"]
    missing = [c for c in required if c not in doh_df.columns]
    if missing:
        raise ValueError(f"DOH missing columns: {missing}")
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
    out = out.replace([np.inf, -np.inf], np.nan)
    denom = out["delivery_frequency_unique_dates"].replace(0, np.nan)
    out["avg_quantity_per_delivery"] = out["quantity_total"] / denom
    out["avg_amount_per_delivery"] = out["total_amount_total"] / denom
    return out.fillna(0)


def _rid_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ["RECIPIENT", "top_program", "top_item_description"] if c in df.columns]


def frame_doh_a(base: pd.DataFrame) -> pd.DataFrame:
    return base.copy()


def frame_doh_b_shortage(base: pd.DataFrame) -> pd.DataFrame:
    d = base.copy()
    d["feature_shortage_depth"] = -(
        np.log1p(d["quantity_total"].clip(lower=0))
        + 0.5 * np.log1p(d["delivery_frequency_unique_dates"].clip(lower=0))
        + 0.5 * np.log1p(d["avg_quantity_per_delivery"].clip(lower=0))
    )
    cols = _rid_cols(d) + [
        "medicines_received_unique",
        "delivery_frequency_unique_dates",
        "quantity_total",
        "avg_quantity_per_delivery",
        "feature_shortage_depth",
    ]
    return d[[c for c in cols if c in d.columns]]


def frame_doh_c_overstock(base: pd.DataFrame) -> pd.DataFrame:
    d = base.copy()
    d["feature_overstock_intensity"] = (
        np.log1p(d["total_amount_total"].clip(lower=0))
        + np.log1p(d["quantity_total"].clip(lower=0))
        + np.log1p(d["avg_amount_per_delivery"].clip(lower=0))
    )
    cols = _rid_cols(d) + [
        "total_amount_total",
        "quantity_total",
        "avg_amount_per_delivery",
        "medicines_received_lines",
        "feature_overstock_intensity",
    ]
    return d[[c for c in cols if c in d.columns]]


def frame_doh_d_inefficient(base: pd.DataFrame) -> pd.DataFrame:
    d = base.copy()
    d["feature_deliveries_per_unit_qty"] = d["delivery_frequency_unique_dates"] / np.maximum(
        d["quantity_total"], 1e-6
    )
    cols = _rid_cols(d) + [
        "delivery_frequency_unique_dates",
        "quantity_total",
        "medicines_received_lines",
        "avg_quantity_per_delivery",
        "feature_deliveries_per_unit_qty",
    ]
    return d[[c for c in cols if c in d.columns]]


def frame_doh_e_unequal(base: pd.DataFrame) -> pd.DataFrame:
    d = base.copy()
    mu = np.maximum(d["medicines_received_unique"], 1)
    d["feature_amount_per_unique_med"] = d["total_amount_total"] / mu
    d["feature_lines_per_unique_med"] = d["medicines_received_lines"] / mu
    d["feature_qty_per_unique_med"] = d["quantity_total"] / mu
    cols = _rid_cols(d) + [
        "total_amount_total",
        "quantity_total",
        "medicines_received_lines",
        "medicines_received_unique",
        "feature_amount_per_unique_med",
        "feature_lines_per_unique_med",
        "feature_qty_per_unique_med",
    ]
    return d[[c for c in cols if c in d.columns]]


def _scale_export(dfin: pd.DataFrame, base_path: str) -> None:
    id_keep = [c for c in dfin.columns if not pd.api.types.is_numeric_dtype(dfin[c])]
    # IDs are object; numeric features all others numeric
    num_cols = [c for c in dfin.columns if c not in id_keep and pd.api.types.is_numeric_dtype(dfin[c])]
    z = scale_numeric_features(dfin, feature_cols=num_cols, method="zscore")
    m = scale_numeric_features(dfin, feature_cols=num_cols, method="minmax")
    _safe_write_csv(z, base_path.replace(".csv", "_zscore.csv"))
    _safe_write_csv(m, base_path.replace(".csv", "_minmax.csv"))


def main() -> None:
    # _run_main owns the per-lens loops, EDA calls, and CSV writes.
    root = _project_root()
    ensure_all_base_dirs()
    term_log = os.path.join(logs_dir_doh(), "02_data_transformation_doh_terminal.txt")
    with tee_stdio_to_file(term_log):
        _run_main(root)


def _run_main(root: str) -> None:
    clean_dir = os.path.join(data_root_doh(), "01_data_cleaning")
    trans_dir = os.path.join(data_root_doh(), "02_data_transformation")
    doh_in = os.path.join(clean_dir, "doh_medicine_distribution_2022_2025.csv")
    os.makedirs(trans_dir, exist_ok=True)

    vis_paths = build_source_visualization_paths(root, "DOH")
    eda_steps = os.path.join(vis_paths["vis_eda_preprocessing_dir"], "steps")
    for d in ("07_feature_engineering", "09_handle_outliers", "11_final_dataset_structure"):
        _ensure_dir(os.path.join(eda_steps, d))

    if not os.path.exists(doh_in) or os.path.getsize(doh_in) == 0:
        print("DOH: Input missing/empty. Run DOH/01_data_cleaning_doh.py first.")
        return

    doh_df = _safe_read_csv(doh_in)
    if doh_df.empty:
        print("DOH: Input empty. Skipping.")
        return

    base = preprocess_doh_base_recipient_features(doh_df)
    bundles = [
        ("DOH_A_distribution_recipient", frame_doh_a(base), "DOH A — distribution recipient (unit: RECIPIENT)"),
        ("DOH_B_high_risk_shortage", frame_doh_b_shortage(base), "DOH B — high risk of shortage (dedicated features)"),
        ("DOH_C_overstocking", frame_doh_c_overstock(base), "DOH C — overstocking (dedicated features)"),
        ("DOH_D_inefficient_distribution", frame_doh_d_inefficient(base), "DOH D — inefficient distribution (dedicated features)"),
        ("DOH_E_unequal_supply_regions", frame_doh_e_unequal(base), "DOH E — unequal supply / concentration (dedicated features)"),
    ]

    for slug, frame, label in bundles:
        out_csv = os.path.join(trans_dir, f"clustering_{slug}_features.csv")
        _safe_write_csv(frame, out_csv)
        print(f"DOH: Saved {slug} -> {out_csv} (rows={len(frame)})")

        num_for_eda = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        generate_eda_visualizations(
            frame,
            out_dir=os.path.join(eda_steps, "07_feature_engineering", slug),
            dataset_label=label,
            stage_label="Feature engineering",
            numeric_focus=num_for_eda[:10],
            include_correlation=True,
            data_source="DOH",
        )
        analyze_outliers_iqr_and_boxplots(
            frame,
            feature_cols=num_for_eda,
            dataset_label=slug,
            out_dir=os.path.join(eda_steps, "09_handle_outliers", slug),
            anomalies_out_csv=f"{slug}_anomalies.csv",
            summary_out_csv=f"{slug}_outlier_summary.csv",
            data_source="DOH",
        )
        _scale_export(frame, out_csv)

    print("DOH: Transformation complete (A-E feature sets + minmax).")


if __name__ == "__main__":
    main()

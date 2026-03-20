"""
Clustering Data Preprocessing (A/B/C)

Builds clustering-ready feature tables for:

A. Supplier / Awardee Clustering (PhilGEPS)
B. Medicine Procurement Pattern Clustering (PhilGEPS)
C. Distribution Pattern Clustering (DOH)

Pipeline steps (labeled for identification):
  07 Feature Engineering (aggregate, derive features)
  08 Encode Categorical Data (one-hot for B)
  09 Handle Outliers (IQR analysis, retain for anomaly detection)
  10 Standardize / Normalize Data (zscore, minmax)
  11 Final Clean Dataset Structure (interpretation tables, scaled outputs)

Outputs: this_datasets/ ; EDA: webp/EDA/data_preprocessing/steps/
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Iterable

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore[import-not-found]

# Allow importing data_understanding for EDA
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

try:
    from data_understanding import generate_eda_visualizations
except ImportError:
    generate_eda_visualizations = None


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    return df


def _safe_write_csv(df: pd.DataFrame, path: str) -> str:
    """
    Write CSV, but if Windows denies access (e.g., file open in Excel),
    fall back to a timestamped filename.
    """
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root, ext = os.path.splitext(path)
        alt_path = f"{root}_{ts}{ext}"
        df.to_csv(alt_path, index=False)
        return alt_path


def _coerce_numeric_and_impute(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Convert `cols` to numeric (errors -> NaN), then impute:
    - median if possible
    - else 0 (if entire column missing)
    """
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med if pd.notna(med) else 0)
    return df


def _fill_categorical(df: pd.DataFrame, cols: Iterable[str], placeholder: str = "N/A") -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = df[c].replace({None: np.nan, "nan": np.nan, "None": np.nan})
        df[c] = df[c].fillna(placeholder).astype(str)
    return df


def _top_categories(series: pd.Series, top_k: int) -> list[str]:
    vc = series.value_counts(dropna=True)
    chosen = vc.head(top_k).index.tolist()
    return [str(x) for x in chosen]


def preprocess_A_supplier_awardee(philgeps_df: pd.DataFrame, *, top_k_modes: int = 10, top_k_regions: int = 8) -> pd.DataFrame:
    """
    STEP 07 Feature Engineering - A: Supplier/Awardee
    Aggregate per Awardee: contract_amount_total, num_awards, regions_served_nunique,
    procurement/region mode proportions. Grain: one row per Awardee.
    """
    required = ["Awardee Organization Name", "Procurement Mode", "Region of Awardee", "Contract Amount", "Award Reference No."]
    missing = [c for c in required if c not in philgeps_df.columns]
    if missing:
        raise ValueError(f"PhilGEPS dataset missing required columns for A: {missing}")

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

    # Procurement mode mix: proportions for top modes
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
        denom = agg_base.set_index(group_key)["rows_count"]
        mode_counts[c] = mode_counts[c] / denom[mode_counts[group_key]].values

    # Region mix (optional): proportions for top regions served
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
        denom = agg_base.set_index(group_key)["rows_count"]
        region_counts[c] = region_counts[c] / denom[region_counts[group_key]].values

    out = agg_base.merge(mode_counts, on=group_key, how="left").merge(region_counts, on=group_key, how="left")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def build_A_interpretation_table(philgeps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpretation-only table for A (do not use for clustering features directly).
    Produces per-awardee categorical summaries (most common mode/region/province/city).
    """
    required = ["Awardee Organization Name", "Procurement Mode", "Region of Awardee", "Province of Awardee", "City/Municipality of Awardee"]
    missing = [c for c in required if c not in philgeps_df.columns]
    if missing:
        # Return minimal interpretation table if some columns are missing
        return philgeps_df[["Awardee Organization Name"]].drop_duplicates().copy()

    df = philgeps_df.copy()
    df = _fill_categorical(df, required)

    def _mode_or_na(s: pd.Series) -> str:
        m = s.mode(dropna=True)
        return str(m.iloc[0]) if len(m) > 0 else "N/A"

    out = (
        df.groupby("Awardee Organization Name")
        .agg(
            top_procurement_mode=("Procurement Mode", _mode_or_na),
            top_region=("Region of Awardee", _mode_or_na),
            top_province=("Province of Awardee", _mode_or_na),
            top_city=("City/Municipality of Awardee", _mode_or_na),
        )
        .reset_index()
    )
    return out


def preprocess_B_medicine_procurement_pattern(
    philgeps_df: pd.DataFrame,
    *,
    top_k_modes: int = 10,
    top_k_funding: int = 10,
) -> pd.DataFrame:
    """
    STEP 07 Feature Engineering + STEP 08 Encode Categorical Data - B: Medicine Procurement
    Row-level: item_budget, quantity, log1p transforms; one-hot for Procurement Mode,
    Funding Source (top-k bucketed). Grain: one row per line item.
    """
    required = ["Item Budget", "Quantity", "Procurement Mode", "Funding Source"]
    missing = [c for c in required if c not in philgeps_df.columns]
    if missing:
        raise ValueError(f"PhilGEPS dataset missing required columns for B: {missing}")

    df = philgeps_df.copy()
    df = _coerce_numeric_and_impute(df, ["Item Budget", "Quantity"])
    df = _fill_categorical(df, ["Procurement Mode", "Funding Source"])

    top_modes = _top_categories(df["Procurement Mode"], top_k_modes)
    top_funding = _top_categories(df["Funding Source"], top_k_funding)

    # Interpretation-only categorical/context columns (kept, but not used in numeric clustering matrix)
    meta_cols = [
        c
        for c in [
            "Awardee Organization Name",
            "Item Name",
            "UNSPSC Description",
            "Region of Awardee",
            "Procurement Mode",
            "Funding Source",
        ]
        if c in df.columns
    ]

    # Traceability columns (may be numeric due to CSV inference; exclude from scaling later)
    trace_cols = [c for c in ["Award Reference No.", "UNSPSC Code"] if c in df.columns]

    df["Procurement Mode bucket"] = np.where(df["Procurement Mode"].isin(top_modes), df["Procurement Mode"], "Other")
    df["Funding Source bucket"] = np.where(df["Funding Source"].isin(top_funding), df["Funding Source"], "Other")

    ohe_mode = pd.get_dummies(df["Procurement Mode bucket"], prefix="mode", dtype=int)
    ohe_funding = pd.get_dummies(df["Funding Source bucket"], prefix="funding", dtype=int)

    numeric = df[["Item Budget", "Quantity"]].copy()
    numeric = numeric.rename(columns={"Item Budget": "item_budget", "Quantity": "quantity"})

    # log1p is undefined for x < -1; clip defensive
    numeric["item_budget"] = numeric["item_budget"].clip(lower=0)
    numeric["quantity"] = numeric["quantity"].clip(lower=0)
    numeric["log1p_item_budget"] = np.log1p(numeric["item_budget"])
    numeric["log1p_quantity"] = np.log1p(numeric["quantity"])

    features = pd.concat(
        [
            df[meta_cols + trace_cols].reset_index(drop=True),
            numeric.reset_index(drop=True),
            ohe_mode.reset_index(drop=True),
            ohe_funding.reset_index(drop=True),
        ],
        axis=1,
    )
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features


def preprocess_C_distribution_pattern(doh_df: pd.DataFrame) -> pd.DataFrame:
    """
    STEP 07 Feature Engineering - C: Distribution Pattern
    Aggregate per RECIPIENT: medicines_received_lines, quantity_total, total_amount_total,
    delivery_frequency_unique_dates. Grain: one row per recipient.
    """
    required = ["RECIPIENT", "ITEM DESCRIPTION", "QUANTITY", "TOTAL AMOUNT"]
    missing = [c for c in required if c not in doh_df.columns]
    if missing:
        raise ValueError(f"DOH dataset missing required columns for C: {missing}")

    df = doh_df.copy()
    df = _coerce_numeric_and_impute(df, ["QUANTITY", "TOTAL AMOUNT"])
    df = _fill_categorical(df, ["RECIPIENT", "ITEM DESCRIPTION"])

    # Detect delivery date column robustly
    date_col_candidates = [
        c
        for c in df.columns
        if "DATE" in str(c).upper() and ("DELIVER" in str(c).upper() or "PICK" in str(c).upper())
    ]
    delivery_col = date_col_candidates[0] if date_col_candidates else None

    if delivery_col:
        df[delivery_col] = pd.to_datetime(df[delivery_col], errors="coerce")
        df["delivery_date_only"] = df[delivery_col].dt.date
    else:
        df["delivery_date_only"] = pd.NaT

    # Optional categorical columns for interpretation (not clustering)
    program_col_candidates = [c for c in doh_df.columns if "PROGRAM" in str(c).upper()]
    program_col = program_col_candidates[0] if program_col_candidates else None

    df = df.copy()
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
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def _iqr_fences(series: pd.Series) -> tuple[float, float]:
    """Compute IQR fences for a 1D numeric series."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    # If IQR is 0, fences collapse to the same value; still valid for outlier checks.
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def analyze_outliers_iqr_and_boxplots(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    dataset_label: str,
    out_dir: str,
    anomalies_out_csv: str,
    summary_out_csv: str,
    outlier_rule: str = "any",
    boxplot_cols: list[str] | None = None,
) -> None:
    """
    STEP 09 Handle Outliers: IQR fences + boxplots.
    Outliers are RETAINED for anomaly detection; we export anomaly subset + summary only.
    """
    _ensure_dir(out_dir)

    # Work on a copy to avoid mutating caller's dataframe
    df = df.copy()
    for c in feature_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        print(f"{dataset_label}: No numeric feature cols for outlier analysis. Skipping.")
        return

    # Boxplots (optional): default to same columns
    box_cols = boxplot_cols or feature_cols
    for c in box_cols:
        if c not in df.columns:
            continue
        series = pd.to_numeric(df[c], errors="coerce").dropna()
        if series.empty:
            continue
        plt.figure(figsize=(7, 4), dpi=160)
        plt.boxplot(series, vert=True)
        plt.title(f"{dataset_label} - Boxplot: {c}")
        plt.ylabel(c)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"boxplot__{c}.png"))
        plt.close()

    # IQR outlier detection per feature
    outlier_masks: dict[str, pd.Series] = {}
    for c in feature_cols:
        lower, upper = _iqr_fences(pd.to_numeric(df[c], errors="coerce").dropna())
        mask = (pd.to_numeric(df[c], errors="coerce") < lower) | (pd.to_numeric(df[c], errors="coerce") > upper)
        outlier_masks[c] = mask.fillna(False)

    if outlier_rule == "any":
        is_anomaly = pd.concat(outlier_masks.values(), axis=1).any(axis=1)
    elif outlier_rule == "all":
        is_anomaly = pd.concat(outlier_masks.values(), axis=1).all(axis=1)
    else:
        raise ValueError("outlier_rule must be 'any' or 'all'")

    # Export anomaly subset
    anomalies_path = os.path.join(out_dir, anomalies_out_csv)
    df[is_anomaly].to_csv(anomalies_path, index=False)

    # Export summary
    summary_rows = []
    for c, mask in outlier_masks.items():
        summary_rows.append(
            {
                "feature_col": c,
                "outlier_count": int(mask.sum()),
                "outlier_ratio": float(mask.mean()),
            }
        )
    summary_rows.append(
        {
            "feature_col": "__ANY__",
            "outlier_count": int(is_anomaly.sum()),
            "outlier_ratio": float(is_anomaly.mean()),
        }
    )
    summary_df = pd.DataFrame(summary_rows).sort_values(by=["outlier_count"], ascending=False)
    summary_path = os.path.join(out_dir, summary_out_csv)
    summary_df.to_csv(summary_path, index=False)

    print(f"{dataset_label}: Outlier analysis saved anomalies={int(is_anomaly.sum())} to {anomalies_path}")
    print(f"{dataset_label}: Outlier summary saved to {summary_path}")


def scale_numeric_features(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    method: str,
) -> pd.DataFrame:
    """
    STEP 10 Standardize / Normalize Data:
    - zscore -> StandardScaler
    - minmax -> MinMaxScaler
    """
    out = df.copy()
    cols = [c for c in feature_cols if c in out.columns]
    if not cols:
        return out

    X = out[cols].to_numpy(dtype=float)
    if method == "zscore":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'zscore' or 'minmax'")

    X_scaled = scaler.fit_transform(X)
    out[cols] = X_scaled
    return out


def main() -> None:
    root = _project_root()
    this_dir = os.path.join(root, "this_datasets")
    _ensure_dir(this_dir)

    # EDA output: step-based folders under webp/EDA/data_preprocessing/steps/
    eda_steps = os.path.join(root, "webp", "EDA", "data_preprocessing", "steps")
    eda_07 = os.path.join(eda_steps, "07_feature_engineering")
    eda_08 = os.path.join(eda_steps, "08_encode_categorical")
    eda_09 = os.path.join(eda_steps, "09_handle_outliers")
    eda_10 = os.path.join(eda_steps, "10_standardize_normalize")
    eda_11 = os.path.join(eda_steps, "11_final_dataset_structure")
    for d in (eda_07, eda_08, eda_09, eda_10, eda_11):
        _ensure_dir(d)

    phil_in = os.path.join(this_dir, "philgeps_2025_medical_procurement.csv")
    doh_in = os.path.join(this_dir, "doh_medicine_distribution_2022_2025.csv")

    A_out = os.path.join(this_dir, "clustering_A_supplier_awardee_features.csv")
    B_out = os.path.join(this_dir, "clustering_B_medicine_procurement_pattern_features.csv")
    C_out = os.path.join(this_dir, "clustering_C_distribution_recipient_features.csv")

    A_interp_out = os.path.join(this_dir, "clustering_A_supplier_awardee_interpretation.csv")
    B_interp_out = os.path.join(this_dir, "clustering_B_medicine_procurement_pattern_interpretation.csv")
    C_interp_out = os.path.join(this_dir, "clustering_C_distribution_recipient_interpretation.csv")

    # ===== A: Supplier/Awardee Clustering =====
    if os.path.exists(phil_in) and os.path.getsize(phil_in) > 0:
        phil_df = _safe_read_csv(phil_in)
        if not phil_df.empty:
            # ---------------------------------------------------------------------
            # STEP 07 Feature Engineering
            # ---------------------------------------------------------------------
            A_features = preprocess_A_supplier_awardee(phil_df)
            saved = _safe_write_csv(A_features, A_out)
            print(f"A: Saved supplier features -> {saved} (rows={len(A_features)})")

            # EDA: STEP 07 - Feature distributions + correlation
            A_num_cols = ["contract_amount_total", "contract_amount_mean", "num_awards", "regions_served_nunique", "rows_count"]
            if generate_eda_visualizations:
                generate_eda_visualizations(
                    A_features,
                    out_dir=os.path.join(eda_07, "A_supplier_awardee"),
                    dataset_label="A",
                    stage_label="Feature Engineering (supplier/awardee)",
                    numeric_focus=[c for c in A_num_cols if c in A_features.columns],
                    include_correlation=True,
                )

            # ---------------------------------------------------------------------
            # STEP 11 Final Clean Dataset Structure - Interpretation table
            # ---------------------------------------------------------------------
            A_interp = build_A_interpretation_table(phil_df)
            _safe_write_csv(A_interp, A_interp_out)
            # EDA: STEP 11 - Final interpretation table (categorical composition)
            if generate_eda_visualizations:
                generate_eda_visualizations(
                    A_interp,
                    out_dir=os.path.join(eda_11, "A_supplier_awardee_interpretation"),
                    dataset_label="A",
                    stage_label="Final (interpretation table)",
                    numeric_focus=None,
                    include_correlation=False,
                )

            # ---------------------------------------------------------------------
            # STEP 09 Handle Outliers (retain; export anomaly subset)
            # ---------------------------------------------------------------------
            outlier_dir = os.path.join(eda_09, "A_supplier_awardee")
            analyze_outliers_iqr_and_boxplots(
                A_features,
                feature_cols=[
                    "contract_amount_total",
                    "contract_amount_mean",
                    "num_awards",
                    "regions_served_nunique",
                    "rows_count",
                ],
                dataset_label="A",
                out_dir=outlier_dir,
                anomalies_out_csv="clustering_A_supplier_awardee_features_anomalies.csv",
                summary_out_csv="clustering_A_supplier_awardee_outlier_summary.csv",
            )

            # ---------------------------------------------------------------------
            # STEP 10 Standardize / Normalize Data
            # ---------------------------------------------------------------------
            # Keep identifier columns (Awardee Organization Name) unscaled.
            A_scale_cols = [
                "contract_amount_total",
                "contract_amount_mean",
                "num_awards",
                "regions_served_nunique",
                "rows_count",
            ] + [c for c in A_features.columns if c.startswith("mode_count__") or c.startswith("region_count__")]

            A_z = scale_numeric_features(A_features, feature_cols=A_scale_cols, method="zscore")
            A_m = scale_numeric_features(A_features, feature_cols=A_scale_cols, method="minmax")
            _safe_write_csv(A_z, A_out.replace(".csv", "_zscore.csv"))
            _safe_write_csv(A_m, A_out.replace(".csv", "_minmax.csv"))

            # EDA: STEP 10 - Scaled feature distributions
            if generate_eda_visualizations:
                generate_eda_visualizations(
                    A_z,
                    out_dir=os.path.join(eda_10, "A_supplier_awardee_zscore"),
                    dataset_label="A",
                    stage_label="Standardized (zscore)",
                    numeric_focus=[c for c in A_scale_cols if c in A_z.columns],
                    include_correlation=True,
                )
        else:
            print("A: PhilGEPS input exists but is empty. Skipping A.")
    else:
        print("A: PhilGEPS input missing/empty. Skipping A.")

    # ===== B: Medicine Procurement Pattern =====
    if os.path.exists(phil_in) and os.path.getsize(phil_in) > 0:
        phil_df = _safe_read_csv(phil_in)
        if not phil_df.empty:
            # ---------------------------------------------------------------------
            # STEP 07 Feature Engineering + STEP 08 Encode Categorical Data
            # ---------------------------------------------------------------------
            B_features = preprocess_B_medicine_procurement_pattern(phil_df)
            saved = _safe_write_csv(B_features, B_out)
            print(f"B: Saved medicine pattern features -> {saved} (rows={len(B_features)})")

            # STEP 08 Encode Categorical - run log (B only; matches print output)
            with open(os.path.join(eda_08, "step_log.txt"), "w", encoding="utf-8") as f:
                f.write(f"B: Saved medicine pattern features -> {saved} (rows={len(B_features)})\n")

            # EDA: STEP 07/08 - Feature distributions
            B_num_cols = ["item_budget", "quantity", "log1p_item_budget", "log1p_quantity"]
            if generate_eda_visualizations:
                generate_eda_visualizations(
                    B_features,
                    out_dir=os.path.join(eda_07, "B_medicine_procurement_pattern"),
                    dataset_label="B",
                    stage_label="Feature Engineering + Encode Categorical",
                    numeric_focus=[c for c in B_num_cols if c in B_features.columns],
                    include_correlation=True,
                )

            # STEP 11 Final Clean Dataset Structure - Interpretation table
            keep_cols = [c for c in ["Awardee Organization Name", "Item Name", "UNSPSC Description", "Region of Awardee", "Procurement Mode", "Funding Source", "item_budget", "quantity"] if c in B_features.columns]
            B_interp = B_features[keep_cols].copy()
            _safe_write_csv(B_interp, B_interp_out)
            # EDA: STEP 11 - Final interpretation table
            if generate_eda_visualizations:
                generate_eda_visualizations(
                    B_interp,
                    out_dir=os.path.join(eda_11, "B_medicine_procurement_pattern_interpretation"),
                    dataset_label="B",
                    stage_label="Final (interpretation table)",
                    numeric_focus=[c for c in ["item_budget", "quantity"] if c in B_interp.columns],
                    include_correlation=True,
                )

            # STEP 09 Handle Outliers
            outlier_dir = os.path.join(eda_09, "B_medicine_pattern")
            analyze_outliers_iqr_and_boxplots(
                B_features,
                feature_cols=[
                    "item_budget",
                    "quantity",
                    "log1p_item_budget",
                    "log1p_quantity",
                ],
                dataset_label="B",
                out_dir=outlier_dir,
                anomalies_out_csv="clustering_B_medicine_procurement_pattern_features_anomalies.csv",
                summary_out_csv="clustering_B_medicine_procurement_pattern_outlier_summary.csv",
            )

            # ---------------------------------------------------------------------
            # STEP 10 Standardize / Normalize Data
            # ---------------------------------------------------------------------
            # Identifiers can be numeric due to CSV dtype inference; scaling them would bias clusters.
            exclude_from_scaling = {"Award Reference No.", "UNSPSC Code"}
            B_scale_cols = [
                c
                for c in B_features.columns
                if c not in exclude_from_scaling and pd.api.types.is_numeric_dtype(B_features[c])
            ]

            B_z = scale_numeric_features(B_features, feature_cols=B_scale_cols, method="zscore")
            B_m = scale_numeric_features(B_features, feature_cols=B_scale_cols, method="minmax")
            _safe_write_csv(B_z, B_out.replace(".csv", "_zscore.csv"))
            _safe_write_csv(B_m, B_out.replace(".csv", "_minmax.csv"))

            # EDA: STEP 10 - Scaled distributions
            if generate_eda_visualizations and B_scale_cols:
                generate_eda_visualizations(
                    B_z,
                    out_dir=os.path.join(eda_10, "B_medicine_procurement_pattern_zscore"),
                    dataset_label="B",
                    stage_label="Standardized (zscore)",
                    numeric_focus=B_scale_cols[:12],
                    include_correlation=True,
                )
        else:
            print("B: PhilGEPS input exists but is empty. Skipping B.")
    else:
        print("B: PhilGEPS input missing/empty. Skipping B.")

    # ===== C: Distribution Pattern =====
    if os.path.exists(doh_in) and os.path.getsize(doh_in) > 0:
        doh_df = _safe_read_csv(doh_in)
        if not doh_df.empty:
            # ---------------------------------------------------------------------
            # STEP 07 Feature Engineering
            # ---------------------------------------------------------------------
            C_features = preprocess_C_distribution_pattern(doh_df)
            saved = _safe_write_csv(C_features, C_out)
            print(f"C: Saved recipient distribution features -> {saved} (rows={len(C_features)})")

            # EDA: STEP 07 - Feature distributions
            C_num_cols = ["medicines_received_lines", "medicines_received_unique", "quantity_total", "total_amount_total", "delivery_frequency_unique_dates"]
            if generate_eda_visualizations:
                generate_eda_visualizations(
                    C_features,
                    out_dir=os.path.join(eda_07, "C_distribution_recipient"),
                    dataset_label="C",
                    stage_label="Feature Engineering (distribution)",
                    numeric_focus=[c for c in C_num_cols if c in C_features.columns],
                    include_correlation=True,
                )

            # STEP 11 Final Clean Dataset Structure - Interpretation table
            keep_cols = [c for c in ["RECIPIENT", "top_program", "top_item_description", "quantity_total", "total_amount_total", "delivery_frequency_unique_dates"] if c in C_features.columns]
            C_interp = C_features[keep_cols].copy()
            _safe_write_csv(C_interp, C_interp_out)
            # EDA: STEP 11 - Final interpretation table
            if generate_eda_visualizations:
                generate_eda_visualizations(
                    C_interp,
                    out_dir=os.path.join(eda_11, "C_distribution_recipient_interpretation"),
                    dataset_label="C",
                    stage_label="Final (interpretation table)",
                    numeric_focus=[c for c in ["quantity_total", "total_amount_total", "delivery_frequency_unique_dates"] if c in C_interp.columns],
                    include_correlation=True,
                )

            # STEP 09 Handle Outliers
            outlier_dir = os.path.join(eda_09, "C_distribution_recipient")
            analyze_outliers_iqr_and_boxplots(
                C_features,
                feature_cols=[
                    "medicines_received_lines",
                    "medicines_received_unique",
                    "quantity_total",
                    "total_amount_total",
                    "delivery_frequency_unique_dates",
                ],
                dataset_label="C",
                out_dir=outlier_dir,
                anomalies_out_csv="clustering_C_distribution_recipient_features_anomalies.csv",
                summary_out_csv="clustering_C_distribution_recipient_outlier_summary.csv",
            )

            # ---------------------------------------------------------------------
            # STEP 10 Standardize / Normalize Data
            # ---------------------------------------------------------------------
            C_scale_cols = [c for c in C_features.columns if c != "RECIPIENT" and pd.api.types.is_numeric_dtype(C_features[c])]
            C_z = scale_numeric_features(C_features, feature_cols=C_scale_cols, method="zscore")
            C_m = scale_numeric_features(C_features, feature_cols=C_scale_cols, method="minmax")
            _safe_write_csv(C_z, C_out.replace(".csv", "_zscore.csv"))
            _safe_write_csv(C_m, C_out.replace(".csv", "_minmax.csv"))

            # EDA: STEP 10 - Scaled distributions
            if generate_eda_visualizations and C_scale_cols:
                generate_eda_visualizations(
                    C_z,
                    out_dir=os.path.join(eda_10, "C_distribution_recipient_zscore"),
                    dataset_label="C",
                    stage_label="Standardized (zscore)",
                    numeric_focus=C_scale_cols[:12],
                    include_correlation=True,
                )
        else:
            print("C: DOH input exists but is empty. Skipping C.")
    else:
        print("C: DOH input missing/empty. Skipping C.")

    # Write export summary for step 11
    summary_path = os.path.join(eda_11, "export_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MedFlow — Step 11 Final Dataset Structure — Export Summary\n")
        f.write("=" * 60 + "\n")
        for name, path in [
            ("A features", A_out),
            ("A zscore", A_out.replace(".csv", "_zscore.csv")),
            ("A minmax", A_out.replace(".csv", "_minmax.csv")),
            ("A interpretation", A_interp_out),
            ("B features", B_out),
            ("B zscore", B_out.replace(".csv", "_zscore.csv")),
            ("B minmax", B_out.replace(".csv", "_minmax.csv")),
            ("B interpretation", B_interp_out),
            ("C features", C_out),
            ("C zscore", C_out.replace(".csv", "_zscore.csv")),
            ("C minmax", C_out.replace(".csv", "_minmax.csv")),
            ("C interpretation", C_interp_out),
        ]:
            if os.path.exists(path):
                with open(path, encoding="utf-8", errors="ignore") as rf:
                    n = sum(1 for _ in rf) - 1
                f.write(f"{name}: {path} ({n} rows)\n")
            else:
                f.write(f"{name}: (not generated)\n")


if __name__ == "__main__":
    main()


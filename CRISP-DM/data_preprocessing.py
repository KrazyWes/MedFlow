"""
Clustering Data Preprocessing (A/B/C)

Builds clustering-ready feature tables for:

A. Supplier / Awardee Clustering (PhilGEPS)
   - contract amount (sum)
   - number of awards
   - regions served (nunique)
   - procurement mode mix (one-hot top modes)

B. Medicine Procurement Pattern Clustering (PhilGEPS)
   - item budget, quantity (numeric)
   - procurement mode + funding source (one-hot top categories)

C. Distribution Pattern Clustering (DOH)
   - medicines received (line count + unique item count)
   - total quantity + total amount
   - delivery frequency derived from delivery dates

Outputs saved into: `this_datasets/`
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd


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
    Aggregate per Awardee and produce numeric-ready features for clustering.

    Grain (rows): one per Awardee Organization Name
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


def preprocess_B_medicine_procurement_pattern(
    philgeps_df: pd.DataFrame,
    *,
    top_k_modes: int = 10,
    top_k_funding: int = 10,
) -> pd.DataFrame:
    """
    Row-level features for clustering procurement patterns.

    Grain (rows): each PhilGEPS line item row in the medical-filtered dataset.
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

    trace_cols = [c for c in ["Awardee Organization Name", "Award Reference No.", "UNSPSC Code", "Item Name"] if c in df.columns]

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
            df[trace_cols].reset_index(drop=True),
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
    Aggregate distribution patterns per recipient.

    Grain (rows): one per RECIPIENT
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

    out = (
        df.groupby("RECIPIENT")
        .agg(
            medicines_received_lines=("ITEM DESCRIPTION", "count"),
            medicines_received_unique=("ITEM DESCRIPTION", "nunique"),
            quantity_total=("QUANTITY", "sum"),
            total_amount_total=("TOTAL AMOUNT", "sum"),
            delivery_frequency_unique_dates=("delivery_date_only", "nunique"),
        )
        .reset_index()
    )
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def main() -> None:
    root = _project_root()
    this_dir = os.path.join(root, "this_datasets")
    _ensure_dir(this_dir)

    phil_in = os.path.join(this_dir, "philgeps_2025_medical_procurement.csv")
    doh_in = os.path.join(this_dir, "doh_medicine_distribution_2022_2025.csv")

    A_out = os.path.join(this_dir, "clustering_A_supplier_awardee_features.csv")
    B_out = os.path.join(this_dir, "clustering_B_medicine_procurement_pattern_features.csv")
    C_out = os.path.join(this_dir, "clustering_C_distribution_recipient_features.csv")

    # ===== A =====
    if os.path.exists(phil_in) and os.path.getsize(phil_in) > 0:
        phil_df = _safe_read_csv(phil_in)
        if not phil_df.empty:
            A_features = preprocess_A_supplier_awardee(phil_df)
            saved = _safe_write_csv(A_features, A_out)
            print(f"A: Saved supplier features -> {saved} (rows={len(A_features)})")
        else:
            print("A: PhilGEPS input exists but is empty. Skipping A.")
    else:
        print("A: PhilGEPS input missing/empty. Skipping A.")

    # ===== B =====
    if os.path.exists(phil_in) and os.path.getsize(phil_in) > 0:
        phil_df = _safe_read_csv(phil_in)
        if not phil_df.empty:
            B_features = preprocess_B_medicine_procurement_pattern(phil_df)
            saved = _safe_write_csv(B_features, B_out)
            print(f"B: Saved medicine pattern features -> {saved} (rows={len(B_features)})")
        else:
            print("B: PhilGEPS input exists but is empty. Skipping B.")
    else:
        print("B: PhilGEPS input missing/empty. Skipping B.")

    # ===== C =====
    if os.path.exists(doh_in) and os.path.getsize(doh_in) > 0:
        doh_df = _safe_read_csv(doh_in)
        if not doh_df.empty:
            C_features = preprocess_C_distribution_pattern(doh_df)
            saved = _safe_write_csv(C_features, C_out)
            print(f"C: Saved recipient distribution features -> {saved} (rows={len(C_features)})")
        else:
            print("C: DOH input exists but is empty. Skipping C.")
    else:
        print("C: DOH input missing/empty. Skipping C.")


if __name__ == "__main__":
    main()


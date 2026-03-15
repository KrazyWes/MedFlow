"""
Modeling Preparation for PhilGEPS and DOH Medical Data

Prepares cleaned data for K-means clustering. Supports three objectives:
  A. Supplier / Awardee Clustering (PhilGEPS)   - aggregate to supplier level
  B. Medicine Procurement Pattern (PhilGEPS)   - line-item level
  C. Distribution Pattern Clustering (DOH)     - aggregate to recipient level

Outputs: modeling-ready DataFrames and scaled arrays saved to model_prep_outputs/.
Usage: python CRISP-DM/modeling_preparation.py
"""

import os
import pickle
import pandas as pd
import numpy as np

# Optional: sklearn for scaling and encoding
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# =============================================================================
# Configuration
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
philgeps_path = os.path.join(project_root, "this_datasets", "philgeps_2025_medical_procurement.csv")
doh_path = os.path.join(project_root, "this_datasets", "doh_medicine_distribution_2022_2025.csv")
output_dir = os.path.join(project_root, "model_prep_outputs")
os.makedirs(output_dir, exist_ok=True)

# Outlier handling: "remove" (drop rows beyond IQR) or "winsorize" (cap at percentiles)
OUTLIER_METHOD = "remove"
OUTLIER_IQR_MULT = 1.5       # standard outliers: beyond Q1 - 1.5*IQR, Q3 + 1.5*IQR
EXTREME_OUTLIER_IQR_MULT = 3.0   # extreme outliers: beyond Q1 - 3*IQR, Q3 + 3*IQR
REMOVE_EXTREME_FIRST = True  # if True, remove extreme outliers first, then standard


# =============================================================================
# Outlier handling
# =============================================================================
def clean_outliers(df, numeric_cols, method=None, iqr_mult=None):
    """
    Remove or winsorize outliers on numeric columns using IQR rule.
    numeric_cols: list of column names present in df.
    method: "remove" or "winsorize". Uses OUTLIER_METHOD if None.
    iqr_mult: IQR multiplier (e.g. 1.5 = standard, 3.0 = extreme). Uses OUTLIER_IQR_MULT if None.
    Returns: (df_cleaned, n_removed)
    """
    method = method or OUTLIER_METHOD
    iqr_mult = iqr_mult if iqr_mult is not None else OUTLIER_IQR_MULT
    cols = [c for c in numeric_cols if c in df.columns]
    if not cols:
        return df, 0

    n_before = len(df)
    out = df.copy()

    for col in cols:
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            iqr = out[col].std() or 1e-8
        low = q1 - iqr_mult * iqr
        high = q3 + iqr_mult * iqr

        if method == "remove":
            out = out[(out[col] >= low) & (out[col] <= high)]
        else:
            out[col] = out[col].clip(lower=low, upper=high)

    n_after = len(out)
    return out, n_before - n_after


def clean_outliers_two_stage(df, numeric_cols):
    """
    First remove extreme outliers (3*IQR), then standard outliers (1.5*IQR).
    Returns: (df_cleaned, n_removed_total)
    """
    if not REMOVE_EXTREME_FIRST:
        return clean_outliers(df, numeric_cols)
    out, n_extreme = clean_outliers(df, numeric_cols, iqr_mult=EXTREME_OUTLIER_IQR_MULT)
    out, n_standard = clean_outliers(out, numeric_cols, iqr_mult=OUTLIER_IQR_MULT)
    if n_extreme or n_standard:
        print(f"  Outlier breakdown: extreme (3*IQR)={n_extreme}, standard (1.5*IQR)={n_standard}")
    return out, n_extreme + n_standard


# =============================================================================
# Load Data
# =============================================================================
def load_philgeps():
    """Load cleaned PhilGEPS dataset."""
    if not os.path.isfile(philgeps_path):
        raise FileNotFoundError(f"Run data_preparation.py first. Expected: {philgeps_path}")
    df = pd.read_csv(philgeps_path, low_memory=False)
    print(f"Loaded {len(df)} rows from {philgeps_path}")
    return df


def load_doh():
    """Load cleaned DOH distribution dataset."""
    if not os.path.isfile(doh_path):
        raise FileNotFoundError(f"Run data_preparation.py first. Expected: {doh_path}")
    df = pd.read_csv(doh_path, low_memory=False)
    print(f"Loaded {len(df)} rows from {doh_path}")
    return df


# =============================================================================
# Objective A: Supplier / Awardee Clustering
# Cluster by: contract amounts, number of awards, regions served, procurement modes
# =============================================================================
def prepare_supplier_clustering(df):
    """
    Aggregate line items to supplier (awardee) level and create modeling features.
    Returns: aggregated df, X (array), feature_names, scaler, metadata
    """
    key_col = "Awardee Organization Name"
    if key_col not in df.columns:
        raise ValueError(f"Required column '{key_col}' not found")

    # Ensure numeric
    amount_col = "Contract Amount"
    for c in [amount_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Aggregate per supplier
    agg = df.groupby(key_col).agg(
        total_contract_amount=("Contract Amount", "sum"),
        num_awards=("Award Reference No.", "nunique"),
    ).reset_index()

    # Regions served: one row per award - use first region/province/city per supplier
    region_cols = ["Region of Awardee", "Province of Awardee", "City/Municipality of Awardee"]
    for col in region_cols:
        if col in df.columns:
            first_region = df.groupby(key_col)[col].first().reset_index()
            first_region.columns = [key_col, col]
            agg = agg.merge(first_region, on=key_col, how="left")

    # Procurement mode: use mode per supplier (most common)
    if "Procurement Mode" in df.columns:
        mode_per_supplier = (
            df.groupby(key_col)["Procurement Mode"]
            .apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "N/A")
            .reset_index()
        )
        mode_per_supplier.columns = [key_col, "Procurement Mode"]
        agg = agg.merge(mode_per_supplier, on=key_col, how="left")

    # Handle missing
    agg["total_contract_amount"] = agg["total_contract_amount"].fillna(0)
    agg["num_awards"] = agg["num_awards"].fillna(0)

    # Remove extreme then standard outliers on numeric features
    agg, n_removed = clean_outliers_two_stage(agg, ["total_contract_amount", "num_awards"])
    if n_removed:
        print(f"  Outliers removed: {n_removed} suppliers")

    # Build feature matrix
    numeric_cols = ["total_contract_amount", "num_awards"]
    cat_cols = ["Region of Awardee", "Procurement Mode"]
    cat_cols = [c for c in cat_cols if c in agg.columns]

    X_numeric = agg[numeric_cols].values
    feature_names = list(numeric_cols)

    # One-hot encode categorical (top N categories to avoid explosion)
    X_cat_list = []
    encoders = {}
    for col in cat_cols:
        vals = agg[col].fillna("N/A").astype(str)
        top_n = vals.value_counts().head(15).index.tolist()
        vals = vals.apply(lambda x: x if x in top_n else "Other")
        if HAS_SKLEARN:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            enc.fit(vals.values.reshape(-1, 1))
            X_cat = enc.transform(vals.values.reshape(-1, 1))
            encoders[col] = enc
            for i, cat in enumerate(enc.categories_[0]):
                feature_names.append(f"{col}__{cat}")
        else:
            dummies = pd.get_dummies(vals, prefix=col, prefix_sep="__")
            X_cat = dummies.values
            feature_names.extend(dummies.columns.tolist())
            encoders[col] = None
        X_cat_list.append(X_cat)

    if X_cat_list:
        X = np.hstack([X_numeric] + X_cat_list)
    else:
        X = X_numeric

    # Scale
    scaler = None
    if HAS_SKLEARN:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        # Manual z-score scaling
        X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    metadata = {
        "objective": "A_supplier_awardee",
        "n_samples": len(agg),
        "n_features": X.shape[1],
        "feature_names": feature_names,
    }
    return agg, X_scaled, feature_names, scaler, encoders, metadata


# =============================================================================
# Objective B: Medicine Procurement Pattern Clustering
# Cluster by: item budget, quantity, procurement mode, funding source
# =============================================================================
def prepare_procurement_pattern_clustering(df):
    """
    Prepare line-item features for medicine procurement pattern clustering.
    Returns: df subset, X (array), feature_names, scaler, encoders, metadata
    """
    numeric_cols = ["Item Budget", "Quantity"]
    cat_cols = ["Procurement Mode", "Funding Source"]

    for c in numeric_cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found")
    for c in cat_cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found")

    sub = df[numeric_cols + cat_cols].copy()
    sub[numeric_cols] = sub[numeric_cols].apply(pd.to_numeric, errors="coerce")
    sub = sub.fillna({"Item Budget": 0, "Quantity": 0})
    sub = sub.replace([np.inf, -np.inf], 0)

    # Remove extreme then standard outliers on numeric features
    sub, n_removed = clean_outliers_two_stage(sub, numeric_cols)
    if n_removed:
        print(f"  Outliers removed: {n_removed} line items")

    X_numeric = sub[numeric_cols].values
    feature_names = list(numeric_cols)

    X_cat_list = []
    encoders = {}
    for col in cat_cols:
        vals = sub[col].fillna("N/A").astype(str)
        top_n = vals.value_counts().head(20).index.tolist()
        vals = vals.apply(lambda x: x if x in top_n else "Other")
        if HAS_SKLEARN:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            enc.fit(vals.values.reshape(-1, 1))
            X_cat = enc.transform(vals.values.reshape(-1, 1))
            encoders[col] = enc
            for cat in enc.categories_[0]:
                feature_names.append(f"{col}__{cat}")
        else:
            dummies = pd.get_dummies(vals, prefix=col, prefix_sep="__")
            X_cat = dummies.values
            feature_names.extend(dummies.columns.tolist())
            encoders[col] = None
        X_cat_list.append(X_cat)

    X = np.hstack([X_numeric] + X_cat_list)

    scaler = None
    if HAS_SKLEARN:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    metadata = {
        "objective": "B_medicine_procurement_pattern",
        "n_samples": len(sub),
        "n_features": X.shape[1],
        "feature_names": feature_names,
    }
    return sub, X_scaled, feature_names, scaler, encoders, metadata


# =============================================================================
# Objective C: Distribution Pattern Clustering (DOH)
# Cluster recipients by: medicines received, quantity, delivery frequency, total amount
# =============================================================================
def prepare_distribution_clustering(df):
    """
    Aggregate line items to recipient level and create modeling features.
    Returns: aggregated df, X (array), feature_names, scaler, encoders, metadata
    """
    key_col = "RECIPIENT"
    date_col = None
    for c in df.columns:
        if "DATE" in c.upper() and ("DELIVER" in c.upper() or "PICKED" in c.upper()):
            date_col = c
            break
    if not date_col:
        date_col = [c for c in df.columns if "DATE" in c.upper()]
        date_col = date_col[0] if date_col else None

    if key_col not in df.columns:
        raise ValueError(f"Required column '{key_col}' not found")
    if "QUANTITY" not in df.columns or "TOTAL AMOUNT" not in df.columns:
        raise ValueError("Required columns 'QUANTITY' and 'TOTAL AMOUNT' not found")

    df = df.copy()
    df["QUANTITY"] = pd.to_numeric(df["QUANTITY"], errors="coerce")
    df["TOTAL AMOUNT"] = pd.to_numeric(df["TOTAL AMOUNT"], errors="coerce")
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    agg_args = {
        "total_quantity": ("QUANTITY", "sum"),
        "total_amount": ("TOTAL AMOUNT", "sum"),
        "num_distinct_medicines": ("ITEM DESCRIPTION", "nunique"),
    }
    if date_col:
        agg_args["delivery_frequency"] = (date_col, "nunique")
    agg = df.groupby(key_col).agg(**agg_args).reset_index()

    agg["total_quantity"] = agg["total_quantity"].fillna(0)
    agg["total_amount"] = agg["total_amount"].fillna(0)
    agg["num_distinct_medicines"] = agg["num_distinct_medicines"].fillna(0)
    if "delivery_frequency" in agg.columns:
        agg["delivery_frequency"] = agg["delivery_frequency"].fillna(0)

    # Remove extreme then standard outliers on numeric features
    numeric_cols = ["total_quantity", "total_amount", "num_distinct_medicines"]
    if "delivery_frequency" in agg.columns:
        numeric_cols.append("delivery_frequency")
    agg, n_removed = clean_outliers_two_stage(agg, numeric_cols)
    if n_removed:
        print(f"  Outliers removed: {n_removed} recipients")

    X_numeric = agg[numeric_cols].values
    X_numeric = np.nan_to_num(X_numeric, nan=0.0, posinf=0.0, neginf=0.0)
    feature_names = list(numeric_cols)

    scaler = None
    if HAS_SKLEARN:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
    else:
        X_scaled = (X_numeric - X_numeric.mean(axis=0)) / (X_numeric.std(axis=0) + 1e-8)

    metadata = {
        "objective": "C_distribution_pattern",
        "n_samples": len(agg),
        "n_features": X_scaled.shape[1],
        "feature_names": feature_names,
    }
    return agg, X_scaled, feature_names, scaler, {}, metadata


# =============================================================================
# Main
# =============================================================================
def main():
    if not HAS_SKLEARN:
        print("WARNING: scikit-learn not installed. Using manual scaling. Install: pip install scikit-learn")

    df = load_philgeps()

    # --- Objective A: Supplier Clustering ---
    print("\n" + "=" * 60)
    print("Objective A: Supplier / Awardee Clustering")
    print("=" * 60)
    agg_a, X_a, names_a, scaler_a, enc_a, meta_a = prepare_supplier_clustering(df)
    out_agg = os.path.join(output_dir, "philgeps_supplier_aggregated.csv")
    agg_a.to_csv(out_agg, index=False)
    np.save(os.path.join(output_dir, "philgeps_supplier_X.npy"), X_a)
    with open(os.path.join(output_dir, "philgeps_supplier_meta.pkl"), "wb") as f:
        pickle.dump({"feature_names": names_a, "scaler": scaler_a}, f)
    print(f"  Samples: {meta_a['n_samples']}, Features: {meta_a['n_features']}")
    print(f"  Saved: {out_agg}")

    # --- Objective B: Medicine Procurement Pattern ---
    print("\n" + "=" * 60)
    print("Objective B: Medicine Procurement Pattern Clustering")
    print("=" * 60)
    sub_b, X_b, names_b, scaler_b, enc_b, meta_b = prepare_procurement_pattern_clustering(df)
    out_sub = os.path.join(output_dir, "philgeps_procurement_line_items.csv")
    sub_b.to_csv(out_sub, index=False)
    np.save(os.path.join(output_dir, "philgeps_procurement_X.npy"), X_b)
    with open(os.path.join(output_dir, "philgeps_procurement_meta.pkl"), "wb") as f:
        pickle.dump({"feature_names": names_b, "scaler": scaler_b}, f)
    print(f"  Samples: {meta_b['n_samples']}, Features: {meta_b['n_features']}")
    print(f"  Saved: {out_sub}")

    # --- Objective C: DOH Distribution Pattern ---
    print("\n" + "=" * 60)
    print("Objective C: Distribution Pattern Clustering (DOH)")
    print("=" * 60)
    df_doh = load_doh()
    agg_c, X_c, names_c, scaler_c, enc_c, meta_c = prepare_distribution_clustering(df_doh)
    out_agg_c = os.path.join(output_dir, "doh_distribution_aggregated.csv")
    agg_c.to_csv(out_agg_c, index=False)
    np.save(os.path.join(output_dir, "doh_distribution_X.npy"), X_c)
    with open(os.path.join(output_dir, "doh_distribution_meta.pkl"), "wb") as f:
        pickle.dump({"feature_names": names_c, "scaler": scaler_c}, f)
    print(f"  Samples: {meta_c['n_samples']}, Features: {meta_c['n_features']}")
    print(f"  Saved: {out_agg_c}")

    print("\nModeling preparation complete. Outputs in model_prep_outputs/")


if __name__ == "__main__":
    main()

import pandas as pd


def remove_peso_and_currency_chars(value):
    """Remove peso sign (₱), PHP, commas from currency values."""
    if pd.isna(value):
        return value
    s = str(value).strip()
    for char in ("\u20b1", "₱", "PHP", "Php", "php"):
        s = s.replace(char, "")
    s = s.replace(",", "").strip()
    return s if s else value


def clean_doh_dataframe(df: pd.DataFrame, currency_cols=None) -> pd.DataFrame:
    """Remove peso signs and convert currency columns to numeric."""
    if currency_cols is None:
        currency_cols = ["UNIT COST", "TOTAL AMOUNT"]

    df = df.copy()
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]
    currency_cols_clean = [c.replace("\n", " ").strip() for c in currency_cols]

    # Strip peso from all object columns to reduce downstream encoding/type issues.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: str(x).replace("\u20b1", "").replace("₱", "").strip()
                if pd.notna(x)
                else x
            )

    # Convert currency columns to numeric.
    for col in currency_cols_clean:
        if col in df.columns:
            df[col] = df[col].apply(remove_peso_and_currency_chars)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def handle_missing_values_doh(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in DOH dataset.
    - Computable: TOTAL AMOUNT = QUANTITY * UNIT COST (and vice versa where possible)
    - Numeric: impute with median
    - Categorical: impute with 'N/A'
    """
    df = df.copy()
    q_col = "QUANTITY"
    uc_col = "UNIT COST"
    ta_col = "TOTAL AMOUNT"

    for col in [q_col, uc_col, ta_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if all(c in df.columns for c in [q_col, uc_col, ta_col]):
        # TOTAL AMOUNT missing
        mask = df[ta_col].isna() & df[q_col].notna() & df[uc_col].notna()
        df.loc[mask, ta_col] = (df.loc[mask, q_col] * df.loc[mask, uc_col]).astype(float)

        # UNIT COST missing
        mask = df[uc_col].isna() & df[ta_col].notna() & (df[q_col] > 0)
        df.loc[mask, uc_col] = (df.loc[mask, ta_col] / df.loc[mask, q_col]).astype(float)

        # QUANTITY missing
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


def handle_missing_values_philgeps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in PhilGEPS dataset.
    - Numeric (amounts, quantities): impute with median (0 if all missing)
    - Categorical: impute with mode, or 'N/A' if no mode
    """
    df = df.copy()

    numeric_cols = [
        "Contract Amount",
        "Item Budget",
        "Quantity",
        "Approved Budget of the Contract",
        "Line Item No",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)

    cat_cols = [c for c in df.columns if df[c].dtype == object]
    for col in cat_cols:
        mode_val = df[col].mode()
        fill_val = mode_val.iloc[0] if len(mode_val) > 0 and pd.notna(mode_val.iloc[0]) else "N/A"
        df[col] = df[col].fillna(fill_val).replace("", fill_val)
        df[col] = df[col].replace("nan", "N/A").replace("None", "N/A")

    return df


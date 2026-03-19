import pandas as pd


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


"""
Data Preparation

Loads and prepares data from raw_datasets/ for downstream clustering analysis.
Supports both CSV and Excel (.xlsx, .xls) inputs. Skips processing if a source
folder is empty or has no supported files.

Missing value handling:
- Computable: DOH TOTAL AMOUNT = QUANTITY * UNIT COST (and inverse) where possible
- Numeric: impute with median (0 if all missing)
- Categorical: impute with mode, or 'N/A' if no mode; empty strings -> 'N/A'

1. DOH Medicine Procurement and Distribution
   Input: raw_datasets/DOH/  (CSV or Excel)
   Output: this_datasets/doh_medicine_distribution_2022_2025.csv
   Use case: Distribution Pattern Clustering (C)

2. PhilGEPS Medical Procurement
   Input: raw_datasets/PhilGEPS/  (CSV or Excel)
   Output: this_datasets/philgeps_2025_medical_procurement.csv
   Use cases: Supplier/Awardee Clustering (A), Medicine Procurement Pattern (B)

================================================================================
CLUSTERING OBJECTIVES (informs required columns for cleaning)
================================================================================

A. Supplier / Awardee Clustering (PhilGEPS)
   Cluster suppliers based on:
   - Contract Amount
   - Number of awards (aggregate)
   - Regions served (Region of Awardee, Province of Awardee, City/Municipality)
   - Procurement Mode
   Key columns: Awardee Organization Name, Contract Amount, Procurement Mode,
                Region of Awardee, Province of Awardee, City/Municipality of Awardee

B. Medicine Procurement Pattern Clustering (PhilGEPS)
   Cluster procurement records based on:
   - Item Budget
   - Quantity
   - Procurement Mode
   - Funding Source
   Key columns: Item Budget, Quantity, Procurement Mode, Funding Source,
                Item Name, Item Description, UNSPSC Description

C. Distribution Pattern Clustering (DOH)
   Cluster recipients based on:
   - Medicines received (ITEM DESCRIPTION)
   - Quantity
   - Delivery frequency (derive from delivery dates per recipient)
   - Total amount
   Key columns: RECIPIENT, ITEM DESCRIPTION, QUANTITY, TOTAL AMOUNT,
                DATE DELIVERED (or equivalent)
================================================================================
"""

import os
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
output_dir = os.path.join(project_root, "this_datasets")
os.makedirs(output_dir, exist_ok=True)

# Supported file extensions for flexible input
CSV_EXTENSIONS = (".csv",)
EXCEL_EXTENSIONS = (".xlsx", ".xls")
ALL_DATA_EXTENSIONS = CSV_EXTENSIONS + EXCEL_EXTENSIONS


def get_supported_files(directory):
    """Return list of (filepath, extension) for CSV/Excel files in directory."""
    if not os.path.isdir(directory):
        return []
    files = []
    for f in os.listdir(directory):
        fp = os.path.join(directory, f)
        if os.path.isfile(fp):
            ext = os.path.splitext(f)[1].lower()
            if ext in ALL_DATA_EXTENSIONS:
                files.append((fp, ext))
    return sorted(files, key=lambda x: x[0])


def load_csv(path):
    """Load a CSV file. Uses low_memory=False to avoid mixed-type warnings."""
    return pd.read_csv(path, low_memory=False)


def load_excel(path, dtype=str):
    """Load an Excel file. Returns list of DataFrames (one per sheet)."""
    xl = pd.ExcelFile(path)
    kwargs = {"dtype": dtype} if dtype is not None else {}
    return [pd.read_excel(path, sheet_name=name, **kwargs) for name in xl.sheet_names]


def remove_peso_and_currency_chars(value):
    """Remove peso sign (₱), PHP, commas from currency values."""
    if pd.isna(value):
        return value
    s = str(value).strip()
    for char in ("\u20b1", "₱", "PHP", "Php", "php"):
        s = s.replace(char, "")
    s = s.replace(",", "").strip()
    return s if s else value


def clean_doh_dataframe(df, currency_cols=None):
    """Remove peso signs and convert currency columns to numeric."""
    if currency_cols is None:
        currency_cols = ["UNIT COST", "TOTAL AMOUNT"]
    # Normalize column names (handle newlines in headers)
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]
    currency_cols_clean = [c.replace("\n", " ").strip() for c in currency_cols]
    # Remove peso from all object columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: str(x).replace("\u20b1", "").replace("₱", "").strip()
                if pd.notna(x) else x
            )
    for col in currency_cols_clean:
        if col in df.columns:
            df[col] = df[col].apply(remove_peso_and_currency_chars)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def handle_missing_values_doh(df):
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

    # Ensure numeric columns are numeric
    for col in [q_col, uc_col, ta_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Step 1: Compute derived values where possible
    if all(c in df.columns for c in [q_col, uc_col, ta_col]):
        # TOTAL AMOUNT missing, but QUANTITY and UNIT COST present
        mask = df[ta_col].isna() & df[q_col].notna() & df[uc_col].notna()
        df.loc[mask, ta_col] = (df.loc[mask, q_col] * df.loc[mask, uc_col]).astype(float)
        # UNIT COST missing, but TOTAL and QUANTITY present (avoid div by zero)
        mask = df[uc_col].isna() & df[ta_col].notna() & (df[q_col] > 0)
        df.loc[mask, uc_col] = (df.loc[mask, ta_col] / df.loc[mask, q_col]).astype(float)
        # QUANTITY missing, but TOTAL and UNIT COST present
        mask = df[q_col].isna() & df[ta_col].notna() & (df[uc_col] > 0)
        df.loc[mask, q_col] = (df.loc[mask, ta_col] / df.loc[mask, uc_col]).astype(float)

    # Step 2: Impute remaining numeric with median
    numeric_cols = [c for c in [q_col, uc_col, ta_col] if c in df.columns]
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)

    # Step 3: Impute categorical with 'N/A'
    cat_cols = [c for c in df.columns if c not in numeric_cols and df[c].dtype == object]
    for col in cat_cols:
        df[col] = df[col].fillna("N/A").replace("", "N/A").replace("nan", "N/A")

    return df


def handle_missing_values_philgeps(df):
    """
    Handle missing values in PhilGEPS dataset.
    - Numeric (amounts, quantities): impute with median (0 if all missing)
    - Categorical: impute with mode, or 'N/A' if no mode
    """
    df = df.copy()
    numeric_cols = [
        "Contract Amount", "Item Budget", "Quantity", "Approved Budget of the Contract",
        "Line Item No"
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)

    # Categorical: mode or 'N/A'
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    for col in cat_cols:
        mode_val = df[col].mode()
        fill_val = mode_val.iloc[0] if len(mode_val) > 0 and pd.notna(mode_val.iloc[0]) else "N/A"
        df[col] = df[col].fillna(fill_val).replace("", fill_val)
        # Handle 'nan' string from dtype conversion
        df[col] = df[col].replace("nan", "N/A").replace("None", "N/A")

    # Date columns: fill with NaT stays, or we could use a placeholder - for CSV, NaT becomes empty
    # Leaving dates as-is; they'll export as empty string if NaN
    return df


# =============================================================================
# Data Preparation - DOH Medicine Procurement and Distribution
# For: Distribution Pattern Clustering (C)
# =============================================================================
doh_input_dir = os.path.join(project_root, "raw_datasets", "DOH")
doh_output_filename = "doh_medicine_distribution_2022_2025.csv"
doh_sheets_years = ["2022", "2023", "2024", "2025"]  # Excel sheets to use if present

doh_files = get_supported_files(doh_input_dir)
if doh_files:
    doh_dfs = []
    for filepath, ext in doh_files:
        try:
            if ext in CSV_EXTENSIONS:
                df = load_csv(filepath)
                df = clean_doh_dataframe(df)
                df["Year"] = "CSV"  # Source indicator when year unknown
                doh_dfs.append(df)
                print(f"DOH: Loaded {os.path.basename(filepath)} ({len(df)} rows)")
            elif ext in EXCEL_EXTENSIONS:
                sheets = load_excel(filepath)
                xl = pd.ExcelFile(filepath)
                for i, sheet_name in enumerate(xl.sheet_names):
                    # Use sheet if it matches a year, or use all sheets
                    if sheet_name in doh_sheets_years or not doh_sheets_years:
                        df = sheets[i]
                        df = clean_doh_dataframe(df)
                        try:
                            df["Year"] = int(sheet_name) if sheet_name.isdigit() else i
                        except (ValueError, TypeError):
                            df["Year"] = sheet_name
                        doh_dfs.append(df)
                        print(f"DOH: Loaded {os.path.basename(filepath)} sheet '{sheet_name}' ({len(df)} rows)")
        except Exception as e:
            print(f"DOH: Error loading {filepath}: {e}")
    if doh_dfs:
        doh_merged = pd.concat(doh_dfs, ignore_index=True)
        doh_merged = handle_missing_values_doh(doh_merged)
        doh_output_path = os.path.join(output_dir, doh_output_filename)
        doh_merged.to_csv(doh_output_path, index=False)
        print(f"DOH: Merged {len(doh_merged)} records. Saved to {doh_output_path}")
    else:
        print("DOH: No valid data loaded from files.")
else:
    print(f"DOH: Folder empty or no CSV/Excel files in {doh_input_dir}. Skipping.")


# =============================================================================
# Data Preparation - PhilGEPS Medical Procurement Filter
# For: Supplier/Awardee Clustering (A), Medicine Procurement Pattern (B)
# =============================================================================
philgeps_input_dir = os.path.join(project_root, "raw_datasets", "PhilGEPS")
philgeps_output_filename = "philgeps_2025_medical_procurement.csv"

medical_keywords = [
    "medical", "medicine", "pharmaceutical", "drug", "vaccine",
    "hospital", "laboratory", "diagnostic", "surgical",
    "clinic", "health", "therapeutic", "antibiotic",
    "syringe", "test kit", "reagent", "biomedical"
]
pattern = "|".join(medical_keywords)

philgeps_files = get_supported_files(philgeps_input_dir)
if philgeps_files:
    dfs = []
    for filepath, ext in philgeps_files:
        try:
            if ext in CSV_EXTENSIONS:
                df = load_csv(filepath)
                dfs.append(df)
                print(f"PhilGEPS: Loaded {os.path.basename(filepath)} ({len(df)} rows)")
            elif ext in EXCEL_EXTENSIONS:
                sheets = load_excel(filepath, dtype=None)  # Infer types for PhilGEPS
                for df in sheets:
                    dfs.append(df)
                print(f"PhilGEPS: Loaded {os.path.basename(filepath)} ({sum(len(s) for s in sheets)} rows from {len(sheets)} sheet(s))")
        except Exception as e:
            print(f"PhilGEPS: Error loading {filepath}: {e}")
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        if "UNSPSC Description" in df.columns:
            medical_df = df[df["UNSPSC Description"].str.contains(pattern, case=False, na=False)]
            medical_df = handle_missing_values_philgeps(medical_df)
            output_file = os.path.join(output_dir, philgeps_output_filename)
            medical_df.to_csv(output_file, index=False)
            print(f"PhilGEPS: Filtered {len(medical_df)} medical records. Saved to {output_file}")
        else:
            print("PhilGEPS: 'UNSPSC Description' column not found. Saving unfiltered.")
            df = handle_missing_values_philgeps(df)
            output_file = os.path.join(output_dir, philgeps_output_filename.replace(".csv", "_unfiltered.csv"))
            df.to_csv(output_file, index=False)
    else:
        print("PhilGEPS: No valid data loaded from files.")
else:
    print(f"PhilGEPS: Folder empty or no CSV/Excel files in {philgeps_input_dir}. Skipping.")

print("Data preparation complete.")

import pandas as pd
import numpy as np
import re
from pathlib import Path

# =========================================================
# 1. FILE PATHS
# =========================================================

# Project root (parent of CRISP-DM)
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_datasets"
OUTPUT_DIR = BASE_DIR / "new_data"

distribution_file = RAW_DATA_DIR / "DISTRIBUTION REPORT 2022-205 (FOI REQUEST).xlsx"
delivery_file = RAW_DATA_DIR / "FOI_DELIVERY DATA.xlsx"

# Output files (combined 2022-2024 at root)
dist_merged_file = OUTPUT_DIR / "distribution_2022_2024.csv"
delivery_merged_file = OUTPUT_DIR / "delivery_2022_2024.csv"
output_file = OUTPUT_DIR / "MEDFLOW_MASTER_DATASET.csv"
audit_file = OUTPUT_DIR / "MERGE_AUDIT_REPORT.csv"

# Per-year folders: 2020, 2021, 2022, 2023, 2024, 2025
YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]
DIST_SHEETS = ["2022", "2023", "2024", "2025"]  # distribution has no 2020, 2021
DELIVERY_SHEETS = ["2020", "2021", "2022", "2023", "2024", "2025"]


def load_distribution_sheet(sheet_name, header=0):
    """Load single distribution sheet, adding source_year."""
    df = pd.read_excel(distribution_file, sheet_name=sheet_name, header=header)
    df["source_year"] = int(sheet_name)
    return df


def load_delivery_sheet(sheet_name, header=1):
    """Load single delivery sheet (header=1 for title row), adding source_year."""
    df = pd.read_excel(delivery_file, sheet_name=sheet_name, header=header)
    df["source_year"] = int(sheet_name)
    return df


def load_distribution_sheets(sheet_names, header=0):
    """Load and concatenate distribution sheets, adding source_year column."""
    dfs = []
    for sheet in sheet_names:
        try:
            df = pd.read_excel(distribution_file, sheet_name=sheet, header=header)
            df["source_year"] = int(sheet)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not load distribution sheet '{sheet}': {e}")
    if not dfs:
        raise ValueError("No distribution sheets could be loaded")
    return pd.concat(dfs, ignore_index=True)


def load_delivery_sheets(sheet_names, header=1):
    """Load and concatenate delivery sheets (header=1 for title row), adding source_year."""
    dfs = []
    for sheet in sheet_names:
        try:
            df = pd.read_excel(delivery_file, sheet_name=sheet, header=header)
            df["source_year"] = int(sheet)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not load delivery sheet '{sheet}': {e}")
    if not dfs:
        raise ValueError("No delivery sheets could be loaded")
    return pd.concat(dfs, ignore_index=True)


# =========================================================
# 2. LOAD COMBINED 2022-2024 (FOR MAIN TRAINING DATASET)
# =========================================================

DIST_SHEETS_EXCL_2025 = ["2022", "2023", "2024"]
DELIVERY_SHEETS_EXCL_2025 = ["2020", "2021", "2022", "2023", "2024"]

print("Loading distribution (2022-2024)...")
distribution = load_distribution_sheets(DIST_SHEETS_EXCL_2025)
print("Loading delivery (2020-2024)...")
delivery = load_delivery_sheets(DELIVERY_SHEETS_EXCL_2025)
print("Distribution (excl. 2025) rows:", len(distribution))
print("Delivery (excl. 2025) rows:", len(delivery))


# =========================================================
# 3. SAVE COMBINED RAW FILES AT ROOT
# =========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

distribution.to_csv(dist_merged_file, index=False)
print("Saved:", dist_merged_file)
delivery.to_csv(delivery_merged_file, index=False)
print("Saved:", delivery_merged_file)


# =========================================================
# 4. CLEAN COLUMN NAMES
# =========================================================

distribution.columns = distribution.columns.str.strip()
delivery.columns = delivery.columns.str.strip()


# =========================================================
# 5. DEFINE PO CLEANING FUNCTIONS
# =========================================================

def clean_po(po):
    if pd.isna(po):
        return np.nan
    po = str(po).upper().strip()
    po = re.sub(r"\s+", " ", po)
    return po


def normalize_po(po):
    if pd.isna(po):
        return np.nan
    po = str(po).upper().strip()
    po = re.sub(r"-\d+(ST|ND|RD|TH)", "", po)
    po = re.sub(r"#\d+", "", po)
    po = re.sub(r"\(\d+(ST|ND|RD|TH).*?\)", "", po)
    po = po.strip()
    return po


# =========================================================
# 6. MERGE DISTRIBUTION + DELIVERY (PO MATCHING)
# =========================================================

def merge_distribution_delivery(dist_df, deliv_df):
    """Merge distribution with delivery using exact and base PO matching."""
    dist = dist_df.copy()
    deliv = deliv_df.copy()

    if len(dist) == 0:
        return pd.DataFrame()

    dist["po_clean"] = dist["PURCHASE ORDER /CONTRACT NO./SOURCE"].apply(clean_po)
    deliv["po_clean"] = deliv["P.O NUMBER/CONTRACT"].apply(clean_po)
    dist["po_base"] = dist["po_clean"].apply(normalize_po)
    deliv["po_base"] = deliv["po_clean"].apply(normalize_po)

    merged_exact = dist.merge(
        deliv,
        how="left",
        left_on="po_clean",
        right_on="po_clean",
        suffixes=("", "_delivery"),
    )
    merged_exact["match_type"] = np.where(merged_exact["SUPPLIER"].notna(), "exact", "none")

    unmatched = merged_exact[merged_exact["match_type"] == "none"].copy()
    cols_to_drop = [
        c for c in unmatched.columns
        if (c in deliv.columns or c.endswith("_delivery")) and c != "po_base"
    ]
    unmatched_for_base = unmatched.drop(columns=cols_to_drop, errors="ignore")

    base_merge = unmatched_for_base.merge(
        deliv, how="left", left_on="po_base", right_on="po_base", suffixes=("", "_delivery"),
    )
    base_merge["match_type"] = np.where(base_merge["SUPPLIER"].notna(), "base_po", "N/A")

    matched_exact = merged_exact[merged_exact["match_type"] == "exact"]
    final = pd.concat([matched_exact, base_merge], ignore_index=True)

    for col in ["SUPPLIER", "TRANCHE", "P.O QUANTITY", "LATEST DELIVERY DATE", "REMARKS"]:
        if col in final.columns:
            final[col] = final[col].fillna("N/A")

    return final


# =========================================================
# 7. MERGE COMBINED 2022-2024 DATA (AT ROOT)
# =========================================================

print("Merging distribution + delivery (2022-2024 combined)...")
final_dataset = merge_distribution_delivery(distribution, delivery)
final_dataset.to_csv(output_file, index=False)
print("Saved:", output_file)

audit = final_dataset["match_type"].value_counts().reset_index()
audit.columns = ["match_type", "count"]
audit.to_csv(audit_file, index=False)
print("Saved:", audit_file)


# =========================================================
# 8. MERGE EACH YEAR SEPARATELY (IN YEAR FOLDERS)
# =========================================================

for year in YEARS:
    year_dir = OUTPUT_DIR / year
    year_dir.mkdir(parents=True, exist_ok=True)

    dist_year = pd.DataFrame()
    deliv_year = pd.DataFrame()

    if year in DIST_SHEETS:
        try:
            dist_year = load_distribution_sheet(year)
        except Exception as e:
            print(f"  Warning: No distribution for {year}: {e}")
    if year in DELIVERY_SHEETS:
        try:
            deliv_year = load_delivery_sheet(year)
        except Exception as e:
            print(f"  Warning: No delivery for {year}: {e}")

    if len(dist_year) > 0:
        dist_year.columns = dist_year.columns.str.strip()
    if len(deliv_year) > 0:
        deliv_year.columns = deliv_year.columns.str.strip()

    merged = merge_distribution_delivery(dist_year, deliv_year)

    out_csv = year_dir / "MEDFLOW_MASTER_DATASET.csv"
    merged.to_csv(out_csv, index=False)
    print(f"Saved {year}:", out_csv)

    audit_year = merged["match_type"].value_counts().reset_index() if len(merged) > 0 and "match_type" in merged.columns else pd.DataFrame(columns=["match_type", "count"])
    audit_year.columns = ["match_type", "count"]
    audit_year.to_csv(year_dir / "MERGE_AUDIT_REPORT.csv", index=False)

"""
3. Exploratory Data Analysis - PhilGEPS (CRISP-DM)

Runs EDA on PhilGEPS cleaned and transformed data.
Inputs: this_datasets/01_data_cleaning/philgeps_*, this_datasets/02_data_transformation/clustering_A_*, clustering_B_*
Outputs: webp/EDA_and_visualization/03_exploratory_data_analysis/PhilGEPS/
"""

from __future__ import annotations

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from _common import _ensure_dir, _project_root, generate_eda_visualizations


def main() -> None:
    # --- Step 0: Setup paths ---
    root = _project_root()
    clean_dir = os.path.join(root, "this_datasets", "01_data_cleaning")
    trans_dir = os.path.join(root, "this_datasets", "02_data_transformation")
    out_dir = os.path.join(root, "webp", "EDA_and_visualization", "03_exploratory_data_analysis", "PhilGEPS")
    _ensure_dir(out_dir)

    import pandas as pd

    # --- Step 1: EDA on cleaned PhilGEPS data ---
    phil_clean = os.path.join(clean_dir, "philgeps_2025_medical_procurement.csv")
    if os.path.exists(phil_clean) and os.path.getsize(phil_clean) > 0:
        df = pd.read_csv(phil_clean, low_memory=False)
        if not df.empty:
            generate_eda_visualizations(
                df,
                out_dir=os.path.join(out_dir, "01_cleaned"),
                dataset_label="PhilGEPS",
                stage_label="Cleaned (post data cleaning)",
                numeric_focus=["Contract Amount", "Item Budget", "Quantity", "Approved Budget of the Contract"],
                include_correlation=True, data_source="PhilGEPS",
            )
            print(f"PhilGEPS EDA: Cleaned data -> {out_dir}/01_cleaned")

    # --- Step 2: EDA on A and B transformed features ---
    for name, path in [
        ("A_supplier", "clustering_A_supplier_awardee_features.csv"),
        ("A_minmax", "clustering_A_supplier_awardee_features_minmax.csv"),
        ("B_procurement", "clustering_B_medicine_procurement_pattern_features.csv"),
        ("B_minmax", "clustering_B_medicine_procurement_pattern_features_minmax.csv"),
    ]:
        fp = os.path.join(trans_dir, path)
        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            df = pd.read_csv(fp, low_memory=False)
            if not df.empty:
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:12]
                generate_eda_visualizations(
                    df,
                    out_dir=os.path.join(out_dir, f"02_{name}"),
                    dataset_label="PhilGEPS",
                    stage_label=f"Transformed ({name})",
                    numeric_focus=num_cols,
                    include_correlation=True, data_source="PhilGEPS",
                )
                print(f"PhilGEPS EDA: {name} -> {out_dir}/02_{name}")

    print("PhilGEPS EDA complete.")


if __name__ == "__main__":
    main()

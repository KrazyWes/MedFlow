"""
3. Exploratory Data Analysis - DOH (CRISP-DM)

Runs EDA on DOH cleaned and transformed data.
Inputs: this_datasets/01_data_cleaning/doh_*, this_datasets/02_data_transformation/clustering_C_*
Outputs: webp/EDA_and_visualization/03_exploratory_data_analysis/DOH/
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
    out_dir = os.path.join(root, "webp", "EDA_and_visualization", "03_exploratory_data_analysis", "DOH")
    _ensure_dir(out_dir)

    doh_clean = os.path.join(clean_dir, "doh_medicine_distribution_2022_2025.csv")
    c_features = os.path.join(trans_dir, "clustering_C_distribution_recipient_features.csv")
    c_minmax = os.path.join(trans_dir, "clustering_C_distribution_recipient_features_minmax.csv")

    import pandas as pd

    if os.path.exists(doh_clean) and os.path.getsize(doh_clean) > 0:
        df = pd.read_csv(doh_clean, low_memory=False)
        if not df.empty:
            generate_eda_visualizations(
                df,
                out_dir=os.path.join(out_dir, "01_cleaned"),
                dataset_label="DOH",
                stage_label="Cleaned (post data cleaning)",
                numeric_focus=["UNIT COST", "TOTAL AMOUNT", "QUANTITY", "Year"],
                include_correlation=True, data_source="DOH",
            )
            print(f"DOH EDA: Cleaned data -> {out_dir}/01_cleaned")

    # --- Step 2: EDA on C transformed features ---
    if os.path.exists(c_features) and os.path.getsize(c_features) > 0:
        df = pd.read_csv(c_features, low_memory=False)
        if not df.empty:
            num_cols = ["medicines_received_lines", "medicines_received_unique", "quantity_total", "total_amount_total", "delivery_frequency_unique_dates"]
            num_cols = [c for c in num_cols if c in df.columns]
            generate_eda_visualizations(
                df,
                out_dir=os.path.join(out_dir, "02_transformed_features"),
                dataset_label="DOH",
                stage_label="Transformed (C distribution features)",
                numeric_focus=num_cols,
                include_correlation=True, data_source="DOH",
            )
            print(f"DOH EDA: Transformed features -> {out_dir}/02_transformed_features")

    # --- Step 3: EDA on C minmax-scaled (clustering-ready) ---
    if os.path.exists(c_minmax) and os.path.getsize(c_minmax) > 0:
        df = pd.read_csv(c_minmax, low_memory=False)
        if not df.empty:
            num_cols = [c for c in df.columns if df[c].dtype in ("int64", "float64")]
            generate_eda_visualizations(
                df,
                out_dir=os.path.join(out_dir, "03_scaled_minmax"),
                dataset_label="DOH",
                stage_label="Scaled (minmax for clustering)",
                numeric_focus=num_cols[:12],
                include_correlation=True, data_source="DOH",
            )
            print(f"DOH EDA: Scaled minmax -> {out_dir}/03_scaled_minmax")

    print("DOH EDA complete.")


if __name__ == "__main__":
    main()

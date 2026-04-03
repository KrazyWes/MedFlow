"""
Step 03 — PhilGEPS exploratory pass.

Rebuilds summary plots from the cleaned procurement CSV and early transformed outputs.
Read-only with respect to this_datasets/; writes PNG/WebP under
webp/EDA_and_visualization/PhilGEPS/03_exploratory_data_analysis/.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

# Shared `_common`, `sources_paths`, etc. live in parent `CRISP-DM/`.
script_dir = os.path.dirname(os.path.abspath(__file__))
_crisp_dm_root = os.path.dirname(script_dir)
if _crisp_dm_root not in sys.path:
    sys.path.insert(0, _crisp_dm_root)

from _common import _ensure_dir, _project_root, generate_eda_visualizations
from log_tee import tee_stdio_to_file
from sources_paths import data_root_philgeps, logs_dir_philgeps, webp_root_philgeps


def main() -> None:
    root = _project_root()  # repo root via _common._project_root
    term_log = os.path.join(logs_dir_philgeps(), "03_exploratory_data_analysis_philgeps_terminal.txt")
    with tee_stdio_to_file(term_log):
        _run(root)


def _run(root: str) -> None:
    clean_dir = os.path.join(data_root_philgeps(), "01_data_cleaning")
    trans_dir = os.path.join(data_root_philgeps(), "02_data_transformation")
    out_dir = os.path.join(webp_root_philgeps(), "03_exploratory_data_analysis")
    _ensure_dir(out_dir)

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
                include_correlation=True,
                data_source="PhilGEPS",
            )
            print(f"PhilGEPS EDA: Cleaned -> {out_dir}/01_cleaned")

    extra_files = [
        ("PhilGEPS_A_features", "clustering_PhilGEPS_A_supplier_awardee_features.csv"),
        ("PhilGEPS_A_minmax", "clustering_PhilGEPS_A_supplier_awardee_features_minmax.csv"),
        ("PhilGEPS_B_features", "clustering_PhilGEPS_B_medicine_procurement_pattern_features.csv"),
        ("PhilGEPS_B_minmax", "clustering_PhilGEPS_B_medicine_procurement_pattern_features_minmax.csv"),
        ("PhilGEPS_C_minmax", "clustering_PhilGEPS_C_distribution_recipient_features_minmax.csv"),
    ]
    for name, path in extra_files:
        fp = os.path.join(trans_dir, path)
        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            df = pd.read_csv(fp, low_memory=False)
            if not df.empty:
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:12]
                generate_eda_visualizations(
                    df,
                    out_dir=os.path.join(out_dir, f"02_{name}"),
                    dataset_label="PhilGEPS",
                    stage_label=f"Features ({name})",
                    numeric_focus=num_cols,
                    include_correlation=True,
                    data_source="PhilGEPS",
                )
                print(f"PhilGEPS EDA: {name} -> {out_dir}/02_{name}")

    print("PhilGEPS EDA complete.")


if __name__ == "__main__":
    main()

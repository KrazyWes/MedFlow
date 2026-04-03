"""
Step 03 — DOH exploratory pass.

Re-scans the cleaned delivery file and the transformed feature exports (at least lens A) to
regenerate correlation / distribution plots after cleaning + engineering settled. Does not
mutate CSVs; only adds figures under webp/.../DOH/03_exploratory_data_analysis/.

Safe to rerun any time; it is slower than step 02 because it recomputes summaries from disk.
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
from sources_paths import data_root_doh, logs_dir_doh, webp_root_doh


def main() -> None:
    # _project_root comes from _common (repo root); data paths use sources_paths helpers inside _run.
    root = _project_root()
    term_log = os.path.join(logs_dir_doh(), "03_exploratory_data_analysis_doh_terminal.txt")
    with tee_stdio_to_file(term_log):
        _run(root)


def _run(root: str) -> None:
    clean_dir = os.path.join(data_root_doh(), "01_data_cleaning")
    trans_dir = os.path.join(data_root_doh(), "02_data_transformation")
    out_dir = os.path.join(webp_root_doh(), "03_exploratory_data_analysis")
    _ensure_dir(out_dir)

    doh_clean = os.path.join(clean_dir, "doh_medicine_distribution_2022_2025.csv")
    a_features = os.path.join(trans_dir, "clustering_DOH_A_distribution_recipient_features.csv")
    a_minmax = os.path.join(trans_dir, "clustering_DOH_A_distribution_recipient_features_minmax.csv")

    if os.path.exists(doh_clean) and os.path.getsize(doh_clean) > 0:
        df = pd.read_csv(doh_clean, low_memory=False)
        if not df.empty:
            generate_eda_visualizations(
                df,
                out_dir=os.path.join(out_dir, "01_cleaned"),
                dataset_label="DOH",
                stage_label="Cleaned (post data cleaning)",
                numeric_focus=["UNIT COST", "TOTAL AMOUNT", "QUANTITY", "Year"],
                include_correlation=True,
                data_source="DOH",
            )
            print(f"DOH EDA: Cleaned -> {out_dir}/01_cleaned")

    if os.path.exists(a_features) and os.path.getsize(a_features) > 0:
        df = pd.read_csv(a_features, low_memory=False)
        if not df.empty:
            num_cols = [
                "medicines_received_lines",
                "medicines_received_unique",
                "quantity_total",
                "total_amount_total",
                "delivery_frequency_unique_dates",
            ]
            num_cols = [c for c in num_cols if c in df.columns]
            generate_eda_visualizations(
                df,
                out_dir=os.path.join(out_dir, "02_transformed_A_distribution"),
                dataset_label="DOH",
                stage_label="Transformed (A — distribution recipient)",
                numeric_focus=num_cols,
                include_correlation=True,
                data_source="DOH",
            )
            print(f"DOH EDA: Transformed A -> {out_dir}/02_transformed_A_distribution")

    if os.path.exists(a_minmax) and os.path.getsize(a_minmax) > 0:
        df = pd.read_csv(a_minmax, low_memory=False)
        if not df.empty:
            num_cols = [c for c in df.columns if df[c].dtype in ("int64", "float64")]
            generate_eda_visualizations(
                df,
                out_dir=os.path.join(out_dir, "03_scaled_minmax_A"),
                dataset_label="DOH",
                stage_label="Scaled minmax (A — clustering input)",
                numeric_focus=num_cols[:12],
                include_correlation=True,
                data_source="DOH",
            )
            print(f"DOH EDA: Minmax A -> {out_dir}/03_scaled_minmax_A")

    print("DOH EDA complete.")


if __name__ == "__main__":
    main()

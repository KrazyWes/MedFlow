"""
Step 08 — DOH thesis bundle.

Thin wrapper: calls output_bundle.run_bundles_for_source("DOH") so every DOH lens (A–E) gets
paired k-means + DBSCAN folders under webp/EDA_and_visualization/DOH/08_output_to_use/.
See output_bundle.py for plot definitions and thematic scoring.
"""

from __future__ import annotations

import os
import sys

# Shared `_common`, `sources_paths`, etc. live in parent `CRISP-DM/`.
script_dir = os.path.dirname(os.path.abspath(__file__))
_crisp_dm_root = os.path.dirname(script_dir)
if _crisp_dm_root not in sys.path:
    sys.path.insert(0, _crisp_dm_root)

from log_tee import tee_stdio_to_file
from output_bundle import run_bundles_for_source
from sources_paths import logs_dir_doh


def main() -> None:
    # Single tee log; all per-dataset messages still print inside process_dataset_bundle.
    term_log = os.path.join(logs_dir_doh(), "08_final_output_bundle_doh_terminal.txt")
    with tee_stdio_to_file(term_log):
        run_bundles_for_source("DOH")


if __name__ == "__main__":
    main()

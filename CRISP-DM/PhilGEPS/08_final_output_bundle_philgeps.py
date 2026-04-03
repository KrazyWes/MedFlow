"""
Step 08 — PhilGEPS thesis bundle.

Same as the DOH wrapper but run_bundles_for_source("PhilGEPS") covers lenses A–G.
Outputs: webp/EDA_and_visualization/PhilGEPS/08_output_to_use/{kmeans|dbscan}/{slug}/.
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
from sources_paths import logs_dir_philgeps


def main() -> None:
    term_log = os.path.join(logs_dir_philgeps(), "08_final_output_bundle_philgeps_terminal.txt")
    with tee_stdio_to_file(term_log):
        run_bundles_for_source("PhilGEPS")


if __name__ == "__main__":
    main()

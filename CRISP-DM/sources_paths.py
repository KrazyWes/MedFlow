"""
Single place for filesystem roots used by the split DOH / PhilGEPS pipelines.

Convention:
  - `project_root()`  → repo root (parent of the `CRISP-DM/` folder)
  - `data_root_*()`   → `this_datasets/DOH` or `.../PhilGEPS` (cleaned CSVs, features, cluster labels)
  - `webp_root_*()`   → figures and EDA under `webp/EDA_and_visualization/`
  - `logs_dir_*()`    → step transcripts from `tee_stdio_to_file`

`_common.py` resolves the repo root from its own file path; this module does the same so both
agree. Step scripts under `DOH/` or `PhilGEPS/` import these helpers after pushing `CRISP-DM/`
onto `sys.path`.
"""

from __future__ import annotations

import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)


def project_root() -> str:
    """MedFlow repository root (contains `CRISP-DM/`, `raw_datasets/`, `webp/`, …)."""
    return _PROJECT_ROOT


def data_root_doh() -> str:
    """Processed DOH tables: cleaning, transformation, clustering, etc."""
    return os.path.join(_PROJECT_ROOT, "this_datasets", "DOH")


def data_root_philgeps() -> str:
    """Same pipeline stages as DOH, for PhilGEPS."""
    return os.path.join(_PROJECT_ROOT, "this_datasets", "PhilGEPS")


def webp_root_doh() -> str:
    """EDA PNG/WebP tree for DOH (mirrors step numbers 01–08)."""
    return os.path.join(_PROJECT_ROOT, "webp", "EDA_and_visualization", "DOH")


def webp_root_philgeps() -> str:
    """EDA tree for PhilGEPS."""
    return os.path.join(_PROJECT_ROOT, "webp", "EDA_and_visualization", "PhilGEPS")


def logs_dir_doh() -> str:
    """One `*_terminal.txt` per DOH step (stdout/stderr tee target)."""
    return os.path.join(_PROJECT_ROOT, "webp", "logs", "DOH")


def logs_dir_philgeps() -> str:
    """Terminal logs for PhilGEPS steps."""
    return os.path.join(_PROJECT_ROOT, "webp", "logs", "PhilGEPS")


def ensure_all_base_dirs() -> None:
    """Create DOH/PhilGEPS roots if missing (safe no-op if they already exist)."""
    for d in (
        data_root_doh(),
        data_root_philgeps(),
        webp_root_doh(),
        webp_root_philgeps(),
        logs_dir_doh(),
        logs_dir_philgeps(),
    ):
        os.makedirs(d, exist_ok=True)

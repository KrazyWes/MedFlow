"""
Wipe regenerated pipeline artifacts — used by `main.py --fresh` or when you want a clean rerun.

What gets removed:
  Everything *under* `webp/EDA_and_visualization/`, `webp/logs/`, `this_datasets/DOH/`,
  and `this_datasets/PhilGEPS/` (files and subfolders deleted, top folder kept empty then refilled).
  Any old flat folders `this_datasets/01_data_cleaning`, `02_data_transformation`, `04_clustering`
  are removed entirely if they still exist (leftovers from an earlier layout).

What stays: `raw_datasets/` (your inputs) and all Python/source code.

After clearing, `ensure_all_base_dirs()` recreates the empty DOH/PhilGEPS roots so the next step
does not fail on a missing directory.

Usage: `python CRISP-DM/00_clear_pipeline_outputs.py` from repo root, or `python 00_clear_pipeline_outputs.py`
from inside `CRISP-DM/` (this file adds its directory to `sys.path` for `sources_paths`).
"""

from __future__ import annotations

import os
import shutil
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from sources_paths import ensure_all_base_dirs, project_root  # noqa: E402


def _clear_directory(path: str) -> None:
    """Delete all children of `path` but leave `path` itself as an empty directory."""
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        full = os.path.join(path, name)
        try:
            if os.path.isdir(full):
                shutil.rmtree(full)
            else:
                os.unlink(full)
        except OSError as e:
            print(f"Warning: could not remove {full}: {e}")


def _remove_dir_if_exists(path: str) -> None:
    """Drop the whole directory (used for legacy flat `this_datasets/*` step folders)."""
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except OSError as e:
            print(f"Warning: could not remove {path}: {e}")


def main() -> None:
    root = project_root()
    td = os.path.join(root, "this_datasets")
    # Main trees: empty in place so `this_datasets/DOH` still exists after clearing.
    targets = [
        os.path.join(root, "webp", "EDA_and_visualization"),
        os.path.join(root, "webp", "logs"),
        os.path.join(td, "DOH"),
        os.path.join(td, "PhilGEPS"),
    ]
    for t in targets:
        print(f"Clearing: {t}")
        _clear_directory(t)
    for name in ("01_data_cleaning", "02_data_transformation", "04_clustering"):
        leg = os.path.join(td, name)
        if os.path.isdir(leg):
            print(f"Removing legacy: {leg}")
            _remove_dir_if_exists(leg)
    ensure_all_base_dirs()
    print("Pipeline output directories cleared and base folders recreated.")


if __name__ == "__main__":
    main()

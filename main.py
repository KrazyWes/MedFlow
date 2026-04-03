"""
MedFlow: thin wrapper around the CRISP-DM clustering pipeline.

What runs:
  `CRISP-DM/run_all.py` executes every step script in order: full DOH block (01→08),
  then full PhilGEPS (01→08). Individual steps live in `CRISP-DM/DOH/` and
  `CRISP-DM/PhilGEPS/`; shared code is in `CRISP-DM/_common.py`, `sources_paths.py`, etc.

Why subprocess instead of imports: each step is a standalone script with its own
`if __name__ == "__main__"` and logging setup, so the orchestrator shells out the
same way you would from a terminal.

Usage (repo root):
  python main.py           # keep existing outputs, run pipeline
  python main.py --fresh   # wipe generated dirs first (see 00_clear_pipeline_outputs.py)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="MedFlow CRISP-DM pipeline runner.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear pipeline outputs under webp/ and this_datasets/{DOH,PhilGEPS} before running.",
    )
    args = parser.parse_args()

    # Repo root = parent of this file; CRISP-DM is always a sibling folder.
    root = os.path.dirname(os.path.abspath(__file__))
    crisp_dm = os.path.join(root, "CRISP-DM")
    run_all = os.path.join(crisp_dm, "run_all.py")
    if not os.path.isfile(run_all):
        print("CRISP-DM/run_all.py not found.", file=sys.stderr)
        sys.exit(1)

    if args.fresh:
        # Clears webp EDA + logs + this_datasets/{DOH,PhilGEPS} and legacy flat folders;
        # raw_datasets/ is never touched.
        clear_script = os.path.join(crisp_dm, "00_clear_pipeline_outputs.py")
        if not os.path.isfile(clear_script):
            print("CRISP-DM/00_clear_pipeline_outputs.py not found.", file=sys.stderr)
            sys.exit(1)
        print("Running with --fresh: clearing pipeline outputs first.\n")
        code = subprocess.call([sys.executable, clear_script], cwd=crisp_dm)
        if code != 0:
            sys.exit(code)

    # cwd=crisp_dm so paths like DOH/01_*.py resolve the same as manual `cd CRISP-DM`.
    code = subprocess.call([sys.executable, run_all], cwd=crisp_dm)
    sys.exit(code)


if __name__ == "__main__":
    main()

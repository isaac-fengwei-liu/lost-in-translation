#!/usr/bin/env python3
"""Uniqueness check, M-E- rematch and Latin-square counterbalancing.

Run this after the design has been manually reviewed (the reviewed file defaults to
manual_review/uniqueness_checked3.csv).

    python counterbalance.py
    python counterbalance.py --reviewed-file uniqueness_checked4.csv
    python counterbalance.py --max-total-target-uses 3 --n-anneal-iters 500000
    python counterbalance.py --help                # every parameter

Writes uniqueness_issue4.csv and full_design.xlsx into the output directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stimuli_pipeline.cli import counterbalance_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(counterbalance_main())

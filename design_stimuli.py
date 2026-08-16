#!/usr/bin/env python3
"""Select the stimulus design from the SWOW norms.

    python design_stimuli.py                       # defaults = the original notebook
    python design_stimuli.py --dist-cutoff 1.4 --no-run-permutation-test
    python design_stimuli.py --config my_run.json --dump-config effective.json
    python design_stimuli.py --help                # every parameter

Writes candidate_pairs.csv, matched_cue_target_pairs.csv, design.csv and
qc_report.txt into the output directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stimuli_pipeline.cli import design_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(design_main())

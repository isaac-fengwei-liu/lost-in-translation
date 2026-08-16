"""Stimulus design pipeline for the bilingual semantics experiment.

Two stages, sharing one set of matrices:

1. `design`         - build the translation bridge and the four M x E cells, rank
                      every eligible cue-target pair, match on concreteness /
                      frequency / length, and select the 3-cell design.
                      (was `design_stimuli.ipynb`)
2. `counterbalance` - take the manually reviewed design, check global uniqueness,
                      rematch the M-E- cell from the M+E+ pool, and build the
                      Latin-square lists. (was `counterbalance.ipynb`)

Typical use from a notebook or REPL:

    from stimuli_pipeline import Config, build_context, select_design
    from stimuli_pipeline.counterbalance import run_counterbalance

    cfg = Config(dist_cutoff=1.5)
    ctx = build_context(cfg)              # matrices; RW files are cached on disk
    design, extras = select_design(ctx)
    result = run_counterbalance(ctx)
"""

from .config import Config
from .context import StimulusContext, build_context
from .selection import select_design
from .qc import qc_report
from .permutation import run_permutation_test, benjamini_hochberg
from .counterbalance import run_counterbalance
from .utils import log, set_verbose

__all__ = [
    "Config",
    "StimulusContext",
    "build_context",
    "select_design",
    "qc_report",
    "run_permutation_test",
    "benjamini_hochberg",
    "run_counterbalance",
    "log",
    "set_verbose",
]

__version__ = "1.0.0"

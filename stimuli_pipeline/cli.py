"""Command-line front end.

    python design_stimuli.py --help
    python counterbalance.py --help
    python -m stimuli_pipeline design --dist-cutoff 1.4

Every field of `Config` is exposed as a flag (``--min-count-en 4``,
``--no-exclude-capitalised``), and a whole configuration can be loaded from JSON
with ``--config`` and written back out with ``--dump-config``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import Config
from .utils import log, set_verbose

PATH_FIELDS = {"stim_dir", "data_dir", "out_dir", "cache_dir", "review_dir"}
_DEFAULTS = Config()


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """One flag per Config field, typed from its default value."""
    group = parser.add_argument_group("pipeline parameters")
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        default = getattr(_DEFAULTS, f.name)
        helptext = f"default: {default}"
        if f.name in PATH_FIELDS:
            group.add_argument(flag, dest=f.name, type=Path, default=None, help=helptext)
        elif isinstance(default, bool):
            group.add_argument(flag, dest=f.name, default=None,
                               action=argparse.BooleanOptionalAction, help=helptext)
        elif isinstance(default, tuple):
            group.add_argument(flag, dest=f.name, nargs="+", default=None, help=helptext)
        elif isinstance(default, int):
            group.add_argument(flag, dest=f.name, type=int, default=None, help=helptext)
        elif isinstance(default, float):
            group.add_argument(flag, dest=f.name, type=float, default=None, help=helptext)
        else:
            group.add_argument(flag, dest=f.name, type=str, default=None, help=helptext)


def config_from_args(args: argparse.Namespace) -> Config:
    """--config file first, then any explicitly given flags on top.

    Only the keys actually present in the JSON are taken, so an unset directory
    stays derived from --stim-dir instead of being frozen by the file.
    """
    base: Dict[str, Any] = (json.loads(Path(args.config).read_text(encoding="utf-8"))
                            if getattr(args, "config", None) else {})
    names = {f.name for f in fields(Config)}
    overrides = {k: v for k, v in vars(args).items() if k in names and v is not None}
    base.update(overrides)
    cfg = Config.from_dict(base)
    set_verbose(cfg.verbose)
    if getattr(args, "dump_config", None):
        cfg.to_json(args.dump_config)
        log(f"wrote effective configuration to {args.dump_config}")
    return cfg


def _base_parser(prog: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=Path, default=None,
                   help="JSON file with any subset of the pipeline parameters")
    p.add_argument("--dump-config", type=Path, default=None,
                   help="write the effective configuration to this JSON file and continue")
    add_config_arguments(p)
    return p


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def run_design(cfg: Config):
    """Load norms -> build bridge -> four conditions -> select design -> QC -> permutation."""
    from .context import build_context
    from .permutation import run_permutation_test
    from .qc import qc_report
    from .selection import select_design

    ctx = build_context(cfg)
    design, extras = select_design(ctx)
    qc_report(design, cfg, clique=extras.get("clique"))
    if cfg.run_permutation_test:
        design = run_permutation_test(design, ctx)
    return ctx, design, extras


def run_counterbalance_stage(cfg: Config, ctx=None):
    from .context import build_context
    from .counterbalance import run_counterbalance

    if ctx is None:
        ctx = build_context(cfg)
    return ctx, run_counterbalance(ctx)


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def design_main(argv: Optional[Sequence[str]] = None) -> int:
    p = _base_parser("design_stimuli.py",
                     "Build the candidate pairs and select the 3-cell design "
                     "(design_stimuli.ipynb as a script).")
    args = p.parse_args(argv)
    cfg = config_from_args(args)
    log(f"outputs will go to {cfg.out_dir}")
    run_design(cfg)
    return 0


def counterbalance_main(argv: Optional[Sequence[str]] = None) -> int:
    p = _base_parser("counterbalance.py",
                     "Uniqueness check, M-E- rematch and Latin-square lists "
                     "(counterbalance.ipynb as a script).")
    args = p.parse_args(argv)
    cfg = config_from_args(args)
    log(f"outputs will go to {cfg.out_dir}")
    # the counterbalancing stage only needs the matrices, not the permutation test
    run_counterbalance_stage(cfg)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """`python -m stimuli_pipeline <design|counterbalance|all>`."""
    p = argparse.ArgumentParser(prog="stimuli_pipeline",
                                description="Bilingual semantics stimulus pipeline")
    sub = p.add_subparsers(dest="stage", required=True)
    for name, helptext in (("design", "select the 3-cell design"),
                           ("counterbalance", "rematch M-E- and build the lists"),
                           ("all", "design, then counterbalance, reusing the matrices")):
        sp = sub.add_parser(name, help=helptext,
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        sp.add_argument("--config", type=Path, default=None)
        sp.add_argument("--dump-config", type=Path, default=None)
        add_config_arguments(sp)

    args = p.parse_args(argv)
    cfg = config_from_args(args)
    log(f"outputs will go to {cfg.out_dir}")
    if args.stage == "design":
        run_design(cfg)
    elif args.stage == "counterbalance":
        run_counterbalance_stage(cfg)
    else:
        ctx, _design, _extras = run_design(cfg)
        run_counterbalance_stage(cfg, ctx=ctx)
    return 0

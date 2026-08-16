"""Turning the association matrices into the four M x E design cells."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from .config import Config
from .utils import log


def calibrate_ceiling(rw: np.ndarray, related: np.ndarray, overlap: float) -> float:
    """RW value below which only `overlap` of related pairs would fall."""
    vals = rw[related & np.isfinite(rw)]
    return float(np.percentile(vals, 100 * overlap))


def build_ok_mask(nodes: Sequence[str], shared_translation: np.ndarray,
                  cfg: Config) -> np.ndarray:
    """Pairs that are admissible before any relatedness test.

    Drops the diagonal, shared-translation pairs, and morphological near-duplicates
    (run / running, book / booking). Used both when filtering cues and when
    rematching the M-E- cell, so the two stages cannot drift apart.
    """
    n = len(nodes)
    ok = np.ones((n, n), dtype=bool)
    np.fill_diagonal(ok, False)
    if cfg.exclude_shared_translation:
        ok &= ~shared_translation

    pref: Dict[str, list] = {}
    for i, w in enumerate(nodes):
        pref.setdefault(w[:cfg.min_edit_distinctness].lower(), []).append(i)
    for group in pref.values():
        if len(group) > 1:
            g = np.array(group)
            ok[np.ix_(g, g)] = False
    return ok


def build_cells(rel_en, unrel_en, rel_zh2en, unrel_zh2en, cfg: Config) -> Dict[str, np.ndarray]:
    cells = {
        "M+E+": rel_zh2en & rel_en,
        "M+E-": rel_zh2en & unrel_en,
        "M-E+": unrel_zh2en & rel_en,
        "M-E-": unrel_zh2en & unrel_en,
    }
    for c, m_ in cells.items():
        log(f"  {c}: {m_.sum():,} candidate pairs")
    return {c: cells[c] for c in cfg.conditions}


def filter_cues(nodes: Sequence[str], cells: Dict[str, np.ndarray], ok: np.ndarray,
                cfg: Config) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    """Which cues can fill every condition with at least `n_per_cell` targets."""
    avail = {c: (cells[c] & ok) for c in cfg.conditions}      # whether a pair is eligible
    counts = {c: avail[c].sum(axis=1) for c in cfg.conditions}

    worst = np.min(np.stack([counts[c] for c in cfg.conditions]), axis=0)
    feasible = np.where(worst >= cfg.n_per_cell)[0]
    log(f"  {len(feasible):,} cues can fill the constrained cells")
    for c in cfg.conditions:
        log(f"    {c}: median {np.median(counts[c]):.0f} targets/cue available")

    return avail, counts, feasible

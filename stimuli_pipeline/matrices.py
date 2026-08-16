"""Dense monolingual association matrices (strength / count) and row cosines."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .utils import log


def strength_matrices(d: pd.DataFrame, nodes: Sequence[str], lang: str,
                      cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Dense directed (strength, count) matrices over `nodes`."""
    idx = {w: i for i, w in enumerate(nodes)}

    if lang == "en" and cfg.renormalise_en_to_cue_support:
        # Put English on the same footing as Chinese: probabilities over cue-words only.
        all_cues = set(d["cue"])
        on_support = d["response"].isin(all_cues)
        mass = d.loc[on_support].groupby("cue")["strength"].sum()
        d = d.loc[on_support].copy()
        d["strength"] = d["strength"] / d["cue"].map(mass)
        log(f"  {lang}: renormalised onto cue-support (median retained mass "
            f"{mass.median():.3f})")

    sub = d[d["cue"].isin(idx) & d["response"].isin(idx)]
    m = len(nodes)
    S = np.zeros((m, m), dtype=np.float32)
    C = np.zeros((m, m), dtype=np.float32)
    i = sub["cue"].map(idx).to_numpy(np.int32)
    j = sub["response"].map(idx).to_numpy(np.int32)
    S[i, j] = sub["strength"].to_numpy(np.float32)
    C[i, j] = sub["count"].to_numpy(np.float32)
    log(f"  {lang}: {len(sub):,} directed pairs inside the lexicon "
        f"({len(sub)/(m*m):.2%} dense)")
    return S, C


def cosine_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two matrices of the same shape."""
    A = np.nan_to_num(A); B = np.nan_to_num(B)
    num = (A * B).sum(axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)

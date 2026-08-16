"""The bipartite English-Chinese translation bridge and the route algebra built on it."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .utils import log


def is_usable_word(w, cfg: Config) -> bool:
    """Single English word, sensible length, not a proper noun."""
    if not isinstance(w, str) or not cfg.word_re.match(w):
        return False
    if not (cfg.min_len <= len(w) <= cfg.max_len):
        return False
    return not (cfg.exclude_capitalised and w[0].isupper())


def build_translation_map(vocab_en: set, vocab_zh: set, cfg: Config) -> pd.DataFrame:
    align = pd.read_excel(cfg.path_align)
    align.columns = ["en", "zh", "rcs", "shared"]
    n0 = len(align)

    # exclude negative rcs (dissimilar translations)
    align = align[align["rcs"] >= 0].copy()

    # translation edge weight that consolidates the ranks in both directions
    # P calculated before filtering -- highly aligned translations may be OOV
    align["p_z_e"] = align["rcs"] / align.groupby("en")["rcs"].transform("sum")  # p(zh | en)
    align["p_e_z"] = align["rcs"] / align.groupby("zh")["rcs"].transform("sum")  # p(en | zh)
    align["w"] = np.sqrt(align["p_z_e"] * align["p_e_z"])

    # keep only pairs where BOTH words are usable cues, and the translation is confident
    align = align[align["en"].isin(vocab_en) & align["zh"].isin(vocab_zh)]
    align = align[align["rcs"] >= cfg.align_word_floor]
    align = align[align["en"].map(lambda w: is_usable_word(w, cfg))].copy()
    log(f"  {n0:,} candidate translations -> {len(align):,} usable "
        f"({align['en'].nunique():,} English, {align['zh'].nunique():,} Chinese)")

    # candidate translations must be among the best in both directions
    # --- Chinese -> English: keep top K English forms ---
    align = (align.sort_values("w", ascending=False)
             .groupby("zh", sort=False).head(cfg.max_translations_per_word).copy())
    align["w"] = align["w"] / align.groupby("zh")["w"].transform("sum")

    # --- English -> Chinese: keep top K Chinese forms ---
    align = (align.sort_values("w", ascending=False)
             .groupby("en", sort=False).head(cfg.max_translations_per_word).copy())
    align["w"] = align["w"] / align.groupby("en")["w"].transform("sum")

    per_en = align.groupby("en").size()
    log(f"  bridge: {len(align):,} links over {align['en'].nunique():,} English words, "
        f"{per_en.mean():.1f} routes per word")
    return align.sort_values(["en", "w"], ascending=[True, False]).reset_index(drop=True)


def build_route_table(edges: pd.DataFrame, nodes_en: Sequence[str], nodes_zh: Sequence[str],
                      cfg: Config) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
    """TIDX[i, k] = index of the k-th Chinese route of English word i (-1 = no route).
    TW[i, k]     = that route's weight.
    """
    K = cfg.max_translations_per_word
    n_en = len(nodes_en)
    zpos = {z: i for i, z in enumerate(nodes_zh)}
    wpos = {w: i for i, w in enumerate(nodes_en)}

    TIDX = np.full((n_en, K), -1, dtype=np.int32)
    TW = np.zeros((n_en, K), dtype=np.float32)
    for en_w, grp in edges.groupby("en", sort=False):
        i = wpos[en_w]
        for k, (z, wt) in enumerate(zip(grp["zh"], grp["w"])):
            if k >= K:
                break
            TIDX[i, k] = zpos[z]
            TW[i, k] = wt
    log(f"route table: {(TIDX >= 0).sum():,} routes for {n_en:,} English words")
    return TIDX, TW, wpos, zpos


def aggregate_routes(ZZ_assoc: np.ndarray, TIDX: np.ndarray, TW: np.ndarray) -> np.ndarray:
    """result[i, j] = max over routes a of i, b of j, of ZZ_assoc[a, b]. Continuous
    values are weighted by the translation edges in both cue and target directions
    before finding the max; Boolean values are not.

    P(cue_zh|target_zh) * W(cue_en,cue_zh) * W(target_en,target_zh)

    Done as two separable passes. np.fmax is used for float matrices so that a NaN
    on the RW diagonal is ignored rather than poisoning the result. Self-pair routes
    need no special handling: the norms contain no self-pairs, so ZZ_assoc[a, a] is
    0 or NaN and cannot win a maximum.
    """
    rows, K_ = TIDX.shape
    is_float = ZZ_assoc.dtype.kind == "f"
    mx = np.fmax if is_float else np.maximum
    init = np.nan if is_float else 0

    EZ_assoc = np.full((rows, ZZ_assoc.shape[1]), init, dtype=ZZ_assoc.dtype)   # collapse over cue routes
    for k in range(K_):
        idx = TIDX[:, k]; ok = idx >= 0
        weights = TW[:, k]
        if ok.any():
            if is_float:
                # element-wise multiplication with route probability
                a_weighted = ZZ_assoc[idx[ok], :] * weights[ok].reshape(-1, 1)
            else:
                a_weighted = ZZ_assoc[idx[ok], :]      # a route is either related or not
            EZ_assoc[ok] = mx(EZ_assoc[ok], a_weighted)

    EE_assoc = np.full((rows, rows), init, dtype=ZZ_assoc.dtype)          # collapse over target routes
    for k in range(K_):
        idx = TIDX[:, k]; ok = idx >= 0
        weights = TW[:, k]
        if ok.any():
            if is_float:
                a_weighted = EZ_assoc[:, idx[ok]] * weights[ok]   # EZ_assoc columns are ZH
            else:
                a_weighted = EZ_assoc[:, idx[ok]]
            EE_assoc[:, ok] = mx(EE_assoc[:, ok], a_weighted)
    return EE_assoc


def shared_translation_matrix(TIDX: np.ndarray, TW: np.ndarray, n_en: int, n_zh: int,
                              cfg: Config) -> np.ndarray:
    """True where two English words can map onto the same Chinese form.

    Cue and target sharing a Chinese form are translation-equivalent: the Mandarin
    "pair" would be a word with itself, so they cannot honestly be called unrelated.
    """
    Wmat = np.zeros((n_en, n_zh), dtype=np.float32)
    for k in range(cfg.max_translations_per_word):
        idx = TIDX[:, k]; ok = idx >= 0
        Wmat[np.where(ok)[0], idx[ok]] += TW[ok, k]
    return ((Wmat > 0).astype(np.float32) @ (Wmat > 0).astype(np.float32).T) > 0

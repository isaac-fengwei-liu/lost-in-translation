"""Builds everything the design and the counterbalancing both need.

`build_context(cfg)` reproduces sections *Initialize* through *Four Conditions* of
`design_stimuli.ipynb` and hands back a single object holding the lexicons, the
translation bridge, the monolingual and translated matrices, and the four cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import bridge as _bridge
from . import conditions as _cond
from . import data_io
from .config import Config
from .matrices import strength_matrices
from .utils import log, set_verbose


@dataclass
class StimulusContext:
    cfg: Config

    # lexicons and bridge
    edges: pd.DataFrame = None
    nodes_en: List[str] = field(default_factory=list)
    nodes_zh: List[str] = field(default_factory=list)
    wpos: Dict[str, int] = field(default_factory=dict)
    zpos: Dict[str, int] = field(default_factory=dict)
    TIDX: np.ndarray = None
    TW: np.ndarray = None

    # raw association tables
    str_en: pd.DataFrame = None
    str_zh: pd.DataFrame = None

    # monolingual matrices
    s_en: np.ndarray = None
    c_en: np.ndarray = None
    rw_en: np.ndarray = None
    s_zh: np.ndarray = None
    c_zh: np.ndarray = None
    rw_zh: np.ndarray = None

    # Chinese matrices mapped into English space
    rel_zh: np.ndarray = None
    produced_zh: np.ndarray = None
    rel_zh_mapped: np.ndarray = None
    produced_zh2en: np.ndarray = None
    rw_zh2en: np.ndarray = None
    s_zh2en: np.ndarray = None
    c_zh2en: np.ndarray = None
    rel_zh2en: np.ndarray = None
    median_mapped_strength: float = float("nan")
    shared_translation: np.ndarray = None

    # relatedness / unrelatedness
    rel_en: np.ndarray = None
    unrel_en: np.ndarray = None
    unrel_zh2en: np.ndarray = None
    ceil_en: float = float("nan")
    ceil_zh2en: float = float("nan")

    # design cells
    cells: Dict[str, np.ndarray] = field(default_factory=dict)
    ok: np.ndarray = None
    avail: Dict[str, np.ndarray] = field(default_factory=dict)
    counts: Dict[str, np.ndarray] = field(default_factory=dict)
    feasible: np.ndarray = None

    top_translations: pd.DataFrame = None

    # ------------------------------------------------------------------ views --
    @property
    def n_en(self) -> int:
        return len(self.nodes_en)

    @property
    def n_zh(self) -> int:
        return len(self.nodes_zh)

    @property
    def report(self) -> Dict[str, np.ndarray]:
        """The metric columns attached to every candidate pair."""
        return {"rw_en": self.rw_en, "rw_zh, alignment weighted": self.rw_zh2en,
                "str_en": self.s_en, "str_zh, alignment weighted": self.s_zh2en,
                "cnt_en": self.c_en, "cnt_zh, alignment weighted": self.c_zh2en}

    # ---------------------------------------------------------------- routes --
    def find_latent_route(self, cue_en: str, target_en: str) -> Optional[Dict[str, Any]]:
        """Return the winning (cue_zh, target_zh) route for a Mandarin-related English
        pair, plus its Mandarin association strength and respondent count on that
        route, each side's Rcs/weight, and the word's other candidate translations.

        Mirrors aggregate_routes(s_zh, ...): argmax over route pairs of s_zh, restricted
        to routes where rel_zh is actually true (the criterion rel_zh2en is built from).
        Returns None if the pair is not Mandarin-related via any route (e.g. M-E+ pairs,
        or same-word shared-translation pairs already excluded upstream).
        """
        edges, TIDX, TW = self.edges, self.TIDX, self.TW
        ci, ti = self.wpos.get(cue_en), self.wpos.get(target_en)
        if ci is None or ti is None:
            return None

        cue_idx = TIDX[ci][TIDX[ci] >= 0]
        tgt_idx = TIDX[ti][TIDX[ti] >= 0]
        cue_w = TW[ci][TIDX[ci] >= 0]
        tgt_w = TW[ti][TIDX[ti] >= 0]
        if len(cue_idx) == 0 or len(tgt_idx) == 0:
            return None

        sub_rel = self.rel_zh[np.ix_(cue_idx, tgt_idx)]
        sub_s = self.s_zh[np.ix_(cue_idx, tgt_idx)]
        sub_s = np.where(sub_rel, sub_s, -np.inf)
        if not np.isfinite(sub_s).any():
            return None   # not Mandarin-related through any route

        a, b = np.unravel_index(np.argmax(sub_s), sub_s.shape)
        cue_zh, target_zh = self.nodes_zh[cue_idx[a]], self.nodes_zh[tgt_idx[b]]

        def other_translations(w, chosen_zh):
            zh_list, rcs_list, w_list = self.top_translations.loc[w, ["zh", "rcs", "w"]]
            return [{"zh": z, "rcs": float(r), "w": float(wt)}
                    for z, r, wt in zip(zh_list, rcs_list, w_list) if z != chosen_zh]

        return {
            "route_cue_zh": cue_zh,
            "route_target_zh": target_zh,
            "str_zh": float(self.s_zh[cue_idx[a], tgt_idx[b]]),
            "cnt_zh": float(self.c_zh[cue_idx[a], tgt_idx[b]]),
            "route_cue_rcs": float(edges.loc[(edges["en"] == cue_en) & (edges["zh"] == cue_zh), "rcs"].iloc[0]),
            "route_target_rcs": float(edges.loc[(edges["en"] == target_en) & (edges["zh"] == target_zh), "rcs"].iloc[0]),
            "route_cue_w": float(cue_w[a]),
            "route_target_w": float(tgt_w[b]),
            "route_cue_other_translations": other_translations(cue_en, cue_zh),
            "route_target_other_translations": other_translations(target_en, target_zh),
        }


def build_context(cfg: Config) -> StimulusContext:
    """Load the norms, build the bridge, and derive the four condition cells."""
    set_verbose(cfg.verbose)
    cfg.ensure_dirs()
    ctx = StimulusContext(cfg=cfg)

    # ---- lexicons ----------------------------------------------------------
    vocab_en, vocab_zh, ctx.str_en, ctx.str_zh = data_io.load_vocabularies(cfg)

    # ---- bipartite translation bridge --------------------------------------
    ctx.edges = _bridge.build_translation_map(vocab_en, vocab_zh, cfg)
    ctx.nodes_en = sorted(ctx.edges["en"].unique())
    ctx.nodes_zh = sorted(ctx.edges["zh"].unique())
    log(f"Final lexicon: {ctx.n_en:,} English words connected with {ctx.n_zh:,} Chinese words.")

    # ---- monolingual network matrices --------------------------------------
    ctx.s_en, ctx.c_en = strength_matrices(ctx.str_en, ctx.nodes_en, "en", cfg)
    ctx.rw_en = data_io.load_rw(cfg.path_rw_en, ctx.nodes_en, "rw_en", cfg)
    ctx.s_zh, ctx.c_zh = strength_matrices(ctx.str_zh, ctx.nodes_zh, "zh", cfg)
    ctx.rw_zh = data_io.load_rw(cfg.path_rw_zh, ctx.nodes_zh, "rw_zh", cfg)

    # ---- route table --------------------------------------------------------
    ctx.TIDX, ctx.TW, ctx.wpos, ctx.zpos = _bridge.build_route_table(
        ctx.edges, ctx.nodes_en, ctx.nodes_zh, cfg)
    ctx.top_translations = (ctx.edges.sort_values("w", ascending=False)
                            .groupby("en")[["zh", "rcs", "w"]].agg(list))

    # ---- apply the RELATED test in Chinese space FIRST, then combine --------
    ctx.rel_zh = ((ctx.c_zh >= cfg.min_count["zh"]) &
                  (ctx.s_zh >= cfg.min_strength["zh"])).astype(bool)
    ctx.produced_zh = ((ctx.s_zh > 0) | (ctx.s_zh.T > 0)).astype(bool)   # said in either direction

    agg = _bridge.aggregate_routes
    ctx.rel_zh_mapped = agg(ctx.rel_zh, ctx.TIDX, ctx.TW).astype(bool)   # first 2 relatedness criteria

    # matrices in the English space
    ctx.produced_zh2en = agg(ctx.produced_zh, ctx.TIDX, ctx.TW).astype(bool)  # any route produced?
    ctx.rw_zh2en = agg(ctx.rw_zh, ctx.TIDX, ctx.TW)                 # weighted RW over routes
    ctx.s_zh2en = agg(ctx.s_zh, ctx.TIDX, ctx.TW)
    ctx.c_zh2en = agg(ctx.c_zh, ctx.TIDX, ctx.TW)

    ctx.median_mapped_strength = float(
        np.nanmedian(np.where(ctx.rel_zh_mapped, ctx.s_zh2en, np.nan)))
    ctx.rel_zh2en = ctx.rel_zh_mapped & (ctx.s_zh2en >= ctx.median_mapped_strength)

    ctx.shared_translation = _bridge.shared_translation_matrix(
        ctx.TIDX, ctx.TW, ctx.n_en, ctx.n_zh, cfg)

    log(f"Mandarin-related English pairs: {ctx.rel_zh2en.sum():,}")
    log(f"pairs sharing a Chinese form (excluded): "
        f"{int(ctx.shared_translation.sum() - ctx.n_en):,}")

    # ---- four conditions ----------------------------------------------------
    ctx.rel_en = (ctx.c_en >= cfg.min_count["en"]) & (ctx.s_en >= cfg.min_strength["en"])
    ctx.ceil_en = _cond.calibrate_ceiling(ctx.rw_en, ctx.rel_en, cfg.unrel_rw_overlap)
    absent_en = (ctx.s_en <= 0) & (ctx.s_en.T <= 0)          # never said, either direction
    ctx.unrel_en = absent_en & np.isfinite(ctx.rw_en) & (ctx.rw_en <= ctx.ceil_en)

    # Chinese side, determined after weighting by the translation edges
    ctx.ceil_zh2en = _cond.calibrate_ceiling(ctx.rw_zh2en, ctx.rel_zh2en, cfg.unrel_rw_overlap)
    ctx.unrel_zh2en = ((~ctx.produced_zh2en) & np.isfinite(ctx.rw_zh2en) &
                       (ctx.rw_zh2en <= ctx.ceil_zh2en)).astype(bool)

    for m_ in (ctx.rel_en, ctx.unrel_en, ctx.rel_zh2en, ctx.unrel_zh2en):
        np.fill_diagonal(m_, False)

    log(f"RW ceilings: English {ctx.ceil_en:.4f}, Chinese {ctx.ceil_zh2en:.4f}")
    log(f"English: {ctx.rel_en.sum():,} related, {ctx.unrel_en.sum():,} unrelated")
    log(f"Chinese: {ctx.rel_zh2en.sum():,} related, {ctx.unrel_zh2en.sum():,} unrelated")

    ctx.cells = _cond.build_cells(ctx.rel_en, ctx.unrel_en, ctx.rel_zh2en,
                                  ctx.unrel_zh2en, cfg)

    # ---- eligible cues ------------------------------------------------------
    ctx.ok = _cond.build_ok_mask(ctx.nodes_en, ctx.shared_translation, cfg)
    ctx.avail, ctx.counts, ctx.feasible = _cond.filter_cues(
        ctx.nodes_en, ctx.cells, ctx.ok, cfg)
    return ctx

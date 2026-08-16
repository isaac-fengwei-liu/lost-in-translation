"""Bridge-shuffle permutation test + Benjamini-Hochberg FDR correction.

Validates that the latent M+ pairs are related above chance *because of* the
semantic alignment carried by the translation bridge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .context import StimulusContext
from .utils import log


def bridge_shuffle_test(design: pd.DataFrame, ctx: StimulusContext,
                        n_perm: int | None = None) -> pd.DataFrame:
    """Rewire the translation bridge, holding the English pair fixed.

    Preserved under the null, per English word:
      * the NUMBER of Chinese translations (substitution is in place)
      * each edge's Rcs and weight      -> equally good translations
      * each Chinese word's degree      -> equally connectable inside Chinese

    Reuses TIDX from the route table rather than rebuilding one, and reuses the
    rw_zh2en already stored on the design rather than recomputing the observed
    statistic. Both were verified identical to freshly built versions; rebuilding
    them also made the result depend on how ties in `w` happened to be ordered
    (e.g. `account` has 账户 and 账号 at exactly the same weight).
    """
    cfg = ctx.cfg
    n_perm = cfg.n_permutations if n_perm is None else n_perm
    nodes_en, nodes_zh = ctx.nodes_en, ctx.nodes_zh
    edges, rw_zh, rel_zh = ctx.edges, ctx.rw_zh, ctx.rel_zh
    TIDX, TW = ctx.TIDX, ctx.TW

    rng = np.random.default_rng(cfg.seed)
    wpos = {w: i for i, w in enumerate(nodes_en)}
    zpos = {z: i for i, z in enumerate(nodes_zh)}

    # association degree = how many Chinese words this one is genuinely related to
    deg = np.asarray(rel_zh.sum(axis=1)).ravel()

    ed = edges.copy()
    ed["zi"] = ed["zh"].map(zpos)
    ed["ei"] = ed["en"].map(wpos)
    ed = ed.dropna(subset=["zi", "ei"]).copy()
    ed["zi"] = ed["zi"].astype(int)
    ed["ei"] = ed["ei"].astype(int)
    ed["deg"] = deg[ed["zi"].to_numpy()]

    # stratum = association degree, 4 buckets
    ed["strat"] = pd.qcut(ed["deg"].rank(method="first"), 1,
                          labels=False, duplicates="drop")
    pools = {s: g["zi"].unique() for s, g in ed.groupby("strat")}
    log(f"  {len(pools)} strata (degree quartile), "
        f"pool sizes {min(len(p) for p in pools.values())}-"
        f"{max(len(p) for p in pools.values())}")

    # Stratum label for each slot of TIDX, looked up per (English word, Chinese
    # word) edge, so the two tables are aligned by construction.
    strat_of_edge = {(r.ei, r.zi): r.strat for r in ed.itertuples()}
    STRAT = np.full(TIDX.shape, -1, dtype=np.int32)
    for i in range(TIDX.shape[0]):
        for k in range(TIDX.shape[1]):
            z = TIDX[i, k]
            if z >= 0:
                STRAT[i, k] = strat_of_edge.get((i, int(z)), -1)

    ci = design["cue_en"].map(wpos).to_numpy()
    ti = design["target_en"].map(wpos).to_numpy()
    NEG = np.float32(-1e9)

    def statistic(T, TW):
        """max Chinese RW over every cue-route x target-route combination,
        dampened by both cue and target edges. Same weighting as aggregate_routes():
        P(target_zh|cue_zh) * W(cue_en,cue_zh) * W(target_en,target_zh)
        """
        A, B = T[ci], T[ti]                       # (rows, K)
        Aw, Bw = TW[ci], TW[ti]
        Ac, Bc = np.clip(A, 0, None), np.clip(B, 0, None)
        vals = rw_zh[Ac[:, :, None], Bc[:, None, :]]
        vals_w = vals * Aw[:, :, None] * Bw[:, None, :]
        ok = ((A >= 0)[:, :, None] & (B >= 0)[:, None, :]
              & (Ac[:, :, None] != Bc[:, None, :])      # a word is not its own pair
              & np.isfinite(vals))
        return np.where(ok, vals_w, NEG).max(axis=(1, 2))

    # The observed statistic is exactly the rw_zh2en already on each row.
    observed = design["rw_zh, alignment weighted"].to_numpy(dtype=float)
    valid = np.isfinite(observed)
    obs_cmp = np.where(valid, observed, -np.inf)   # invalid rows get p = NaN below

    hits = np.zeros(len(design), dtype=np.int32)
    log(f"  {n_perm:,} rewired bridges")
    for _ in range(n_perm):
        P = TIDX.copy()
        # Sampling is WITH REPLACEMENT and may redraw a word's original
        # translation (~0.27% of edges per shuffle). That is deliberate: under
        # the null the real assignment is one of the exchangeable outcomes, so
        # excluding it would bias the null downward and inflate significance.
        for s, pool in pools.items():
            m = (STRAT == s)
            if m.any():
                P[m] = pool[rng.integers(0, len(pool), int(m.sum()))]
        hits += (statistic(P, TW) >= obs_cmp)

    d = design.copy()
    d["zh_link_observed"] = np.where(valid, observed, np.nan)
    d["p_bridge"] = np.where(valid, (hits + 1) / (n_perm + 1), np.nan)
    return d


def benjamini_hochberg(p) -> np.ndarray:
    """Standard BH step-up FDR correction. Returns q-values aligned to input."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]      # enforce monotonicity
    res = np.empty(m)
    res[order] = np.clip(q, 0, 1)
    out[ok] = res
    return out


def run_permutation_test(design: pd.DataFrame, ctx: StimulusContext) -> pd.DataFrame:
    """Permutation test + BH correction, with the same summaries the notebook printed."""
    design = bridge_shuffle_test(design, ctx)

    print(design.groupby("condition").agg(
        n=("p_bridge", "size"),
        median_p=("p_bridge", "median"),
        frac_p_lt_05=("p_bridge", lambda s: (s < 0.05).mean()),
        zh_link=("zh_link_observed", "mean")).round(4).to_string())

    design["q_bridge"] = benjamini_hochberg(design["p_bridge"].to_numpy())

    for c in ctx.cfg.conditions:
        s = design[design.condition == c]
        if len(s) == 0:
            continue
        log(f"{c}: median p {s.p_bridge.median():.3f} | "
            f"raw p<.05 {int((s.p_bridge < .05).sum())}/{len(s)} | "
            f"BH q<.05 {int((s.q_bridge < .05).sum())}/{len(s)}")
    return design

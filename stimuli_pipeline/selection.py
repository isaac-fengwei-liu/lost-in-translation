"""Choosing the stimuli: rank candidates, add similarity, match covariates, pick the design."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .config import Config
from .context import StimulusContext
from .matrices import cosine_rows
from .utils import argsort_desc, log

COVARIATES = ["conc_avg", "freq_zipf", "length"]


def fill_conditions(cfg: Config) -> List[str]:
    """Conditions filled directly here; M-E- is rematched during counterbalancing."""
    return [c for c in cfg.conditions if not (cfg.rematched and c == "M-E-")]


# --------------------------------------------------------------------------- #
# 1. rank every eligible cue-target pair by weighted association strength
# --------------------------------------------------------------------------- #
def rank_targets(ctx: StimulusContext) -> pd.DataFrame:
    cfg = ctx.cfg
    nodes, edges, avail, feasible = ctx.nodes_en, ctx.edges, ctx.avail, ctx.feasible
    s_en, s_zh2en, report = ctx.s_en, ctx.s_zh2en, ctx.report

    cells_to_fill = fill_conditions(cfg)
    targs_by_cue: Dict[int, Dict[str, List[int]]] = {}

    for ci in feasible:
        targs = {}
        for cond in cells_to_fill:
            cand = np.array([t for t in np.where(avail[cond][ci])[0]])
            key = s_zh2en[ci, cand] if cond == "M+E-" else s_en[ci, cand]
            targs[cond] = cand[argsort_desc(key)].tolist()
        targs_by_cue[ci] = targs

    top_route = (edges.sort_values("w", ascending=False)
                 .drop_duplicates("en").set_index("en"))

    # M+ conditions (M+E+, M+E-) are driven by a latent Chinese route; recover and
    # attach it so the bridge behind every M+ pair can be inspected and reported.
    m_plus_conds = {"M+E+", "M+E-"}

    rows = []
    for ci in feasible:
        for cond in cells_to_fill:
            for t in targs_by_cue[ci][cond]:
                row = {
                    "cue_en": nodes[ci], "target_en": nodes[t], "condition": cond,
                    "cue_zh": top_route.at[nodes[ci], "zh"],
                    "target_zh": top_route.at[nodes[t], "zh"],
                    "cue_len": len(nodes[ci]), "target_len": len(nodes[t]),
                    **{k: float(M[ci, t]) for k, M in report.items()},
                }
                if cond in m_plus_conds:
                    latent = ctx.find_latent_route(nodes[ci], nodes[t])
                    if latent is not None:
                        row.update(latent)
                rows.append(row)
    log(f"found and ranked {len(feasible)} cues, {len(rows)} cue-target pairs")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. cross-linguistic neighbourhood similarity
# --------------------------------------------------------------------------- #
def add_similarity(ctx: StimulusContext, all_pairs: pd.DataFrame,
                   save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = ctx.cfg
    a_ = np.nan_to_num(ctx.s_en) > 0
    b_ = np.nan_to_num(ctx.s_zh2en) > 0
    inter = (a_ & b_).sum(axis=1).astype(float)
    union = (a_ | b_).sum(axis=1).astype(float)

    similarity = pd.DataFrame({
        "en": ctx.nodes_en,
        "sim_cos_rw": cosine_rows(ctx.rw_en, ctx.rw_zh2en),    # the usable one
        "sim_cos_str": cosine_rows(ctx.s_en, ctx.s_zh2en),     # noisy, for contrast
        "sim_jaccard": np.where(union > 0, inter / union, np.nan),
        "n_assoc_shared": inter.astype(int),
    })

    if cfg.verbose:
        print(similarity[["sim_cos_rw", "sim_cos_str", "sim_jaccard"]]
              .describe(percentiles=[.5]).round(3).to_string())

    for c in ("sim_cos_rw", "sim_cos_str"):
        col = similarity.set_index("en")[c]
        all_pairs[f"cue_{c}"] = all_pairs["cue_en"].map(col)
        all_pairs[f"target_{c}"] = all_pairs["target_en"].map(col)
        all_pairs[f"pair_{c}"] = all_pairs[[f"cue_{c}", f"target_{c}"]].mean(axis=1)

    if save:
        out = cfg.out_dir / cfg.candidate_pairs_file
        all_pairs.to_csv(out, index=False, encoding="utf-8-sig")
        log(f"saved to {out}")
    return all_pairs, similarity


# --------------------------------------------------------------------------- #
# 3. match on concreteness and frequency
# --------------------------------------------------------------------------- #
@dataclass
class CovariateFit:
    norm: pd.DataFrame
    all_words: pd.DataFrame
    variance: np.ndarray
    cues: pd.DataFrame
    targets: pd.DataFrame
    filtered_pairs: pd.DataFrame


def load_norms(cfg: Config) -> pd.DataFrame:
    norm_conc = pd.read_excel(cfg.path_conc)
    norm_freq = pd.read_excel(cfg.path_freq)
    norm = pd.merge(norm_conc[["Word", "Conc.M"]], norm_freq[["Word", "Zipf-value"]],
                    on="Word", how="inner")
    norm.columns = ["word", "conc_avg", "freq_zipf"]
    return norm


def _seuclidean_paired(A: np.ndarray, B: np.ndarray, V: np.ndarray,
                       block: int = 2048) -> np.ndarray:
    """Row-aligned standardized Euclidean distance, i.e. cdist(A, B, ...).diagonal().

    Computed in row blocks so the full n x n matrix is never allocated; each value
    still comes out of `cdist`, so it is numerically identical to the notebook.
    """
    n = len(A)
    out = np.empty(n, dtype=float)
    for i in range(0, n, block):
        j = min(i + block, n)
        out[i:j] = cdist(A[i:j], B[i:j], metric="seuclidean", V=V).diagonal()
    return out


def add_covariate_distance(all_pairs: pd.DataFrame, norm: pd.DataFrame) -> CovariateFit:
    """Standardized Euclidean distance between each cue and its target.

    Variance comes from the set of ALL words appearing in `all_pairs`, so the metric
    is comparable across cues.
    """
    vocab_norm = norm["word"].unique().tolist()
    filtered_pairs = all_pairs[all_pairs["cue_en"].isin(vocab_norm)
                               & all_pairs["target_en"].isin(vocab_norm)].copy()

    lost_pairs = all_pairs.shape[0] - filtered_pairs.shape[0]
    affected_cues = all_pairs["cue_en"].nunique() - filtered_pairs["cue_en"].nunique()
    print(f"{lost_pairs} out of {all_pairs.shape[0]} pairs not in the concreteness norm.")
    print(f"{affected_cues} out of {all_pairs['cue_en'].nunique()} cues are affected.")

    # every distinct word in first-appearance order (cue before target, row by row)
    seen: Dict[str, int] = {}
    for cue, target, clen, tlen in zip(filtered_pairs["cue_en"], filtered_pairs["target_en"],
                                       filtered_pairs["cue_len"], filtered_pairs["target_len"]):
        if cue not in seen:
            seen[cue] = clen
        if target not in seen:
            seen[target] = tlen
    all_words = pd.DataFrame({"word": list(seen), "length": list(seen.values())})
    all_words = pd.merge(all_words, norm[["word", "conc_avg", "freq_zipf"]],
                         on="word", how="inner").reset_index(drop=True)

    variance = np.var(all_words[["conc_avg", "freq_zipf", "length"]].to_numpy(),
                      axis=0, ddof=1)

    cues = filtered_pairs[["cue_en", "condition", "cue_len"]]
    cues.columns = ["word", "condition", "length"]
    cues = pd.merge(cues, norm[["word", "conc_avg", "freq_zipf"]],
                    on="word", how="inner").reset_index(drop=True)

    targets = filtered_pairs[["target_en", "condition", "target_len",
                              "str_zh, alignment weighted"]]
    targets.columns = ["word", "condition", "length", "strength"]
    targets = pd.merge(targets, norm[["word", "conc_avg", "freq_zipf"]],
                       on="word", how="inner").reset_index(drop=True)

    if not (len(cues) == len(targets) == len(filtered_pairs)):
        raise RuntimeError(
            "cue/target covariate tables lost their row alignment with the pair table "
            f"({len(cues)} / {len(targets)} vs {len(filtered_pairs)}) -- the norm file "
            "probably contains duplicate words.")

    euc = _seuclidean_paired(cues[COVARIATES].to_numpy(),
                             targets[COVARIATES].to_numpy(), variance)
    filtered_pairs["cov_distance"] = euc
    return CovariateFit(norm=norm, all_words=all_words, variance=variance,
                        cues=cues, targets=targets, filtered_pairs=filtered_pairs)


def match_pairs(fit: CovariateFit, cfg: Config, save: bool = True) -> pd.DataFrame:
    """Pairs inside the covariate cutoff, for cues that survive in every filled cell."""
    matched = fit.filtered_pairs[fit.filtered_pairs["cov_distance"] <= cfg.dist_cutoff]
    conds = fill_conditions(cfg)
    cues_all3 = (matched
                 .groupby("cue_en")["condition"].nunique()
                 .loc[lambda s: s == len(conds)].index)
    matched = matched[matched["cue_en"].isin(cues_all3)].copy()
    if save:
        out = cfg.out_dir / cfg.matched_pairs_file
        matched.reset_index(drop=True).to_csv(out, index=False, encoding="utf-8-sig")
        log(f"saved to {out}")
    return matched


@dataclass
class CliqueResult:
    """The cue set, plus which cues are only there because a tie went their way.

    The "largest mutually-close set of cues" is usually NOT unique: several maximum
    cliques of the same size exist, and `max(nx.find_cliques(G), key=len)` returns
    whichever the enumerator happens to emit first — an order that changes between
    networkx versions. Enumerating them all makes the result reproducible and shows
    exactly which cues are contested.
    """
    cues: List[str]                      # the selected cue set
    maximum_cliques: List[frozenset]     # every clique of the maximum size
    core: Set[str]                       # cues present in EVERY maximum clique
    tied: Set[str]                       # cues present in some but not all
    size: int                            # size of one maximum clique
    rule: str                            # how `cues` was chosen
    complete: bool = True                # False if enumeration hit the cap

    def status(self, cue: str) -> str:
        return "core" if cue in self.core else ("tied" if cue in self.tied else "")


def _clique_graph(matched: pd.DataFrame, fit: CovariateFit, cfg: Config):
    cues_all3 = matched["cue_en"].unique()
    cues_u = fit.cues[fit.cues["word"].isin(cues_all3)].drop_duplicates(["word"])
    cues_dist = cdist(cues_u[COVARIATES], cues_u[COVARIATES],
                      metric="seuclidean", V=fit.variance)

    adj = cues_dist <= cfg.dist_cutoff
    np.fill_diagonal(adj, False)

    G = nx.Graph()
    G.add_nodes_from(cues_u["word"])
    words_arr = cues_u["word"].to_numpy()
    cue_ed = [(words_arr[i], words_arr[j])
              for i, j in zip(*np.where(adj))
              if i < j]
    G.add_edges_from(cue_ed)
    return G, cues_u, cues_dist, words_arr


def _mean_pairwise(cues: Sequence[str], words_arr: np.ndarray,
                   cues_dist: np.ndarray) -> float:
    pos = {w: i for i, w in enumerate(words_arr)}
    idx = [pos[c] for c in cues]
    sub = cues_dist[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), 1)
    return float(sub[iu].mean()) if len(idx) > 1 else 0.0


def largest_cue_clique(matched: pd.DataFrame, fit: CovariateFit,
                       cfg: Config) -> CliqueResult:
    """Largest set of cues that are mutually within `dist_cutoff` of each other.

    (i.e. max clique in the graph where edges connect cues with cues_dist <= cutoff)

    All maximum cliques are enumerated, then `cfg.clique_rule` decides what to keep:

    - ``union``          every cue that appears in any maximum clique. Superset of a
                         clique, so the contested cues are NOT mutually close to each
                         other — they are flagged `tied` for manual review.
    - ``tightest``       the maximum clique with the smallest mean pairwise distance
                         (ties broken alphabetically).
    - ``lexicographic``  the alphabetically first maximum clique.
    - ``first``          legacy behaviour: whichever clique the enumerator emits
                         first. Reproducible only within one networkx version.
    """
    G, cues_u, cues_dist, words_arr = _clique_graph(matched, fit, cfg)

    best, found, total, complete = 0, [], 0, True
    for c in nx.find_cliques(G):
        total += 1
        if len(c) > best:
            best, found = len(c), []
        if len(c) == best:
            found.append(frozenset(c))
        if total >= cfg.max_clique_enumeration:
            complete = False
            log(f"  WARNING: stopped after {total:,} maximal cliques "
                f"(max_clique_enumeration) -- the tie report may be incomplete")
            break

    maximum_cliques = sorted(set(found), key=lambda s: sorted(s))
    core = set.intersection(*[set(s) for s in maximum_cliques])
    union = set().union(*[set(s) for s in maximum_cliques])
    tied = union - core

    rule = cfg.clique_rule
    if rule == "union":
        cues = sorted(union)
    elif rule == "tightest":
        cues = sorted(min(maximum_cliques,
                          key=lambda s: (_mean_pairwise(sorted(s), words_arr, cues_dist),
                                         sorted(s))))
    elif rule == "lexicographic":
        cues = sorted(maximum_cliques[0])
    elif rule == "first":
        cues = list(max(nx.find_cliques(G), key=len))
    else:
        raise ValueError(f"unknown clique_rule {rule!r} -- expected one of "
                         "union / tightest / lexicographic / first")

    print(f"Largest mutually-close set ({best} words, cutoff={cfg.dist_cutoff}):")
    log(f"  {len(maximum_cliques)} maximum clique(s) of size {best} "
        f"among {total:,} maximal cliques; {len(core)} cues in all of them")
    if tied:
        log(f"  {len(tied)} contested cue(s), each in some but not all maximum "
            f"cliques: {sorted(tied)}")
        for i, s in enumerate(maximum_cliques, 1):
            log(f"    clique {i}: {sorted(set(s) & tied)} "
                f"(mean pairwise d {_mean_pairwise(sorted(s), words_arr, cues_dist):.4f})")
    log(f"  rule '{rule}' -> {len(cues)} cues kept"
        + (" (union of all maximum cliques: the tied cues are not mutually close, "
           "so the kept set is not itself a clique)" if rule == "union" and tied else ""))
    return CliqueResult(cues=cues, maximum_cliques=maximum_cliques, core=core,
                        tied=tied, size=best, rule=rule, complete=complete)


def build_design(matched: pd.DataFrame, clique, cfg: Config,
                 save: bool = True) -> pd.DataFrame:
    """The strongest target in each condition, for every cue in the clique.

    `matched` is already ranked by strength, so keeping the first row per
    (cue, condition) takes the strongest surviving target.

    When `clique` is a `CliqueResult`, a `clique_status` column marks each cue as
    `core` (in every maximum clique) or `tied` (kept only because it won a tie), and
    the tied rows are also written out on their own for easy copy-paste into the
    manual-review file.
    """
    result = clique if isinstance(clique, CliqueResult) else None
    cues = list(result.cues) if result is not None else list(clique)

    design = (matched[matched["cue_en"].isin(cues)]
              .drop_duplicates(["cue_en", "condition"], keep="first")
              .reset_index(drop=True))

    if result is not None:
        design["clique_status"] = [result.status(c) for c in design["cue_en"]]

    if save:
        out = cfg.out_dir / cfg.design_file
        design.to_csv(out, index_label="id", encoding="utf-8-sig")
        log(f"saved to {out}")

        if result is not None and result.tied:
            tied_rows = design[design["clique_status"] == "tied"]
            if len(tied_rows):
                out_tied = cfg.out_dir / cfg.tied_cues_file
                tied_rows.to_csv(out_tied, index_label="id", encoding="utf-8-sig")
                log(f"saved the {tied_rows['cue_en'].nunique()} contested cues "
                    f"({len(tied_rows)} rows) to {out_tied}")
    return design


# --------------------------------------------------------------------------- #
# the whole selection stage in one call
# --------------------------------------------------------------------------- #
def select_design(ctx: StimulusContext, save: bool = True):
    """rank -> similarity -> covariate matching -> clique -> design."""
    cfg = ctx.cfg
    all_pairs = rank_targets(ctx)
    all_pairs, similarity = add_similarity(ctx, all_pairs, save=save)
    norm = load_norms(cfg)
    fit = add_covariate_distance(all_pairs, norm)
    matched = match_pairs(fit, cfg, save=save)
    clique = largest_cue_clique(matched, fit, cfg)
    design = build_design(matched, clique, cfg, save=save)
    return design, dict(all_pairs=all_pairs, similarity=similarity, fit=fit,
                        matched=matched, clique=clique)

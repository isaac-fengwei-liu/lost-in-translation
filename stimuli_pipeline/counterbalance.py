"""Global uniqueness check, M-E- rematch, and the 4-list Latin square.

Consumes the manually reviewed 3-cell design (`manual_review/uniqueness_checked3.csv`
by default) plus the matrices in a `StimulusContext`, and produces `full_design.xlsx`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .conditions import build_ok_mask
from .config import Config
from .context import StimulusContext
from .utils import log


# --------------------------------------------------------------------------- #
# 1. global uniqueness check
# --------------------------------------------------------------------------- #
def load_reviewed(cfg: Config) -> pd.DataFrame:
    v_design = pd.read_csv(cfg.path_reviewed)
    log(f"loaded {len(v_design)} reviewed rows from {cfg.path_reviewed}")
    return v_design


# metric columns that can be read straight back out of the matrices
MATRIX_COLUMNS = {
    "rw_en": "rw_en",
    "rw_zh, alignment weighted": "rw_zh2en",
    "str_en": "s_en",
    "str_zh, alignment weighted": "s_zh2en",
    "cnt_en": "c_en",
    "cnt_zh, alignment weighted": "c_zh2en",
}

# Unweighted Chinese association strength on the pair's own translation route.
# `find_latent_route` returns it as "str_zh"; older stage-1 output called the same
# quantity "route_s_zh", so both names are accepted when reading a row back.
ROUTE_STRENGTH_COLUMN = "str_zh"
ROUTE_STRENGTH_ALIASES = ("str_zh", "route_s_zh")


def _route_strength(latent: Dict[str, float]) -> Optional[float]:
    for k in ROUTE_STRENGTH_ALIASES:
        if k in latent:
            return float(latent[k])
    return None


def patch_reviewed_metrics(ctx: StimulusContext, v_design: pd.DataFrame,
                           columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Fill metric cells left blank by manual editing. Never overwrites a value.

    Rows pasted in during review arrive with holes -- e.g. rows from
    `matched_pairs_reordered.csv`, whose `str_zh` column is empty because the stage-1
    run that produced it wrote that quantity under the name `route_s_zh`. Everything here is recomputable, so the holes are filled from
    the same matrices the design was built from:

    - `MATRIX_COLUMNS`  -> matrix[cue, target], looked up by `ctx.wpos`
    - `str_zh`          -> s_zh[route_cue_zh, route_target_zh], i.e. the strength on
                           the route recorded on that row. If the row has no route
                           recorded, the winning route is recovered with
                           `ctx.find_latent_route`; if the pair is not Mandarin-related
                           through any route (every M-E+ row), the cell stays blank.
                           A column still named `route_s_zh` is handled the same way.
    - `cue_len` / `target_len` -> the word lengths.
    """
    df = v_design.copy()
    wanted = set(columns) if columns else None
    filled: Dict[str, int] = {}

    def _target(col: str) -> bool:
        return (wanted is None or col in wanted) and col in df.columns

    # ---- straight matrix look-ups ------------------------------------------
    for col, attr in MATRIX_COLUMNS.items():
        if not _target(col):
            continue
        M = getattr(ctx, attr)
        gaps = df.index[df[col].isna()]
        n = 0
        for i in gaps:
            ci, ti = ctx.wpos.get(df.at[i, "cue_en"]), ctx.wpos.get(df.at[i, "target_en"])
            if ci is not None and ti is not None:
                df.at[i, col] = float(M[ci, ti])
                n += 1
        if n:
            filled[col] = n

    # ---- the route strength ------------------------------------------------
    column = next((c for c in ROUTE_STRENGTH_ALIASES if _target(c)), None)
    if column is not None:
        gaps = df.index[df[column].isna()]
        n_route, n_recovered, no_route = 0, 0, []
        for i in gaps:
            a = df.at[i, "route_cue_zh"] if "route_cue_zh" in df.columns else None
            b = df.at[i, "route_target_zh"] if "route_target_zh" in df.columns else None
            za, zb = ctx.zpos.get(a), ctx.zpos.get(b)
            if za is not None and zb is not None:
                df.at[i, column] = float(ctx.s_zh[za, zb])
                n_route += 1
                continue
            latent = ctx.find_latent_route(df.at[i, "cue_en"], df.at[i, "target_en"])
            value = _route_strength(latent) if latent is not None else None
            if value is not None:
                df.at[i, column] = value
                n_recovered += 1
            else:
                no_route.append((df.at[i, "cue_en"], df.at[i, "target_en"],
                                 df.at[i, "condition"]))
        if n_route or n_recovered:
            filled[column] = n_route + n_recovered
            log(f"  {column}: {n_route} filled from the route recorded "
                f"on the row, {n_recovered} from the recovered winning route")
        if no_route:
            by_cond = Counter(c for _, _, c in no_route)
            log(f"  {len(no_route)} row(s) left blank -- no Mandarin route "
                f"(expected for M-E-/M-E+): {dict(by_cond)}")

    # ---- word lengths ------------------------------------------------------
    for col, src in (("cue_len", "cue_en"), ("target_len", "target_en")):
        if not _target(col):
            continue
        gaps = df.index[df[col].isna()]
        for i in gaps:
            df.at[i, col] = len(str(df.at[i, src]))
        if len(gaps):
            filled[col] = len(gaps)

    if filled:
        log(f"patched {sum(filled.values())} blank metric cell(s): {filled}")
    else:
        log("no blank metric cells to patch")
    return df


def uniqueness_check(v_design: pd.DataFrame, cfg: Config, save: bool = True) -> pd.DataFrame:
    """Flag words that appear more than once in the stimulus set.

    Each cue is counted ONCE (not once per condition -- every cue legitimately has
    one row per condition, so counting rows would flag everything trivially) and each
    cue-target pair once. A word is "duplicated" if:
      - it is the target in more than one distinct cue-target pair, or
      - it is used as both a cue (for its own row) and a target (for some pair)

    This runs BEFORE the M-E- rematch below, so it reflects the 3-cell
    (M+E+/M+E-/M-E+) set only -- the M-E- step deliberately reuses M+E+ targets
    afterwards, which is by design, not a uniqueness violation.
    """
    cue_words = set(v_design["cue_en"])
    pairs = v_design[["cue_en", "target_en"]].drop_duplicates()
    target_counts = pairs["target_en"].value_counts()   # times used as a target, across distinct pairs

    dupe_targets = set(target_counts[target_counts > 1].index)
    cue_as_target = cue_words & set(pairs["target_en"])

    dupe_words = dupe_targets | cue_as_target

    v_design["cue_duplicated"] = v_design["cue_en"].isin(cue_as_target)
    v_design["target_duplicated"] = v_design["target_en"].isin(dupe_words)
    v_design["has_duplicate"] = v_design["cue_duplicated"] | v_design["target_duplicated"]

    n_dupe_rows = int(v_design["has_duplicate"].sum())
    log(f"{len(dupe_targets)} words used as a target in more than one cue-target pair "
        f"({sorted(dupe_targets)})")
    log(f"{len(cue_as_target)} words used as a cue AND as (someone else's) target: "
        f"{sorted(cue_as_target)}")
    log(f"{len(dupe_words)} words duplicated overall ({n_dupe_rows}/{len(v_design)} rows flagged)")

    if dupe_targets:
        print(target_counts[target_counts > 1].sort_values(ascending=False).to_string())

    if save:
        out = cfg.out_dir / cfg.uniqueness_issue_file
        v_design.to_csv(out, index=False, encoding="utf-8-sig")
        log(f"saved to {out}")
    return v_design


# --------------------------------------------------------------------------- #
# 2. rematch the M-E- cell from the M+E+ target pool
# --------------------------------------------------------------------------- #
def rematch_minus_minus(ctx: StimulusContext, v_design: pd.DataFrame) -> pd.DataFrame:
    """Fill M-E- by reusing the M+E+ target pool.

    The critical (M+E+) and baseline (M-E-) cells then share the same vocabulary, so
    word frequency/length/concreteness cannot confound the comparison. A candidate
    target is admissible for a cue only if it is unrelated to that cue in BOTH English
    (unrel_en) and Mandarin (unrel_zh2en), excludes shared-translation and
    morphological near-duplicates, and isn't already used by that cue.

    `max_total_target_uses` caps how many (cue, condition) slots any single word may
    occupy across the WHOLE 4-condition design (not just M-E-). This matters for the
    later Latin-square counterbalancing: with exactly 4 lists, a word used at 5+
    distinct (cue, condition) slots is guaranteed by pigeonhole to collide with
    itself in some list no matter how cues are grouped -- and even exactly 4 slots
    leaves zero slack for the solver. Capping at 3 keeps one list's worth of margin.
    """
    cfg = ctx.cfg
    nodes_en, wpos, edges = ctx.nodes_en, ctx.wpos, ctx.edges
    max_uses = cfg.max_total_target_uses

    rng = np.random.default_rng(cfg.seed)

    ok = build_ok_mask(nodes_en, ctx.shared_translation, cfg)
    avail_mm = ctx.unrel_en & ctx.unrel_zh2en & ok

    pool_words = sorted(v_design.loc[v_design["condition"] == "M+E+", "target_en"].unique())
    pool = [wpos[w] for w in pool_words if w in wpos]
    log(f"rematching M-E- from the {len(pool):,}-word M+E+ pool")

    lens = np.array([len(w) for w in nodes_en])
    anchor = float(np.median(lens))
    len_penalty = 0.10 * np.maximum(0.0, np.abs(lens - anchor) - 1)

    # pre-existing usage of each word across the 3 already-fixed conditions
    # (M+E+, M+E-, M-E+) -- the M-E- draw must not push any word past the cap
    existing_target_counts = Counter(v_design["target_en"])
    over_cap_before_mm = {w: c for w, c in existing_target_counts.items() if c >= max_uses}
    if over_cap_before_mm:
        log(f"  WARNING: {len(over_cap_before_mm)} words already at/over the cap "
            f"before M-E- is even added: {over_cap_before_mm} "
            f"-- these cannot receive an M-E- match and may still block the Latin "
            f"square if their existing slots land in the same list.")

    cue_rows = v_design[["cue_en"]].drop_duplicates()
    reuse = Counter({wpos[w]: c for w, c in existing_target_counts.items() if w in wpos})
    fallback, mm_rows = 0, []

    for cue in cue_rows["cue_en"]:
        ci = wpos.get(cue)
        own_words = set(v_design.loc[v_design["cue_en"] == cue, "target_en"]) | {cue}
        own = {wpos[w] for w in own_words if w in wpos}

        cand = [t for t in pool if t not in own and avail_mm[ci, t]
                and reuse[t] < max_uses]
        if len(cand) < cfg.n_per_cell:
            extra = [t for t in np.where(avail_mm[ci])[0]
                     if t not in own and t not in cand and reuse[t] < max_uses]
            rng.shuffle(extra)
            cand += extra[: cfg.n_per_cell - len(cand)]
            fallback += 1
        if len(cand) < cfg.n_per_cell:
            raise RuntimeError(
                f"cue '{cue}' has no M-E- candidate left that keeps every target's "
                f"total use under {max_uses} -- relax max_total_target_uses, "
                f"add eligible words, or drop/replace this cue.")
        cand.sort(key=lambda t: (reuse[t], len_penalty[t], rng.random()))
        sel = cand[:cfg.n_per_cell]
        for t in sel:
            reuse[t] += 1
            mm_rows.append({
                "cue_en": cue, "target_en": nodes_en[t], "condition": "M-E-",
                "cue_zh": v_design.loc[v_design["cue_en"] == cue, "cue_zh"].iloc[0],
                "target_zh": edges.loc[edges["en"] == nodes_en[t], "zh"].iloc[0]
                             if (edges["en"] == nodes_en[t]).any() else np.nan,
                "cue_len": len(cue), "target_len": len(nodes_en[t]),
                "rw_en": float(ctx.rw_en[ci, t]),
                "rw_zh, alignment weighted": float(ctx.rw_zh2en[ci, t]),
                "str_en": float(ctx.s_en[ci, t]),
                "str_zh, alignment weighted": float(ctx.s_zh2en[ci, t]),
                "cnt_en": float(ctx.c_en[ci, t]),
                "cnt_zh, alignment weighted": float(ctx.c_zh2en[ci, t]),
            })

    log(f"  {fallback} cues needed a fallback draw (want 0)")

    mm_design = pd.DataFrame(mm_rows)
    assert mm_design["cue_en"].nunique() == cue_rows["cue_en"].nunique()
    assert mm_design.groupby("cue_en").size().eq(cfg.n_per_cell).all()

    full_design = pd.concat(
        [v_design.drop(columns=["Issue", "cue_duplicated", "target_duplicated",
                                "has_duplicate"], errors="ignore"),
         mm_design], ignore_index=True)
    full_design.index.name = "stim_id"

    pp = set(v_design.loc[v_design.condition == "M+E+", "target_en"])
    mm = set(mm_design["target_en"])
    log(f"M+E+ and M-E- share a word set: {pp == mm}  (expect True)")
    log(f"max reuse of any one M-E- target: {mm_design['target_en'].value_counts().max()}")

    final_word_counts = pd.concat([full_design["target_en"]]).value_counts()
    log(f"max total uses of any target across all 4 conditions: {final_word_counts.max()} "
        f"(cap is {max_uses})")
    return full_design


# --------------------------------------------------------------------------- #
# 3. counterbalanced lists (Latin square by simulated annealing)
# --------------------------------------------------------------------------- #
def build_lists(full_design: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build the Latin-square lists (A-D by default).

    Every participant sees all cues exactly once; which condition a cue is tested in
    rotates across lists, so across the lists each cue appears once in every condition.
    Within a single list no target word may repeat (global uniqueness *per list*, not
    across the whole design -- the M+E+/M-E- pool sharing means the same word
    legitimately serves two different cues overall, but never within the same list).

    Cues are split into `n_lists` groups. In list `L`, a cue from group `g` is shown at
    condition `conditions[(g + L) % n_lists]`. The group assignment is found by
    simulated annealing, minimizing (a) within-list target collisions and (b)
    group-size imbalance, until zero collisions are found.
    """
    conditions = list(cfg.conditions)
    n_lists = cfg.n_lists

    pivot = full_design.pivot(index="cue_en", columns="condition", values="target_en")
    cues = pivot.index.tolist()
    n_cues = len(cues)
    target_of = {c: {cond: pivot.loc[c, cond] for cond in conditions} for c in cues}

    def list_conflicts(groups):
        """Total within-list duplicate-target collisions, summed over all lists."""
        total, per_list = 0, []
        for L in range(n_lists):
            seen = defaultdict(int)
            for i, c in enumerate(cues):
                cond = conditions[(groups[i] + L) % n_lists]
                seen[target_of[c][cond]] += 1
            dup = sum(k - 1 for k in seen.values() if k > 1)
            total += dup
            per_list.append(dup)
        return total, per_list

    def group_balance_penalty(groups, target_size=n_cues / n_lists):
        counts = np.bincount(groups, minlength=n_lists)
        return float(np.sum((counts - target_size) ** 2))

    def cost(groups):
        conf, _ = list_conflicts(groups)
        return conf * 100 + group_balance_penalty(groups)

    lsq_rng = np.random.default_rng(cfg.seed)
    groups = np.array([i % n_lists for i in range(n_cues)])
    lsq_rng.shuffle(groups)

    cur_cost = cost(groups)
    best, best_cost = groups.copy(), cur_cost

    n_iters = cfg.n_anneal_iters
    T0 = cfg.anneal_t0
    for it in range(n_iters):
        T = T0 * (1 - it / n_iters)
        i, j = lsq_rng.integers(0, n_cues, size=2)
        if groups[i] == groups[j]:
            continue
        groups[i], groups[j] = groups[j], groups[i]
        new_cost = cost(groups)
        if new_cost <= cur_cost or lsq_rng.random() < np.exp(-(new_cost - cur_cost) / max(T, 1e-6)):
            cur_cost = new_cost
            if cur_cost < best_cost:
                best_cost, best = cur_cost, groups.copy()
        else:
            groups[i], groups[j] = groups[j], groups[i]
        if best_cost == 0:
            break

    conflicts, per_list_conflicts = list_conflicts(best)
    log(f"Latin square search: best cost {best_cost}, within-list conflicts {per_list_conflicts}, "
        f"group sizes {np.bincount(best, minlength=n_lists).tolist()}")
    assert conflicts == 0, ("no collision-free Latin square found -- rerun with more "
                            "iterations or fewer cues")

    rows = []
    for L in range(n_lists):
        list_name = chr(ord("A") + L)
        for i, c in enumerate(cues):
            cond = conditions[(best[i] + L) % n_lists]
            rows.append({"list": list_name, "cue_en": c, "condition": cond,
                         "target_en": target_of[c][cond]})
    lists_df = pd.DataFrame(rows)
    return lists_df, best


def qc_lists(lists_df: pd.DataFrame, cfg: Config) -> bool:
    """Verify the counterbalanced lists."""
    conditions = list(cfg.conditions)
    list_names = sorted(lists_df["list"].unique())
    n_cues = lists_df["cue_en"].nunique()

    ok = True
    for L in list_names:
        sub = lists_df[lists_df["list"] == L]
        n_cues_ok = sub["cue_en"].nunique() == n_cues == len(sub)
        dup_targets = int(sub["target_en"].duplicated().sum())
        cue_target_overlap = set(sub["cue_en"]) & set(sub["target_en"])
        cond_counts = sub["condition"].value_counts().to_dict()
        log(f"List {L}: {sub['cue_en'].nunique()} cues, dup targets {dup_targets}, "
            f"cue/target overlap {len(cue_target_overlap)}, per-condition n {cond_counts}")
        ok &= n_cues_ok and dup_targets == 0 and not cue_target_overlap

    rotation_ok = (lists_df.pivot(index="cue_en", columns="list", values="condition")
                   .apply(lambda r: sorted(r) == sorted(conditions), axis=1).all())
    log(f"every cue tested once per condition across the {len(list_names)} lists: {rotation_ok}")
    ok &= bool(rotation_ok)

    log(f"ALL CHECKS PASSED: {ok}")
    return bool(ok)


# --------------------------------------------------------------------------- #
# 4. save
# --------------------------------------------------------------------------- #
def save_full_design(full_design: pd.DataFrame, lists_df: pd.DataFrame,
                     cfg: Config) -> Tuple[pd.DataFrame, Path]:
    """Write `full_design.xlsx`.

    - `all_pairs`: every cue-target pair (the full metric set, minus the manual
      Issue/duplicate-flag columns), plus a `list` column marking which of the
      counterbalanced lists shows that (cue, condition) pairing.
    - `List A`-`List D`: the rows shown to participants assigned to that list, with
      the full metric columns and `stim_id` carried over from `all_pairs`.
    """
    # which list shows each (cue_en, condition) pairing
    pair_to_list = {(r.cue_en, r.condition): r.list for r in lists_df.itertuples()}

    all_pairs = full_design.reset_index()  # brings back stim_id as a column
    all_pairs["list"] = [pair_to_list[(r.cue_en, r.condition)] for r in all_pairs.itertuples()]

    out_path = cfg.out_dir / cfg.full_design_file
    list_names = sorted(lists_df["list"].unique())
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        all_pairs.to_excel(writer, sheet_name="all_pairs", index=False)
        for L in list_names:
            sheet = all_pairs[all_pairs["list"] == L].reset_index(drop=True)
            sheet.to_excel(writer, sheet_name=f"List {L}", index=False)

    per_list = len(all_pairs) // max(len(list_names), 1)
    log(f"saved {cfg.full_design_file}: 1 all_pairs sheet ({len(all_pairs)} rows) + "
        f"{len(list_names)} per-list sheets ({per_list} rows each) to {out_path}")
    return all_pairs, out_path


# --------------------------------------------------------------------------- #
# the whole counterbalancing stage in one call
# --------------------------------------------------------------------------- #
def run_counterbalance(ctx: StimulusContext, v_design: Optional[pd.DataFrame] = None,
                       save: bool = True):
    cfg = ctx.cfg
    cfg.ensure_dirs()
    if v_design is None:
        v_design = load_reviewed(cfg)
    if cfg.patch_reviewed_metrics:
        v_design = patch_reviewed_metrics(ctx, v_design)
    v_design = uniqueness_check(v_design, cfg, save=save)
    full_design = rematch_minus_minus(ctx, v_design)
    lists_df, groups = build_lists(full_design, cfg)
    checks_passed = qc_lists(lists_df, cfg)
    all_pairs, out_path = (save_full_design(full_design, lists_df, cfg)
                           if save else (None, None))
    return dict(v_design=v_design, full_design=full_design, lists_df=lists_df,
                groups=groups, checks_passed=checks_passed, all_pairs=all_pairs,
                out_path=out_path)

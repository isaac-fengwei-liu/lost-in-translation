# Stimulus pipeline

Refactor of `design_stimuli.ipynb` + `counterbalance.ipynb` into a package that runs
from the terminal with configurable parameters. The logic is unchanged — same filters,
same matrices, same ranking, same annealer — with one deliberate exception: the
cue-clique tie-break, which the notebook left non-deterministic (see *The cue-clique
tie* below). The original notebooks are untouched; nothing here overwrites them.

```
Stimuli/
├── design_stimuli.ipynb          <- original, untouched
├── counterbalance.ipynb          <- original, untouched
├── data/  cache/  output/  manual_review/   <- unchanged, shared with the notebooks
└── pipeline/                     <- everything new lives here
    ├── design_stimuli.py         <- CLI: stage 1
    ├── counterbalance.py         <- CLI: stage 2
    ├── counterbalance.ipynb      <- thin notebook that calls the package
    ├── config.example.json
    └── stimuli_pipeline/
        ├── config.py             every tunable parameter (dataclass + JSON)
        ├── utils.py              logging, stable descending argsort
        ├── data_io.py            RW vocabularies, strength files, cached RW matrices
        ├── bridge.py             translation bridge, route table, aggregate_routes
        ├── matrices.py           strength/count matrices, row cosines
        ├── conditions.py         RW ceilings, the four M x E cells, eligible cues
        ├── context.py            build_context(): everything both stages share
        ├── selection.py          ranking, similarity, covariate matching, clique, design
        ├── qc.py                 qc_report.txt
        ├── permutation.py        bridge-shuffle test + Benjamini-Hochberg
        ├── counterbalance.py     uniqueness check, M-E- rematch, Latin square, xlsx
        └── cli.py                argparse front end
```

## Running it

```bash
cd .../Stimuli/pipeline

# stage 1 - candidate pairs + 3-cell design  (was design_stimuli.ipynb)
python design_stimuli.py

# ... manual review, producing manual_review/uniqueness_checked3.csv ...

# stage 2 - patch blanks, uniqueness check, M-E- rematch, 4 counterbalanced lists
python counterbalance.py

# or both at once (only useful when the reviewed file already exists)
python -m stimuli_pipeline all
```

Both scripts locate `data/`, `cache/`, `output/` and `manual_review/` relative to the
parent folder, so they work from anywhere as long as the repo layout is intact.

## Configurable parameters

Every field of `Config` is a flag; `--help` lists them with their defaults.

```bash
python design_stimuli.py --dist-cutoff 1.4 --min-count-en 4 --no-run-permutation-test
python design_stimuli.py --unrel-rw-overlap 0.05 --max-translations-per-word 3
python counterbalance.py --reviewed-file uniqueness_checked4.csv --n-anneal-iters 500000
python counterbalance.py --seed 20260901 --full-design-file full_design_v2.xlsx
```

Booleans take a `--no-` form (`--no-exclude-shared-translation`). Paths
(`--data-dir`, `--out-dir`, `--cache-dir`, `--review-dir`) let you point a run at a
scratch directory without touching the real outputs:

```bash
python design_stimuli.py --out-dir ../output_test --cache-dir ../cache   # reuse the cache
```

A whole configuration can live in a JSON file (see `config.example.json`):

```bash
python design_stimuli.py --config my_run.json           # load
python design_stimuli.py --dist-cutoff 1.4 --dump-config effective.json   # save
```

CLI flags win over `--config`, which wins over the defaults. The defaults reproduce
the notebooks exactly.

## Using it from a notebook / REPL

`pipeline/counterbalance.ipynb` is the notebook version of stage 2: instead of
`%run design_stimuli.ipynb` it imports the same functions, so nothing is duplicated
and nothing is re-executed that stage 2 doesn't need.

```python
from stimuli_pipeline import Config, build_context, select_design
from stimuli_pipeline.counterbalance import (load_reviewed, uniqueness_check,
                                             rematch_minus_minus, build_lists,
                                             qc_lists, save_full_design)

cfg = Config()                     # or Config(dist_cutoff=1.4, seed=123)
ctx = build_context(cfg)           # lexicons, bridge, matrices, the four cells
```

`ctx` carries everything the old notebook globals did — `ctx.nodes_en`, `ctx.wpos`,
`ctx.edges`, `ctx.rw_en`, `ctx.rw_zh2en`, `ctx.s_en`, `ctx.s_zh2en`, `ctx.c_en`,
`ctx.c_zh2en`, `ctx.unrel_en`, `ctx.unrel_zh2en`, `ctx.shared_translation`,
`ctx.cells`, `ctx.avail`, `ctx.feasible`, plus `ctx.find_latent_route(cue, target)`.

Stage boundaries are all callable on their own, so you can stop and inspect anywhere:
`rank_targets`, `add_similarity`, `add_covariate_distance`, `match_pairs`,
`largest_cue_clique`, `build_design`, `qc_report`, `run_permutation_test`.

## Outputs

| file | written by |
|---|---|
| `output/candidate_pairs.csv` | stage 1, after the similarity columns are added |
| `output/matched_cue_target_pairs.csv` | stage 1, pairs inside `dist_cutoff` |
| `output/design.csv` | stage 1, one target per cue per condition (+ `clique_status`) |
| `output/design_tied_cues.csv` | stage 1, only the rows for contested cues (see below) |
| `output/qc_report.txt` | stage 1 |
| `output/uniqueness_issue4.csv` | stage 2, reviewed design (blanks patched) + duplicate flags |
| `output/full_design.xlsx` | stage 2, `all_pairs` + `List A`-`List D` |

The RW matrix cache in `cache/` is keyed by a hash of the lexicon, exactly as before,
so the first run after a filter change rebuilds and later runs are fast.

## Blank cells left by manual review

Stage 2 starts by refilling metric cells that manual editing left empty
(`patch_reviewed_metrics`, before the uniqueness check and the M-E- rematch, so
everything downstream sees complete rows). Existing values are never overwritten and
every fill is logged.

The common case: rows pasted from `manual_review/matched_pairs_reordered.csv` have no
`str_zh`, because the stage-1 run that produced that file wrote the quantity under the
name `route_s_zh` — the column exists in the reordered file but is empty for all 1,543
rows, so the paste brings a blank. (`find_latent_route` now emits it as `str_zh`, so
files produced from here on carry the value; both names are still accepted, so older
CSVs keep working.) It is refilled as `s_zh[route_cue_zh, route_target_zh]`: the unweighted Chinese
strength on the route recorded on that row. If a row has no route recorded, the winning
route is recovered with `ctx.find_latent_route`; if the pair is not Mandarin-related
through any route — every M-E+ row — the cell stays blank, as it should.

`rw_en`, `rw_zh, alignment weighted`, `str_en`, `str_zh, alignment weighted`, `cnt_en`,
`cnt_zh, alignment weighted`, `cue_len` and `target_len` are refilled the same way,
straight out of the matrices. Turn the whole step off with
`--no-patch-reviewed-metrics`.

## The cue-clique tie (why two runs can disagree)

The cue set is the largest group of cues that are all mutually within `dist_cutoff`
of each other in covariate space — a maximum clique. That clique is **not unique**:
on the current data there are four maximum cliques of 111 cues, sharing 108 and
disagreeing on six (`candle, kidney, lift, pier, soft, strike`), with essentially the
same tightness (mean pairwise d 0.8125-0.8187, identical max 1.4938).

The notebook's `max(nx.find_cliques(G), key=len)` returns whichever one the enumerator
emits first, and that order is not stable: `nx.find_cliques` iterates over sets of node
names, so it depends on Python's per-process string-hash randomisation *and* on the
networkx version. The same graph gave `candle+kidney` on one run and `lift+soft` on the
next, on the same machine.

`largest_cue_clique` now enumerates **all** maximum cliques and applies `clique_rule`:

| `--clique-rule` | keeps |
|---|---|
| `union` (default) | every cue in any maximum clique (114 here). Contested cues are marked `tied` |
| `tightest` | the maximum clique with the smallest mean pairwise distance |
| `lexicographic` | the alphabetically first maximum clique |
| `first` | the old behaviour — reproducible only within one process/version |

With `union`, `design.csv` gains a `clique_status` column (`core` / `tied`) and the
contested rows are also written to `design_tied_cues.csv` for easy copy-paste into the
manual-review file. Note that the union is a *superset* of a clique: the tied cues are
not all mutually close to one another, which is exactly why they are flagged. The
count of maximum cliques and the contested cues are logged and recorded in
`qc_report.txt` on every run.

## Notes on the port

Apart from the clique tie-break above, behaviour is identical; three implementation
details are worth flagging.

1. **`np.argsort(key, descending=True)`** only exists on newer NumPy. `utils.argsort_desc`
   uses it when available and otherwise falls back to a stable descending sort, which
   is the same ordering (ties keep their original order).
2. **The cue↔target covariate distance** was `cdist(cues, targets).diagonal()`, which
   allocates an n×n matrix (~1.3 GB at 12,900 candidate pairs). It is now computed by
   `cdist` in row blocks and is numerically identical, just without the big allocation.
   The cue↔cue distance matrix used for the clique is small and still computed whole.
3. **The permutation test** is stage 1 only, runs after `design.csv` is written, and
   changes nothing downstream — it adds `p_bridge` / `q_bridge` to the in-memory frame.
   `--no-run-permutation-test` skips it; stage 2 never runs it.

Two small cleanups: the exploratory plot cells (`sns.histplot`, `all_pairs[...=='waiter']`)
live in the notebook rather than the library, and one unused variable in the QC cell
(`other = ...`) was dropped. `forbid_cue_as_target` is kept as a config field for the
record — as in the notebook, nothing reads it; the constraint is enforced by the
uniqueness check and the per-list QC.

# Stimulus pipeline

```
Stimuli/
├── data/  cache/  output/  manual_review/
├── design_stimuli.py         <- stage 1
├── counterbalance.py         <- stage 2
├── counterbalance.ipynb      <- stage 2 notebook for interactiveness
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

## Running

```bash
cd .../Stimuli

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

`ctx` carries global variables — `ctx.nodes_en`, `ctx.wpos`,
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

## Relationship with legacy scripts

Uses the same logic as `design_stimuli.ipynb` + `counterbalance.ipynb`. Fixed the
cue-clique tie-break handling in the original notebooks. The two notebooks are not version controlled.

"""Every tunable parameter of the stimulus pipeline, in one place.

The defaults reproduce `design_stimuli.ipynb` / `counterbalance.ipynb` exactly.
Anything here can be overridden on the command line (``--min-count-en 4``) or in a
JSON file passed with ``--config``.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# .../Stimuli/stimuli_pipeline/config.py -> .../Stimuli
DEFAULT_STIM_DIR = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # ------------------------------------------------------------------ paths --
    stim_dir: Path = DEFAULT_STIM_DIR
    data_dir: Optional[Path] = None      # default: <stim_dir>/data
    out_dir: Optional[Path] = None       # default: <stim_dir>/output
    cache_dir: Optional[Path] = None     # default: <stim_dir>/cache
    review_dir: Optional[Path] = None    # default: <stim_dir>/manual_review

    align_file: str = "Aligned_Translation_ENCH.xlsx"
    rw_en_file: str = "RWsim_R123_EN.csv"
    rw_zh_file: str = "RWsim_R123_CH.csv"
    strength_en_file: str = "strength.SWOW-EN.R123.20180827.csv"
    strength_zh_file: str = "strength.SWOWZH.R123.20230423.csv"
    concreteness_file: str = "concreteness_Brysbaert2014.xlsx"
    frequency_file: str = "SUBTLEX-US.xlsx"

    # manual-review input consumed by the counterbalancing stage
    reviewed_file: str = "uniqueness_checked3.csv"       # inside review_dir

    # output file names
    candidate_pairs_file: str = "candidate_pairs.csv"
    matched_pairs_file: str = "matched_cue_target_pairs.csv"
    design_file: str = "design.csv"
    tied_cues_file: str = "design_tied_cues.csv"
    qc_report_file: str = "qc_report.txt"
    uniqueness_issue_file: str = "uniqueness_issue4.csv"
    full_design_file: str = "full_design.xlsx"

    # ----------------------------------------------------------------- design --
    n_per_cell: int = 1                  # targets per cue per condition
    conditions: Tuple[str, ...] = ("M+E+", "M+E-", "M-E+", "M-E-")

    # ------------------------------------------------------ translation bridge --
    align_word_floor: float = 0.20       # ignore any translation candidate below this Rcs
    max_translations_per_word: int = 5   # keep at most this many routes per word

    # ------------------------------------------------------ what counts RELATED --
    # Both conditions must hold.
    min_count_en: int = 3                # minimum respondents who produced the pair
    min_count_zh: int = 3
    min_strength_en: float = 0.01        # minimum probability, on a common scale
    min_strength_zh: float = 0.01

    # English strengths are computed over ALL responses, but only ~62% of English
    # responses are themselves cue words. Chinese strengths were already restricted
    # to cue words. Renormalising English onto the same support makes them comparable.
    renormalise_en_to_cue_support: bool = True

    # ---------------------------------------------------- what counts UNRELATED --
    # Never produced in EITHER direction, AND below a calibrated RW ceiling.
    unrel_rw_overlap: float = 0.1        # ceiling set where this share of related pairs passes

    # ----------------------------------------------------------- word filters --
    word_pattern: str = r"^[A-Za-z]+$"   # single word, letters only
    min_len: int = 3
    max_len: int = 12
    exclude_capitalised: bool = True     # drop proper nouns (Abel, Africa, ...)

    # -------------------------------------------- avoiding accidental repetition --
    forbid_cue_as_target: bool = True    # declared in the original notebook; the
                                         # constraint is enforced by the uniqueness
                                         # check + per-list QC, not by a matrix mask
    min_edit_distinctness: int = 4       # reject target sharing a 4-letter prefix with its cue
    exclude_shared_translation: bool = True   # cue and target may not share a Chinese form

    # Fill M-E- from the M+E+ target pool instead of demanding globally unique
    # targets, so the critical cell and the baseline cell contain the SAME words.
    rematched: bool = True

    # ------------------------------------------ controlling for covariate distance --
    dist_cutoff: float = 1.5

    # The "largest mutually-close set of cues" is rarely unique. How to resolve it:
    #   union         - keep every cue appearing in ANY maximum clique; cues that are
    #                   not in all of them are flagged `tied` in design.csv and written
    #                   to `tied_cues_file` (the kept set is then a superset of a clique)
    #   tightest      - the maximum clique with the smallest mean pairwise distance
    #   lexicographic - the alphabetically first maximum clique
    #   first         - legacy max(nx.find_cliques(G), key=len); depends on the
    #                   networkx version, so not reproducible across environments
    clique_rule: str = "union"
    max_clique_enumeration: int = 2_000_000   # safety cap on the clique enumeration

    # -------------------------------------------------------------- validation --
    n_permutations: int = 1000
    run_permutation_test: bool = True

    # ---------------------------------------------------------- counterbalancing --
    # Manual review can leave blank cells (e.g. rows pasted from
    # matched_pairs_reordered.csv have no `str_zh`, because stage 1 writes that
    # quantity as `route_s_zh`). Refill them from the matrices before anything
    # downstream reads them. Only blanks are touched; existing values are kept.
    patch_reviewed_metrics: bool = True

    # Caps how many (cue, condition) slots any single word may occupy across the
    # whole 4-condition design. With 4 lists, a word at 5+ slots is guaranteed by
    # pigeonhole to collide with itself in some list; 3 keeps one list of margin.
    max_total_target_uses: int = 3
    n_anneal_iters: int = 200_000
    anneal_t0: float = 5.0

    # ------------------------------------------------------------------ misc --
    seed: int = 20260812
    chunk_rows: int = 4_000_000          # rows per chunk when streaming the huge RW files
    verbose: bool = True

    # ---------------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.stim_dir = Path(self.stim_dir).expanduser()
        self.data_dir = Path(self.data_dir).expanduser() if self.data_dir else self.stim_dir / "data"
        self.out_dir = Path(self.out_dir).expanduser() if self.out_dir else self.stim_dir / "output"
        self.cache_dir = Path(self.cache_dir).expanduser() if self.cache_dir else self.stim_dir / "cache"
        self.review_dir = (Path(self.review_dir).expanduser() if self.review_dir
                           else self.stim_dir / "manual_review")
        self.conditions = tuple(self.conditions)

    # ---- convenience views on the notebook's original names -------------------
    @property
    def min_count(self) -> Dict[str, int]:
        return {"en": self.min_count_en, "zh": self.min_count_zh}

    @property
    def min_strength(self) -> Dict[str, float]:
        return {"en": self.min_strength_en, "zh": self.min_strength_zh}

    @property
    def word_re(self) -> re.Pattern:
        return re.compile(self.word_pattern)

    @property
    def n_lists(self) -> int:
        return len(self.conditions)

    # ---- paths ----------------------------------------------------------------
    @property
    def path_align(self) -> Path:
        return self.data_dir / self.align_file

    @property
    def path_rw_en(self) -> Path:
        return self.data_dir / self.rw_en_file

    @property
    def path_rw_zh(self) -> Path:
        return self.data_dir / self.rw_zh_file

    @property
    def path_str_en(self) -> Path:
        return self.data_dir / self.strength_en_file

    @property
    def path_str_zh(self) -> Path:
        return self.data_dir / self.strength_zh_file

    @property
    def path_conc(self) -> Path:
        return self.data_dir / self.concreteness_file

    @property
    def path_freq(self) -> Path:
        return self.data_dir / self.frequency_file

    @property
    def path_reviewed(self) -> Path:
        return self.review_dir / self.reviewed_file

    def ensure_dirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- (de)serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, tuple):
                d[k] = list(v)
        return d

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

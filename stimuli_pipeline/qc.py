"""QC report on the selected design (written to `qc_report.txt`)."""

from __future__ import annotations

import pandas as pd

from .config import Config
from .utils import log


def qc_report(design: pd.DataFrame, cfg: Config, save: bool = True,
              clique=None) -> str:
    lines = []

    def A(s: str = "") -> None:
        lines.append(s)
        print(s)

    A("=" * 70)
    A("QC REPORT")
    A("=" * 70)
    A(f"cues {design.cue_en.nunique()} | rows {len(design)} | "
      f"distinct target words {design.target_en.nunique()}")
    A("")
    A("-- cell balance ---------------------------------------------------")
    A(design.groupby("condition")[["rw_en", "rw_zh, alignment weighted", "str_en",
                                   "str_zh, alignment weighted",
                                   "target_len"]].mean().round(4).to_string())
    A("")
    A("rw_en should be high in E+ and low in E-; rw_zh2en high in M+ and low in M-.")
    A("target_len should be similar across cells.")
    A("")
    A("-- overlap checks -------------------------------------------------")
    A(f"cue also used as a target        : "
      f"{len(set(design.cue_en) & set(design.target_en))}   (expect 0)")
    pp = set(design.loc[design.condition == 'M+E+', 'target_en'])
    mm = set(design.loc[design.condition == 'M-E-', 'target_en'])
    A(f"M+E+ and M-E- share a word set   : {pp == mm}   (expect True, by design)")
    A(f"max uses of any one target       : {design.target_en.value_counts().max()}"
      f"   (expect 2)")
    A("")
    if clique is not None:
        A("-- clique tie -----------------------------------------------------")
        A(f"maximum cliques of size {clique.size}     : {len(clique.maximum_cliques)}")
        A(f"cues in every maximum clique     : {len(clique.core)}")
        A(f"contested (tied) cues            : {len(clique.tied)} "
          f"{sorted(clique.tied) if clique.tied else ''}")
        A(f"rule '{clique.rule}' kept          : {len(clique.cues)} cues")
        A("")
    A("-- completeness ---------------------------------------------------")
    inc = design.groupby(['cue_en', 'condition']).size().unstack(fill_value=0)
    A(f"cues with an incomplete cell     : {int((inc != cfg.n_per_cell).any(axis=1).sum())}"
      f"   (expect 0)")
    A("")

    text = "\n".join(lines)
    if save:
        out = cfg.out_dir / cfg.qc_report_file
        out.write_text(text, encoding="utf-8")
        log(f"saved to {out}")
    return text

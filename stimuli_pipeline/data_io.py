"""Reading the raw norms: RW vocabularies, association strengths, RW matrices.

Everything expensive is cached under ``cfg.cache_dir`` exactly as in the notebook.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .utils import log


def rw_lex(path: Path, cache_name: str, cfg: Config) -> List[str]:
    """Return every word appearing in either column of a RW file."""
    cache = cfg.cache_dir / cache_name
    if cache.exists():
        return json.loads(cache.read_text())

    log(f"scanning vocabulary of {path.name} (slow, cached afterwards)")
    seen = set()
    if shutil.which("awk"):
        r = subprocess.run(
            ["awk", "-F,", "NR>1{a[$1];a[$2]} END{for(k in a) print k}", str(path)],
            capture_output=True, check=True, env={**os.environ, "LC_ALL": "C"})
        seen = set(r.stdout.decode("utf-8").split("\n"))
    else:
        for chunk in pd.read_csv(path, usecols=[0, 1], header=0, names=["a", "b"],
                                 dtype=str, chunksize=cfg.chunk_rows):
            seen.update(chunk["a"].unique()); seen.update(chunk["b"].unique())

    vocab = sorted(v for v in seen if isinstance(v, str) and v)
    cache.write_text(json.dumps(vocab, ensure_ascii=False))
    return vocab


def read_strength(path: Path, lang: str) -> pd.DataFrame:
    """Load a strength file as [cue, response, strength, count, n].

    English gives counts directly. Chinese does not, but the file was renormalised
    over cue-words only, so 1/min(strength) recovers the denominator exactly - a
    fact we verify rather than assume.
    """
    if lang == "en":
        d = pd.read_csv(path, sep="\t").rename(
            columns={"R123": "count", "R123.Strength": "strength", "N": "n"})
        d = d[["cue", "response", "strength", "count", "n"]]
    else:
        d = pd.read_csv(path).rename(columns={"R123.Strength": "strength"})
        n = (1.0 / d.groupby("cue")["strength"].min()).round()
        d["n"] = d["cue"].map(n)
        d["count"] = (d["strength"] * d["n"]).round()
        err = (d["strength"] * d["n"] - d["count"]).abs().max()
        log(f"  {lang}: recovered counts, largest rounding error {err:.1e}")

    log(f"  {lang}: {len(d):,} pairs, {d['cue'].nunique():,} cues, "
        f"median {d.groupby('cue')['n'].first().median():.0f} respondents/cue, "
        f"{(d['count'] == 1).mean():.0%} of pairs from a single respondent")
    return d


def load_rw(path: Path, nodes: Sequence[str], cache_name: str, cfg: Config) -> np.ndarray:
    """Dense symmetric RW matrix over `nodes`. NaN = pair not in the file.

    The cache key is a hash of the node list, so a changed lexicon transparently
    rebuilds (and the stale file is removed).
    """
    fp = hashlib.sha1("\n".join(nodes).encode()).hexdigest()[:12]
    cache = cfg.cache_dir / f"{cache_name}_{fp}.npy"
    if cache.exists():
        log(f"  loading cached {cache.name}")
        return np.load(cache)
    for stale in cfg.cache_dir.glob(f"{cache_name}_*.npy"):
        stale.unlink()

    idx = {w: i for i, w in enumerate(nodes)}
    m = len(nodes)
    M = np.full((m, m), np.nan, dtype=np.float32)
    log(f"  streaming {path.name} into a {m} x {m} matrix")

    src = path
    if shutil.which("awk"):                       # fast path: prefilter with awk
        keep = cfg.cache_dir / f"_keep_{cache_name}.txt"
        out = cfg.cache_dir / f"_filt_{cache_name}.csv"
        keep.write_text("\n".join(nodes), encoding="utf-8")
        tmp = out.with_suffix(".partial")         # write-then-rename, so a killed
        with tmp.open("w") as fh:                 # run cannot leave a truncated cache
            subprocess.run(["awk", "-F,",
                            "NR==FNR{k[$0];next} FNR==1{next} ($1 in k)&&($2 in k)",
                            str(keep), str(path)],
                           stdout=fh, check=True, env={**os.environ, "LC_ALL": "C"})
        tmp.replace(out)
        src = out

    for chunk in pd.read_csv(src, usecols=[0, 1, 2],
                             header=0 if src is path else None,
                             names=["a", "b", "rw"],
                             dtype={"a": str, "b": str, "rw": np.float32},
                             chunksize=cfg.chunk_rows):
        ia = chunk["a"].map(idx); ib = chunk["b"].map(idx)
        ok = ia.notna() & ib.notna()
        if ok.any():
            ia = ia[ok].to_numpy(np.int32); ib = ib[ok].to_numpy(np.int32)
            v = chunk.loc[ok, "rw"].to_numpy(np.float32)
            M[ia, ib] = v
            M[ib, ia] = v          # symmetrise: the file only stores one triangle
    np.fill_diagonal(M, np.nan)    # a word is never its own stimulus
    np.save(cache, M)
    if src is not path:
        Path(src).unlink(missing_ok=True)
    return M


def load_vocabularies(cfg: Config) -> Tuple[set, set, pd.DataFrame, pd.DataFrame]:
    """RW vocabularies intersected with the words that are cues in their own norm."""
    vocab_en = set(rw_lex(cfg.path_rw_en, "rw_lexicon_en.json", cfg))
    vocab_zh = set(rw_lex(cfg.path_rw_zh, "rw_lexicon_zh.json", cfg))
    log(f"RW vocabularies: English {len(vocab_en):,}, Chinese {len(vocab_zh):,}")

    str_en = read_strength(cfg.path_str_en, "en")
    str_zh = read_strength(cfg.path_str_zh, "zh")

    # A word must be a CUE in its own norm to be usable.
    vocab_en &= set(str_en["cue"])
    vocab_zh &= set(str_zh["cue"])
    log(f"usable after requiring cue status: English {len(vocab_en):,}, "
        f"Chinese {len(vocab_zh):,}")
    return vocab_en, vocab_zh, str_en, str_zh

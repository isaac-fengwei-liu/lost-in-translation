"""Small shared helpers."""

from __future__ import annotations

import time

import numpy as np

_VERBOSE = True


def set_verbose(flag: bool) -> None:
    global _VERBOSE
    _VERBOSE = bool(flag)


def log(msg: str) -> None:
    """Timestamped progress line (identical format to the notebooks)."""
    if _VERBOSE:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def argsort_desc(key: np.ndarray) -> np.ndarray:
    """Stable descending argsort.

    The notebook used ``np.argsort(key, descending=True)``, which only exists on
    newer NumPy. The fallback is the array-API definition of a stable descending
    sort (ties keep their original order), so the ranking is identical either way.
    """
    key = np.asarray(key)
    try:
        return np.argsort(key, descending=True)
    except TypeError:
        return np.argsort(-key, kind="stable")

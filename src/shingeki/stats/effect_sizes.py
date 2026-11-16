"""Effect-size utilities."""

from __future__ import annotations

import numpy as np


def cohens_dz(diffs: np.ndarray) -> float:
    """Cohen's d_z for paired data: mean(diffs) / std(diffs, ddof=1)."""
    diffs = np.asarray(diffs, dtype=float)
    s = diffs.std(ddof=1)
    if s == 0.0:
        return float("inf") if diffs.mean() != 0 else 0.0
    return float(diffs.mean() / s)


def fraction_positive(diffs: np.ndarray) -> float:
    """Fraction of paired differences > 0."""
    diffs = np.asarray(diffs, dtype=float)
    return float((diffs > 0).mean())

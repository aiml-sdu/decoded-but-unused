"""Paired bootstrap CI over triplets."""

from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) of the bootstrap distribution of the mean.

    `diffs` is a length-N array of paired differences (one per triplet)."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = diffs[idx].mean()
    lo = np.quantile(means, (1 - confidence) / 2)
    hi = np.quantile(means, 1 - (1 - confidence) / 2)
    return float(diffs.mean()), float(lo), float(hi)

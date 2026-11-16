"""Paired and grouped permutation tests."""

from __future__ import annotations

import numpy as np


def paired_permutation_p(
    a: np.ndarray, b: np.ndarray, n_perms: int = 10_000, seed: int = 42
) -> float:
    """Two-sided p-value for the null that mean(a) == mean(b) under random
    sign-flips of the paired differences. `a`, `b` must be length-N."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    obs = abs(diffs.mean())
    n = len(diffs)
    count = 0
    for _ in range(n_perms):
        signs = rng.choice([-1.0, 1.0], size=n)
        if abs((diffs * signs).mean()) >= obs:
            count += 1
    return (count + 1) / (n_perms + 1)


def grouped_permutation_test(
    metric_fn,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_perms: int = 1000,
    seed: int = 42,
) -> tuple[float, np.ndarray, float]:
    """Permute labels *within* the same group structure (i.e. shuffling group→
    label assignments, not within-group). Returns (observed, null distribution,
    p-value)."""
    rng = np.random.default_rng(seed)
    obs = float(metric_fn(X, y, groups))
    null = np.empty(n_perms)
    unique_groups = np.unique(groups)
    for i in range(n_perms):
        # Permute the per-group label set.
        perm = rng.permutation(len(unique_groups))
        group_to_perm = dict(zip(unique_groups, unique_groups[perm], strict=True))
        # Within each group, keep the same label distribution but reshuffle which
        # *group* gets which set of labels. The simpler operation: pick a random
        # permutation of `y` that respects group block structure.
        new_y = y.copy()
        for g_old, g_new in group_to_perm.items():
            new_y[groups == g_old] = y[groups == g_new]
        null[i] = float(metric_fn(X, new_y, groups))
    p = float(((null >= obs).sum() + 1) / (n_perms + 1))
    return obs, null, p

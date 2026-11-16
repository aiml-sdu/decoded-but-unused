"""Leave-one-triplet-out cross-validation splits."""

from __future__ import annotations

import numpy as np


def loto_splits(triplet_ids: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each fold leaves *all* framings of one triplet out for the test set."""
    triplet_ids = np.asarray(triplet_ids)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for tid in np.unique(triplet_ids):
        test_mask = triplet_ids == tid
        out.append((np.where(~test_mask)[0], np.where(test_mask)[0]))
    return out

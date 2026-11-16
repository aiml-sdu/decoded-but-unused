"""Probe-selectivity utilities (Hewitt & Liang 2019).

The control task takes the same input and assigns each *unique input* a fixed
random label drawn from the same label distribution. A probe trained on the
control task should perform near chance if the original probe captures genuine
linguistic structure rather than memorising probe capacity.
"""

from __future__ import annotations

import numpy as np


def make_control_labels(
    triplet_ids: np.ndarray, frames: np.ndarray, n_classes: int, seed: int = 42
) -> np.ndarray:
    """Assign a stable random label per *triplet_id* (so all three framings of
    the same dilemma share the same control label). This gives the probe the
    same group structure it sees in the real task."""
    rng = np.random.default_rng(seed)
    unique = np.unique(triplet_ids)
    mapping = {tid: int(rng.integers(0, n_classes)) for tid in unique}
    return np.array([mapping[tid] for tid in triplet_ids])


def selectivity(real_acc: float, control_acc: float) -> float:
    """Selectivity = real_acc - control_acc."""
    return float(real_acc - control_acc)

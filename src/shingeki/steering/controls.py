"""Steering and ablation controls (Makelov, Lange, Nanda 2024).

  - random-direction: norm-matched Gaussian in residual-stream space.
  - same-rank random subspace: random orthonormal k-frame.
  - random-label probe direction: train a probe on random group labels and
    use its weight contrast as the "direction".
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from shingeki.stats.selectivity import make_control_labels


def random_subspace(d_model: int, k: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(d_model, k, generator=g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(M)
    return Q.T  # [k, D]


def random_label_probe_direction(
    X_layer: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 42
) -> np.ndarray:
    n_classes = len(np.unique(y))
    y_rand = make_control_labels(groups, y, n_classes, seed=seed)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
    clf.fit(X_layer, y_rand)
    if len(clf.classes_) < 2:
        return np.zeros(X_layer.shape[1], dtype=np.float32)
    cls = list(clf.classes_)
    pos, neg = cls[0], cls[1]
    return (clf.coef_[cls.index(pos)] - clf.coef_[cls.index(neg)]).astype(np.float32)

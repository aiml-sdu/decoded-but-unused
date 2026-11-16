"""PCA helpers for residual-stream contrast directions."""

from __future__ import annotations

import numpy as np
import torch


def pca_top_k(matrix: torch.Tensor | np.ndarray, k: int) -> torch.Tensor:
    """Top-k singular vectors of a [N, D] matrix. Returns [k, D]."""
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)
    matrix = matrix.float()
    # center
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    # SVD; V is [D, D]; top-k right-singular vectors are V[:, :k].T
    _u, _s, vh = torch.linalg.svd(matrix, full_matrices=False)
    return vh[:k]

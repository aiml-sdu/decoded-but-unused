"""Steering-direction builders.

Eight kinds of direction:
  - mean       : mean(S) - mean(C) per layer
  - probe      : the trained probe's pos_class - neg_class weight vector
  - pca_top1   : first principal component of (S - C) differences
  - pca_topk   : top-k PCA subspace
  - random     : Gaussian, norm-matched to the reference direction
  - random_lbl : direction trained from random per-triplet labels (a probe)
  - judgment   : W_U[:, yes_id] - W_U[:, no_id]
  - ortho_fr_j : framing direction with the judgment component projected out
"""

from __future__ import annotations

import numpy as np
import torch

from shingeki.geometry.directions import judgment_direction
from shingeki.geometry.pca import pca_top_k
from shingeki.models.hooked import HookedModel
from shingeki.models.tokens import AnswerTokens


def mean_direction(acts_S_layer: torch.Tensor, acts_C_layer: torch.Tensor) -> torch.Tensor:
    return (acts_S_layer.mean(dim=0) - acts_C_layer.mean(dim=0)).float()


def random_norm_matched(reference: torch.Tensor, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(reference.shape, generator=g, dtype=torch.float32)
    return v * (reference.norm() / (v.norm() + 1e-8))


def orthogonalize(d_frame: torch.Tensor, d_judge: torch.Tensor) -> torch.Tensor:
    """Project d_judge out of d_frame."""
    j = d_judge.float()
    j_hat = j / (j.norm() + 1e-8)
    f = d_frame.float()
    return f - (f @ j_hat) * j_hat


def pca_direction(diffs_NxD: torch.Tensor | np.ndarray, k: int = 1) -> torch.Tensor:
    """First k principal components of N (S - C) layer-l difference vectors."""
    return pca_top_k(diffs_NxD, k)


__all__ = [
    "mean_direction",
    "random_norm_matched",
    "orthogonalize",
    "pca_direction",
    "judgment_direction",
]

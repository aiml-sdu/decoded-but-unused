"""Directions and layerwise alignment.

  - `framing_direction(acts_S, acts_C)` = mean(S) - mean(C) per layer.
  - `judgment_direction(model, ans)` = W_U[:, yes_id] - W_U[:, no_id] (TL conv).
  - `layerwise_alignment(d_frame, d_judge)` = cosine per layer.
"""

from __future__ import annotations

import torch

from shingeki.models.hooked import HookedModel
from shingeki.models.tokens import AnswerTokens


def framing_direction_per_layer(acts_sym: torch.Tensor, acts_cond: torch.Tensor) -> torch.Tensor:
    """Inputs are [N, L, D] tensors of per-triplet activations. Returns [L, D]."""
    return acts_sym.mean(dim=0) - acts_cond.mean(dim=0)


def judgment_direction(model: HookedModel, ans: AnswerTokens) -> torch.Tensor:
    """Returns a [d_model] direction in residual-stream space.
    Requires single-token answer pair."""
    if not ans.single_token:
        raise ValueError("judgment_direction requires single-token answer pair")
    W_U = model.W_U.detach().to(torch.float32)  # [D, V] in TL convention
    return (W_U[:, ans.pos_id] - W_U[:, ans.neg_id]).cpu()


def layerwise_alignment(d_frame: torch.Tensor, d_judge: torch.Tensor) -> torch.Tensor:
    """`d_frame` is [L, D]; `d_judge` is [D]. Returns [L] cosine."""
    a = d_frame.float()
    b = d_judge.float()
    num = a @ b
    den = a.norm(dim=-1) * b.norm() + 1e-8
    return num / den

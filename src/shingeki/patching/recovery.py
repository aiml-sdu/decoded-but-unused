"""Patching-recovery metric and KL/malformed-rate helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def recovery(patched_score: float, clean_score: float, corrupt_score: float, eps: float = 1e-8) -> float:
    """Fraction of the (clean - corrupt) gap that the patched run recovers.

    0.0 → no transfer, 1.0 → full transfer, negative → patch pushes the wrong way."""
    return float((patched_score - corrupt_score) / (clean_score - corrupt_score + eps))


def logits_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """KL(a || b) on the final-token softmax."""
    a = F.log_softmax(logits_a[0, -1].float(), dim=-1)
    b = F.log_softmax(logits_b[0, -1].float(), dim=-1)
    return float((a.exp() * (a - b)).sum().item())


def malformed_rate(top_token_id: int, allowed_ids: set[int]) -> int:
    """Helper: returns 1 if the argmax token is not in `allowed_ids`."""
    return int(top_token_id not in allowed_ids)

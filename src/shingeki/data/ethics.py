"""ETHICS (D4) appendix loader — stub.

ETHICS spans commonsense morality, deontology, justice, utilitarianism, and
virtue. For the workshop we only need a small behavioural sanity check, so this
stays a stub that the appendix script can flesh out by passing
`subset="commonsense"` etc.
"""

from __future__ import annotations


def load_d4(subset: str = "commonsense", n: int = 50, seed: int = 42) -> list[dict]:
    """Sample `n` items from `hendrycks/ethics`."""
    from datasets import load_dataset

    ds = load_dataset("hendrycks/ethics", subset, split="test")
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return list(ds)

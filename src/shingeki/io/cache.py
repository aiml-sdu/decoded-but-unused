"""On-disk activation / embedding cache.

Keyed on `(model_id, prompt_hash, hook_name, dtype, position_spec)`. Backed by
torch `.pt` files under `runs/.cache/`. Atomic writes (tmp → rename).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch

from shingeki.config import RUNS_DIR

CACHE_ROOT = RUNS_DIR / ".cache"


def _key(parts: tuple[Any, ...]) -> str:
    blob = "|".join(str(p) for p in parts).encode()
    return hashlib.sha1(blob).hexdigest()


class ActivationCache:
    def __init__(self, root: Path = CACHE_ROOT, enabled: bool = True):
        self.root = root
        self.enabled = enabled
        if enabled:
            root.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: Any) -> Path:
        return self.root / f"{_key(parts)}.pt"

    def get(self, *parts: Any) -> torch.Tensor | None:
        if not self.enabled:
            return None
        p = self.path(*parts)
        if not p.exists():
            return None
        return torch.load(p, map_location="cpu", weights_only=False)

    def put(self, value: torch.Tensor, *parts: Any) -> None:
        if not self.enabled:
            return
        p = self.path(*parts)
        tmp = p.with_suffix(".tmp")
        torch.save(value, tmp)
        os.replace(tmp, p)

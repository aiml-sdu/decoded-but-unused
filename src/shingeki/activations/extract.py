"""Activation extraction helpers.

Given a `HookedModel` and a `BuiltPrompt`, return tensors per layer at the
requested position. Position can be:

  - "final"          : the last token of the prompt
  - int              : a specific token index
  - "all"            : return the full sequence (B=1, T, D)

Hook names are TL-style.
"""

from __future__ import annotations

from typing import Literal

import torch

from shingeki.models.hooked import HookedModel
from shingeki.prompts.builders import BuiltPrompt

Position = Literal["final", "all"] | int


def _resolve_pos(pos: Position, T: int) -> int | slice:
    if pos == "final":
        return T - 1
    if pos == "all":
        return slice(None)
    return int(pos)


def _gather_layers(layers: list[int] | None, n_layers: int) -> list[int]:
    return list(range(n_layers)) if layers is None else list(layers)


@torch.inference_mode()
def get_residual_stream(
    model: HookedModel,
    prompt: BuiltPrompt,
    layers: list[int] | None = None,
    position: Position = "final",
    which: Literal["pre", "post"] = "post",
) -> torch.Tensor:
    """Return tensor [L, D] (position scalar) or [L, T, D] (position 'all')."""
    L = _gather_layers(layers, model.cfg.n_layers)
    suffix = "hook_resid_pre" if which == "pre" else "hook_resid_post"
    names = [f"blocks.{l}.{suffix}" for l in L]
    _logits, cache = model.run_with_cache(
        prompt.input_ids, names, attention_mask=prompt.attention_mask
    )
    out = []
    for n in names:
        t = cache[n][0]  # [T, D]
        idx = _resolve_pos(position, t.shape[0])
        out.append(t[idx] if isinstance(idx, int) else t)
    return torch.stack(out, dim=0)  # [L, D] or [L, T, D]


@torch.inference_mode()
def get_attn_out(
    model: HookedModel,
    prompt: BuiltPrompt,
    layers: list[int] | None = None,
    position: Position = "final",
) -> torch.Tensor:
    L = _gather_layers(layers, model.cfg.n_layers)
    names = [f"blocks.{l}.hook_attn_out" for l in L]
    _l, cache = model.run_with_cache(
        prompt.input_ids, names, attention_mask=prompt.attention_mask
    )
    out = []
    for n in names:
        t = cache[n][0]
        idx = _resolve_pos(position, t.shape[0])
        out.append(t[idx] if isinstance(idx, int) else t)
    return torch.stack(out, dim=0)


@torch.inference_mode()
def get_mlp_out(
    model: HookedModel,
    prompt: BuiltPrompt,
    layers: list[int] | None = None,
    position: Position = "final",
) -> torch.Tensor:
    L = _gather_layers(layers, model.cfg.n_layers)
    names = [f"blocks.{l}.hook_mlp_out" for l in L]
    _l, cache = model.run_with_cache(
        prompt.input_ids, names, attention_mask=prompt.attention_mask
    )
    out = []
    for n in names:
        t = cache[n][0]
        idx = _resolve_pos(position, t.shape[0])
        out.append(t[idx] if isinstance(idx, int) else t)
    return torch.stack(out, dim=0)


@torch.inference_mode()
def get_logits(model: HookedModel, prompt: BuiltPrompt, position: Position = "final") -> torch.Tensor:
    out = model(prompt.input_ids, prompt.attention_mask)
    if position == "all":
        return out[0]
    idx = _resolve_pos(position, out.shape[1])
    return out[0, idx] if isinstance(idx, int) else out[0]

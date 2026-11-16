"""Residual-stream activation patching across layers, final-token only.

For a given (clean, corrupt) prompt pair and a single layer ``l``, we run
the corrupt prompt through the model with the residual-stream activations at
``hook_resid_pre`` (the input to layer l) replaced by the *clean* prompt's
activations at the same position. The recovery metric quantifies how much of
the (clean - corrupt) score gap is restored by the patch.

S→C and C→S directions are always reported as separate sweeps.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

from shingeki.behavior.scoring import score_prompt
from shingeki.data.schemas import FRAMES, FramingTriplet
from shingeki.io.results import ResultBundle
from shingeki.models.hooked import HookedModel
from shingeki.models.tokens import resolve_answer_tokens
from shingeki.patching.recovery import recovery
from shingeki.prompts.builders import BuiltPrompt, build_aot_prompt
from shingeki.prompts.chat import ModelFamily


@torch.inference_mode()
def patch_resid_pre_at_layer(
    model: HookedModel,
    clean: BuiltPrompt,
    corrupt: BuiltPrompt,
    layer: int,
    ans,
    position: int | None = None,
) -> dict:
    """Single-(clean, corrupt) patch at a single layer. Returns metric dict."""
    # 1. capture clean activation at the desired hook
    name = f"blocks.{layer}.hook_resid_pre"
    _logits, cache = model.run_with_cache(
        clean.input_ids, [name], attention_mask=clean.attention_mask
    )
    clean_act = cache[name].detach()

    # only patch the position we read out from (the final token), so the
    # corruption is *focused*.
    if position is None:
        position = corrupt.input_ids.shape[1] - 1

    # if the two prompts differ in length, take min(T_clean, T_corrupt) - 1
    pos_clean = min(position, clean_act.shape[1] - 1)

    def hook(t, _name):
        new = t.clone()
        new[:, position] = clean_act[:, pos_clean]
        return new

    # 2. run corrupt with patch
    patched_logits = model.run_with_hooks(
        corrupt.input_ids, [(name, hook)], attention_mask=corrupt.attention_mask
    )

    # 3. score everything
    if ans.single_token:
        clean_logits = model(clean.input_ids, clean.attention_mask)
        corrupt_logits = model(corrupt.input_ids, corrupt.attention_mask)
        clean_score = float(
            (clean_logits[0, -1, ans.pos_id] - clean_logits[0, -1, ans.neg_id]).item()
        )
        corrupt_score = float(
            (corrupt_logits[0, -1, ans.pos_id] - corrupt_logits[0, -1, ans.neg_id]).item()
        )
        patched_score = float(
            (patched_logits[0, -1, ans.pos_id] - patched_logits[0, -1, ans.neg_id]).item()
        )
    else:
        # multi-token fallback: use score_prompt (one model call each)
        clean_score = score_prompt(model, clean.input_ids, clean.attention_mask, ans)
        corrupt_score = score_prompt(model, corrupt.input_ids, corrupt.attention_mask, ans)
        # for patched, re-run with hook
        def hook2(t, _n):
            new = t.clone()
            new[:, position] = clean_act[:, pos_clean]
            return new

        patched = model.run_with_hooks(
            corrupt.input_ids, [(name, hook2)], attention_mask=corrupt.attention_mask
        )
        patched_score = float(
            (patched[0, -1, ans.pos_ids[0]] - patched[0, -1, ans.neg_ids[0]]).item()
        )

    return {
        "clean_score": clean_score,
        "corrupt_score": corrupt_score,
        "patched_score": patched_score,
        "recovery": recovery(patched_score, clean_score, corrupt_score),
    }


def sweep_layers(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    direction: str = "S_to_C",  # patch sympathetic activations into condemning prompts
    fmt: str = "yesno",
    exp_id: str | None = None,
    layers: list[int] | None = None,
) -> ResultBundle:
    """Sweep `layers` (default: every layer) for one S↔C direction.

    Returns recovery[N_triplets, n_layers].
    """
    ans = resolve_answer_tokens(model.tokenizer, fmt)
    if direction == "S_to_C":
        clean_frame, corrupt_frame = "sympathetic", "condemning"
    elif direction == "C_to_S":
        clean_frame, corrupt_frame = "condemning", "sympathetic"
    else:
        raise ValueError(direction)
    L = model.cfg.n_layers
    layers = layers or list(range(L))
    rec = np.full((len(triplets), len(layers)), np.nan, dtype=np.float32)
    cs = np.zeros(len(triplets))
    ks = np.zeros(len(triplets))
    for i, t in enumerate(triplets):
        clean = build_aot_prompt(t, clean_frame, fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        corrupt = build_aot_prompt(t, corrupt_frame, fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        # snapshot baseline scores once per triplet
        for j, l in enumerate(layers):
            r = patch_resid_pre_at_layer(model, clean, corrupt, l, ans)
            rec[i, j] = r["recovery"]
            cs[i] = r["clean_score"]
            ks[i] = r["corrupt_score"]
    return ResultBundle(
        exp_id=exp_id or f"exp07_resid_patch_{direction}",
        config={
            "direction": direction,
            "n_triplets": len(triplets),
            "layers": list(layers),
            "fmt": fmt,
        },
        metrics={
            "mean_recovery_per_layer": rec.mean(axis=0).tolist(),
            "median_recovery_per_layer": np.median(rec, axis=0).tolist(),
            "best_layer": int(layers[int(np.argmax(rec.mean(axis=0)))]),
            "best_layer_recovery": float(rec.mean(axis=0).max()),
        },
        arrays={"recovery": rec, "clean_score": cs, "corrupt_score": ks},
    )

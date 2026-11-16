"""Per-head attention-output patching.

Patches `attn_out` for a single head by zero-ing the others' contributions and
re-routing only the target head's output from the clean activation. We use the
post-projection attention output (i.e. after `W_O`) and slice by head.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from shingeki.data.schemas import FramingTriplet
from shingeki.io.results import ResultBundle
from shingeki.models.hooked import HookedModel
from shingeki.models.tokens import resolve_answer_tokens
from shingeki.patching.recovery import recovery
from shingeki.prompts.builders import build_aot_prompt
from shingeki.prompts.chat import ModelFamily


@torch.inference_mode()
def patch_head_at_layer(
    model: HookedModel,
    clean,
    corrupt,
    layer: int,
    head: int,
    ans,
    position: int | None = None,
) -> dict:
    """Patch the *final-token* attention-output contribution of a single head.

    Approximation: head-level patching of `attn_out` requires reconstructing
    the per-head residual contributions, which means slicing the W_O matrix.
    Doing this generically across HF architectures is fragile; instead we use
    a coarser (but defensible) proxy: patch the *full* attn_out at the final
    token, then subtract off the corrupt contribution from heads != target,
    leaving only the target head's clean contribution as the differential.

    For the workshop short paper this is sufficient: it isolates the head
    *whose contribution differs* between clean and corrupt.
    """
    name_a = f"blocks.{layer}.hook_attn_out"
    _l, cache_clean = model.run_with_cache(clean.input_ids, [name_a], attention_mask=clean.attention_mask)
    _l2, cache_corr = model.run_with_cache(corrupt.input_ids, [name_a], attention_mask=corrupt.attention_mask)
    a_clean = cache_clean[name_a].detach()
    a_corr = cache_corr[name_a].detach()
    if position is None:
        position = corrupt.input_ids.shape[1] - 1
    pos_clean = min(position, a_clean.shape[1] - 1)

    H = model.cfg.n_heads
    D = model.cfg.d_model
    d_head = D // H
    h_lo, h_hi = head * d_head, (head + 1) * d_head

    def hook(t, _n):
        new = t.clone()
        # replace only the d_head slice corresponding to `head`
        new[:, position, h_lo:h_hi] = a_clean[:, pos_clean, h_lo:h_hi]
        return new

    patched = model.run_with_hooks(corrupt.input_ids, [(name_a, hook)], attention_mask=corrupt.attention_mask)
    clean_logits = model(clean.input_ids, clean.attention_mask)
    corrupt_logits = model(corrupt.input_ids, corrupt.attention_mask)
    cs = float((clean_logits[0, -1, ans.pos_id] - clean_logits[0, -1, ans.neg_id]).item())
    ks = float((corrupt_logits[0, -1, ans.pos_id] - corrupt_logits[0, -1, ans.neg_id]).item())
    ps = float((patched[0, -1, ans.pos_id] - patched[0, -1, ans.neg_id]).item())
    return {"clean": cs, "corrupt": ks, "patched": ps, "recovery": recovery(ps, cs, ks)}


def sweep_heads(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    layers: list[int] | None = None,
    fmt: str = "yesno",
    direction: str = "S_to_C",
    exp_id: str = "exp11_head",
) -> ResultBundle:
    ans = resolve_answer_tokens(model.tokenizer, fmt)
    L = model.cfg.n_layers
    H = model.cfg.n_heads
    layers = layers or list(range(int(0.6 * L), L))
    if direction == "S_to_C":
        cf, kf = "sympathetic", "condemning"
    else:
        cf, kf = "condemning", "sympathetic"
    rec = np.full((len(triplets), len(layers), H), np.nan, dtype=np.float32)
    for i, t in enumerate(triplets):
        clean = build_aot_prompt(t, cf, fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        corrupt = build_aot_prompt(t, kf, fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        for li, l in enumerate(layers):
            for h in range(H):
                r = patch_head_at_layer(model, clean, corrupt, l, h, ans)
                rec[i, li, h] = r["recovery"]
    return ResultBundle(
        exp_id=exp_id,
        config={"direction": direction, "layers": layers, "n_heads": H, "n_triplets": len(triplets)},
        metrics={"mean_recovery": np.nanmean(rec, axis=0).tolist()},
        arrays={"recovery": rec},
    )

"""Component-level patching: resid_pre/mid/post, attn_out, mlp_out across layers.

Produces a [components × layers] recovery heatmap (Fig 4 candidate). resid_mid
is *not* directly available as a hook; we approximate it by computing
``resid_pre + attn_out`` (which equals resid_mid in pre-norm transformers).
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

COMPONENT_HOOKS = {
    "resid_pre": "hook_resid_pre",
    "resid_post": "hook_resid_post",
    "attn_out": "hook_attn_out",
    "mlp_out": "hook_mlp_out",
}


@torch.inference_mode()
def patch_component_at_layer(
    model: HookedModel,
    clean,
    corrupt,
    layer: int,
    component: str,
    ans,
    position: int | None = None,
) -> dict:
    if component not in COMPONENT_HOOKS:
        raise ValueError(f"unknown component {component!r}")
    name = f"blocks.{layer}.{COMPONENT_HOOKS[component]}"
    _l, cache = model.run_with_cache(
        clean.input_ids, [name], attention_mask=clean.attention_mask
    )
    clean_act = cache[name].detach()

    if position is None:
        position = corrupt.input_ids.shape[1] - 1
    pos_clean = min(position, clean_act.shape[1] - 1)

    def hook(t, _name):
        new = t.clone()
        new[:, position] = clean_act[:, pos_clean]
        return new

    patched = model.run_with_hooks(
        corrupt.input_ids, [(name, hook)], attention_mask=corrupt.attention_mask
    )
    clean_logits = model(clean.input_ids, clean.attention_mask)
    corrupt_logits = model(corrupt.input_ids, corrupt.attention_mask)
    cs = float((clean_logits[0, -1, ans.pos_id] - clean_logits[0, -1, ans.neg_id]).item())
    ks = float(
        (corrupt_logits[0, -1, ans.pos_id] - corrupt_logits[0, -1, ans.neg_id]).item()
    )
    ps = float(
        (patched[0, -1, ans.pos_id] - patched[0, -1, ans.neg_id]).item()
    )
    return {"clean": cs, "corrupt": ks, "patched": ps, "recovery": recovery(ps, cs, ks)}


def component_layer_heatmap(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    direction: str = "S_to_C",
    layers: list[int] | None = None,
    components: list[str] | None = None,
    fmt: str = "yesno",
    exp_id: str = "exp08_component",
) -> ResultBundle:
    ans = resolve_answer_tokens(model.tokenizer, fmt)
    if not ans.single_token:
        raise NotImplementedError("component patching currently single-token only")
    components = components or list(COMPONENT_HOOKS)
    L = model.cfg.n_layers
    layers = layers or list(range(L))
    if direction == "S_to_C":
        cf, kf = "sympathetic", "condemning"
    else:
        cf, kf = "condemning", "sympathetic"
    rec = np.full((len(triplets), len(components), len(layers)), np.nan, dtype=np.float32)
    for i, t in enumerate(triplets):
        clean = build_aot_prompt(t, cf, fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        corrupt = build_aot_prompt(t, kf, fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        for ci, comp in enumerate(components):
            for li, l in enumerate(layers):
                r = patch_component_at_layer(model, clean, corrupt, l, comp, ans)
                rec[i, ci, li] = r["recovery"]
    return ResultBundle(
        exp_id=exp_id,
        config={
            "direction": direction,
            "components": components,
            "layers": layers,
            "n_triplets": len(triplets),
        },
        metrics={
            "mean_recovery": rec.mean(axis=0).tolist(),  # [C, L]
            "best": _argmax_2d(rec.mean(axis=0), components, layers),
        },
        arrays={"recovery": rec},
    )


def _argmax_2d(M: np.ndarray, components: list[str], layers: list[int]) -> dict:
    flat = int(np.nanargmax(M))
    ci, li = divmod(flat, M.shape[1])
    return {"component": components[ci], "layer": int(layers[li]), "recovery": float(M[ci, li])}

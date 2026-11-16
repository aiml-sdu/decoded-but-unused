"""Attention-source analysis (layers near the readout transition).

For each (triplet, frame) we capture per-layer attention probabilities to
prior tokens at the *final* query position. We then aggregate by token
category. This identifies which token roles the model attends to when
forming its answer, and which heads/layers are candidates for head-level
patching.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from shingeki.data.annotators import load_annotations
from shingeki.data.schemas import CATEGORIES, FramingTriplet
from shingeki.io.results import ResultBundle
from shingeki.models.hooked import HookedModel
from shingeki.patching.token_position import _annotation_for, _resolve_token_indices_in_chat
from shingeki.prompts.builders import build_aot_prompt
from shingeki.prompts.chat import ModelFamily


@torch.inference_mode()
def attention_to_categories_per_layer(
    model: HookedModel,
    prompt,
    layers: list[int],
    cat_to_idxs: dict[str, list[int]],
) -> np.ndarray:
    """Returns array [len(layers), n_heads, n_categories]: mean attention from
    the final token to each category's tokens, at each (layer, head)."""
    if model.backend == "tl":
        names = [f"blocks.{l}.attn.hook_pattern" for l in layers]
        _logits, cache = model.run_with_cache(prompt.input_ids, names)
    else:
        # HF backend: pass `output_attentions=True` and consume the resulting
        # tensor list. Many modern HF models default to flash/sdpa attention,
        # which does not materialise the attention probabilities; we ask
        # eager attention only for this call.
        out = model.model(
            input_ids=prompt.input_ids,
            attention_mask=prompt.attention_mask,
            output_attentions=True,
        )
        attentions = getattr(out, "attentions", None)
        if attentions is None or len(attentions) == 0:
            raise RuntimeError(
                "model returned no attention probabilities; "
                "load with attn_implementation='eager' to enable Exp 10"
            )
        # Map decoder-layer index -> attention tensor. With multimodal Gemma-3,
        # `attentions` covers only the language-model layers (decoder layers),
        # which is what we want.
        cache = {f"blocks.{l}.attn.hook_pattern": attentions[l] for l in layers if l < len(attentions)}

    final_idx = prompt.input_ids.shape[1] - 1
    n_heads = list(cache.values())[0].shape[1]
    cats = list(CATEGORIES)
    res = np.zeros((len(layers), n_heads, len(cats)), dtype=np.float32)
    for li, l in enumerate(layers):
        att = cache[f"blocks.{l}.attn.hook_pattern"][0]  # [H, T_q, T_k]
        att_q = att[:, final_idx]  # [H, T_k]
        for ci, cat in enumerate(cats):
            idxs = cat_to_idxs.get(cat, [])
            if not idxs:
                continue
            res[li, :, ci] = att_q[:, idxs].sum(dim=-1).float().cpu().numpy()
    return res


def attention_source_bundle(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    layers: list[int] | None = None,
    fmt: str = "yesno",
    exp_id: str = "exp10_attn_source",
) -> ResultBundle:
    annotations = load_annotations()
    L = model.cfg.n_layers
    layers = layers or list(range(int(0.6 * L), L))
    arrs: dict[str, np.ndarray] = {}
    for frame in ("sympathetic", "condemning"):
        per_triplet = []
        for t in triplets:
            ann = _annotation_for(annotations, t.id, frame)
            if ann is None:
                continue
            p = build_aot_prompt(t, frame, fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
            cat_to_idxs = _resolve_token_indices_in_chat(p, ann, model.tokenizer)
            per_triplet.append(
                attention_to_categories_per_layer(model, p, layers, cat_to_idxs)
            )
        if per_triplet:
            arrs[frame] = np.stack(per_triplet)  # [N, L, H, C]
    metrics: dict = {}
    if "sympathetic" in arrs and "condemning" in arrs:
        delta = arrs["sympathetic"].mean(0) - arrs["condemning"].mean(0)  # [L, H, C]
        metrics["S_minus_C_attn_to_categories"] = delta.tolist()
        # rank top heads by max-abs delta over categories
        per_head_score = np.max(np.abs(delta), axis=-1)  # [L, H]
        top_flat = np.argsort(per_head_score.flatten())[-10:][::-1]
        top = [(int(idx // per_head_score.shape[1]), int(idx % per_head_score.shape[1])) for idx in top_flat]
        metrics["top10_layer_head"] = top
    return ResultBundle(
        exp_id=exp_id,
        config={"layers": layers, "fmt": fmt, "n_triplets": len(triplets)},
        metrics=metrics,
        arrays={k: v for k, v in arrs.items()},
    )

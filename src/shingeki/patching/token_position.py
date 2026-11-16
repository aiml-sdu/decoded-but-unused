"""Token-position patching using LLM-annotated category spans.

For each token category (actor / victim / harm / justification / group_id /
action), patch the *clean* activations at *every token in that category* into
the corrupt run, sweeping layers. The output heatmap is [Categories × Layers].

Requires `data/token_categories/aot_token_spans.jsonl` to be populated by
`shingeki data annotate-tokens`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from shingeki.data.annotators import load_annotations
from shingeki.data.schemas import CATEGORIES, Category, FramingTriplet, TokenSpanAnnotation
from shingeki.io.results import ResultBundle
from shingeki.models.hooked import HookedModel
from shingeki.models.tokens import resolve_answer_tokens
from shingeki.patching.recovery import recovery
from shingeki.prompts.builders import BuiltPrompt, build_aot_prompt
from shingeki.prompts.chat import ModelFamily


def _annotation_for(annotations: list[TokenSpanAnnotation], triplet_id: str, frame: str) -> TokenSpanAnnotation | None:
    for a in annotations:
        if a.triplet_id == triplet_id and a.frame == frame:
            return a
    return None


def _resolve_token_indices_in_chat(
    prompt: BuiltPrompt, ann: TokenSpanAnnotation, tokenizer
) -> dict[Category, list[int]]:
    """The annotator gave token spans relative to the *scenario only*. The
    chat-formatted prompt has additional prefix/suffix tokens. Convert by
    locating the scenario char span in the chat text and re-aligning."""
    if prompt.scenario_char_span is None:
        return {c: [] for c in CATEGORIES}
    scen_char_start, scen_char_end = prompt.scenario_char_span
    enc = tokenizer(prompt.chat_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    # find the token index window that covers the scenario substring
    scen_token_start = next(
        (i for i, (a, _b) in enumerate(offsets) if a >= scen_char_start), None
    )
    if scen_token_start is None:
        return {c: [] for c in CATEGORIES}
    out: dict[Category, list[int]] = {c: [] for c in CATEGORIES}
    for cat, spans in ann.token_spans.items():
        for s, e in spans:
            for tok in range(s, e):
                idx = scen_token_start + tok
                if 0 <= idx < prompt.input_ids.shape[1]:
                    out[cat].append(idx)
    return out


@torch.inference_mode()
def patch_category_at_layer(
    model: HookedModel,
    clean: BuiltPrompt,
    corrupt: BuiltPrompt,
    layer: int,
    clean_idxs: list[int],
    corrupt_idxs: list[int],
    ans,
) -> dict:
    """Replace the residual-stream at all `corrupt_idxs` with the clean
    activations at the union of `clean_idxs`. If the position counts differ,
    reuse cyclically (the simple approach, since spans rarely match exactly).
    """
    name = f"blocks.{layer}.hook_resid_pre"
    _l, cache = model.run_with_cache(clean.input_ids, [name], attention_mask=clean.attention_mask)
    clean_act = cache[name].detach()

    def hook(t, _n):
        new = t.clone()
        if not corrupt_idxs or not clean_idxs:
            return new
        for j, ci in enumerate(corrupt_idxs):
            src = clean_idxs[j % len(clean_idxs)]
            if src < clean_act.shape[1] and ci < new.shape[1]:
                new[:, ci] = clean_act[:, src]
        return new

    patched = model.run_with_hooks(corrupt.input_ids, [(name, hook)], attention_mask=corrupt.attention_mask)
    clean_logits = model(clean.input_ids, clean.attention_mask)
    corrupt_logits = model(corrupt.input_ids, corrupt.attention_mask)
    cs = float((clean_logits[0, -1, ans.pos_id] - clean_logits[0, -1, ans.neg_id]).item())
    ks = float((corrupt_logits[0, -1, ans.pos_id] - corrupt_logits[0, -1, ans.neg_id]).item())
    ps = float((patched[0, -1, ans.pos_id] - patched[0, -1, ans.neg_id]).item())
    return {"clean": cs, "corrupt": ks, "patched": ps, "recovery": recovery(ps, cs, ks)}


def token_position_heatmap(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    direction: str = "S_to_C",
    layers: list[int] | None = None,
    fmt: str = "yesno",
    exp_id: str = "exp09_token_position",
) -> ResultBundle:
    ans = resolve_answer_tokens(model.tokenizer, fmt)
    annotations = load_annotations()
    L = model.cfg.n_layers
    layers = layers or list(range(L))
    if direction == "S_to_C":
        cf, kf = "sympathetic", "condemning"
    else:
        cf, kf = "condemning", "sympathetic"
    cats = list(CATEGORIES)
    rec = np.full((len(triplets), len(cats), len(layers)), np.nan, dtype=np.float32)
    for i, t in enumerate(triplets):
        clean = build_aot_prompt(t, cf, fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        corrupt = build_aot_prompt(t, kf, fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        clean_ann = _annotation_for(annotations, t.id, cf)
        corrupt_ann = _annotation_for(annotations, t.id, kf)
        if clean_ann is None or corrupt_ann is None:
            continue
        clean_idx_by_cat = _resolve_token_indices_in_chat(clean, clean_ann, model.tokenizer)
        corrupt_idx_by_cat = _resolve_token_indices_in_chat(corrupt, corrupt_ann, model.tokenizer)
        for ci, cat in enumerate(cats):
            cl_idxs = clean_idx_by_cat.get(cat, [])
            ck_idxs = corrupt_idx_by_cat.get(cat, [])
            if not cl_idxs or not ck_idxs:
                continue
            for li, l in enumerate(layers):
                r = patch_category_at_layer(model, clean, corrupt, l, cl_idxs, ck_idxs, ans)
                rec[i, ci, li] = r["recovery"]
    return ResultBundle(
        exp_id=exp_id,
        config={
            "direction": direction,
            "categories": cats,
            "layers": layers,
            "n_triplets": len(triplets),
        },
        metrics={
            "mean_recovery": np.nanmean(rec, axis=0).tolist(),
        },
        arrays={"recovery": rec},
    )

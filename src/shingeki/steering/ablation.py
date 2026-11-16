"""Amnesic-style projection ablation (Elazar et al. 2021).

Train a probe; remove its weight subspace from intermediate activations during
the forward pass; measure whether the final Yes/No gap shrinks.

Controls:
  - random rank-k subspace (resampled per layer, norm-matched)
  - length-control direction (the regression coefficient on prompt length)
  - random-label probe direction (expects no shrinkage)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from shingeki.data.schemas import FramingTriplet
from shingeki.io.results import ResultBundle
from shingeki.models.hooked import HookedModel
from shingeki.models.tokens import resolve_answer_tokens
from shingeki.prompts.builders import build_aot_prompt
from shingeki.prompts.chat import ModelFamily
from shingeki.steering.interventions import projection_removal_hook, subspace_removal_hook


@torch.inference_mode()
def projection_ablation(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    layer: int,
    direction: torch.Tensor,
    fmt: str = "yesno",
) -> dict:
    ans = resolve_answer_tokens(model.tokenizer, fmt)
    name, hook = projection_removal_hook(layer, direction, position="final")
    sym, cond = [], []
    sym_abl, cond_abl = [], []
    for t in triplets:
        ps = build_aot_prompt(t, "sympathetic", fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        pc = build_aot_prompt(t, "condemning", fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        ls = model(ps.input_ids, ps.attention_mask)
        lc = model(pc.input_ids, pc.attention_mask)
        ls_a = model.run_with_hooks(ps.input_ids, [(name, hook)], attention_mask=ps.attention_mask)
        lc_a = model.run_with_hooks(pc.input_ids, [(name, hook)], attention_mask=pc.attention_mask)
        sym.append(float(ls[0, -1, ans.pos_id] - ls[0, -1, ans.neg_id]))
        cond.append(float(lc[0, -1, ans.pos_id] - lc[0, -1, ans.neg_id]))
        sym_abl.append(float(ls_a[0, -1, ans.pos_id] - ls_a[0, -1, ans.neg_id]))
        cond_abl.append(float(lc_a[0, -1, ans.pos_id] - lc_a[0, -1, ans.neg_id]))
    sc = np.array(sym) - np.array(cond)
    sca = np.array(sym_abl) - np.array(cond_abl)
    return {
        "S_C_baseline": float(sc.mean()),
        "S_C_ablated": float(sca.mean()),
        "shrink_frac": float(1 - sca.mean() / (sc.mean() + 1e-8)),
        "raw_baseline": sc.tolist(),
        "raw_ablated": sca.tolist(),
    }


@torch.inference_mode()
def subspace_ablation(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    layer: int,
    basis: torch.Tensor,
    fmt: str = "yesno",
) -> dict:
    ans = resolve_answer_tokens(model.tokenizer, fmt)
    name, hook = subspace_removal_hook(layer, basis, position="final")
    sym, cond, sym_a, cond_a = [], [], [], []
    for t in triplets:
        ps = build_aot_prompt(t, "sympathetic", fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        pc = build_aot_prompt(t, "condemning", fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        ls = model(ps.input_ids, ps.attention_mask)
        lc = model(pc.input_ids, pc.attention_mask)
        ls_a = model.run_with_hooks(ps.input_ids, [(name, hook)], attention_mask=ps.attention_mask)
        lc_a = model.run_with_hooks(pc.input_ids, [(name, hook)], attention_mask=pc.attention_mask)
        sym.append(float(ls[0, -1, ans.pos_id] - ls[0, -1, ans.neg_id]))
        cond.append(float(lc[0, -1, ans.pos_id] - lc[0, -1, ans.neg_id]))
        sym_a.append(float(ls_a[0, -1, ans.pos_id] - ls_a[0, -1, ans.neg_id]))
        cond_a.append(float(lc_a[0, -1, ans.pos_id] - lc_a[0, -1, ans.neg_id]))
    sc = np.array(sym) - np.array(cond)
    sca = np.array(sym_a) - np.array(cond_a)
    return {
        "S_C_baseline": float(sc.mean()),
        "S_C_ablated": float(sca.mean()),
        "shrink_frac": float(1 - sca.mean() / (sc.mean() + 1e-8)),
    }

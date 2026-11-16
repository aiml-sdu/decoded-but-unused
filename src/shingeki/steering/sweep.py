"""Top-level steering sweep: 8 directions × 3 interventions × controls.

Used for Exp 13 (steering beyond the mean).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from shingeki.activations.extract import get_residual_stream
from shingeki.data.schemas import FRAMES, FramingTriplet
from shingeki.io.results import ResultBundle
from shingeki.models.hooked import HookedModel
from shingeki.models.tokens import resolve_answer_tokens
from shingeki.prompts.builders import build_aot_prompt
from shingeki.prompts.chat import ModelFamily
from shingeki.steering.controls import random_subspace
from shingeki.steering.directions import (
    judgment_direction,
    mean_direction,
    orthogonalize,
    pca_direction,
    random_norm_matched,
)
from shingeki.steering.interventions import (
    addition_hook,
    projection_removal_hook,
    subspace_removal_hook,
)


def _collect_layer_acts(
    model: HookedModel, triplets: Sequence[FramingTriplet], family: ModelFamily, layer: int, fmt: str = "yesno"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (S, C) tensors of shape [N, D] for one layer."""
    s_list, c_list = [], []
    for t in triplets:
        ps = build_aot_prompt(t, "sympathetic", fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        pc = build_aot_prompt(t, "condemning", fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        with torch.inference_mode():
            s = get_residual_stream(model, ps, layers=[layer], position="final", which="post")[0]
            c = get_residual_stream(model, pc, layers=[layer], position="final", which="post")[0]
        s_list.append(s.float().cpu())
        c_list.append(c.float().cpu())
    return torch.stack(s_list), torch.stack(c_list)


@torch.inference_mode()
def measure_intervention(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    intervention,
    fmt: str = "yesno",
) -> dict:
    """Apply a (name, hook) intervention to each prompt; return paired metrics.

    The metric reported is recovery toward the *opposite frame*: i.e. when we
    apply the intervention to the condemning prompt, does its score move
    toward the sympathetic baseline?
    """
    ans = resolve_answer_tokens(model.tokenizer, fmt)
    name, hook = intervention
    sym, cond, sym_p, cond_p = [], [], [], []
    for t in triplets:
        ps = build_aot_prompt(t, "sympathetic", fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        pc = build_aot_prompt(t, "condemning", fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        ls = model(ps.input_ids, ps.attention_mask)
        lc = model(pc.input_ids, pc.attention_mask)
        ls_p = model.run_with_hooks(ps.input_ids, [(name, hook)], attention_mask=ps.attention_mask)
        lc_p = model.run_with_hooks(pc.input_ids, [(name, hook)], attention_mask=pc.attention_mask)
        sym.append(float(ls[0, -1, ans.pos_id] - ls[0, -1, ans.neg_id]))
        cond.append(float(lc[0, -1, ans.pos_id] - lc[0, -1, ans.neg_id]))
        sym_p.append(float(ls_p[0, -1, ans.pos_id] - ls_p[0, -1, ans.neg_id]))
        cond_p.append(float(lc_p[0, -1, ans.pos_id] - lc_p[0, -1, ans.neg_id]))
    sym = np.array(sym); cond = np.array(cond)
    sym_p = np.array(sym_p); cond_p = np.array(cond_p)
    sc_baseline = (sym - cond).mean()
    sc_after = (sym_p - cond_p).mean()
    return {
        "S_C_baseline_mean": float(sc_baseline),
        "S_C_after_mean": float(sc_after),
        "shrink_frac": float(1 - sc_after / (sc_baseline + 1e-8)),
        "sym_shift": float((sym_p - sym).mean()),
        "cond_shift": float((cond_p - cond).mean()),
    }


def steering_sweep_bundle(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    layer: int,
    alpha: float = 4.0,
    fmt: str = "yesno",
    seed: int = 42,
    exp_id: str = "exp13_steering",
    pca_ks: tuple[int, ...] = (1, 2, 4, 8),
) -> ResultBundle:
    ans = resolve_answer_tokens(model.tokenizer, fmt)
    S, C = _collect_layer_acts(model, triplets, family, layer, fmt)  # [N, D]
    d_mean = mean_direction(S, C)
    d_judge = judgment_direction(model, ans).cpu()
    d_random = random_norm_matched(d_mean, seed=seed)
    d_ortho = orthogonalize(d_mean, d_judge)
    diffs = (S - C).numpy()
    d_pca1 = pca_direction(diffs, k=1)[0]
    pca_topk = {k: pca_direction(diffs, k=k) for k in pca_ks}
    rs_subspace = random_subspace(model.cfg.d_model, max(pca_ks), seed=seed)

    results: dict[str, dict] = {}
    # --- addition (use clear single-direction labels)
    for label, d in [
        ("mean", d_mean),
        ("pca_top1", d_pca1),
        ("random_norm_matched", d_random),
        ("judgment", d_judge),
        ("ortho_frame", d_ortho),
    ]:
        results[f"add[{label}]"] = measure_intervention(
            model, triplets, family, addition_hook(layer, d, alpha), fmt=fmt
        )
    # --- projection removal
    for label, d in [("mean", d_mean), ("pca_top1", d_pca1), ("random", d_random), ("judgment", d_judge)]:
        results[f"proj_rm[{label}]"] = measure_intervention(
            model, triplets, family, projection_removal_hook(layer, d), fmt=fmt
        )
    # --- subspace removal (PCA-k vs random-k)
    for k in pca_ks:
        results[f"subspace_rm[pca_top{k}]"] = measure_intervention(
            model, triplets, family, subspace_removal_hook(layer, pca_topk[k]), fmt=fmt
        )
        results[f"subspace_rm[random_top{k}]"] = measure_intervention(
            model, triplets, family, subspace_removal_hook(layer, rs_subspace[:k]), fmt=fmt
        )

    return ResultBundle(
        exp_id=exp_id,
        config={
            "layer": layer,
            "alpha": alpha,
            "fmt": fmt,
            "pca_ks": list(pca_ks),
            "n_triplets": len(triplets),
        },
        metrics=results,
    )

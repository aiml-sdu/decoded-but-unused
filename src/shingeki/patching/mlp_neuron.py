"""MLP-neuron localization.

For a given layer, rank neurons by their |S - C| activation difference at the
final token, then ablate the top-k (zero them) and measure the recovery toward
the corrupt baseline. Compare against random-top-k controls.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from shingeki.activations.extract import get_mlp_out
from shingeki.data.schemas import FramingTriplet
from shingeki.io.results import ResultBundle
from shingeki.models.hooked import HookedModel
from shingeki.models.tokens import resolve_answer_tokens
from shingeki.prompts.builders import build_aot_prompt
from shingeki.prompts.chat import ModelFamily


def rank_mlp_neurons_by_sc_delta(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    layer: int,
    fmt: str = "yesno",
) -> np.ndarray:
    """Returns an array of neuron indices sorted by descending |S - C|."""
    diffs = []
    for t in triplets:
        ps = build_aot_prompt(t, "sympathetic", fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        pc = build_aot_prompt(t, "condemning", fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        with torch.inference_mode():
            s = get_mlp_out(model, ps, layers=[layer], position="final")[0].float().cpu().numpy()
            c = get_mlp_out(model, pc, layers=[layer], position="final")[0].float().cpu().numpy()
        diffs.append(s - c)
    diffs = np.stack(diffs)
    abs_mean = np.abs(diffs.mean(axis=0))
    # `[::-1]` gives a view with negative stride; copy so torch indexing works.
    return np.ascontiguousarray(np.argsort(abs_mean)[::-1])


@torch.inference_mode()
def ablate_top_k_neurons(
    model: HookedModel,
    triplet,
    family,
    layer: int,
    neuron_ids: np.ndarray,
    ans,
    fmt: str = "yesno",
) -> dict:
    """Zero the listed neurons in `mlp_out` at all positions; score sympathetic
    prompt; report the new logit gap (relative to corrupt baseline = condemning
    prompt with no ablation)."""
    p_sym = build_aot_prompt(triplet, "sympathetic", fmt, family, model.tokenizer, model.device)
    p_con = build_aot_prompt(triplet, "condemning", fmt, family, model.tokenizer, model.device)

    name = f"blocks.{layer}.hook_mlp_out"

    neuron_idx_t = torch.from_numpy(np.asarray(neuron_ids, dtype=np.int64).copy())

    def hook(t, _n):
        new = t.clone()
        new[..., neuron_idx_t.to(new.device)] = 0.0
        return new

    abl_logits = model.run_with_hooks(
        p_sym.input_ids, [(name, hook)], attention_mask=p_sym.attention_mask
    )
    sym_logits = model(p_sym.input_ids, p_sym.attention_mask)
    con_logits = model(p_con.input_ids, p_con.attention_mask)
    a = float((abl_logits[0, -1, ans.pos_id] - abl_logits[0, -1, ans.neg_id]).item())
    s = float((sym_logits[0, -1, ans.pos_id] - sym_logits[0, -1, ans.neg_id]).item())
    c = float((con_logits[0, -1, ans.pos_id] - con_logits[0, -1, ans.neg_id]).item())
    return {"sym_score": s, "ablated_score": a, "cond_score": c, "shrink_frac": (s - a) / (s - c + 1e-8)}


def neuron_localization_bundle(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    layer: int,
    k_values: tuple[int, ...] = (16, 64, 256),
    fmt: str = "yesno",
    exp_id: str = "exp12_mlp_neuron",
    seed: int = 42,
) -> ResultBundle:
    ans = resolve_answer_tokens(model.tokenizer, fmt)
    rng = np.random.default_rng(seed)
    ranking = rank_mlp_neurons_by_sc_delta(model, triplets, family, layer, fmt=fmt)
    D = ranking.shape[0]
    metrics: dict = {}
    arrays: dict = {"ranking": ranking}
    for k in k_values:
        top = ranking[:k]
        rand = rng.choice(D, size=k, replace=False)
        top_shrink = []
        rand_shrink = []
        for t in triplets:
            top_shrink.append(ablate_top_k_neurons(model, t, family, layer, top, ans)["shrink_frac"])
            rand_shrink.append(ablate_top_k_neurons(model, t, family, layer, rand, ans)["shrink_frac"])
        metrics[str(k)] = {
            "top_k_mean_shrink": float(np.mean(top_shrink)),
            "random_k_mean_shrink": float(np.mean(rand_shrink)),
        }
        arrays[f"top_shrink_k{k}"] = np.array(top_shrink)
        arrays[f"rand_shrink_k{k}"] = np.array(rand_shrink)
    return ResultBundle(
        exp_id=exp_id, config={"layer": layer, "k_values": list(k_values)}, metrics=metrics, arrays=arrays
    )

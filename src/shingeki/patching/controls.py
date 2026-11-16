"""Patching controls.

  - random clean source: use a *different* triplet's sympathetic activation as
    the clean source for the corrupt prompt of triplet i.
  - same-framing patch: clean = corrupt = corrupt-frame from a different
    triplet (should yield ~0 recovery; non-zero implies leakage).
  - random-layer / random-token patch: same as residual patching but with the
    layer or token position permuted.
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
def random_clean_source_control(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    layer: int,
    fmt: str = "yesno",
    seed: int = 42,
    exp_id: str = "exp_ctl_random_clean_source",
) -> ResultBundle:
    """For each triplet i, patch resid_pre at layer `layer` from triplet
    perm(i)'s sympathetic prompt into triplet i's condemning prompt."""
    ans = resolve_answer_tokens(model.tokenizer, fmt)
    rng = np.random.default_rng(seed)
    n = len(triplets)
    perm = rng.permutation(n)
    rec = np.full(n, np.nan, dtype=np.float32)
    name = f"blocks.{layer}.hook_resid_pre"
    for i, t in enumerate(triplets):
        donor = triplets[int(perm[i])]
        clean = build_aot_prompt(donor, "sympathetic", fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        corrupt = build_aot_prompt(t, "condemning", fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        _l, cache = model.run_with_cache(clean.input_ids, [name], attention_mask=clean.attention_mask)
        clean_act = cache[name].detach()
        position = corrupt.input_ids.shape[1] - 1
        pos_clean = min(position, clean_act.shape[1] - 1)

        def hook(tt, _n):
            new = tt.clone()
            new[:, position] = clean_act[:, pos_clean]
            return new

        patched = model.run_with_hooks(corrupt.input_ids, [(name, hook)], attention_mask=corrupt.attention_mask)
        # the "clean score" baseline here is the donor's clean — but we report
        # recovery toward the donor, since cross-triplet patching may be ill-defined.
        donor_logits = model(clean.input_ids, clean.attention_mask)
        corr_logits = model(corrupt.input_ids, corrupt.attention_mask)
        cs = float((donor_logits[0, -1, ans.pos_id] - donor_logits[0, -1, ans.neg_id]).item())
        ks = float((corr_logits[0, -1, ans.pos_id] - corr_logits[0, -1, ans.neg_id]).item())
        ps = float((patched[0, -1, ans.pos_id] - patched[0, -1, ans.neg_id]).item())
        rec[i] = recovery(ps, cs, ks)
    return ResultBundle(
        exp_id=exp_id,
        config={"layer": layer, "n_triplets": n, "seed": seed},
        metrics={"mean_recovery": float(np.nanmean(rec)), "median_recovery": float(np.nanmedian(rec))},
        arrays={"recovery": rec},
    )

"""Off-target check for projection ablation (gap E).

If projecting out the framing subspace at L=28 just *destroys Yes/No (or A/B)
confidence in general*, the "removes framing causally" claim is wrong. We
check this by running the same projection-removal hook on prompts where
framing should not matter:

  - MoralChoice low-ambiguity items, no narrative prefix.
  - Each item has a clearly correct A/B answer in the dataset.

Metrics:
  - mean confidence |gap| before vs after ablation
  - accuracy before vs after (does the model still pick the right A/B?)
  - shrink_frac of the *correct-answer logit margin*

Verdict: if the off-target |gap| is largely preserved (e.g. <30% shrink), the
intervention is reasonably specific. If both shrink ~equally, the on-target
shrink is just confidence destruction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from shingeki.data.schemas import MoralChoiceItem
from shingeki.models.hooked import HookedModel
from shingeki.models.tokens import resolve_answer_tokens
from shingeki.prompts.builders import build_moralchoice_prompt
from shingeki.prompts.chat import ModelFamily
from shingeki.steering.interventions import projection_removal_hook, subspace_removal_hook


@torch.inference_mode()
def offtarget_projection(
    model: HookedModel,
    items: Sequence[MoralChoiceItem],
    family: ModelFamily,
    layer: int,
    direction: torch.Tensor,
    *,
    label_field: str = "correct",
) -> dict:
    """Run projection-removal on neutral MoralChoice prompts; report drift.

    The MoralChoice items in our pipeline are framed (sym/cond/neutral); for an
    off-target check we want *only* the neutral-frame items so the prompt has
    no sympathetic / condemning prefix.
    """
    ans = resolve_answer_tokens(model.tokenizer, "ab")
    name, hook = projection_removal_hook(layer, direction, position="final")

    base_gaps: list[float] = []
    abl_gaps: list[float] = []
    base_correct: list[int] = []
    abl_correct: list[int] = []
    n_skipped = 0
    for it in items:
        # only use neutral-prefix items for off-target
        if it.frame != "neutral":
            n_skipped += 1
            continue
        p = build_moralchoice_prompt(it, family, model.tokenizer, model.device)
        l_base = model(p.input_ids, p.attention_mask)
        l_abl = model.run_with_hooks(p.input_ids, [(name, hook)], attention_mask=p.attention_mask)
        gap_base = float(l_base[0, -1, ans.pos_id] - l_base[0, -1, ans.neg_id])
        gap_abl = float(l_abl[0, -1, ans.pos_id] - l_abl[0, -1, ans.neg_id])
        base_gaps.append(gap_base)
        abl_gaps.append(gap_abl)
        # If the dataset records a correct answer, track accuracy.
        correct = getattr(it, label_field, None)
        if correct in ("A", "B"):
            want_pos = correct == "A"
            base_correct.append(int((gap_base > 0) == want_pos))
            abl_correct.append(int((gap_abl > 0) == want_pos))

    if not base_gaps:
        return {"error": "no neutral-frame items found", "n_skipped": int(n_skipped)}

    base = np.array(base_gaps)
    abl = np.array(abl_gaps)
    out: dict = {
        "n": int(base.size),
        "n_skipped_non_neutral": int(n_skipped),
        "mean_abs_gap_base": float(np.mean(np.abs(base))),
        "mean_abs_gap_ablated": float(np.mean(np.abs(abl))),
        "abs_gap_shrink": float(1 - np.mean(np.abs(abl)) / (np.mean(np.abs(base)) + 1e-8)),
        "raw_base": base.tolist(),
        "raw_ablated": abl.tolist(),
    }
    if base_correct:
        out["accuracy_base"] = float(np.mean(base_correct))
        out["accuracy_ablated"] = float(np.mean(abl_correct))
        out["accuracy_drop"] = float(out["accuracy_base"] - out["accuracy_ablated"])
    return out


def write_offtarget_audit(
    out_dir: Path,
    model: HookedModel,
    items: Sequence[MoralChoiceItem],
    family: ModelFamily,
    layer: int,
    direction: torch.Tensor,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = offtarget_projection(model, items, family, layer=layer, direction=direction)
    (out_dir / "result.json").write_text(json.dumps(res, indent=2, default=str))
    return out_dir

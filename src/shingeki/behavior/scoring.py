"""Behavioural scoring: Yes/No logit gap, A/B logit gap, paired stats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from shingeki.config import BOOTSTRAP_RESAMPLES, PERMUTATION_SAMPLES, SEED
from shingeki.data.schemas import Frame, FramingTriplet, MoralChoiceItem
from shingeki.io.results import ResultBundle
from shingeki.models.hooked import HookedModel
from shingeki.models.tokens import (
    AnswerTokens,
    resolve_answer_tokens,
    sequence_logprob,
)
from shingeki.prompts.builders import build_aot_prompt, build_moralchoice_prompt
from shingeki.prompts.chat import ModelFamily
from shingeki.prompts.templates import AnswerFormat
from shingeki.stats.bootstrap import paired_bootstrap_ci
from shingeki.stats.effect_sizes import cohens_dz, fraction_positive
from shingeki.stats.permutation import paired_permutation_p


@dataclass
class TripletScore:
    triplet_id: str
    sympathetic: float
    condemning: float
    abstract: float

    @property
    def s_minus_c(self) -> float:
        return self.sympathetic - self.condemning


@torch.inference_mode()
def score_prompt(
    model: HookedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    ans: AnswerTokens,
) -> float:
    """Return the answer-pair logit gap (positive favours `ans.pos_str`)."""
    if ans.single_token:
        logits = model(input_ids, attention_mask)
        fl = logits[0, -1].float()
        return float(fl[ans.pos_id] - fl[ans.neg_id])
    # multi-token fallback: difference of sequence log-probs
    pos = sequence_logprob(model.model, model.tokenizer, input_ids[0], ans.pos_ids)
    neg = sequence_logprob(model.model, model.tokenizer, input_ids[0], ans.neg_ids)
    return pos - neg


def score_aot(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    fmt: AnswerFormat = "yesno",
) -> list[TripletScore]:
    ans = resolve_answer_tokens(model.tokenizer, fmt)
    scores: list[TripletScore] = []
    for t in triplets:
        per_frame = {}
        for frame in ("sympathetic", "condemning", "abstract"):
            p = build_aot_prompt(t, frame, fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
            per_frame[frame] = score_prompt(model, p.input_ids, p.attention_mask, ans)
        scores.append(
            TripletScore(
                triplet_id=t.id,
                sympathetic=per_frame["sympathetic"],
                condemning=per_frame["condemning"],
                abstract=per_frame["abstract"],
            )
        )
    return scores


def paired_shift(
    sympathetic: Iterable[float],
    condemning: Iterable[float],
    seed: int = SEED,
) -> dict:
    s = np.asarray(list(sympathetic), dtype=float)
    c = np.asarray(list(condemning), dtype=float)
    diffs = s - c
    mean, lo, hi = paired_bootstrap_ci(diffs, BOOTSTRAP_RESAMPLES, seed=seed)
    return {
        "n_triplets": int(len(diffs)),
        "mean_shift": float(diffs.mean()),
        "median_shift": float(np.median(diffs)),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "frac_sym_gt_cond": fraction_positive(diffs),
        "perm_p": paired_permutation_p(s, c, PERMUTATION_SAMPLES, seed=seed),
        "cohens_dz": cohens_dz(diffs),
        "shifts": diffs.tolist(),
    }


def behavioral_validation_bundle(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    fmt: AnswerFormat = "yesno",
    exp_id: str = "exp01_behavior",
) -> ResultBundle:
    scores = score_aot(model, triplets, family, fmt)
    sym = [s.sympathetic for s in scores]
    cond = [s.condemning for s in scores]
    abst = [s.abstract for s in scores]
    s_vs_c = paired_shift(sym, cond)
    s_vs_a = paired_shift(sym, abst)
    c_vs_a = paired_shift(cond, abst)
    return ResultBundle(
        exp_id=exp_id,
        config={
            "model_id": _model_id_of(model),
            "fmt": fmt,
            "n_triplets": len(scores),
            "answer_format_single_token": resolve_answer_tokens(
                model.tokenizer, fmt
            ).single_token,
        },
        metrics={
            "S_vs_C": s_vs_c,
            "S_vs_A": s_vs_a,
            "C_vs_A": c_vs_a,
            "scores_per_triplet": [
                {
                    "id": s.triplet_id,
                    "sympathetic": s.sympathetic,
                    "condemning": s.condemning,
                    "abstract": s.abstract,
                }
                for s in scores
            ],
        },
        arrays={
            "sym": np.array(sym),
            "cond": np.array(cond),
            "abst": np.array(abst),
        },
    )


def _model_id_of(model: HookedModel) -> str:
    if model.backend == "tl":
        return getattr(model._tl.cfg, "model_name", "tl-model")  # type: ignore[union-attr]
    return getattr(model.model.config, "_name_or_path", "hf-model")


def score_moralchoice(
    model: HookedModel,
    items: Sequence[MoralChoiceItem],
    family: ModelFamily,
) -> list[dict]:
    """Score MoralChoice-Frame items with the A/B logit gap."""
    ans = resolve_answer_tokens(model.tokenizer, "ab")
    rows = []
    for it in items:
        p = build_moralchoice_prompt(it, family, model.tokenizer, model.device)
        gap = score_prompt(model, p.input_ids, p.attention_mask, ans)
        rows.append(
            {
                "id": it.id,
                "frame": it.frame,
                "ambiguity": it.ambiguity,
                "ab_gap": gap,
            }
        )
    return rows

"""Answer-token robustness sweep across the four AnswerFormats."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from shingeki.behavior.scoring import score_aot, paired_shift
from shingeki.data.schemas import FramingTriplet
from shingeki.io.results import ResultBundle
from shingeki.models.hooked import HookedModel
from shingeki.prompts.chat import ModelFamily
from shingeki.prompts.templates import ANSWER_FORMATS


def robustness_bundle(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    formats: tuple[str, ...] = ANSWER_FORMATS,
    exp_id: str = "exp02_robustness",
) -> ResultBundle:
    metrics: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    for fmt in formats:
        scores = score_aot(model, triplets, family, fmt)  # type: ignore[arg-type]
        sym = np.array([s.sympathetic for s in scores])
        cond = np.array([s.condemning for s in scores])
        metrics[fmt] = paired_shift(sym, cond)
        arrays[f"{fmt}_sym"] = sym
        arrays[f"{fmt}_cond"] = cond
    return ResultBundle(
        exp_id=exp_id,
        config={"formats": list(formats), "n_triplets": len(triplets)},
        metrics={"per_format_S_vs_C": metrics},
        arrays=arrays,
    )

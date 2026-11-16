"""Linear probes with leave-one-triplet-out CV.

The probe is a multinomial logistic regression. Inputs are [N_prompts, D]
hidden states at a single layer; labels are encoded framings. Groups are
triplet ids - every fold leaves out *all three framings* of one dilemma so
the probe is never tested on any token sequence it was trained on.

The fitted probe's weight vector for the (sympathetic, condemning) contrast
is recovered as `probe_direction(probe)` and is used as a steering target in
Tier 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from shingeki.activations.extract import get_residual_stream
from shingeki.data.schemas import FRAMES, FramingTriplet
from shingeki.io.results import ResultBundle
from shingeki.models.hooked import HookedModel
from shingeki.probing.splits import loto_splits
from shingeki.prompts.builders import build_aot_prompt
from shingeki.prompts.chat import ModelFamily


@dataclass
class ProbeResult:
    layer: int
    accuracy: float
    macro_f1: float
    per_fold: list[float]
    weights: np.ndarray
    intercept: np.ndarray
    classes: list[str]


def cache_activations(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    fmt: str = "yesno",
    pool: str = "final",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X[N, L, D], y[N], groups[N]).

    `pool='final'` keeps the last-token activations; `pool='mean'` averages
    across the (masked) prompt.
    """
    L = model.cfg.n_layers
    D = model.cfg.d_model
    X = np.zeros((len(triplets) * 3, L, D), dtype=np.float32)
    y = np.empty(len(triplets) * 3, dtype="<U16")
    groups = np.empty(len(triplets) * 3, dtype="<U32")
    label_idx = {f: i for i, f in enumerate(FRAMES)}
    row = 0
    for t in triplets:
        for f in FRAMES:
            p = build_aot_prompt(t, f, fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
            with torch.inference_mode():
                full = get_residual_stream(model, p, position="all", which="post")  # [L, T, D]
            if pool == "final":
                v = full[:, -1]
            elif pool == "mean":
                mask = p.attention_mask[0].bool()
                v = full[:, mask].mean(dim=1)
            else:
                raise ValueError(pool)
            X[row] = v.detach().to("cpu", torch.float32).numpy()
            y[row] = f
            groups[row] = t.id
            row += 1
    return X, y, groups


def fit_probe_at_layer(
    X_layer: np.ndarray, y: np.ndarray, groups: np.ndarray, c: float = 1.0, seed: int = 42
) -> ProbeResult:
    splits = loto_splits(groups)
    per_fold = []
    preds_all = np.empty_like(y)
    for tr, te in splits:
        clf = LogisticRegression(
            C=c,
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
        clf.fit(X_layer[tr], y[tr])
        pred = clf.predict(X_layer[te])
        per_fold.append(accuracy_score(y[te], pred))
        preds_all[te] = pred
    # Re-fit on all data to extract a stable weight vector.
    clf_all = LogisticRegression(
        C=c, max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=seed
    )
    clf_all.fit(X_layer, y)
    return ProbeResult(
        layer=-1,  # filled in by caller
        accuracy=float(accuracy_score(y, preds_all)),
        macro_f1=float(f1_score(y, preds_all, average="macro")),
        per_fold=[float(a) for a in per_fold],
        weights=clf_all.coef_.astype(np.float32),
        intercept=clf_all.intercept_.astype(np.float32),
        classes=list(clf_all.classes_),
    )


def fit_probes_all_layers(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    c: float = 1.0,
    seed: int = 42,
    layers: list[int] | None = None,
) -> list[ProbeResult]:
    L = X.shape[1]
    layers = layers or list(range(L))
    out: list[ProbeResult] = []
    for l in layers:
        r = fit_probe_at_layer(X[:, l], y, groups, c=c, seed=seed)
        r.layer = int(l)
        out.append(r)
    return out


def probe_direction(probe: ProbeResult, pos_class: str = "sympathetic", neg_class: str = "condemning") -> np.ndarray:
    """Return the (D,) contrast vector probe.weights[pos] - probe.weights[neg]."""
    cls = list(probe.classes)
    return probe.weights[cls.index(pos_class)] - probe.weights[cls.index(neg_class)]

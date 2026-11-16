"""Cosine-similarity baseline by layer.

Computes layerwise cosine similarity between framings using both:

  - mean-pooled hidden states across the prompt
  - final-token hidden states

Output: arrays of shape [N_triplets, n_layers] for each pair (S↔C, S↔A, C↔A).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from shingeki.activations.extract import get_residual_stream
from shingeki.data.schemas import FRAMES, FramingTriplet
from shingeki.io.results import ResultBundle
from shingeki.models.hooked import HookedModel
from shingeki.prompts.builders import build_aot_prompt
from shingeki.prompts.chat import ModelFamily


@torch.inference_mode()
def _per_triplet_layer_acts(
    model: HookedModel,
    triplet: FramingTriplet,
    family: ModelFamily,
    fmt: str,
    pool: str,
) -> dict[str, torch.Tensor]:
    """Returns {frame: [L, D]} pooled across token positions."""
    out: dict[str, torch.Tensor] = {}
    for frame in FRAMES:
        p = build_aot_prompt(triplet, frame, fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        full = get_residual_stream(
            model, p, layers=None, position="all", which="post"
        )  # [L, T, D]
        if pool == "final":
            v = full[:, -1]  # [L, D]
        elif pool == "mean":
            mask = p.attention_mask[0].bool()
            v = full[:, mask].mean(dim=1)  # [L, D]
        else:
            raise ValueError(pool)
        out[frame] = v.detach().to("cpu", torch.float32)
    return out


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def cosine_bundle(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    fmt: str = "yesno",
    exp_id: str = "exp03_cosine",
) -> ResultBundle:
    L = model.cfg.n_layers
    sc_final = np.full((len(triplets), L), np.nan)
    sa_final = np.full_like(sc_final, np.nan)
    ca_final = np.full_like(sc_final, np.nan)
    sc_mean = np.full_like(sc_final, np.nan)
    sa_mean = np.full_like(sc_final, np.nan)
    ca_mean = np.full_like(sc_final, np.nan)
    for i, t in enumerate(triplets):
        f = _per_triplet_layer_acts(model, t, family, fmt, pool="final")
        m = _per_triplet_layer_acts(model, t, family, fmt, pool="mean")
        sc_final[i] = _cos(f["sympathetic"], f["condemning"]).numpy()
        sa_final[i] = _cos(f["sympathetic"], f["abstract"]).numpy()
        ca_final[i] = _cos(f["condemning"], f["abstract"]).numpy()
        sc_mean[i] = _cos(m["sympathetic"], m["condemning"]).numpy()
        sa_mean[i] = _cos(m["sympathetic"], m["abstract"]).numpy()
        ca_mean[i] = _cos(m["condemning"], m["abstract"]).numpy()
    return ResultBundle(
        exp_id=exp_id,
        config={"fmt": fmt, "n_triplets": len(triplets), "n_layers": L},
        metrics={
            "final_token": {
                "S_vs_C_mean_per_layer": sc_final.mean(axis=0).tolist(),
                "S_vs_A_mean_per_layer": sa_final.mean(axis=0).tolist(),
                "C_vs_A_mean_per_layer": ca_final.mean(axis=0).tolist(),
            },
            "mean_pooled": {
                "S_vs_C_mean_per_layer": sc_mean.mean(axis=0).tolist(),
                "S_vs_A_mean_per_layer": sa_mean.mean(axis=0).tolist(),
                "C_vs_A_mean_per_layer": ca_mean.mean(axis=0).tolist(),
            },
        },
        arrays={
            "final_sc": sc_final,
            "final_sa": sa_final,
            "final_ca": ca_final,
            "mean_sc": sc_mean,
            "mean_sa": sa_mean,
            "mean_ca": ca_mean,
        },
    )

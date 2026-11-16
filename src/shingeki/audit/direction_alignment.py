"""PT vs IT framing/judgment direction alignment (gap D).

Hypothesis: instruction tuning does not *create* the moral-framing
representation; it *aligns* that representation with the judgment readout. To
test this we compare two same-architecture checkpoints (Gemma-3-4B-PT vs
Gemma-3-4B-IT) on:

  1. cosine similarity between the PT framing direction and the IT framing
     direction at every layer (does IT preserve the PT direction?)
  2. cos(d_frame, d_judge) per layer for each model (does only IT show late
     alignment?)
  3. cosine between PT d_judge and IT d_judge (sanity)

Inputs are the per-layer activations and the answer-token unembeds we already
compute under exp06_geometry. We can re-derive d_judge in-memory from any
loaded model. d_frame is computed from saved residuals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from shingeki.activations.extract import get_residual_stream
from shingeki.data.schemas import FramingTriplet
from shingeki.geometry.directions import (
    framing_direction_per_layer,
    judgment_direction,
    layerwise_alignment,
)
from shingeki.models.hooked import HookedModel
from shingeki.models.tokens import resolve_answer_tokens
from shingeki.prompts.builders import build_aot_prompt
from shingeki.prompts.chat import ModelFamily


def collect_frame_acts(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    fmt: str = "yesno",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sym_acts, cond_acts), each [N, L, D] at the final-token position."""
    L = model.cfg.n_layers
    D = model.cfg.d_model
    sym = np.zeros((len(triplets), L, D), dtype=np.float32)
    cond = np.zeros((len(triplets), L, D), dtype=np.float32)
    for i, t in enumerate(triplets):
        ps = build_aot_prompt(t, "sympathetic", fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        pc = build_aot_prompt(t, "condemning", fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        with torch.inference_mode():
            sym[i] = get_residual_stream(model, ps, position="final", which="post").float().cpu().numpy()
            cond[i] = get_residual_stream(model, pc, position="final", which="post").float().cpu().numpy()
    return sym, cond


def model_direction_block(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    fmt: str = "yesno",
) -> dict:
    """For one model, compute d_frame (per layer), d_judge, layerwise alignment."""
    sym, cond = collect_frame_acts(model, triplets, family, fmt=fmt)
    d_frame = framing_direction_per_layer(torch.from_numpy(sym), torch.from_numpy(cond))  # [L, D]
    ans = resolve_answer_tokens(model.tokenizer, fmt)
    d_judge = judgment_direction(model, ans).cpu()  # [D]
    align = layerwise_alignment(d_frame, d_judge).numpy()
    return {
        "d_frame": d_frame.numpy(),
        "d_judge": d_judge.numpy(),
        "alignment_per_layer": align,
        "n_layers": int(d_frame.shape[0]),
        "d_model": int(d_frame.shape[1]),
    }


def cosine_per_layer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Layerwise cosine between [L, D] arrays of the same shape."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    return num / den


def compute_pt_it_alignment(pt_block: dict, it_block: dict) -> dict:
    """Combine two model blocks (must have matching layer count and dim)."""
    if pt_block["n_layers"] != it_block["n_layers"]:
        return {"error": f"layer mismatch PT={pt_block['n_layers']} IT={it_block['n_layers']}"}
    if pt_block["d_model"] != it_block["d_model"]:
        return {"error": f"d_model mismatch PT={pt_block['d_model']} IT={it_block['d_model']}"}

    frame_cos = cosine_per_layer(pt_block["d_frame"], it_block["d_frame"])
    # judge directions are vectors, not per-layer
    judge_cos = float(
        np.dot(pt_block["d_judge"], it_block["d_judge"])
        / (np.linalg.norm(pt_block["d_judge"]) * np.linalg.norm(it_block["d_judge"]) + 1e-8)
    )
    return {
        "n_layers": int(pt_block["n_layers"]),
        "frame_cos_per_layer": frame_cos.tolist(),
        "frame_cos_mean": float(frame_cos.mean()),
        "frame_cos_late_mean": float(frame_cos[len(frame_cos) // 2 :].mean()),
        "judge_judge_cos": judge_cos,
        "pt_alignment_per_layer": pt_block["alignment_per_layer"].tolist(),
        "it_alignment_per_layer": it_block["alignment_per_layer"].tolist(),
        "pt_peak_alignment": float(np.max(np.abs(pt_block["alignment_per_layer"]))),
        "it_peak_alignment": float(np.max(np.abs(it_block["alignment_per_layer"]))),
    }


def write_alignment_audit(out_dir: Path, pt_block: dict, it_block: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = compute_pt_it_alignment(pt_block, it_block)
    (out_dir / "result.json").write_text(json.dumps(res, indent=2, default=str))
    if "error" not in res:
        # save raw direction arrays for downstream plots
        np.savez_compressed(
            out_dir / "arrays.npz",
            pt_d_frame=pt_block["d_frame"],
            it_d_frame=it_block["d_frame"],
            pt_d_judge=pt_block["d_judge"],
            it_d_judge=it_block["d_judge"],
            pt_alignment=pt_block["alignment_per_layer"],
            it_alignment=it_block["alignment_per_layer"],
        )
    return out_dir

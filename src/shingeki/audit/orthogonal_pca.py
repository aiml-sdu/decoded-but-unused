"""Orthogonal-PCA projection audit (concern #4).

The original audit B compared rank-4 PCA-of-(sym - cond) at L=28 against
random rank-4 controls. But PCA-of-(sym - cond) at a deep layer can capture
the *judgment* direction itself (since sym/cond differ in their judgment as
well as their framing). This audit isolates the framing subspace by:

  1. Compute d_judge (the answer-token unembedding direction for "Yes" - "No").
  2. Compute PCA top-K of (sym - cond) residuals at the target layer.
  3. Project each PCA component to be ORTHOGONAL to d_judge.
  4. Re-orthonormalize the resulting basis (so it remains a valid subspace).
  5. Run subspace_ablation with the resulting orthogonal-to-d_judge basis.

If shrink_frac remains ~0.95 (matching the un-orthogonalized PCA result), the
framing subspace is *causally separable* from the judgment direction. If it
crashes toward 0, the v1 effect was really "we removed the judgment readout."

Also runs a paired control: ablate ONLY d_judge (rank-1 along the judgment
direction). This bounds how much the judgment direction alone can shrink the
gap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from shingeki.activations.extract import get_residual_stream
from shingeki.data.schemas import FramingTriplet
from shingeki.geometry.directions import judgment_direction
from shingeki.models.hooked import HookedModel
from shingeki.models.tokens import resolve_answer_tokens
from shingeki.prompts.builders import build_aot_prompt
from shingeki.prompts.chat import ModelFamily
from shingeki.steering.ablation import projection_ablation, subspace_ablation


def _orthogonalize(basis: np.ndarray, against: np.ndarray) -> np.ndarray:
    """Remove the `against` component from each row of `basis`, then re-orthonormalize.

    basis: [k, D]
    against: [D]
    returns: [k, D] orthonormal-ish, with each row orthogonal to `against`.
    """
    a = against / (np.linalg.norm(against) + 1e-8)
    proj = basis @ a  # [k]
    basis_perp = basis - np.outer(proj, a)  # [k, D]
    # re-orthonormalize via QR
    q, _ = np.linalg.qr(basis_perp.T)  # [D, <=k]
    out = q.T  # [<=k, D]
    return out.astype(basis.dtype)


@torch.inference_mode()
def collect_frame_residuals(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    layer: int,
    fmt: str = "yesno",
) -> tuple[np.ndarray, np.ndarray]:
    sym = np.zeros((len(triplets), model.cfg.d_model), dtype=np.float32)
    cond = np.zeros((len(triplets), model.cfg.d_model), dtype=np.float32)
    for i, t in enumerate(triplets):
        ps = build_aot_prompt(t, "sympathetic", fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        pc = build_aot_prompt(t, "condemning", fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
        rs = get_residual_stream(model, ps, position="final", which="post")  # [L, D]
        rc = get_residual_stream(model, pc, position="final", which="post")
        sym[i] = rs[layer].float().cpu().numpy()
        cond[i] = rc[layer].float().cpu().numpy()
    return sym, cond


def orthogonal_pca_audit(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    layers: tuple[int, ...] = (20, 25, 28),
    rank: int = 4,
    fmt: str = "yesno",
) -> dict:
    """Run full audit at each layer in `layers`."""
    ans = resolve_answer_tokens(model.tokenizer, fmt)
    d_judge = judgment_direction(model, ans).cpu().float().numpy()  # [D]

    out: dict = {"layers": list(layers), "rank": int(rank), "d_judge_norm": float(np.linalg.norm(d_judge))}
    per_layer: dict = {}
    for layer in layers:
        sym, cond = collect_frame_residuals(model, triplets, family, layer, fmt=fmt)
        diffs = sym - cond  # [N, D]
        diffs_centered = diffs - diffs.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
        pca_basis = Vt[:rank].astype(np.float32)  # [k, D]

        # 1. baseline: rank-1 along d_judge alone
        d_judge_t = torch.from_numpy(d_judge).float()
        only_judge = projection_ablation(
            model, triplets, family, layer=layer, direction=d_judge_t, fmt=fmt
        )

        # 2. baseline: rank-k PCA WITHOUT orthogonalization (matches original audit B)
        pca_basis_t = torch.from_numpy(pca_basis).float()
        pca_full = subspace_ablation(model, triplets, family, layer=layer, basis=pca_basis_t, fmt=fmt)

        # 3. main audit: rank-k PCA WITH d_judge orthogonalized out
        pca_perp = _orthogonalize(pca_basis, d_judge)
        pca_perp_t = torch.from_numpy(pca_perp).float()
        pca_perp_res = subspace_ablation(
            model, triplets, family, layer=layer, basis=pca_perp_t, fmt=fmt
        )

        # how much of pca_basis was the judgment direction?
        proj_norms = np.linalg.norm(pca_basis @ (d_judge / (np.linalg.norm(d_judge) + 1e-8)), keepdims=True)

        per_layer[str(layer)] = {
            "judge_only_rank1": {
                "S_C_baseline": only_judge["S_C_baseline"],
                "S_C_ablated": only_judge["S_C_ablated"],
                "shrink_frac": only_judge["shrink_frac"],
            },
            "pca_full_rank4": {
                "S_C_baseline": pca_full["S_C_baseline"],
                "S_C_ablated": pca_full["S_C_ablated"],
                "shrink_frac": pca_full["shrink_frac"],
            },
            "pca_perp_d_judge_rank4": {
                "S_C_baseline": pca_perp_res["S_C_baseline"],
                "S_C_ablated": pca_perp_res["S_C_ablated"],
                "shrink_frac": pca_perp_res["shrink_frac"],
                "rank_after_orthogonalization": int(pca_perp.shape[0]),
            },
            "pca_to_d_judge_projection_norm_per_component": proj_norms.flatten().tolist(),
        }
    out["per_layer"] = per_layer
    return out


def write_orthogonal_pca_audit(
    out_dir: Path,
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    layers: tuple[int, ...] = (20, 25, 28),
    rank: int = 4,
    fmt: str = "yesno",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = orthogonal_pca_audit(model, triplets, family, layers=layers, rank=rank, fmt=fmt)
    (out_dir / "result.json").write_text(json.dumps(res, indent=2, default=str))

    # Verdict CSV
    lines = [
        "layer,judge_only_shrink,pca_full_shrink,pca_perp_shrink,verdict_separable_from_judgment"
    ]
    for layer_str, sub in res["per_layer"].items():
        ju = sub["judge_only_rank1"]["shrink_frac"]
        pf = sub["pca_full_rank4"]["shrink_frac"]
        pp = sub["pca_perp_d_judge_rank4"]["shrink_frac"]
        # If pca_perp >= 80% of pca_full, the framing subspace is mostly orthogonal to d_judge
        verdict = "SEPARABLE" if abs(pp) >= 0.8 * abs(pf) and abs(pp) >= 0.5 else "FUSED-WITH-JUDGE"
        lines.append(f"{layer_str},{ju:+.4f},{pf:+.4f},{pp:+.4f},{verdict}")
    (out_dir / "verdict.csv").write_text("\n".join(lines) + "\n")
    return out_dir

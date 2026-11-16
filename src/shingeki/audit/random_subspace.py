"""Random same-rank subspace controls for projection ablation (gap B).

The current paper claims a rank-4 PCA-removal at L=28 closes ~96% of the
sympathetic-condemning gap, vs. ~1% for random rank-4 subspaces. That number
should be supported by a *distribution*, not a single random seed:

  - Run K independent random rank-r orthonormal subspaces, norm-matched to the
    layer's residual norm.
  - Run subspace_ablation for each.
  - Report mean / median / 95% CI / max of the random shrink_frac distribution.
  - Compare against the *true* shrink (probe direction or PCA top-k).

We also include controls that are *not* purely random but should still fail to
shrink the gap meaningfully:
  - random-label probe direction (probe trained against scrambled labels)
  - length-only direction (regression coefficient on prompt length)

The off-target check is in `offtarget.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from shingeki.data.schemas import FramingTriplet
from shingeki.models.hooked import HookedModel
from shingeki.prompts.chat import ModelFamily
from shingeki.steering.ablation import projection_ablation, subspace_ablation


def _norm_match_basis(D: int, k: int, target_norm: float, seed: int) -> torch.Tensor:
    """Return [k, D] random orthonormal basis scaled so each row has `target_norm`."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(D, k, generator=g)
    Q, _ = torch.linalg.qr(A)  # [D, k] orthonormal columns
    Q = Q.T  # [k, D]
    Q = Q / (Q.norm(dim=1, keepdim=True) + 1e-8) * target_norm
    return Q


def random_subspace_distribution(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    layer: int,
    rank: int,
    target_norm: float,
    n_seeds: int = 25,
    fmt: str = "yesno",
) -> dict:
    """Run subspace_ablation with K random same-rank subspaces."""
    shrinks: list[float] = []
    raws: list[dict] = []
    for s in range(n_seeds):
        basis = _norm_match_basis(model.cfg.d_model, rank, target_norm, seed=1000 + s)
        res = subspace_ablation(model, triplets, family, layer=layer, basis=basis, fmt=fmt)
        shrinks.append(res["shrink_frac"])
        raws.append(
            {
                "seed": 1000 + s,
                "S_C_baseline": res["S_C_baseline"],
                "S_C_ablated": res["S_C_ablated"],
                "shrink_frac": res["shrink_frac"],
            }
        )
    arr = np.array(shrinks)
    return {
        "n_seeds": int(n_seeds),
        "rank": int(rank),
        "layer": int(layer),
        "target_norm": float(target_norm),
        "mean_shrink": float(arr.mean()),
        "median_shrink": float(np.median(arr)),
        "std_shrink": float(arr.std()),
        "min_shrink": float(arr.min()),
        "max_shrink": float(arr.max()),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "raw": raws,
    }


def random_subspace_controls_full(
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    layer: int,
    true_directions: dict[str, torch.Tensor],
    ranks: tuple[int, ...] = (1, 4),
    n_seeds: int = 25,
    fmt: str = "yesno",
) -> dict:
    """Compare true-direction shrink vs random same-rank distribution.

    `true_directions[name]` may be either rank-1 (D,) or rank-r ([k, D]).
    For each rank in `ranks`, build a matching random distribution.
    """
    out: dict = {"layer": int(layer), "n_seeds": int(n_seeds), "ranks": list(ranks)}

    # Estimate the layer's residual norm from a couple of forward passes so the
    # random subspaces sit at the same scale as the activations they perturb.
    norms = []
    from shingeki.activations.extract import get_residual_stream
    from shingeki.prompts.builders import build_aot_prompt

    with torch.inference_mode():
        for t in triplets[:5]:
            for fr in ("sympathetic", "condemning"):
                p = build_aot_prompt(t, fr, fmt, family, model.tokenizer, model.device)  # type: ignore[arg-type]
                rs = get_residual_stream(model, p, position="final", which="post")  # [L, D]
                norms.append(float(rs[layer].float().norm().item()))
    target_norm = float(np.mean(norms))
    out["target_norm"] = target_norm

    # True-direction shrinks (rank-1 projection_ablation OR rank-r subspace_ablation)
    true_results: dict[str, dict] = {}
    for name, d in true_directions.items():
        if d.ndim == 1:
            res = projection_ablation(model, triplets, family, layer=layer, direction=d, fmt=fmt)
        else:
            res = subspace_ablation(model, triplets, family, layer=layer, basis=d, fmt=fmt)
        true_results[name] = {
            "rank": int(d.shape[0]) if d.ndim > 1 else 1,
            "S_C_baseline": res["S_C_baseline"],
            "S_C_ablated": res["S_C_ablated"],
            "shrink_frac": res["shrink_frac"],
        }
    out["true"] = true_results

    # Random distribution per rank
    out["random_per_rank"] = {
        f"rank{r}": random_subspace_distribution(
            model, triplets, family, layer, r, target_norm, n_seeds=n_seeds, fmt=fmt
        )
        for r in ranks
    }
    return out


def write_random_subspace_audit(
    out_dir: Path,
    model: HookedModel,
    triplets: Sequence[FramingTriplet],
    family: ModelFamily,
    layer: int,
    true_directions: dict[str, torch.Tensor],
    ranks: tuple[int, ...] = (1, 4),
    n_seeds: int = 25,
    fmt: str = "yesno",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = random_subspace_controls_full(
        model, triplets, family, layer, true_directions, ranks=ranks, n_seeds=n_seeds, fmt=fmt
    )
    (out_dir / "result.json").write_text(json.dumps(res, indent=2, default=str))

    # Verdict line
    lines: list[str] = ["true_name,rank,true_shrink,random_mean,random_p95,random_max,verdict"]
    for name, sub in res["true"].items():
        rank = sub["rank"]
        rand = res["random_per_rank"][f"rank{rank}"]
        verdict = "SPECIFIC" if abs(sub["shrink_frac"]) > rand["p95"] + 0.05 else "NON-SPECIFIC"
        lines.append(
            f"{name},{rank},{sub['shrink_frac']:+.4f},"
            f"{rand['mean_shrink']:+.4f},{rand['p95']:+.4f},{rand['max_shrink']:+.4f},{verdict}"
        )
    (out_dir / "verdict.csv").write_text("\n".join(lines) + "\n")
    return out_dir

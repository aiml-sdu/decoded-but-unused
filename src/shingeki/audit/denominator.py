"""Denominator-stability audit for activation patching (gap C).

The recovery metric `(patched - corrupt) / (clean - corrupt)` blows up when the
clean-corrupt gap is small. The current paper reports median recovery for that
reason. This audit makes it explicit:

  - Threshold τ = 2.0 logits on |clean - corrupt|.
  - Split triplets into "stable" (|gap| >= τ) and "small" (|gap| < τ).
  - Report mean / median / winsorized-mean / fraction-positive for each subset
    and for "stable" only, with bootstrap CI over triplets.
  - Emit per-layer curves.

Inputs are the saved arrays.npz from sweep_layers (fields: recovery [N, L],
clean_score [N], corrupt_score [N]).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import trim_mean


def _bootstrap_ci(values: np.ndarray, n: int = 5000, seed: int = 42) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, values.size, values.size)
        boots[i] = float(np.median(values[idx]))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _summarize(rec_subset: np.ndarray) -> dict:
    if rec_subset.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "trimmed_mean_20": float("nan"),
            "frac_positive": float("nan"),
            "ci95_median_lo": float("nan"),
            "ci95_median_hi": float("nan"),
        }
    finite = rec_subset[np.isfinite(rec_subset)]
    if finite.size == 0:
        finite = rec_subset
    lo, hi = _bootstrap_ci(finite)
    return {
        "n": int(rec_subset.size),
        "n_finite": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "trimmed_mean_20": float(trim_mean(finite, 0.2)) if finite.size >= 5 else float("nan"),
        "frac_positive": float((finite > 0).mean()),
        "ci95_median_lo": lo,
        "ci95_median_hi": hi,
    }


def split_by_denominator(
    bundle_dir: Path,
    threshold: float = 2.0,
    layers_of_interest: Sequence[int] | None = None,
) -> dict:
    """Re-aggregate a sweep_layers bundle by denominator stability.

    Parameters
    ----------
    bundle_dir : Path
        Directory containing arrays.npz (recovery, clean_score, corrupt_score)
        and result.json (config with "layers" key).
    threshold : float
        |clean - corrupt| logit threshold; below it triplets are denominator-small.
    layers_of_interest : list[int] | None
        Restrict per-layer reporting to these (still report all layers in arrays).
    """
    arrays_path = bundle_dir / "arrays.npz"
    result_path = bundle_dir / "result.json"
    if not arrays_path.exists() or not result_path.exists():
        return {"error": f"missing artifacts at {bundle_dir}"}
    arrs = np.load(arrays_path)
    res = json.loads(result_path.read_text())
    rec = arrs["recovery"]  # [N, L]
    clean = arrs["clean_score"]
    corrupt = arrs["corrupt_score"]
    layers = res.get("config", {}).get("layers", list(range(rec.shape[1])))

    gap = np.abs(np.asarray(clean) - np.asarray(corrupt))
    stable_mask = gap >= threshold
    small_mask = ~stable_mask

    out: dict = {
        "bundle_dir": str(bundle_dir),
        "threshold": float(threshold),
        "n_triplets": int(rec.shape[0]),
        "n_layers": int(rec.shape[1]),
        "n_stable": int(stable_mask.sum()),
        "n_small": int(small_mask.sum()),
        "stable_mean_gap": float(gap[stable_mask].mean()) if stable_mask.any() else float("nan"),
        "small_mean_gap": float(gap[small_mask].mean()) if small_mask.any() else float("nan"),
        "per_layer_stable": [],
        "per_layer_small": [],
        "per_layer_all": [],
    }
    for j in range(rec.shape[1]):
        out["per_layer_all"].append(_summarize(rec[:, j]))
        out["per_layer_stable"].append(_summarize(rec[stable_mask, j]) if stable_mask.any() else _summarize(np.array([])))
        out["per_layer_small"].append(_summarize(rec[small_mask, j]) if small_mask.any() else _summarize(np.array([])))

    # Best-layer and best-stable-layer summaries
    means_all = np.array([s["median"] for s in out["per_layer_all"]])
    means_stable = np.array([s["median"] for s in out["per_layer_stable"]])
    out["best_layer_all"] = int(layers[int(np.nanargmax(means_all))])
    out["best_layer_all_median"] = float(np.nanmax(means_all))
    if stable_mask.any():
        out["best_layer_stable"] = int(layers[int(np.nanargmax(means_stable))])
        out["best_layer_stable_median"] = float(np.nanmax(means_stable))
    return out


def write_denominator_audit(
    out_dir: Path,
    bundle_dirs: dict[str, Path],
    threshold: float = 2.0,
) -> Path:
    """Run denominator audit on multiple bundles (e.g. S_to_C and C_to_S)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, dict] = {}
    for name, bd in bundle_dirs.items():
        payload[name] = split_by_denominator(bd, threshold=threshold)
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, default=str))

    csv_lines = ["bundle,n_stable,n_small,best_layer_all,best_median_all,best_layer_stable,best_median_stable"]
    for name, sub in payload.items():
        if "error" in sub:
            csv_lines.append(f"{name},,,,,,ERROR")
            continue
        csv_lines.append(
            f"{name},{sub['n_stable']},{sub['n_small']},"
            f"{sub.get('best_layer_all', '')},{sub.get('best_layer_all_median', float('nan')):.4f},"
            f"{sub.get('best_layer_stable', '')},{sub.get('best_layer_stable_median', float('nan')):.4f}"
        )
    (out_dir / "summary.csv").write_text("\n".join(csv_lines) + "\n")
    return out_dir

"""Intermediate-layer patching audit (concern #3).

The headline "C->S patching at L=33 recovers 0.96" risks being trivially close
to "swap the unembedding logit." This audit reports stable-subset median
recovery at intermediate layers (10, 15, 20, 25) so the *real* mechanism
evidence is anchored in the middle of the network, not at L=33.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def intermediate_recovery(
    bundle_dirs: dict[str, Path],
    layers: tuple[int, ...] = (10, 15, 20, 25, 28, 30, 32, 33),
    threshold: float = 2.0,
) -> dict:
    out: dict = {"layers": list(layers), "threshold": float(threshold), "per_bundle": {}}
    for name, bd in bundle_dirs.items():
        arr = np.load(bd / "arrays.npz")
        rec = arr["recovery"]  # [N, L]
        clean = np.asarray(arr["clean_score"])
        corrupt = np.asarray(arr["corrupt_score"])
        gap = np.abs(clean - corrupt)
        stable_mask = gap >= threshold
        rows: list[dict] = []
        for l in layers:
            if l >= rec.shape[1]:
                rows.append({"layer": int(l), "error": "out of range"})
                continue
            all_finite = rec[:, l][np.isfinite(rec[:, l])]
            stable_finite = rec[stable_mask, l][np.isfinite(rec[stable_mask, l])]
            rows.append(
                {
                    "layer": int(l),
                    "n_stable": int(stable_mask.sum()),
                    "median_all": float(np.median(all_finite)) if all_finite.size else float("nan"),
                    "mean_all": float(np.mean(all_finite)) if all_finite.size else float("nan"),
                    "median_stable": float(np.median(stable_finite)) if stable_finite.size else float("nan"),
                    "mean_stable": float(np.mean(stable_finite)) if stable_finite.size else float("nan"),
                    "frac_positive_stable": float((stable_finite > 0).mean()) if stable_finite.size else float("nan"),
                }
            )
        out["per_bundle"][name] = rows
    return out


def write_intermediate_audit(out_dir: Path, bundle_dirs: dict[str, Path]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = intermediate_recovery(bundle_dirs)
    (out_dir / "result.json").write_text(json.dumps(res, indent=2, default=str))
    csv_lines = ["bundle,layer,n_stable,median_stable,mean_stable,frac_positive_stable"]
    for name, rows in res["per_bundle"].items():
        for r in rows:
            if "error" in r:
                continue
            csv_lines.append(
                f"{name},{r['layer']},{r['n_stable']},{r['median_stable']:.4f},"
                f"{r['mean_stable']:.4f},{r['frac_positive_stable']:.4f}"
            )
    (out_dir / "summary.csv").write_text("\n".join(csv_lines) + "\n")
    return out_dir

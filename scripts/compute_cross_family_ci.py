"""Compute paired bootstrap CIs for the cross-family Table 6.

Reads per-model behavioural arrays from runs/, computes the paired sympathetic
minus condemning diff per triplet, and reports the bootstrap mean + 95% CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shingeki.stats.bootstrap import paired_bootstrap_ci

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"

# Map: model display name -> run directory (most recent N=100 behavior bundle)
MODELS = {
    "Gemma-3-4B-PT":     "f195bace571b__7c764dda87/exp20_pt_behavior_n100",
    "Gemma-3-4B-IT":     "f195bace571b__07c13e30bc/exp01_behavior_n100",
    "Gemma-2-2B-IT":     "ff3be407d878__bbfcc98ada/exp30_gemma2_2b_n100_behavior",
    "Gemma-2-9B-IT":     "c6c1c149bd20__3457409ae2/exp23_gemma2_9b_n100_behavior",
    "Gemma-2-27B-IT":    "ff3be407d878__1d153ef8a3/exp32_gemma2_27b_n100_behavior",
    "Gemma-3-12B-IT":    "c6c1c149bd20__aabaf661e7/exp24_gemma3_12b_n100_behavior",
    "Qwen-2.5-3B-Inst":  "ff3be407d878__3b0874841e/exp31_qwen_3b_n100_behavior",
    "Qwen-2.5-7B-Inst":  "c6c1c149bd20__e9d888a907/exp25_qwen_7b_n100_behavior",
    "Qwen-2.5-14B-Inst": "c6c1c149bd20__86d6c44973/exp26_qwen_14b_n100_behavior",
    "Qwen-2.5-32B-Inst": "ff3be407d878__097a67e846/exp33_qwen_32b_n100_behavior",
    "Llama-3.1-8B-IT":   "67dac9f6192e__0d356f3b64/exp28_llama31_8b_n100_behavior",
}


def main() -> None:
    rows = []
    for name, run in MODELS.items():
        path = RUNS / run / "arrays.npz"
        if not path.exists():
            print(f"[skip] {name}: no arrays at {path}")
            continue
        arr = np.load(path)
        sym, cond = arr["sym"], arr["cond"]
        n = len(sym)
        diffs = sym - cond
        mean, lo, hi = paired_bootstrap_ci(diffs, n_resamples=10_000, seed=42)
        frac_pos = float((diffs > 0).mean())
        # Sign-flip permutation p-value
        rng = np.random.default_rng(42)
        signs = rng.choice([-1, 1], size=(10_000, n))
        null_means = (signs * np.abs(diffs)).mean(axis=1)
        p = float((np.abs(null_means) >= abs(mean)).mean())
        rows.append({
            "model": name,
            "n": int(n),
            "mean": round(mean, 3),
            "ci_lo": round(lo, 3),
            "ci_hi": round(hi, 3),
            "frac_pos": round(frac_pos, 3),
            "p": p,
        })
        print(f"{name:20s} n={n:3d}  S-C={mean:+7.2f}  "
              f"CI=[{lo:+6.2f}, {hi:+6.2f}]  frac>0={frac_pos:.2f}  p={p:.2e}")

    out = REPO / "manuscript" / "data" / "cross_family_ci.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

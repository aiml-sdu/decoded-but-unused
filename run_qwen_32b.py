"""Qwen-2.5-32B-Instruct on n=100.

Isolates the Gemma-2-27B drop story: is the +4.47 we saw on Gemma-2-27B
specific to Gemma post-training, or is it a general size-32B effect?

Runs behavior + probe + lens. ~30 min on B200.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    out_dir = Path("runs/qwen_32b_check")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pid").write_text(str(os.getpid()))

    from shingeki.behavior.scoring import behavioral_validation_bundle
    from shingeki.data.aot import load_combined_n100
    from shingeki.io.results import ResultBundle, write_bundle
    from shingeki.lens.logit_lens import logit_lens_bundle
    from shingeki.models.loader import load_hooked
    from shingeki.probing.linear import cache_activations, fit_probes_all_layers

    triplets = load_combined_n100()
    print(f"[{_now()}] loading qwen25-32b-it ...", flush=True)
    t0 = time.time()
    m = load_hooked("qwen25-32b-it")
    family = "qwen25-it"
    print(f"  loaded in {time.time()-t0:.1f}s; L={m.cfg.n_layers} D={m.cfg.d_model}", flush=True)

    # Behavior
    t0 = time.time()
    beh = behavioral_validation_bundle(m, triplets, family, fmt="yesno", exp_id="exp29_qwen_32b_n100_behavior")
    write_bundle(beh)
    sub = beh.metrics["S_vs_C"]
    print(
        f"  S-C mean={sub['mean_shift']:+.3f} median={sub['median_shift']:+.3f} "
        f"p={sub['perm_p']:.4f} frac>0={sub['frac_sym_gt_cond']:.2f}  ({time.time()-t0:.1f}s)",
        flush=True,
    )
    sources = {"aot": [], "aot_ext": [], "got": []}
    sym_arr = beh.arrays["sym"]; cond_arr = beh.arrays["cond"]
    for i, t in enumerate(triplets):
        gap = float(sym_arr[i] - cond_arr[i])
        if t.id.startswith("aot-0") and int(t.id.split("-")[1]) <= 40:
            sources["aot"].append(gap)
        elif t.id.startswith("aot-"):
            sources["aot_ext"].append(gap)
        elif t.id.startswith("got-"):
            sources["got"].append(gap)
    print("  by source:", {k: round(float(np.mean(v)), 3) for k, v in sources.items() if v}, flush=True)

    # Probe
    t0 = time.time()
    X, y, g = cache_activations(m, triplets, family)
    probes = fit_probes_all_layers(X, y, g)
    accs = [p.accuracy for p in probes]
    bundle = ResultBundle(
        exp_id="exp29_qwen_32b_n100_probe",
        config={"model_id": "qwen25-32b-it", "dataset": "n100"},
        metrics={"accuracy_per_layer": accs, "best_layer": int(np.argmax(accs)), "best_acc": float(np.max(accs))},
        arrays={"X": X.astype(np.float32)},
    )
    write_bundle(bundle)
    print(f"  probe best L={int(np.argmax(accs))} acc={np.max(accs):.3f}  ({time.time()-t0:.1f}s)", flush=True)

    # Lens
    try:
        lens = logit_lens_bundle(m, triplets, family, fmt="yesno", exp_id="exp29_qwen_32b_n100_lens")
        write_bundle(lens)
        gap = lens.metrics["sym_minus_cond_per_layer"]
        print(f"  lens S-C: L0={gap[0]:+.2f} mid={gap[len(gap)//2]:+.2f} last={gap[-1]:+.2f}", flush=True)
    except Exception as e:
        print(f"  lens failed: {type(e).__name__}: {e}", flush=True)

    print(f"\n[{_now()}] qwen-2.5-32b check DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

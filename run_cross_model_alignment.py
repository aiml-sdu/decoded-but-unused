"""Cross-model alignment-cosine audit on n=100.

For each wave-2 IT checkpoint, compute layerwise cos(d_frame, d_judge)
on the AoT-50 + GoT-50 testbed and report the peak. Establishes the
routing-metric multi-model claim: every IT model has a non-trivial
alignment cosine, the matched PT does not.

Sequential per-model so peak GPU stays ~30 GB; doesn't conflict with the
n=100 orchestrator running on Gemma-3-4B.
"""

from __future__ import annotations

import json
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


# (spec_id, family) pairs to audit
TARGETS = [
    ("gemma2-2b-it",   "gemma2-it"),
    ("gemma2-9b-it",   "gemma2-it"),
    ("gemma3-12b-it",  "gemma3-it"),
    ("qwen25-3b-it",   "qwen25-it"),
    ("qwen25-7b-it",   "qwen25-it"),
    ("qwen25-14b-it",  "qwen25-it"),
    ("gemma2-27b-it",  "gemma2-it"),
    ("llama31-8b-it",  "llama31-it"),
    ("qwen25-32b-it",  "qwen25-it"),
]


def main() -> int:
    out_dir = Path("runs/cross_model_alignment")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pid").write_text(str(os.getpid()))
    print(f"[{_now()}] cross-model alignment audit -> {out_dir}", flush=True)

    from shingeki.audit.direction_alignment import model_direction_block
    from shingeki.data.aot import load_combined_n100
    from shingeki.models.loader import load_hooked

    triplets = load_combined_n100()
    print(f"[{_now()}] n100 triplets: {len(triplets)}", flush=True)

    summary = []
    for spec_id, family in TARGETS:
        print(f"\n=== {spec_id} ({family}) ===", flush=True)
        t0 = time.time()
        try:
            m = load_hooked(spec_id)
            print(f"  loaded in {time.time()-t0:.1f}s; L={m.cfg.n_layers} D={m.cfg.d_model}", flush=True)

            t1 = time.time()
            block = model_direction_block(m, triplets, family, fmt="yesno")
            align = block["alignment_per_layer"]
            peak_layer = int(np.argmax(align))
            peak_cos = float(align[peak_layer])
            n_layers = int(block["n_layers"])
            d_model = int(block["d_model"])
            print(
                f"  alignment: peak L={peak_layer}/{n_layers-1} cos={peak_cos:+.4f}  "
                f"({time.time()-t1:.1f}s)",
                flush=True,
            )

            np.savez(
                out_dir / f"{spec_id}_align.npz",
                alignment_per_layer=align,
                d_frame=block["d_frame"],
                d_judge=block["d_judge"],
            )
            summary.append({
                "spec_id": spec_id,
                "family": family,
                "n_layers": n_layers,
                "d_model": d_model,
                "peak_layer": peak_layer,
                "peak_cos": peak_cos,
                "alignment_per_layer": align.tolist(),
            })

            # Free GPU memory before loading the next model
            import torch
            del m, block
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}", flush=True)
            summary.append({"spec_id": spec_id, "family": family, "error": f"{type(e).__name__}: {e}"})
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    # Write summary CSV + JSON
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    csv_lines = ["spec_id,family,n_layers,d_model,peak_layer,peak_cos"]
    for r in summary:
        if "error" in r:
            csv_lines.append(f"{r['spec_id']},{r['family']},,,,ERROR")
        else:
            csv_lines.append(
                f"{r['spec_id']},{r['family']},{r['n_layers']},{r['d_model']},"
                f"{r['peak_layer']},{r['peak_cos']:.4f}"
            )
    (out_dir / "summary.csv").write_text("\n".join(csv_lines) + "\n")

    print(f"\n[{_now()}] cross-model alignment DONE -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

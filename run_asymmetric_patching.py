"""Asymmetric-patching component breakdown (concern: why is C->S 0.92 vs S->C 0.38 at L=20?).

For both directions, run:
  - resid (already known, baseline)
  - attn_out only
  - mlp_out only
at L=20 on D0 (n=40, sufficient since the asymmetry is already visible there).
Goal: localize the asymmetry to attention vs MLP if possible.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timezone.utc.utcoffset(None) or 0).replace("+00:00", "+00:00")


def main() -> int:
    out_dir = Path("runs/asymmetric_patching")
    out_dir.mkdir(parents=True, exist_ok=True)
    pid_file = out_dir / "pid"
    pid_file.write_text(str(os.getpid()))
    print(f"asymmetric patching, out_dir: {out_dir}", flush=True)

    from shingeki.behavior.scoring import score_prompt
    from shingeki.data.aot import load_d0
    from shingeki.io.results import ResultBundle, write_bundle
    from shingeki.models.loader import load_hooked
    from shingeki.models.tokens import resolve_answer_tokens
    from shingeki.patching.recovery import recovery
    from shingeki.prompts.builders import build_aot_prompt

    triplets = load_d0()
    print(f"loading gemma-3-4b-it ...", flush=True)
    t0 = time.time()
    m = load_hooked("gemma3-4b-it")
    fam = "gemma3-it"
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    ans = resolve_answer_tokens(m.tokenizer, "yesno")

    results: dict = {"layer": 20, "directions": {}, "n_triplets": len(triplets)}

    @torch.inference_mode()
    def patch_one(clean_built, corrupt_built, hook_name: str, layer: int) -> dict:
        # Cache clean activation at hook_name
        _logits, cache = m.run_with_cache(
            clean_built.input_ids, [hook_name], attention_mask=clean_built.attention_mask
        )
        clean_act = cache[hook_name].detach()
        position = corrupt_built.input_ids.shape[1] - 1
        pos_clean = min(position, clean_act.shape[1] - 1)

        def hook(t, _name):
            new = t.clone()
            if isinstance(new, tuple):
                # attn_out etc. sometimes return tuples; handled by hooked_model
                new = new[0]
            new[:, position] = clean_act[:, pos_clean]
            return new

        patched_logits = m.run_with_hooks(
            corrupt_built.input_ids, [(hook_name, hook)], attention_mask=corrupt_built.attention_mask
        )
        clean_logits = m(clean_built.input_ids, clean_built.attention_mask)
        corrupt_logits = m(corrupt_built.input_ids, corrupt_built.attention_mask)
        cs = float(clean_logits[0, -1, ans.pos_id] - clean_logits[0, -1, ans.neg_id])
        ks = float(corrupt_logits[0, -1, ans.pos_id] - corrupt_logits[0, -1, ans.neg_id])
        ps = float(patched_logits[0, -1, ans.pos_id] - patched_logits[0, -1, ans.neg_id])
        return {
            "clean_score": cs,
            "corrupt_score": ks,
            "patched_score": ps,
            "recovery": recovery(ps, cs, ks),
        }

    LAYER = 20
    for direction in ("S_to_C", "C_to_S"):
        if direction == "S_to_C":
            cf, qf = "sympathetic", "condemning"
        else:
            cf, qf = "condemning", "sympathetic"
        for hook_kind in ("hook_resid_pre", "hook_resid_post", "hook_attn_out", "hook_mlp_out"):
            hook_name = f"blocks.{LAYER}.{hook_kind}"
            recs = []
            cs_arr = []
            ks_arr = []
            for t in triplets:
                clean = build_aot_prompt(t, cf, "yesno", fam, m.tokenizer, m.device)
                corrupt = build_aot_prompt(t, qf, "yesno", fam, m.tokenizer, m.device)
                try:
                    res = patch_one(clean, corrupt, hook_name, LAYER)
                except Exception as e:
                    print(f"  patch failed for {direction} {hook_kind}: {type(e).__name__}: {e}", flush=True)
                    continue
                recs.append(res["recovery"])
                cs_arr.append(res["clean_score"])
                ks_arr.append(res["corrupt_score"])
            arr = np.array(recs)
            gap = np.abs(np.array(cs_arr) - np.array(ks_arr))
            stable_mask = gap >= 2.0
            stable = arr[stable_mask]
            row = {
                "direction": direction,
                "hook": hook_kind,
                "n": len(arr),
                "n_stable": int(stable_mask.sum()),
                "mean_recovery": float(np.mean(arr)) if arr.size else float("nan"),
                "median_recovery": float(np.median(arr)) if arr.size else float("nan"),
                "stable_mean_recovery": float(np.mean(stable)) if stable.size else float("nan"),
                "stable_median_recovery": float(np.median(stable)) if stable.size else float("nan"),
            }
            print(
                f"  {direction:>7s}  {hook_kind:>20s}  med_stable={row['stable_median_recovery']:+.3f}  "
                f"mean_stable={row['stable_mean_recovery']:+.3f}  n_stable={row['n_stable']}",
                flush=True,
            )
            results["directions"].setdefault(direction, []).append(row)

    import json
    (out_dir / "result.json").write_text(json.dumps(results, indent=2))
    csv_lines = ["direction,hook,n_stable,stable_mean_recovery,stable_median_recovery"]
    for direction, rows in results["directions"].items():
        for r in rows:
            csv_lines.append(f"{r['direction']},{r['hook']},{r['n_stable']},{r['stable_mean_recovery']:.4f},{r['stable_median_recovery']:.4f}")
    (out_dir / "summary.csv").write_text("\n".join(csv_lines) + "\n")
    print(f"\nasymmetric-patching breakdown -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

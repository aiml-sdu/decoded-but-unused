"""Standalone orthogonal-PCA audit (concern #4).

Loads Gemma-3-4B-IT, runs the rank-1 d_judge / rank-4 PCA-full /
rank-4 PCA-orthogonalized-to-d_judge ablations at L=20, 25, 28 on D0
(or n=100 if available). Writes verdict.csv + result.json.

Cheap (~5 min). Designed to be safe to run alongside LoRA and n=100.

Usage:
  python run_orthogonal_pca_audit.py --out-dir runs/orthogonal_pca_<TS>
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--use-n100", action="store_true", help="Use n=100 dataset instead of D0.")
    parser.add_argument("--layers", default="20,25,28")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--pid-file", default="/tmp/orthopca.pid")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.pid_file).write_text(str(os.getpid()))
    print(f"[{_now()}] orthogonal-PCA audit starting", flush=True)
    print(f"  out_dir: {out_dir}", flush=True)
    print(f"  layers: {args.layers} rank: {args.rank}", flush=True)

    from shingeki.audit.orthogonal_pca import write_orthogonal_pca_audit
    from shingeki.data.aot import load_combined_n100, load_d0
    from shingeki.models.loader import load_hooked

    triplets = load_combined_n100() if args.use_n100 else load_d0()
    print(f"  triplets: {len(triplets)}", flush=True)

    print(f"[{_now()}] loading gemma-3-4b-it ...", flush=True)
    t0 = time.time()
    m = load_hooked("gemma3-4b-it")
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    layers = tuple(int(l) for l in args.layers.split(","))
    t0 = time.time()
    write_orthogonal_pca_audit(
        out_dir, m, triplets, family="gemma3-it",
        layers=layers, rank=args.rank, fmt="yesno",
    )
    print(f"[{_now()}] audit done in {time.time()-t0:.1f}s", flush=True)
    print(f"  verdict.csv: {out_dir / 'verdict.csv'}", flush=True)
    print(f"  result.json: {out_dir / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

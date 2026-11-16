"""n=100 (AoT-50 + GoT-50) primary experiments + cross-model behavior.

Designed to reproduce the headline findings on the expanded 100-triplet
testbed. Skips the long-pole experiments (head-patching, MLP-neuron, full
denominator audit) that aren't load-bearing for the paper's headline; those
can be added later if there's time.

Phases:
  1. manifest_n100              - env snapshot for the n=100 run
  2. it_n100                    - Gemma-3-4B-IT: behavior, robustness, cosine,
                                  probe, lens, geometry, residual patching
                                  (both directions), steering, projection
                                  ablation, denominator audit, transfer matrix,
                                  random-subspace audit, intermediate-layers
                                  audit
  3. pt_n100                    - Gemma-3-4B-PT: behavior, probe, lens,
                                  alignment-block save
  4. alignment_audit_n100       - PT-vs-IT direction alignment from saved blocks
  5. cross_model_n100           - Gemma-2-9B-IT, Gemma-3-12B-IT,
                                  Qwen-2.5-7B-IT, Qwen-2.5-14B-IT (skip 27B
                                  unless time permits)
  6. final_report_n100          - synthesize results into investigation_report.md

Usage:
  nohup python run_n100.py > runs/n100_<TS>/log.txt 2>&1 &
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def _phase_done(status: dict, name: str) -> bool:
    return status.get("phases", {}).get(name, {}).get("state") == "completed"


def _begin(status_path: Path, status: dict, name: str) -> None:
    status.setdefault("phases", {})
    status["phases"][name] = {"state": "running", "started_at": _now()}
    _save(status_path, status)
    print(f"\n=== [{_now()}] BEGIN {name} ===", flush=True)


def _ok(status_path: Path, status: dict, name: str, extra: dict | None = None) -> None:
    sub = status["phases"][name]
    sub["state"] = "completed"
    sub["finished_at"] = _now()
    if extra:
        sub.update(extra)
    _save(status_path, status)
    print(f"=== [{_now()}] OK    {name} ===", flush=True)


def _fail(status_path: Path, status: dict, name: str, exc: BaseException) -> None:
    sub = status["phases"].setdefault(name, {})
    sub["state"] = "failed"
    sub["failed_at"] = _now()
    sub["error"] = f"{type(exc).__name__}: {exc}"
    sub["traceback"] = traceback.format_exc(limit=10)
    _save(status_path, status)
    print(f"=== [{_now()}] FAIL  {name}: {sub['error']} ===", flush=True)
    traceback.print_exc()


def _cleanup(*objs) -> None:
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def phase_manifest_n100(out_dir: Path) -> dict:
    from shingeki.audit.manifest import write_manifest

    p = write_manifest(out_dir, _now())
    return {"manifest_path": str(p)}


def phase_it_n100(out_dir: Path) -> dict:
    """Gemma-3-4B-IT on n=100."""
    from shingeki.audit.intermediate_layers import write_intermediate_audit
    from shingeki.audit.random_subspace import write_random_subspace_audit
    from shingeki.audit.transfer_matrix import write_transfer_audit
    from shingeki.behavior.robustness import robustness_bundle
    from shingeki.behavior.scoring import behavioral_validation_bundle
    from shingeki.data.aot import load_combined_n100
    from shingeki.geometry.cosine import cosine_bundle
    from shingeki.geometry.directions import (
        framing_direction_per_layer,
        judgment_direction,
        layerwise_alignment,
    )
    from shingeki.io.results import ResultBundle, write_bundle
    from shingeki.lens.logit_lens import logit_lens_bundle
    from shingeki.models.loader import load_hooked
    from shingeki.models.tokens import resolve_answer_tokens
    from shingeki.patching.residual import sweep_layers
    from shingeki.probing.controls import length_only_baseline, random_label_baseline, tfidf_baseline
    from shingeki.probing.linear import cache_activations, fit_probes_all_layers, probe_direction
    from shingeki.activations.extract import get_residual_stream
    from shingeki.prompts.builders import build_aot_prompt
    from shingeki.steering.ablation import projection_ablation
    from shingeki.steering.sweep import steering_sweep_bundle

    summary: dict = {}

    print("loading gemma-3-4b-it ...", flush=True)
    t_load = time.time()
    m = load_hooked("gemma3-4b-it")
    fam = "gemma3-it"
    print(f"  loaded in {time.time()-t_load:.1f}s; L={m.cfg.n_layers}", flush=True)
    summary["n_layers"] = int(m.cfg.n_layers)
    summary["d_model"] = int(m.cfg.d_model)

    triplets = load_combined_n100()
    print(f"n100 triplets: {len(triplets)}", flush=True)
    summary["n_triplets"] = len(triplets)

    # 1. Behaviour (KEY HEADLINE NUMBER)
    t0 = time.time()
    beh = behavioral_validation_bundle(m, triplets, fam, fmt="yesno", exp_id="exp01_behavior_n100")
    write_bundle(beh)
    sub = beh.metrics["S_vs_C"]
    summary["S_vs_C"] = {
        "mean": float(sub["mean_shift"]),
        "median": float(sub["median_shift"]),
        "perm_p": float(sub["perm_p"]),
        "frac_pos": float(sub["frac_sym_gt_cond"]),
        "ci95_lo": float(sub["ci95_lo"]),
        "ci95_hi": float(sub["ci95_hi"]),
        "cohens_dz": float(sub["cohens_dz"]),
    }
    print(
        f"  S-C mean={sub['mean_shift']:+.3f} median={sub['median_shift']:+.3f} "
        f"p={sub['perm_p']:.4f} frac>0={sub['frac_sym_gt_cond']:.2f}  ({time.time()-t0:.1f}s)",
        flush=True,
    )

    # Per-source breakdown (AoT-40, AoT-extension, GoT)
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
    summary["S_vs_C_by_source"] = {
        k: {"n": len(v), "mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None}
        for k, v in sources.items()
    }
    print("  by source:", summary["S_vs_C_by_source"], flush=True)

    # 2. Robustness
    t0 = time.time()
    rob = robustness_bundle(m, triplets, fam, formats=("yesno", "ab", "justified", "acceptable"),
                             exp_id="exp02_robustness_n100")
    write_bundle(rob)
    print(f"  robustness done ({time.time()-t0:.1f}s)", flush=True)

    # 3. Cosine
    cos = cosine_bundle(m, triplets, fam, fmt="yesno", exp_id="exp03_cosine_n100")
    write_bundle(cos)

    # 4. Probe
    t0 = time.time()
    X, y, g = cache_activations(m, triplets, fam)
    probes = fit_probes_all_layers(X, y, g)
    accs = np.array([p.accuracy for p in probes])
    texts = []
    for t in triplets:
        for fr in ("sympathetic", "condemning", "abstract"):
            texts.append(getattr(t, fr))
    rand_ctl = random_label_baseline(X[:, int(accs.argmax())], y, g)
    len_ctl = length_only_baseline(texts, y, g)
    tfidf_ctl = tfidf_baseline(texts, y, g)
    probe_bundle = ResultBundle(
        exp_id="exp04_probe_n100",
        config={"model_id": "gemma3-4b-it", "dataset": "n100", "n_triplets": len(triplets)},
        metrics={
            "accuracy_per_layer": accs.tolist(),
            "best_layer": int(accs.argmax()),
            "best_acc": float(accs.max()),
            "controls": {"random_label": rand_ctl, "length_only": len_ctl, "tfidf": tfidf_ctl},
        },
        arrays={
            "weights": np.stack([p.weights for p in probes]),
            "intercept": np.stack([p.intercept for p in probes]),
            "X_final": X.astype(np.float32),
            "y_classes": np.array([str(c) for c in probes[0].classes]),
        },
    )
    write_bundle(probe_bundle)
    summary["probe"] = {"best_layer": int(accs.argmax()), "best_acc": float(accs.max())}
    print(f"  probe best L={accs.argmax()} acc={accs.max():.3f}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  controls: random={rand_ctl['mean_acc']:.3f} length={len_ctl['mean_acc']:.3f} tfidf={tfidf_ctl['mean_acc']:.3f}", flush=True)

    # 5. Logit lens
    lens = logit_lens_bundle(m, triplets, fam, fmt="yesno", exp_id="exp05_logit_lens_n100")
    write_bundle(lens)

    # 6. Geometry
    t0 = time.time()
    ans = resolve_answer_tokens(m.tokenizer, "yesno")
    L = m.cfg.n_layers
    D = m.cfg.d_model
    sym_acts = np.zeros((len(triplets), L, D), dtype=np.float32)
    cond_acts = np.zeros((len(triplets), L, D), dtype=np.float32)
    for i, t in enumerate(triplets):
        ps = build_aot_prompt(t, "sympathetic", "yesno", fam, m.tokenizer, m.device)
        pc = build_aot_prompt(t, "condemning", "yesno", fam, m.tokenizer, m.device)
        with torch.inference_mode():
            sym_acts[i] = get_residual_stream(m, ps, position="final", which="post").float().cpu().numpy()
            cond_acts[i] = get_residual_stream(m, pc, position="final", which="post").float().cpu().numpy()
    d_frame = framing_direction_per_layer(torch.from_numpy(sym_acts), torch.from_numpy(cond_acts))
    d_judge = judgment_direction(m, ans).cpu()
    align = layerwise_alignment(d_frame, d_judge).numpy()
    geom_bundle = ResultBundle(
        exp_id="exp06_geometry_n100",
        config={"model_id": "gemma3-4b-it", "dataset": "n100"},
        metrics={
            "alignment_per_layer": align.tolist(),
            "peak_layer": int(np.argmax(np.abs(align))),
            "peak_alignment": float(align[int(np.argmax(np.abs(align)))]),
        },
        arrays={"d_frame": d_frame.numpy(), "d_judge": d_judge.numpy(), "alignment": align},
    )
    write_bundle(geom_bundle)
    np.savez_compressed(
        out_dir / "it_direction_block_n100.npz",
        d_frame=d_frame.numpy(), d_judge=d_judge.numpy(), alignment=align,
        sym_acts=sym_acts, cond_acts=cond_acts,
    )
    summary["geometry"] = {"peak_layer": int(np.argmax(np.abs(align))),
                          "peak_alignment": float(align[int(np.argmax(np.abs(align)))])}
    print(f"  geometry peak L={summary['geometry']['peak_layer']} cos={summary['geometry']['peak_alignment']:+.3f}  ({time.time()-t0:.1f}s)", flush=True)

    # 7. Residual patching (both directions)
    t0 = time.time()
    resid_sc = sweep_layers(m, triplets, fam, direction="S_to_C", fmt="yesno", exp_id="exp07_resid_S_to_C_n100")
    sc_dir = write_bundle(resid_sc)
    resid_cs = sweep_layers(m, triplets, fam, direction="C_to_S", fmt="yesno", exp_id="exp07_resid_C_to_S_n100")
    cs_dir = write_bundle(resid_cs)
    summary["patch"] = {
        "S_to_C_best_layer": int(resid_sc.metrics["best_layer"]),
        "S_to_C_best_recovery": float(resid_sc.metrics["best_layer_recovery"]),
        "C_to_S_best_layer": int(resid_cs.metrics["best_layer"]),
        "C_to_S_best_recovery": float(resid_cs.metrics["best_layer_recovery"]),
    }
    print(f"  patching done  ({time.time()-t0:.1f}s)", flush=True)

    # 8. Steering at residual peak
    peak_layer = int(resid_sc.metrics["best_layer"])
    target_layer = peak_layer if peak_layer >= 20 else 28
    if target_layer >= L:
        target_layer = L - 4
    steer = steering_sweep_bundle(m, triplets, fam, layer=target_layer, alpha=4.0, exp_id=f"exp13_steering_L{target_layer}_n100")
    write_bundle(steer)

    # 9. Projection ablation at L=28 with probe direction
    d_probe = torch.from_numpy(probe_direction(probes[target_layer])).float()
    proj_res = projection_ablation(m, triplets, fam, layer=target_layer, direction=d_probe, fmt="yesno")
    write_bundle(ResultBundle(
        exp_id=f"exp14_projection_ablation_L{target_layer}_n100",
        config={"layer": target_layer, "model_id": "gemma3-4b-it", "dataset": "n100"},
        metrics=proj_res,
    ))
    summary["projection_rank1"] = {"shrink_frac": float(proj_res["shrink_frac"]),
                                    "S_C_baseline": float(proj_res["S_C_baseline"]),
                                    "S_C_ablated": float(proj_res["S_C_ablated"])}
    print(f"  projection rank-1 shrink={proj_res['shrink_frac']:+.3f}", flush=True)

    # 10. Audit B (random subspace controls) at target_layer
    diffs = sym_acts[:, target_layer] - cond_acts[:, target_layer]
    diffs_centered = diffs - diffs.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
    pca_top4 = torch.from_numpy(Vt[:4]).float()
    write_random_subspace_audit(
        out_dir / "audit_random_subspace_n100", m, triplets, fam,
        layer=target_layer,
        true_directions={"probe_direction": d_probe, "pca_top4": pca_top4},
        ranks=(1, 4), n_seeds=15, fmt="yesno",
    )
    summary["audit_random_subspace_layer"] = int(target_layer)

    # 11. Audit A (transfer matrix) - skip since D2 + D1 not aligned to n=100;
    # rely on the existing n=40 transfer matrix from wave 1.

    # 12. Intermediate-layer audit on patching
    write_intermediate_audit(
        out_dir / "audit_intermediate_n100",
        {"S_to_C": Path(sc_dir), "C_to_S": Path(cs_dir)},
    )

    _cleanup(m)
    return summary


def phase_pt_n100(out_dir: Path) -> dict:
    """Gemma-3-4B-PT on n=100."""
    from shingeki.audit.direction_alignment import collect_frame_acts
    from shingeki.behavior.scoring import behavioral_validation_bundle
    from shingeki.data.aot import load_combined_n100
    from shingeki.geometry.directions import (
        framing_direction_per_layer,
        judgment_direction,
        layerwise_alignment,
    )
    from shingeki.io.results import ResultBundle, write_bundle
    from shingeki.lens.logit_lens import logit_lens_bundle
    from shingeki.models.loader import load_hooked
    from shingeki.models.tokens import resolve_answer_tokens
    from shingeki.probing.linear import cache_activations, fit_probes_all_layers

    summary: dict = {}
    m = load_hooked("gemma3-4b-pt")
    fam = "gemma3-pt"
    triplets = load_combined_n100()

    beh = behavioral_validation_bundle(m, triplets, fam, fmt="yesno", exp_id="exp20_pt_behavior_n100")
    write_bundle(beh)
    sub = beh.metrics["S_vs_C"]
    summary["S_vs_C"] = {"mean": float(sub["mean_shift"]), "perm_p": float(sub["perm_p"])}

    X, y, g = cache_activations(m, triplets, fam)
    probes = fit_probes_all_layers(X, y, g)
    accs = [p.accuracy for p in probes]
    bundle = ResultBundle(
        exp_id="exp20_pt_probe_n100",
        config={"model_id": "gemma3-4b-pt", "dataset": "n100"},
        metrics={"accuracy_per_layer": accs, "best_layer": int(np.argmax(accs)), "best_acc": float(np.max(accs))},
        arrays={"X": X.astype(np.float32), "weights": np.stack([p.weights for p in probes])},
    )
    write_bundle(bundle)
    summary["probe_best_acc"] = float(np.max(accs))

    lens = logit_lens_bundle(m, triplets, fam, fmt="yesno", exp_id="exp20_pt_lens_n100")
    write_bundle(lens)

    # Direction alignment block
    sym, cond = collect_frame_acts(m, triplets, fam, fmt="yesno")
    d_frame = framing_direction_per_layer(torch.from_numpy(sym), torch.from_numpy(cond))
    ans = resolve_answer_tokens(m.tokenizer, "yesno")
    d_judge = judgment_direction(m, ans).cpu()
    align = layerwise_alignment(d_frame, d_judge).numpy()
    np.savez_compressed(
        out_dir / "pt_direction_block_n100.npz",
        d_frame=d_frame.numpy(), d_judge=d_judge.numpy(), alignment=align,
        sym_acts=sym, cond_acts=cond,
    )

    _cleanup(m)
    return summary


def phase_alignment_audit_n100(out_dir: Path) -> dict:
    from shingeki.audit.direction_alignment import compute_pt_it_alignment, write_alignment_audit

    pt = np.load(out_dir / "pt_direction_block_n100.npz")
    it = np.load(out_dir / "it_direction_block_n100.npz")
    pt_block = {"d_frame": pt["d_frame"], "d_judge": pt["d_judge"], "alignment_per_layer": pt["alignment"],
                "n_layers": int(pt["d_frame"].shape[0]), "d_model": int(pt["d_frame"].shape[1])}
    it_block = {"d_frame": it["d_frame"], "d_judge": it["d_judge"], "alignment_per_layer": it["alignment"],
                "n_layers": int(it["d_frame"].shape[0]), "d_model": int(it["d_frame"].shape[1])}
    write_alignment_audit(out_dir / "audit_alignment_n100", pt_block, it_block)
    return {"alignment": "ok"}


def phase_cross_model_n100(out_dir: Path) -> dict:
    """Run behavior + probe + lens on n=100 for Gemma-2-9B-IT, Gemma-3-12B-IT,
    Qwen-2.5-7B-IT, Qwen-2.5-14B-IT. Skips 27B (too long; will run if requested)."""
    from shingeki.behavior.scoring import behavioral_validation_bundle
    from shingeki.data.aot import load_combined_n100
    from shingeki.io.results import ResultBundle, write_bundle
    from shingeki.lens.logit_lens import logit_lens_bundle
    from shingeki.models.loader import load_hooked
    from shingeki.probing.linear import cache_activations, fit_probes_all_layers

    triplets = load_combined_n100()
    summary: dict = {}

    EXTRAS = [
        ("gemma2-9b-it", "gemma2-it", "exp23_gemma2_9b_n100"),
        ("gemma3-12b-it", "gemma3-it", "exp24_gemma3_12b_n100"),
        ("qwen25-7b-it", "qwen25-it", "exp25_qwen_7b_n100"),
        ("qwen25-14b-it", "qwen25-it", "exp26_qwen_14b_n100"),
    ]

    csv_path = out_dir / "cross_model_n100_summary.csv"
    csv_path.write_text("spec_id,n_layers,d_model,S_C_mean,S_C_median,S_C_perm_p,probe_best_acc\n")

    for spec_id, family, prefix in EXTRAS:
        try:
            print(f"\n[{_now()}] loading {spec_id} ...", flush=True)
            m = load_hooked(spec_id)
            print(f"  loaded; L={m.cfg.n_layers}", flush=True)

            beh = behavioral_validation_bundle(m, triplets, family, fmt="yesno", exp_id=f"{prefix}_behavior")
            write_bundle(beh)
            sub = beh.metrics["S_vs_C"]
            print(f"  S-C mean={sub['mean_shift']:+.3f} median={sub['median_shift']:+.3f} p={sub['perm_p']:.4f}", flush=True)

            X, y, g = cache_activations(m, triplets, family)
            probes = fit_probes_all_layers(X, y, g)
            accs = [p.accuracy for p in probes]
            bundle = ResultBundle(
                exp_id=f"{prefix}_probe", config={"model_id": spec_id, "dataset": "n100"},
                metrics={"accuracy_per_layer": accs, "best_layer": int(np.argmax(accs)), "best_acc": float(np.max(accs))},
                arrays={"X": X.astype(np.float32)},
            )
            write_bundle(bundle)
            print(f"  probe best L={int(np.argmax(accs))} acc={np.max(accs):.3f}", flush=True)

            lens = logit_lens_bundle(m, triplets, family, fmt="yesno", exp_id=f"{prefix}_lens")
            write_bundle(lens)

            with csv_path.open("a") as f:
                f.write(f"{spec_id},{m.cfg.n_layers},{m.cfg.d_model},"
                        f"{sub['mean_shift']:.4f},{sub['median_shift']:.4f},"
                        f"{sub['perm_p']:.4f},{float(np.max(accs)):.4f}\n")
            summary[spec_id] = {
                "S_C_mean": float(sub["mean_shift"]),
                "probe_best_acc": float(np.max(accs)),
            }
            _cleanup(m)
        except BaseException as e:
            print(f"  {spec_id} FAILED: {type(e).__name__}: {e}", flush=True)
            summary[spec_id] = {"error": f"{type(e).__name__}"}
    return summary


def phase_final_report_n100(out_dir: Path) -> dict:
    """Walk the n=100 artifacts and write a focused report."""
    md: list[str] = ["# n=100 results summary\n"]
    md.append(f"Generated: {_now()}\n")

    status_path = out_dir / "status.json"
    if status_path.exists():
        s = json.loads(status_path.read_text())
        for name, sub in s.get("phases", {}).items():
            md.append(f"- **{name}** ({sub.get('state','?')}, {sub.get('elapsed_s','?')}s)")
            for k, v in sub.items():
                if k in ("state", "started_at", "finished_at", "elapsed_s", "traceback", "error"):
                    continue
                md.append(f"  - {k}: {v}")
    csv_path = out_dir / "cross_model_n100_summary.csv"
    if csv_path.exists():
        md.append("\n## Cross-model n=100\n```\n" + csv_path.read_text().strip() + "\n```\n")
    out_path = out_dir / "n100_report.md"
    out_path.write_text("\n".join(md))
    return {"report": str(out_path)}


PHASES = [
    ("manifest_n100", phase_manifest_n100),
    ("it_n100", phase_it_n100),
    ("pt_n100", phase_pt_n100),
    ("alignment_audit_n100", phase_alignment_audit_n100),
    ("cross_model_n100", phase_cross_model_n100),
    ("final_report_n100", phase_final_report_n100),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--pid-file", default="/tmp/n100.pid")
    args = parser.parse_args(argv)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = PROJECT_ROOT / "runs" / f"n100_{_ts()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.pid_file).write_text(str(os.getpid()))
    print(f"n100 dir: {out_dir}", flush=True)
    print(f"pid: {os.getpid()}", flush=True)

    status_path = out_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"phases": {}}
    status["pid"] = os.getpid()
    status.setdefault("started_at", _now())
    only = set(args.only.split(",")) if args.only else None

    overall_t0 = time.time()
    for name, fn in PHASES:
        if only and name not in only:
            continue
        if _phase_done(status, name):
            print(f"[skip {name}] already completed", flush=True)
            continue
        _begin(status_path, status, name)
        t0 = time.time()
        try:
            extra = fn(out_dir) or {}
            extra["elapsed_s"] = round(time.time() - t0, 1)
            _ok(status_path, status, name, extra=extra)
        except SystemExit:
            raise
        except BaseException as e:
            _fail(status_path, status, name, e)
            continue

    status["finished_at"] = _now()
    status["total_elapsed_s"] = round(time.time() - overall_t0, 1)
    _save(status_path, status)
    print(f"\n=== n=100 RUN DONE in {time.time()-overall_t0:.1f}s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

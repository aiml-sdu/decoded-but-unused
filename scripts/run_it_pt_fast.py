"""Core IT+PT N=100 sweep: per-layer probe, logit lens, geometry, patching,
steering and projection ablation. Probes use StandardScaler and lbfgs with
C=0.01, max_iter=200, as reported in the paper.

Behaviour (exp01, exp20_pt_behavior) comes from run_n100.py. This script produces: exp04_probe_n100, exp05_logit_lens_n100, exp06_geometry_n100,
exp07_resid_S_to_C_n100, exp07_resid_C_to_S_n100, exp13_steering_LX_n100,
exp14_projection_ablation_LX_n100, exp20_pt_probe_n100, exp20_pt_lens_n100,
audit_random_subspace_n100, audit_alignment_n100, plus PT/IT direction blocks.
"""
import sys, os, time, gc, json
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from shingeki.audit.random_subspace import write_random_subspace_audit
from shingeki.behavior.scoring import behavioral_validation_bundle  # noqa
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
from shingeki.patching.residual import sweep_layers
from shingeki.activations.extract import get_residual_stream
from shingeki.prompts.builders import build_aot_prompt
from shingeki.steering.ablation import projection_ablation
from shingeki.steering.sweep import steering_sweep_bundle
from shingeki.probing.linear import ProbeResult
from shingeki.probing.splits import loto_splits

OUT = Path("runs/n100_20260508T161113Z")
OUT.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fast_cache_activations(model, triplets, family, fmt="yesno"):
    L, D = model.cfg.n_layers, model.cfg.d_model
    X = np.zeros((len(triplets) * 3, L, D), dtype=np.float32)
    y = np.empty(len(triplets) * 3, dtype="<U16")
    groups = np.empty(len(triplets) * 3, dtype="<U32")
    from shingeki.data.schemas import FRAMES
    row = 0
    t0 = time.time()
    for ti, t in enumerate(triplets):
        for f in FRAMES:
            p = build_aot_prompt(t, f, fmt, family, model.tokenizer, model.device)
            with torch.inference_mode():
                full = get_residual_stream(model, p, position="all", which="post")
            X[row] = full[:, -1].detach().to("cpu", torch.float32).numpy()
            y[row] = f
            groups[row] = t.id
            row += 1
        if (ti+1) % 20 == 0:
            print(f"    cached {ti+1}/{len(triplets)} triplets ({time.time()-t0:.1f}s)", flush=True)
    return X, y, groups


def fast_probes_all_layers(X, y, groups):
    """Fast binary probe per layer (sym vs cond, abst dropped).
    Uses StandardScaler + heavy-reg lbfgs for fast convergence."""
    L = X.shape[1]
    keep = (y == "sympathetic") | (y == "condemning")
    Xb = X[keep]
    yb = y[keep]
    gb = groups[keep]
    splits_b = loto_splits(gb)
    out = []
    t0 = time.time()
    for l in range(L):
        Xl_raw = Xb[:, l]
        scaler = StandardScaler()
        Xl = scaler.fit_transform(Xl_raw)
        per_fold = []
        preds = np.empty_like(yb)
        for tr, te in splits_b:
            clf = LogisticRegression(C=0.01, max_iter=200,
                                       solver="lbfgs", random_state=42)
            clf.fit(Xl[tr], yb[tr])
            preds[te] = clf.predict(Xl[te])
            per_fold.append(accuracy_score(yb[te], preds[te]))
        clf_all = LogisticRegression(C=0.01, max_iter=200,
                                       solver="lbfgs", random_state=42)
        clf_all.fit(Xl, yb)
        out.append(ProbeResult(
            layer=l,
            accuracy=float(accuracy_score(yb, preds)),
            macro_f1=float(f1_score(yb, preds, average="macro")),
            per_fold=[float(a) for a in per_fold],
            weights=clf_all.coef_.astype(np.float32),
            intercept=clf_all.intercept_.astype(np.float32),
            classes=list(clf_all.classes_),
        ))
        if (l+1) % 5 == 0:
            print(f"    probe L={l}/{L-1} acc={out[l].accuracy:.3f} ({time.time()-t0:.1f}s)", flush=True)
    return out


def run_it():
    print(f"\n=== [{_now()}] IT N=100 fast run ===", flush=True)
    m = load_hooked("gemma3-4b-it")
    fam = "gemma3-it"
    print(f"  loaded gemma3-4b-it L={m.cfg.n_layers} D={m.cfg.d_model}", flush=True)
    triplets = load_combined_n100()
    print(f"  triplets: {len(triplets)}", flush=True)

    # 4. Probe
    print(f"\n[{_now()}] caching activations ...", flush=True)
    t0 = time.time()
    X, y, g = fast_cache_activations(m, triplets, fam)
    print(f"  cache done ({time.time()-t0:.1f}s)", flush=True)
    t0 = time.time()
    print(f"[{_now()}] fitting probes ...", flush=True)
    probes = fast_probes_all_layers(X, y, g)
    accs = np.array([p.accuracy for p in probes])
    bundle = ResultBundle(
        exp_id="exp04_probe_n100",
        config={"model_id": "gemma3-4b-it", "dataset": "n100", "n_triplets": len(triplets)},
        metrics={
            "accuracy_per_layer": accs.tolist(),
            "best_layer": int(accs.argmax()),
            "best_acc": float(accs.max()),
        },
        arrays={
            "weights": np.stack([p.weights for p in probes]),
            "intercept": np.stack([p.intercept for p in probes]),
            "X_final": X.astype(np.float32),
            "y_classes": np.array([str(c) for c in probes[0].classes]),
        },
    )
    write_bundle(bundle)
    print(f"  IT probe best L={accs.argmax()} acc={accs.max():.3f}  ({time.time()-t0:.1f}s)", flush=True)

    # 5. Lens
    print(f"\n[{_now()}] logit lens ...", flush=True)
    t0 = time.time()
    lens = logit_lens_bundle(m, triplets, fam, fmt="yesno", exp_id="exp05_logit_lens_n100")
    write_bundle(lens)
    print(f"  lens done ({time.time()-t0:.1f}s)", flush=True)

    # 6. Geometry
    print(f"\n[{_now()}] geometry ...", flush=True)
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
        if (i+1) % 25 == 0:
            print(f"    geom collected {i+1}/{len(triplets)}", flush=True)
    d_frame = framing_direction_per_layer(torch.from_numpy(sym_acts), torch.from_numpy(cond_acts))
    d_judge = judgment_direction(m, ans).cpu()
    align = layerwise_alignment(d_frame, d_judge).numpy()
    geom = ResultBundle(
        exp_id="exp06_geometry_n100",
        config={"model_id": "gemma3-4b-it", "dataset": "n100"},
        metrics={
            "alignment_per_layer": align.tolist(),
            "peak_layer": int(np.argmax(np.abs(align))),
            "peak_alignment": float(align[int(np.argmax(np.abs(align)))]),
        },
        arrays={"d_frame": d_frame.numpy(), "d_judge": d_judge.numpy(), "alignment": align},
    )
    write_bundle(geom)
    np.savez_compressed(
        OUT / "it_direction_block_n100.npz",
        d_frame=d_frame.numpy(), d_judge=d_judge.numpy(), alignment=align,
        sym_acts=sym_acts, cond_acts=cond_acts,
    )
    print(f"  geom peak L={int(np.argmax(np.abs(align)))} cos={align[int(np.argmax(np.abs(align)))]:+.3f}  ({time.time()-t0:.1f}s)", flush=True)

    # 7. Residual patching both directions
    print(f"\n[{_now()}] residual patching S->C ...", flush=True)
    t0 = time.time()
    rsc = sweep_layers(m, triplets, fam, direction="S_to_C", fmt="yesno", exp_id="exp07_resid_S_to_C_n100")
    sc_dir = write_bundle(rsc)
    print(f"  S->C best L={rsc.metrics['best_layer']} rec={rsc.metrics['best_layer_recovery']:.3f}  ({time.time()-t0:.1f}s)", flush=True)
    t0 = time.time()
    rcs = sweep_layers(m, triplets, fam, direction="C_to_S", fmt="yesno", exp_id="exp07_resid_C_to_S_n100")
    cs_dir = write_bundle(rcs)
    print(f"  C->S best L={rcs.metrics['best_layer']} rec={rcs.metrics['best_layer_recovery']:.3f}  ({time.time()-t0:.1f}s)", flush=True)

    # 8. Steering at peak layer
    peak = int(rsc.metrics["best_layer"])
    target = peak if peak >= 20 else 28
    if target >= L: target = L - 4
    print(f"\n[{_now()}] steering at L={target} ...", flush=True)
    t0 = time.time()
    steer = steering_sweep_bundle(m, triplets, fam, layer=target, alpha=4.0, exp_id=f"exp13_steering_L{target}_n100")
    write_bundle(steer)
    print(f"  steering done  ({time.time()-t0:.1f}s)", flush=True)

    # 9. Projection ablation
    print(f"\n[{_now()}] projection ablation ...", flush=True)
    t0 = time.time()
    from shingeki.probing.linear import probe_direction
    d_probe = torch.from_numpy(probe_direction(probes[target])).float()
    proj_res = projection_ablation(m, triplets, fam, layer=target, direction=d_probe, fmt="yesno")
    write_bundle(ResultBundle(
        exp_id=f"exp14_projection_ablation_L{target}_n100",
        config={"layer": target, "model_id": "gemma3-4b-it", "dataset": "n100"},
        metrics=proj_res,
    ))
    print(f"  proj rank-1 shrink={proj_res['shrink_frac']:+.3f}  ({time.time()-t0:.1f}s)", flush=True)

    # 10. Random subspace audit
    print(f"\n[{_now()}] random subspace audit ...", flush=True)
    t0 = time.time()
    diffs = sym_acts[:, target] - cond_acts[:, target]
    diffs_centered = diffs - diffs.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
    pca_top4 = torch.from_numpy(Vt[:4]).float()
    write_random_subspace_audit(
        OUT / "audit_random_subspace_n100", m, triplets, fam,
        layer=target,
        true_directions={"probe_direction": d_probe, "pca_top4": pca_top4},
        ranks=(1, 4), n_seeds=15, fmt="yesno",
    )
    print(f"  audit done  ({time.time()-t0:.1f}s)", flush=True)

    del m
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"\n=== [{_now()}] IT N=100 DONE ===\n", flush=True)


def run_pt():
    print(f"\n=== [{_now()}] PT N=100 fast run ===", flush=True)
    from shingeki.audit.direction_alignment import collect_frame_acts
    m = load_hooked("gemma3-4b-pt")
    fam = "gemma3-pt"
    triplets = load_combined_n100()
    print(f"  loaded gemma3-4b-pt L={m.cfg.n_layers}", flush=True)

    # PT probe (use fast path)
    print(f"\n[{_now()}] PT caching activations ...", flush=True)
    t0 = time.time()
    X, y, g = fast_cache_activations(m, triplets, fam)
    print(f"  cache done ({time.time()-t0:.1f}s)", flush=True)
    t0 = time.time()
    probes = fast_probes_all_layers(X, y, g)
    accs = np.array([p.accuracy for p in probes])
    bundle = ResultBundle(
        exp_id="exp20_pt_probe_n100",
        config={"model_id": "gemma3-4b-pt", "dataset": "n100"},
        metrics={"accuracy_per_layer": accs.tolist(), "best_layer": int(accs.argmax()), "best_acc": float(accs.max())},
        arrays={"X": X.astype(np.float32), "weights": np.stack([p.weights for p in probes])},
    )
    write_bundle(bundle)
    print(f"  PT probe best L={accs.argmax()} acc={accs.max():.3f} ({time.time()-t0:.1f}s)", flush=True)

    # PT lens
    print(f"\n[{_now()}] PT lens ...", flush=True)
    t0 = time.time()
    lens = logit_lens_bundle(m, triplets, fam, fmt="yesno", exp_id="exp20_pt_lens_n100")
    write_bundle(lens)
    print(f"  PT lens done ({time.time()-t0:.1f}s)", flush=True)

    # PT direction block
    print(f"\n[{_now()}] PT direction block ...", flush=True)
    t0 = time.time()
    sym, cond = collect_frame_acts(m, triplets, fam, fmt="yesno")
    d_frame = framing_direction_per_layer(torch.from_numpy(sym), torch.from_numpy(cond))
    ans = resolve_answer_tokens(m.tokenizer, "yesno")
    d_judge = judgment_direction(m, ans).cpu()
    align = layerwise_alignment(d_frame, d_judge).numpy()
    np.savez_compressed(
        OUT / "pt_direction_block_n100.npz",
        d_frame=d_frame.numpy(), d_judge=d_judge.numpy(), alignment=align,
        sym_acts=sym, cond_acts=cond,
    )
    print(f"  PT align peak L={int(np.argmax(np.abs(align)))} cos={align[int(np.argmax(np.abs(align)))]:+.3f}  ({time.time()-t0:.1f}s)", flush=True)

    del m
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"\n=== [{_now()}] PT N=100 DONE ===\n", flush=True)


def run_alignment_audit():
    print(f"\n=== [{_now()}] alignment audit (PT vs IT) ===", flush=True)
    from shingeki.audit.direction_alignment import write_alignment_audit
    write_alignment_audit(
        OUT / "audit_alignment_n100",
        pt_block=OUT / "pt_direction_block_n100.npz",
        it_block=OUT / "it_direction_block_n100.npz",
    )
    print(f"=== [{_now()}] alignment audit DONE ===", flush=True)


if __name__ == "__main__":
    run_it()
    run_pt()
    try:
        run_alignment_audit()
    except Exception as e:
        print(f"alignment_audit failed: {type(e).__name__}: {e}", flush=True)

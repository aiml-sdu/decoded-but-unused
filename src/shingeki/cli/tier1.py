"""`shingeki tier1 ...` — behaviour, robustness, cosine, probe, lens, geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console

from shingeki.config import MANUSCRIPT_DIR
from shingeki.data.aot import load_d0, load_d1, load_d2
from shingeki.data.schemas import FRAMES, FramingTriplet
from shingeki.io.results import write_bundle
from shingeki.models.loader import load_hooked
from shingeki.models.registry import get
from shingeki.prompts.chat import ModelFamily

app = typer.Typer(help="Tier 1: behavioural, representational, timing experiments.")
console = Console()


def _triplets(name: str) -> list[FramingTriplet]:
    if name == "d0":
        return load_d0()
    if name == "d1":
        return load_d1(include_v2=True)
    if name == "d2":
        return load_d2()
    raise typer.BadParameter(f"unknown dataset {name!r}")


def _family(spec_id: str) -> ModelFamily:
    return get(spec_id).family  # type: ignore[return-value]


@app.command("behavior")
def behavior_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    dataset: str = typer.Option("d0"),
    fmt: str = typer.Option("yesno"),
    n: Optional[int] = typer.Option(None, help="limit to first n triplets"),
):
    """Exp 1 — behavioural validation."""
    from shingeki.behavior.scoring import behavioral_validation_bundle

    m = load_hooked(model_id)
    ts = _triplets(dataset)
    if n:
        ts = ts[:n]
    bundle = behavioral_validation_bundle(m, ts, _family(model_id), fmt=fmt, exp_id="exp01_behavior")
    out = write_bundle(bundle)
    console.print(f"[green]wrote[/] {out}")
    sub = bundle.metrics["S_vs_C"]
    console.print(
        f"S–C: mean={sub['mean_shift']:+.3f} CI95=[{sub['ci95_lo']:+.3f}, {sub['ci95_hi']:+.3f}] "
        f"p={sub['perm_p']:.4f} d_z={sub['cohens_dz']:+.2f} frac>0={sub['frac_sym_gt_cond']:.2f}"
    )


@app.command("robustness")
def robustness_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    dataset: str = typer.Option("d0"),
    formats: str = typer.Option("yesno,ab,justified,acceptable"),
):
    """Exp 2 — answer-token robustness across formats."""
    from shingeki.behavior.robustness import robustness_bundle

    fmts = tuple(s.strip() for s in formats.split(","))
    m = load_hooked(model_id)
    ts = _triplets(dataset)
    bundle = robustness_bundle(m, ts, _family(model_id), formats=fmts)  # type: ignore[arg-type]
    out = write_bundle(bundle)
    console.print(f"[green]wrote[/] {out}")
    for f, sub in bundle.metrics["per_format_S_vs_C"].items():
        console.print(f"  {f:>11s}: mean={sub['mean_shift']:+.3f} p={sub['perm_p']:.3f}")


@app.command("cosine")
def cosine_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    dataset: str = typer.Option("d0"),
    fmt: str = typer.Option("yesno"),
):
    """Exp 3 — cosine similarity baseline."""
    from shingeki.geometry.cosine import cosine_bundle

    m = load_hooked(model_id)
    ts = _triplets(dataset)
    bundle = cosine_bundle(m, ts, _family(model_id), fmt=fmt)
    out = write_bundle(bundle)
    console.print(f"[green]wrote[/] {out}")


@app.command("probe")
def probe_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    dataset: str = typer.Option("d0"),
    fmt: str = typer.Option("yesno"),
    layers: Optional[str] = typer.Option(None, help="comma-separated, e.g. 0,8,16,24,32"),
):
    """Exp 4 — linear probes with LOTO + controls."""
    from shingeki.io.results import ResultBundle
    from shingeki.probing.controls import length_only_baseline, random_label_baseline, tfidf_baseline
    from shingeki.probing.linear import cache_activations, fit_probes_all_layers

    m = load_hooked(model_id)
    ts = _triplets(dataset)
    fam = _family(model_id)

    X, y, groups = cache_activations(m, ts, fam, fmt=fmt)
    layer_list = (
        [int(x) for x in layers.split(",")] if layers else list(range(m.cfg.n_layers))
    )
    probes = fit_probes_all_layers(X, y, groups, layers=layer_list)
    accs = np.array([p.accuracy for p in probes])
    f1s = np.array([p.macro_f1 for p in probes])

    texts: list[str] = []
    for t in ts:
        for f in FRAMES:
            texts.append(getattr(t, f))
    rand_ctl = random_label_baseline(X[:, layer_list[len(layer_list) // 2]], y, groups)
    len_ctl = length_only_baseline(texts, y, groups)
    tfidf_ctl = tfidf_baseline(texts, y, groups)

    bundle = ResultBundle(
        exp_id="exp04_probe",
        config={
            "model_id": model_id,
            "dataset": dataset,
            "fmt": fmt,
            "n_triplets": len(ts),
            "layers": layer_list,
        },
        metrics={
            "accuracy_per_layer": accs.tolist(),
            "macro_f1_per_layer": f1s.tolist(),
            "best_layer": int(layer_list[int(accs.argmax())]),
            "best_acc": float(accs.max()),
            "controls": {
                "random_label": rand_ctl,
                "length_only": len_ctl,
                "tfidf": tfidf_ctl,
            },
        },
        arrays={
            "weights": np.stack([p.weights for p in probes]),
            "intercept": np.stack([p.intercept for p in probes]),
            "X_final": X.astype(np.float32),
            "y_classes": np.array([str(c) for c in probes[0].classes]),
        },
    )
    out = write_bundle(bundle)
    console.print(f"[green]wrote[/] {out}; best layer={bundle.metrics['best_layer']} acc={bundle.metrics['best_acc']:.3f}")


@app.command("lens")
def lens_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    dataset: str = typer.Option("d0"),
    fmt: str = typer.Option("yesno"),
):
    """Exp 5 — logit-lens trajectory."""
    from shingeki.lens.logit_lens import logit_lens_bundle

    m = load_hooked(model_id)
    ts = _triplets(dataset)
    bundle = logit_lens_bundle(m, ts, _family(model_id), fmt=fmt)
    out = write_bundle(bundle)
    console.print(f"[green]wrote[/] {out}")


@app.command("geometry")
def geometry_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    dataset: str = typer.Option("d0"),
    fmt: str = typer.Option("yesno"),
):
    """Exp 6 — framing-vs-judgment alignment by layer."""
    import torch

    from shingeki.activations.extract import get_residual_stream
    from shingeki.geometry.directions import (
        framing_direction_per_layer,
        judgment_direction,
        layerwise_alignment,
    )
    from shingeki.io.results import ResultBundle
    from shingeki.models.tokens import resolve_answer_tokens
    from shingeki.prompts.builders import build_aot_prompt

    m = load_hooked(model_id)
    ts = _triplets(dataset)
    fam = _family(model_id)
    ans = resolve_answer_tokens(m.tokenizer, fmt)

    # Build per-frame layer activations [N, L, D]
    L = m.cfg.n_layers
    D = m.cfg.d_model
    sym = np.zeros((len(ts), L, D), dtype=np.float32)
    cond = np.zeros((len(ts), L, D), dtype=np.float32)
    for i, t in enumerate(ts):
        ps = build_aot_prompt(t, "sympathetic", fmt, fam, m.tokenizer, m.device)
        pc = build_aot_prompt(t, "condemning", fmt, fam, m.tokenizer, m.device)
        with torch.inference_mode():
            sym[i] = get_residual_stream(m, ps, position="final", which="post").float().cpu().numpy()
            cond[i] = get_residual_stream(m, pc, position="final", which="post").float().cpu().numpy()
    d_frame = framing_direction_per_layer(torch.from_numpy(sym), torch.from_numpy(cond))  # [L, D]
    d_judge = judgment_direction(m, ans).cpu()
    align = layerwise_alignment(d_frame, d_judge).numpy()
    bundle = ResultBundle(
        exp_id="exp06_geometry",
        config={"model_id": model_id, "dataset": dataset, "fmt": fmt, "n_triplets": len(ts)},
        metrics={
            "alignment_per_layer": align.tolist(),
            "peak_layer": int(np.argmax(np.abs(align))),
            "peak_alignment": float(align[int(np.argmax(np.abs(align)))]),
        },
        arrays={"d_frame": d_frame.numpy(), "d_judge": d_judge.numpy(), "alignment": align},
    )
    out = write_bundle(bundle)
    console.print(f"[green]wrote[/] {out}; peak layer {bundle.metrics['peak_layer']} cos={bundle.metrics['peak_alignment']:+.3f}")

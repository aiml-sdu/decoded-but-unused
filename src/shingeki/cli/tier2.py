"""`shingeki tier2 ...` — patching, steering, ablation."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from shingeki.cli.tier1 import _family, _triplets
from shingeki.io.results import write_bundle
from shingeki.models.loader import load_hooked

app = typer.Typer(help="Tier 2: causal interventions.")
console = Console()


@app.command("patch-resid")
def patch_resid_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    dataset: str = typer.Option("d0"),
    direction: str = typer.Option("S_to_C", help="S_to_C or C_to_S"),
    fmt: str = typer.Option("yesno"),
    n: Optional[int] = typer.Option(None),
):
    """Exp 7 — residual-stream patching, final token, all layers."""
    from shingeki.patching.residual import sweep_layers

    m = load_hooked(model_id)
    ts = _triplets(dataset)
    if n:
        ts = ts[:n]
    bundle = sweep_layers(m, ts, _family(model_id), direction=direction, fmt=fmt)
    out = write_bundle(bundle)
    console.print(f"[green]wrote[/] {out}; best L={bundle.metrics['best_layer']} rec={bundle.metrics['best_layer_recovery']:.3f}")


@app.command("patch-component")
def patch_component_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    dataset: str = typer.Option("d0"),
    direction: str = typer.Option("S_to_C"),
    layers: Optional[str] = typer.Option(None, help="comma-separated layer indices"),
):
    """Exp 8 — component patching across layers."""
    from shingeki.patching.component import component_layer_heatmap

    m = load_hooked(model_id)
    ts = _triplets(dataset)
    layer_list = [int(x) for x in layers.split(",")] if layers else list(range(int(0.5 * m.cfg.n_layers), m.cfg.n_layers))
    bundle = component_layer_heatmap(m, ts, _family(model_id), direction=direction, layers=layer_list)
    out = write_bundle(bundle)
    console.print(f"[green]wrote[/] {out}; best={bundle.metrics['best']}")


@app.command("patch-token")
def patch_token_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    direction: str = typer.Option("S_to_C"),
    layers: Optional[str] = typer.Option(None),
):
    """Exp 9 — token-position patching."""
    from shingeki.patching.token_position import token_position_heatmap

    m = load_hooked(model_id)
    ts = _triplets("d0")
    layer_list = [int(x) for x in layers.split(",")] if layers else list(range(int(0.5 * m.cfg.n_layers), m.cfg.n_layers))
    bundle = token_position_heatmap(m, ts, _family(model_id), direction=direction, layers=layer_list)
    out = write_bundle(bundle)
    console.print(f"[green]wrote[/] {out}")


@app.command("attn-source")
def attn_source_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    layers: Optional[str] = typer.Option(None),
):
    """Exp 10 — attention-source analysis."""
    from shingeki.patching.attention_source import attention_source_bundle

    m = load_hooked(model_id)
    ts = _triplets("d0")
    layer_list = [int(x) for x in layers.split(",")] if layers else None
    bundle = attention_source_bundle(m, ts, _family(model_id), layers=layer_list)
    out = write_bundle(bundle)
    console.print(f"[green]wrote[/] {out}")


@app.command("patch-head")
def patch_head_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    direction: str = typer.Option("S_to_C"),
    layers: Optional[str] = typer.Option(None),
):
    """Exp 11 — per-head patching."""
    from shingeki.patching.head import sweep_heads

    m = load_hooked(model_id)
    ts = _triplets("d0")
    layer_list = [int(x) for x in layers.split(",")] if layers else None
    bundle = sweep_heads(m, ts, _family(model_id), layers=layer_list, direction=direction)
    out = write_bundle(bundle)
    console.print(f"[green]wrote[/] {out}")


@app.command("mlp-neuron")
def mlp_neuron_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    layer: int = typer.Option(...),
    k_values: str = typer.Option("16,64,256"),
):
    """Exp 12 — MLP neuron localisation."""
    from shingeki.patching.mlp_neuron import neuron_localization_bundle

    m = load_hooked(model_id)
    ts = _triplets("d0")
    ks = tuple(int(x) for x in k_values.split(","))
    bundle = neuron_localization_bundle(m, ts, _family(model_id), layer=layer, k_values=ks)
    out = write_bundle(bundle)
    console.print(f"[green]wrote[/] {out}")


@app.command("steer")
def steer_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    dataset: str = typer.Option("d0"),
    layer: int = typer.Option(...),
    alpha: float = typer.Option(4.0),
):
    """Exp 13 — steering sweep (8 directions × 3 modes + controls)."""
    from shingeki.steering.sweep import steering_sweep_bundle

    m = load_hooked(model_id)
    ts = _triplets(dataset)
    bundle = steering_sweep_bundle(m, ts, _family(model_id), layer=layer, alpha=alpha)
    out = write_bundle(bundle)
    console.print(f"[green]wrote[/] {out}")
    for label, sub in bundle.metrics.items():
        if isinstance(sub, dict) and "shrink_frac" in sub:
            console.print(f"  {label:>30s}: shrink={sub['shrink_frac']:+.3f}")


@app.command("ablate")
def ablate_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    layer: int = typer.Option(...),
    probe_run_dir: str = typer.Option(..., help="path to a written exp04_probe bundle"),
    pos_class: str = typer.Option("sympathetic"),
    neg_class: str = typer.Option("condemning"),
):
    """Exp 14 — projection ablation using a saved probe direction."""
    import json
    from pathlib import Path

    import numpy as np
    import torch

    from shingeki.io.results import ResultBundle
    from shingeki.steering.ablation import projection_ablation

    m = load_hooked(model_id)
    ts = _triplets("d0")

    pdir = Path(probe_run_dir)
    weights = np.load(pdir / "arrays.npz")["weights"]  # [L, K, D]
    classes = np.load(pdir / "arrays.npz")["y_classes"].tolist()
    pos_idx = classes.index(pos_class)
    neg_idx = classes.index(neg_class)
    direction = torch.from_numpy(weights[layer, pos_idx] - weights[layer, neg_idx]).float()
    res = projection_ablation(m, ts, _family(model_id), layer=layer, direction=direction)
    bundle = ResultBundle(
        exp_id="exp14_projection_ablation",
        config={"layer": layer, "model_id": model_id, "probe_run_dir": str(pdir)},
        metrics=res,
    )
    out = write_bundle(bundle)
    console.print(f"[green]wrote[/] {out}")
    console.print(f"S–C baseline = {res['S_C_baseline']:+.3f}, ablated = {res['S_C_ablated']:+.3f}, shrink = {res['shrink_frac']:.3f}")

"""`shingeki run ...` - composite runners."""

from __future__ import annotations

import typer
from rich.console import Console

from shingeki.cli import tier1, tier2, tier3

app = typer.Typer(help="Composite runners.")
console = Console()


@app.command("submission-min")
def submission_min(
    model_id: str = typer.Option("gemma3-4b-it"),
    fmt: str = typer.Option("yesno"),
):
    """Minimal run: all Tier-1 experiments plus the key Tier-2 and Tier-3 pieces."""
    tier1.behavior_cmd(model_id=model_id, dataset="d0", fmt=fmt)
    tier1.robustness_cmd(model_id=model_id, dataset="d0", formats="yesno,ab,justified,acceptable")
    tier1.cosine_cmd(model_id=model_id, dataset="d0", fmt=fmt)
    tier1.probe_cmd(model_id=model_id, dataset="d0", fmt=fmt)
    tier1.lens_cmd(model_id=model_id, dataset="d0", fmt=fmt)
    tier1.geometry_cmd(model_id=model_id, dataset="d0", fmt=fmt)
    # style + anonymized
    tier3.style_cmd(model_id=model_id, fmt=fmt)
    tier3.anon_cmd(model_id=model_id, fmt=fmt)
    # residual patching
    tier2.patch_resid_cmd(model_id=model_id, dataset="d0", direction="S_to_C", fmt=fmt)
    tier2.patch_resid_cmd(model_id=model_id, dataset="d0", direction="C_to_S", fmt=fmt)
    # MoralChoice
    tier3.moralchoice_cmd(model_id=model_id)


@app.command("deep-discovery")
def deep_discovery(
    model_id: str = typer.Option("gemma3-4b-it"),
    component_layer: str = typer.Option(None, help="comma-separated layer list for component patching"),
    steer_layer: int = typer.Option(28),
):
    """Follow-up run: the remaining Tier-2 and Tier-3 experiments."""
    tier2.patch_component_cmd(model_id=model_id, dataset="d0", direction="S_to_C", layers=component_layer)
    tier2.patch_token_cmd(model_id=model_id, direction="S_to_C", layers=component_layer)
    tier2.attn_source_cmd(model_id=model_id, layers=component_layer)
    tier2.patch_head_cmd(model_id=model_id, direction="S_to_C", layers=component_layer)
    tier2.steer_cmd(model_id=model_id, dataset="d0", layer=steer_layer)
    tier3.gemma_pt_vs_it_cmd()
    tier3.gemma2_cmd()
    tier3.qwen_cmd()


@app.command("all")
def all_cmd(model_id: str = typer.Option("gemma3-4b-it"), steer_layer: int = typer.Option(28)):
    """Run submission-min then deep-discovery on the same model."""
    submission_min(model_id=model_id)
    deep_discovery(model_id=model_id, steer_layer=steer_layer)

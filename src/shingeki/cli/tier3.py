"""`shingeki tier3 ...` — generality experiments."""

from __future__ import annotations

import typer
from rich.console import Console

from shingeki.cli import tier1, tier2

app = typer.Typer(help="Tier 3: generality and cross-model replication.")
console = Console()


@app.command("anon")
def anon_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    fmt: str = typer.Option("yesno"),
):
    """Exp 16 — anonymized AoT (D2): behaviour + probe + lens."""
    tier1.behavior_cmd(model_id=model_id, dataset="d2", fmt=fmt, n=None)
    tier1.probe_cmd(model_id=model_id, dataset="d2", fmt=fmt, layers=None)
    tier1.lens_cmd(model_id=model_id, dataset="d2", fmt=fmt)


@app.command("style")
def style_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    fmt: str = typer.Option("yesno"),
):
    """Exp 17 — style-controlled (D1): behaviour + probe + lens."""
    tier1.behavior_cmd(model_id=model_id, dataset="d1", fmt=fmt, n=None)
    tier1.probe_cmd(model_id=model_id, dataset="d1", fmt=fmt, layers=None)
    tier1.lens_cmd(model_id=model_id, dataset="d1", fmt=fmt)


@app.command("moralchoice")
def moralchoice_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
):
    """Exp 18 — MoralChoice-Frame ambiguity test (A/B logit gap)."""
    from shingeki.behavior.scoring import score_moralchoice
    from shingeki.cli.tier1 import _family
    from shingeki.data.moralchoice import load_d3
    from shingeki.io.results import ResultBundle, write_bundle
    from shingeki.models.loader import load_hooked

    m = load_hooked(model_id)
    items = load_d3()
    if not items:
        console.print("[red]D3 not built — run `shingeki data build-d3` first.[/]")
        return
    rows = score_moralchoice(m, items, _family(model_id))
    bundle = ResultBundle(
        exp_id="exp18_moralchoice",
        config={"model_id": model_id, "n_items": len(items)},
        metrics={"rows": rows},
    )
    out = write_bundle(bundle)
    console.print(f"[green]wrote[/] {out}")


@app.command("ethics")
def ethics_cmd(
    model_id: str = typer.Option("gemma3-4b-it"),
    n: int = typer.Option(50),
    subset: str = typer.Option("commonsense"),
):
    """Exp 19 — ETHICS appendix sanity check."""
    from shingeki.data.ethics import load_d4

    rows = load_d4(subset=subset, n=n)
    console.print(f"loaded {len(rows)} rows from ETHICS/{subset}; appendix run not yet wired (stub)")


@app.command("gemma-pt-vs-it")
def gemma_pt_vs_it_cmd(fmt: str = typer.Option("yesno")):
    """Exp 20 — PT vs IT comparison."""
    tier1.behavior_cmd(model_id="gemma3-4b-it", dataset="d0", fmt=fmt)
    tier1.behavior_cmd(model_id="gemma3-4b-pt", dataset="d0", fmt=fmt)
    tier1.probe_cmd(model_id="gemma3-4b-it", dataset="d0", fmt=fmt)
    tier1.probe_cmd(model_id="gemma3-4b-pt", dataset="d0", fmt=fmt)


@app.command("gemma2")
def gemma2_cmd(fmt: str = typer.Option("yesno")):
    """Exp 21 — Gemma-2 2B-IT cheap replication."""
    tier1.behavior_cmd(model_id="gemma2-2b-it", dataset="d0", fmt=fmt)
    tier1.probe_cmd(model_id="gemma2-2b-it", dataset="d0", fmt=fmt)
    tier1.lens_cmd(model_id="gemma2-2b-it", dataset="d0", fmt=fmt)


@app.command("qwen")
def qwen_cmd(fmt: str = typer.Option("yesno")):
    """Exp 22 — Qwen-2.5 3B-Instruct cross-family check."""
    tier1.behavior_cmd(model_id="qwen25-3b-it", dataset="d0", fmt=fmt)
    tier1.probe_cmd(model_id="qwen25-3b-it", dataset="d0", fmt=fmt)
    tier1.lens_cmd(model_id="qwen25-3b-it", dataset="d0", fmt=fmt)

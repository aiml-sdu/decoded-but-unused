"""`shingeki data ...` subcommands."""

from __future__ import annotations

import typer
from rich.console import Console

from shingeki.config import D2_PATH, D3_PATH, DATA_DIR

app = typer.Typer(help="Dataset builders.")
console = Console()


@app.command("build-d1")
def build_d1_cmd():
    """Print the curation template; v2 expansion is human work (see stub_template.md)."""
    p = DATA_DIR / "d1_style_controlled" / "stub_template.md"
    console.print(f"D1 expansion is curated by hand. See [bold]{p}[/].")


@app.command("build-d2")
def build_d2_cmd(n: int = typer.Option(None, help="limit to first n triplets")):
    """Build the anonymized AoT JSONL (D2)."""
    from shingeki.data.curate_d2 import build_d2

    out = build_d2(n=n)
    console.print(f"Wrote [bold]{D2_PATH}[/] with {len(out)} triplets.")


@app.command("build-d3")
def build_d3_cmd(
    n_high: int = typer.Option(20, help="number of high-ambiguity items"),
    n_low: int = typer.Option(20, help="number of low-ambiguity items"),
):
    """Build the MoralChoice-Frame JSONL (D3)."""
    from shingeki.data.moralchoice import build_d3, write_d3, write_prefix_templates

    write_prefix_templates(D3_PATH.parent / "prefix_templates.json")
    items = build_d3(n_high=n_high, n_low=n_low)
    write_d3(items)
    console.print(f"Wrote [bold]{D3_PATH}[/] with {len(items)} wrapped items.")


@app.command("annotate-tokens")
def annotate_tokens_cmd(
    model_id: str = typer.Option("gemma3-4b-it", help="HookedModel id (for tokenizer)"),
):
    """Run the token-category annotator on D0 (resumable)."""
    from shingeki.data.annotators import annotate_dataset
    from shingeki.data.aot import load_d0
    from shingeki.models.loader import load_hooked

    m = load_hooked(model_id)
    triplets = load_d0()
    out = annotate_dataset(triplets, m.tokenizer)
    console.print(f"Annotated {len(out)} (triplet, frame) pairs.")

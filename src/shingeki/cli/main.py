"""`shingeki` Typer entry point."""

from __future__ import annotations

import typer

from shingeki.cli import all as all_app
from shingeki.cli import data_cmds, tier1, tier2, tier3

app = typer.Typer(
    no_args_is_help=True,
    help="shingeki-no-features: mechanistic study of moral framing in LLMs.",
)
app.add_typer(data_cmds.app, name="data")
app.add_typer(tier1.app, name="tier1")
app.add_typer(tier2.app, name="tier2")
app.add_typer(tier3.app, name="tier3")
app.add_typer(all_app.app, name="run")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

"""Consistent matplotlib style for publication figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

ICML_COL_INCH = 3.25
ICML_FULL_INCH = 6.75


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


PALETTE = {
    "sympathetic": "#2a9d8f",
    "condemning": "#e76f51",
    "abstract": "#264653",
    "S-C": "#7b1d3f",
    "judgment": "#a08d3a",
    "framing": "#3a6c8c",
    "random": "#9c9c9c",
}

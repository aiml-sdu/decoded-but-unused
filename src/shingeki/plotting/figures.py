"""Manuscript figures.

Each function takes a `ResultBundle` (or several) and returns the path to a
saved PDF (PNG also written for previews). The figures are sized to ICML
2026 column widths: 3.25 inches for `figure`, ~6.75 inches for `figure*`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shingeki.io.results import ResultBundle
from shingeki.plotting.style import ICML_COL_INCH, ICML_FULL_INCH, PALETTE, set_style


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    return pdf


def fig1_dataset_and_behavior(
    behavior: ResultBundle, out_dir: Path, *, name: str = "fig1_dataset_behavior"
) -> Path:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(ICML_FULL_INCH, 2.4))

    sym = np.asarray(behavior.arrays["sym"])
    cond = np.asarray(behavior.arrays["cond"])
    n = len(sym)
    sub = behavior.metrics["S_vs_C"]

    ax = axes[0]
    ax.scatter(cond, sym, s=22, alpha=0.65, color=PALETTE["S-C"], edgecolors="white", linewidths=0.4)
    lim = max(abs(sym).max(), abs(cond).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="0.6", linestyle="--", linewidth=0.7, zorder=0)
    ax.axhline(0, color="0.85", linewidth=0.5, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.5, zorder=0)
    ax.set_xlabel("Condemning prompt:  $\\log p(\\text{Yes}) - \\log p(\\text{No})$")
    ax.set_ylabel("Sympathetic prompt", labelpad=2)
    ax.set_title("Paired Yes / No shift across triplets", fontsize=9, pad=4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(False)

    diffs = sym - cond
    ax = axes[1]
    bins = np.linspace(diffs.min(), diffs.max(), 16)
    ax.hist(diffs, bins=bins, color=PALETTE["S-C"], alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="0.5", linewidth=0.6)
    ax.axvline(diffs.mean(), color="black", linewidth=1.0)
    ax.set_xlabel("$S - C$ shift  (logit gap)")
    ax.set_ylabel("\\# triplets", labelpad=2)
    ax.set_title(
        f"$N{{=}}{n}$, mean $={sub['mean_shift']:+.2f}$, frac $S>C={sub['frac_sym_gt_cond']:.2f}$, $p={sub['perm_p']:.0e}$",
        fontsize=8,
        pad=4,
    )
    ax.grid(False)

    fig.tight_layout()
    return _save(fig, out_dir, name)


def fig2_cosine_vs_probe(
    cosine: ResultBundle,
    probe_acc_per_layer: dict[str, np.ndarray] | None,
    out_dir: Path,
    *,
    name: str = "fig2_cosine_vs_probe",
) -> Path:
    set_style()
    has_probe = bool(probe_acc_per_layer)
    fig, axes = plt.subplots(
        1, 2 if has_probe else 1, figsize=(ICML_FULL_INCH, 2.3)
    )
    if not has_probe:
        axes = [axes]

    L = len(cosine.metrics["final_token"]["S_vs_C_mean_per_layer"])
    layers = np.arange(L)
    ax = axes[0]
    ax.plot(
        layers,
        cosine.metrics["final_token"]["S_vs_C_mean_per_layer"],
        color=PALETTE["S-C"],
        linewidth=1.4,
        label="final-token",
    )
    ax.plot(
        layers,
        cosine.metrics["mean_pooled"]["S_vs_C_mean_per_layer"],
        color=PALETTE["framing"],
        linestyle="--",
        linewidth=1.0,
        label="mean-pooled",
    )
    ax.set_ylim(0.0, 1.04)
    ax.set_xlabel("Layer")
    ax.set_ylabel("$\\cos(\\bar{S},\\,\\bar{C})$", labelpad=1)
    ax.set_title("Coarse cosine misses framing", fontsize=9, pad=4)
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    if has_probe:
        ax = axes[1]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
        for (label, accs), c in zip(probe_acc_per_layer.items(), colors, strict=False):
            ax.plot(np.arange(len(accs)), accs, color=c, linewidth=1.2, label=label)
        ax.axhline(1 / 3, color="0.55", linestyle=":", linewidth=0.7, label="chance")
        ax.set_xlabel("Layer")
        ax.set_ylabel("LOTO probe accuracy", labelpad=1)
        ax.set_ylim(0.25, 1.04)
        ax.set_title("Probes decode framing", fontsize=9, pad=4)
        ax.legend(frameon=False, loc="lower right", handlelength=1.6, fontsize=7)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    fig.tight_layout()
    return _save(fig, out_dir, name)


def fig3_late_readout(
    lens: ResultBundle,
    geom_alignment: np.ndarray | None,
    out_dir: Path,
    *,
    name: str = "fig3_late_readout",
) -> Path:
    set_style()
    cols = 3 if geom_alignment is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(ICML_FULL_INCH, 2.2))

    L = len(lens.metrics["sympathetic"]["mean_per_layer"])
    layers = np.arange(L)

    ax = axes[0]
    for f in ("sympathetic", "condemning", "abstract"):
        ax.plot(layers, lens.metrics[f]["mean_per_layer"], color=PALETTE[f], linewidth=1.2, label=f)
    ax.axhline(0, color="0.5", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("$\\log p(\\text{Yes}) - \\log p(\\text{No})$", labelpad=1)
    ax.set_title("Logit-lens trajectories", fontsize=9, pad=4)
    ax.legend(frameon=False, loc="upper left", fontsize=7, handlelength=1.4)
    ax.grid(True, linestyle=":", alpha=0.35)

    ax = axes[1]
    sc = np.array(lens.metrics["sym_minus_cond_per_layer"])
    ax.fill_between(layers, sc, 0, color=PALETTE["S-C"], alpha=0.18)
    ax.plot(layers, sc, color=PALETTE["S-C"], linewidth=1.4)
    ax.axhline(0, color="0.5", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("$S - C$ lensed gap", labelpad=1)
    ax.set_title("Readout transition", fontsize=9, pad=4)
    ax.grid(True, linestyle=":", alpha=0.35)

    if geom_alignment is not None:
        ax = axes[2]
        ax.plot(layers, geom_alignment, color=PALETTE["judgment"], linewidth=1.4)
        ax.axhline(0, color="0.5", linewidth=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("$\\cos(d_{\\mathrm{frame}}, d_{\\mathrm{judge}})$", labelpad=1)
        ax.set_title("Framing $\\times$ judgment", fontsize=9, pad=4)
        ax.grid(True, linestyle=":", alpha=0.35)

    fig.tight_layout()
    return _save(fig, out_dir, name)


def fig4_causal(
    component_heatmap: ResultBundle | None,
    resid_recovery: ResultBundle | None,
    out_dir: Path,
    *,
    name: str = "fig4_causal",
) -> Path:
    set_style()
    if component_heatmap is not None:
        rec = np.array(component_heatmap.metrics["mean_recovery"])  # [C, L]
        comps = component_heatmap.config["components"]
        layers = component_heatmap.config["layers"]
        fig, ax = plt.subplots(figsize=(ICML_COL_INCH, 1.8))
        im = ax.imshow(
            rec,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            origin="upper",
            extent=[layers[0] - 0.5, layers[-1] + 0.5, len(comps) - 0.5, -0.5],
        )
        ax.set_yticks(range(len(comps)))
        ax.set_yticklabels(comps)
        ax.set_xlabel("Layer")
        ax.set_title("Component $\\times$ layer recovery", fontsize=9, pad=4)
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cbar.set_label("recovery", fontsize=7)
        cbar.ax.tick_params(labelsize=7)
        fig.tight_layout()
        return _save(fig, out_dir, name)

    assert resid_recovery is not None, "need component heatmap or resid recovery"
    fig, ax = plt.subplots(figsize=(ICML_COL_INCH, 1.8))
    layers = resid_recovery.config["layers"]
    median = np.array(resid_recovery.metrics["median_recovery_per_layer"])
    mean = np.array(resid_recovery.metrics["mean_recovery_per_layer"])
    ax.plot(layers, median, color=PALETTE["S-C"], linewidth=1.4, label="median")
    ax.plot(layers, mean, color=PALETTE["framing"], linestyle="--", linewidth=0.9, label="mean")
    ax.axhline(0, color="0.5", linewidth=0.5)
    ax.axhline(1.0, color="0.5", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Recovery (S\\,$\\to$\\,C)", labelpad=1)
    ax.set_title("Residual-stream patching", fontsize=9, pad=4)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4)
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def fig5_steering(
    steering: ResultBundle,
    out_dir: Path,
    *,
    name: str = "fig5_steering",
) -> Path:
    """Horizontal bar chart of `shrink_frac` across (intervention, direction)."""
    set_style()
    rows: list[tuple[str, float]] = []
    for label, sub in steering.metrics.items():
        if isinstance(sub, dict) and "shrink_frac" in sub:
            rows.append((label, float(sub["shrink_frac"])))
    rows.sort(key=lambda p: p[1])

    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(ICML_COL_INCH, 0.18 * max(6, len(rows)) + 0.6))
    colors = []
    for lab, v in rows:
        if "random" in lab:
            colors.append("#9c9c9c")
        elif "pca" in lab:
            colors.append(PALETTE["S-C"])
        elif "judgment" in lab:
            colors.append(PALETTE["judgment"])
        else:
            colors.append(PALETTE["framing"])
    ax.barh(range(len(rows)), vals, color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="0.4", linewidth=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("$1 - (S{-}C)_{\\text{after}}/(S{-}C)_{\\text{baseline}}$  (shrink)")
    layer = steering.config.get("layer", "?")
    ax.set_title(f"Interventions at layer {layer}", fontsize=9, pad=4)
    ax.grid(True, axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def fig6_cross_model(
    rows: list[dict],
    out_dir: Path,
    *,
    name: str = "fig6_cross_model",
) -> Path:
    set_style()
    fig, ax = plt.subplots(figsize=(ICML_COL_INCH, 0.32 * len(rows) + 0.6))
    sym_colors = {"Gemma-3-4B-IT": PALETTE["S-C"], "Gemma-3-4B-PT": PALETTE["framing"],
                  "Gemma-2-2B-IT": PALETTE["judgment"], "Qwen-2.5-3B-IT": "#404040"}
    colors = []
    for r in rows:
        # match the model prefix
        c = "#7c7c7c"
        for prefix, col in sym_colors.items():
            if r["label"].startswith(prefix):
                c = col
                break
        colors.append(c)
    means = [r["S_C"] for r in rows]
    err_lo = [m - r["ci_lo"] for m, r in zip(means, rows)]
    err_hi = [r["ci_hi"] - m for m, r in zip(means, rows)]
    ax.barh(
        range(len(rows)),
        means,
        xerr=[err_lo, err_hi],
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        error_kw={"elinewidth": 0.7, "capsize": 2.0},
    )
    ax.axvline(0, color="0.4", linewidth=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=7)
    ax.set_xlabel("$S - C$  logit shift  (95\\% CI)")
    ax.set_title("Cross-model behavioural shift", fontsize=9, pad=4)
    ax.grid(True, axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    return _save(fig, out_dir, name)

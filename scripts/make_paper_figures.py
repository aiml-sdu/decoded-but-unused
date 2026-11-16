"""Generate all paper figures with matplotlib + seaborn (Chisel style).

One script, six figures, consistent palette and rcParams:
  figure1.pdf  schematic (PT vs IT alignment cartoon)
  figure2.pdf  probe acc + alignment cosine per layer (PT vs IT)
  figure3.pdf  cross-model behaviour + alignment cosine (2-panel, full width)
  figure4.pdf  consolidated intervention evidence
  figure5.pdf  standalone subspace ablation backup
  figure7.pdf  standalone component x layer patching backup

ICML column width = 3.25 in single, 6.75 in full text width.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- ICML-style global rcParams ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.2,
    "ytick.minor.size": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.which": "both",
    "grid.alpha": 0.18,
    "grid.linewidth": 0.4,
    "grid.linestyle": ":",
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def add_minor_grid(ax, axis="y", minor_alpha=0.10):
    """Subtle minor gridlines for fine reading of small values."""
    from matplotlib.ticker import AutoMinorLocator
    if axis in ("y", "both"):
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    if axis in ("x", "both"):
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which="major", alpha=0.24, linestyle=":", linewidth=0.45)
    ax.grid(which="minor", alpha=minor_alpha, linestyle=":", linewidth=0.35)

# ---------- consistent palette ----------
# One color = one concept across every figure
COL = {
    "PT":      "#777777",          # neutral grey, pretrained reference
    "IT":      "#0072B2",          # Okabe-Ito blue, instruction-tuned primary
    "Gemma2":  "#D55E00",          # Okabe-Ito vermillion
    "Gemma3":  "#0072B2",          # Okabe-Ito blue
    "Qwen":    "#009E73",          # Okabe-Ito bluish green
    "Llama":   "#CC79A7",          # Okabe-Ito reddish purple
    "frame":   "#777777",          # neutral grey for framing/reference geometry
    "judge":   "#0072B2",          # Okabe-Ito blue for d_judge
    "C_to_S":  "#0072B2",
    "S_to_C":  "#777777",
    "success": "#0072B2",
    "fail":    "#9b9b9b",
}


# =====================================================================
# FIGURE 1 - schematic (PT vs IT routing geometry)
# =====================================================================
def figure1():
    from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch

    fig, axes = plt.subplots(1, 2, figsize=(3.25, 1.72))
    panel_specs = [
        ("PT: unused", "Gemma-3-4B-PT", 0.05, 87, "+0.04", "0.95", False),
        ("IT: routed", "Gemma-3-4B-IT", 0.31, 72, "+12.18", "0.98", True),
    ]

    for ax, (label, model, cos_val, angle_deg, sc, probe, has_routing) in zip(axes, panel_specs):
        ax.set_xlim(-0.18, 1.18)
        ax.set_ylim(-0.56, 1.17)
        ax.set_aspect("equal")
        ax.axis("off")

        ax.add_patch(FancyBboxPatch(
            (-0.15, -0.57), 1.30, 1.70,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor="white", edgecolor="#777777", linewidth=0.55,
            clip_on=False, zorder=0,
        ))

        origin = (0.03, -0.01)
        rad = np.deg2rad(angle_deg)
        f_tip = (origin[0] + 0.82 * np.cos(rad), origin[1] + 0.82 * np.sin(rad))
        j_tip = (origin[0] + 0.92, origin[1])

        ax.plot(*origin, "o", color="#222", ms=2.0, zorder=4)
        ax.add_patch(FancyArrowPatch(origin, j_tip,
                                     arrowstyle="-|>",
                                     mutation_scale=8, lw=1.4,
                                     color=COL["judge"], zorder=3))
        ax.text(j_tip[0] - 0.05, -0.13, r"$d_{\mathrm{judge}}$",
                color=COL["judge"], fontsize=6.2, fontweight="bold", ha="center")

        ax.add_patch(FancyArrowPatch(origin, f_tip,
                                     arrowstyle="-|>",
                                     mutation_scale=8, lw=1.4,
                                     color=COL["frame"], zorder=3))
        ax.text(f_tip[0], f_tip[1] + 0.08, r"$d_{\mathrm{frame}}$",
                color=COL["frame"], fontsize=6.2, fontweight="bold", ha="center")

        ax.add_patch(Arc(origin, 0.34, 0.34, theta1=0, theta2=angle_deg,
                         color="#666", lw=0.55, zorder=2))
        mid = np.deg2rad(angle_deg / 2)
        ax.text(origin[0] + 0.29 * np.cos(mid), origin[1] + 0.29 * np.sin(mid),
                f"{angle_deg}°", fontsize=5.6, color="#333",
                ha="center", va="center")

        proj_len = origin[0] + 0.82 * np.cos(rad)
        if has_routing:
            ax.add_patch(FancyArrowPatch(origin, (proj_len, origin[1]),
                                         arrowstyle="-|>",
                                         mutation_scale=7, lw=1.8,
                                         color=COL["frame"], zorder=4,
                                         shrinkA=0, shrinkB=0))
            ax.plot([f_tip[0], proj_len], [f_tip[1], origin[1]],
                    ":", lw=0.6, color="#777", zorder=2)
        else:
            ax.text(0.31, -0.13, "near-orthogonal",
                    color="#555", fontsize=5.2, ha="center", style="italic")

        ax.text(0.50, 1.03, label, fontsize=6.8, fontweight="bold",
                ha="center", color="#222")
        ax.text(0.50, -0.27, model, ha="center", va="center", fontsize=5.8,
                style="italic", color="#555")
        ax.text(0.50, -0.39, rf"$\cos={cos_val:.2f}$   probe {probe}",
                ha="center", va="center", fontsize=5.7, color="#222")
        ax.text(0.50, -0.49, rf"$S{{-}}C={sc}$", ha="center", va="center",
                fontsize=5.7, color="#222")

    fig.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.03, wspace=0.18)
    fig.savefig(OUT / "figure1.pdf")
    fig.savefig(OUT / "figure1.png", dpi=200)
    plt.close(fig)
    print("figure1: schematic")


# =====================================================================
# FIGURE 2 - combined probe, alignment, and intervention summary
# =====================================================================
def figure2():
    pt_probe = [0.933,0.908,0.875,0.858,0.883,0.867,0.858,0.892,0.883,0.925,0.900,0.908,0.858,0.917,0.875,0.850,0.883,0.883,0.875,0.900,0.908,0.942,0.925,0.925,0.917,0.925,0.908,0.925,0.908,0.858,0.875,0.858,0.842,0.825]
    it_probe = [0.900,0.958,0.950,0.933,0.933,0.942,0.933,0.950,0.950,0.983,0.942,0.967,0.967,0.975,0.975,0.983,0.983,0.983,0.983,0.967,0.967,0.975,0.975,0.950,0.958,0.958,0.967,0.967,0.967,0.958,0.950,0.958,0.958,0.967]
    pt_align = [-0.002,-0.003,0.024,0.031,0.027,0.040,0.000,0.017,-0.016,0.002,0.013,-0.012,-0.020,-0.019,0.039,-0.021,-0.011,0.013,0.013,0.018,0.012,0.014,0.027,0.030,0.022,0.036,0.032,0.027,0.033,0.013,0.027,0.029,0.029,0.005]
    it_align = [0.000,0.001,-0.011,0.013,0.009,0.019,-0.012,-0.028,-0.002,-0.002,-0.022,-0.013,-0.056,-0.038,-0.024,0.003,0.044,0.020,0.003,0.024,0.017,0.019,0.079,0.137,0.118,0.248,0.226,0.245,0.220,0.197,0.217,0.188,0.153,0.044]
    layers = list(range(34))

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(6.85, 2.25))
    gs = GridSpec(
        1, 3, figure=fig,
        width_ratios=[1.14, 1.14, 1.18],
        wspace=0.34,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # TOP: probe accuracy
    ax1.axvspan(22, 32.5, color="#000000", alpha=0.035, lw=0, zorder=0)
    ax1.plot(layers, pt_probe, color=COL["PT"], lw=1.4, label="PT", zorder=3)
    ax1.plot(layers, it_probe, color=COL["IT"], lw=1.6, label="IT", zorder=4)
    # shaded band showing probe accuracy is high in BOTH
    ax1.fill_between(layers,
                     [min(p, q) for p, q in zip(pt_probe, it_probe)],
                     [max(p, q) for p, q in zip(pt_probe, it_probe)],
                     alpha=0.06, color="#888", zorder=1)
    ax1.axhline(1/3, color="#bbb", ls="--", lw=0.5)
    ax1.text(0.2, 0.36, "chance", fontsize=6, color="#888", va="bottom")

    ax1.set_ylim(0.30, 1.02)
    ax1.set_xlim(-0.5, 33.6)
    ax1.set_xticks([0, 10, 20, 30])
    ax1.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax1.set_xlabel(r"layer $\ell$")
    ax1.set_ylabel("probe accuracy")
    ax1.text(31.5, pt_probe[-1] - 0.035, "PT",  color=COL["PT"], fontsize=7.2, va="center")
    ax1.text(31.5, it_probe[-1] + 0.025, "IT",  color=COL["IT"], fontsize=7.2,
             va="center", fontweight="bold")
    ax1.set_title("A  decodability", fontsize=8, loc="left", fontweight="bold")
    add_minor_grid(ax1, axis="both")

    # BOTTOM: alignment cosine
    ax2.axvspan(22, 32.5, color="#000000", alpha=0.035, lw=0, zorder=0)
    ax2.axhline(0, color="#999", lw=0.5, zorder=2)
    ax2.plot(layers, pt_align, color=COL["PT"], lw=1.4, zorder=3)
    ax2.plot(layers, it_align, color=COL["IT"], lw=1.6, zorder=4)

    # mark peak
    pk = 25
    ax2.scatter([pk], [it_align[pk]], s=24, color=COL["IT"], zorder=6,
                edgecolors="white", linewidths=0.8)
    ax2.annotate(f"L={pk}\n{it_align[pk]:+.2f}",
                 xy=(pk, it_align[pk]), xytext=(14.2, 0.275),
                 fontsize=6.5, color=COL["IT"],
                 arrowprops=dict(arrowstyle="-", lw=0.4, color=COL["IT"]))

    ax2.set_ylim(-0.10, 0.34)
    ax2.set_xlim(-0.5, 33.6)
    ax2.set_xticks([0, 10, 20, 30])
    ax2.set_xlabel(r"layer $\ell$")
    ax2.set_ylabel(r"alignment cosine")
    ax2.text(31.5, pt_align[-1] - 0.020, "PT", color=COL["PT"], fontsize=7.2, va="center")
    ax2.text(31.5, it_align[-1] + 0.020, "IT", color=COL["IT"], fontsize=7.2,
             va="center", fontweight="bold")
    ax2.set_title("B  readout alignment", fontsize=8, loc="left", fontweight="bold")
    add_minor_grid(ax2, axis="both")

    # C: compact intervention summary from the projection and patching figure.
    labels = [
        "random\nsubspace",
        "PCA\n" + r"$\perp d_j$",
        "resid\npatch",
        "attn/MLP\npatch",
    ]
    vals = [0.105, 0.951, 0.916, 0.231]
    colors = [COL["fail"], COL["success"], COL["C_to_S"], COL["fail"]]
    hatches = ["//", "", "", "//"]
    x = np.arange(len(labels))
    bars = ax3.bar(x, vals, color=colors, edgecolor="#333", linewidth=0.25, width=0.62)
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)
    for xi, v in zip(x, vals, strict=True):
        ax3.text(xi, v + 0.035, f"{v:.2f}", ha="center", fontsize=6.2, color="#222")
    ax3.axhline(1.0, color="#bbb", ls="--", lw=0.5)
    ax3.text(-0.46, 1.02, "full", fontsize=6, color="#888", ha="left")
    ax3.set_ylim(0, 1.18)
    ax3.set_yticks([0, 0.5, 1.0])
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=6)
    ax3.set_ylabel("fraction")
    ax3.set_title("C  intervention summary", fontsize=8, loc="left", fontweight="bold")
    add_minor_grid(ax3, axis="y")

    fig.savefig(OUT / "figure2.pdf")
    fig.savefig(OUT / "figure2.png", dpi=200)
    plt.close(fig)
    print("figure2: probe + alignment + intervention summary")


# =====================================================================
# FIGURE 3 - cross-model behaviour + alignment (2 panels, FULL WIDTH)
# =====================================================================
def figure3():
    rows = [
        # (label, family, S-C, alignment_peak)
        ("Gemma-3-4B-PT (ref.)", "PT",     -0.01, 0.0512),
        ("Gemma-2-2B-IT",        "Gemma2", 10.31, 0.206),
        ("Gemma-2-9B-IT",        "Gemma2",  6.61, 0.218),
        ("Gemma-2-27B-IT",       "Gemma2",  4.47, 0.229),
        ("Gemma-3-4B-IT",        "Gemma3",  8.96, 0.248),
        ("Gemma-3-12B-IT",       "Gemma3", 11.56, 0.338),
        ("Qwen-2.5-3B-Inst",     "Qwen",    8.22, 0.350),
        ("Qwen-2.5-7B-Inst",     "Qwen",   13.35, 0.276),
        ("Qwen-2.5-14B-Inst",    "Qwen",   17.89, 0.290),
        ("Qwen-2.5-32B-Inst",    "Qwen",   19.13, 0.282),
        ("Llama-3.1-8B-Inst",    "Llama",   3.10, 0.249),
    ]
    labels = [r[0] for r in rows]
    fams   = [r[1] for r in rows]
    sc     = [r[2] for r in rows]
    al     = [r[3] for r in rows]
    colors = [COL[f] for f in fams]
    hatches = {"PT": "///", "Gemma2": "", "Gemma3": "\\\\", "Qwen": "..", "Llama": "xx"}
    y      = np.arange(len(rows))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.75, 2.9),
                                   gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05})

    # LEFT: paired S-C shift
    bars_l = axL.barh(y, sc, color=colors, edgecolor="#333", linewidth=0.25, height=0.65)
    for bar, fam in zip(bars_l, fams, strict=True):
        bar.set_hatch(hatches[fam])
    axL.axvline(0, color="#888", lw=0.4)
    axL.set_xlim(-2, 22)
    axL.set_xticks([0, 5, 10, 15, 20])
    axL.set_xlabel("paired $S{-}C$ shift (logits)")
    axL.set_yticks(y)
    axL.set_yticklabels(labels)
    axL.invert_yaxis()
    axL.set_title("behaviour", fontsize=8, style="italic", pad=2)
    for yi, v in zip(y, sc):
        axL.text(v + 0.4, yi, f"{v:+.2f}", fontsize=6, va="center", color="#222")
    add_minor_grid(axL, axis="x")

    # RIGHT: alignment cosine peak
    bars_r = axR.barh(y, al, color=colors, edgecolor="#333", linewidth=0.25, height=0.65)
    for bar, fam in zip(bars_r, fams, strict=True):
        bar.set_hatch(hatches[fam])
    axR.axvline(0.0512, color="#888", lw=0.5, ls="--")
    axR.set_xlim(0, 0.42)
    axR.set_xticks([0, 0.10, 0.20, 0.30, 0.40])
    axR.set_xlabel(r"peak $\cos(\bar{S}{-}\bar{C}, d_{\mathrm{judge}})$")
    axR.set_yticks(y)
    axR.set_yticklabels([])
    axR.invert_yaxis()
    axR.set_title("routing alignment", fontsize=8, style="italic", pad=2)
    for yi, v in zip(y, al):
        axR.text(v + 0.005, yi, f"{v:.2f}", fontsize=6, va="center", color="#222")
    add_minor_grid(axR, axis="x")

    # family legend (single)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COL["PT"], edgecolor="#333", hatch=hatches["PT"], label="Pretrained ref."),
        Patch(facecolor=COL["Gemma2"], edgecolor="#333", hatch=hatches["Gemma2"], label="Gemma-2"),
        Patch(facecolor=COL["Gemma3"], edgecolor="#333", hatch=hatches["Gemma3"], label="Gemma-3"),
        Patch(facecolor=COL["Qwen"], edgecolor="#333", hatch=hatches["Qwen"], label="Qwen-2.5"),
        Patch(facecolor=COL["Llama"], edgecolor="#333", hatch=hatches["Llama"], label="Llama-3.1"),
    ]
    fig.legend(handles=legend_handles, ncol=5, loc="upper center",
               bbox_to_anchor=(0.55, 1.04), frameon=False, fontsize=7)

    fig.savefig(OUT / "figure3.pdf")
    fig.savefig(OUT / "figure3.png", dpi=200)
    plt.close(fig)
    print("figure3: cross-model 2-panel")


# =====================================================================
# FIGURE 4 - consolidated intervention evidence
# =====================================================================
def figure4():
    import json
    from matplotlib.gridspec import GridSpec

    subspace_labels = [
        "rank-1\nprobe",
        "rank-4\nrandom\n" + r"$p_{95}$",
        "rank-4\nPCA\n" + r"$\perp d_j$",
        "rank-4\nPCA",
    ]
    subspace_vals = [0.000, 0.105, 0.951, 1.051]
    subspace_colors = [COL["fail"], COL["fail"], COL["success"], COL["success"]]
    subspace_hatches = ["//", "//", "", ""]

    components = ["resid_pre", "resid_post", "attn_out", "mlp_out"]
    c_to_s = [0.916, 0.934, 0.069, 0.231]
    s_to_c = [0.379, 0.337, 0.062, 0.050]

    resid_path = OUT.parent / "data" / "fig4b_resid.dat"
    resid_table = np.genfromtxt(resid_path, names=True)

    result_path = Path("runs/9728c2897b22__6411b6f2e4/exp08_component/result.json")
    data_path = OUT.parent / "data" / "fig4_component_heatmap.dat"
    mr = None
    layers = None
    comps = ["resid_pre", "resid_post", "attn_out", "mlp_out"]
    if result_path.exists():
        d = json.load(open(result_path))
        mr = np.array(d["metrics"]["mean_recovery"])
        comps = d["config"]["components"]
        layers = d["config"]["layers"]
    elif data_path.exists():
        table = np.genfromtxt(data_path, names=True)
        layers = sorted(set(table["layer"].astype(int)))
        comp_ids = sorted(set(table["comp_idx"].astype(int)))
        mr = np.full((len(comp_ids), len(layers)), np.nan)
        layer_to_col = {layer: i for i, layer in enumerate(layers)}
        for row in table:
            r = int(row["comp_idx"])
            c = layer_to_col[int(row["layer"])]
            mr[r, c] = row["rec"]

    fig = plt.figure(figsize=(6.85, 4.35))
    gs = GridSpec(
        2, 2, figure=fig,
        height_ratios=[1.0, 1.08],
        width_ratios=[1.0, 1.0],
        hspace=0.62,
        wspace=0.34,
    )
    ax_sub = fig.add_subplot(gs[0, 0])
    ax_comp = fig.add_subplot(gs[0, 1])
    ax_line = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])

    # A: subspace removal specificity.
    x = np.arange(len(subspace_labels))
    bars = ax_sub.bar(
        x, subspace_vals, color=subspace_colors, edgecolor="#333",
        linewidth=0.25, width=0.62,
    )
    for bar, hatch in zip(bars, subspace_hatches, strict=True):
        bar.set_hatch(hatch)
    for xi, v in zip(x, subspace_vals):
        ax_sub.text(xi, v + 0.035, f"{v:.2f}", ha="center", fontsize=6, color="#222")
    ax_sub.axhline(1.0, color="#bbb", ls="--", lw=0.5)
    ax_sub.text(-0.45, 1.02, "full ablation", fontsize=6, color="#888", ha="left")
    ax_sub.axhline(0, color="#444", lw=0.4)
    ax_sub.set_xticks(x)
    ax_sub.set_xticklabels(subspace_labels, fontsize=5.4)
    ax_sub.set_ylim(-0.10, 1.23)
    ax_sub.set_yticks([0, 0.5, 1.0])
    ax_sub.set_ylabel(r"shrink fraction")
    ax_sub.set_title("A  subspace removal", fontsize=8, loc="left", fontweight="bold")
    add_minor_grid(ax_sub, axis="y")

    # B: component patching asymmetry at L=20.
    x = np.arange(len(components))
    w = 0.36
    ax_comp.bar(x - w/2, c_to_s, w, color=COL["C_to_S"], edgecolor="#333",
           linewidth=0.25, label=r"C$\to$S")
    ax_comp.bar(x + w/2, s_to_c, w, color=COL["S_to_C"], edgecolor="#333",
           linewidth=0.25, hatch="//", label=r"S$\to$C")
    for xi, v in zip(x - w/2, c_to_s):
        ax_comp.text(xi - 0.03, v + 0.040, f"{v:.2f}", ha="center", fontsize=6, color="#222")
    for i, (xi, v) in enumerate(zip(x + w/2, s_to_c)):
        dx = 0.05 if i < 2 else 0.02
        ax_comp.text(xi + dx, v + 0.040, f"{v:.2f}", ha="center", fontsize=6, color="#222")
    ax_comp.axhline(1.0, color="#bbb", ls="--", lw=0.5)
    ax_comp.text(3.40, 1.02, "full", fontsize=6, color="#888", ha="right")
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(["resid\npre", "resid\npost", "attn\nout", "mlp\nout"], fontsize=5.8)
    ax_comp.set_ylim(0, 1.24)
    ax_comp.set_yticks([0, 0.5, 1.0])
    ax_comp.set_ylabel(r"patch recovery")
    ax_comp.legend(loc="upper center", frameon=False, ncol=2,
                   handlelength=1.0, columnspacing=0.75, bbox_to_anchor=(0.56, 1.05))
    ax_comp.set_title(r"B  component patching, $L{=}20$", fontsize=8,
                      loc="left", fontweight="bold", y=1.16)
    add_minor_grid(ax_comp, axis="y")

    # C: residual-stream layer sweep across the full network.
    layer = resid_table["layer"]
    ax_line.axhline(0.0, color="#777", lw=0.45)
    ax_line.axhline(1.0, color="#bbb", ls="--", lw=0.5)
    ax_line.axvspan(17, 33, color="#000000", alpha=0.035, linewidth=0)
    ax_line.plot(layer, resid_table["median"], color=COL["C_to_S"], lw=1.45,
                 marker="o", ms=2.2)
    ax_line.set_xlim(0, 33)
    ax_line.set_ylim(-0.08, 1.05)
    ax_line.set_xticks([0, 5, 10, 15, 20, 25, 30, 33])
    ax_line.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_line.set_xlabel(r"layer $\ell$")
    ax_line.set_ylabel("median residual recovery")
    ax_line.set_title("C  residual-stream layer sweep", fontsize=8,
                      loc="left", fontweight="bold")
    add_minor_grid(ax_line, axis="both")

    # D: component x layer sweep for S->C patching.
    if mr is None or layers is None:
        ax_heat.axis("off")
        ax_heat.text(0.5, 0.5, "heatmap data missing", ha="center", va="center")
    else:
        norm = mpl.colors.TwoSlopeNorm(vmin=-0.8, vcenter=0.0, vmax=1.0)
        im = ax_heat.imshow(mr, aspect="auto", cmap="RdBu_r", norm=norm,
                            interpolation="nearest")
        late_start = layers.index(28) if 28 in layers else len(layers) // 2
        tick_idx = [0, 3, 6, 9, 12, 15]
        tick_idx = [i for i in tick_idx if i < len(layers)]
        ax_heat.set_xticks(tick_idx)
        ax_heat.set_xticklabels([layers[i] for i in tick_idx], fontsize=6)
        ax_heat.set_yticks(range(len(comps)))
        ax_heat.set_yticklabels(["resid pre", "resid post", "attn out", "mlp out"], fontsize=6)
        ax_heat.set_xlabel(r"layer $\ell$")
        ax_heat.tick_params(axis="both", which="both", length=0)
        ax_heat.set_xticks(np.arange(-0.5, len(layers), 1), minor=True)
        ax_heat.set_yticks(np.arange(-0.5, len(comps), 1), minor=True)
        ax_heat.grid(which="minor", color="white", linewidth=0.65)
        ax_heat.grid(which="major", visible=False)
        ax_heat.axvspan(late_start - 0.5, len(layers) - 0.5,
                        color="#000000", alpha=0.035, linewidth=0)
        ax_heat.axvline(late_start - 0.5, color="#333333",
                        linewidth=0.45, alpha=0.32)
        ax_heat.set_title(r"D  component layer sweep, S$\to$C", fontsize=8,
                          loc="left", fontweight="bold")
        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.042, pad=0.025,
                            ticks=[-0.8, 0, 0.5, 1.0])
        cbar.ax.tick_params(labelsize=6, length=0)
        cbar.outline.set_linewidth(0.4)
        cbar.set_label("recovery", fontsize=7)

    fig.savefig(OUT / "figure4.pdf")
    fig.savefig(OUT / "figure4.png", dpi=200)
    plt.close(fig)
    print("figure4: consolidated interventions")


# =====================================================================
# FIGURE 5 - subspace ablation specificity (L=25, AoT-40)
# =====================================================================
def figure5():
    labels = [
        "rank-1\nprobe dir.",
        "rank-4\nrandom ($p_{95}$)",
        r"rank-4 PCA""\n"r"$\perp\,d_{\mathrm{judge}}$",
        "rank-4\nPCA",
    ]
    vals    = [0.000, 0.105, 0.951, 1.051]
    colors  = [COL["fail"], COL["fail"], COL["success"], COL["success"]]
    hatches = ["//", "//", "", ""]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(3.25, 2.15))
    bars = ax.bar(x, vals, color=colors, edgecolor="#333", linewidth=0.25, width=0.6)
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)

    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=7, color="#222")

    ax.axhline(1.0, color="#bbb", ls="--", lw=0.5)
    ax.text(-0.4, 1.02, "full ablation", fontsize=6, color="#888", ha="left")
    ax.axhline(0, color="#444", lw=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(-0.10, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(r"shrink fraction of $S{-}C$ gap")
    add_minor_grid(ax, axis="both")

    fig.savefig(OUT / "figure5.pdf")
    fig.savefig(OUT / "figure5.png", dpi=200)
    plt.close(fig)
    print("figure5: subspace ablation")


# =====================================================================
# FIGURE 7 - component x layer recovery heatmap (Gemma-3-4B-IT, S->C)
# =====================================================================
def figure7():
    import json
    result_path = Path("runs/9728c2897b22__6411b6f2e4/exp08_component/result.json")
    data_path = OUT.parent / "data" / "fig4_component_heatmap.dat"
    if result_path.exists():
        d = json.load(open(result_path))
        mr = np.array(d["metrics"]["mean_recovery"])  # [4, 16]
        comps = d["config"]["components"]
        layers = d["config"]["layers"]
    elif data_path.exists():
        table = np.genfromtxt(data_path, names=True)
        layers = sorted(set(table["layer"].astype(int)))
        comp_ids = sorted(set(table["comp_idx"].astype(int)))
        comps = ["resid_pre", "resid_post", "attn_out", "mlp_out"]
        mr = np.full((len(comp_ids), len(layers)), np.nan)
        layer_to_col = {layer: i for i, layer in enumerate(layers)}
        for row in table:
            r = int(row["comp_idx"])
            c = layer_to_col[int(row["layer"])]
            mr[r, c] = row["rec"]
    else:
        print("figure7: keeping existing output; missing", result_path, "and", data_path)
        return

    fig, ax = plt.subplots(figsize=(3.25, 1.75))
    norm = mpl.colors.TwoSlopeNorm(vmin=-0.8, vcenter=0.0, vmax=1.0)
    im = ax.imshow(mr, aspect="auto", cmap="RdBu_r", norm=norm,
                   interpolation="nearest")

    late_start = layers.index(28)
    tick_idx = [0, 3, 6, 9, 12, 15]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([layers[i] for i in tick_idx], fontsize=6)
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels(["resid pre", "resid post", "attn out", "mlp out"])
    ax.set_xlabel(r"layer $\ell$")
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(layers), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(comps), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.65)
    ax.grid(which="major", visible=False)
    ax.axvspan(late_start - 0.5, len(layers) - 0.5,
               color="#000000", alpha=0.035, linewidth=0)
    ax.axvline(late_start - 0.5, color="#333333", linewidth=0.45, alpha=0.32)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.035,
                        ticks=[-0.8, 0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=6, length=0)
    cbar.outline.set_linewidth(0.4)
    cbar.set_label("recovery", fontsize=7)

    fig.savefig(OUT / "figure7.pdf")
    fig.savefig(OUT / "figure7.png", dpi=200)
    plt.close(fig)
    print("figure7: component x layer heatmap")


def main():
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure7()
    print(f"\nWrote 6 figures to {OUT}")


if __name__ == "__main__":
    main()

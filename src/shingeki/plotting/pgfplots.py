"""pgfplots emitter.

Each `emit_figN(...)` writes:
  - one or more `.dat` tables under `manuscript/data/`
  - a TikZ/pgfplots `.tex` snippet under `manuscript/figures/` that loads them

The manuscript's `main.tex` includes pgfplots and `\\input`s these snippets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from shingeki.config import MANUSCRIPT_DIR

DATA_DIR = MANUSCRIPT_DIR / "data"
FIG_DIR = MANUSCRIPT_DIR / "figures"


# ----------------------------- helpers -----------------------------


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _write_table(path: Path, header: list[str], rows: Iterable[Iterable]) -> None:
    """Whitespace-delimited pgfplots table.

    pgfplots `header=has colnames` treats the first non-comment line as the
    column-name row, so we write `col1 col2 col3` *without* a leading `#`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(" ".join(header) + "\n")
        for row in rows:
            f.write(" ".join(_fmt(v) for v in row) + "\n")


def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "nan"
        return f"{float(v):.6g}"
    return str(v)


# ----------------------------- Fig 1 -----------------------------


def emit_fig1(behavior_metrics: dict, sym: np.ndarray, cond: np.ndarray) -> Path:
    """Paired scatter + histogram of S-C shifts."""
    _ensure_dirs()
    scatter_path = DATA_DIR / "fig1_scatter.dat"
    hist_path = DATA_DIR / "fig1_shift_hist.dat"
    _write_table(
        scatter_path,
        ["cond", "sym"],
        [(c, s) for c, s in zip(cond, sym, strict=True)],
    )

    diffs = np.array(sym) - np.array(cond)
    hist, edges = np.histogram(diffs, bins=18)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = float(edges[1] - edges[0])
    _write_table(hist_path, ["center", "count"], list(zip(centers, hist, strict=True)))

    sub = behavior_metrics["S_vs_C"]
    n = sub["n_triplets"]
    title = f"$N{{=}}{n}$, mean $S{{-}}C={sub['mean_shift']:+.2f}$, $p={sub['perm_p']:.0e}$"

    tex = rf"""% Auto-generated; do not edit
\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=2 by 1, horizontal sep=1.4cm}},
  width=4.4cm, height=4.0cm, scale only axis,
  axis x line=bottom, axis y line=left,
  enlargelimits=0.05,
  tick label style={{font=\footnotesize}},
  label style={{font=\footnotesize}},
]
\nextgroupplot[
  xlabel={{Condemning prompt: $\log p(\text{{Yes}}) - \log p(\text{{No}})$}},
  ylabel={{Sympathetic prompt}},
  title style={{font=\footnotesize}},
  title={{Paired Yes/No shift}},
]
\addplot[only marks, mark=*, mark size=1.2pt, color=red!60!black, fill opacity=0.5]
  table[x=cond, y=sym, header=has colnames] {{data/fig1_scatter.dat}};
\addplot[gray, dashed, domain=-30:35] {{x}};

\nextgroupplot[
  xlabel={{$S - C$ shift (logit gap)}},
  ylabel={{\# triplets}},
  ybar, bar width={width:.3f}*1pt,
  title style={{font=\footnotesize}},
  title={{{title}}},
]
\addplot[fill=red!50!black, draw=none]
  table[x=center, y=count, header=has colnames] {{data/fig1_shift_hist.dat}};
\addplot[mark=none, gray, sharp plot, line width=0.4pt] coordinates {{(0,0) (0,10)}};
\end{{groupplot}}
\end{{tikzpicture}}
"""
    out = FIG_DIR / "fig1_dataset_behavior.tex"
    out.write_text(tex)
    return out


# ----------------------------- Fig 2 -----------------------------


def emit_fig2(
    cosine_metrics: dict,
    probe_acc_per_layer_per_dataset: Mapping[str, np.ndarray],
    n_layers: int,
) -> Path:
    """Two-panel: cosine misses, probes find."""
    _ensure_dirs()
    cos_path = DATA_DIR / "fig2_cosine.dat"
    final_pool = cosine_metrics["final_token"]["S_vs_C_mean_per_layer"]
    mean_pool = cosine_metrics["mean_pooled"]["S_vs_C_mean_per_layer"]
    rows = [(l, f, m) for l, (f, m) in enumerate(zip(final_pool, mean_pool, strict=True))]
    _write_table(cos_path, ["layer", "final", "meanpool"], rows)

    probe_path = DATA_DIR / "fig2_probe_accuracy.dat"
    name_to_col = {n: _col_safe(n) for n in probe_acc_per_layer_per_dataset.keys()}
    cols = ["layer"] + list(name_to_col.values())
    arrs = list(probe_acc_per_layer_per_dataset.values())
    rows = []
    for l in range(n_layers):
        rows.append([l] + [(arr[l] if l < len(arr) else float("nan")) for arr in arrs])
    _write_table(probe_path, cols, rows)

    n_classes_chance = 1.0 / 3.0
    probe_lines = "\n".join(
        rf"\addplot[smooth, line width=0.9pt] table[x=layer, y={name_to_col[name]}, header=has colnames] {{data/fig2_probe_accuracy.dat}}; \addlegendentry{{{_legend_safe(name)}}}"
        for name in probe_acc_per_layer_per_dataset.keys()
    )

    tex = rf"""% Auto-generated; do not edit
\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=2 by 1, horizontal sep=1.4cm}},
  width=4.4cm, height=4.0cm, scale only axis,
  axis x line=bottom, axis y line=left,
  enlargelimits=0.02,
  tick label style={{font=\footnotesize}},
  label style={{font=\footnotesize}},
  legend style={{font=\footnotesize, draw=none, fill opacity=0.7}},
]
\nextgroupplot[
  xlabel=Layer, ylabel={{cos$(\bar{{S}}, \bar{{C}})$}},
  ymin=0.0, ymax=1.05,
  title style={{font=\footnotesize}},
  title={{Coarse cosine}},
  legend pos=south west,
]
\addplot[red!70!black, line width=0.9pt]
  table[x=layer, y=final, header=has colnames] {{data/fig2_cosine.dat}};
\addlegendentry{{final-token}}
\addplot[blue!60!black, dashed, line width=0.9pt]
  table[x=layer, y=meanpool, header=has colnames] {{data/fig2_cosine.dat}};
\addlegendentry{{mean-pool}}

\nextgroupplot[
  xlabel=Layer, ylabel={{LOTO probe acc.}},
  ymin=0.0, ymax=1.05,
  title style={{font=\footnotesize}},
  title={{Probes decode framing}},
  legend pos=south east,
]
{probe_lines}
\addplot[gray, dotted, line width=0.5pt, domain=0:{n_layers - 1}] {{{n_classes_chance:.4f}}};
\addlegendentry{{chance}}
\end{{groupplot}}
\end{{tikzpicture}}
"""
    out = FIG_DIR / "fig2_cosine_vs_probe.tex"
    out.write_text(tex)
    return out


def _legend_safe(name: str) -> str:
    return name.replace("_", r"\_")


def _col_safe(name: str) -> str:
    """Pgfplots column names must be alphanumeric; sanitise for header row."""
    out = []
    for ch in name:
        if ch.isalnum():
            out.append(ch)
        elif ch in ("_",):
            out.append("_")
        else:
            out.append("_")
    s = "".join(out).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    if not s or not s[0].isalpha():
        s = "c_" + s
    return s


# ----------------------------- Fig 3 -----------------------------


def emit_fig3(lens_metrics: dict, alignment_per_layer: np.ndarray) -> Path:
    """Three-panel: logit-lens trajectories per framing; S-C gap; cos(d_frame, d_judge)."""
    _ensure_dirs()
    traj_path = DATA_DIR / "fig3_lens_trajectories.dat"
    sym = lens_metrics["sympathetic"]["mean_per_layer"]
    cond = lens_metrics["condemning"]["mean_per_layer"]
    abst = lens_metrics["abstract"]["mean_per_layer"]
    rows = [(l, s, c, a) for l, (s, c, a) in enumerate(zip(sym, cond, abst, strict=True))]
    _write_table(traj_path, ["layer", "sym", "cond", "abst"], rows)

    sc_path = DATA_DIR / "fig3_sc_gap.dat"
    sc = lens_metrics["sym_minus_cond_per_layer"]
    _write_table(sc_path, ["layer", "sc_gap"], [(l, v) for l, v in enumerate(sc)])

    align_path = DATA_DIR / "fig3_alignment.dat"
    _write_table(
        align_path,
        ["layer", "cos_align"],
        [(l, v) for l, v in enumerate(alignment_per_layer)],
    )

    tex = rf"""% Auto-generated; do not edit
\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=3 by 1, horizontal sep=1.1cm}},
  width=3.0cm, height=3.4cm, scale only axis,
  axis x line=bottom, axis y line=left,
  tick label style={{font=\footnotesize}},
  label style={{font=\footnotesize}},
  title style={{font=\footnotesize}},
  legend style={{font=\scriptsize, draw=none, fill opacity=0.7}},
]
\nextgroupplot[xlabel=Layer, ylabel={{$\log p(\text{{Yes}}) - \log p(\text{{No}})$}}, title=Logit-lens]
\addplot[teal!60!black, line width=0.9pt]   table[x=layer, y=sym,  header=has colnames] {{data/fig3_lens_trajectories.dat}};
\addlegendentry{{sym.}}
\addplot[orange!70!black, line width=0.9pt] table[x=layer, y=cond, header=has colnames] {{data/fig3_lens_trajectories.dat}};
\addlegendentry{{cond.}}
\addplot[gray!50!black, line width=0.9pt]   table[x=layer, y=abst, header=has colnames] {{data/fig3_lens_trajectories.dat}};
\addlegendentry{{abst.}}

\nextgroupplot[xlabel=Layer, ylabel={{$S - C$ gap}}, title={{S$-$C readout transition}}]
\addplot[red!60!black, line width=1pt] table[x=layer, y=sc_gap, header=has colnames] {{data/fig3_sc_gap.dat}};

\nextgroupplot[xlabel=Layer, ylabel={{cos$(d_{{\text{{frame}}}}, d_{{\text{{judge}}}})$}}, title=Geometry alignment]
\addplot[olive!70!black, line width=1pt] table[x=layer, y=cos_align, header=has colnames] {{data/fig3_alignment.dat}};
\end{{groupplot}}
\end{{tikzpicture}}
"""
    out = FIG_DIR / "fig3_late_readout.tex"
    out.write_text(tex)
    return out


# ----------------------------- Fig 4 -----------------------------


def emit_fig4_component(component_metrics: dict, components: list[str], layers: list[int]) -> Path:
    """Component x layer recovery heatmap (matrix plot)."""
    _ensure_dirs()
    rec = np.array(component_metrics["mean_recovery"])  # [C, L]
    table_path = DATA_DIR / "fig4_component_heatmap.dat"
    rows = []
    for ci, c in enumerate(components):
        for li, l in enumerate(layers):
            rows.append((l, ci, float(rec[ci, li])))
    _write_table(table_path, ["layer", "comp_idx", "rec"], rows)

    yticks = ", ".join(f"{i}" for i in range(len(components)))
    yticklabels = ", ".join(_legend_safe(c) for c in components)
    tex = rf"""% Auto-generated; do not edit
\begin{{tikzpicture}}
\begin{{axis}}[
  width=7.0cm, height=4.0cm, scale only axis,
  view={{0}}{{90}},
  enlargelimits=false,
  axis on top,
  xlabel=Layer, ylabel={{Component}},
  ytick={{{yticks}}}, yticklabels={{{yticklabels}}},
  tick label style={{font=\footnotesize}},
  label style={{font=\footnotesize}},
  title style={{font=\footnotesize}},
  title={{Component $\times$ layer recovery (S\,$\to$\,C)}},
  point meta min=-1, point meta max=1,
  colormap name=RdBuDiv, colorbar,
  colorbar style={{
    width=0.25cm,
    tick label style={{font=\footnotesize}},
    title=rec.,
    title style={{font=\scriptsize}},
  }},
]
\addplot[matrix plot*, mesh/cols={len(layers)}, point meta=explicit]
  table[x=layer, y=comp_idx, meta=rec, header=has colnames] {{data/fig4_component_heatmap.dat}};
\end{{axis}}
\end{{tikzpicture}}
"""
    out = FIG_DIR / "fig4_causal.tex"
    out.write_text(tex)
    return out


def emit_fig4b_resid(resid_metrics: dict, layers: list[int]) -> Path:
    """Residual-stream patching mean + median per layer."""
    _ensure_dirs()
    median = resid_metrics["median_recovery_per_layer"]
    mean = resid_metrics["mean_recovery_per_layer"]
    rows = [(l, m, mn) for l, m, mn in zip(layers, median, mean, strict=True)]
    table_path = DATA_DIR / "fig4b_resid.dat"
    _write_table(table_path, ["layer", "median", "mean"], rows)

    tex = rf"""% Auto-generated; do not edit
\begin{{tikzpicture}}
\begin{{axis}}[
  width=5.0cm, height=3.6cm, scale only axis,
  axis x line=bottom, axis y line=left,
  xlabel=Layer, ylabel={{Recovery (S\,$\to$\,C)}},
  tick label style={{font=\footnotesize}},
  label style={{font=\footnotesize}},
  title style={{font=\footnotesize}},
  title={{Residual-stream patching (final token)}},
  legend style={{font=\footnotesize, draw=none, at={{(0.02,0.98)}}, anchor=north west}},
]
\addplot[red!60!black, line width=1pt] table[x=layer, y=median, header=has colnames] {{data/fig4b_resid.dat}};
\addlegendentry{{median}}
\addplot[blue!50!black, dashed, line width=0.7pt] table[x=layer, y=mean, header=has colnames] {{data/fig4b_resid.dat}};
\addlegendentry{{mean}}
\addplot[gray, dotted, line width=0.4pt, domain={layers[0]}:{layers[-1]}] {{1.0}};
\end{{axis}}
\end{{tikzpicture}}
"""
    out = FIG_DIR / "fig4b_resid_recovery.tex"
    out.write_text(tex)
    return out


# ----------------------------- Fig 5 (steering controls) -----------------------------


def emit_fig5_steering(steering_metrics: dict) -> Path:
    """Bar chart of `shrink_frac` across intervention × direction at the
    chosen patching peak layer. Highlights the random-control separations."""
    _ensure_dirs()
    rows: list[tuple[str, float]] = []
    for label, sub in steering_metrics.items():
        if isinstance(sub, dict) and "shrink_frac" in sub:
            rows.append((label, float(sub["shrink_frac"])))
    rows.sort(key=lambda p: p[1])
    table_path = DATA_DIR / "fig5_steering.dat"
    with table_path.open("w") as f:
        f.write("# label shrink\n")
        for lab, val in rows:
            safe = lab.replace(" ", "_")
            f.write(f"{safe} {val:.6g}\n")

    coords = "\n  ".join(f"({val:.4f}, {idx})" for idx, (_lab, val) in enumerate(rows))
    yticks = ", ".join(str(i) for i in range(len(rows)))
    yticklabels = ", ".join(_legend_safe(lab) for lab, _ in rows)

    tex = rf"""% Auto-generated; do not edit
\begin{{tikzpicture}}
\begin{{axis}}[
  width=5.6cm, height=6.5cm, scale only axis,
  xbar, bar width=4pt,
  xlabel={{$1 - (S{{-}}C)_{{\text{{ablated}}}}/(S{{-}}C)_{{\text{{baseline}}}}$ \,(``shrink'')}},
  ytick={{{yticks}}}, yticklabels={{{yticklabels}}},
  tick label style={{font=\scriptsize}},
  label style={{font=\footnotesize}},
  title style={{font=\footnotesize}},
  title={{Steering / ablation interventions (L=28)}},
  enlarge y limits=0.02,
]
\addplot[fill=red!50!black, draw=none] coordinates {{
  {coords}
}};
\end{{axis}}
\end{{tikzpicture}}
"""
    out = FIG_DIR / "fig5_steering.tex"
    out.write_text(tex)
    return out


# ----------------------------- Cross-model panel -----------------------------


def emit_fig6_cross_model(rows: list[dict]) -> Path:
    """One row per (model, dataset). Columns: behavior shift, behaviour CI,
    probe best layer / total layers, lens transition (last layer S-C gap)."""
    _ensure_dirs()
    table_path = DATA_DIR / "fig6_cross_model.dat"
    header = ["idx", "S_C", "ci_lo", "ci_hi", "lens_last", "probe_best_layer_norm"]
    out_rows = []
    for i, r in enumerate(rows):
        out_rows.append(
            [
                i,
                r["S_C"],
                r["ci_lo"],
                r["ci_hi"],
                r["lens_last"],
                r["probe_best_layer_norm"],
            ]
        )
    _write_table(table_path, header, out_rows)

    yticks = ", ".join(str(i) for i in range(len(rows)))
    yticklabels = ", ".join(_legend_safe(r["label"]) for r in rows)
    tex = rf"""% Auto-generated; do not edit
\begin{{tikzpicture}}
\begin{{axis}}[
  width=5.4cm, height=4.0cm, scale only axis,
  xbar, bar width=5pt,
  xlabel={{$S - C$ logit shift}},
  ytick={{{yticks}}}, yticklabels={{{yticklabels}}},
  tick label style={{font=\scriptsize}},
  label style={{font=\footnotesize}},
  title style={{font=\footnotesize}},
  title=Cross-model behavioural shift,
  enlarge y limits=0.08,
  /pgf/bar shift=0pt,
  /pgfplots/error bars/x dir=both,
  /pgfplots/error bars/x explicit,
  nodes near coords,
  nodes near coords style={{font=\tiny, color=black, /pgf/number format/fixed, /pgf/number format/precision=2}},
  every node near coord/.style={{xshift=2pt}},
]
\addplot+[fill=red!50!black, draw=none, mark=none]
  plot[error bars/.cd, x dir=both, x explicit]
  table[x=S_C, y=idx,
        x error plus expr=\thisrow{{ci_hi}}-\thisrow{{S_C}},
        x error minus expr=\thisrow{{S_C}}-\thisrow{{ci_lo}},
        header=has colnames] {{data/fig6_cross_model.dat}};
\end{{axis}}
\end{{tikzpicture}}
"""
    out = FIG_DIR / "fig6_cross_model.tex"
    out.write_text(tex)
    return out


# ----------------------------- Table 1 -----------------------------


def emit_table1(rows: list[dict]) -> Path:
    """LaTeX booktabs table summarising per-(model, dataset) row."""
    _ensure_dirs()
    out = FIG_DIR / "table1_robustness.tex"
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"row & $S-C$ shift & 95\% CI & $p$ & probe acc.\ (peak L) & lens last $S-C$ \\",
        r"\midrule",
    ]
    for r in rows:
        ci = f"[{r['ci_lo']:+.2f},\\,{r['ci_hi']:+.2f}]"
        probe = f"{r['probe_acc']:.2f} (L{r['probe_best_layer']})" if r.get("probe_acc") is not None else "--"
        lens_last = f"{r['lens_last']:+.2f}" if r.get("lens_last") is not None else "--"
        p = f"{r['p']:.0e}" if r.get("p") is not None else "--"
        lines.append(
            rf"{r['label']} & {r['S_C']:+.2f} & {ci} & {p} & {probe} & {lens_last} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    return out

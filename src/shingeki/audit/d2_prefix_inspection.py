"""D0 vs D2 framing-prefix textual inspection (concern #1, tier 1).

For every triplet pair, compute character-level edit ratios and flag whether
the framing prose is identical (just names changed) vs structurally different.
Output: per-frame edit-ratio histograms.

This isn't a probe; it's a sanity check for the "D0->D2 = 100%" claim.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence

from shingeki.data.aot import load_d0, load_d2
from shingeki.data.schemas import FRAMES, FramingTriplet


def _ratio(a: str, b: str) -> float:
    """SequenceMatcher ratio in [0, 1]; 1.0 = identical."""
    return SequenceMatcher(None, a, b).ratio()


def per_triplet_edit_ratios(d0: Sequence[FramingTriplet], d2: Sequence[FramingTriplet]) -> list[dict]:
    by_id_d0 = {t.id: t for t in d0}
    by_id_d2 = {t.id: t for t in d2}
    out: list[dict] = []
    # try matching by index (D0 has aot-001 ... aot-040, D2 has anon-001 ...)
    n = min(len(d0), len(d2))
    for i in range(n):
        a, b = d0[i], d2[i]
        row: dict = {"d0_id": a.id, "d2_id": b.id}
        for fr in FRAMES:
            row[f"ratio_{fr}"] = _ratio(getattr(a, fr), getattr(b, fr))
            row[f"len_d0_{fr}"] = len(getattr(a, fr))
            row[f"len_d2_{fr}"] = len(getattr(b, fr))
        out.append(row)
    return out


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"error": "no rows"}
    summary: dict = {"n_triplets": len(rows), "per_frame": {}}
    for fr in FRAMES:
        ratios = [r[f"ratio_{fr}"] for r in rows]
        summary["per_frame"][fr] = {
            "mean_ratio": float(sum(ratios) / len(ratios)),
            "min_ratio": float(min(ratios)),
            "max_ratio": float(max(ratios)),
            "n_above_0.95": int(sum(1 for r in ratios if r >= 0.95)),
            "n_above_0.85": int(sum(1 for r in ratios if r >= 0.85)),
            "n_below_0.5": int(sum(1 for r in ratios if r < 0.5)),
        }
    return summary


def write_inspection(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    d0 = load_d0()
    d2 = load_d2()
    rows = per_triplet_edit_ratios(d0, d2)
    summary = summarise(rows)
    payload = {"summary": summary, "rows": rows}
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    csv_lines = ["d0_id,d2_id,ratio_sym,ratio_cond,ratio_abst,len_d0_sym,len_d2_sym"]
    for r in rows:
        csv_lines.append(
            f"{r['d0_id']},{r['d2_id']},"
            f"{r['ratio_sympathetic']:.3f},{r['ratio_condemning']:.3f},{r['ratio_abstract']:.3f},"
            f"{r['len_d0_sympathetic']},{r['len_d2_sympathetic']}"
        )
    (out_dir / "per_triplet.csv").write_text("\n".join(csv_lines) + "\n")
    return out_dir

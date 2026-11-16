"""Prose diagnostics on AoT vs GoT framing prompts.

Tests whether GoT > AoT framing effect could be a prose-style artifact rather
than genuine moral-content generalization. Five checks:

  1. Length distribution per (source, frame) - outliers, variance, range.
  2. Type-token ratio per (source, frame) - vocabulary diversity.
  3. Within-source frame-template overlap - do all GoT sympathetic prompts
     start with similar tokens? Same for condemning.
  4. Lexical "sentinel" tokens - counts of common opening tokens like
     "I had", "I was", "I had to" that might correlate with frame label.
  5. Cross-source vocabulary intersection - do AoT and GoT share enough
     vocabulary that the moral concept is genuinely cross-domain?

Output: prose_diagnostics.json + prose_diagnostics_summary.csv.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Sequence

from shingeki.data.aot import load_d0, load_d6_aot_extension, load_d7_got
from shingeki.data.schemas import FRAMES, FramingTriplet


def _tokens(text: str) -> list[str]:
    """Lower-cased, simple-space-split tokens, punctuation stripped."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return [t for t in text.split() if t]


def length_stats(prompts: Sequence[str]) -> dict:
    chars = [len(p) for p in prompts]
    toks = [len(_tokens(p)) for p in prompts]
    return {
        "n": len(prompts),
        "char_mean": statistics.mean(chars),
        "char_stdev": statistics.stdev(chars) if len(chars) > 1 else 0.0,
        "char_min": min(chars),
        "char_max": max(chars),
        "tok_mean": statistics.mean(toks),
        "tok_stdev": statistics.stdev(toks) if len(toks) > 1 else 0.0,
    }


def type_token_ratio(prompts: Sequence[str]) -> float:
    all_tokens: list[str] = []
    for p in prompts:
        all_tokens.extend(_tokens(p))
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)


def opening_overlap(prompts: Sequence[str], k: int = 5) -> dict:
    """Distribution of the first k tokens across prompts. Higher concentration
    means more templated openings."""
    openings = [tuple(_tokens(p)[:k]) for p in prompts]
    counter = Counter(openings)
    most_common = counter.most_common(5)
    total = sum(counter.values())
    return {
        "n_unique_openings": len(counter),
        "frac_most_common_opening": most_common[0][1] / total if most_common and total else 0.0,
        "top_openings": [{"tokens": list(o), "count": c} for o, c in most_common],
    }


SENTINEL_BIGRAMS = [
    "i had", "i was", "i did", "she was", "he was", "i am", "they had",
    "i could", "i have", "we had", "we were", "they killed",
]


def sentinel_counts(prompts: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in SENTINEL_BIGRAMS:
        c = 0
        for p in prompts:
            c += p.lower().count(s)
        out[s] = c
    return out


def vocab(prompts: Sequence[str]) -> set[str]:
    s: set[str] = set()
    for p in prompts:
        s.update(_tokens(p))
    return s


def diagnostics(triplets: Sequence[FramingTriplet], label: str) -> dict:
    out: dict = {"label": label, "n_triplets": len(triplets), "by_frame": {}}
    for fr in FRAMES:
        prompts = [getattr(t, fr) for t in triplets]
        out["by_frame"][fr] = {
            "length": length_stats(prompts),
            "ttr": type_token_ratio(prompts),
            "openings": opening_overlap(prompts, k=5),
            "sentinels": sentinel_counts(prompts),
        }
    return out


def main() -> None:
    out_dir = Path("runs/prose_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    d0 = load_d0()
    d6 = load_d6_aot_extension()
    d7 = load_d7_got()
    aot50 = d0 + d6  # combined AoT-50

    payload = {
        "aot_40": diagnostics(d0, "AoT-40 (D0, human-curated)"),
        "aot_extension_10": diagnostics(d6, "AoT-extension (D6, machine-curated)"),
        "aot_50_combined": diagnostics(aot50, "AoT-50 (D0 + D6)"),
        "got_50": diagnostics(d7, "GoT-50 (D7, machine-curated)"),
    }

    # Cross-source vocabulary overlap per frame
    for fr in FRAMES:
        aot_vocab = vocab([getattr(t, fr) for t in aot50])
        got_vocab = vocab([getattr(t, fr) for t in d7])
        intersect = aot_vocab & got_vocab
        union = aot_vocab | got_vocab
        payload[f"cross_vocab_{fr}"] = {
            "aot_vocab_size": len(aot_vocab),
            "got_vocab_size": len(got_vocab),
            "intersection_size": len(intersect),
            "jaccard": len(intersect) / max(len(union), 1),
        }

    (out_dir / "prose_diagnostics.json").write_text(json.dumps(payload, indent=2))

    # Tabular summary
    rows = ["dataset,frame,n,char_mean,char_stdev,tok_mean,ttr,frac_most_common_opening,top_opening"]
    for key, sub in payload.items():
        if not key.startswith("cross_") and "by_frame" in sub:
            for fr, fsub in sub["by_frame"].items():
                top = " ".join(fsub["openings"]["top_openings"][0]["tokens"])
                rows.append(
                    f"{sub['label']},{fr},{fsub['length']['n']},"
                    f"{fsub['length']['char_mean']:.0f},{fsub['length']['char_stdev']:.0f},"
                    f"{fsub['length']['tok_mean']:.1f},{fsub['ttr']:.3f},"
                    f"{fsub['openings']['frac_most_common_opening']:.3f},{top}"
                )
    (out_dir / "prose_diagnostics_summary.csv").write_text("\n".join(rows) + "\n")
    print(f"wrote {out_dir / 'prose_diagnostics.json'}")
    print(f"wrote {out_dir / 'prose_diagnostics_summary.csv'}")


if __name__ == "__main__":
    main()

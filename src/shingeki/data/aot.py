"""Loaders for the AoT dataset variants (D0, D1, D2) and extensions (D6, D7)."""

from __future__ import annotations

import json
from pathlib import Path

from shingeki.config import (
    D0_PATH,
    D1_PATH,
    D1_V2_PATH,
    D2_PATH,
    D6_AOT_EXTENSION_PATH,
    D7_GOT_PATH,
)
from shingeki.data.schemas import FramingTriplet


def _load_jsonl(path: Path) -> list[FramingTriplet]:
    if not path.exists():
        return []
    out: list[FramingTriplet] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(
                FramingTriplet(
                    id=d["id"],
                    dilemma=d["dilemma"],
                    source=d["source"],
                    decision_maker=d.get("decision_maker", "Unknown"),
                    ethical_frameworks=d.get(
                        "ethical_frameworks", d.get("decision_frameworks", [])
                    ),
                    sympathetic=d["sympathetic"],
                    condemning=d["condemning"],
                    abstract=d["abstract"],
                )
            )
    return out


def load_d0() -> list[FramingTriplet]:
    """D0: original 40 AoT triplets."""
    return _load_jsonl(D0_PATH)


def load_d1(include_v2: bool = True) -> list[FramingTriplet]:
    """D1: style-controlled subset (10 originally; v2 adds 10 more)."""
    items = _load_jsonl(D1_PATH)
    if include_v2:
        items += _load_jsonl(D1_V2_PATH)
    return items


def load_d2() -> list[FramingTriplet]:
    """D2: anonymized AoT (names/lore replaced)."""
    return _load_jsonl(D2_PATH)


def load_d6_aot_extension() -> list[FramingTriplet]:
    """D6: 10 additional AoT triplets covering canonical events not in D0."""
    return _load_jsonl(D6_AOT_EXTENSION_PATH)


def load_d7_got() -> list[FramingTriplet]:
    """D7: 50 Game of Thrones triplets (canonical book/show events)."""
    return _load_jsonl(D7_GOT_PATH)


def load_combined_n100() -> list[FramingTriplet]:
    """Combined n=100: D0 (40) + D6 (10 AoT extension) + D7 (50 GoT)."""
    return load_d0() + load_d6_aot_extension() + load_d7_got()

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FRAMES = ("sympathetic", "condemning", "abstract")

TRIPLET_FILES = {
    "data/d0_aot/aot_framing_triplets.jsonl": 40,
    "data/d1_style_controlled/aot_style_controlled.jsonl": 10,
    "data/d2_anonymized/aot_anonymized.jsonl": 40,
    "data/d6_aot_extension/aot_framing_extension.jsonl": 10,
    "data/d7_got/got_framing_triplets.jsonl": 50,
}

PRIMARY_TESTBED = {
    "data/d0_aot/aot_framing_triplets.jsonl": 40,
    "data/d6_aot_extension/aot_framing_extension.jsonl": 10,
    "data/d7_got/got_framing_triplets.jsonl": 50,
}

FORBIDDEN_ANON_TERMS = (
    "Eren",
    "Yeager",
    "Mikasa",
    "Reiner",
    "Marley",
    "Eldian",
    "Paradis",
    "Titan",
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path}:{line_no} invalid JSON: {exc}") from exc
    return rows


def validate_triplet(path: Path, row: dict) -> None:
    required = {"id", "dilemma", "source", *FRAMES}
    missing = sorted(required - set(row))
    assert not missing, f"{path}: {row.get('id', '<missing id>')} missing {missing}"
    if "ethical_frameworks" in row:
        assert isinstance(row["ethical_frameworks"], list), f"{path}: frameworks must be a list"
    for frame in FRAMES:
        text = row[frame]
        assert isinstance(text, str) and text.strip(), f"{path}: empty {frame} for {row['id']}"


def main() -> None:
    total_primary = 0
    total_triplets = 0
    for rel, expected in TRIPLET_FILES.items():
        path = ROOT / rel
        rows = read_jsonl(path)
        assert len(rows) == expected, f"{rel}: expected {expected}, got {len(rows)}"
        total_triplets += len(rows)
        for row in rows:
            validate_triplet(path, row)
        if rel in PRIMARY_TESTBED:
            total_primary += len(rows)

    assert total_primary == 100, f"primary testbed expected 100, got {total_primary}"

    residual_terms = 0
    anon_rows = read_jsonl(ROOT / "data/d2_anonymized/aot_anonymized.jsonl")
    for row in anon_rows:
        for frame in FRAMES:
            text = row[frame]
            for term in FORBIDDEN_ANON_TERMS:
                if term in text:
                    residual_terms += 1

    spans = read_jsonl(ROOT / "data/token_categories/aot_token_spans.jsonl")
    assert len(spans) == 120, f"token spans expected 120, got {len(spans)}"

    d3_rows = read_jsonl(ROOT / "data/d3_moralchoice/moralchoice_frame.jsonl")
    assert d3_rows, "D3 file is empty"

    result_root = ROOT / "paper_results"
    if not result_root.exists():
        result_root = ROOT / "runs"
    result_files = sorted(result_root.glob("**/*.json")) + sorted(result_root.glob("**/*.csv"))
    assert result_files, "paper_results contains no lightweight result summaries"

    print(f"ok: validated {total_triplets} triplets and {len(spans)} token-span annotations")
    print(f"ok: primary paper testbed has {total_primary} triplets")
    print(f"ok: found {len(result_files)} lightweight result summary files under {result_root.name}/")
    if residual_terms:
        print(f"warning: D2 contains {residual_terms} residual source-specific term matches")


if __name__ == "__main__":
    main()

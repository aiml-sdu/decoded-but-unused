"""LLM-assisted token-category annotation for AoT triplets.

Categories follow `data.schemas.Category`. Annotations are stored at
`data/token_categories/aot_token_spans.jsonl` with one row per (triplet_id,
frame). Token-position patching uses the resolved `token_spans`.

The annotator emits CHAR spans (not token spans). The token resolver runs
locally with the target-model's tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shingeki.config import ANNOTATOR_MODEL, TOKEN_SPANS_PATH
from shingeki.data.schemas import (
    CATEGORIES,
    FramingTriplet,
    TokenSpanAnnotation,
)
from shingeki.env import resolve_openrouter_key

PROMPT = """\
You will annotate a moral-dilemma narrative for downstream interpretability research.

Given the SCENARIO text below, return a JSON object whose keys are the following
categories and whose values are arrays of [start, end) character offsets into the
SCENARIO (NOT the surrounding text — the offsets must be valid for the SCENARIO
exactly as given). Multiple disjoint spans per category are allowed; no overlaps
between categories. Use empty arrays for categories with no occurrence.

Categories:
- actor: words referring to the decision-maker (names, pronouns, role-titles)
- victim: words referring to victims or affected parties
- harm: words describing harm, suffering, or destruction
- justification: words describing the actor's reasons, intentions, constraints, or moral
  reasoning
- group_id: ethnic / national / faction labels
- action: verbs and verb phrases describing the morally salient action itself
- final_question: not present in the scenario; leave empty
- final_answer: not present in the scenario; leave empty

Return ONLY the JSON object.

SCENARIO:
{scenario}
"""


def _call_openrouter(scenario: str, model: str = ANNOTATOR_MODEL) -> dict[str, list[list[int]]]:
    from openai import OpenAI

    key = resolve_openrouter_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set; cannot annotate")
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(scenario=scenario)}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or "{}"
    spans = json.loads(raw)
    return {k: [list(s) for s in v] for k, v in spans.items()}


def char_spans_to_token_spans(
    text: str, tokenizer, char_spans: dict[str, list[list[int]]]
) -> dict[str, list[list[int]]]:
    """Resolve [start, end) character spans to [start, end) token-index spans.

    A token at offset [a, b) is included iff it overlaps [cs, ce), i.e.
    `a < ce and b > cs`."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    out: dict[str, list[list[int]]] = {c: [] for c in char_spans}
    for cat, spans in char_spans.items():
        for cs, ce in spans:
            cs, ce = int(cs), int(ce)
            ts = next((i for i, (a, b) in enumerate(offsets) if b > cs), None)
            te = None
            for i in range(len(offsets) - 1, -1, -1):
                a, _b = offsets[i]
                if a < ce:
                    te = i
                    break
            if ts is None or te is None or te < ts:
                continue
            out[cat].append([int(ts), int(te + 1)])
    return out


def annotate_triplet(
    triplet: FramingTriplet,
    tokenizer,
    model: str = ANNOTATOR_MODEL,
) -> list[TokenSpanAnnotation]:
    annotations: list[TokenSpanAnnotation] = []
    for frame in ("sympathetic", "condemning", "abstract"):
        scenario = getattr(triplet, frame)
        char_spans = _call_openrouter(scenario, model=model)
        char_spans = {
            c: [list(s) for s in char_spans.get(c, [])] for c in CATEGORIES
        }
        token_spans = char_spans_to_token_spans(scenario, tokenizer, char_spans)
        annotations.append(
            TokenSpanAnnotation(
                triplet_id=triplet.id,
                frame=frame,  # type: ignore[arg-type]
                char_spans={c: [tuple(s) for s in char_spans[c]] for c in CATEGORIES},
                token_spans={c: [tuple(s) for s in token_spans[c]] for c in CATEGORIES},
            )
        )
    return annotations


def write_annotations(annotations: list[TokenSpanAnnotation], path: Path = TOKEN_SPANS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for a in annotations:
            f.write(
                json.dumps(
                    {
                        "triplet_id": a.triplet_id,
                        "frame": a.frame,
                        "char_spans": {k: [list(s) for s in v] for k, v in a.char_spans.items()},
                        "token_spans": {k: [list(s) for s in v] for k, v in a.token_spans.items()},
                    }
                )
                + "\n"
            )


def load_annotations(path: Path = TOKEN_SPANS_PATH) -> list[TokenSpanAnnotation]:
    if not path.exists():
        return []
    out: list[TokenSpanAnnotation] = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out.append(
                TokenSpanAnnotation(
                    triplet_id=d["triplet_id"],
                    frame=d["frame"],
                    char_spans={k: [tuple(s) for s in v] for k, v in d["char_spans"].items()},
                    token_spans={k: [tuple(s) for s in v] for k, v in d["token_spans"].items()},
                )
            )
    return out


def annotate_dataset(
    triplets: list[FramingTriplet],
    tokenizer,
    out_path: Path = TOKEN_SPANS_PATH,
    model: str = ANNOTATOR_MODEL,
    resume: bool = True,
) -> list[TokenSpanAnnotation]:
    """Annotate every (triplet, frame) and write to `out_path` (resumable)."""
    existing = {(a.triplet_id, a.frame): a for a in load_annotations(out_path)} if resume else {}
    all_anns: list[TokenSpanAnnotation] = list(existing.values())
    for triplet in triplets:
        if all(((triplet.id, fr) in existing) for fr in ("sympathetic", "condemning", "abstract")):
            continue
        all_anns.extend(annotate_triplet(triplet, tokenizer, model=model))
        write_annotations(all_anns, out_path)
    return all_anns


def _resolve_path(path: Any) -> Path:
    return path if isinstance(path, Path) else Path(path)

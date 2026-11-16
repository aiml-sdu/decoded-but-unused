"""Heuristic token-span annotator (regex / keyword based).

A fallback for when no OpenRouter API key is available. The categories follow
`schemas.Category`. The output JSONL has the same shape as the LLM annotator.

Quality is lower than an LLM annotator, but the annotation is reproducible
and zero-cost. Used to drive Tier-2 token-position patching when the LLM
annotator can't run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from shingeki.config import TOKEN_SPANS_PATH
from shingeki.data.annotators import char_spans_to_token_spans, write_annotations
from shingeki.data.aot import load_d0
from shingeki.data.schemas import (
    CATEGORIES,
    Frame,
    FramingTriplet,
    TokenSpanAnnotation,
)


# ---------- keyword pools (lowercased; matched word-bounded) ----------

PRONOUNS_FIRST = {"i", "my", "me", "mine", "myself", "we", "our", "us", "ours", "ourselves"}
PRONOUNS_THIRD = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"}

HARM_KEYWORDS = {
    "kill", "killed", "killing", "kills",
    "die", "died", "dying", "dies", "dead", "death", "deaths",
    "destroy", "destroyed", "destroying", "destroys", "destruction",
    "crush", "crushed", "crushing",
    "trample", "trampled", "trampling",
    "annihilate", "annihilated", "annihilation",
    "exterminate", "exterminated", "extermination",
    "massacre", "massacred",
    "slaughter", "slaughtered",
    "victim", "victims",
    "casualties", "casualty",
    "innocent", "innocents",
    "civilian", "civilians",
    "refugee", "refugees",
    "wounded", "injured", "harmed", "suffering",
    "blood", "bodies",
}

JUSTIFICATION_KEYWORDS = {
    "had to", "no choice", "no other way", "no alternative",
    "believed", "thought", "convinced", "knew",
    "must", "had no", "could not", "couldn't",
    "duty", "responsibility", "obligation",
    "intended", "intent", "meant to",
    "in order to", "to protect", "to save", "to defend",
    "the only way", "only path",
    "would have", "would die",
    "alternative", "deterrence", "deterrent",
    "compelled", "forced", "coerced",
    "chose", "choose", "chosen", "decision", "decided",
}

ACTION_KEYWORDS = {
    "activated", "deployed", "launched", "ordered",
    "charged", "attacked", "fired", "released",
    "executed", "performed", "carried out",
    "betrayed", "abandoned", "sacrificed",
    "stabbed", "shot", "ambushed",
}

GROUP_ID_KEYWORDS = {
    # AoT proper nouns (kept for D0; D2 uses anonymized variants)
    "eldians", "eldian", "marleyans", "marleyan", "paradis", "marley",
    "hizuru", "shiganshina", "trost", "liberio",
    # anonymized variants from curate_d2
    "group x", "group-x", "group y", "group-y", "homeland h",
    "nation y", "allied nation z",
    # generic group labels likely to appear in the abstract framings
    "ethnic group", "nation", "nations", "coalition", "people", "community",
    "village", "villages",
}


def _word_spans(text: str, vocab: Iterable[str], lower_text: str | None = None) -> list[tuple[int, int]]:
    """Return character spans (start, end) for any word in `vocab` in `text`."""
    if lower_text is None:
        lower_text = text.lower()
    out: list[tuple[int, int]] = []
    for term in vocab:
        for m in re.finditer(r"\b" + re.escape(term) + r"\b", lower_text):
            out.append((m.start(), m.end()))
    return out


def _multi_word_spans(text: str, phrases: Iterable[str], lower_text: str | None = None) -> list[tuple[int, int]]:
    if lower_text is None:
        lower_text = text.lower()
    out = []
    for phrase in phrases:
        idx = 0
        while True:
            i = lower_text.find(phrase, idx)
            if i < 0:
                break
            # require word boundary on both sides
            left_ok = i == 0 or not lower_text[i - 1].isalnum()
            right_ok = (i + len(phrase) >= len(lower_text)) or not lower_text[i + len(phrase)].isalnum()
            if left_ok and right_ok:
                out.append((i, i + len(phrase)))
            idx = i + len(phrase)
    return out


def _decision_maker_spans(text: str, name: str) -> list[tuple[int, int]]:
    if not name or name == "Unknown":
        return []
    parts = name.split()
    spans: list[tuple[int, int]] = []
    # full name + each part
    for term in [name] + parts:
        spans += _multi_word_spans(text, [term.lower()])
    # also pronoun-class (first or third) — pick by whether the framing reads first-person
    # we don't know that here, so we accept both first-person and third-person pronouns
    spans += _word_spans(text, PRONOUNS_FIRST | PRONOUNS_THIRD)
    return spans


def _harm_spans(text: str) -> list[tuple[int, int]]:
    return _word_spans(text, HARM_KEYWORDS)


def _victim_spans(text: str) -> list[tuple[int, int]]:
    # We don't have a clean victim cue list; reuse a slim subset of harm
    # vocabulary that picks out victim references rather than harm verbs.
    victims = {"victim", "victims", "innocent", "innocents", "civilian", "civilians",
               "refugee", "refugees", "child", "children", "boy", "girl", "mother", "father"}
    return _word_spans(text, victims)


def _justification_spans(text: str) -> list[tuple[int, int]]:
    multi = [p for p in JUSTIFICATION_KEYWORDS if " " in p]
    single = JUSTIFICATION_KEYWORDS - set(multi)
    return _word_spans(text, single) + _multi_word_spans(text, multi)


def _action_spans(text: str) -> list[tuple[int, int]]:
    return _word_spans(text, ACTION_KEYWORDS)


def _group_spans(text: str) -> list[tuple[int, int]]:
    multi = [p for p in GROUP_ID_KEYWORDS if " " in p]
    single = GROUP_ID_KEYWORDS - set(multi)
    return _word_spans(text, single) + _multi_word_spans(text, multi)


def _dedupe_overlap(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans = sorted(set(spans))
    out = []
    last_end = -1
    for s, e in spans:
        if s >= last_end:
            out.append((s, e))
            last_end = e
    return out


def annotate_triplet(triplet: FramingTriplet, tokenizer) -> list[TokenSpanAnnotation]:
    """Heuristic annotator. Returns one annotation per framing."""
    annotations: list[TokenSpanAnnotation] = []
    name = triplet.decision_maker
    for frame in ("sympathetic", "condemning", "abstract"):
        text: str = getattr(triplet, frame)
        char_spans: dict = {
            "actor": _dedupe_overlap(_decision_maker_spans(text, name)),
            "victim": _dedupe_overlap(_victim_spans(text)),
            "harm": _dedupe_overlap(_harm_spans(text)),
            "justification": _dedupe_overlap(_justification_spans(text)),
            "group_id": _dedupe_overlap(_group_spans(text)),
            "action": _dedupe_overlap(_action_spans(text)),
            "final_question": [],
            "final_answer": [],
        }
        token_spans = char_spans_to_token_spans(text, tokenizer, char_spans)
        annotations.append(
            TokenSpanAnnotation(
                triplet_id=triplet.id,
                frame=frame,  # type: ignore[arg-type]
                char_spans={c: [tuple(s) for s in char_spans.get(c, [])] for c in CATEGORIES},
                token_spans={c: [tuple(s) for s in token_spans.get(c, [])] for c in CATEGORIES},
            )
        )
    return annotations


def annotate_dataset_heuristic(
    triplets: list[FramingTriplet] | None = None,
    tokenizer=None,
    out_path: Path = TOKEN_SPANS_PATH,
) -> list[TokenSpanAnnotation]:
    if triplets is None:
        triplets = load_d0()
    if tokenizer is None:
        from transformers import AutoTokenizer

        from shingeki.config import GEMMA_3_4B_IT
        from shingeki.env import resolve_hf_token

        tokenizer = AutoTokenizer.from_pretrained(GEMMA_3_4B_IT, token=resolve_hf_token())
    out: list[TokenSpanAnnotation] = []
    for t in triplets:
        out.extend(annotate_triplet(t, tokenizer))
    write_annotations(out, out_path)
    return out

"""Anonymized AoT (D2) builder.

Applies a deterministic name/lore mapping to each AoT triplet. The mapping is
load-bearing — it is what removes the "model recognises AoT lore" confound — so
the map is checked into the repo (`data/d2_anonymized/name_map.json`).

This is intentionally a stub: only a minimal core mapping ships. Manual curation
is expected to fill in remaining entity references; running this builder
end-to-end yields drafts that need a human pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from shingeki.config import D2_PATH, DATA_DIR
from shingeki.data.aot import load_d0
from shingeki.data.schemas import FramingTriplet

NAME_MAP_PATH = DATA_DIR / "d2_anonymized" / "name_map.json"

DEFAULT_NAME_MAP: dict[str, str] = {
    # Decision-makers
    "Eren Yeager": "Leader A",
    "Eren": "Leader A",
    "Erwin Smith": "Commander C",
    "Erwin": "Commander C",
    "Reiner Braun": "Soldier R",
    "Reiner": "Soldier R",
    "Levi Ackerman": "Captain L",
    "Levi": "Captain L",
    "Armin Arlert": "Strategist M",
    "Armin": "Strategist M",
    "Mikasa Ackerman": "Soldier S",
    "Mikasa": "Soldier S",
    "Hange Zoe": "Researcher H",
    "Hange": "Researcher H",
    "Ymir Fritz": "Founder F",
    "Ymir": "Founder F",
    "Historia Reiss": "Sovereign K",
    "Historia": "Sovereign K",
    "Frieda Reiss": "Sovereign K's predecessor",
    "Marco Bodt": "Recruit B",
    "Marco": "Recruit B",
    "Annie Leonhart": "Soldier A",
    "Annie": "Soldier A",
    "Bertholdt Hoover": "Soldier B",
    "Bertholdt": "Soldier B",
    "Zeke Yeager": "Leader Z",
    "Zeke": "Leader Z",
    "Floch Forster": "Recruit F",
    "Floch": "Recruit F",
    "Willy Tybur": "Diplomat W",
    "Ramzi": "Civilian R",
    "Kenny Ackerman": "Operative K",
    "Rod Reiss": "Patriarch X",
    "Grisha Yeager": "Father G",
    # Groups / nations / lore
    "Eldians": "Group X",
    "Eldian": "Group-X",
    "Marleyans": "Group Y",
    "Marleyan": "Group-Y",
    "Marley": "Nation Y",
    "Paradis Island": "Homeland H",
    "Paradis": "Homeland H",
    "the Survey Corps": "the Reconnaissance Corps",
    "Survey Corps": "Reconnaissance Corps",
    "Military Police": "Internal Security Force",
    "Garrison": "Wall Garrison Force",
    "Hizuru": "Allied Nation Z",
    # Concepts
    "the Rumbling": "the mass-deployment weapon",
    "The Rumbling": "The mass-deployment weapon",
    "Titans": "biological war machines",
    "Titan": "biological war machine",
    "Founding Titan": "command-class war machine",
    "Beast Titan": "long-range war machine",
    "Colossal Titan": "siege-class war machine",
    "Colossus Titan": "siege-class war machine",
    "Armored Titan": "fortified war machine",
    "Female Titan": "elite war machine",
    "Attack Titan": "vanguard war machine",
    # Walls
    "Wall Maria": "the outer barrier",
    "Wall Rose": "the middle barrier",
    "Wall Sina": "the inner barrier",
    # Tech / artefacts
    "Liberio": "the foreign internment district",
    "Shiganshina": "border town",
    "Trost": "border outpost",
}


def load_name_map(path: Path = NAME_MAP_PATH) -> dict[str, str]:
    if path.exists():
        return json.loads(path.read_text())
    return dict(DEFAULT_NAME_MAP)


def write_name_map(name_map: dict[str, str], path: Path = NAME_MAP_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(name_map, indent=2, ensure_ascii=False))


def anonymize_text(text: str, name_map: dict[str, str]) -> str:
    # apply longest keys first to avoid partial overlaps (e.g. "Eren Yeager" before "Eren")
    keys = sorted(name_map.keys(), key=len, reverse=True)
    for k in keys:
        # use word-boundary regex; for multi-word keys this still works
        pattern = re.compile(r"\b" + re.escape(k) + r"\b")
        text = pattern.sub(name_map[k], text)
    return text


def anonymize_triplet(t: FramingTriplet, name_map: dict[str, str]) -> FramingTriplet:
    return FramingTriplet(
        id=t.id.replace("aot-", "anon-"),
        dilemma=anonymize_text(t.dilemma, name_map),
        source=t.source,
        decision_maker=name_map.get(t.decision_maker, t.decision_maker),
        ethical_frameworks=list(t.ethical_frameworks),
        sympathetic=anonymize_text(t.sympathetic, name_map),
        condemning=anonymize_text(t.condemning, name_map),
        abstract=anonymize_text(t.abstract, name_map),
    )


def build_d2(
    out_path: Path = D2_PATH,
    name_map_path: Path = NAME_MAP_PATH,
    n: int | None = None,
) -> list[FramingTriplet]:
    name_map = load_name_map(name_map_path)
    if not name_map_path.exists():
        write_name_map(name_map, name_map_path)
    triplets = load_d0()
    if n is not None:
        triplets = triplets[:n]
    out = [anonymize_triplet(t, name_map) for t in triplets]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for a in out:
            f.write(
                json.dumps(
                    {
                        "id": a.id,
                        "dilemma": a.dilemma,
                        "source": a.source,
                        "decision_maker": a.decision_maker,
                        "ethical_frameworks": a.ethical_frameworks,
                        "sympathetic": a.sympathetic,
                        "condemning": a.condemning,
                        "abstract": a.abstract,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out

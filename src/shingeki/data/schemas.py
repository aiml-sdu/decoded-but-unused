"""Dataclasses shared across data loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Frame = Literal["sympathetic", "condemning", "abstract"]
FRAMES: tuple[Frame, Frame, Frame] = ("sympathetic", "condemning", "abstract")

Ambiguity = Literal["high", "low"]
MoralChoiceFrame = Literal["sympathetic", "condemning", "neutral"]
MORALCHOICE_FRAMES: tuple[MoralChoiceFrame, MoralChoiceFrame, MoralChoiceFrame] = (
    "sympathetic",
    "condemning",
    "neutral",
)

Category = Literal[
    "actor",
    "victim",
    "harm",
    "justification",
    "group_id",
    "action",
    "final_question",
    "final_answer",
]
CATEGORIES: tuple[Category, ...] = (
    "actor",
    "victim",
    "harm",
    "justification",
    "group_id",
    "action",
    "final_question",
    "final_answer",
)


@dataclass
class FramingTriplet:
    id: str
    dilemma: str
    source: str
    decision_maker: str
    ethical_frameworks: list[str]
    sympathetic: str
    condemning: str
    abstract: str

    def text(self, frame: Frame) -> str:
        return getattr(self, frame)


@dataclass
class MoralChoiceItem:
    id: str
    ambiguity: Ambiguity
    scenario: str
    action_a: str
    action_b: str
    frame: MoralChoiceFrame
    frame_prefix: str

    @property
    def composite_id(self) -> str:
        return f"{self.id}__{self.frame}"


@dataclass
class TokenSpanAnnotation:
    triplet_id: str
    frame: Frame
    char_spans: dict[Category, list[tuple[int, int]]] = field(default_factory=dict)
    token_spans: dict[Category, list[tuple[int, int]]] = field(default_factory=dict)

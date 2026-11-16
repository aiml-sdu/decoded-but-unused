"""Prompt builders.

`BuiltPrompt` carries everything downstream code needs:
  - `prompt_text`: the raw user-message text (pre chat template)
  - `chat_text`:   the chat-template-formatted string fed to the tokenizer
  - `input_ids`:   torch.LongTensor of shape [1, T]
  - `attention_mask`: torch.LongTensor of shape [1, T]
  - `final_token_idx`: int, the index of the position whose logits we read for
    the answer score (T - 1 if the chat template ends right where generation
    starts, which is the standard case)
  - `scenario_char_span`: (start, end) char span in `chat_text` of the scenario
    body, useful for token-categoriser alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from shingeki.data.schemas import Frame, FramingTriplet, MoralChoiceItem
from shingeki.prompts.chat import ModelFamily, apply_chat_template
from shingeki.prompts.templates import (
    AB_V2,
    ACCEPTABLE_V2,
    AnswerFormat,
    JUSTIFIED_V2,
    LEGACY_EXPLAIN,
    MORALCHOICE_AB,
    YESNO_V2,
)


@dataclass
class BuiltPrompt:
    prompt_text: str
    chat_text: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    final_token_idx: int
    scenario_char_span: tuple[int, int] | None = None
    meta: dict[str, Any] | None = None


def _format_aot(
    triplet: FramingTriplet, frame: Frame, fmt: AnswerFormat, ab_alt: str = "Refuse and accept the consequences"
) -> tuple[str, tuple[int, int]]:
    text = triplet.text(frame)
    if fmt == "yesno":
        prompt = YESNO_V2.format(text=text)
        scen = prompt.find(text)
    elif fmt == "justified":
        prompt = JUSTIFIED_V2.format(text=text)
        scen = prompt.find(text)
    elif fmt == "acceptable":
        prompt = ACCEPTABLE_V2.format(text=text)
        scen = prompt.find(text)
    elif fmt == "ab":
        prompt = AB_V2.format(text=text, action_a=triplet.dilemma, action_b=ab_alt)
        scen = prompt.find(text)
    elif fmt == "legacy":
        prompt = LEGACY_EXPLAIN.format(text=text)
        scen = prompt.find(text)
    else:
        raise ValueError(f"unknown fmt={fmt}")
    return prompt, (scen, scen + len(text))


def build_aot_prompt(
    triplet: FramingTriplet,
    frame: Frame,
    fmt: AnswerFormat,
    family: ModelFamily,
    tokenizer,
    device: str = "cpu",
) -> BuiltPrompt:
    prompt_text, scen_span = _format_aot(triplet, frame, fmt)
    chat_text = apply_chat_template(prompt_text, family, tokenizer)
    enc = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    attention_mask = (
        attn.to(device) if attn is not None else torch.ones_like(input_ids)
    )
    final_idx = input_ids.shape[1] - 1
    # the scenario span moved if the chat template was applied (it adds a
    # prefix); recompute against `chat_text`.
    scen_in_chat = chat_text.find(triplet.text(frame))
    if scen_in_chat >= 0:
        scen_span = (scen_in_chat, scen_in_chat + len(triplet.text(frame)))
    return BuiltPrompt(
        prompt_text=prompt_text,
        chat_text=chat_text,
        input_ids=input_ids,
        attention_mask=attention_mask,
        final_token_idx=final_idx,
        scenario_char_span=scen_span,
        meta={"triplet_id": triplet.id, "frame": frame, "fmt": fmt},
    )


def build_moralchoice_prompt(
    item: MoralChoiceItem,
    family: ModelFamily,
    tokenizer,
    device: str = "cpu",
) -> BuiltPrompt:
    prompt_text = MORALCHOICE_AB.format(
        frame_prefix=item.frame_prefix,
        scenario=item.scenario,
        action_a=item.action_a,
        action_b=item.action_b,
    )
    chat_text = apply_chat_template(prompt_text, family, tokenizer)
    enc = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    attention_mask = (
        attn.to(device) if attn is not None else torch.ones_like(input_ids)
    )
    return BuiltPrompt(
        prompt_text=prompt_text,
        chat_text=chat_text,
        input_ids=input_ids,
        attention_mask=attention_mask,
        final_token_idx=input_ids.shape[1] - 1,
        meta={"id": item.id, "frame": item.frame, "ambiguity": item.ambiguity},
    )

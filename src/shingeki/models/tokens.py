"""Answer-token resolution.

For each `AnswerFormat`, return either two single-token ids or the
`fallback`-style sequence-logprob path. The dispatcher chooses based on
whether the strings tokenise to single ids in the target tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from shingeki.prompts.templates import ANSWER_PAIRS, AnswerFormat


@dataclass
class AnswerTokens:
    fmt: AnswerFormat
    pos_str: str
    neg_str: str
    pos_ids: list[int]
    neg_ids: list[int]
    single_token: bool

    @property
    def pos_id(self) -> int:
        if not self.single_token:
            raise RuntimeError("not single-token; use sequence_logprob")
        return self.pos_ids[0]

    @property
    def neg_id(self) -> int:
        if not self.single_token:
            raise RuntimeError("not single-token; use sequence_logprob")
        return self.neg_ids[0]


def resolve_answer_tokens(tokenizer, fmt: AnswerFormat) -> AnswerTokens:
    pos_str, neg_str = ANSWER_PAIRS[fmt]
    pos_ids = tokenizer.encode(pos_str, add_special_tokens=False)
    neg_ids = tokenizer.encode(neg_str, add_special_tokens=False)
    # also try the leading-space variants without the space, in case the
    # tokenizer prefers the no-space form; pick the shorter encoding.
    alt_pos = tokenizer.encode(pos_str.lstrip(), add_special_tokens=False)
    alt_neg = tokenizer.encode(neg_str.lstrip(), add_special_tokens=False)
    if len(alt_pos) < len(pos_ids):
        pos_ids, pos_str = alt_pos, pos_str.lstrip()
    if len(alt_neg) < len(neg_ids):
        neg_ids, neg_str = alt_neg, neg_str.lstrip()
    return AnswerTokens(
        fmt=fmt,
        pos_str=pos_str,
        neg_str=neg_str,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        single_token=(len(pos_ids) == 1 and len(neg_ids) == 1),
    )


@torch.inference_mode()
def sequence_logprob(model, tokenizer, prefix_ids: torch.Tensor, completion_ids: list[int]) -> float:
    """Sum-of-log-probs of `completion_ids` after `prefix_ids` (single sequence)."""
    full = torch.cat(
        [prefix_ids.flatten(), torch.tensor(completion_ids, device=prefix_ids.device)]
    ).unsqueeze(0)
    logits = model(full).logits if hasattr(model, "lm_head") else model(full)
    if isinstance(logits, tuple):
        logits = logits[0]
    # log probs at positions T-len(c) .. T-1 predict completion_ids
    T = full.shape[1]
    n = len(completion_ids)
    target_logits = logits[0, T - n - 1 : T - 1]  # predicts positions T-n .. T-1
    target_logp = F.log_softmax(target_logits.float(), dim=-1)
    targets = torch.tensor(completion_ids, device=target_logp.device)
    return float(target_logp.gather(1, targets.unsqueeze(1)).sum().item())

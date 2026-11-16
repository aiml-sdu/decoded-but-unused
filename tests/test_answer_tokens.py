"""Answer-token resolution invariants."""

import pytest
from transformers import AutoTokenizer

from shingeki.env import resolve_hf_token
from shingeki.models.tokens import resolve_answer_tokens
from shingeki.prompts.templates import ANSWER_FORMATS


@pytest.mark.parametrize("model", ["google/gemma-3-1b-it"])
def test_yes_no_single_token(model):
    token = resolve_hf_token()
    if not token:
        pytest.skip("Gemma tokenization check requires HF_TOKEN or HF_API_KEY for gated model access")
    tok = AutoTokenizer.from_pretrained(model, token=token)
    ans = resolve_answer_tokens(tok, "yesno")
    assert ans.single_token, f"{model}: ' Yes'/' No' should be single tokens"
    ans_ab = resolve_answer_tokens(tok, "ab")
    assert ans_ab.single_token


def test_all_formats_resolve_without_error():
    tok = AutoTokenizer.from_pretrained("gpt2")
    for fmt in ANSWER_FORMATS:
        ans = resolve_answer_tokens(tok, fmt)
        assert len(ans.pos_ids) >= 1
        assert len(ans.neg_ids) >= 1

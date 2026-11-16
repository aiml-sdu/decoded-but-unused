"""Prompt-builder snapshot tests."""

from shingeki.data.aot import load_d0
from shingeki.prompts.builders import _format_aot
from shingeki.prompts.templates import ANSWER_FORMATS


def test_yesno_prompt_shape():
    t = load_d0()[0]
    prompt, span = _format_aot(t, "sympathetic", "yesno")
    assert "Answer only Yes or No." in prompt
    assert "Scenario:" in prompt
    body = prompt[span[0] : span[1]]
    assert body == t.sympathetic


def test_all_formats_buildable():
    t = load_d0()[0]
    for fmt in ANSWER_FORMATS:
        prompt, _ = _format_aot(t, "condemning", fmt)
        assert "Scenario:" in prompt or "scenario" in prompt.lower()

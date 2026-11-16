"""Chat-template wrappers per model family.

Each adapter takes a single user-message string and returns the fully formatted
input suitable for tokenisation. Instruction-tuned models use their own chat
template; pretraining (PT) models use the raw text.
"""

from __future__ import annotations

from typing import Literal

ModelFamily = Literal[
    "gemma3-it", "gemma3-pt", "gemma2-it", "qwen25-it", "llama31-it", "gpt2"
]


def apply_chat_template(text: str, family: ModelFamily, tokenizer) -> str:
    """Return the chat-formatted input string. The tokenizer is consulted only
    when its `apply_chat_template` knows about the family."""
    if family in ("gemma3-it", "gemma2-it", "qwen25-it", "llama31-it"):
        # Use the tokenizer's chat template if available — it handles each
        # family's special tokens correctly.
        try:
            messages = [{"role": "user", "content": text}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
        # Fallbacks (rarely needed; here for safety).
        if family.startswith("gemma"):
            return (
                f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
            )
        if family.startswith("qwen"):
            return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

    # PT and gpt2 use raw text.
    return text

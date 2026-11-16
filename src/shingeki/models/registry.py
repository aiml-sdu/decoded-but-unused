"""Model registry: identifiers, families, and the architectural specs that
matter to downstream analysis (layer count, residual dim, layer-name template).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from shingeki.config import (
    GEMMA_2_2B_IT,
    GEMMA_3_1B_IT,
    GEMMA_3_4B_IT,
    GEMMA_3_4B_PT,
    QWEN_25_3B_IT,
)
from shingeki.prompts.chat import ModelFamily

Backend = Literal["hf", "tl"]


@dataclass
class ModelSpec:
    id: str
    hf_name: str
    family: ModelFamily
    backend: Backend  # "hf" → custom hook wrapper, "tl" → TransformerLens
    primary: bool = False


# transformer-lens 2.16 has zero Gemma-3 support; use the HF backend for those.
# Gemma-2 and gpt2 are fine on TL.
MODELS: dict[str, ModelSpec] = {
    "gemma3-4b-it": ModelSpec(
        "gemma3-4b-it", GEMMA_3_4B_IT, "gemma3-it", "hf", primary=True
    ),
    "gemma3-4b-pt": ModelSpec("gemma3-4b-pt", GEMMA_3_4B_PT, "gemma3-pt", "hf"),
    "gemma3-1b-it": ModelSpec("gemma3-1b-it", GEMMA_3_1B_IT, "gemma3-it", "hf"),
    "gemma3-12b-it": ModelSpec("gemma3-12b-it", "google/gemma-3-12b-it", "gemma3-it", "hf"),
    "gemma2-2b-it": ModelSpec("gemma2-2b-it", GEMMA_2_2B_IT, "gemma2-it", "hf"),
    "gemma2-2b-pt": ModelSpec("gemma2-2b-pt", "google/gemma-2-2b", "gemma2-it", "hf"),
    "gemma2-9b-it": ModelSpec("gemma2-9b-it", "google/gemma-2-9b-it", "gemma2-it", "hf"),
    "gemma2-27b-it": ModelSpec("gemma2-27b-it", "google/gemma-2-27b-it", "gemma2-it", "hf"),
    "qwen25-3b-it": ModelSpec("qwen25-3b-it", QWEN_25_3B_IT, "qwen25-it", "hf"),
    "qwen25-7b-it": ModelSpec("qwen25-7b-it", "Qwen/Qwen2.5-7B-Instruct", "qwen25-it", "hf"),
    "qwen25-14b-it": ModelSpec("qwen25-14b-it", "Qwen/Qwen2.5-14B-Instruct", "qwen25-it", "hf"),
    "qwen25-32b-it": ModelSpec("qwen25-32b-it", "Qwen/Qwen2.5-32B-Instruct", "qwen25-it", "hf"),
    "llama31-8b-it": ModelSpec("llama31-8b-it", "meta-llama/Llama-3.1-8B-Instruct", "llama31-it", "hf"),
    "gpt2": ModelSpec("gpt2", "gpt2", "gpt2", "tl"),
}


def get(spec_id: str) -> ModelSpec:
    if spec_id not in MODELS:
        raise KeyError(f"unknown model id {spec_id!r}; known: {sorted(MODELS)}")
    return MODELS[spec_id]

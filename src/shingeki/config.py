"""Project-wide configuration: paths, model identifiers, seed."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
LEGACY_DIR = PROJECT_ROOT / "legacy"

# Datasets
D0_PATH = DATA_DIR / "d0_aot" / "aot_framing_triplets.jsonl"
D1_PATH = DATA_DIR / "d1_style_controlled" / "aot_style_controlled.jsonl"
D1_V2_PATH = DATA_DIR / "d1_style_controlled" / "aot_style_controlled_v2.jsonl"
D2_PATH = DATA_DIR / "d2_anonymized" / "aot_anonymized.jsonl"
D3_PATH = DATA_DIR / "d3_moralchoice" / "moralchoice_frame.jsonl"
D6_AOT_EXTENSION_PATH = DATA_DIR / "d6_aot_extension" / "aot_framing_extension.jsonl"
D7_GOT_PATH = DATA_DIR / "d7_got" / "got_framing_triplets.jsonl"
TOKEN_SPANS_PATH = DATA_DIR / "token_categories" / "aot_token_spans.jsonl"

# Models
GEMMA_3_4B_IT = "google/gemma-3-4b-it"
GEMMA_3_4B_PT = "google/gemma-3-4b-pt"
GEMMA_3_1B_IT = "google/gemma-3-1b-it"
GEMMA_2_2B_IT = "google/gemma-2-2b-it"
QWEN_25_3B_IT = "Qwen/Qwen2.5-3B-Instruct"

PRIMARY_MODEL = GEMMA_3_4B_IT

# OpenRouter (for behavioural validation + token-span annotation)
OPENROUTER_MODELS = [
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash",
    "openai/gpt-4.1-nano",
]
ANNOTATOR_MODEL = "anthropic/claude-haiku-4.5"

# SAE config (Gemma Scope 2) — kept for future SAE experiments; not used in Tier 1+2+3.
GEMMA_SCOPE_REPO = "google/gemma-scope-3-1b-pt-res"

SEED = 42
BOOTSTRAP_RESAMPLES = 10_000
PERMUTATION_SAMPLES = 10_000

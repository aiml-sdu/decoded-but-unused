"""Environment variable resolution.

Loads `.env` (if present) into `os.environ` without overriding existing values,
then exposes typed accessors. Both `HF_TOKEN` and `HF_API_KEY` are accepted
because users in the wild use both names.
"""

from __future__ import annotations

import os
from pathlib import Path

from shingeki.config import PROJECT_ROOT


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


_load_dotenv(PROJECT_ROOT / ".env")


def resolve_hf_token() -> str | None:
    """Return the Hugging Face token, accepting either env-var name."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")


def resolve_openrouter_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY")

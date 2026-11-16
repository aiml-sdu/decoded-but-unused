"""Reproducibility manifest for the nightly run.

Captures:
  - python / package versions
  - git sha + dirty flag
  - model identifiers (and HF revision SHAs when available)
  - tokenizer ids for every answer format we use
  - chat-template hashes for every model family
  - dataset row counts and file SHAs
  - GPU / driver string

Single output: `<nightly_dir>/manifest.json`.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from shingeki.config import (
    D0_PATH,
    D1_PATH,
    D1_V2_PATH,
    D2_PATH,
    D3_PATH,
    GEMMA_2_2B_IT,
    GEMMA_3_4B_IT,
    GEMMA_3_4B_PT,
    QWEN_25_3B_IT,
)


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _file_sha(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _file_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open() as f:
        return sum(1 for _ in f)


def _gpu_string() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _pkg_versions() -> dict[str, str | None]:
    names = (
        "torch",
        "transformers",
        "transformer-lens",
        "datasets",
        "scikit-learn",
        "numpy",
        "scipy",
        "huggingface-hub",
        "tokenizers",
        "accelerate",
    )
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = None
    return out


def _resolve_tokens(hf_id: str) -> dict[str, Any]:
    """Load tokenizer once and resolve answer-token ids for every format."""
    from transformers import AutoTokenizer

    from shingeki.env import resolve_hf_token
    from shingeki.models.tokens import resolve_answer_tokens

    token = resolve_hf_token()
    kwargs = {"token": token} if token else {}
    tok = AutoTokenizer.from_pretrained(hf_id, **kwargs)
    out: dict[str, Any] = {
        "tokenizer_class": type(tok).__name__,
        "vocab_size": int(getattr(tok, "vocab_size", -1)),
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
        "answer_formats": {},
        "chat_template_sha": None,
    }
    for fmt in ("yesno", "ab", "justified", "acceptable"):
        try:
            ans = resolve_answer_tokens(tok, fmt)
            out["answer_formats"][fmt] = {
                "pos_str": ans.pos_str,
                "neg_str": ans.neg_str,
                "pos_ids": ans.pos_ids,
                "neg_ids": ans.neg_ids,
                "single_token": ans.single_token,
            }
        except Exception as e:
            out["answer_formats"][fmt] = {"error": f"{type(e).__name__}: {e}"}
    template = getattr(tok, "chat_template", None)
    if template:
        out["chat_template_sha"] = hashlib.sha256(template.encode()).hexdigest()[:16]
        out["chat_template_len"] = len(template)
    return out


@dataclass
class NightlyManifest:
    nightly_dir: str
    started_at: str
    python: str = field(default_factory=platform.python_version)
    sys_executable: str = field(default_factory=lambda: sys.executable)
    git_sha: str = field(default_factory=lambda: _git(["rev-parse", "HEAD"]))
    git_short: str = field(default_factory=lambda: _git(["rev-parse", "--short=12", "HEAD"]))
    git_dirty: bool = field(default_factory=lambda: bool(_git(["status", "--porcelain"])))
    git_branch: str = field(default_factory=lambda: _git(["rev-parse", "--abbrev-ref", "HEAD"]))
    git_origin: str = field(default_factory=lambda: _git(["config", "--get", "remote.origin.url"]))
    deps: dict[str, str | None] = field(default_factory=_pkg_versions)
    gpu: str = field(default_factory=_gpu_string)
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)

    def fill_datasets(self) -> None:
        for name, path in (
            ("d0_aot", D0_PATH),
            ("d1_style", D1_PATH),
            ("d1_style_v2", D1_V2_PATH),
            ("d2_anonymized", D2_PATH),
            ("d3_moralchoice", D3_PATH),
        ):
            self.datasets[name] = {
                "path": str(path),
                "exists": path.exists(),
                "n_lines": _file_lines(path),
                "sha256_16": _file_sha(path),
            }

    def fill_models(self, *, ids: tuple[str, ...] | None = None) -> None:
        ids = ids or (GEMMA_3_4B_IT, GEMMA_3_4B_PT, GEMMA_2_2B_IT, QWEN_25_3B_IT)
        for hf_id in ids:
            try:
                self.models[hf_id] = _resolve_tokens(hf_id)
            except Exception as e:
                self.models[hf_id] = {"error": f"{type(e).__name__}: {e}"}


def write_manifest(nightly_dir: Path, started_at: str) -> Path:
    nightly_dir.mkdir(parents=True, exist_ok=True)
    m = NightlyManifest(nightly_dir=str(nightly_dir), started_at=started_at)
    m.fill_datasets()
    m.fill_models()
    out = nightly_dir / "manifest.json"
    out.write_text(json.dumps(asdict(m), indent=2, default=str))
    return out

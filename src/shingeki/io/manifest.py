"""Run-provenance manifest: git sha, config hash, dependency snapshot."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "nogit"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        )
        return bool(out.decode().strip())
    except Exception:
        return False


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def _pkg_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


@dataclass
class Manifest:
    exp_id: str
    config: dict[str, Any]
    git_sha: str = field(default_factory=_git_sha)
    git_dirty: bool = field(default_factory=_git_dirty)
    python: str = field(default_factory=platform.python_version)
    deps: dict[str, str | None] = field(
        default_factory=lambda: {
            name: _pkg_version(name)
            for name in (
                "torch",
                "transformers",
                "transformer-lens",
                "datasets",
                "scikit-learn",
                "numpy",
                "scipy",
            )
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

"""Deterministic result-bundle writer.

Every experiment returns a `ResultBundle`. Each bundle is materialised under
`runs/<git_sha>__<config_hash>/<exp_id>/` with `result.json` (scalars/metadata),
`arrays.npz` (numpy arrays), `manifest.json` (provenance), and figure paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shingeki.config import RUNS_DIR
from shingeki.io.manifest import Manifest, config_hash


@dataclass
class ResultBundle:
    exp_id: str
    config: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    figures: list[Path] = field(default_factory=list)
    notes: str = ""


def make_run_dir(exp_id: str, cfg: dict[str, Any]) -> Path:
    manifest = Manifest(exp_id=exp_id, config=cfg)
    name = f"{manifest.git_sha}__{config_hash(cfg)}"
    out = RUNS_DIR / name / exp_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_bundle(bundle: ResultBundle, run_dir: Path | None = None) -> Path:
    if run_dir is None:
        run_dir = make_run_dir(bundle.exp_id, bundle.config)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(exp_id=bundle.exp_id, config=bundle.config).to_dict()

    payload = {
        "exp_id": bundle.exp_id,
        "config": bundle.config,
        "metrics": _jsonable(bundle.metrics),
        "figures": [str(p) for p in bundle.figures],
        "notes": bundle.notes,
    }
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if bundle.arrays:
        np.savez_compressed(run_dir / "arrays.npz", **bundle.arrays)
    return run_dir


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj

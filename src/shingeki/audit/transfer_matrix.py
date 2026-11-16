"""Cross-domain probe transfer matrix (gap A).

Trains an LR probe on dataset X at the best layer (chosen by within-X CV) and
evaluates on dataset Y. Reports a 3x3 matrix over D0/D1/D2 plus the within-CV
diagonal for reference, and the per-layer transfer curve from D0 to D1/D2.

Why this matters:
  Probe accuracy on D0 alone could be driven by lexical or lore cues. If the
  same probe transfers to D1 (style-controlled) and D2 (anonymized) at high
  accuracy, the framing representation is largely non-lexical and non-lore.

The probe activations cache (X, y, groups) is reused if already loaded; we
expect the orchestrator to pass these dicts from the in-memory IT run rather
than re-running activations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def _fit_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    c: float = 1.0,
    seed: int = 42,
) -> dict:
    clf = LogisticRegression(
        C=c,
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return {
        "accuracy": float(accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro")),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }


def _ensure_same_classes(y_tr: np.ndarray, y_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Restrict to classes present in both splits - defensive guard for tiny D1."""
    common = np.intersect1d(np.unique(y_tr), np.unique(y_te))
    tr_mask = np.isin(y_tr, common)
    te_mask = np.isin(y_te, common)
    return tr_mask, te_mask


def transfer_matrix_per_layer(
    caches: dict[str, dict],
    layers: list[int] | None = None,
) -> dict:
    """Build a transfer matrix at every layer for every (train, test) pair.

    Parameters
    ----------
    caches : dict
        ``caches[ds] = {"X": [N, L, D], "y": [N], "groups": [N]}`` for ds in
        {"d0", "d1", "d2"}.
    layers : list[int] | None
        Layers to evaluate. Default: all layers.

    Returns
    -------
    dict with keys:
      "datasets" : sorted list of dataset names
      "layers"   : list[int]
      "matrix_per_layer" : list of L dicts, each mapping
                           "{train}->{test}" -> {accuracy, macro_f1, ...}
      "best_per_pair"    : dict mapping pair -> {best_layer, best_accuracy}
    """
    names = sorted(caches.keys())
    L = next(iter(caches.values()))["X"].shape[1]
    layers = layers or list(range(L))
    per_layer: list[dict] = []
    for li in layers:
        layer_block: dict = {}
        for tr_name in names:
            tr = caches[tr_name]
            X_tr_full, y_tr_full = tr["X"][:, li], tr["y"]
            for te_name in names:
                te = caches[te_name]
                X_te_full, y_te_full = te["X"][:, li], te["y"]
                if tr_name == te_name:
                    # within-domain: leave-one-triplet-out; aggregate over folds
                    groups = tr["groups"]
                    preds = np.empty_like(y_tr_full)
                    for tid in np.unique(groups):
                        te_mask = groups == tid
                        tr_mask = ~te_mask
                        clf = LogisticRegression(
                            C=1.0,
                            max_iter=5000,
                            class_weight="balanced",
                            solver="lbfgs",
                            random_state=42,
                        )
                        clf.fit(X_tr_full[tr_mask], y_tr_full[tr_mask])
                        preds[te_mask] = clf.predict(X_tr_full[te_mask])
                    layer_block[f"{tr_name}->{te_name}"] = {
                        "accuracy": float(accuracy_score(y_tr_full, preds)),
                        "macro_f1": float(f1_score(y_tr_full, preds, average="macro")),
                        "n_train": int(len(y_tr_full)),
                        "n_test": int(len(y_tr_full)),
                        "kind": "within_loto",
                    }
                else:
                    tr_mask, te_mask = _ensure_same_classes(y_tr_full, y_te_full)
                    res = _fit_predict(
                        X_tr_full[tr_mask],
                        y_tr_full[tr_mask],
                        X_te_full[te_mask],
                        y_te_full[te_mask],
                    )
                    res["kind"] = "cross_full"
                    layer_block[f"{tr_name}->{te_name}"] = res
        per_layer.append(layer_block)

    # collapse to best-per-pair
    best: dict[str, dict] = {}
    for tr_name in names:
        for te_name in names:
            key = f"{tr_name}->{te_name}"
            accs = [per_layer[i][key]["accuracy"] for i in range(len(layers))]
            j = int(np.argmax(accs))
            best[key] = {
                "best_layer": int(layers[j]),
                "best_accuracy": float(accs[j]),
                "best_macro_f1": float(per_layer[j][key]["macro_f1"]),
            }
    return {
        "datasets": names,
        "layers": layers,
        "matrix_per_layer": per_layer,
        "best_per_pair": best,
    }


def write_transfer_audit(out_dir: Path, caches: dict[str, dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = transfer_matrix_per_layer(caches)
    (out_dir / "result.json").write_text(json.dumps(res, indent=2, default=str))
    # also emit a flat CSV for easy inspection
    csv_lines = ["pair,best_layer,best_accuracy,best_macro_f1"]
    for pair, sub in res["best_per_pair"].items():
        csv_lines.append(
            f"{pair},{sub['best_layer']},{sub['best_accuracy']:.4f},{sub['best_macro_f1']:.4f}"
        )
    (out_dir / "best_per_pair.csv").write_text("\n".join(csv_lines) + "\n")
    return out_dir

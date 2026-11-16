"""Probe controls: random labels, length-only, TF-IDF baselines."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from shingeki.probing.splits import loto_splits
from shingeki.stats.selectivity import make_control_labels


def random_label_baseline(
    X_layer: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 42
) -> dict:
    """Stable random labels per group → probe should hit chance."""
    n_classes = len(np.unique(y))
    y_rand = make_control_labels(groups, y, n_classes, seed=seed)
    accs = []
    for tr, te in loto_splits(groups):
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
        clf.fit(X_layer[tr], y_rand[tr])
        accs.append(accuracy_score(y_rand[te], clf.predict(X_layer[te])))
    return {"mean_acc": float(np.mean(accs)), "per_fold": [float(a) for a in accs]}


def length_only_baseline(
    texts: list[str], y: np.ndarray, groups: np.ndarray, seed: int = 42
) -> dict:
    X = np.array([[len(t)] for t in texts], dtype=np.float32)
    accs = []
    for tr, te in loto_splits(groups):
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
        clf.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(X[te])))
    return {"mean_acc": float(np.mean(accs)), "per_fold": [float(a) for a in accs]}


def tfidf_baseline(
    texts: list[str], y: np.ndarray, groups: np.ndarray, seed: int = 42, max_features: int = 5000
) -> dict:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    accs = []
    for tr, te in loto_splits(groups):
        clf = LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=seed, solver="lbfgs"
        )
        clf.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(X[te])))
    return {
        "mean_acc": float(np.mean(accs)),
        "macro_f1": float(
            f1_score(y, _global_predictions(X, y, groups, seed=seed), average="macro")
        ),
        "per_fold": [float(a) for a in accs],
    }


def _global_predictions(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 42
) -> np.ndarray:
    preds = np.empty_like(y)
    for tr, te in loto_splits(groups):
        clf = LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=seed, solver="lbfgs"
        )
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
    return preds

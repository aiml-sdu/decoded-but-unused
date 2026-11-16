"""Cross-domain probe transfer: train on one dataset, test on another."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def transfer_probe(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, seed: int = 42
) -> dict:
    clf = LogisticRegression(
        max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=seed
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }

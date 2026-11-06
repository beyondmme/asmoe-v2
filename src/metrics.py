"""Metric computation and cross-fold aggregation."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> Dict:
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": 100.0 * accuracy_score(y_true, y_pred),
        "macro_f1": 100.0 * f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": 100.0 * f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i, c in enumerate(class_names)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def aggregate_folds(fold_metrics: List[Dict], class_names: Sequence[str]) -> Dict:
    """Mean ± SD (sample SD, ddof=1) across folds + pooled confusion matrix."""
    out: Dict = {"n_folds": len(fold_metrics)}
    for k in ("accuracy", "macro_f1", "weighted_f1", "mcc"):
        v = np.array([m[k] for m in fold_metrics], dtype=float)
        out[f"{k}_mean"], out[f"{k}_sd"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
    out["per_class"] = {}
    for c in class_names:
        out["per_class"][c] = {}
        for k in ("precision", "recall", "f1"):
            v = np.array([m["per_class"][c][k] for m in fold_metrics], dtype=float)
            out["per_class"][c][f"{k}_mean"] = float(v.mean())
            out["per_class"][c][f"{k}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    out["confusion_matrix_pooled"] = np.sum([m["confusion_matrix"] for m in fold_metrics], axis=0).tolist()
    return out

"""Rare-event evaluation metrics.

Two precision-recall summaries are reported and must not be conflated:

* ``average_precision`` -- scikit-learn's ``average_precision_score``: the
  step-wise sum of precision weighted by recall increments. This is the
  quantity used for model selection and reported in the historical results.
* ``pr_auc_trapezoidal`` -- ``auc(recall, precision)`` over the same curve,
  which linearly interpolates between operating points and is typically
  optimistic. It is reported only for reference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from .thresholds import apply_threshold


def confusion_counts(y_true, predictions) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def ranking_metrics(y_true, scores) -> dict[str, float | None]:
    """Threshold-free summaries; ``None`` when a class is absent (undefined)."""
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return {"average_precision": None, "pr_auc_trapezoidal": None, "roc_auc": None}
    precision, recall, _ = precision_recall_curve(y, scores)
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "pr_auc_trapezoidal": float(auc(recall, precision)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def evaluate_scores(
    model_name: str,
    policy_name: str,
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Confusion-derived and ranking metrics for one model/policy on one split."""
    predictions = apply_threshold(scores, threshold)
    counts = confusion_counts(y_true, predictions)
    tp, fp, fn, tn = (
        counts["true_positives"],
        counts["false_positives"],
        counts["false_negatives"],
        counts["true_negatives"],
    )
    return {
        "model": model_name,
        "policy": policy_name,
        "threshold": float(threshold),
        "precision": _safe_divide(tp, tp + fp),
        "recall": _safe_divide(tp, tp + fn),
        # 2TP / (2TP + FP + FN) is algebraically F1 and matches sklearn's f1_score bit-for-bit.
        "f1": _safe_divide(2 * tp, 2 * tp + fp + fn),
        **ranking_metrics(y_true, scores),
        **counts,
        "false_positives_per_1000_legitimate": _safe_divide(fp, tn + fp) * 1000,
        "alert_rate": _safe_divide(tp + fp, len(y_true)),
        "test_rows": int(len(y_true)),
        "test_fraud_cases": int(tp + fn),
    }

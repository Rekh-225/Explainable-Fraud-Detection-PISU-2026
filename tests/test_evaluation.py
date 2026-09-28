"""Metric arithmetic checked against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from fraud_pipeline.evaluation import confusion_counts, evaluate_scores, ranking_metrics


def test_confusion_and_derived_metrics_match_hand_computation():
    y = pd.Series([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.4, 0.2, 0.7, 0.3, 0.1, 0.05, 0.02, 0.01])
    m = evaluate_scores("m", "p", y, s, threshold=0.5)
    # score >= 0.5: rows 0,1 (TP) and 4 (FP)
    assert (m["true_positives"], m["false_positives"], m["false_negatives"], m["true_negatives"]) == (2, 1, 2, 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))
    assert m["false_positives_per_1000_legitimate"] == pytest.approx(1 / 6 * 1000)
    assert m["alert_rate"] == pytest.approx(3 / 10)
    assert m["test_rows"] == 10 and m["test_fraud_cases"] == 4
    assert m["threshold"] == 0.5


def test_zero_alerts_yield_zero_precision_without_error():
    y = pd.Series([1, 0, 0])
    m = evaluate_scores("m", "p", y, np.array([0.1, 0.1, 0.1]), threshold=0.9)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    assert m["alert_rate"] == 0.0 and m["false_positives_per_1000_legitimate"] == 0.0


def test_average_precision_differs_from_trapezoidal_pr_auc():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    s = np.array([0.9, 0.85, 0.7, 0.6, 0.5, 0.45, 0.2, 0.1])
    r = ranking_metrics(y, s)
    # AP: sum over positives of precision at each recall step (1/3 each).
    expected_ap = (1 / 3) * (1 / 1 + 2 / 3 + 3 / 6)
    assert r["average_precision"] == pytest.approx(expected_ap)
    assert r["pr_auc_trapezoidal"] != pytest.approx(r["average_precision"])
    assert 0 < r["roc_auc"] < 1


def test_ranking_metrics_are_undefined_with_one_class():
    r = ranking_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    assert r == {"average_precision": None, "pr_auc_trapezoidal": None, "roc_auc": None}


def test_confusion_counts_uses_fixed_label_order():
    assert confusion_counts([0, 0], [0, 0]) == {
        "true_negatives": 2,
        "false_positives": 0,
        "false_negatives": 0,
        "true_positives": 0,
    }

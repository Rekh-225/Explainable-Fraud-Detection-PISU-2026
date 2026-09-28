"""Threshold policy: target recall, ties, comparison semantics, degenerate inputs."""

import numpy as np
import pytest

from fraud_pipeline.thresholds import (
    ThresholdSelectionError,
    apply_threshold,
    select_operating_threshold,
)


def test_highest_precision_threshold_reaching_recall_target():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.85, 0.5])
    d = select_operating_threshold(y, s, minimum_recall=0.8)
    # Candidates with recall >= 0.8: 0.6 (TP=4, FP=1 -> P=0.8) and 0.5 (TP=5, FP=1 -> P=5/6).
    assert d.threshold == pytest.approx(0.5)
    assert d.validation_recall == pytest.approx(1.0)
    assert d.validation_precision == pytest.approx(5 / 6)
    assert d.rule.startswith("highest precision")
    predictions = apply_threshold(s, d.threshold)
    assert predictions.sum() == 6
    # With the 0.5-scored positive removed, 0.6 becomes the best feasible candidate.
    d2 = select_operating_threshold(y[:-1], s[:-1], minimum_recall=0.8)
    assert d2.threshold == pytest.approx(0.6)
    assert d2.validation_precision == pytest.approx(0.8)
    assert d2.validation_recall == pytest.approx(1.0)


def test_precision_ties_resolve_to_highest_threshold():
    # Any threshold in (0.3, 0.9] gives precision 1.0 with recall >= 0.5.
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.3, 0.8, 0.9])
    d = select_operating_threshold(y, s, minimum_recall=0.5)
    assert d.threshold == pytest.approx(0.9)
    assert d.validation_recall == pytest.approx(0.5)
    assert any("tied on precision" in note for note in d.notes)


def test_tied_scores_are_flagged_together_and_metrics_match_decision_rule():
    y = np.array([1, 1, 0, 0, 1, 0])
    s = np.array([0.7, 0.7, 0.7, 0.2, 0.9, 0.1])
    d = select_operating_threshold(y, s, minimum_recall=0.9)
    assert d.threshold == pytest.approx(0.7)
    predictions = apply_threshold(s, d.threshold)
    tp = int(((predictions == 1) & (y == 1)).sum())
    fp = int(((predictions == 1) & (y == 0)).sum())
    assert tp == 3 and fp == 1
    assert d.validation_precision == pytest.approx(tp / (tp + fp))
    assert d.validation_recall == pytest.approx(1.0)


def test_full_recall_target_is_always_feasible_at_lowest_positive_score():
    # The lowest candidate threshold equals the minimum positive score and
    # reaches recall 1.0, so the F1 fallback cannot trigger on valid inputs.
    y = np.array([1, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    d = select_operating_threshold(y, s, minimum_recall=1.0)
    assert d.threshold == pytest.approx(0.1)
    assert d.validation_recall == 1.0
    assert d.rule.startswith("highest precision")
    d2 = select_operating_threshold(y[:-1], s[:-1], minimum_recall=1.0)
    assert d2.threshold == pytest.approx(0.9)
    assert d2.validation_precision == 1.0


def test_constant_scores_yield_single_all_flag_threshold():
    y = np.array([0, 1, 0, 1])
    s = np.full(4, 0.42)
    d = select_operating_threshold(y, s, minimum_recall=0.8)
    assert d.threshold == pytest.approx(0.42)
    assert d.validation_recall == 1.0
    assert d.validation_precision == pytest.approx(0.5)
    assert apply_threshold(s, d.threshold).sum() == 4


@pytest.mark.parametrize(
    "y, s, message",
    [
        ([0, 1], [0.1, np.nan], "NaN or infinite"),
        ([0, 1], [0.1, np.inf], "NaN or infinite"),
        ([0, 0, 0], [0.1, 0.2, 0.3], "Both classes"),
        ([1, 1], [0.1, 0.2], "Both classes"),
        ([0, 1, 2], [0.1, 0.2, 0.3], "only 0 and 1"),
        ([0, 1], [0.1], "Length mismatch"),
        ([], [], "zero rows"),
    ],
)
def test_invalid_inputs_raise(y, s, message):
    with pytest.raises(ThresholdSelectionError, match=message):
        select_operating_threshold(np.array(y, dtype=float), np.array(s, dtype=float))


def test_invalid_minimum_recall_raises():
    with pytest.raises(ThresholdSelectionError):
        select_operating_threshold([0, 1], [0.1, 0.9], minimum_recall=0)
    with pytest.raises(ThresholdSelectionError):
        select_operating_threshold([0, 1], [0.1, 0.9], minimum_recall=1.5)


def test_apply_threshold_uses_greater_or_equal_and_rejects_bad_inputs():
    assert apply_threshold([0.5, 0.49, 0.51], 0.5).tolist() == [1, 0, 1]
    with pytest.raises(ThresholdSelectionError):
        apply_threshold([0.5], np.nan)
    with pytest.raises(ThresholdSelectionError):
        apply_threshold([np.nan], 0.5)


def test_decision_serialises_policy_metadata():
    d = select_operating_threshold([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]).to_dict()
    assert d["comparison"] == ">="
    assert d["minimum_recall"] == 0.8
    assert "tie_break" in d and isinstance(d["notes"], list)

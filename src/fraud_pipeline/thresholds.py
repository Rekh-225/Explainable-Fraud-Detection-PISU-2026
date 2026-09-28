"""Validation-only operating-threshold selection.

Policy (unchanged from the historical experiment):

1. Among all distinct score values, keep those whose validation recall is at
   least ``minimum_recall``.
2. Choose the one with the highest validation precision.
3. If several thresholds tie on precision, choose the *highest* threshold
   (fewest alerts).
4. If no threshold reaches the recall target, fall back to the threshold with
   maximum validation F1 (ties resolve to the lowest such threshold, i.e. the
   first candidate). The fallback is kept for parity with the original
   script, but with valid inputs the lowest candidate (the minimum positive
   score) always reaches recall 1.0, so it is effectively unreachable.

Defined behaviour at the edges:

* Predictions use ``score >= threshold``. ``precision_recall_curve`` produces
  one candidate per *distinct* score, so rows tied exactly at the threshold
  are all flagged and the reported validation precision/recall correspond
  exactly to that decision rule.
* Constant scores yield a single candidate equal to that constant; it flags
  every row (recall 1, precision = prevalence) and is reported with a note.
* Inputs with NaN/inf scores, non-binary labels, a single class, mismatched
  lengths or an empty set raise :class:`ThresholdSelectionError` because
  precision/recall are undefined or meaningless.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

COMPARISON = ">="
TIE_BREAK = "highest threshold among equal-precision candidates"


class ThresholdSelectionError(ValueError):
    """Raised for degenerate or invalid threshold-selection inputs."""


@dataclass(frozen=True)
class ThresholdDecision:
    threshold: float
    rule: str
    validation_precision: float
    validation_recall: float
    minimum_recall: float
    comparison: str = COMPARISON
    tie_break: str = TIE_BREAK
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_inputs(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(pd.Series(y_true).to_numpy())
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1:
        raise ThresholdSelectionError("y_true and scores must be one-dimensional")
    if len(y) != len(s):
        raise ThresholdSelectionError(f"Length mismatch: {len(y)} labels vs {len(s)} scores")
    if len(y) == 0:
        raise ThresholdSelectionError("Cannot select a threshold from zero rows")
    if not np.all(np.isfinite(s)):
        raise ThresholdSelectionError("Scores contain NaN or infinite values")
    try:
        y_float = y.astype(float)
    except (TypeError, ValueError) as exc:
        raise ThresholdSelectionError("Labels must be numeric 0/1") from exc
    if not np.all(np.isin(y_float, (0.0, 1.0))):
        raise ThresholdSelectionError("Labels must contain only 0 and 1")
    positives = int(y_float.sum())
    if positives == 0 or positives == len(y_float):
        raise ThresholdSelectionError(
            "Both classes must be present to select a threshold "
            f"(positives={positives}, rows={len(y_float)})"
        )
    return y_float.astype(int), s


def apply_threshold(scores, threshold: float) -> np.ndarray:
    """Binary decisions using ``score >= threshold``."""
    if not np.isfinite(threshold):
        raise ThresholdSelectionError("Threshold must be finite")
    s = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(s)):
        raise ThresholdSelectionError("Scores contain NaN or infinite values")
    return (s >= threshold).astype(int)


def select_operating_threshold(
    y_true,
    scores,
    minimum_recall: float = 0.80,
) -> ThresholdDecision:
    """Select the highest-precision validation threshold reaching target recall."""
    if not 0 < minimum_recall <= 1:
        raise ThresholdSelectionError("minimum_recall must lie in (0, 1]")
    y, s = _validate_inputs(y_true, scores)

    precision, recall, thresholds = precision_recall_curve(y, s)
    precision_at_threshold = precision[:-1]
    recall_at_threshold = recall[:-1]
    notes: list[str] = []
    if thresholds.size == 1:
        notes.append("all scores are identical; the single candidate threshold flags every row")

    feasible = np.flatnonzero(recall_at_threshold >= minimum_recall)
    if feasible.size:
        best_precision = precision_at_threshold[feasible].max()
        candidates = feasible[precision_at_threshold[feasible] == best_precision]
        if candidates.size > 1:
            notes.append(f"{candidates.size} candidates tied on precision; {TIE_BREAK}")
        index = int(candidates[np.argmax(thresholds[candidates])])
        rule = f"highest precision with validation recall >= {minimum_recall:.0%}"
    else:
        f1_values = np.divide(
            2 * precision_at_threshold * recall_at_threshold,
            precision_at_threshold + recall_at_threshold,
            out=np.zeros_like(precision_at_threshold),
            where=(precision_at_threshold + recall_at_threshold) > 0,
        )
        index = int(np.argmax(f1_values))
        rule = "maximum validation F1 because recall target was not reached"

    return ThresholdDecision(
        threshold=float(thresholds[index]),
        rule=rule,
        validation_precision=float(precision_at_threshold[index]),
        validation_recall=float(recall_at_threshold[index]),
        minimum_recall=minimum_recall,
        notes=notes,
    )

"""Compatibility entry point for the original analysis command.

``python src/run_analysis.py`` now runs the same experiment through the
``fraud_pipeline`` package (``fraud-pipeline reproduce-historical``):
Kaggle ``creditcard.csv``, exact-duplicate removal, stratified 64/16/20
split with seed 42, Dummy / Logistic Regression / Random Forest, validation-
only model and threshold selection, test-set evaluation.

Two behaviours changed deliberately:

* The dataset is no longer fetched from OpenML as a silent fallback. The
  OpenML mirror omits ``Time``, which the historical models use as a feature,
  so it is not equivalent. Pass ``--dataset`` / ``--source openml`` through
  ``fraud-pipeline run`` if you want that experiment explicitly.
* Outputs go to a fresh directory under ``output/runs/<run_id>/`` with a run
  manifest instead of overwriting the historical artifacts in
  ``output/analysis/``.

The original single-file script is preserved in git history
(commit 811654d, ``src/run_analysis.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fraud_pipeline.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["reproduce-historical", *sys.argv[1:]]))

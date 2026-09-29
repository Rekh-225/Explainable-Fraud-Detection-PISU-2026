"""Helpers to build small deterministic runs for UI tests (no network, no Kaggle file)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fraud_pipeline.modeling import ModelConfig
from fraud_pipeline.pipeline import RunConfig, RunResult, run_pipeline
from fraud_pipeline.synthetic import make_synthetic_frame

NONLINEAR_FEATURES = ("V4", "V10", "V12", "V14", "V17")


def linear_frame(seed: int, rows: int = 900):
    """Mean-shifted fraud signal: Logistic Regression wins validation AP."""
    return make_synthetic_frame(n_rows=rows, fraud_rate=0.05, seed=seed)


def nonlinear_frame(seed: int, rows: int = 900):
    """Symmetric +/- fraud signal with zero mean shift: Random Forest wins, LR cannot."""
    frame = make_synthetic_frame(n_rows=rows, fraud_rate=0.05, seed=seed)
    rng = np.random.default_rng(seed)
    fraud = frame["Class"] == 1
    n_fraud, n_legit = int(fraud.sum()), int((~fraud).sum())
    for column in NONLINEAR_FEATURES:
        frame.loc[fraud, column] = rng.choice([-3.5, 3.5], size=n_fraud) + rng.normal(0, 0.3, size=n_fraud)
    frame.loc[~fraud, list(NONLINEAR_FEATURES)] = rng.normal(0, 0.8, size=(n_legit, len(NONLINEAR_FEATURES)))
    return frame


def make_run(
    root: Path,
    run_id: str,
    *,
    source: str = "synthetic",
    seed: int = 1,
    winner: str = "Logistic Regression",
    figures: bool = False,
) -> RunResult:
    """Write a fixture CSV with the Kaggle column layout and run the pipeline on it.

    ``source="kaggle"`` only declares the *layout* (Time, V1-V28, Amount, Class);
    the rows are generated, not the ULB data. Tests use it to exercise
    Reproduced Experiment mode mechanics without weakening its rejection of
    ``source="synthetic"`` runs and without touching real-dataset provenance.
    """
    frame = nonlinear_frame(seed) if winner == "Random Forest" else linear_frame(seed)
    csv_path = root / "data" / f"{run_id}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    result = run_pipeline(
        RunConfig(
            dataset_path=csv_path,
            source=source,
            output_root=root / "runs",
            run_id=run_id,
            seed=seed,
            model=ModelConfig(seed=seed, rf_n_estimators=16, rf_max_depth=6),
            figures=figures,
        )
    )
    assert result.selected_model == winner, f"fixture produced {result.selected_model}, expected {winner}"
    return result

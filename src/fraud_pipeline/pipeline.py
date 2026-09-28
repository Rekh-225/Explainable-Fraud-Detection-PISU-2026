"""End-to-end orchestration of the historical experiment.

Order of operations (unchanged from the original script):
load + validate -> de-duplicate -> stratified split -> fit on train ->
score validation -> select model (validation AP) and thresholds (validation)
-> evaluate on test -> write artifacts -> write manifest.

Every run writes into its own directory under ``output_root``; historical
outputs under ``output/analysis`` are never touched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import reporting
from .artifacts import save_model_bundle
from .data import Dataset, DuplicatePolicy, Source, load_dataset
from .evaluation import evaluate_scores, ranking_metrics
from .manifest import build_manifest, save_json, write_manifest
from .modeling import (
    DUMMY_MODEL,
    FOREST_MODEL,
    LEARNED_MODELS,
    ModelConfig,
    build_models,
    describe_model,
    fit_model,
    positive_scores,
)
from .splitting import SplitResult, split_dataset
from .thresholds import COMPARISON, TIE_BREAK, ThresholdDecision, select_operating_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "runs"
HISTORICAL_KAGGLE_CSV = PROJECT_ROOT / "data" / "raw" / "creditcardfraud" / "creditcard.csv"
PROJECT_TITLE = (
    "Explainable AI-Powered Fraud Detection and Risk Analytics for Digital Payment Transactions"
)
DUMMY_THRESHOLD = 0.5
DEFAULT_POLICY = "default_0.5"
OPERATING_POLICY = "validation_operating_point"


@dataclass(frozen=True)
class RunConfig:
    dataset_path: Path
    source: Source
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_id: str | None = None
    seed: int = 42
    test_size: float = 0.20
    validation_size: float = 0.20
    minimum_recall: float = 0.80
    duplicate_policy: DuplicatePolicy = "drop_exact"
    model: ModelConfig = field(default_factory=ModelConfig)
    figures: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_path"] = str(self.dataset_path)
        payload["output_root"] = str(self.output_root)
        return payload


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    dataset: Dataset
    splits: SplitResult
    selected_model: str
    thresholds: dict[str, float]
    results: pd.DataFrame
    manifest: dict[str, Any]


def make_run_id(source: str, dataset_sha256: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{source}_{dataset_sha256[:8]}"


def prepare_run_dir(output_root: Path, run_id: str) -> Path:
    run_dir = output_root / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory {run_dir} already exists and is not empty; runs are never overwritten"
        )
    for sub in ("figures", "models"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def data_quality_summary(dataset: Dataset) -> dict[str, Any]:
    info = dataset.info
    return {
        "source": info.source,
        "source_description": info.source_description,
        "path": info.path,
        "sha256": info.sha256,
        "size_bytes": info.size_bytes,
        "features": info.features,
        "has_time": info.has_time,
        "time_column_note": (
            "Time (seconds since first transaction) is a model input, as in the Kaggle release."
            if info.has_time
            else "Time is absent (OpenML mirror layout); results are not comparable to the "
            "historical Kaggle experiment, which used Time as a feature."
        ),
        "ignored_columns": info.ignored_columns,
        "original_rows": info.raw_rows,
        "original_class_counts": info.raw_class_counts,
        "missing_values": info.missing_values,
        "duplicate_policy": info.duplicate_policy,
        "exact_duplicate_rows_removed": info.duplicates_removed,
        "cleaned_rows": info.rows,
        "cleaned_class_counts": info.class_counts,
        "cleaned_fraud_prevalence": info.fraud_prevalence,
    }


def run_pipeline(config: RunConfig) -> RunResult:
    dataset = load_dataset(config.dataset_path, config.source, config.duplicate_policy)
    run_id = config.run_id or make_run_id(config.source, dataset.info.sha256)
    run_dir = prepare_run_dir(Path(config.output_root), run_id)
    figure_dir = run_dir / "figures"

    save_json(run_dir / "data_quality.json", data_quality_summary(dataset))
    if config.figures:
        reporting.set_style()
        reporting.plot_class_imbalance(dataset.frame, figure_dir)
        reporting.plot_amount_distribution(dataset.frame, figure_dir)
        reporting.plot_fraud_rate_by_amount_band(dataset.frame, figure_dir)

    splits = split_dataset(
        dataset.X, dataset.y, config.seed, config.test_size, config.validation_size
    )
    split_summary = splits.summary()
    save_json(run_dir / "split_summary.json", split_summary)

    models = build_models(config.model)
    validation_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}
    validation_ap: dict[str, float] = {}
    decisions: dict[str, dict[str, Any]] = {}
    thresholds: dict[str, float] = {}

    for name, model in models.items():
        fit_model(model, splits.train.X, splits.train.y)
        validation_scores[name] = positive_scores(model, splits.validation.X)
        test_scores[name] = positive_scores(model, splits.test.X)
        validation_ap[name] = ranking_metrics(splits.validation.y, validation_scores[name])[
            "average_precision"
        ]
        if name == DUMMY_MODEL:
            thresholds[name] = DUMMY_THRESHOLD
            decisions[name] = {
                "threshold": DUMMY_THRESHOLD,
                "rule": "fixed non-learning reference threshold",
            }
        else:
            decision: ThresholdDecision = select_operating_threshold(
                splits.validation.y, validation_scores[name], config.minimum_recall
            )
            thresholds[name] = decision.threshold
            decisions[name] = decision.to_dict()

    selected = max(LEARNED_MODELS, key=lambda name: validation_ap[name])

    rows: list[dict[str, Any]] = []
    for name in models:
        rows.append(evaluate_scores(name, DEFAULT_POLICY, splits.test.y, test_scores[name], 0.5))
        if name != DUMMY_MODEL:
            rows.append(
                evaluate_scores(name, OPERATING_POLICY, splits.test.y, test_scores[name], thresholds[name])
            )
    results = pd.DataFrame(rows)
    results.to_csv(run_dir / "model_comparison.csv", index=False)

    selection = {
        "validation_average_precision": validation_ap,
        "threshold_selection": decisions,
        "selected_model": selected,
        "selection_rule": "highest validation Average Precision among learned models",
    }
    save_json(run_dir / "validation_summary.json", selection)

    reporting.export_validation_scores(
        run_dir,
        splits.validation.row_ids,
        splits.validation.y,
        {name: validation_scores[name] for name in LEARNED_MODELS},
        {name: thresholds[name] for name in LEARNED_MODELS},
    )

    importance = reporting.feature_importance_table(models[FOREST_MODEL], dataset.features)
    importance.to_csv(run_dir / "feature_importance.csv", index=False)
    if config.figures:
        learned_scores = {name: test_scores[name] for name in LEARNED_MODELS}
        reporting.plot_precision_recall_curves(splits.test.y, learned_scores, figure_dir)
        reporting.plot_confusion_matrices(
            splits.test.y, learned_scores, {n: thresholds[n] for n in LEARNED_MODELS}, figure_dir
        )
        reporting.plot_feature_importance(importance, figure_dir)

    selected_row = results[
        (results["model"] == selected) & (results["policy"] == OPERATING_POLICY)
    ].iloc[0].to_dict()
    summary = {
        "project_title": PROJECT_TITLE,
        "run_id": run_id,
        "data_quality": data_quality_summary(dataset),
        "splits": split_summary,
        "selected_model": selected,
        "selected_threshold": thresholds[selected],
        "selected_model_test_metrics": selected_row,
        "top_random_forest_features": importance.head(10).to_dict("records"),
        "model_selection_basis": "validation Average Precision",
        "threshold_policy": decisions[selected],
        "limitations": [
            "The transactions cover only two days in September 2013.",
            "V1-V28 are anonymized PCA components without business-semantic labels.",
            (
                "Time records elapsed seconds from the first transaction, not calendar timestamps or local time."
                if dataset.info.has_time
                else "The OpenML mirror omits the original Time field."
            ),
            "A stratified random split does not simulate future fraud drift.",
            "The dataset contains no reliable investigation-cost or fraud-loss matrix.",
            "This is an offline academic prototype, not a production payment control.",
        ],
    }
    save_json(run_dir / "analysis_summary.json", summary)

    save_model_bundle(run_dir, models[selected], selected, thresholds[selected], dataset.features, run_id)

    manifest = build_manifest(
        run_id=run_id,
        run_dir=run_dir,
        repo_root=PROJECT_ROOT,
        dataset={
            "path": dataset.info.path,
            "source": dataset.info.source,
            "source_description": dataset.info.source_description,
            "sha256": dataset.info.sha256,
            "size_bytes": dataset.info.size_bytes,
            "raw_rows": dataset.info.raw_rows,
            "raw_class_counts": dataset.info.raw_class_counts,
            "rows_after_duplicate_policy": dataset.info.rows,
            "class_counts_after_duplicate_policy": dataset.info.class_counts,
            "has_time": dataset.info.has_time,
            "ignored_columns": dataset.info.ignored_columns,
        },
        features=dataset.features,
        duplicate_policy={
            "policy": dataset.info.duplicate_policy,
            "definition": "exact match on all retained feature columns and Class; first occurrence kept",
            "rows_removed": dataset.info.duplicates_removed,
        },
        split=split_summary,
        models={name: describe_model(model) for name, model in models.items()},
        threshold_policy={
            "minimum_recall": config.minimum_recall,
            "comparison": COMPARISON,
            "tie_break": TIE_BREAK,
            "fallback": "maximum validation F1 when the recall target is unreachable",
            "dummy_threshold": DUMMY_THRESHOLD,
            "decisions": decisions,
        },
        selection={
            "rule": selection["selection_rule"],
            "selected_model": selected,
            "validation_average_precision": validation_ap,
        },
        config=config.to_dict(),
    )
    write_manifest(run_dir, manifest)

    return RunResult(
        run_id=run_id,
        run_dir=run_dir,
        dataset=dataset,
        splits=splits,
        selected_model=selected,
        thresholds=thresholds,
        results=results,
        manifest=manifest,
    )

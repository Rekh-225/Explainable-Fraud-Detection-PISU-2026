"""Loaders and calculations behind the Streamlit interface (no Streamlit here).

Three modes, none of which falls back to another:

* ``load_historical`` reads the committed July-2026 artifacts in
  ``output/analysis`` and lists what provenance they lack.
* ``ensure_synthetic_demo_run`` creates (once) or reuses a small run on a
  deterministic synthetic fixture.
* ``load_run`` opens a run directory written by ``fraud_pipeline`` after
  verifying its manifest, artifact checksums and version compatibility.

All interactive calculations operate on *validation* predictions. Test-set
results are read from ``model_comparison.csv`` and never recomputed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .. import __version__
from ..artifacts import ArtifactIntegrityError, ModelBundle, load_model_bundle
from ..data import LABEL_COLUMN, sha256_file
from ..evaluation import evaluate_scores
from ..manifest import MANIFEST_NAME, load_manifest, verify_run
from ..modeling import LEARNED_MODELS, ModelConfig
from ..pipeline import PROJECT_ROOT, RunConfig, run_pipeline
from ..reporting import VALIDATION_SCORES_FILENAME
from ..synthetic import write_synthetic_csv

HISTORICAL_DIR = PROJECT_ROOT / "output" / "analysis"
RUNS_ROOT = PROJECT_ROOT / "output" / "runs"
DEMO_ROOT = PROJECT_ROOT / "output" / "ui_demo"

MAX_SCORE_ROWS = 1_000_000
MAX_DATASET_BYTES = 600 * 1024 * 1024
MAX_QUEUE_ROWS = 500
COMPATIBLE_MANIFEST_VERSIONS = {1}
OPERATING_POLICY = "validation_operating_point"

REQUIRED_RUN_FILES = (
    "analysis_summary.json",
    "data_quality.json",
    "model_comparison.csv",
    "split_summary.json",
    "validation_summary.json",
    "feature_importance.csv",
    VALIDATION_SCORES_FILENAME,
)
HISTORICAL_FILES = (
    "analysis_summary.json",
    "data_quality.json",
    "model_comparison.csv",
    "split_summary.json",
    "validation_summary.json",
    "feature_importance.csv",
)
FIGURE_NAMES = {
    "class_imbalance": "01_class_imbalance.png",
    "amount_distribution": "02_amount_distribution.png",
    "fraud_rate_by_amount_band": "03_fraud_rate_by_amount_band.png",
    "precision_recall_curves": "04_precision_recall_curves.png",
    "confusion_matrices": "05_confusion_matrices.png",
    "feature_importance": "06_random_forest_feature_importance.png",
}
ANNOTATION_CHOICES = ("unreviewed", "confirmed_fraud", "legitimate", "escalate")

SCORE_DISCLAIMER = (
    "Scores are uncalibrated model outputs, not probabilities of fraud; compare them only "
    "against thresholds selected on validation data."
)
FEATURE_DISCLAIMER = (
    "V1-V28 are anonymized PCA components of the ULB dataset. The transformation that "
    "produced them is not public, so this model cannot score arbitrary bank statements or "
    "raw transaction exports."
)
IMPORTANCE_DISCLAIMER = (
    "Impurity-based importance describes what the Random Forest relies on globally. It is "
    "not a causal explanation of any individual transaction."
)


class UILoadError(RuntimeError):
    """Raised when a mode cannot be entered; the UI shows it instead of falling back."""


# --------------------------------------------------------------------------- helpers


def model_slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def score_column(name: str) -> str:
    return f"score__{model_slug(name)}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _figures(directory: Path) -> dict[str, Path]:
    figure_dir = directory / "figures"
    return {key: figure_dir / name for key, name in FIGURE_NAMES.items() if (figure_dir / name).is_file()}


# --------------------------------------------------------------------------- historical


@dataclass
class HistoricalResults:
    directory: Path
    analysis_summary: dict[str, Any]
    validation_summary: dict[str, Any]
    data_quality: dict[str, Any]
    split_summary: dict[str, Any]
    model_comparison: pd.DataFrame
    feature_importance: pd.DataFrame
    figures: dict[str, Path]
    missing_files: list[str]
    provenance_gaps: list[str]

    @property
    def selected_model(self) -> str:
        return self.analysis_summary["selected_model"]

    @property
    def selected_threshold(self) -> float:
        return float(self.analysis_summary["selected_threshold"])


def load_historical(directory: Path = HISTORICAL_DIR) -> HistoricalResults:
    """Load the committed artifacts and enumerate the provenance they lack."""
    directory = Path(directory)
    if not directory.is_dir():
        raise UILoadError(f"Historical results directory not found: {directory}")
    missing = [name for name in HISTORICAL_FILES if not (directory / name).is_file()]
    if missing:
        raise UILoadError(f"Historical results in {directory} are incomplete; missing {missing}")

    gaps = []
    if not (directory / MANIFEST_NAME).is_file():
        gaps.append("No run manifest: dataset checksum, split membership fingerprints, exact model "
                    "parameters, Python/package versions and code commit were not recorded.")
    if not (directory / VALIDATION_SCORES_FILENAME).is_file():
        gaps.append("No row-level validation predictions: the threshold explorer and review queue "
                    "are unavailable in this mode.")
    model_path = directory / "models" / "selected_model.joblib"
    if model_path.is_file():
        gaps.append("A serialized model exists locally but has no schema sidecar or manifest "
                    "checksum, so it is not loaded.")
    else:
        gaps.append("The serialized model is not committed (git-ignored).")
    gaps.append("Figures are committed PNG files; their generating run cannot be re-verified.")

    return HistoricalResults(
        directory=directory,
        analysis_summary=_read_json(directory / "analysis_summary.json"),
        validation_summary=_read_json(directory / "validation_summary.json"),
        data_quality=_read_json(directory / "data_quality.json"),
        split_summary=_read_json(directory / "split_summary.json"),
        model_comparison=pd.read_csv(directory / "model_comparison.csv", float_precision="round_trip"),
        feature_importance=pd.read_csv(directory / "feature_importance.csv"),
        figures=_figures(directory),
        missing_files=missing,
        provenance_gaps=gaps,
    )


# --------------------------------------------------------------------------- runs


@dataclass
class LoadedRun:
    run_dir: Path
    manifest: dict[str, Any]
    verification: dict[str, list[str]]
    compatibility_notes: list[str]
    analysis_summary: dict[str, Any]
    validation_summary: dict[str, Any]
    data_quality: dict[str, Any]
    split_summary: dict[str, Any]
    model_comparison: pd.DataFrame
    feature_importance: pd.DataFrame
    validation_scores: pd.DataFrame
    figures: dict[str, Path]

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    @property
    def source(self) -> str:
        return self.manifest["dataset"]["source"]

    @property
    def is_synthetic(self) -> bool:
        return self.source == "synthetic"

    @property
    def selected_model(self) -> str:
        return self.manifest["selection"]["selected_model"]

    @property
    def selected_threshold(self) -> float:
        return float(self.manifest["threshold_policy"]["decisions"][self.selected_model]["threshold"])

    @property
    def scored_models(self) -> list[str]:
        return [m for m in LEARNED_MODELS if score_column(m) in self.validation_scores.columns]

    def threshold_for(self, model: str) -> float:
        return float(self.manifest["threshold_policy"]["decisions"][model]["threshold"])

    def dataset_path(self) -> Path:
        path = Path(self.manifest["dataset"]["path"])
        return path if path.is_absolute() else PROJECT_ROOT / path


def list_runs(runs_root: Path = RUNS_ROOT) -> list[Path]:
    root = Path(runs_root)
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if (p / MANIFEST_NAME).is_file()), reverse=True)


def _compatibility_notes(manifest: dict[str, Any]) -> list[str]:
    notes = []
    version = manifest.get("manifest_version")
    if version not in COMPATIBLE_MANIFEST_VERSIONS:
        raise UILoadError(
            f"Manifest version {version!r} is not supported by this UI (supported: "
            f"{sorted(COMPATIBLE_MANIFEST_VERSIONS)})."
        )
    run_version = str(manifest.get("pipeline_version", ""))
    if run_version.split(".")[:2] != __version__.split(".")[:2]:
        notes.append(f"Run was produced by pipeline {run_version}; this UI ships with {__version__}.")
    packages = manifest.get("environment", {}).get("packages", {})
    try:
        from importlib import metadata

        for name in ("scikit-learn", "numpy", "pandas"):
            installed = metadata.version(name)
            if packages.get(name) and packages[name] != installed:
                notes.append(f"{name} {packages[name]} at run time vs {installed} installed now.")
    except Exception:  # pragma: no cover - metadata lookup is best effort
        pass
    return notes


def load_run(run_dir: Path) -> LoadedRun:
    """Open a run directory after verifying manifest, checksums and file layout."""
    run_dir = Path(run_dir)
    if not (run_dir / MANIFEST_NAME).is_file():
        raise UILoadError(f"{run_dir} has no {MANIFEST_NAME}; only runs written by fraud_pipeline can be opened.")
    manifest = load_manifest(run_dir)
    notes = _compatibility_notes(manifest)

    verification = verify_run(run_dir)
    problems = {k: v for k, v in verification.items() if v and k != "unexpected"}
    if problems:
        raise UILoadError(f"Artifact verification failed for {run_dir.name}: {problems}")
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise UILoadError(f"Run {run_dir.name} is missing required files: {missing}")

    # round_trip parsing keeps scores bit-identical to the run, so rows tied
    # exactly at a stored threshold are flagged the same way the pipeline did.
    scores = pd.read_csv(run_dir / VALIDATION_SCORES_FILENAME, float_precision="round_trip")
    if len(scores) > MAX_SCORE_ROWS:
        raise UILoadError(f"validation_scores.csv has {len(scores):,} rows; the UI caps input at {MAX_SCORE_ROWS:,}.")
    required_columns = {"row_id", "split", "label"}
    if not required_columns <= set(scores.columns):
        raise UILoadError(f"validation_scores.csv lacks columns {sorted(required_columns - set(scores.columns))}")
    if not (scores["split"] == "validation").all():
        raise UILoadError("validation_scores.csv contains rows from a split other than validation.")
    if not scores["label"].isin([0, 1]).all():
        raise UILoadError("validation_scores.csv contains labels other than 0/1.")
    if scores["row_id"].duplicated().any():
        raise UILoadError("validation_scores.csv contains duplicate row ids.")
    if not any(score_column(m) in scores.columns for m in LEARNED_MODELS):
        raise UILoadError("validation_scores.csv contains no model score columns.")

    return LoadedRun(
        run_dir=run_dir,
        manifest=manifest,
        verification=verification,
        compatibility_notes=notes,
        analysis_summary=_read_json(run_dir / "analysis_summary.json"),
        validation_summary=_read_json(run_dir / "validation_summary.json"),
        data_quality=_read_json(run_dir / "data_quality.json"),
        split_summary=_read_json(run_dir / "split_summary.json"),
        model_comparison=pd.read_csv(run_dir / "model_comparison.csv", float_precision="round_trip"),
        feature_importance=pd.read_csv(run_dir / "feature_importance.csv"),
        validation_scores=scores,
        figures=_figures(run_dir),
    )


def load_reproduced_run(run_dir: Path) -> LoadedRun:
    """Reproduced-experiment mode: a verified run on real (non-synthetic) data."""
    run = load_run(run_dir)
    if run.is_synthetic:
        raise UILoadError(
            f"Run {run.run_id} was produced on synthetic data. Open it in Synthetic Demo mode; "
            "Reproduced Experiment mode only accepts kaggle/openml runs."
        )
    return run


def load_run_bundle(run: LoadedRun) -> ModelBundle:
    """Checksum-verified model bundle from the run directory (never a user-supplied path)."""
    try:
        return load_model_bundle(run.run_dir)
    except ArtifactIntegrityError as exc:
        raise UILoadError(str(exc)) from exc


def load_dataset_rows(run: LoadedRun, row_ids: list[int]) -> pd.DataFrame:
    """Feature rows for the given row ids from the run's source CSV, checksum-verified."""
    path = run.dataset_path()
    if not path.is_file():
        raise UILoadError(f"Source dataset not found at {path}; transaction features cannot be shown.")
    size = path.stat().st_size
    if size > MAX_DATASET_BYTES:
        raise UILoadError(f"Dataset is {size / 1e6:.0f} MB, above the UI cap of {MAX_DATASET_BYTES / 1e6:.0f} MB.")
    if sha256_file(path) != run.manifest["dataset"]["sha256"]:
        raise UILoadError("Source dataset checksum differs from the manifest; refusing to display its rows.")
    frame = pd.read_csv(path)
    frame.index = pd.RangeIndex(len(frame), name="row_id")
    wanted = [i for i in row_ids if 0 <= i < len(frame)]
    return frame.loc[wanted, run.manifest["features"]]


# --------------------------------------------------------------------------- synthetic demo


def synthetic_demo_paths(root: Path, rows: int, seed: int) -> tuple[Path, Path]:
    stem = f"synthetic_demo_rows{rows}_seed{seed}"
    return root / "data" / f"{stem}.csv", root / "runs" / stem


def ensure_synthetic_demo_run(
    root: Path = DEMO_ROOT,
    rows: int = 6000,
    seed: int = 2026,
    fraud_rate: float = 0.02,
    n_duplicates: int = 40,
) -> tuple[Path, bool]:
    """Create the demo run once and reuse it afterwards. Returns (run_dir, created)."""
    if not 500 <= rows <= 50_000:
        raise ValueError("rows must be between 500 and 50,000 for the local demo")
    csv_path, run_dir = synthetic_demo_paths(Path(root), rows, seed)
    if (run_dir / MANIFEST_NAME).is_file():
        return run_dir, False
    write_synthetic_csv(csv_path, n_rows=rows, fraud_rate=fraud_rate, seed=seed, n_duplicates=n_duplicates)
    run_pipeline(
        RunConfig(
            dataset_path=csv_path,
            source="synthetic",
            output_root=run_dir.parent,
            run_id=run_dir.name,
            seed=seed,
            model=ModelConfig(seed=seed, rf_n_estimators=60, rf_max_depth=8),
            figures=True,
        )
    )
    return run_dir, True


# --------------------------------------------------------------------------- threshold explorer


@dataclass(frozen=True)
class ThresholdSummary:
    model: str
    threshold: float
    rows: int
    fraud_total: int
    alerts: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    alert_rate: float
    false_positives_per_1000_legitimate: float

    @property
    def missed_fraud(self) -> int:
        return self.false_negatives

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["missed_fraud"] = self.missed_fraud
        return payload


def threshold_summary(model: str, labels, scores, threshold: float) -> ThresholdSummary:
    """Validation metrics at ``threshold`` using the core evaluation arithmetic."""
    metrics = evaluate_scores(model, "explorer", pd.Series(np.asarray(labels)), np.asarray(scores, dtype=float), threshold)
    return ThresholdSummary(
        model=model,
        threshold=float(threshold),
        rows=metrics["test_rows"],
        fraud_total=metrics["test_fraud_cases"],
        alerts=metrics["true_positives"] + metrics["false_positives"],
        true_positives=metrics["true_positives"],
        false_positives=metrics["false_positives"],
        false_negatives=metrics["false_negatives"],
        true_negatives=metrics["true_negatives"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        alert_rate=metrics["alert_rate"],
        false_positives_per_1000_legitimate=metrics["false_positives_per_1000_legitimate"],
    )


def threshold_curve(labels, scores) -> pd.DataFrame:
    """Precision/recall/alerts for every distinct candidate threshold (validation)."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y, s)
    ordered = np.sort(s)
    alerts = len(s) - np.searchsorted(ordered, thresholds, side="left")
    return pd.DataFrame(
        {"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1], "alerts": alerts}
    )


def threshold_for_capacity(scores, capacity: int) -> float:
    """Highest threshold whose alert count (``score >= t``) is at least ``capacity``.

    With tied scores the realised alert count can exceed the capacity; callers
    should display the realised count from :func:`threshold_summary`.
    """
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    if capacity < 1:
        raise ValueError("capacity must be at least 1")
    return float(s[min(capacity, len(s)) - 1])


def capacity_scenarios(model: str, labels, scores, capacities: list[int]) -> pd.DataFrame:
    rows = []
    for capacity in capacities:
        t = threshold_for_capacity(scores, capacity)
        summary = threshold_summary(model, labels, scores, t)
        rows.append({"review_capacity": int(capacity), **summary.to_dict()})
    return pd.DataFrame(rows)


def parse_capacities(text: str, maximum: int) -> list[int]:
    values = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit() or int(token) < 1:
            raise ValueError(f"Capacities must be positive integers; got {token!r}")
        values.append(min(int(token), maximum))
    return sorted(set(values))


# --------------------------------------------------------------------------- cost scenario


@dataclass(frozen=True)
class CostInputs:
    review_cost_per_alert: float
    loss_per_missed_fraud: float
    recovered_per_caught_fraud: float
    currency: str = "units"

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name == "currency":
                continue
            if value is None or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be an explicit non-negative number")


def cost_scenario(summary: ThresholdSummary, inputs: CostInputs) -> dict[str, Any]:
    """Hypothetical scenario arithmetic. Not realized savings."""
    inputs.validate()
    review = summary.alerts * inputs.review_cost_per_alert
    missed = summary.false_negatives * inputs.loss_per_missed_fraud
    recovered = summary.true_positives * inputs.recovered_per_caught_fraud
    return {
        "kind": "hypothetical_scenario",
        "disclaimer": "Scenario arithmetic from user-supplied hypothetical unit costs on validation counts; "
        "not realized or projected savings.",
        "currency": inputs.currency,
        "inputs": asdict(inputs),
        "alerts_reviewed": summary.alerts,
        "review_cost_total": review,
        "missed_fraud_cases": summary.false_negatives,
        "missed_fraud_loss_total": missed,
        "caught_fraud_cases": summary.true_positives,
        "recovered_total": recovered,
        "net_scenario_value": recovered - review - missed,
    }


# --------------------------------------------------------------------------- review queue


GROUND_TRUTH_COLUMN = "ground_truth_label"


def build_review_queue(run: LoadedRun, model: str, threshold: float, limit: int = 50) -> pd.DataFrame:
    """Validation rows flagged at ``threshold``, highest score first.

    The dataset label is exposed under ``ground_truth_label`` so it is never
    confused with analyst annotations, which live in :class:`AnnotationStore`.
    """
    limit = max(1, min(int(limit), MAX_QUEUE_ROWS))
    column = score_column(model)
    if column not in run.validation_scores.columns:
        raise UILoadError(f"No validation scores for model {model!r} in this run.")
    frame = run.validation_scores[["row_id", "split", "label", column]].rename(
        columns={column: "model_score", "label": GROUND_TRUTH_COLUMN}
    )
    frame = frame[frame["model_score"] >= threshold].sort_values(
        ["model_score", "row_id"], ascending=[False, True]
    )
    frame.insert(1, "queue_rank", range(1, len(frame) + 1))
    frame["model"] = model
    return frame.head(limit).reset_index(drop=True)


@dataclass
class AnnotationStore:
    """Analyst annotations keyed by row id, kept apart from dataset labels."""

    records: dict[int, dict[str, Any]] = field(default_factory=dict)

    def annotate(self, row_id: int, annotation: str, note: str = "", when: datetime | None = None) -> None:
        if annotation not in ANNOTATION_CHOICES:
            raise ValueError(f"annotation must be one of {ANNOTATION_CHOICES}")
        if annotation == "unreviewed":
            self.records.pop(int(row_id), None)
            return
        self.records[int(row_id)] = {
            "annotation": annotation,
            "note": note.strip()[:500],
            "annotated_at_utc": (when or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        }

    def get(self, row_id: int) -> str:
        return self.records.get(int(row_id), {}).get("annotation", "unreviewed")

    def frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=["row_id", "annotation", "note", "annotated_at_utc"])
        return pd.DataFrame(
            [{"row_id": rid, **rec} for rid, rec in sorted(self.records.items())]
        )

    def summary(self) -> dict[str, int]:
        counts = {choice: 0 for choice in ANNOTATION_CHOICES if choice != "unreviewed"}
        for rec in self.records.values():
            counts[rec["annotation"]] += 1
        return counts


def queue_with_annotations(queue: pd.DataFrame, store: AnnotationStore) -> pd.DataFrame:
    frame = queue.copy()
    frame["analyst_annotation"] = frame["row_id"].map(store.get)
    return frame


def annotation_agreement(queue: pd.DataFrame, store: AnnotationStore) -> dict[str, int]:
    """Compare analyst annotations with dataset labels for the rows in the queue."""
    merged = queue_with_annotations(queue, store)
    reviewed = merged[merged["analyst_annotation"].isin(["confirmed_fraud", "legitimate"])]
    as_label = reviewed["analyst_annotation"].eq("confirmed_fraud").astype(int)
    return {
        "reviewed": int(len(reviewed)),
        "agree_with_ground_truth": int((as_label == reviewed[GROUND_TRUTH_COLUMN]).sum()),
        "disagree_with_ground_truth": int((as_label != reviewed[GROUND_TRUTH_COLUMN]).sum()),
        "escalated": int(merged["analyst_annotation"].eq("escalate").sum()),
    }


# --------------------------------------------------------------------------- local explanation


def linear_contributions(bundle: ModelBundle, row: pd.Series) -> pd.DataFrame:
    """Per-feature additive terms of a scaled logistic regression for one row.

    Contribution_i = coef_i * (x_i - mean_i) / scale_i (log-odds units). Only
    the Logistic Regression pipeline supports this; other models raise.
    """
    steps = getattr(bundle.model, "named_steps", None)
    if not steps or "scaler" not in steps or not hasattr(steps.get("model"), "coef_"):
        raise NotImplementedError(
            f"Local additive explanations are only available for the Logistic Regression pipeline, "
            f"not {bundle.model_name}."
        )
    scaler, model = steps["scaler"], steps["model"]
    x = row[bundle.features].to_numpy(dtype=float)
    z = (x - scaler.mean_) / scaler.scale_
    contributions = model.coef_[0] * z
    frame = pd.DataFrame(
        {
            "feature": bundle.features,
            "description": [
                "anonymized PCA component" if f.startswith("V") else f.lower() for f in bundle.features
            ],
            "value": x,
            "standardized_value": z,
            "log_odds_contribution": contributions,
        }
    )
    frame.attrs["intercept"] = float(model.intercept_[0])
    frame.attrs["log_odds"] = float(model.intercept_[0] + contributions.sum())
    return frame.reindex(frame["log_odds_contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# --------------------------------------------------------------------------- exports


def markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.{float_digits}f}"
        return str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def export_config(
    mode: str,
    run: LoadedRun | None,
    model: str | None,
    threshold: float | None,
    capacities: list[int],
    cost_inputs: CostInputs | None,
    annotations: AnnotationStore | None,
) -> dict[str, Any]:
    return {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ui_version": __version__,
        "mode": mode,
        "run_id": run.run_id if run else None,
        "run_dir": str(run.run_dir) if run else None,
        "dataset_source": run.source if run else None,
        "dataset_sha256": run.manifest["dataset"]["sha256"] if run else None,
        "synthetic": run.is_synthetic if run else None,
        "explored_model": model,
        "explored_validation_threshold": threshold,
        "validation_selected_threshold": run.threshold_for(model) if run and model else None,
        "review_capacities": capacities,
        "cost_scenario_inputs": asdict(cost_inputs) if cost_inputs else None,
        "annotation_counts": annotations.summary() if annotations else None,
        "note": "Explored thresholds are validation-only explorations; they do not re-validate the model "
        "or alter the frozen test result.",
    }


def render_report(
    mode: str,
    title_source: str,
    data_quality: dict[str, Any],
    model_comparison: pd.DataFrame,
    selected_model: str,
    selected_threshold: float,
    explorer: ThresholdSummary | None = None,
    capacity_table: pd.DataFrame | None = None,
    cost: dict[str, Any] | None = None,
    annotations: AnnotationStore | None = None,
    manifest: dict[str, Any] | None = None,
    provenance_gaps: list[str] | None = None,
    limitations: list[str] | None = None,
) -> str:
    """Markdown analytical report for the current view."""
    synthetic = data_quality.get("source") == "synthetic"
    lines = [f"# Fraud-detection analytical report - {mode}", ""]
    if synthetic:
        lines += ["> **SYNTHETIC DATA.** Every number in this report comes from a generated fixture. "
                  "None of it describes the historical experiment or real transactions.", ""]
    lines += [f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC by fraud_pipeline UI {__version__}.", ""]
    lines += ["## Data", "", f"- Source: {title_source}"]
    for key in ("sha256", "original_rows", "exact_duplicate_rows_removed", "cleaned_rows", "cleaned_class_counts", "cleaned_fraud_prevalence"):
        if key in data_quality:
            lines.append(f"- {key}: {data_quality[key]}")
    lines += ["", "## Test-set results (frozen)", "",
              "Computed once at the validation-selected threshold; not recomputed by the UI.", ""]
    columns = [c for c in ("model", "policy", "threshold", "precision", "recall", "f1", "average_precision", "roc_auc", "true_positives", "false_positives", "false_negatives") if c in model_comparison.columns]
    lines.append(markdown_table(model_comparison[columns]))
    lines += ["", f"Selected model: **{selected_model}** at validation-selected threshold **{selected_threshold:.6f}**.", ""]
    if explorer:
        lines += ["## Validation threshold exploration", "",
                  f"Model {explorer.model} at threshold {explorer.threshold:.6f} on {explorer.rows:,} validation rows "
                  f"({explorer.fraud_total} fraud): {explorer.alerts} alerts, precision {explorer.precision:.3f}, "
                  f"recall {explorer.recall:.3f}, {explorer.false_positives} false positives, "
                  f"{explorer.missed_fraud} missed fraud cases.",
                  "", "This is an exploration on validation predictions only. It is not a newly validated operating "
                  "point and the frozen test result above was not re-evaluated at this threshold.", ""]
    if capacity_table is not None and len(capacity_table):
        lines += ["## Review-capacity scenarios (validation)", "",
                  markdown_table(capacity_table[["review_capacity", "threshold", "alerts", "precision", "recall", "false_positives", "missed_fraud"]]), ""]
    if cost:
        lines += ["## Cost scenario (hypothetical)", "", f"> {cost['disclaimer']}", "",
                  f"- Inputs ({cost['currency']}): {cost['inputs']}",
                  f"- Review cost total: {cost['review_cost_total']:.2f}",
                  f"- Missed-fraud loss total: {cost['missed_fraud_loss_total']:.2f}",
                  f"- Recovered total: {cost['recovered_total']:.2f}",
                  f"- Net scenario value: {cost['net_scenario_value']:.2f}", ""]
    if annotations and annotations.records:
        lines += ["## Analyst annotations", "", f"Counts: {annotations.summary()}",
                  "Annotations are stored separately from dataset labels and do not retrain the model.", ""]
    if manifest:
        lines += ["## Provenance", "", f"- run_id: {manifest['run_id']}",
                  f"- pipeline_version: {manifest['pipeline_version']}",
                  f"- dataset sha256: {manifest['dataset']['sha256']}",
                  f"- code commit: {manifest['code'].get('commit')}",
                  f"- environment: Python {manifest['environment']['python']}, {manifest['environment']['packages']}", ""]
    if provenance_gaps:
        lines += ["## Provenance gaps", ""] + [f"- {gap}" for gap in provenance_gaps] + [""]
    lines += ["## Limitations", ""] + [f"- {item}" for item in (limitations or [])]
    lines += [f"- {SCORE_DISCLAIMER}", f"- {FEATURE_DISCLAIMER}", f"- {IMPORTANCE_DISCLAIMER}", ""]
    return "\n".join(lines)

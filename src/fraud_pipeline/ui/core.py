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

import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .. import __version__
from ..artifacts import (
    LINEAR_SPEC_FILENAME,
    MAX_LINEAR_SPEC_BYTES,
    MODEL_DIRNAME,
    ArtifactIntegrityError,
    LinearModelSpec,
    parse_linear_spec,
)
from ..data import LABEL_COLUMN, REQUIRED_FEATURES, TIME_COLUMN, sha256_file
from ..evaluation import evaluate_scores
from ..manifest import MANIFEST_NAME, ManifestError, artifact_checksums, load_manifest, verify_run
from ..modeling import LEARNED_MODELS, LOGISTIC_MODEL, ModelConfig
from ..pipeline import REQUIRED_RUN_FILES, RunConfig, run_pipeline
from ..reporting import VALIDATION_SCORES_FILENAME
from ..synthetic import write_synthetic_csv
from ..workspace import historical_results_dir, default_output_root, ui_demo_root, workspace_root

# Defaults resolved from the workspace (explicit FRAUD_PIPELINE_ROOT or the current
# directory at import time), never from the installed package location. The app
# additionally honours FRAUD_UI_* overrides.
HISTORICAL_DIR = historical_results_dir()
RUNS_ROOT = default_output_root()
DEMO_ROOT = ui_demo_root()

MAX_SCORE_ROWS = 1_000_000
MAX_SCORES_BYTES = 256 * 1024 * 1024
MAX_TABLE_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_DATASET_BYTES = 600 * 1024 * 1024
MAX_QUEUE_ROWS = 500
COMPATIBLE_MANIFEST_VERSIONS = {1}
OPERATING_POLICY = "validation_operating_point"

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


def _committed_figures(directory: Path) -> dict[str, Path]:
    """Figures present on disk (historical mode only; no manifest exists to verify them)."""
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
        model_comparison=_read_csv(directory / "model_comparison.csv", MAX_TABLE_BYTES),
        feature_importance=_read_csv(directory / "feature_importance.csv", MAX_TABLE_BYTES),
        figures=_committed_figures(directory),
        missing_files=missing,
        provenance_gaps=gaps,
    )


# --------------------------------------------------------------------------- runs


@dataclass
class LoadedRun:
    """A verified snapshot of a run directory.

    Every consumed file was read once, its bytes hashed and compared with the
    digest recorded in ``run_manifest.json``, and then parsed *from those same
    bytes*; ``content_key`` fingerprints the whole directory state at that
    moment so callers can detect later modifications (see
    :func:`run_content_key`). Files displayed later (figures, the linear spec)
    go through :func:`read_verified_artifact`, which re-reads and re-hashes.

    Integrity vs. authenticity: all of this proves the directory is internally
    consistent with its own manifest. It does not prove who produced it or
    that the declared dataset source is genuine; see :func:`classify_provenance`.
    """

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
    unverified_extra_files: list[str]
    linear_spec_path: Path | None
    content_key: str
    consistency_checks: dict[str, Any]

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
    def verified(self) -> bool:
        """True when every required and every displayed artifact matched its digest at load."""
        return not any(self.verification[k] for k in ("missing", "modified", "unlisted_required"))

    @property
    def selected_model(self) -> str:
        return self.manifest["selection"]["selected_model"]

    @property
    def selected_threshold(self) -> float:
        return self.threshold_for(self.selected_model)

    @property
    def scored_models(self) -> list[str]:
        return [m for m in LEARNED_MODELS if score_column(m) in self.validation_scores.columns]

    def threshold_for(self, model: str) -> float:
        decisions = self.manifest["threshold_policy"]["decisions"]
        if model not in decisions:
            raise UILoadError(f"No threshold decision recorded for model {model!r}.")
        return float(decisions[model]["threshold"])

    def dataset_path(self) -> Path:
        path = Path(self.manifest["dataset"]["path"])
        return path if path.is_absolute() else workspace_root() / path

    def recorded_digest(self, relpath: str) -> str | None:
        return self.manifest["artifacts"].get(relpath)


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
    if isinstance(packages, dict):
        from importlib import metadata

        for name in ("scikit-learn", "numpy", "pandas"):
            try:
                installed = metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
            if packages.get(name) and packages[name] != installed:
                notes.append(f"{name} {packages[name]} at run time vs {installed} installed now.")
    return notes


def _check_size(path: Path, limit: int, what: str) -> None:
    size = path.stat().st_size
    if size > limit:
        raise UILoadError(f"{what} is {size:,} bytes, above the UI cap of {limit:,} bytes.")


def _read_verified_bytes(run_dir: Path, relpath: str, recorded: dict[str, str], limit: int) -> bytes:
    """Read an artifact and check its bytes against the manifest digest before use."""
    path = run_dir / relpath
    if relpath not in recorded:
        raise UILoadError(f"{relpath} is not listed in the manifest and cannot be verified.")
    if not path.is_file():
        raise UILoadError(f"Required artifact {relpath} is missing.")
    _check_size(path, limit, f"artifact {relpath}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != recorded[relpath]:
        raise UILoadError(f"Artifact {relpath} does not match the digest recorded in the manifest.")
    return data


def _parse_json(data: bytes, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UILoadError(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise UILoadError(f"{name} must contain a JSON object")
    return payload


def _parse_csv(data: bytes, name: str, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(data), float_precision="round_trip", **kwargs)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise UILoadError(f"{name} could not be parsed as CSV: {exc}") from exc


def _read_json(path: Path, limit: int = MAX_JSON_BYTES) -> dict[str, Any]:
    """Unverified JSON read (historical mode only, where no manifest exists)."""
    _check_size(path, limit, path.name)
    return _parse_json(path.read_bytes(), path.name)


def _read_csv(path: Path, limit: int, **kwargs) -> pd.DataFrame:
    """Unverified CSV read (historical mode and the checksum-verified source dataset)."""
    _check_size(path, limit, path.name)
    try:
        return pd.read_csv(path, float_precision="round_trip", **kwargs)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise UILoadError(f"{path.name} could not be parsed as CSV: {exc}") from exc


def _validate_validation_scores(scores: pd.DataFrame, name: str = VALIDATION_SCORES_FILENAME) -> pd.DataFrame:
    if len(scores) > MAX_SCORE_ROWS:
        raise UILoadError(f"{name} has more than {MAX_SCORE_ROWS:,} rows; the UI caps input there.")
    required_columns = {"row_id", "split", "label"}
    if not required_columns <= set(scores.columns):
        raise UILoadError(f"{name} lacks columns {sorted(required_columns - set(scores.columns))}")
    if len(scores) == 0:
        raise UILoadError(f"{name} contains no rows.")
    if not (scores["split"] == "validation").all():
        raise UILoadError(f"{name} contains rows from a split other than validation.")
    if not scores["label"].isin([0, 1]).all():
        raise UILoadError(f"{name} contains labels other than 0/1.")
    row_ids = scores["row_id"]
    if not pd.api.types.is_integer_dtype(row_ids) or (row_ids < 0).any():
        raise UILoadError(f"{name} row_id must be non-negative integers.")
    if row_ids.duplicated().any():
        raise UILoadError(f"{name} contains duplicate row ids.")
    score_columns = [score_column(m) for m in LEARNED_MODELS if score_column(m) in scores.columns]
    if not score_columns:
        raise UILoadError(f"{name} contains no model score columns.")
    for column in score_columns:
        values = scores[column]
        if not pd.api.types.is_numeric_dtype(values) or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise UILoadError(f"{name} column {column} must be numeric and finite.")
        if ((values < 0) | (values > 1)).any():
            raise UILoadError(f"{name} column {column} must lie in [0, 1].")
    return scores


# Documented tolerances for cross-file consistency (all quantities were written by the
# same pipeline from the same arrays; disagreement beyond these bounds means the files
# do not describe the same run).
DECISION_METRIC_TOLERANCE = 1e-9
LINEAR_SPEC_SCORE_TOLERANCE = 1e-6
LINEAR_SPEC_CHECK_ROWS = 200


def _finite_unit(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and 0.0 <= value <= 1.0


def _validate_consistency(
    manifest: dict[str, Any],
    scores: pd.DataFrame,
    model_comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
    validation_summary: dict[str, Any],
    data_quality: dict[str, Any],
    split_summary: dict[str, Any],
    analysis_summary: dict[str, Any],
) -> dict[str, Any]:
    """Cross-file invariants that must hold before anything is rendered.

    Establishes that the files describe one and the same run (internal
    consistency). It does not authenticate externally supplied data.
    """
    checks: dict[str, Any] = {}
    selection = manifest.get("selection")
    policy = manifest.get("threshold_policy")
    if not isinstance(selection, dict) or not isinstance(policy, dict):
        raise UILoadError("Manifest selection/threshold_policy must be objects.")
    decisions = policy.get("decisions")
    if not isinstance(decisions, dict):
        raise UILoadError("Manifest threshold_policy.decisions is missing or malformed.")
    minimum_recall = policy.get("minimum_recall")
    if not isinstance(minimum_recall, (int, float)) or not 0 < minimum_recall <= 1:
        raise UILoadError("Manifest threshold_policy.minimum_recall must lie in (0, 1].")

    selected = selection.get("selected_model")
    scored = [m for m in LEARNED_MODELS if score_column(m) in scores.columns]
    if selected not in LEARNED_MODELS:
        raise UILoadError(f"Manifest selected_model {selected!r} is not a known learned model {list(LEARNED_MODELS)}.")
    if selected not in scored:
        raise UILoadError(f"Selected model {selected!r} has no validation score column.")
    if validation_summary.get("selected_model") != selected:
        raise UILoadError("validation_summary.json and the manifest disagree on the selected model.")
    if analysis_summary.get("selected_model") != selected:
        raise UILoadError("analysis_summary.json and the manifest disagree on the selected model.")

    ap = selection.get("validation_average_precision")
    if not isinstance(ap, dict) or any(m not in ap or not _finite_unit(ap[m]) for m in scored):
        raise UILoadError("Manifest validation_average_precision must contain a finite value in [0, 1] for every scored model.")
    if validation_summary.get("validation_average_precision") != ap:
        raise UILoadError("validation_summary.json and the manifest disagree on validation Average Precision.")

    required_columns = {"model", "policy", "threshold", "precision", "recall", "true_positives", "false_positives",
                        "false_negatives", "true_negatives", "test_rows", "test_fraud_cases"}
    if not required_columns <= set(model_comparison.columns):
        raise UILoadError(f"model_comparison.csv lacks columns {sorted(required_columns - set(model_comparison.columns))}.")
    numeric = model_comparison.select_dtypes("number")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise UILoadError("model_comparison.csv contains non-finite values.")
    if ((model_comparison["threshold"] < 0) | (model_comparison["threshold"] > 1)).any():
        raise UILoadError("model_comparison.csv thresholds must lie in [0, 1].")

    y = scores["label"].to_numpy()
    for model in scored:
        decision = decisions.get(model)
        if not isinstance(decision, dict) or not _finite_unit(decision.get("threshold")):
            raise UILoadError(f"Manifest threshold decision for {model!r} must be a finite value in [0, 1].")
        threshold = float(decision["threshold"])
        stored = validation_summary.get("threshold_selection", {}).get(model, {})
        if not isinstance(stored, dict) or stored.get("threshold") != decision["threshold"]:
            raise UILoadError(f"validation_summary.json and the manifest disagree on the threshold for {model!r}.")
        rows = model_comparison[model_comparison["model"] == model]
        if set(rows["policy"]) != {"default_0.5", OPERATING_POLICY}:
            raise UILoadError(f"model_comparison.csv must contain both policies for {model!r}.")
        operating = rows[rows["policy"] == OPERATING_POLICY].iloc[0]
        if float(operating["threshold"]) != threshold:
            raise UILoadError(f"model_comparison.csv operating threshold for {model!r} differs from the manifest decision.")
        # Recompute the claimed validation metrics from the stored labels and scores.
        summary = evaluate_scores(model, "check", pd.Series(y), scores[score_column(model)].to_numpy(dtype=float), threshold)
        for key, metric in (("validation_precision", "precision"), ("validation_recall", "recall")):
            claimed = decision.get(key)
            if claimed is None:
                continue
            if not _finite_unit(claimed) or abs(float(claimed) - summary[metric]) > DECISION_METRIC_TOLERANCE:
                raise UILoadError(
                    f"Recomputed validation {metric} for {model!r} at {threshold} is {summary[metric]:.9f}; "
                    f"the manifest claims {claimed} (tolerance {DECISION_METRIC_TOLERANCE})."
                )
        checks[f"decision_recomputed::{model}"] = {"precision": summary["precision"], "recall": summary["recall"]}

    features = manifest["features"]
    if data_quality.get("features") not in (None, features):
        raise UILoadError("data_quality.json features differ from the manifest feature order.")
    if set(feature_importance.get("feature", pd.Series(dtype=str))) != set(features):
        raise UILoadError("feature_importance.csv does not cover exactly the manifest features.")
    if data_quality.get("source") != manifest["dataset"]["source"]:
        raise UILoadError("data_quality.json and the manifest disagree on the dataset source.")
    if data_quality.get("sha256") not in (None, manifest["dataset"]["sha256"]):
        raise UILoadError("data_quality.json and the manifest disagree on the dataset checksum.")
    nested = analysis_summary.get("data_quality", {})
    if isinstance(nested, dict) and nested.get("source") not in (None, manifest["dataset"]["source"]):
        raise UILoadError("analysis_summary.json and the manifest disagree on the dataset source.")
    if analysis_summary.get("run_id") not in (None, manifest["run_id"]):
        raise UILoadError("analysis_summary.json run_id differs from the manifest run_id.")
    if analysis_summary.get("selected_threshold") != decisions[selected]["threshold"]:
        raise UILoadError("analysis_summary.json selected_threshold differs from the manifest decision.")

    validation_split = split_summary.get("validation", {})
    if validation_split.get("rows") != len(scores) or validation_split.get("fraud") != int(y.sum()):
        raise UILoadError("split_summary.json validation rows/fraud do not match validation_scores.csv.")
    test_split = split_summary.get("test", {})
    if (model_comparison["test_rows"] != test_split.get("rows")).any() or (
        model_comparison["test_fraud_cases"] != test_split.get("fraud")
    ).any():
        raise UILoadError("model_comparison.csv test counts do not match split_summary.json.")
    checks["cross_file"] = "ok"
    return checks


def run_content_key(run_dir: Path) -> str:
    """Fingerprint of the directory's current content (manifest bytes + every file digest).

    Computed on every UI rerun; a cached ``LoadedRun`` is reused only while
    this value is unchanged, so a same-size or timestamp-preserving edit of
    any file forces a fresh load, which then fails verification.
    """
    run_dir = Path(run_dir)
    digest = hashlib.sha256()
    manifest_path = run_dir / MANIFEST_NAME
    if manifest_path.is_file():
        _check_size(manifest_path, MAX_JSON_BYTES, MANIFEST_NAME)
        digest.update(manifest_path.read_bytes())
    for rel, value in sorted(artifact_checksums(run_dir).items()):
        digest.update(rel.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def load_run(run_dir: Path) -> LoadedRun:
    """Open a run directory as a verified, internally consistent snapshot.

    Rejects: no manifest, malformed manifest, unsupported manifest version,
    required file absent, required file not listed in ``artifacts`` (it would
    otherwise escape verification), any listed file whose digest differs, and
    any cross-file inconsistency (see :func:`_validate_consistency`). Files
    present on disk but not listed are never consumed; they are reported as
    ``unverified_extra_files``. Optional displayed files (figures, the linear
    spec) are offered only if listed and matching, and are re-verified when read.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise UILoadError(f"{run_dir} has no {MANIFEST_NAME}; only runs written by fraud_pipeline can be opened.")
    content_key = run_content_key(run_dir)
    try:
        manifest = load_manifest(run_dir, validate=True)
    except ManifestError as exc:
        raise UILoadError(f"Malformed manifest in {run_dir.name}: {exc}") from exc
    notes = _compatibility_notes(manifest)

    recorded: dict[str, str] = manifest["artifacts"]
    for name in REQUIRED_RUN_FILES:
        if not (run_dir / name).is_file():
            raise UILoadError(f"Run {run_dir.name} is missing required file {name}.")
        if name not in recorded:
            raise UILoadError(
                f"Run {run_dir.name}: required artifact {name} has no checksum entry in the manifest, "
                "so it cannot be verified."
            )
    for name in recorded:
        if (run_dir / name).is_file():
            _check_size(run_dir / name, MAX_ARTIFACT_BYTES, f"artifact {name}")
    try:
        verification = verify_run(run_dir, required=REQUIRED_RUN_FILES)
    except ManifestError as exc:
        raise UILoadError(str(exc)) from exc
    problems = {k: v for k, v in verification.items() if v and k != "unexpected"}
    if problems:
        raise UILoadError(f"Artifact verification failed for {run_dir.name}: {problems}")

    verified_files = set(recorded)
    figures = {
        key: run_dir / "figures" / name
        for key, name in FIGURE_NAMES.items()
        if f"figures/{name}" in verified_files
    }
    linear_rel = f"{MODEL_DIRNAME}/{LINEAR_SPEC_FILENAME}"
    linear_spec_path = run_dir / linear_rel if linear_rel in verified_files else None

    def verified_json(name: str) -> dict[str, Any]:
        return _parse_json(_read_verified_bytes(run_dir, name, recorded, MAX_JSON_BYTES), name)

    def verified_csv(name: str, limit: int, **kwargs) -> pd.DataFrame:
        return _parse_csv(_read_verified_bytes(run_dir, name, recorded, limit), name, **kwargs)

    scores = _validate_validation_scores(
        verified_csv(VALIDATION_SCORES_FILENAME, MAX_SCORES_BYTES, nrows=MAX_SCORE_ROWS + 1)
    )
    analysis_summary = verified_json("analysis_summary.json")
    validation_summary = verified_json("validation_summary.json")
    data_quality = verified_json("data_quality.json")
    split_summary = verified_json("split_summary.json")
    model_comparison = verified_csv("model_comparison.csv", MAX_TABLE_BYTES)
    feature_importance = verified_csv("feature_importance.csv", MAX_TABLE_BYTES)

    checks = _validate_consistency(
        manifest, scores, model_comparison, feature_importance, validation_summary,
        data_quality, split_summary, analysis_summary,
    )

    return LoadedRun(
        run_dir=run_dir,
        manifest=manifest,
        verification=verification,
        compatibility_notes=notes,
        analysis_summary=analysis_summary,
        validation_summary=validation_summary,
        data_quality=data_quality,
        split_summary=split_summary,
        model_comparison=model_comparison,
        feature_importance=feature_importance,
        validation_scores=scores,
        figures=figures,
        unverified_extra_files=verification["unexpected"],
        linear_spec_path=linear_spec_path,
        content_key=content_key,
        consistency_checks=checks,
    )


def read_verified_artifact(run: LoadedRun, relpath: str, limit: int = MAX_ARTIFACT_BYTES) -> bytes:
    """Re-read and re-hash a listed artifact immediately before it is displayed."""
    return _read_verified_bytes(run.run_dir, relpath, run.manifest["artifacts"], limit)


def read_verified_figure(run: LoadedRun, key: str) -> bytes:
    if key not in run.figures:
        raise UILoadError(f"Figure {key!r} is not a verified artifact of this run.")
    return read_verified_artifact(run, f"figures/{FIGURE_NAMES[key]}", MAX_TABLE_BYTES)


def load_reproduced_run(run_dir: Path) -> LoadedRun:
    """Reproduced-experiment mode: a verified run on declared non-synthetic data.

    The declared source is *not* authenticated here; :func:`classify_provenance`
    tells the UI whether the run is on the recognized historical dataset.
    """
    run = load_run(run_dir)
    if run.is_synthetic:
        raise UILoadError(
            f"Run {run.run_id} was produced on synthetic data. Open it in Synthetic Demo mode; "
            "Reproduced Experiment mode only accepts kaggle/openml runs."
        )
    return run


# --------------------------------------------------------------------------- provenance


HISTORICAL_DATASET_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
HISTORICAL_FEATURES = [TIME_COLUMN] + REQUIRED_FEATURES
HISTORICAL_CONFIG = {
    "seed": 42, "test_size": 0.2, "validation_size": 0.2, "minimum_recall": 0.8, "duplicate_policy": "drop_exact",
}
HISTORICAL_MODEL_CONFIG = {
    "rf_n_estimators": 200, "rf_max_depth": 12, "rf_min_samples_leaf": 2, "rf_max_features": "sqrt",
    "rf_class_weight": "balanced_subsample",
}


class Provenance(str, Enum):
    HISTORICAL_FROZEN = "historical_frozen"
    HISTORICAL_REPRODUCTION = "historical_reproduction"
    RECOGNIZED_DATASET_RUN = "recognized_dataset_run"
    DECLARED_SOURCE_RUN = "declared_source_run"
    SYNTHETIC = "synthetic"


PROVENANCE_LABELS = {
    Provenance.HISTORICAL_FROZEN: "frozen historical artifacts (July 2026, committed)",
    Provenance.HISTORICAL_REPRODUCTION: "historical reproduction: recognized ULB dataset, historical configuration, comparison checks passed",
    Provenance.RECOGNIZED_DATASET_RUN: "new run on the recognized ULB dataset (checksum matches); configuration or results differ from the historical experiment",
    Provenance.DECLARED_SOURCE_RUN: "run on data declared as {source}; dataset identity NOT verified against the documented historical dataset",
    Provenance.SYNTHETIC: "synthetic fixture; metrics say nothing about real transactions",
}


def recognizes_historical_dataset(manifest: dict[str, Any]) -> bool:
    dataset = manifest.get("dataset", {})
    return (
        dataset.get("source") == "kaggle"
        and dataset.get("sha256") == HISTORICAL_DATASET_SHA256
        and list(manifest.get("features", [])) == HISTORICAL_FEATURES
    )


def uses_historical_configuration(manifest: dict[str, Any]) -> bool:
    config = manifest.get("config", {})
    model = config.get("model", {}) if isinstance(config, dict) else {}
    return all(config.get(k) == v for k, v in HISTORICAL_CONFIG.items()) and all(
        model.get(k) == v for k, v in HISTORICAL_MODEL_CONFIG.items()
    )


def compare_with_historical(run: LoadedRun, historical_dir: Path) -> dict[str, Any]:
    """Defined comparison checks between a run and the frozen artifacts (exact equality)."""
    checks: dict[str, Any] = {"historical_dir_available": Path(historical_dir).is_dir()}
    if not checks["historical_dir_available"]:
        return checks
    try:
        hist = load_historical(historical_dir)
    except UILoadError as exc:
        checks["error"] = str(exc)
        return checks
    checks["selected_model"] = hist.selected_model == run.selected_model
    checks["thresholds"] = all(
        hist.validation_summary["threshold_selection"].get(m, {}).get("threshold") == run.threshold_for(m)
        for m in run.scored_models
    )
    checks["validation_average_precision"] = (
        hist.validation_summary["validation_average_precision"] == run.manifest["selection"]["validation_average_precision"]
    )
    common = [c for c in hist.model_comparison.columns if c in run.model_comparison.columns]
    checks["model_comparison_historical_columns"] = bool(
        len(hist.model_comparison) == len(run.model_comparison)
        and hist.model_comparison[common].reset_index(drop=True).equals(run.model_comparison[common].reset_index(drop=True))
    )
    h_fi = hist.feature_importance.set_index("feature")["importance"]
    r_fi = run.feature_importance.set_index("feature")["importance"]
    checks["feature_importance"] = bool(set(h_fi.index) == set(r_fi.index) and h_fi.equals(r_fi.reindex(h_fi.index)))
    checks["duplicates_removed"] = hist.data_quality.get("exact_duplicate_rows_removed") == run.data_quality.get(
        "exact_duplicate_rows_removed"
    )
    checks["split_sizes"] = all(hist.split_summary.get(k) == run.split_summary.get(k) for k in ("train", "validation", "test"))
    checks["passed"] = all(v is True for k, v in checks.items() if k not in ("historical_dir_available", "passed"))
    return checks


def classify_provenance(run: LoadedRun, historical_dir: Path | None = None) -> tuple[Provenance, str, dict[str, Any]]:
    """Classify what a run's results may be called, from evidence rather than declarations.

    * synthetic source -> SYNTHETIC
    * dataset digest == documented ULB digest and the Kaggle feature schema ->
      RECOGNIZED_DATASET_RUN, upgraded to HISTORICAL_REPRODUCTION only if the
      historical configuration was used *and* the comparison checks against
      the frozen artifacts pass;
    * anything else -> DECLARED_SOURCE_RUN (the source string is the producer's
      claim; a Kaggle-shaped CSV is not the historical dataset).
    """
    if run.is_synthetic:
        return Provenance.SYNTHETIC, PROVENANCE_LABELS[Provenance.SYNTHETIC], {}
    if not recognizes_historical_dataset(run.manifest):
        label = PROVENANCE_LABELS[Provenance.DECLARED_SOURCE_RUN].format(source=run.source)
        return Provenance.DECLARED_SOURCE_RUN, label, {}
    comparison = compare_with_historical(run, historical_dir) if historical_dir is not None else {}
    if uses_historical_configuration(run.manifest) and comparison.get("passed"):
        return Provenance.HISTORICAL_REPRODUCTION, PROVENANCE_LABELS[Provenance.HISTORICAL_REPRODUCTION], comparison
    return Provenance.RECOGNIZED_DATASET_RUN, PROVENANCE_LABELS[Provenance.RECOGNIZED_DATASET_RUN], comparison


def frozen_test_label(provenance: Provenance) -> str:
    return {
        Provenance.SYNTHETIC: "frozen test results for this synthetic run",
        Provenance.DECLARED_SOURCE_RUN: "frozen test results for this run (declared source, dataset identity not verified)",
        Provenance.RECOGNIZED_DATASET_RUN: "frozen test results for this run on the recognized ULB dataset",
        Provenance.HISTORICAL_REPRODUCTION: "frozen test results of the historical reproduction (untouched test split)",
        Provenance.HISTORICAL_FROZEN: "frozen historical test results (committed July 2026)",
    }[provenance]


# --------------------------------------------------------------------------- explanations


def load_linear_explainer(run: LoadedRun) -> LinearModelSpec:
    """Validated, non-executable Logistic Regression description, re-verified against the manifest.

    Structural validation only; see :func:`verify_linear_explainer` for the
    semantic check against stored scores. Never touches joblib or pickle.
    """
    if run.linear_spec_path is None:
        raise UILoadError(
            "This run has no verified Logistic Regression explanation file "
            f"({MODEL_DIRNAME}/{LINEAR_SPEC_FILENAME}); re-run the pipeline to produce one."
        )
    data = read_verified_artifact(run, f"{MODEL_DIRNAME}/{LINEAR_SPEC_FILENAME}", MAX_LINEAR_SPEC_BYTES)
    try:
        spec = parse_linear_spec(json.loads(data.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UILoadError(f"Linear spec is not valid JSON: {exc}") from exc
    except ArtifactIntegrityError as exc:
        raise UILoadError(str(exc)) from exc
    if spec.run_id != run.run_id:
        raise UILoadError("Linear spec belongs to a different run id.")
    if spec.features != list(run.manifest["features"]):
        raise UILoadError("Linear spec feature order differs from the run manifest.")
    if spec.model_name != LOGISTIC_MODEL:
        raise UILoadError(f"Linear spec describes {spec.model_name!r}, not {LOGISTIC_MODEL!r}.")
    return spec


def verify_linear_explainer(run: LoadedRun, spec: LinearModelSpec, feature_rows: pd.DataFrame) -> dict[str, Any]:
    """Semantic check: the spec must reproduce the stored Logistic Regression scores.

    ``feature_rows`` are checksum-verified source rows indexed by ``row_id``
    (bounded to :data:`LINEAR_SPEC_CHECK_ROWS`). Reconstructed
    ``sigmoid(log_odds)`` must match ``validation_scores.csv`` within
    :data:`LINEAR_SPEC_SCORE_TOLERANCE` for every checked row; a spec with a
    valid checksum but unrelated coefficients therefore fails here.
    """
    column = score_column(LOGISTIC_MODEL)
    if column not in run.validation_scores.columns:
        raise UILoadError("This run has no stored Logistic Regression scores to check the explanation against.")
    stored = run.validation_scores.set_index("row_id")[column]
    ids = [int(i) for i in feature_rows.index if int(i) in stored.index][:LINEAR_SPEC_CHECK_ROWS]
    if not ids:
        raise UILoadError("No checksum-verified feature rows are available to check the explanation.")
    X = feature_rows.loc[ids, spec.features].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise UILoadError("Feature rows for the explanation check contain non-finite values.")
    log_odds = spec.intercept + ((X - spec.mean) / spec.scale) @ spec.coef
    reconstructed = 1.0 / (1.0 + np.exp(-log_odds))
    diff = np.abs(reconstructed - stored.loc[ids].to_numpy(dtype=float))
    result = {"rows_checked": len(ids), "max_abs_difference": float(diff.max()), "tolerance": LINEAR_SPEC_SCORE_TOLERANCE}
    if result["max_abs_difference"] > LINEAR_SPEC_SCORE_TOLERANCE:
        raise UILoadError(
            f"The Logistic Regression explanation file does not reproduce the stored scores "
            f"(max |difference| {result['max_abs_difference']:.3g} over {len(ids)} rows, tolerance "
            f"{LINEAR_SPEC_SCORE_TOLERANCE}); explanations are disabled for this run."
        )
    return result


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
    frame = _read_csv(path, MAX_DATASET_BYTES, nrows=MAX_SCORE_ROWS * 10)
    frame.index = pd.RangeIndex(len(frame), name="row_id")
    features = run.manifest["features"]
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise UILoadError(f"Source dataset lacks manifest features {missing}.")
    wanted = [i for i in row_ids if 0 <= i < len(frame)]
    return frame.loc[wanted, features]


def escape_markdown(text: Any, limit: int = 500) -> str:
    """Render untrusted metadata (run ids, source strings, limitations) as inert text.

    Escapes Markdown control characters so supplied text cannot inject links,
    images or formatting, drops control characters and truncates.
    """
    s = str(text)
    s = "".join(ch for ch in s if ch == " " or ch.isprintable())
    s = re.sub(r"([\\`*_{}\[\]()#+\-.!<>|~])", r"\\\1", s)
    return s[:limit] + ("..." if len(s) > limit else "")


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


CAPACITY_COLUMNS = [
    "review_capacity", "threshold", "alerts", "precision", "recall", "false_positives", "missed_fraud",
]


def capacity_scenarios(model: str, labels, scores, capacities: list[int]) -> pd.DataFrame:
    """One row per capacity; an empty list yields an empty table with the full column set."""
    rows = []
    for capacity in capacities:
        t = threshold_for_capacity(scores, capacity)
        summary = threshold_summary(model, labels, scores, t)
        rows.append({"review_capacity": int(capacity), **summary.to_dict()})
    columns = CAPACITY_COLUMNS + [c for c in ThresholdSummary.__dataclass_fields__ if c not in CAPACITY_COLUMNS]
    table = pd.DataFrame(rows, columns=columns)
    table.attrs["model"] = model
    return table


def parse_capacities(text: str, maximum: int) -> list[int]:
    """Parse ``"10, 25; 50"`` into sorted unique positive capacities capped at ``maximum``.

    Blank input returns ``[]`` (callers show an instruction). Zero, negative,
    non-integer and non-numeric tokens raise ``ValueError`` naming the token.
    """
    if maximum < 1:
        raise ValueError("maximum must be at least 1")
    values = []
    for token in (text or "").replace(";", ",").replace("\n", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if not re.fullmatch(r"\+?[0-9]+", token) or int(token) < 1:
            raise ValueError(f"Capacities must be positive whole numbers; got {token!r}")
        value = int(token)
        values.append(min(value, maximum))
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


def explanation_availability(run: "LoadedRun", model: str) -> tuple[bool, str]:
    """Whether a local explanation exists for *the model that produced the displayed score*.

    Only the Logistic Regression pipeline has a supported, non-executable
    explanation artifact. Any other model is reported as unavailable; the
    winning model is never substituted.
    """
    if model != LOGISTIC_MODEL:
        return False, (
            f"No local explanation is available for {model}. The only supported local explanation "
            f"is the additive log-odds decomposition of the {LOGISTIC_MODEL} pipeline. "
            + IMPORTANCE_DISCLAIMER
        )
    if run.linear_spec_path is None:
        return False, (
            f"This run has no verified {MODEL_DIRNAME}/{LINEAR_SPEC_FILENAME}; re-run the pipeline "
            "to export the Logistic Regression coefficients."
        )
    return True, f"Additive log-odds terms of the {LOGISTIC_MODEL} pipeline that produced this score."


def linear_contributions(spec: LinearModelSpec, row: pd.Series) -> pd.DataFrame:
    """Per-feature additive terms of the scaled logistic regression for one row.

    Contribution_i = coef_i * (x_i - mean_i) / scale_i (log-odds units),
    computed from the validated numeric spec - no model object is unpickled.
    """
    x = spec._vector(row)
    z = (x - spec.mean) / spec.scale
    contributions = spec.coef * z
    frame = pd.DataFrame(
        {
            "feature": spec.features,
            "description": [
                "anonymized PCA component" if f.startswith("V") else f.lower() for f in spec.features
            ],
            "value": x,
            "standardized_value": z,
            "log_odds_contribution": contributions,
        }
    )
    frame.attrs["model"] = spec.model_name
    frame.attrs["intercept"] = float(spec.intercept)
    frame.attrs["log_odds"] = float(spec.intercept + contributions.sum())
    frame.attrs["score"] = float(1.0 / (1.0 + np.exp(-frame.attrs["log_odds"])))
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
    provenance: "Provenance | None" = None,
    provenance_label: str | None = None,
) -> dict[str, Any]:
    return {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ui_version": __version__,
        "mode": mode,
        "provenance": provenance.value if provenance else None,
        "provenance_label": provenance_label,
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
    provenance_label: str | None = None,
    frozen_label: str = "frozen test results for this run",
) -> str:
    """Markdown analytical report for the current view.

    Untrusted metadata (source description, limitations, run ids) is escaped so
    the report cannot carry injected links or images.
    """
    synthetic = data_quality.get("source") == "synthetic"
    lines = [f"# Fraud-detection analytical report - {mode}", ""]
    if synthetic:
        lines += ["> **SYNTHETIC DATA.** Every number in this report comes from a generated fixture. "
                  "None of it describes the historical experiment or real transactions.", ""]
    if provenance_label:
        lines += [f"> **Provenance:** {escape_markdown(provenance_label)}", ""]
    lines += [f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC by fraud_pipeline UI {__version__}.", ""]
    lines += ["## Data", "", f"- Source: {escape_markdown(title_source)}"]
    for key in ("sha256", "original_rows", "exact_duplicate_rows_removed", "cleaned_rows", "cleaned_class_counts", "cleaned_fraud_prevalence"):
        if key in data_quality:
            lines.append(f"- {key}: {data_quality[key]}")
    lines += ["", f"## Test-set results ({escape_markdown(frozen_label)})", "",
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
        lines += ["## Provenance", "", f"- run_id: {escape_markdown(manifest['run_id'])}",
                  f"- pipeline_version: {manifest['pipeline_version']}",
                  f"- dataset sha256: {manifest['dataset']['sha256']}",
                  f"- code commit: {manifest['code'].get('commit')}",
                  f"- environment: Python {manifest['environment']['python']}, {manifest['environment']['packages']}", ""]
    if provenance_gaps:
        lines += ["## Provenance gaps", ""] + [f"- {gap}" for gap in provenance_gaps] + [""]
    lines += ["## Limitations", ""] + [f"- {escape_markdown(item)}" for item in (limitations or [])]
    lines += [f"- {SCORE_DISCLAIMER}", f"- {FEATURE_DISCLAIMER}", f"- {IMPORTANCE_DISCLAIMER}", ""]
    return "\n".join(lines)

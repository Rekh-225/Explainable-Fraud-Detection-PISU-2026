"""Model artifacts: a checksum-bound joblib bundle and a non-executable linear spec.

Trust boundary
--------------
``joblib``/pickle files execute arbitrary code on load. A SHA-256 digest in a
sidecar or manifest only detects *accidental* modification; whoever supplies
the model can also supply matching digests, so a digest match is **not**
authentication. Consequently:

* ``load_model_bundle`` refuses to run unless the caller passes
  ``trusted_source=True``, asserting that the run directory was produced by
  this pipeline on a machine they control. The UI never calls it.
* The UI's local explanations use ``LinearModelSpec``: plain JSON with the
  feature order, scaler statistics, coefficients and intercept of the
  Logistic Regression pipeline. It is validated field by field (types,
  dimensions, finite values) and cannot execute code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .manifest import load_manifest, save_json, sha256_path
from .modeling import positive_scores

MODEL_DIRNAME = "models"
MODEL_FILENAME = "selected_model.joblib"
SCHEMA_FILENAME = "selected_model.schema.json"
LINEAR_SPEC_FILENAME = "logistic_regression_linear.json"
LINEAR_SPEC_KIND = "standard_scaler_logistic_regression"
LINEAR_SPEC_VERSION = 1
MAX_LINEAR_SPEC_BYTES = 1 << 20


class ArtifactIntegrityError(RuntimeError):
    """Raised when a model artifact cannot be trusted or does not validate."""


# --------------------------------------------------------------------------- joblib bundle


@dataclass(frozen=True)
class ModelBundle:
    model: BaseEstimator
    model_name: str
    threshold: float
    features: list[str]
    run_id: str

    def score(self, X: pd.DataFrame):
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing model features: {missing}")
        return positive_scores(self.model, X[self.features])


def save_model_bundle(
    run_dir: Path,
    model: BaseEstimator,
    model_name: str,
    threshold: float,
    features: list[str],
    run_id: str,
) -> dict[str, Path]:
    model_dir = run_dir / MODEL_DIRNAME
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILENAME
    joblib.dump(
        {
            "model": model,
            "threshold": float(threshold),
            "features": list(features),
            "selected_model_name": model_name,
            "run_id": run_id,
        },
        model_path,
        compress=3,
    )
    schema_path = model_dir / SCHEMA_FILENAME
    save_json(
        schema_path,
        {
            "run_id": run_id,
            "model_file": MODEL_FILENAME,
            "model_sha256": sha256_path(model_path),
            "selected_model_name": model_name,
            "threshold": float(threshold),
            "comparison": ">=",
            "features": list(features),
            "n_features": len(features),
        },
    )
    return {"model": model_path, "schema": schema_path}


def load_model_bundle(run_dir: Path, *, trusted_source: bool = False) -> ModelBundle:
    """Unpickle the selected model from a run directory you produced yourself.

    ``trusted_source=True`` is a deliberate assertion by the caller that the
    directory was written by this pipeline on a machine they control. The
    integrity checks below (sidecar digest, manifest digest, run id) catch
    corruption and mix-ups; they do not make a foreign bundle safe.
    """
    if not trusted_source:
        raise ArtifactIntegrityError(
            "Refusing to unpickle a model bundle without trusted_source=True. Checksums in the "
            "sidecar/manifest detect accidental changes but do not authenticate who produced the "
            "file; only load bundles you generated yourself."
        )
    run_dir = Path(run_dir)
    model_path = run_dir / MODEL_DIRNAME / MODEL_FILENAME
    schema_path = run_dir / MODEL_DIRNAME / SCHEMA_FILENAME
    if not schema_path.is_file() or not model_path.is_file():
        raise ArtifactIntegrityError(
            f"Run directory {run_dir} lacks {SCHEMA_FILENAME} or {MODEL_FILENAME}; "
            "only bundles written by fraud_pipeline are loadable."
        )
    schema: dict[str, Any] = json.loads(schema_path.read_text(encoding="utf-8"))
    actual = sha256_path(model_path)
    if schema.get("model_sha256") != actual:
        raise ArtifactIntegrityError("Model file digest does not match its schema sidecar")

    manifest = load_manifest(run_dir)
    relative = model_path.relative_to(run_dir).as_posix()
    if manifest.get("artifacts", {}).get(relative) != actual:
        raise ArtifactIntegrityError("Model file digest does not match the run manifest")
    if manifest.get("run_id") != schema.get("run_id"):
        raise ArtifactIntegrityError("Schema sidecar and manifest refer to different runs")

    payload = joblib.load(model_path)
    if payload.get("features") != schema["features"]:
        raise ArtifactIntegrityError("Feature order inside the bundle differs from the schema")
    return ModelBundle(
        model=payload["model"],
        model_name=payload["selected_model_name"],
        threshold=float(payload["threshold"]),
        features=list(payload["features"]),
        run_id=schema["run_id"],
    )


# --------------------------------------------------------------------------- linear spec


@dataclass(frozen=True)
class LinearModelSpec:
    """Numeric description of ``StandardScaler -> LogisticRegression``.

    log_odds(x) = intercept + sum_i coef_i * (x_i - mean_i) / scale_i
    """

    model_name: str
    run_id: str
    features: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def log_odds(self, row: pd.Series | np.ndarray) -> float:
        return float(self.intercept + self.contributions(row).sum())

    def contributions(self, row: pd.Series | np.ndarray) -> np.ndarray:
        x = self._vector(row)
        return self.coef * (x - self.mean) / self.scale

    def _vector(self, row: pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(row, pd.Series):
            missing = [f for f in self.features if f not in row.index]
            if missing:
                raise ValueError(f"Row is missing features: {missing}")
            x = row[self.features].to_numpy(dtype=float)
        else:
            x = np.asarray(row, dtype=float)
        if x.shape != (len(self.features),):
            raise ValueError(f"Expected {len(self.features)} feature values, got shape {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("Row contains non-finite feature values")
        return x


def export_linear_spec(pipeline: BaseEstimator, model_name: str, features: list[str], run_id: str, path: Path) -> Path:
    """Write the fitted scaler + logistic regression as validated JSON."""
    steps = getattr(pipeline, "named_steps", {})
    scaler, model = steps.get("scaler"), steps.get("model")
    if scaler is None or model is None or not hasattr(model, "coef_") or not hasattr(scaler, "mean_"):
        raise ValueError("export_linear_spec requires a fitted Pipeline(scaler=StandardScaler, model=LogisticRegression)")
    if list(getattr(model, "classes_", [0, 1])) != [0, 1]:
        raise ValueError("Linear spec export expects classes_ == [0, 1]")
    payload = {
        "kind": LINEAR_SPEC_KIND,
        "spec_version": LINEAR_SPEC_VERSION,
        "run_id": run_id,
        "model_name": model_name,
        "positive_class": 1,
        "features": list(features),
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
    }
    parse_linear_spec(payload)  # validate what we are about to write
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, payload)
    return path


def _finite_vector(payload: dict[str, Any], key: str, n: int) -> np.ndarray:
    values = payload.get(key)
    if not isinstance(values, list) or len(values) != n:
        raise ArtifactIntegrityError(f"Linear spec field {key!r} must be a list of {n} numbers")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        raise ArtifactIntegrityError(f"Linear spec field {key!r} contains non-numeric entries")
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ArtifactIntegrityError(f"Linear spec field {key!r} contains non-finite values")
    return array


def parse_linear_spec(payload: Any) -> LinearModelSpec:
    """Strictly validate a linear-spec dictionary (types, dimensions, finiteness)."""
    if not isinstance(payload, dict):
        raise ArtifactIntegrityError("Linear spec must be a JSON object")
    if payload.get("kind") != LINEAR_SPEC_KIND or payload.get("spec_version") != LINEAR_SPEC_VERSION:
        raise ArtifactIntegrityError(
            f"Unsupported linear spec kind/version: {payload.get('kind')!r}/{payload.get('spec_version')!r}"
        )
    features = payload.get("features")
    if (
        not isinstance(features, list)
        or not features
        or not all(isinstance(f, str) and f for f in features)
        or len(set(features)) != len(features)
    ):
        raise ArtifactIntegrityError("Linear spec 'features' must be a non-empty list of unique strings")
    for key in ("run_id", "model_name"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ArtifactIntegrityError(f"Linear spec field {key!r} must be a non-empty string")
    if payload.get("positive_class") != 1:
        raise ArtifactIntegrityError("Linear spec positive_class must be 1")
    n = len(features)
    mean = _finite_vector(payload, "scaler_mean", n)
    scale = _finite_vector(payload, "scaler_scale", n)
    coef = _finite_vector(payload, "coef", n)
    if np.any(scale <= 0):
        raise ArtifactIntegrityError("Linear spec scaler_scale must be strictly positive")
    intercept = payload.get("intercept")
    if not isinstance(intercept, (int, float)) or isinstance(intercept, bool) or not math.isfinite(intercept):
        raise ArtifactIntegrityError("Linear spec 'intercept' must be a finite number")
    return LinearModelSpec(
        model_name=payload["model_name"],
        run_id=payload["run_id"],
        features=list(features),
        mean=mean,
        scale=scale,
        coef=coef,
        intercept=float(intercept),
    )


def load_linear_spec(path: Path) -> LinearModelSpec:
    """Read and validate a linear spec JSON file (size-capped, non-executable)."""
    path = Path(path)
    if not path.is_file():
        raise ArtifactIntegrityError(f"Linear spec not found: {path}")
    if path.stat().st_size > MAX_LINEAR_SPEC_BYTES:
        raise ArtifactIntegrityError(f"Linear spec {path.name} exceeds {MAX_LINEAR_SPEC_BYTES} bytes")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"Linear spec {path.name} is not valid JSON: {exc}") from exc
    return parse_linear_spec(payload)

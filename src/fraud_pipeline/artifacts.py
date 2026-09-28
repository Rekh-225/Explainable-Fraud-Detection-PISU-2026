"""Model bundle persistence bound to a feature schema and run manifest.

``joblib``/pickle files execute arbitrary code on load. The loader therefore
never accepts a bare path from a user: it loads only from a run directory,
requires the ``selected_model.schema.json`` sidecar written by this package,
and refuses to unpickle unless the file's SHA-256 matches both the sidecar
and the run manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

from .manifest import load_manifest, save_json, sha256_path
from .modeling import positive_scores

MODEL_DIRNAME = "models"
MODEL_FILENAME = "selected_model.joblib"
SCHEMA_FILENAME = "selected_model.schema.json"


class ArtifactIntegrityError(RuntimeError):
    """Raised when a model artifact cannot be trusted."""


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


def load_model_bundle(run_dir: Path) -> ModelBundle:
    """Load the selected model from a run directory after integrity checks."""
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

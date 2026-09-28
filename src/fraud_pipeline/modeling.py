"""Model definitions matching the historical experiment.

Learned preprocessing (the ``StandardScaler`` in the Logistic Regression
pipeline) lives *inside* the estimator, so calling ``fit`` on the training
split is the only place its statistics can come from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DUMMY_MODEL = "Dummy Prior"
LOGISTIC_MODEL = "Logistic Regression"
FOREST_MODEL = "Random Forest"
LEARNED_MODELS = (LOGISTIC_MODEL, FOREST_MODEL)


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters of the historical experiment (defaults) or a reduced variant."""

    seed: int = 42
    rf_n_estimators: int = 200
    rf_max_depth: int | None = 12
    rf_min_samples_leaf: int = 2
    rf_max_features: str | int | float = "sqrt"
    rf_class_weight: str | None = "balanced_subsample"
    lr_max_iter: int = 1000
    n_jobs: int = 1


def build_models(config: ModelConfig = ModelConfig()) -> dict[str, BaseEstimator]:
    return {
        DUMMY_MODEL: DummyClassifier(strategy="prior", random_state=config.seed),
        LOGISTIC_MODEL: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=config.lr_max_iter,
                        random_state=config.seed,
                    ),
                ),
            ]
        ),
        FOREST_MODEL: RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
            max_features=config.rf_max_features,
            class_weight=config.rf_class_weight,
            n_jobs=config.n_jobs,
            random_state=config.seed,
        ),
    }


def fit_model(model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
    """Fit on the training split only. Kept as a single choke point for auditing."""
    return model.fit(X_train, y_train)


def positive_scores(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """Probability-like score for the positive (fraud) class."""
    classes = list(getattr(model, "classes_", [0, 1]))
    return model.predict_proba(X)[:, classes.index(1)]


def describe_model(model: BaseEstimator) -> dict[str, Any]:
    """JSON-friendly class name and parameters for the manifest."""
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "params": _json_params(model.get_params(deep=True)),
    }


def _json_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, BaseEstimator):
            out[key] = describe_model(value)
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], tuple):
            out[key] = [name for name, _ in value]
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = repr(value)
    return out

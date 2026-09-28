"""Deterministic synthetic fixtures shaped like the credit-card dataset.

These fixtures exist so that the pipeline and its tests can run without
network access or the restricted Kaggle file. They are *not* a stand-in for
the real data: metrics computed on them must never be reported as results of
the historical experiment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import LABEL_COLUMN, PCA_FEATURES, TIME_COLUMN

# A handful of components are shifted for fraud rows so that models have
# signal to learn; the rest are pure noise, mirroring the real data where a
# few PCA components carry most of the information.
_SIGNAL_FEATURES = {"V4": 2.5, "V10": -2.0, "V12": -2.5, "V14": -3.0, "V17": -2.0}


def make_synthetic_frame(
    n_rows: int = 4000,
    fraud_rate: float = 0.02,
    seed: int = 0,
    include_time: bool = True,
    n_duplicates: int = 0,
) -> pd.DataFrame:
    """Return a deterministic frame with the Kaggle column layout.

    ``n_duplicates`` appends exact copies of existing rows (label included) so
    that duplicate handling can be exercised.
    """
    if n_rows < 10:
        raise ValueError("n_rows must be at least 10")
    if not 0 < fraud_rate < 1:
        raise ValueError("fraud_rate must be in (0, 1)")
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(round(n_rows * fraud_rate)))
    labels = np.zeros(n_rows, dtype=int)
    labels[rng.choice(n_rows, size=n_fraud, replace=False)] = 1

    frame = pd.DataFrame(rng.normal(size=(n_rows, len(PCA_FEATURES))), columns=PCA_FEATURES)
    for feature, shift in _SIGNAL_FEATURES.items():
        frame.loc[labels == 1, feature] += shift
    frame["Amount"] = np.round(rng.lognormal(mean=3.0, sigma=1.2, size=n_rows), 2)
    frame[LABEL_COLUMN] = labels
    if include_time:
        frame.insert(0, TIME_COLUMN, np.cumsum(rng.integers(0, 4, size=n_rows)).astype(float))

    if n_duplicates:
        duplicated = frame.iloc[rng.choice(n_rows, size=n_duplicates, replace=False)]
        frame = pd.concat([frame, duplicated], ignore_index=True)
    return frame


def write_synthetic_csv(path: str | Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_frame(**kwargs).to_csv(path, index=False)
    return path

from pathlib import Path

import pandas as pd
import pytest

from fraud_pipeline.synthetic import make_synthetic_frame


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    return make_synthetic_frame(n_rows=600, fraud_rate=0.05, seed=11)


@pytest.fixture
def synthetic_csv(tmp_path: Path, synthetic_frame: pd.DataFrame) -> Path:
    path = tmp_path / "synthetic.csv"
    synthetic_frame.to_csv(path, index=False)
    return path


def write_csv(tmp_path: Path, frame: pd.DataFrame, name: str = "data.csv") -> Path:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path

"""Schema rejection, label validation, non-finite input and duplicate handling."""

import numpy as np
import pandas as pd
import pytest

from conftest import write_csv
from fraud_pipeline.data import (
    LABEL_COLUMN,
    REQUIRED_FEATURES,
    SchemaError,
    deduplicate,
    load_dataset,
    validate_frame,
    validate_labels,
)
from fraud_pipeline.synthetic import make_synthetic_frame


def test_kaggle_layout_loads_with_time_first(synthetic_csv):
    dataset = load_dataset(synthetic_csv, "kaggle")
    assert dataset.features == ["Time"] + REQUIRED_FEATURES
    assert dataset.info.has_time is True
    assert dataset.frame.index.name == "row_id"
    assert dataset.info.sha256 and len(dataset.info.sha256) == 64
    assert dataset.info.raw_rows == 600
    assert set(dataset.info.class_counts) == {0, 1}


def test_missing_required_column_is_rejected(tmp_path, synthetic_frame):
    path = write_csv(tmp_path, synthetic_frame.drop(columns=["V7"]))
    with pytest.raises(SchemaError, match=r"missing required columns: \['V7'\]"):
        load_dataset(path, "kaggle")


def test_missing_label_column_is_rejected(tmp_path, synthetic_frame):
    path = write_csv(tmp_path, synthetic_frame.drop(columns=[LABEL_COLUMN]))
    with pytest.raises(SchemaError, match="Class"):
        load_dataset(path, "kaggle")


def test_kaggle_source_without_time_is_rejected(tmp_path):
    frame = make_synthetic_frame(n_rows=200, seed=1, include_time=False)
    path = write_csv(tmp_path, frame)
    with pytest.raises(SchemaError, match="requires the Time column"):
        load_dataset(path, "kaggle")


def test_openml_source_without_time_is_accepted_and_recorded(tmp_path):
    frame = make_synthetic_frame(n_rows=200, seed=1, include_time=False)
    dataset = load_dataset(write_csv(tmp_path, frame), "openml")
    assert dataset.info.has_time is False
    assert dataset.features == REQUIRED_FEATURES
    assert "Time" in dataset.info.source_description


def test_unknown_source_is_rejected(synthetic_frame):
    with pytest.raises(SchemaError, match="Unknown source"):
        validate_frame(synthetic_frame, "kaggle_or_whatever")


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv", "kaggle")


def test_non_numeric_feature_is_rejected(tmp_path, synthetic_frame):
    frame = synthetic_frame.copy()
    frame["Amount"] = frame["Amount"].astype(object)
    frame.loc[3, "Amount"] = "twelve"
    with pytest.raises(SchemaError, match="non-numeric: \\['Amount'\\]"):
        load_dataset(write_csv(tmp_path, frame), "kaggle")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_feature_is_rejected(tmp_path, synthetic_frame, bad):
    frame = synthetic_frame.copy()
    frame.loc[5, "V14"] = bad
    with pytest.raises(SchemaError, match="V14"):
        load_dataset(write_csv(tmp_path, frame), "kaggle")


def test_extra_columns_are_ignored_but_recorded(tmp_path, synthetic_frame):
    frame = synthetic_frame.assign(merchant_id=1)
    dataset = load_dataset(write_csv(tmp_path, frame), "kaggle")
    assert dataset.info.ignored_columns == ["merchant_id"]
    assert "merchant_id" not in dataset.features


# --- labels ---------------------------------------------------------------


@pytest.mark.parametrize("value", [0.7, 2, -1, 1.5])
def test_lossy_labels_are_rejected_before_conversion(tmp_path, synthetic_frame, value):
    frame = synthetic_frame.copy()
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(float)
    frame.loc[0, LABEL_COLUMN] = value
    with pytest.raises(SchemaError, match="must contain only 0 and 1"):
        load_dataset(write_csv(tmp_path, frame), "kaggle")


def test_float_labels_that_are_exactly_binary_are_accepted():
    labels = validate_labels(pd.Series([0.0, 1.0, 0.0]))
    assert labels.tolist() == [0, 1, 0]
    assert labels.dtype.kind == "i"


def test_string_labels_are_accepted_only_when_literal_zero_one():
    assert validate_labels(pd.Series(["0", "1", " 1"])).tolist() == [0, 1, 1]
    with pytest.raises(SchemaError):
        validate_labels(pd.Series(["0", "yes"]))
    with pytest.raises(SchemaError):
        validate_labels(pd.Series(["0", "1.0"]))


def test_missing_label_values_are_rejected():
    with pytest.raises(SchemaError, match="missing"):
        validate_labels(pd.Series([0, 1, None]))


def test_single_class_dataset_is_rejected(tmp_path, synthetic_frame):
    frame = synthetic_frame.copy()
    frame[LABEL_COLUMN] = 0
    with pytest.raises(SchemaError, match="Exactly two classes"):
        load_dataset(write_csv(tmp_path, frame), "kaggle")


def test_three_class_dataset_is_rejected(synthetic_frame):
    frame = synthetic_frame.copy()
    frame.loc[0, LABEL_COLUMN] = 2
    with pytest.raises(SchemaError):
        validate_frame(frame, "kaggle")


# --- duplicates -----------------------------------------------------------


def test_exact_duplicates_are_dropped_keeping_first_and_row_ids(tmp_path):
    frame = make_synthetic_frame(n_rows=300, seed=5, n_duplicates=17)
    path = write_csv(tmp_path, frame)
    dataset = load_dataset(path, "kaggle")
    assert dataset.info.raw_rows == 317
    assert dataset.info.duplicates_removed == 17
    assert dataset.info.rows == 300
    assert dataset.frame.index.tolist() == list(range(300))
    assert dataset.info.duplicate_policy == "drop_exact"


def test_keep_policy_retains_duplicates(tmp_path):
    frame = make_synthetic_frame(n_rows=300, seed=5, n_duplicates=17)
    dataset = load_dataset(write_csv(tmp_path, frame), "kaggle", duplicate_policy="keep")
    assert dataset.info.duplicates_removed == 0
    assert dataset.info.duplicates_retained == 17
    assert dataset.info.rows == 317


def test_rows_with_conflicting_labels_are_not_duplicates(synthetic_frame):
    frame = synthetic_frame.copy()
    twin = frame.iloc[[0]].copy()
    twin[LABEL_COLUMN] = 1 - twin[LABEL_COLUMN]
    frame = pd.concat([frame, twin], ignore_index=True)
    deduped, removed, retained = deduplicate(frame, "drop_exact")
    assert removed == 0 and retained == 0 and len(deduped) == len(frame)


def test_unknown_duplicate_policy_is_rejected(synthetic_frame):
    with pytest.raises(ValueError):
        deduplicate(synthetic_frame, "fuzzy")

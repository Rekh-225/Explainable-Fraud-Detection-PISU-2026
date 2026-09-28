"""Disjoint stratified splits, fingerprints, and train-only preprocessing."""

import numpy as np
import pandas as pd
import pytest

from fraud_pipeline.data import load_dataset
from fraud_pipeline.modeling import LOGISTIC_MODEL, ModelConfig, build_models, fit_model
from fraud_pipeline.splitting import fingerprint_row_ids, split_dataset


@pytest.fixture
def dataset(synthetic_csv):
    return load_dataset(synthetic_csv, "kaggle")


def test_splits_are_disjoint_and_cover_all_rows(dataset):
    splits = split_dataset(dataset.X, dataset.y, seed=42)
    ids = [set(s.row_ids) for s in splits]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert ids[0] | ids[1] | ids[2] == set(dataset.frame.index)
    assert splits.overlaps() == {"train_validation": 0, "train_test": 0, "validation_test": 0}


def test_split_sizes_follow_64_16_20(dataset):
    splits = split_dataset(dataset.X, dataset.y, seed=42)
    n = len(dataset.frame)
    assert len(splits.test.y) == round(0.2 * n)
    assert len(splits.validation.y) == round(0.2 * (n - len(splits.test.y)))
    assert len(splits.train.y) == n - len(splits.test.y) - len(splits.validation.y)


def test_stratification_keeps_positives_in_every_split(dataset):
    splits = split_dataset(dataset.X, dataset.y, seed=42)
    for split in splits:
        assert split.y.sum() >= 1
        assert abs(split.y.mean() - dataset.y.mean()) < 0.02


def test_split_is_deterministic_for_seed_and_changes_with_seed(dataset):
    a = split_dataset(dataset.X, dataset.y, seed=42).fingerprints()
    b = split_dataset(dataset.X, dataset.y, seed=42).fingerprints()
    c = split_dataset(dataset.X, dataset.y, seed=43).fingerprints()
    assert a == b
    assert a != c


def test_fingerprint_is_order_independent_and_membership_sensitive():
    assert fingerprint_row_ids([3, 1, 2]) == fingerprint_row_ids([1, 2, 3])
    assert fingerprint_row_ids([1, 2, 3]) != fingerprint_row_ids([1, 2, 4])


def test_split_requires_two_classes_and_enough_minority_rows(dataset):
    y_single = pd.Series(0, index=dataset.y.index)
    with pytest.raises(ValueError, match="two classes"):
        split_dataset(dataset.X, y_single)
    y_rare = dataset.y.copy()
    y_rare[:] = 0
    y_rare.iloc[:2] = 1
    with pytest.raises(ValueError, match="at least 3"):
        split_dataset(dataset.X, y_rare)


def test_summary_contains_fingerprints_and_counts(dataset):
    summary = split_dataset(dataset.X, dataset.y, seed=7).summary()
    assert summary["random_state"] == 7
    assert set(summary["membership_fingerprints_sha256"]) == {"train", "validation", "test"}
    assert summary["train"]["rows"] + summary["validation"]["rows"] + summary["test"]["rows"] == len(
        dataset.frame
    )


def test_scaler_statistics_come_only_from_training_split(dataset):
    splits = split_dataset(dataset.X, dataset.y, seed=42)
    model = build_models(ModelConfig(seed=0))[LOGISTIC_MODEL]
    fit_model(model, splits.train.X, splits.train.y)
    scaler = model.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, splits.train.X.mean().to_numpy())
    np.testing.assert_allclose(scaler.scale_, splits.train.X.std(ddof=0).to_numpy())
    pooled_mean = pd.concat([splits.train.X, splits.validation.X]).mean().to_numpy()
    assert not np.allclose(scaler.mean_, pooled_mean)


def test_changing_validation_or_test_rows_does_not_change_fitted_model(dataset):
    splits = split_dataset(dataset.X, dataset.y, seed=42)
    model_a = build_models(ModelConfig(seed=0))[LOGISTIC_MODEL]
    fit_model(model_a, splits.train.X, splits.train.y)
    # Corrupt the held-out data wildly; the fitted parameters must be identical.
    splits.validation.X.iloc[:, :] = 1e6
    splits.test.X.iloc[:, :] = -1e6
    model_b = build_models(ModelConfig(seed=0))[LOGISTIC_MODEL]
    fit_model(model_b, splits.train.X, splits.train.y)
    np.testing.assert_array_equal(
        model_a.named_steps["model"].coef_, model_b.named_steps["model"].coef_
    )
    np.testing.assert_array_equal(
        model_a.named_steps["scaler"].mean_, model_b.named_steps["scaler"].mean_
    )

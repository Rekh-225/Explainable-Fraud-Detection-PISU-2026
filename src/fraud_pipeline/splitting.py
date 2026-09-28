"""Stratified train/validation/test partitioning with membership fingerprints.

The two-stage ``train_test_split`` call sequence is kept identical to the
historical script so that, for the same de-duplicated data and seed, the
partitions are the same rows.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class Split:
    X: pd.DataFrame
    y: pd.Series
    name: str

    @property
    def row_ids(self) -> list[int]:
        return [int(i) for i in self.X.index]


@dataclass(frozen=True)
class SplitResult:
    train: Split
    validation: Split
    test: Split
    seed: int
    test_size: float
    validation_size: float

    def __iter__(self):
        return iter((self.train, self.validation, self.test))

    def fingerprints(self) -> dict[str, str]:
        return {split.name: fingerprint_row_ids(split.row_ids) for split in self}

    def overlaps(self) -> dict[str, int]:
        ids = {split.name: set(split.row_ids) for split in self}
        return {
            "train_validation": len(ids["train"] & ids["validation"]),
            "train_test": len(ids["train"] & ids["test"]),
            "validation_test": len(ids["validation"] & ids["test"]),
        }

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "random_state": self.seed,
            "split_method": (
                "stratified random "
                f"{(1 - self.test_size) * (1 - self.validation_size):.0%}/"
                f"{(1 - self.test_size) * self.validation_size:.0%}/"
                f"{self.test_size:.0%} train-validation-test"
            ),
        }
        for split in self:
            summary[split.name] = {
                "rows": int(len(split.y)),
                "fraud": int(split.y.sum()),
                "rate": float(split.y.mean()),
            }
        summary["index_overlap"] = self.overlaps()
        summary["membership_fingerprints_sha256"] = self.fingerprints()
        return summary


def fingerprint_row_ids(row_ids: list[int]) -> str:
    """SHA-256 over the sorted row ids; identical membership => identical digest."""
    payload = ",".join(str(i) for i in sorted(row_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 42,
    test_size: float = 0.20,
    validation_size: float = 0.20,
) -> SplitResult:
    """Stratified 64/16/20 split (defaults) reproducing the historical procedure."""
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index (row ids)")
    if y.nunique() != 2:
        raise ValueError("Stratified splitting requires exactly two classes")
    minority = int(y.value_counts().min())
    if minority < 3:
        raise ValueError(
            f"Minority class has {minority} rows; at least 3 are needed for a stratified 3-way split"
        )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=validation_size, stratify=y_dev, random_state=seed
    )
    result = SplitResult(
        train=Split(X_train, y_train, "train"),
        validation=Split(X_val, y_val, "validation"),
        test=Split(X_test, y_test, "test"),
        seed=seed,
        test_size=test_size,
        validation_size=validation_size,
    )
    if any(result.overlaps().values()):
        raise RuntimeError("Split partitions overlap; this should be impossible")
    return result

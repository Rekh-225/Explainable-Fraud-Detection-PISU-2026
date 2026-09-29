"""Regression coverage for manifest coverage, bounded inputs and malformed artifacts.

Findings addressed: required artifacts escaping checksum coverage (missing
checksum entry, missing file, changed file), malformed manifests/JSON/CSV,
byte and row caps applied before/while parsing, score-column validation.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_pipeline.data import sha256_file
from fraud_pipeline.manifest import (
    MANIFEST_NAME,
    ManifestError,
    load_manifest,
    save_json,
    validate_manifest_structure,
    verify_run,
)
from fraud_pipeline.ui import core


@pytest.fixture(scope="module")
def base_run(tmp_path_factory) -> Path:
    run_dir, _ = core.ensure_synthetic_demo_run(tmp_path_factory.mktemp("integrity"), rows=900, seed=21)
    return run_dir


@pytest.fixture
def run_copy(base_run, tmp_path) -> Path:
    copy = tmp_path / "run"
    shutil.copytree(base_run, copy)
    return copy


def _rewrite(copy: Path, name: str, content: str, *, update_manifest: bool) -> None:
    (copy / name).write_text(content, encoding="utf-8")
    if update_manifest:
        manifest = load_manifest(copy)
        manifest["artifacts"][name] = sha256_file(copy / name)
        save_json(copy / MANIFEST_NAME, manifest)


def _edit_manifest(copy: Path, mutate) -> None:
    manifest = load_manifest(copy)
    mutate(manifest)
    save_json(copy / MANIFEST_NAME, manifest)


# --- the reviewer's exact reproduction ---------------------------------------------


def test_required_artifact_without_checksum_entry_is_rejected(run_copy):
    """Reviewer repro: drop model_comparison.csv from artifacts, modify the file, load_run."""

    def drop(manifest):
        del manifest["artifacts"]["model_comparison.csv"]

    _edit_manifest(run_copy, drop)
    _rewrite(run_copy, "model_comparison.csv", "model,policy,threshold,precision\nRandom Forest,default_0.5,0.5,1.0\n", update_manifest=False)
    report = verify_run(run_copy, required=core.REQUIRED_RUN_FILES)
    assert report["unlisted_required"] == ["model_comparison.csv"]
    assert report["unexpected"] == ["model_comparison.csv"]
    with pytest.raises(core.UILoadError, match="no checksum entry"):
        core.load_run(run_copy)


def test_required_artifact_missing_file_is_rejected(run_copy):
    (run_copy / "feature_importance.csv").unlink()
    assert verify_run(run_copy)["missing"] == ["feature_importance.csv"]
    with pytest.raises(core.UILoadError, match="missing required file feature_importance.csv"):
        core.load_run(run_copy)


def test_required_artifact_changed_is_rejected(run_copy):
    _rewrite(run_copy, "split_summary.json", '{"random_state": 1}', update_manifest=False)
    assert verify_run(run_copy)["modified"] == ["split_summary.json"]
    with pytest.raises(core.UILoadError, match="verification failed.*modified"):
        core.load_run(run_copy)


def test_all_required_files_are_checked_for_checksum_entries(run_copy):
    for name in core.REQUIRED_RUN_FILES:
        copy = run_copy.parent / f"drop_{name.replace('.', '_')}"
        shutil.copytree(run_copy, copy)
        _edit_manifest(copy, lambda m, n=name: m["artifacts"].pop(n))
        with pytest.raises(core.UILoadError, match="no checksum entry"):
            core.load_run(copy)


def test_extra_unlisted_files_are_ignored_but_reported(run_copy):
    (run_copy / "notes.txt").write_text("scratch", encoding="utf-8")
    (run_copy / "figures" / "07_extra.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    run = core.load_run(run_copy)
    assert run.verified
    assert run.unverified_extra_files == ["figures/07_extra.png", "notes.txt"]
    assert all(path.name != "07_extra.png" for path in run.figures.values())


def test_displayed_figure_is_hidden_when_its_checksum_fails(run_copy):
    target = run_copy / "figures" / "04_precision_recall_curves.png"
    target.write_bytes(target.read_bytes() + b"\x00")
    with pytest.raises(core.UILoadError, match="modified"):
        core.load_run(run_copy)


def test_figure_not_listed_in_manifest_is_not_displayed(run_copy):
    _edit_manifest(run_copy, lambda m: m["artifacts"].pop("figures/04_precision_recall_curves.png"))
    run = core.load_run(run_copy)
    assert "precision_recall_curves" not in run.figures
    assert "figures/04_precision_recall_curves.png" in run.unverified_extra_files


def test_linear_spec_used_only_when_listed_and_matching(run_copy):
    rel = "models/logistic_regression_linear.json"
    _edit_manifest(run_copy, lambda m: m["artifacts"].pop(rel))
    run = core.load_run(run_copy)
    assert run.linear_spec_path is None
    with pytest.raises(core.UILoadError, match="no verified"):
        core.load_linear_explainer(run)


# --- manifest structure --------------------------------------------------------------


def test_malformed_manifest_json_is_reported(run_copy):
    (run_copy / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(core.UILoadError, match="Malformed manifest.*not valid JSON"):
        core.load_run(run_copy)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda m: m.pop("artifacts"), "missing keys"),
        (lambda m: m.__setitem__("artifacts", []), "must be a non-empty object"),
        (lambda m: m["artifacts"].__setitem__("model_comparison.csv", "abc"), "malformed sha256"),
        (lambda m: m["artifacts"].__setitem__("../escape.csv", "0" * 64), "not a safe relative path"),
        (lambda m: m.__setitem__("run_id", ""), "run_id"),
        (lambda m: m["dataset"].__setitem__("sha256", "xyz"), "dataset.sha256"),
        (lambda m: m.__setitem__("features", []), "features"),
    ],
)
def test_structurally_invalid_manifests_are_rejected(run_copy, mutate, message):
    _edit_manifest(run_copy, mutate)
    with pytest.raises(core.UILoadError, match=message):
        core.load_run(run_copy)


def test_validate_manifest_structure_accepts_real_manifest(base_run):
    manifest = load_manifest(base_run)
    assert validate_manifest_structure(manifest) is manifest
    with pytest.raises(ManifestError):
        validate_manifest_structure([])


def test_manifest_without_threshold_decisions_is_rejected(run_copy):
    _edit_manifest(run_copy, lambda m: m["threshold_policy"].pop("decisions"))
    with pytest.raises(core.UILoadError, match="decisions"):
        core.load_run(run_copy)


# --- bounded inputs -------------------------------------------------------------------


def test_validation_scores_row_cap_is_applied_during_parsing(run_copy, monkeypatch):
    monkeypatch.setattr(core, "MAX_SCORE_ROWS", 50)
    calls = {}
    original = pd.read_csv

    def spy(path, **kwargs):
        calls["nrows"] = kwargs.get("nrows")
        return original(path, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    with pytest.raises(core.UILoadError, match="more than 50 rows"):
        core.load_run(run_copy)
    assert calls["nrows"] == 51  # parser stopped after cap + 1 rows instead of reading everything


def test_byte_caps_are_checked_before_reading(run_copy, monkeypatch):
    monkeypatch.setattr(core, "MAX_ARTIFACT_BYTES", 10)
    with pytest.raises(core.UILoadError, match="above the UI cap"):
        core.load_run(run_copy)


def test_manifest_size_cap(run_copy, monkeypatch):
    import fraud_pipeline.manifest as manifest_module

    monkeypatch.setattr(manifest_module, "MAX_MANIFEST_BYTES", 10)
    with pytest.raises(core.UILoadError, match="exceeds"):
        core.load_run(run_copy)


def test_dataset_size_cap_checked_before_reading(base_run, monkeypatch):
    run = core.load_run(base_run)
    monkeypatch.setattr(core, "MAX_DATASET_BYTES", 10)
    with pytest.raises(core.UILoadError, match="above the UI cap"):
        core.load_dataset_rows(run, [0])


def _scores_frame(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "validation_scores.csv", float_precision="round_trip")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda f: f.drop(columns=["score__random_forest", "score__logistic_regression"]), "no model score columns"),
        (lambda f: f.assign(score__random_forest=np.nan), "numeric and finite"),
        (lambda f: f.assign(score__random_forest=1.5), r"lie in \[0, 1\]"),
        (lambda f: f.assign(row_id=f["row_id"].astype(float) + 0.5), "non-negative integers"),
        (lambda f: f.assign(row_id=-1), "non-negative integers"),
        (lambda f: pd.concat([f, f.head(1)]), "duplicate row ids"),
        (lambda f: f.assign(label=2), "labels other than 0/1"),
        (lambda f: f.assign(split="test"), "other than validation"),
        (lambda f: f.head(0), "contains no rows"),
    ],
)
def test_invalid_validation_scores_are_rejected(run_copy, mutate, message):
    frame = mutate(_scores_frame(run_copy))
    frame.to_csv(run_copy / "validation_scores.csv", index=False)
    _edit_manifest(run_copy, lambda m: m["artifacts"].__setitem__("validation_scores.csv", sha256_file(run_copy / "validation_scores.csv")))
    with pytest.raises(core.UILoadError, match=message):
        core.load_run(run_copy)


def test_malformed_json_artifact_is_reported(run_copy):
    _rewrite(run_copy, "data_quality.json", "[1, 2", update_manifest=True)
    with pytest.raises(core.UILoadError, match="not valid JSON"):
        core.load_run(run_copy)


def test_malformed_csv_artifact_is_reported(run_copy):
    _rewrite(run_copy, "model_comparison.csv", 'a,b\n1,2,3,"unterminated\n', update_manifest=True)
    with pytest.raises(core.UILoadError, match="could not be parsed as CSV|no rows|lacks columns"):
        core.load_run(run_copy)


def test_historical_loader_reports_malformed_json(tmp_path):
    for name in core.HISTORICAL_FILES:
        (tmp_path / name).write_text("model\n" if name.endswith(".csv") else "{}", encoding="utf-8")
    (tmp_path / "analysis_summary.json").write_text("nope", encoding="utf-8")
    with pytest.raises(core.UILoadError, match="not valid JSON"):
        core.load_historical(tmp_path)


def test_cli_verify_run_reports_unlisted_required(run_copy, capsys):
    from fraud_pipeline.cli import main

    _edit_manifest(run_copy, lambda m: m["artifacts"].pop("model_comparison.csv"))
    assert main(["verify-run", str(run_copy)]) == 1
    assert '"unlisted_required"' in capsys.readouterr().out

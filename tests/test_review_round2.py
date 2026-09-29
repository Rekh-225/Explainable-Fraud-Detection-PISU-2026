"""Regression coverage for the second local review.

Findings: stale cached verification, shallow manifest validation, provenance
labels, installed-package paths, duplicates under ``keep``, run-id
confinement, CSV limits, per-row annotation notes, Markdown escaping.

All fixtures are generated test data (``ui_fixtures``). Runs declared with
``source="kaggle"`` only have the Kaggle *column layout*; they are never the
ULB dataset and must classify as declared-source runs, not historical ones.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_pipeline import workspace
from fraud_pipeline.data import DatasetTooLargeError, load_dataset
from fraud_pipeline.manifest import MANIFEST_NAME, ManifestError, is_safe_artifact_path, load_manifest, save_json, validate_manifest_structure
from fraud_pipeline.modeling import ModelConfig
from fraud_pipeline.pipeline import RunConfig, prepare_run_dir, run_pipeline
from fraud_pipeline.splitting import split_dataset
from fraud_pipeline.synthetic import make_synthetic_frame, write_synthetic_csv
from fraud_pipeline.ui import core
from ui_fixtures import make_run

LR, RF = "Logistic Regression", "Random Forest"


@pytest.fixture(scope="module")
def layout_run(tmp_path_factory):
    """Generated data with Kaggle layout, declared source=kaggle (test-only fixture)."""
    root = tmp_path_factory.mktemp("r2")
    result = make_run(root, "layout", source="kaggle", winner=LR, figures=True)
    return result.run_dir


@pytest.fixture
def run_copy(layout_run, tmp_path):
    copy = tmp_path / "run"
    shutil.copytree(layout_run, copy)
    return copy


def _rehash(copy: Path, *names: str) -> None:
    manifest = load_manifest(copy)
    for name in names:
        manifest["artifacts"][name] = core.sha256_file(copy / name)
    save_json(copy / MANIFEST_NAME, manifest)


def _edit_manifest(copy: Path, mutate) -> None:
    manifest = load_manifest(copy)
    mutate(manifest)
    save_json(copy / MANIFEST_NAME, manifest)


def _same_size_flip(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    path.write_bytes(bytes(data))


def _preserve_times(paths, fn):
    stats = {p: p.stat() for p in paths}
    fn()
    for p, st in stats.items():
        os.utime(p, (st.st_atime, st.st_mtime))


# --------------------------------------------------------------------------- F1 stale verification


def test_content_key_changes_on_same_size_timestamp_preserving_edit(run_copy):
    before = core.run_content_key(run_copy)
    fig = run_copy / "figures" / "04_precision_recall_curves.png"
    _preserve_times([fig, run_copy / MANIFEST_NAME], lambda: _same_size_flip(fig))
    assert fig.stat().st_size == fig.stat().st_size and core.run_content_key(run_copy) != before
    with pytest.raises(core.UILoadError, match="modified"):
        core.load_run(run_copy)


@pytest.mark.parametrize(
    "relpath",
    ["figures/04_precision_recall_curves.png", "model_comparison.csv", "models/logistic_regression_linear.json"],
)
def test_artifact_changed_after_loading_is_refused_on_read(run_copy, relpath):
    run = core.load_run(run_copy)
    target = run_copy / relpath
    _preserve_times([target, run_copy / MANIFEST_NAME], lambda: _same_size_flip(target))
    with pytest.raises(core.UILoadError, match="does not match the digest"):
        core.read_verified_artifact(run, relpath)
    if relpath.startswith("figures/"):
        with pytest.raises(core.UILoadError):
            core.read_verified_figure(run, "precision_recall_curves")
    if relpath.endswith("linear.json"):
        with pytest.raises(core.UILoadError):
            core.load_linear_explainer(run)
    assert core.run_content_key(run_copy) != run.content_key


def test_verified_snapshot_parses_the_bytes_it_hashed(run_copy):
    run = core.load_run(run_copy)
    assert run.verified and run.consistency_checks["cross_file"] == "ok"
    assert len(run.content_key) == 64


# --------------------------------------------------------------------------- F2 cross-file validation


def test_unknown_selected_model_is_controlled_error(run_copy):
    _edit_manifest(run_copy, lambda m: m["selection"].__setitem__("selected_model", "Unknown Model"))
    with pytest.raises(core.UILoadError, match="not a known learned model"):
        core.load_run(run_copy)


def test_selected_model_must_agree_across_files(run_copy):
    _edit_manifest(run_copy, lambda m: m["selection"].__setitem__("selected_model", RF))
    with pytest.raises(core.UILoadError, match="disagree on the selected model"):
        core.load_run(run_copy)


@pytest.mark.parametrize("value", [2.0, -0.1, float("nan"), "0.5", None])
def test_threshold_outside_unit_interval_is_rejected(run_copy, value):
    _edit_manifest(run_copy, lambda m: m["threshold_policy"]["decisions"][LR].__setitem__("threshold", value))
    with pytest.raises(core.UILoadError):
        core.load_run(run_copy)


def test_claimed_decision_metrics_are_recomputed(run_copy):
    _edit_manifest(run_copy, lambda m: m["threshold_policy"]["decisions"][LR].__setitem__("validation_recall", 0.42))
    with pytest.raises(core.UILoadError, match="Recomputed validation recall"):
        core.load_run(run_copy)


def test_run_id_and_source_must_agree_across_files(run_copy):
    summary = json.loads((run_copy / "analysis_summary.json").read_text(encoding="utf-8"))
    summary["run_id"] = "someone_else"
    (run_copy / "analysis_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _rehash(run_copy, "analysis_summary.json")
    with pytest.raises(core.UILoadError, match="run_id differs"):
        core.load_run(run_copy)


def test_source_flip_in_manifest_alone_is_rejected(run_copy):
    _edit_manifest(run_copy, lambda m: m["dataset"].__setitem__("source", "openml"))
    with pytest.raises(core.UILoadError, match="disagree on the dataset source"):
        core.load_run(run_copy)


def test_feature_importance_must_cover_manifest_features(run_copy):
    table = pd.read_csv(run_copy / "feature_importance.csv")
    table.iloc[0, 0] = "V99"
    table.to_csv(run_copy / "feature_importance.csv", index=False)
    _rehash(run_copy, "feature_importance.csv")
    with pytest.raises(core.UILoadError, match="exactly the manifest features"):
        core.load_run(run_copy)


def test_validation_scores_must_match_split_summary(run_copy):
    scores = pd.read_csv(run_copy / "validation_scores.csv", float_precision="round_trip")
    scores = scores.iloc[:-1]
    scores.to_csv(run_copy / "validation_scores.csv", index=False)
    _rehash(run_copy, "validation_scores.csv")
    with pytest.raises(core.UILoadError):
        core.load_run(run_copy)


def test_substituted_linear_spec_with_valid_checksum_fails_semantic_check(run_copy):
    """Coefficients replaced, digest updated: structure passes, prediction agreement fails."""
    spec_path = run_copy / "models" / "logistic_regression_linear.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["coef"] = [0.0] * len(spec["coef"])
    spec["intercept"] = 0.0
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    _rehash(run_copy, "models/logistic_regression_linear.json")
    run = core.load_run(run_copy)
    assert run.verified  # checksums are consistent by construction...
    parsed = core.load_linear_explainer(run)  # ...and the structure is valid...
    rows = core.load_dataset_rows(run, run.validation_scores["row_id"].head(20).tolist())
    with pytest.raises(core.UILoadError, match="does not reproduce the stored scores"):
        core.verify_linear_explainer(run, parsed, rows)  # ...but the semantics are not


def test_genuine_linear_spec_passes_semantic_check(layout_run):
    run = core.load_run(layout_run)
    spec = core.load_linear_explainer(run)
    rows = core.load_dataset_rows(run, run.validation_scores["row_id"].head(50).tolist())
    result = core.verify_linear_explainer(run, spec, rows)
    assert result["rows_checked"] == 50 and result["max_abs_difference"] <= core.LINEAR_SPEC_SCORE_TOLERANCE


def test_semantic_check_needs_stored_lr_scores_and_rows(layout_run):
    run = core.load_run(layout_run)
    spec = core.load_linear_explainer(run)
    with pytest.raises(core.UILoadError, match="No checksum-verified feature rows"):
        core.verify_linear_explainer(run, spec, pd.DataFrame(columns=spec.features))


# --------------------------------------------------------------------------- F3 provenance


def test_kaggle_layout_fixture_is_declared_source_not_historical(layout_run):
    run = core.load_run(layout_run)
    provenance, label, comparison = core.classify_provenance(run, core.HISTORICAL_DIR)
    assert provenance == core.Provenance.DECLARED_SOURCE_RUN
    assert "NOT verified" in label and "kaggle" in label
    assert comparison == {}
    assert "declared source" in core.frozen_test_label(provenance)
    assert "historical" not in core.frozen_test_label(provenance)


def test_synthetic_run_classifies_as_synthetic(tmp_path):
    result = make_run(tmp_path, "syn", source="synthetic")
    run = core.load_run(result.run_dir)
    assert core.classify_provenance(run)[0] == core.Provenance.SYNTHETIC


def test_recognition_requires_documented_digest_and_schema():
    manifest = {"dataset": {"source": "kaggle", "sha256": core.HISTORICAL_DATASET_SHA256}, "features": list(core.HISTORICAL_FEATURES)}
    assert core.recognizes_historical_dataset(manifest)
    assert not core.recognizes_historical_dataset({**manifest, "dataset": {"source": "kaggle", "sha256": "0" * 64}})
    assert not core.recognizes_historical_dataset({**manifest, "features": core.HISTORICAL_FEATURES[1:]})
    assert not core.recognizes_historical_dataset({**manifest, "dataset": {"source": "openml", "sha256": core.HISTORICAL_DATASET_SHA256}})


def test_self_declared_reproduction_flag_is_not_trusted(run_copy):
    """Even with the documented digest injected, a run is not 'historical' without configuration + comparison."""
    def mutate(m):
        m["dataset"]["sha256"] = core.HISTORICAL_DATASET_SHA256
        m["reproduced"] = True
    _edit_manifest(run_copy, mutate)
    dq = json.loads((run_copy / "data_quality.json").read_text(encoding="utf-8"))
    dq["sha256"] = core.HISTORICAL_DATASET_SHA256
    (run_copy / "data_quality.json").write_text(json.dumps(dq), encoding="utf-8")
    _rehash(run_copy, "data_quality.json")
    run = core.load_run(run_copy)
    provenance, _, comparison = core.classify_provenance(run, core.HISTORICAL_DIR)
    assert provenance != core.Provenance.HISTORICAL_REPRODUCTION
    if core.HISTORICAL_DIR.is_dir():
        assert comparison.get("passed") is False


def test_historical_configuration_check():
    manifest = {"config": {**core.HISTORICAL_CONFIG, "model": dict(core.HISTORICAL_MODEL_CONFIG)}}
    assert core.uses_historical_configuration(manifest)
    manifest["config"]["model"]["rf_n_estimators"] = 16
    assert not core.uses_historical_configuration(manifest)


def test_export_and_report_carry_provenance(layout_run):
    run = core.load_run(layout_run)
    provenance, label, _ = core.classify_provenance(run)
    config = core.export_config("Reproduced Experiment", run, LR, 0.5, [], None, None, provenance=provenance, provenance_label=label)
    assert config["provenance"] == "declared_source_run" and "NOT verified" in config["provenance_label"]
    report = core.render_report("Reproduced Experiment", run.source, run.data_quality, run.model_comparison, LR, 0.5,
                                provenance_label=label, frozen_label=core.frozen_test_label(provenance))
    assert "NOT verified" in report and "## Test-set results (frozen test results for this run" in report
    assert "untouched historical" not in report


# --------------------------------------------------------------------------- F4 workspace paths


def test_workspace_root_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv(workspace.ROOT_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    assert workspace.workspace_root() == tmp_path.resolve()
    monkeypatch.setenv(workspace.ROOT_ENV_VAR, str(tmp_path / "env"))
    assert workspace.workspace_root() == (tmp_path / "env").resolve()
    assert workspace.workspace_root(tmp_path / "explicit") == (tmp_path / "explicit").resolve()
    assert workspace.default_output_root(tmp_path / "x") == (tmp_path / "x" / "output" / "runs").resolve()
    package_dir = Path(core.__file__).resolve().parents[1]
    assert not workspace.default_output_root(tmp_path / "x").is_relative_to(package_dir)


def test_default_output_root_is_cwd_not_package_location(monkeypatch, tmp_path):
    monkeypatch.delenv(workspace.ROOT_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    csv = write_synthetic_csv(tmp_path / "d.csv", n_rows=400, fraud_rate=0.05, seed=1)
    config = RunConfig(dataset_path=csv, source="synthetic", run_id="cwd_run", model=ModelConfig(rf_n_estimators=4, rf_max_depth=3), figures=False)
    assert config.resolved_output_root() == (tmp_path / "output" / "runs").resolve()
    result = run_pipeline(config)
    assert result.run_dir == (tmp_path / "output" / "runs" / "cwd_run").resolve()
    assert Path(result.manifest["dataset"]["path"]).is_absolute()
    assert result.manifest["code"]["package_version"]


def test_git_attribution_only_inside_checkout(tmp_path):
    from fraud_pipeline.manifest import git_info

    info = git_info(tmp_path)  # not a git checkout
    assert info["commit"] is None and "not a git checkout" in info["note"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    info = git_info(tmp_path)  # a checkout that does not contain the running package
    assert info["commit"] is None and "not tracked source of the git checkout" in info["note"]
    assert info["package_version"] and Path(info["package_location"]).is_dir()


# --------------------------------------------------------------------------- F5 duplicates under keep


@pytest.fixture
def duplicated_csv(tmp_path) -> Path:
    frame = make_synthetic_frame(n_rows=600, fraud_rate=0.05, seed=3)
    frame = pd.concat([frame] + [frame.head(40)] * 4, ignore_index=True)  # 40 rows x 5 copies
    path = tmp_path / "dups.csv"
    frame.to_csv(path, index=False)
    return path


def test_keep_policy_duplicates_would_cross_all_three_splits(duplicated_csv):
    """The fixture reproduces the reviewer's observation: zero row-id overlap, identical content everywhere."""
    dataset = load_dataset(duplicated_csv, "kaggle", duplicate_policy="keep")
    assert dataset.info.duplicates_retained == 160
    splits = split_dataset(dataset.X, dataset.y, seed=42)
    assert splits.overlaps() == {"train_validation": 0, "train_test": 0, "validation_test": 0}
    overlap = splits.content_overlap()
    assert overlap["duplicate_content_groups_crossing_splits"] > 0
    assert overlap["max_splits_for_identical_content"] == 3
    assert splits.summary()["content_overlap"] == overlap


def test_run_pipeline_rejects_keep_with_retained_duplicates(duplicated_csv, tmp_path):
    with pytest.raises(ValueError, match="retained 160 exact duplicate rows.*drop_exact"):
        run_pipeline(RunConfig(duplicated_csv, "kaggle", tmp_path / "runs", "keep", duplicate_policy="keep",
                               model=ModelConfig(rf_n_estimators=4), figures=False))
    assert not (tmp_path / "runs").exists()


def test_drop_exact_default_is_unchanged_and_has_no_content_overlap(duplicated_csv, tmp_path):
    result = run_pipeline(RunConfig(duplicated_csv, "kaggle", tmp_path / "runs", "drop", model=ModelConfig(rf_n_estimators=4), figures=False))
    assert result.dataset.info.duplicates_removed == 160 and result.dataset.info.duplicates_retained == 0
    assert result.manifest["split"]["content_overlap"] == {
        "duplicate_content_groups_crossing_splits": 0, "max_splits_for_identical_content": 1,
    }


def test_keep_is_allowed_when_no_duplicates_exist(tmp_path):
    csv = write_synthetic_csv(tmp_path / "clean.csv", n_rows=400, fraud_rate=0.05, seed=2)
    result = run_pipeline(RunConfig(csv, "synthetic", tmp_path / "runs", "keep_ok", duplicate_policy="keep",
                                    model=ModelConfig(rf_n_estimators=4), figures=False))
    assert result.manifest["duplicate_policy"]["rows_retained"] == 0


def test_cli_reports_keep_rejection(duplicated_csv, tmp_path, capsys):
    from fraud_pipeline.cli import main

    code = main(["run", "--dataset", str(duplicated_csv), "--source", "kaggle", "--duplicate-policy", "keep",
                 "--output-root", str(tmp_path / "runs"), "--rf-estimators", "4", "--no-figures"])
    assert code == 2 and "drop_exact" in capsys.readouterr().err


# --------------------------------------------------------------------------- F6 run-id and artifact paths


@pytest.mark.parametrize(
    "run_id",
    ["../escape", "..\\escape", "a/b", "a\\b", "C:\\Temp\\x", "C:/x", "/abs", "\\\\server\\share", "..", ".hidden", "",
     "name.", "name ", "CON", "nul.txt", "x" * 101, "a b"],
)
def test_invalid_run_ids_are_rejected(tmp_path, run_id):
    with pytest.raises(ValueError):
        workspace.validate_run_id(run_id)
    with pytest.raises(ValueError):
        prepare_run_dir(tmp_path, run_id)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("run_id", ["historical_reproduction_2026-09-28_review", "run.1", "A-b_c.d", "20260928T120000Z_kaggle_76274b69"])
def test_valid_run_ids_are_contained(tmp_path, run_id):
    run_dir = prepare_run_dir(tmp_path, run_id)
    assert run_dir.parent == tmp_path.resolve() and run_dir.name == run_id
    assert (run_dir / "figures").is_dir()


def test_nonempty_run_dir_protection_preserved(tmp_path):
    prepare_run_dir(tmp_path, "r1")
    (tmp_path / "r1" / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_run_dir(tmp_path, "r1")


def test_cli_rejects_traversal_run_id(tmp_path, capsys):
    from fraud_pipeline.cli import main

    csv = write_synthetic_csv(tmp_path / "s.csv", n_rows=300, fraud_rate=0.05, seed=1)
    code = main(["run", "--dataset", str(csv), "--source", "synthetic", "--output-root", str(tmp_path / "runs"),
                 "--run-id", "..\\escaped", "--rf-estimators", "4", "--no-figures"])
    assert code == 2 and "Invalid run id" in capsys.readouterr().err
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize(
    "path",
    ["../x.csv", "..\\x.csv", "a\\..\\x", "/etc/passwd", "C:\\x", "c:/x", "\\\\server\\share", "//server/share", "a//b", "./a", "a/./b", "", "a\x00b"],
)
def test_unsafe_manifest_artifact_paths(path):
    assert not is_safe_artifact_path(path)
    with pytest.raises(ManifestError):
        validate_manifest_structure({
            "manifest_version": 1, "pipeline_version": "0.2.1", "run_id": "r", "dataset": {"sha256": "0" * 64},
            "features": ["Time"], "split": {}, "models": {}, "threshold_policy": {}, "selection": {}, "environment": {},
            "code": {}, "artifacts": {path: "0" * 64},
        })


def test_safe_manifest_artifact_paths():
    assert is_safe_artifact_path("figures/04_precision_recall_curves.png")
    assert is_safe_artifact_path("models/logistic_regression_linear.json")


@pytest.mark.skipif(sys.platform != "win32" and not hasattr(os, "symlink"), reason="symlink support")
def test_symlinked_file_escaping_run_dir_is_not_hashed(run_copy, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = run_copy / "figures" / "escape.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted for this user")
    from fraud_pipeline.manifest import artifact_checksums

    assert "figures/escape.txt" not in artifact_checksums(run_copy)


# --------------------------------------------------------------------------- F7 smaller repairs


def test_cli_dataset_byte_and_row_limits(tmp_path):
    csv = write_synthetic_csv(tmp_path / "s.csv", n_rows=300, fraud_rate=0.05, seed=1)
    with pytest.raises(DatasetTooLargeError, match="above the limit"):
        load_dataset(csv, "kaggle", max_bytes=1000)
    with pytest.raises(DatasetTooLargeError, match="more than 100 rows"):
        load_dataset(csv, "kaggle", max_rows=100)
    assert load_dataset(csv, "kaggle", max_rows=300).info.raw_rows == 300


def test_row_limit_is_applied_during_parsing(tmp_path, monkeypatch):
    csv = write_synthetic_csv(tmp_path / "s.csv", n_rows=300, fraud_rate=0.05, seed=1)
    seen = {}
    original = pd.read_csv

    def spy(path, **kwargs):
        seen["nrows"] = kwargs.get("nrows")
        return original(path, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    with pytest.raises(DatasetTooLargeError):
        load_dataset(csv, "kaggle", max_rows=50)
    assert seen["nrows"] == 51


def test_cli_limits_are_exposed(tmp_path, capsys):
    from fraud_pipeline.cli import main

    csv = write_synthetic_csv(tmp_path / "s.csv", n_rows=300, fraud_rate=0.05, seed=1)
    code = main(["run", "--dataset", str(csv), "--source", "synthetic", "--output-root", str(tmp_path / "runs"),
                 "--max-dataset-rows", "10", "--rf-estimators", "4", "--no-figures"])
    assert code == 2 and "more than 10 rows" in capsys.readouterr().err


def test_escape_markdown_neutralises_links_images_and_html():
    text = core.escape_markdown("![img](http://evil.example/x.png) [link](http://e) <img src=x> *bold* `code`\x07")
    assert "\x07" not in text
    assert text.startswith("\\!\\[img\\]\\(http://evil\\.example/x\\.png\\)")
    # every Markdown control character is escaped, so no link/image/HTML/emphasis can render
    for i, ch in enumerate(text):
        if ch in "![]()<>*`#_" and not (i > 0 and text[i - 1] == "\\"):
            pytest.fail(f"unescaped {ch!r} at {i} in {text!r}")
    assert core.escape_markdown("x" * 600).endswith("...")


def test_report_escapes_untrusted_metadata(layout_run):
    run = core.load_run(layout_run)
    dq = {**run.data_quality, "source_description": "![x](http://evil.example/p.png) [c](http://e)"}
    report = core.render_report("Reproduced Experiment", dq["source_description"], dq, run.model_comparison, LR, 0.5,
                                limitations=["[bad](http://e)"], manifest={**run.manifest, "run_id": "[r](http://e)"})
    assert "![x](" not in report and "[c](http://e)" not in report and "[bad](http://e)" not in report
    assert "\\[r\\]" in report

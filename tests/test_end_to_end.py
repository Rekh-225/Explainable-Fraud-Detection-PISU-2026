"""Tiny deterministic end-to-end run, manifest generation and artifact integrity."""

import json
from pathlib import Path

import pandas as pd
import pytest

from fraud_pipeline import __version__
from fraud_pipeline.artifacts import ArtifactIntegrityError, load_model_bundle
from fraud_pipeline.cli import main
from fraud_pipeline.manifest import MANIFEST_NAME, verify_run
from fraud_pipeline.modeling import ModelConfig
from fraud_pipeline.pipeline import RunConfig, run_pipeline

SMALL_MODEL = ModelConfig(seed=42, rf_n_estimators=8, rf_max_depth=4)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from fraud_pipeline.synthetic import write_synthetic_csv

    root = tmp_path_factory.mktemp("e2e")
    csv = write_synthetic_csv(root / "synthetic.csv", n_rows=800, fraud_rate=0.05, seed=3, n_duplicates=9)
    config = RunConfig(
        dataset_path=csv,
        source="synthetic",
        output_root=root / "runs",
        run_id="e2e",
        model=SMALL_MODEL,
        figures=False,
    )
    return run_pipeline(config)


EXPECTED_FILES = {
    "analysis_summary.json",
    "data_quality.json",
    "feature_importance.csv",
    "model_comparison.csv",
    "split_summary.json",
    "validation_summary.json",
    "validation_scores.csv",
    "models/selected_model.joblib",
    "models/selected_model.schema.json",
    "models/logistic_regression_linear.json",
    MANIFEST_NAME,
}


def test_all_artifacts_written_and_manifest_verifies(run):
    present = {p.relative_to(run.run_dir).as_posix() for p in run.run_dir.rglob("*") if p.is_file()}
    assert EXPECTED_FILES <= present
    assert verify_run(run.run_dir) == {"missing": [], "unexpected": [], "modified": [], "unlisted_required": []}
    assert set(run.manifest["artifacts"]) == present - {MANIFEST_NAME}


def test_manifest_contents(run):
    m = json.loads((run.run_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert m["pipeline_version"] == __version__ and m["run_id"] == "e2e"
    assert m["dataset"]["source"] == "synthetic" and len(m["dataset"]["sha256"]) == 64
    assert m["dataset"]["raw_rows"] == 809 and m["dataset"]["rows_after_duplicate_policy"] == 800
    assert m["duplicate_policy"] == {
        "policy": "drop_exact",
        "definition": "exact match on all retained feature columns and Class; first occurrence kept",
        "rows_removed": 9,
        "rows_retained": 0,
    }
    assert m["features"][0] == "Time" and len(m["features"]) == 30
    assert set(m["dataset"]["class_counts_after_duplicate_policy"]) == {"0", "1"}
    assert m["split"]["random_state"] == 42
    assert set(m["split"]["membership_fingerprints_sha256"]) == {"train", "validation", "test"}
    assert m["split"]["index_overlap"] == {"train_validation": 0, "train_test": 0, "validation_test": 0}
    assert m["models"]["Random Forest"]["params"]["n_estimators"] == 8
    assert m["models"]["Logistic Regression"]["params"]["model"]["params"]["class_weight"] == "balanced"
    assert m["threshold_policy"]["minimum_recall"] == 0.8 and m["threshold_policy"]["comparison"] == ">="
    assert m["selection"]["selected_model"] in ("Logistic Regression", "Random Forest")
    assert m["environment"]["packages"]["scikit-learn"] and m["environment"]["python"]
    assert "commit" in m["code"]
    assert m["config"]["dataset_path"].endswith("synthetic.csv")


def test_validation_export_has_row_ids_split_and_labels(run):
    table = pd.read_csv(run.run_dir / "validation_scores.csv")
    assert list(table.columns[:3]) == ["row_id", "split", "label"]
    assert (table["split"] == "validation").all()
    assert sorted(table["row_id"]) == sorted(run.splits.validation.row_ids)
    assert set(table["label"].unique()) <= {0, 1}
    assert {"score__random_forest", "flag__random_forest", "score__logistic_regression"} <= set(table.columns)
    assert table["score__random_forest"].between(0, 1).all()
    slug = run.selected_model.lower().replace(" ", "_")
    expected_flags = (table[f"score__{slug}"] >= run.thresholds[run.selected_model]).astype(int)
    assert table[f"flag__{slug}"].tolist() == expected_flags.tolist()


def test_model_comparison_has_default_and_operating_rows(run):
    results = run.results
    assert set(results["policy"]) == {"default_0.5", "validation_operating_point"}
    assert len(results) == 5
    assert {"average_precision", "pr_auc_trapezoidal", "roc_auc"} <= set(results.columns)
    dummy = results[results["model"] == "Dummy Prior"].iloc[0]
    assert dummy["recall"] == 0.0 and dummy["roc_auc"] == 0.5


def test_thresholds_come_from_validation_not_test(run):
    summary = json.loads((run.run_dir / "validation_summary.json").read_text())
    for name in ("Logistic Regression", "Random Forest"):
        decision = summary["threshold_selection"][name]
        assert decision["validation_recall"] >= 0.8
        assert decision["threshold"] == run.thresholds[name]


def test_model_bundle_requires_explicit_trust_opt_in(run):
    with pytest.raises(ArtifactIntegrityError, match="trusted_source=True"):
        load_model_bundle(run.run_dir)


def test_model_bundle_round_trip_and_schema_binding(run):
    bundle = load_model_bundle(run.run_dir, trusted_source=True)
    assert bundle.model_name == run.selected_model
    assert bundle.features == run.dataset.features
    assert bundle.run_id == "e2e"
    scores = bundle.score(run.splits.validation.X)
    assert len(scores) == len(run.splits.validation.y)
    with pytest.raises(ValueError, match="missing model features"):
        bundle.score(run.splits.validation.X.drop(columns=["V1"]))


def test_tampered_model_artifact_is_refused(run, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(run.run_dir, copy)
    model_path = copy / "models" / "selected_model.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"\x00")
    with pytest.raises(ArtifactIntegrityError, match="digest"):
        load_model_bundle(copy, trusted_source=True)
    report = verify_run(copy)
    assert report["modified"] == ["models/selected_model.joblib"]


def test_bare_joblib_without_schema_is_refused(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "selected_model.joblib").write_bytes(b"not a trusted pickle")
    with pytest.raises(ArtifactIntegrityError, match="only bundles written by fraud_pipeline"):
        load_model_bundle(tmp_path, trusted_source=True)


def test_runs_are_never_overwritten(run):
    with pytest.raises(FileExistsError):
        run_pipeline(
            RunConfig(
                dataset_path=Path(run.dataset.info.path),
                source="synthetic",
                output_root=run.run_dir.parent,
                run_id="e2e",
                model=SMALL_MODEL,
                figures=False,
            )
        )


def test_pipeline_is_deterministic(run, tmp_path):
    again = run_pipeline(
        RunConfig(
            dataset_path=Path(run.dataset.info.path),
            source="synthetic",
            output_root=tmp_path,
            run_id="again",
            model=SMALL_MODEL,
            figures=False,
        )
    )
    assert again.thresholds == run.thresholds
    assert again.manifest["split"]["membership_fingerprints_sha256"] == run.manifest["split"]["membership_fingerprints_sha256"]
    pd.testing.assert_frame_equal(again.results, run.results)
    a = pd.read_csv(again.run_dir / "validation_scores.csv")
    b = pd.read_csv(run.run_dir / "validation_scores.csv")
    pd.testing.assert_frame_equal(a, b)


def test_cli_synthetic_run_and_verify(tmp_path, capsys):
    csv = tmp_path / "s.csv"
    assert main(["synthetic", "--output", str(csv), "--rows", "500", "--seed", "9", "--fraud-rate", "0.06"]) == 0
    assert main([
        "run", "--dataset", str(csv), "--source", "synthetic", "--output-root", str(tmp_path / "runs"),
        "--run-id", "cli", "--rf-estimators", "5", "--no-figures",
    ]) == 0
    printed = capsys.readouterr().out
    assert '"run_id": "cli"' in printed and '"selected_model"' in printed
    assert main(["verify-run", str(tmp_path / "runs" / "cli")]) == 0
    assert (tmp_path / "runs" / "cli" / MANIFEST_NAME).is_file()
    assert not (tmp_path / "runs" / "cli" / "figures" / "01_class_imbalance.png").exists()


def test_cli_reproduce_historical_fails_clearly_without_dataset(tmp_path, capsys):
    code = main(["reproduce-historical", "--dataset", str(tmp_path / "missing.csv"), "--output-root", str(tmp_path)])
    assert code == 2
    assert "OpenML mirror is NOT substituted" in capsys.readouterr().err


def test_cli_run_rejects_openml_layout_declared_as_kaggle(tmp_path, capsys):
    from fraud_pipeline.synthetic import write_synthetic_csv

    csv = write_synthetic_csv(tmp_path / "openml_like.csv", n_rows=300, seed=2, include_time=False)
    code = main(["run", "--dataset", str(csv), "--source", "kaggle", "--output-root", str(tmp_path / "runs"), "--no-figures"])
    assert code == 2
    assert "requires the Time column" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())

"""UI calculations against core metrics, loaders, error states, exports."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_pipeline import __version__
from fraud_pipeline.evaluation import evaluate_scores
from fraud_pipeline.manifest import MANIFEST_NAME, load_manifest, save_json
from fraud_pipeline.modeling import ModelConfig
from fraud_pipeline.pipeline import RunConfig, run_pipeline
from fraud_pipeline.synthetic import write_synthetic_csv
from fraud_pipeline.ui import core

REPO_HISTORICAL = Path(__file__).resolve().parents[1] / "output" / "analysis"


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> core.LoadedRun:
    root = tmp_path_factory.mktemp("ui")
    run_dir, created = core.ensure_synthetic_demo_run(root, rows=1500, seed=5)
    assert created
    return core.load_run(run_dir)


def _labels_scores(run: core.LoadedRun, model: str):
    return run.validation_scores["label"].to_numpy(), run.validation_scores[core.score_column(model)].to_numpy()


# --- loaders -----------------------------------------------------------------


def test_demo_run_is_reused_not_retrained(tmp_path):
    run_dir, created = core.ensure_synthetic_demo_run(tmp_path, rows=1000, seed=1)
    mtime = (run_dir / MANIFEST_NAME).stat().st_mtime
    again, created_again = core.ensure_synthetic_demo_run(tmp_path, rows=1000, seed=1)
    assert again == run_dir and created and not created_again
    assert (run_dir / MANIFEST_NAME).stat().st_mtime == mtime


def test_demo_row_bounds():
    with pytest.raises(ValueError):
        core.ensure_synthetic_demo_run(Path("unused"), rows=10, seed=1)


def test_load_run_reports_synthetic_and_models(demo):
    assert demo.is_synthetic and demo.source == "synthetic"
    assert set(demo.scored_models) == {"Logistic Regression", "Random Forest"}
    assert demo.selected_threshold == demo.threshold_for(demo.selected_model)
    assert demo.verification == {"missing": [], "unexpected": [], "modified": [], "unlisted_required": []}
    assert demo.verified


def test_reproduced_mode_refuses_synthetic_run(demo):
    with pytest.raises(core.UILoadError, match="Synthetic Demo mode"):
        core.load_reproduced_run(demo.run_dir)


def test_missing_manifest_is_an_error(tmp_path):
    with pytest.raises(core.UILoadError, match="no run_manifest.json"):
        core.load_run(tmp_path)


def test_modified_artifact_is_refused(demo, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(demo.run_dir, copy)
    (copy / "model_comparison.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(core.UILoadError, match="verification failed"):
        core.load_run(copy)


def test_missing_required_file_is_refused(demo, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(demo.run_dir, copy)
    (copy / "validation_scores.csv").unlink()
    with pytest.raises(core.UILoadError):
        core.load_run(copy)


def test_incompatible_manifest_version_is_refused(demo, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(demo.run_dir, copy)
    manifest = load_manifest(copy)
    manifest["manifest_version"] = 99
    save_json(copy / MANIFEST_NAME, manifest)
    with pytest.raises(core.UILoadError, match="Manifest version"):
        core.load_run(copy)


def test_validation_scores_from_other_split_are_refused(demo, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(demo.run_dir, copy)
    scores = pd.read_csv(copy / "validation_scores.csv")
    scores.loc[0, "split"] = "test"
    scores.to_csv(copy / "validation_scores.csv", index=False)
    manifest = load_manifest(copy)
    manifest["artifacts"]["validation_scores.csv"] = core.sha256_file(copy / "validation_scores.csv")
    save_json(copy / MANIFEST_NAME, manifest)
    with pytest.raises(core.UILoadError, match="other than validation"):
        core.load_run(copy)


def test_compatibility_note_when_pipeline_version_differs(demo, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(demo.run_dir, copy)
    manifest = load_manifest(copy)
    manifest["pipeline_version"] = "9.9.0"
    save_json(copy / MANIFEST_NAME, manifest)
    run = core.load_run(copy)
    assert any("9.9.0" in note and __version__ in note for note in run.compatibility_notes)


def test_list_runs_only_returns_manifest_dirs(tmp_path, demo):
    (tmp_path / "junk").mkdir()
    shutil.copytree(demo.run_dir, tmp_path / "real")
    assert [p.name for p in core.list_runs(tmp_path)] == ["real"]
    assert core.list_runs(tmp_path / "missing") == []


def test_dataset_rows_are_checksum_verified(demo, tmp_path):
    rows = core.load_dataset_rows(demo, [0, 1, 2])
    assert list(rows.columns) == demo.manifest["features"] and len(rows) == 3
    copy = tmp_path / "copy"
    shutil.copytree(demo.run_dir, copy)

    def _set_dataset_sha(value: str) -> None:
        # Keep manifest and data_quality.json consistent so only the *file* check can fail.
        dq = json.loads((copy / "data_quality.json").read_text(encoding="utf-8"))
        dq["sha256"] = value
        (copy / "data_quality.json").write_text(json.dumps(dq), encoding="utf-8")
        manifest = load_manifest(copy)
        manifest["dataset"]["sha256"] = value
        manifest["artifacts"]["data_quality.json"] = core.sha256_file(copy / "data_quality.json")
        save_json(copy / MANIFEST_NAME, manifest)

    _set_dataset_sha("0" * 64)
    with pytest.raises(core.UILoadError, match="checksum differs"):
        core.load_dataset_rows(core.load_run(copy), [0])
    manifest = load_manifest(copy)
    manifest["dataset"]["path"] = str(tmp_path / "gone.csv")
    save_json(copy / MANIFEST_NAME, manifest)
    with pytest.raises(core.UILoadError, match="not found"):
        core.load_dataset_rows(core.load_run(copy), [0])


def test_inconsistent_dataset_checksum_between_files_is_rejected(demo, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(demo.run_dir, copy)
    manifest = load_manifest(copy)
    manifest["dataset"]["sha256"] = "0" * 64
    save_json(copy / MANIFEST_NAME, manifest)
    with pytest.raises(core.UILoadError, match="disagree on the dataset checksum"):
        core.load_run(copy)


def test_ui_core_exposes_no_joblib_loader(demo):
    assert not hasattr(core, "load_run_bundle")
    assert "joblib" not in {m.__name__ for m in vars(core).values() if hasattr(m, "__name__") and hasattr(m, "__file__")}
    spec = core.load_linear_explainer(demo)
    assert spec.features == demo.manifest["features"] and spec.run_id == demo.run_id


# --- historical ---------------------------------------------------------------


@pytest.mark.skipif(not REPO_HISTORICAL.is_dir(), reason="committed historical artifacts not present")
def test_historical_loader_reports_missing_provenance():
    hist = core.load_historical(REPO_HISTORICAL)
    assert hist.selected_model == "Random Forest"
    assert hist.selected_threshold == pytest.approx(0.6735057931524799)
    text = " ".join(hist.provenance_gaps)
    assert "No run manifest" in text and "validation predictions" in text
    assert hist.missing_files == []


def test_historical_loader_fails_on_incomplete_dir(tmp_path):
    with pytest.raises(core.UILoadError, match="not found"):
        core.load_historical(tmp_path / "nope")
    (tmp_path / "model_comparison.csv").write_text("model\n", encoding="utf-8")
    with pytest.raises(core.UILoadError, match="incomplete"):
        core.load_historical(tmp_path)


# --- explorer calculations ------------------------------------------------------


@pytest.mark.parametrize("threshold", [0.05, 0.3, 0.5, 0.9])
def test_threshold_summary_matches_core_evaluation(demo, threshold):
    model = "Random Forest"
    y, s = _labels_scores(demo, model)
    summary = core.threshold_summary(model, y, s, threshold)
    ref = evaluate_scores(model, "x", pd.Series(y), s, threshold)
    assert summary.true_positives == ref["true_positives"]
    assert summary.false_positives == ref["false_positives"]
    assert summary.false_negatives == ref["false_negatives"] == summary.missed_fraud
    assert summary.precision == ref["precision"] and summary.recall == ref["recall"]
    assert summary.alerts == int((s >= threshold).sum())
    assert summary.alert_rate == ref["alert_rate"]
    assert summary.fraud_total == int(y.sum()) and summary.rows == len(y)


def test_explorer_at_selected_threshold_matches_manifest_decision(demo):
    for model in demo.scored_models:
        y, s = _labels_scores(demo, model)
        decision = demo.manifest["threshold_policy"]["decisions"][model]
        summary = core.threshold_summary(model, y, s, decision["threshold"])
        assert summary.precision == pytest.approx(decision["validation_precision"])
        assert summary.recall == pytest.approx(decision["validation_recall"])


def test_explorer_uses_validation_rows_only(demo):
    validation_ids = set(demo.validation_scores["row_id"])
    split = demo.split_summary
    assert len(validation_ids) == split["validation"]["rows"]
    assert int(demo.validation_scores["label"].sum()) == split["validation"]["fraud"]
    assert (demo.validation_scores["split"] == "validation").all()
    test_rows = demo.model_comparison["test_rows"].iloc[0]
    assert test_rows == split["test"]["rows"] != len(validation_ids)


def test_threshold_curve_alert_counts(demo):
    y, s = _labels_scores(demo, "Random Forest")
    curve = core.threshold_curve(y, s)
    for t, alerts in zip(curve["threshold"].head(20), curve["alerts"].head(20)):
        assert alerts == int((s >= t).sum())
    assert curve["alerts"].is_monotonic_decreasing


def test_capacity_scenarios_empty_input_yields_structured_empty_table(demo):
    y, s = _labels_scores(demo, "Random Forest")
    table = core.capacity_scenarios("Random Forest", y, s, core.parse_capacities("   ", len(y)))
    assert len(table) == 0
    assert list(table.columns[: len(core.CAPACITY_COLUMNS)]) == core.CAPACITY_COLUMNS
    assert table[core.CAPACITY_COLUMNS].empty  # the UI's column selection no longer raises KeyError
    assert table.attrs["model"] == "Random Forest"


def test_capacity_scenarios(demo):
    y, s = _labels_scores(demo, "Random Forest")
    table = core.capacity_scenarios("Random Forest", y, s, [1, 5, 20, 10_000])
    assert list(table["review_capacity"]) == [1, 5, 20, 10_000]
    assert (table["alerts"] >= table["review_capacity"].clip(upper=len(y))).all()
    assert table.iloc[-1]["alerts"] == len(y) and table.iloc[-1]["recall"] == 1.0
    assert table["recall"].is_monotonic_increasing
    with pytest.raises(ValueError):
        core.threshold_for_capacity(s, 0)


def test_capacity_with_ties_reports_realised_alerts():
    scores = np.array([0.9, 0.9, 0.9, 0.1])
    t = core.threshold_for_capacity(scores, 2)
    assert t == 0.9 and int((scores >= t).sum()) == 3


def test_parse_capacities():
    assert core.parse_capacities("10, 5;5, 200", maximum=100) == [5, 10, 100]
    assert core.parse_capacities("", 100) == [] and core.parse_capacities("  , ;", 100) == []
    assert core.parse_capacities(" 7 ,\n 3", 100) == [3, 7]
    for bad in ("10, -1", "ten", "0", "1.5", "1e3", "5 5", "\u0663"):
        with pytest.raises(ValueError, match="positive whole numbers"):
            core.parse_capacities(bad, 100)


# --- cost scenario --------------------------------------------------------------


def test_cost_scenario_requires_explicit_inputs_and_is_labelled(demo):
    y, s = _labels_scores(demo, "Random Forest")
    summary = core.threshold_summary("Random Forest", y, s, 0.5)
    scenario = core.cost_scenario(summary, core.CostInputs(2.0, 100.0, 80.0, "EUR"))
    assert scenario["kind"] == "hypothetical_scenario" and "not realized" in scenario["disclaimer"]
    assert scenario["review_cost_total"] == summary.alerts * 2.0
    assert scenario["missed_fraud_loss_total"] == summary.false_negatives * 100.0
    assert scenario["recovered_total"] == summary.true_positives * 80.0
    assert scenario["net_scenario_value"] == pytest.approx(
        scenario["recovered_total"] - scenario["review_cost_total"] - scenario["missed_fraud_loss_total"]
    )
    with pytest.raises(ValueError):
        core.cost_scenario(summary, core.CostInputs(-1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        core.cost_scenario(summary, core.CostInputs(float("nan"), 0.0, 0.0))


# --- review queue and annotations ----------------------------------------------------


def test_review_queue_is_sorted_flagged_validation_rows(demo):
    model = "Random Forest"
    threshold = 0.3
    queue = core.build_review_queue(demo, model, threshold, limit=10)
    y, s = _labels_scores(demo, model)
    assert len(queue) == min(10, int((s >= threshold).sum()))
    assert (queue["model_score"] >= threshold).all()
    assert queue["model_score"].is_monotonic_decreasing
    assert list(queue["queue_rank"]) == list(range(1, len(queue) + 1))
    assert (queue["split"] == "validation").all()
    assert core.GROUND_TRUTH_COLUMN in queue.columns and "label" not in queue.columns
    assert set(queue["row_id"]) <= set(demo.validation_scores["row_id"])


def test_review_queue_unknown_model(demo):
    with pytest.raises(core.UILoadError):
        core.build_review_queue(demo, "Gradient Boosting", 0.5)


def test_annotations_stay_separate_from_ground_truth(demo):
    queue = core.build_review_queue(demo, "Random Forest", 0.2, limit=5)
    store = core.AnnotationStore()
    first, second = int(queue["row_id"].iloc[0]), int(queue["row_id"].iloc[1])
    store.annotate(first, "confirmed_fraud", "looks bad")
    store.annotate(second, "legitimate")
    merged = core.queue_with_annotations(queue, store)
    assert merged.loc[merged["row_id"] == first, "analyst_annotation"].item() == "confirmed_fraud"
    assert merged.loc[merged["row_id"] == first, core.GROUND_TRUTH_COLUMN].item() == queue[core.GROUND_TRUTH_COLUMN].iloc[0]
    assert (merged["analyst_annotation"].iloc[2:] == "unreviewed").all()
    agreement = core.annotation_agreement(queue, store)
    assert agreement["reviewed"] == 2 and agreement["agree_with_ground_truth"] + agreement["disagree_with_ground_truth"] == 2
    store.annotate(first, "unreviewed")
    assert store.get(first) == "unreviewed" and store.summary()["confirmed_fraud"] == 0
    with pytest.raises(ValueError):
        store.annotate(first, "fraudulent")
    frame = store.frame()
    assert list(frame.columns) == ["row_id", "annotation", "note", "annotated_at_utc"]
    assert core.GROUND_TRUTH_COLUMN not in frame.columns


def test_linear_contributions_match_fitted_logistic_regression(demo):
    """Reconstructed log-odds from the exported numeric spec equal the fitted model's decision_function."""
    from fraud_pipeline.modeling import LOGISTIC_MODEL, ModelConfig, build_models
    from fraud_pipeline.splitting import split_dataset
    from fraud_pipeline.data import load_dataset

    dataset = load_dataset(demo.dataset_path(), "synthetic")
    splits = split_dataset(dataset.X, dataset.y, seed=demo.manifest["config"]["seed"])
    lr = build_models(ModelConfig(seed=demo.manifest["config"]["seed"]))[LOGISTIC_MODEL].fit(splits.train.X, splits.train.y)
    spec = core.load_linear_explainer(demo)
    rows = core.load_dataset_rows(demo, demo.validation_scores["row_id"].head(5).tolist())
    for row_id, row in rows.iterrows():
        contributions = core.linear_contributions(spec, row)
        assert contributions.attrs["log_odds"] == pytest.approx(lr.decision_function(rows.loc[[row_id]])[0], abs=1e-9)
        displayed = demo.validation_scores.set_index("row_id").loc[row_id, core.score_column(LOGISTIC_MODEL)]
        assert contributions.attrs["score"] == pytest.approx(displayed, abs=1e-9)
        assert set(contributions["feature"]) == set(demo.manifest["features"])
        assert contributions["description"].isin(["anonymized PCA component", "time", "amount"]).all()
    assert contributions.attrs["model"] == LOGISTIC_MODEL


def test_explanation_availability_is_bound_to_displayed_model(demo):
    ok, message = core.explanation_availability(demo, "Random Forest")
    assert not ok and "Random Forest" in message and "not a causal explanation" in message
    ok, message = core.explanation_availability(demo, "Logistic Regression")
    assert ok
    demo_without = core.LoadedRun(**{**demo.__dict__, "linear_spec_path": None})
    ok, message = core.explanation_availability(demo_without, "Logistic Regression")
    assert not ok and "re-run the pipeline" in message
    with pytest.raises(core.UILoadError, match="no verified"):
        core.load_linear_explainer(demo_without)


# --- exports ---------------------------------------------------------------------


def test_export_config_and_report(demo):
    model = "Random Forest"
    y, s = _labels_scores(demo, model)
    summary = core.threshold_summary(model, y, s, 0.4)
    store = core.AnnotationStore()
    store.annotate(int(demo.validation_scores["row_id"].iloc[0]), "escalate")
    cost = core.cost_scenario(summary, core.CostInputs(1.0, 50.0, 40.0))
    capacity = core.capacity_scenarios(model, y, s, [5, 10])
    config = core.export_config("Synthetic Demo", demo, model, 0.4, [5, 10], core.CostInputs(1.0, 50.0, 40.0), store)
    assert config["synthetic"] is True and config["explored_validation_threshold"] == 0.4
    assert config["validation_selected_threshold"] == demo.threshold_for(model)
    assert config["annotation_counts"]["escalate"] == 1
    json.dumps(config)  # serialisable

    report = core.render_report(
        "Synthetic Demo", "synthetic fixture", demo.data_quality, demo.model_comparison, demo.selected_model,
        demo.selected_threshold, explorer=summary, capacity_table=capacity, cost=cost, annotations=store,
        manifest=demo.manifest, limitations=["l1"],
    )
    assert "SYNTHETIC DATA" in report
    assert "not a newly validated operating point" in report
    assert "not realized" in report
    assert f"{summary.alerts} alerts" in report
    assert "| model | policy |" in report and "Random Forest" in report
    assert demo.manifest["dataset"]["sha256"] in report
    assert core.FEATURE_DISCLAIMER in report and core.IMPORTANCE_DISCLAIMER in report
    assert "- l1" in report


def test_markdown_table_formats_floats():
    table = core.markdown_table(pd.DataFrame({"a": [1, 2], "b": [0.123456, 1.0]}), float_digits=2)
    assert table.splitlines()[0] == "| a | b |"
    assert "| 1 | 0.12 |" in table

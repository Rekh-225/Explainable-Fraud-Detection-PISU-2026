"""Headless Streamlit tests: modes, error states, state handling, deserialization boundary.

Runs the real app script through ``streamlit.testing.v1.AppTest``. Fixtures
come from ``ui_fixtures.make_run`` (generated data); nothing here touches the
Kaggle dataset or the frozen historical outputs except the read-only
Historical Results checks.
"""

import shutil
from pathlib import Path

import joblib
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from fraud_pipeline.data import sha256_file  # noqa: E402
from fraud_pipeline.manifest import MANIFEST_NAME, load_manifest, save_json  # noqa: E402
from fraud_pipeline.ui import core  # noqa: E402
from ui_fixtures import make_run  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "fraud_pipeline" / "ui" / "app.py")
REPO_HISTORICAL = Path(__file__).resolve().parents[1] / "output" / "analysis"
LR, RF = "Logistic Regression", "Random Forest"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUD_UI_DEMO_ROOT", str(tmp_path / "demo"))
    monkeypatch.setenv("FRAUD_UI_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("FRAUD_UI_HISTORICAL_DIR", str(REPO_HISTORICAL))
    (tmp_path / "runs").mkdir()
    return tmp_path


@pytest.fixture
def no_deserialization(monkeypatch):
    """Sentinel: any joblib.load during the test fails loudly (no payload is ever executed)."""
    calls = []

    def sentinel(*args, **kwargs):
        calls.append(args)
        raise AssertionError("joblib.load must not be called by the UI")

    monkeypatch.setattr(joblib, "load", sentinel)
    return calls


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def _texts(elements) -> str:
    return " ".join(str(getattr(e, "value", getattr(e, "label", ""))) for e in elements)


def _open_reproduced(at: AppTest) -> AppTest:
    at.sidebar.radio[0].set_value("Reproduced Experiment").run()
    return at


def _kaggle_layout_run(env, run_id="layout_fixture", winner=LR, seed=1):
    """A run whose manifest says source=kaggle but whose rows are generated (see ui_fixtures)."""
    return make_run(env / "src", run_id, source="kaggle", seed=seed, winner=winner, figures=True), _install(env, run_id)


def _install(env, run_id):
    src = env / "src" / "runs" / run_id
    dst = env / "runs" / run_id
    shutil.copytree(src, dst)
    return dst


# --------------------------------------------------------------------------- historical


@pytest.mark.skipif(not REPO_HISTORICAL.is_dir(), reason="committed historical artifacts not present")
def test_historical_mode_shows_frozen_results_and_gaps(env, no_deserialization):
    at = _app()
    assert not at.exception and not at.error
    assert "Historical Results" in at.header[0].value
    assert "No run manifest" in _texts(at.markdown)
    assert "no threshold explorer or review queue" in _texts(at.warning)
    assert "Threshold explorer" not in [t.label for t in at.tabs]
    assert "0.6735" in _texts(at.markdown)
    assert no_deserialization == []


def test_historical_mode_error_when_directory_missing(env, monkeypatch):
    monkeypatch.setenv("FRAUD_UI_HISTORICAL_DIR", str(env / "absent"))
    at = _app()
    assert any("not found" in e.value for e in at.error)
    assert not at.exception


# --------------------------------------------------------------------------- reproduced: errors


def test_reproduced_mode_without_runs_errors_and_does_not_fall_back(env):
    at = _open_reproduced(_app())
    assert not at.exception
    assert any("does not fall back" in e.value for e in at.error)
    assert not at.metric


def test_reproduced_mode_rejects_synthetic_run(env):
    make_run(env / "src", "syn", source="synthetic")
    _install(env, "syn")
    at = _open_reproduced(_app())
    assert any("Synthetic Demo mode" in e.value for e in at.error)
    assert not at.exception


def test_reproduced_mode_rejects_tampered_run(env):
    make_run(env / "src", "t", source="kaggle")
    target = _install(env, "t")
    (target / "feature_importance.csv").write_text("x\n", encoding="utf-8")
    at = _open_reproduced(_app())
    assert any("verification failed" in e.value for e in at.error)


def test_reproduced_mode_rejects_required_artifact_without_checksum(env):
    make_run(env / "src", "u", source="kaggle")
    target = _install(env, "u")
    manifest = load_manifest(target)
    del manifest["artifacts"]["model_comparison.csv"]
    save_json(target / MANIFEST_NAME, manifest)
    (target / "model_comparison.csv").write_text("model,policy\nx,y\n", encoding="utf-8")
    at = _open_reproduced(_app())
    assert any("no checksum entry" in e.value for e in at.error)
    assert not at.metric


def test_reproduced_mode_custom_path_with_malformed_manifest(env):
    bad = env / "elsewhere"
    bad.mkdir()
    (bad / MANIFEST_NAME).write_text("{oops", encoding="utf-8")
    at = _open_reproduced(_app())
    at.sidebar.text_input[0].set_value(str(bad)).run()
    assert any("Malformed manifest" in e.value for e in at.error)
    assert not at.exception


# --------------------------------------------------------------------------- reproduced: success path (F8)


def test_reproduced_mode_success_path(env, no_deserialization):
    """Full workflow on a verified run with the Kaggle column layout (test-only fixture).

    Provenance note: the fixture is generated data written with source="kaggle"
    so that the mode's mechanics (verification, browsing, exploring, queue,
    exports) can be exercised offline. It is not the ULB dataset, the UI must
    label it as a declared-source run, and this test makes no claim about the
    historical reproduction.
    """
    result, run_dir = _kaggle_layout_run(env)
    run = core.load_run(run_dir)
    at = _open_reproduced(_app())
    assert not at.exception and not at.error
    assert "Artifact checksums verified" in _texts(at.success)
    assert "SYNTHETIC DATA" not in _texts(at.warning)
    assert "checksum-verified: yes" in _texts(at.caption)
    assert [t.label for t in at.tabs][:3] == ["Overview", "Model comparison", "Threshold explorer"]

    # Frozen comparison table present with both policies.
    tables = [d.value for d in at.dataframe if "policy" in d.value.columns]
    assert tables and set(tables[0]["policy"]) == {"default_0.5", "validation_operating_point"}
    # Generated data declared as kaggle is labelled as such - never as the historical test set.
    text = _texts(at.markdown) + _texts(at.warning)
    assert "Frozen test result: frozen test results for this run (declared source, dataset identity not verified)" in text
    assert "untouched historical" not in text
    assert "dataset identity NOT verified" in _texts(at.warning)

    # Explorer metrics equal core at the validation-selected threshold.
    model = run.selected_model
    y = run.validation_scores["label"].to_numpy()
    s = run.validation_scores[core.score_column(model)].to_numpy()
    expected = core.threshold_summary(model, y, s, run.threshold_for(model))
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Alerts"] == f"{expected.alerts:,}" and metrics["Recall"] == f"{expected.recall:.3f}"

    # Review queue rows come from validation and match core.
    queue_tables = [d.value for d in at.dataframe if "queue_rank" in d.value.columns]
    assert queue_tables and (queue_tables[0]["split"] == "validation").all()
    expected_queue = core.build_review_queue(run, model, run.threshold_for(model), 25)
    assert queue_tables[0]["row_id"].tolist() == expected_queue["row_id"].tolist()

    # Exports are rendered for the active model/threshold.
    assert f"Exporting model **{model}** at validation threshold **{run.threshold_for(model):.6f}**" in _texts(at.markdown)
    codes = [c.value for c in at.code]
    assert any("Test-set results (frozen test results for this run" in c and "Provenance" in c for c in codes)
    assert no_deserialization == []


# --------------------------------------------------------------------------- F1: never deserialize


def test_browsing_a_run_never_calls_joblib_load(env, no_deserialization):
    """Open a run, walk every tab, load features, expand explanations - no joblib.load."""
    _, run_dir = _kaggle_layout_run(env, winner=RF)
    at = _open_reproduced(_app())
    assert not at.exception and not at.error
    run = core.load_run(run_dir)
    at.checkbox(key=f"features::{run.run_id}").check().run()
    at.checkbox(key=f"reveal::{run.run_id}").check().run()
    at.slider(key=f"threshold::{run.run_id}::{RF}").set_value(0.2).run()
    at.selectbox(key=f"model::{run.run_id}").select(LR).run()
    at.checkbox(key=f"ack::{run.run_id}").check().run()
    assert not at.exception
    assert no_deserialization == []


def test_externally_assembled_bundle_with_matching_checksums_is_never_loaded(env, no_deserialization):
    """A foreign joblib file with consistent sidecar and manifest digests reaches no loader."""
    _, run_dir = _kaggle_layout_run(env, run_id="foreign")
    model_path = run_dir / "models" / "selected_model.joblib"
    model_path.write_bytes(b"SENTINEL-NOT-A-PICKLE")
    digest = sha256_file(model_path)
    schema_path = run_dir / "models" / "selected_model.schema.json"
    schema = load_manifest(run_dir)  # reuse json loader for the sidecar
    import json

    sidecar = json.loads(schema_path.read_text(encoding="utf-8"))
    sidecar["model_sha256"] = digest
    schema_path.write_text(json.dumps(sidecar), encoding="utf-8")
    manifest = load_manifest(run_dir)
    manifest["artifacts"]["models/selected_model.joblib"] = digest
    manifest["artifacts"]["models/selected_model.schema.json"] = sha256_file(schema_path)
    save_json(run_dir / MANIFEST_NAME, manifest)
    assert core.load_run(run_dir).verified  # integrity checks pass by construction...
    at = _open_reproduced(_app())
    at.checkbox(key=f"features::foreign").check().run()
    assert not at.exception and not at.error
    assert no_deserialization == []  # ...yet nothing unpickles it


def test_app_module_has_no_pickle_imports():
    source = Path(APP).read_text(encoding="utf-8")
    core_source = Path(core.__file__).read_text(encoding="utf-8")
    for text in (source, core_source):
        assert "import joblib" not in text and "import pickle" not in text
        assert "load_model_bundle" not in text


# --------------------------------------------------------------------------- F3: capacities


def test_capacity_field_recovers_from_empty_and_invalid_input(env):
    _, run_dir = _kaggle_layout_run(env)
    run = core.load_run(run_dir)
    key = f"capacities::{run.run_id}"
    at = _open_reproduced(_app())
    cap_tables = lambda: [d.value for d in at.dataframe if "review_capacity" in d.value.columns]
    assert len(cap_tables()[0]) == 4

    at.text_input(key=key).set_value("").run()
    assert not at.exception
    assert "Enter at least one review capacity" in _texts(at.info)
    assert not cap_tables()
    assert "No valid review capacities are set" in _texts(at.caption)

    for bad in ("   ", "abc", "0", "-5, 10", "1.5"):
        at.text_input(key=key).set_value(bad).run()
        assert not at.exception, bad
        assert not cap_tables(), bad
    assert "positive whole numbers" in _texts(at.error)
    assert "No valid review capacities are set" in _texts(at.caption)

    at.text_input(key=key).set_value("5; 20,20").run()
    assert not at.exception and not at.error
    table = cap_tables()[0]
    assert table["review_capacity"].tolist() == [5, 20]
    assert "tied scores can produce more alerts" in _texts(at.caption)
    assert "No valid review capacities are set" not in _texts(at.caption)


# --------------------------------------------------------------------------- F4: threshold state


def test_reset_updates_slider_and_survives_other_interactions(env):
    _, run_dir = _kaggle_layout_run(env)
    run = core.load_run(run_dir)
    model = run.selected_model
    key = f"threshold::{run.run_id}::{model}"
    default = run.threshold_for(model)
    at = _open_reproduced(_app())
    assert at.slider(key=key).value == pytest.approx(default)

    at.slider(key=key).set_value(0.05).run()
    assert at.slider(key=key).value == 0.05
    y = run.validation_scores["label"].to_numpy()
    s = run.validation_scores[core.score_column(model)].to_numpy()
    moved = core.threshold_summary(model, y, s, 0.05)
    assert {m.label: m.value for m in at.metric}["Alerts"] == f"{moved.alerts:,}"
    assert f"threshold **0.0500**" in _texts(at.markdown)  # review queue header uses the same value

    at.button(key=f"reset::{run.run_id}::{model}").click().run()
    assert at.slider(key=key).value == pytest.approx(default)
    back = core.threshold_summary(model, y, s, default)
    assert {m.label: m.value for m in at.metric}["Alerts"] == f"{back.alerts:,}"

    at.slider(key=f"queue_limit::{run.run_id}").set_value(10).run()  # unrelated interaction
    assert at.slider(key=key).value == pytest.approx(default)
    assert f"threshold **{default:.4f}**" in _texts(at.markdown)


def test_threshold_state_is_separate_per_model_and_run(env):
    _, run_dir = _kaggle_layout_run(env, run_id="run_a", winner=RF)
    _, run_dir_b = _kaggle_layout_run(env, run_id="run_b", winner=LR, seed=2)
    run_a, run_b = core.load_run(run_dir), core.load_run(run_dir_b)
    at = _open_reproduced(_app())
    at.sidebar.selectbox[0].select("run_a").run()
    assert at.slider(key=f"threshold::run_a::{RF}").value == pytest.approx(run_a.threshold_for(RF))
    at.slider(key=f"threshold::run_a::{RF}").set_value(0.15).run()

    at.selectbox(key="model::run_a").select(LR).run()
    assert at.slider(key=f"threshold::run_a::{LR}").value == pytest.approx(run_a.threshold_for(LR))
    assert f"Model **{LR}**, threshold **{run_a.threshold_for(LR):.4f}**" in _texts(at.markdown)
    at.slider(key=f"threshold::run_a::{LR}").set_value(0.9).run()

    at.selectbox(key="model::run_a").select(RF).run()
    assert at.slider(key=f"threshold::run_a::{RF}").value == 0.15
    assert f"Model **{RF}**, threshold **0.1500**" in _texts(at.markdown)

    at.sidebar.selectbox[0].select("run_b").run()
    assert not at.exception and not at.error
    assert at.slider(key=f"threshold::run_b::{LR}").value == pytest.approx(run_b.threshold_for(LR))
    at.sidebar.selectbox[0].select("run_a").run()
    assert at.slider(key=f"threshold::run_a::{RF}").value == 0.15
    assert f"Model **{RF}**, threshold **0.1500**" in _texts(at.markdown)
    at.selectbox(key="model::run_a").select(LR).run()
    assert at.slider(key=f"threshold::run_a::{LR}").value == 0.9
    assert f"Model **{LR}**, threshold **0.9000**" in _texts(at.markdown)


def test_alerts_queue_cost_and_export_share_the_active_threshold(env):
    _, run_dir = _kaggle_layout_run(env)
    run = core.load_run(run_dir)
    model = run.selected_model
    at = _open_reproduced(_app())
    at.slider(key=f"threshold::{run.run_id}::{model}").set_value(0.3).run()
    at.checkbox(key=f"ack::{run.run_id}").check().run()
    at.number_input(key=f"cost_review::{run.run_id}").set_value(2.0).run()
    assert not at.exception
    y = run.validation_scores["label"].to_numpy()
    s = run.validation_scores[core.score_column(model)].to_numpy()
    summary = core.threshold_summary(model, y, s, 0.3)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Alerts"] == f"{summary.alerts:,}"
    assert metrics["Review cost total"].startswith(f"{summary.alerts * 2.0:,.2f}")
    text = _texts(at.markdown) + _texts(at.caption)
    assert f"Model **{model}**, threshold **0.3000**" in text
    assert f"at validation threshold 0.3000 for {model}" in text
    assert f"Exporting model **{model}** at validation threshold **0.300000**" in text
    queue = [d.value for d in at.dataframe if "queue_rank" in d.value.columns][0]
    assert (queue["model_score"] >= 0.3).all()
    assert len(queue) == min(25, summary.alerts)


# --------------------------------------------------------------------------- F5: explanations follow the displayed model


def _open_with_features(env, winner, run_id):
    _, run_dir = _kaggle_layout_run(env, run_id=run_id, winner=winner)
    at = _open_reproduced(_app())
    at.checkbox(key=f"features::{run_id}").check().run()
    return at, core.load_run(run_dir)


def test_rf_wins_but_lr_selected_explains_lr(env, no_deserialization):
    at, run = _open_with_features(env, RF, "rf_wins")
    assert "No local explanation is available for Random Forest" in _texts(at.markdown)
    at.selectbox(key="model::rf_wins").select(LR).run()
    assert not at.exception
    text = _texts(at.markdown)
    assert f"Local explanation for {LR}" in _texts(at.expander)
    assert "Additive log-odds terms of the Logistic Regression pipeline" in text
    assert "reconstructed score" in text and "= displayed score" in text and "Verified against" in text
    contributions = [d.value for d in at.dataframe if "log_odds_contribution" in d.value.columns]
    assert contributions and contributions[0]["description"].isin(["anonymized PCA component", "time", "amount"]).all()
    assert no_deserialization == []


def test_lr_wins_but_rf_selected_disables_explanation(env, no_deserialization):
    at, run = _open_with_features(env, LR, "lr_wins")
    assert "Additive log-odds terms" in _texts(at.markdown)
    at.selectbox(key="model::lr_wins").select(RF).run()
    assert not at.exception
    text = _texts(at.markdown)
    assert "No local explanation is available for Random Forest" in text
    assert "not a causal explanation" in text
    assert "Additive log-odds terms of the Logistic Regression pipeline" not in text
    assert f"Local explanation for {RF}" in _texts(at.expander)
    assert no_deserialization == []


def test_run_without_linear_spec_disables_explanation_honestly(env):
    _, run_dir = _kaggle_layout_run(env, run_id="old")
    (run_dir / "models" / "logistic_regression_linear.json").unlink()
    manifest = load_manifest(run_dir)
    del manifest["artifacts"]["models/logistic_regression_linear.json"]
    save_json(run_dir / MANIFEST_NAME, manifest)
    at = _open_reproduced(_app())
    at.checkbox(key="features::old").check().run()
    assert "re-run the pipeline" in _texts(at.markdown)


# --------------------------------------------------------------------------- synthetic


def test_synthetic_mode_creates_once_then_explores(env, no_deserialization):
    at = _app()
    at.sidebar.radio[0].set_value("Synthetic Demo").run()
    assert at.button[0].label == "Create synthetic demo run"
    at.sidebar.number_input[0].set_value(1000).run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert "SYNTHETIC DATA" in _texts(at.warning)
    assert "Frozen test result: frozen test results for this synthetic run" in _texts(at.markdown)
    run_dir = core.synthetic_demo_paths(env / "demo", 1000, 2026)[1]
    mtime = (run_dir / core.MANIFEST_NAME).stat().st_mtime
    at.run()
    assert (run_dir / core.MANIFEST_NAME).stat().st_mtime == mtime
    assert no_deserialization == []


# --------------------------------------------------------------------------- round-2 findings


def _same_size_flip(path: Path) -> None:
    import os

    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    stat = path.stat()
    path.write_bytes(bytes(data))
    os.utime(path, (stat.st_atime, stat.st_mtime))


def test_stale_cache_figure_changed_after_open_is_rejected(env):
    """Same-size, timestamp-preserving edit of a displayed figure after the run was opened."""
    import os

    _, run_dir = _kaggle_layout_run(env)
    at = _open_reproduced(_app())
    assert not at.error and "checksum-verified: yes" in _texts(at.caption)
    manifest = run_dir / MANIFEST_NAME
    stat = manifest.stat()
    _same_size_flip(run_dir / "figures" / "04_precision_recall_curves.png")
    os.utime(manifest, (stat.st_atime, stat.st_mtime))
    at.slider(key="queue_limit::layout_fixture").set_value(10).run()
    assert any("verification failed" in e.value and "04_precision_recall_curves.png" in e.value for e in at.error)
    assert "checksum-verified: yes" not in _texts(at.caption)
    assert not at.metric


def test_stale_cache_report_table_changed_after_open_is_rejected(env):
    import os

    _, run_dir = _kaggle_layout_run(env)
    at = _open_reproduced(_app())
    manifest = run_dir / MANIFEST_NAME
    stat = manifest.stat()
    _same_size_flip(run_dir / "model_comparison.csv")
    os.utime(manifest, (stat.st_atime, stat.st_mtime))
    at.run()
    assert any("model_comparison.csv" in e.value for e in at.error)
    assert not [d for d in at.dataframe if "policy" in d.value.columns]


def test_stale_cache_linear_spec_changed_after_open_is_rejected(env, no_deserialization):
    import os

    _, run_dir = _kaggle_layout_run(env)
    at = _open_reproduced(_app())
    at.checkbox(key="features::layout_fixture").check().run()
    assert "Verified against" in _texts(at.markdown)
    manifest = run_dir / MANIFEST_NAME
    stat = manifest.stat()
    _same_size_flip(run_dir / "models" / "logistic_regression_linear.json")
    os.utime(manifest, (stat.st_atime, stat.st_mtime))
    at.run()
    assert any("logistic_regression_linear.json" in e.value for e in at.error)
    assert "Verified against" not in _texts(at.markdown)


def test_substituted_linear_spec_is_disabled_in_ui(env, no_deserialization):
    import json

    _, run_dir = _kaggle_layout_run(env, run_id="subst")
    spec_path = run_dir / "models" / "logistic_regression_linear.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["coef"] = [0.0] * len(spec["coef"])
    spec["intercept"] = 0.0
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    manifest = load_manifest(run_dir)
    manifest["artifacts"]["models/logistic_regression_linear.json"] = sha256_file(spec_path)
    save_json(run_dir / MANIFEST_NAME, manifest)
    at = _open_reproduced(_app())
    assert not at.error  # viewing is allowed: the run is internally consistent
    at.checkbox(key="features::subst").check().run()
    assert any("Explanation disabled" in e.value and "does not reproduce the stored scores" in e.value for e in at.error)
    assert "Verified against" not in _texts(at.markdown)
    assert no_deserialization == []


def test_unknown_selected_model_is_a_controlled_error(env):
    _, run_dir = _kaggle_layout_run(env, run_id="unk")
    manifest = load_manifest(run_dir)
    manifest["selection"]["selected_model"] = "Unknown Model"
    save_json(run_dir / MANIFEST_NAME, manifest)
    at = _open_reproduced(_app())
    assert not at.exception
    assert any("not a known learned model" in e.value for e in at.error)


def test_annotation_note_is_keyed_per_row(env):
    _, run_dir = _kaggle_layout_run(env)
    run = core.load_run(run_dir)
    at = _open_reproduced(_app())
    queue = [d.value for d in at.dataframe if "queue_rank" in d.value.columns][0]
    first, second = int(queue["row_id"].iloc[0]), int(queue["row_id"].iloc[1])
    at.text_input(key=f"note::{run.run_id}::{first}").set_value("note for first").run()
    at.selectbox(key=f"row::{run.run_id}").select(second).run()
    assert at.text_input(key=f"note::{run.run_id}::{second}").value == ""
    at.text_input(key=f"note::{run.run_id}::{second}").set_value("second note").run()
    at.radio(key=f"choice::{run.run_id}::{second}").set_value("escalate").run()
    at.button(key=f"save::{run.run_id}").click().run()
    at.selectbox(key=f"row::{run.run_id}").select(first).run()
    assert at.text_input(key=f"note::{run.run_id}::{first}").value == "note for first"
    at.selectbox(key=f"row::{run.run_id}").select(second).run()
    assert at.text_input(key=f"note::{run.run_id}::{second}").value == "second note"


def test_untrusted_metadata_is_rendered_inert(env):
    """Markdown in run metadata must not render as links/images; the report preview is plain text."""
    import json

    _, run_dir = _kaggle_layout_run(env, run_id="md")
    payload = "![x](http://evil.example/p.png) [c](http://evil.example)"
    for name in ("data_quality.json", "analysis_summary.json"):
        data = json.loads((run_dir / name).read_text(encoding="utf-8"))
        target = data if name == "data_quality.json" else data["data_quality"]
        target["source_description"] = payload
        if name == "analysis_summary.json":
            data["limitations"] = [payload]
        (run_dir / name).write_text(json.dumps(data), encoding="utf-8")
    manifest = load_manifest(run_dir)
    for name in ("data_quality.json", "analysis_summary.json"):
        manifest["artifacts"][name] = sha256_file(run_dir / name)
    save_json(run_dir / MANIFEST_NAME, manifest)
    at = _open_reproduced(_app())
    assert not at.error
    rendered = _texts(at.markdown)
    assert payload not in rendered
    assert "\\!\\[x\\]" in rendered  # escaped, inert
    codes = [c.value for c in at.code]
    assert any("Fraud-detection analytical report" in c for c in codes)  # preview is st.code, not st.markdown
    assert not any("Fraud-detection analytical report" in m.value for m in at.markdown)

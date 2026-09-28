"""Headless Streamlit smoke tests for the three modes and their error states."""

import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from fraud_pipeline.ui import core  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "fraud_pipeline" / "ui" / "app.py")
REPO_HISTORICAL = Path(__file__).resolve().parents[1] / "output" / "analysis"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUD_UI_DEMO_ROOT", str(tmp_path / "demo"))
    monkeypatch.setenv("FRAUD_UI_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("FRAUD_UI_HISTORICAL_DIR", str(REPO_HISTORICAL))
    return tmp_path


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def _texts(elements) -> str:
    return " ".join(e.value for e in elements)


@pytest.mark.skipif(not REPO_HISTORICAL.is_dir(), reason="committed historical artifacts not present")
def test_historical_mode_shows_frozen_results_and_gaps(env):
    at = _app()
    assert not at.exception and not at.error
    assert "Historical Results" in at.header[0].value
    assert "No run manifest" in _texts(at.markdown)
    assert "no threshold explorer or review queue" in _texts(at.warning)
    assert "Threshold explorer" not in [t.label for t in at.tabs]
    assert "0.6735" in _texts(at.markdown)


def test_historical_mode_error_when_directory_missing(env, monkeypatch):
    monkeypatch.setenv("FRAUD_UI_HISTORICAL_DIR", str(env / "absent"))
    at = _app()
    assert any("not found" in e.value for e in at.error)
    assert not at.exception


def test_reproduced_mode_without_runs_errors_and_does_not_fall_back(env):
    at = _app()
    at.sidebar.radio[0].set_value("Reproduced Experiment").run()
    assert not at.exception
    assert any("does not fall back" in e.value for e in at.error)
    assert not at.metric


def test_reproduced_mode_rejects_synthetic_run(env):
    run_dir, _ = core.ensure_synthetic_demo_run(env / "runs_src", rows=800, seed=3)
    (env / "runs").mkdir()
    import shutil

    shutil.copytree(run_dir, env / "runs" / run_dir.name)
    at = _app()
    at.sidebar.radio[0].set_value("Reproduced Experiment").run()
    assert any("Synthetic Demo mode" in e.value for e in at.error)
    assert not at.exception


def test_reproduced_mode_rejects_tampered_run(env):
    run_dir, _ = core.ensure_synthetic_demo_run(env / "runs_src", rows=800, seed=3)
    import shutil

    target = env / "runs" / "tampered"
    shutil.copytree(run_dir, target)
    (target / "feature_importance.csv").write_text("x\n", encoding="utf-8")
    at = _app()
    at.sidebar.radio[0].set_value("Reproduced Experiment").run()
    assert any("verification failed" in e.value for e in at.error)


def test_synthetic_mode_creates_once_then_explores(env):
    at = _app()
    at.sidebar.radio[0].set_value("Synthetic Demo").run()
    assert at.button[0].label == "Create synthetic demo run"
    at.sidebar.number_input[0].set_value(1000).run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert "SYNTHETIC DATA" in _texts(at.warning)
    labels = {m.label: m.value for m in at.metric}
    assert {"Alerts", "Precision", "Recall", "False positives", "Missed fraud"} <= set(labels)
    assert "Frozen test result (synthetic test set)" in _texts(at.markdown)

    run_dir = core.synthetic_demo_paths(env / "demo", 1000, 2026)[1]
    run = core.load_run(run_dir)
    model = run.selected_model
    y = run.validation_scores["label"].to_numpy()
    s = run.validation_scores[core.score_column(model)].to_numpy()
    expected = core.threshold_summary(model, y, s, run.threshold_for(model))
    assert labels["Alerts"] == f"{expected.alerts:,}"
    assert labels["Precision"] == f"{expected.precision:.3f}"
    assert labels["Missed fraud"] == f"{expected.missed_fraud:,}"

    # Moving the slider changes the explorer but never the frozen comparison table.
    slider = at.slider[0]
    slider.set_value(0.05).run()
    assert not at.exception
    moved = {m.label: m.value for m in at.metric}
    expected_moved = core.threshold_summary(model, y, s, 0.05)
    assert moved["Alerts"] == f"{expected_moved.alerts:,}"
    frozen = [d for d in at.dataframe if "policy" in d.value.columns]
    assert all(
        (frame.value.loc[frame.value["model"] == model, "threshold"].round(6) == round(run.threshold_for(model), 6)).any()
        for frame in frozen if (frame.value["model"] == model).any()
    )
    # Re-running does not retrain: manifest untouched.
    mtime = (run_dir / core.MANIFEST_NAME).stat().st_mtime
    at.run()
    assert (run_dir / core.MANIFEST_NAME).stat().st_mtime == mtime

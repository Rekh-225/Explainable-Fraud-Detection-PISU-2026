"""Streamlit front end. Launch: ``streamlit run src/fraud_pipeline/ui/app.py``.

Layout only; every number is computed in :mod:`fraud_pipeline.ui.core`.

This module never imports joblib or pickle. Opening a run, switching tabs or
exploring stored predictions reads JSON, CSV and PNG artifacts only; local
explanations come from a validated numeric coefficient file.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fraud_pipeline.manifest import json_ready  # noqa: E402
from fraud_pipeline.ui import core  # noqa: E402

# Workspace-relative defaults (FRAUD_PIPELINE_ROOT or the directory `streamlit run` was
# started from), each overridable individually. Never derived from the package location.
HISTORICAL_DIR = Path(os.environ.get("FRAUD_UI_HISTORICAL_DIR", core.HISTORICAL_DIR))
RUNS_ROOT = Path(os.environ.get("FRAUD_UI_RUNS_ROOT", core.RUNS_ROOT))
DEMO_ROOT = Path(os.environ.get("FRAUD_UI_DEMO_ROOT", core.DEMO_ROOT))
EXPORT_DIR = core.workspace_root() / "output" / "ui_exports"

MODE_HISTORICAL = "Historical Results"
MODE_SYNTHETIC = "Synthetic Demo"
MODE_REPRODUCED = "Reproduced Experiment"
MODES = (MODE_HISTORICAL, MODE_SYNTHETIC, MODE_REPRODUCED)

COMPARISON_COLUMNS = [
    "model", "policy", "threshold", "precision", "recall", "f1", "average_precision",
    "pr_auc_trapezoidal", "roc_auc", "true_positives", "false_positives", "false_negatives",
    "true_negatives", "alert_rate",
]


# --------------------------------------------------------------------------- cached loaders


# Caches are keyed by the *content* of the run directory (manifest bytes plus the
# digest of every file), recomputed on each rerun. Editing any artifact - even with
# the same size and preserved timestamps - changes the key, forces a fresh load and
# fails verification. Figures and the linear spec are additionally re-hashed at the
# moment they are displayed.


@st.cache_resource(show_spinner="Verifying run artifacts...")
def _load_run(run_dir: str, content_key: str, reproduced: bool) -> core.LoadedRun:
    loader = core.load_reproduced_run if reproduced else core.load_run
    return loader(Path(run_dir))


@st.cache_data(show_spinner="Reading source dataset (checksum-verified)...")
def _load_rows(run_dir: str, content_key: str, row_ids: tuple[int, ...]) -> pd.DataFrame:
    return core.load_dataset_rows(core.load_run(Path(run_dir)), list(row_ids))


@st.cache_resource(show_spinner="Checking the explanation file against stored scores...")
def _verified_linear_explainer(run_dir: str, content_key: str, dataset_sha: str):
    """Structural + semantic validation of the linear spec; raises UILoadError on failure."""
    run = core.load_run(Path(run_dir))
    spec = core.load_linear_explainer(run)
    ids = run.validation_scores["row_id"].head(core.LINEAR_SPEC_CHECK_ROWS).tolist()
    rows = core.load_dataset_rows(run, [int(i) for i in ids])
    check = core.verify_linear_explainer(run, spec, rows)
    return spec, check


def _content_key(run_dir: Path) -> str:
    return core.run_content_key(run_dir)


def _open_run(run_dir: Path, reproduced: bool) -> core.LoadedRun:
    return _load_run(str(run_dir), _content_key(run_dir), reproduced)


def _figure(run: core.LoadedRun, key: str, **kwargs) -> None:
    """Display a figure only after re-verifying the bytes that are shown."""
    if key not in run.figures:
        return
    try:
        st.image(core.read_verified_figure(run, key), **kwargs)
    except core.UILoadError as exc:
        st.error(f"Figure withheld: {exc}")


def _t(value) -> str:
    """Untrusted metadata rendered as inert text."""
    return core.escape_markdown(value)


def _annotations(run_id: str) -> core.AnnotationStore:
    stores = st.session_state.setdefault("annotations", {})
    return stores.setdefault(run_id, core.AnnotationStore())


# --------------------------------------------------------------------------- shared widgets


def _comparison_table(frame: pd.DataFrame) -> None:
    columns = [c for c in COMPARISON_COLUMNS if c in frame.columns]
    st.dataframe(frame[columns], width="stretch", hide_index=True)
    st.caption(
        "`average_precision` (used for model selection) and `pr_auc_trapezoidal` are different "
        "precision-recall summaries and are reported separately. " + core.SCORE_DISCLAIMER
    )


def _data_quality(dq: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (raw)", f"{dq.get('original_rows', 0):,}")
    c2.metric("Exact duplicates removed", f"{dq.get('exact_duplicate_rows_removed', 0):,}")
    c3.metric("Rows (clean)", f"{dq.get('cleaned_rows', 0):,}")
    c4.metric("Fraud prevalence", f"{dq.get('cleaned_fraud_prevalence', 0):.4%}")
    st.write(f"Source: {_t(dq.get('source'))} - {_t(dq.get('source_description', dq.get('source_note', '')))}")
    if "time_column_note" in dq:
        st.write(_t(dq["time_column_note"]))
    st.write(f"Class counts after duplicate policy: {_t(dq.get('cleaned_class_counts'))}")


def _frozen_test_result(frame: pd.DataFrame, model: str, label: str) -> None:
    st.markdown(f"#### Frozen test result: {label}")
    st.info(
        "Computed once at the validation-selected threshold when the run was produced. "
        "The explorer above changes nothing here: an explored threshold is **not** a newly "
        "validated operating point.",
        icon="🔒",
    )
    _comparison_table(frame[frame["model"] == model])


def _pr_figure(run: core.LoadedRun, model: str, point: core.ThresholdSummary | None):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    y = run.validation_scores["label"].to_numpy()
    for name in run.scored_models:
        curve = core.threshold_curve(y, run.validation_scores[core.score_column(name)].to_numpy())
        ax.plot(curve["recall"], curve["precision"], linewidth=1.8, label=name)
    if point:
        ax.scatter([point.recall], [point.precision], s=70, color="black", zorder=5,
                   label=f"{model} @ {point.threshold:.4f}")
    ax.axhline(y.mean(), color="#777", linestyle="--", linewidth=1, label=f"No-skill ({y.mean():.4f})")
    ax.set_xlabel("Recall (validation)")
    ax.set_ylabel("Precision (validation)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Validation precision-recall" + (" - SYNTHETIC" if run.is_synthetic else ""))
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def _limitations(items: list[str]) -> None:
    for item in items:
        st.markdown(f"- {_t(item)}")
    st.markdown(f"- {core.SCORE_DISCLAIMER}")
    st.markdown(f"- {core.FEATURE_DISCLAIMER}")
    st.markdown(f"- {core.IMPORTANCE_DISCLAIMER}")


def _download_exports(mode: str, run, report: str, config: dict, stem: str) -> None:
    c1, c2 = st.columns(2)
    c1.download_button("Download analytical report (Markdown)", report, f"{stem}_report.md", "text/markdown")
    c2.download_button("Download configuration (JSON)", json.dumps(json_ready(config), indent=2),
                       f"{stem}_config.json", "application/json")
    if st.button("Also save both under output/ui_exports/"):
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        (EXPORT_DIR / f"{stem}_report.md").write_text(report, encoding="utf-8")
        (EXPORT_DIR / f"{stem}_config.json").write_text(json.dumps(json_ready(config), indent=2), encoding="utf-8")
        st.success(f"Saved to {EXPORT_DIR}")
    with st.expander("Preview report (plain text; Markdown is not rendered so supplied metadata cannot inject links or images)"):
        st.code(report, language="markdown")


# --------------------------------------------------------------------------- historical mode


def render_historical() -> None:
    st.header("Historical Results (committed artifacts, July 2026)")
    try:
        hist = core.load_historical(HISTORICAL_DIR)
    except core.UILoadError as exc:
        st.error(str(exc))
        st.stop()

    st.success(
        f"Showing the committed files in `{hist.directory}` exactly as cited by the report. "
        "Nothing is recomputed."
    )
    with st.expander("Missing provenance for these artifacts", expanded=True):
        for gap in hist.provenance_gaps:
            st.markdown(f"- {gap}")
        st.markdown(
            "- To obtain full provenance, run `fraud-pipeline reproduce-historical` locally and open the "
            "result in **Reproduced Experiment** mode. This mode does not do that for you."
        )

    tabs = st.tabs(["Overview", "Model comparison", "Figures", "Features", "Limitations", "Exports"])
    with tabs[0]:
        _data_quality(hist.data_quality)
        st.markdown("#### Splits")
        st.json(hist.split_summary, expanded=False)
        st.markdown("#### Selection")
        st.write(
            f"Selected model **{hist.selected_model}** by validation Average Precision; "
            f"validation-selected threshold **{hist.selected_threshold:.6f}**."
        )
        st.json(hist.validation_summary, expanded=False)
    with tabs[1]:
        st.markdown("#### Test-set comparison (frozen)")
        _comparison_table(hist.model_comparison)
        st.warning(
            "The historical export contains no validation predictions, so there is no threshold "
            "explorer or review queue in this mode.", icon="ℹ️",
        )
    with tabs[2]:
        for key in ("precision_recall_curves", "confusion_matrices", "class_imbalance",
                    "amount_distribution", "fraud_rate_by_amount_band"):
            if key in hist.figures:
                st.image(str(hist.figures[key]), caption=f"Committed figure (no manifest to verify against): {hist.figures[key].name}")
    with tabs[3]:
        st.dataframe(hist.feature_importance, width="stretch", hide_index=True)
        if "feature_importance" in hist.figures:
            st.image(str(hist.figures["feature_importance"]))
        st.warning(core.IMPORTANCE_DISCLAIMER)
        st.info(core.FEATURE_DISCLAIMER)
    with tabs[4]:
        _limitations(hist.analysis_summary.get("limitations", []))
        st.markdown(
            "- Validation recall at the selected threshold was "
            f"{hist.analysis_summary['threshold_policy']['validation_recall']:.1%}; test recall was "
            f"{hist.analysis_summary['selected_model_test_metrics']['recall']:.1%}. The gap is reported, not hidden."
        )
    with tabs[5]:
        report = core.render_report(
            MODE_HISTORICAL, hist.data_quality.get("source", ""), hist.data_quality, hist.model_comparison,
            hist.selected_model, hist.selected_threshold, provenance_gaps=hist.provenance_gaps,
            limitations=hist.analysis_summary.get("limitations", []),
            provenance_label=core.PROVENANCE_LABELS[core.Provenance.HISTORICAL_FROZEN],
            frozen_label=core.frozen_test_label(core.Provenance.HISTORICAL_FROZEN),
        )
        config = core.export_config(
            MODE_HISTORICAL, None, None, None, [], None, None,
            provenance=core.Provenance.HISTORICAL_FROZEN,
            provenance_label=core.PROVENANCE_LABELS[core.Provenance.HISTORICAL_FROZEN],
        )
        config["historical_dir"] = str(hist.directory)
        _download_exports(MODE_HISTORICAL, None, report, config, "historical")


# --------------------------------------------------------------------------- run-backed modes


# Streamlit drops the session-state entry of a keyed widget that is not rendered
# during a rerun (e.g. the Random Forest slider while Logistic Regression is
# shown, or every widget of run A while run B is open). Explored values are
# therefore mirrored into plain "store" entries that survive, and each widget
# is re-seeded from its store when it is created again.


def _slider_key(run: core.LoadedRun, model: str) -> str:
    return f"threshold::{run.run_id}::{model}"


def _threshold_store_key(run: core.LoadedRun, model: str) -> str:
    return f"explored_threshold::{run.run_id}::{model}"


def _model_key(run: core.LoadedRun) -> str:
    return f"model::{run.run_id}"


def _model_store_key(run: core.LoadedRun) -> str:
    return f"explored_model::{run.run_id}"


def _capacity_key(run: core.LoadedRun) -> str:
    return f"capacities::{run.run_id}"


def _capacity_store_key(run: core.LoadedRun) -> str:
    return f"explored_capacities::{run.run_id}"


DEFAULT_CAPACITIES = "10, 25, 50, 100"


def _reset_threshold(slider_key: str, store_key: str, value: float) -> None:
    """Button callback: runs before the rerun, so the slider shows the reset value."""
    st.session_state[slider_key] = value
    st.session_state[store_key] = value


def _active_model(run: core.LoadedRun) -> str:
    model = st.session_state.get(_model_store_key(run), run.selected_model)
    return model if model in run.scored_models else run.scored_models[0]


def _active_threshold(run: core.LoadedRun, model: str) -> float:
    """The explored threshold for (run, model); each pair keeps its own state."""
    return float(st.session_state.get(_threshold_store_key(run, model), run.threshold_for(model)))


def _capacity_text(run: core.LoadedRun) -> str:
    return st.session_state.get(_capacity_store_key(run), DEFAULT_CAPACITIES)


def _labels_scores(run: core.LoadedRun, model: str):
    return run.validation_scores["label"].to_numpy(), run.validation_scores[core.score_column(model)].to_numpy()


def _capacity_table_or_none(run: core.LoadedRun, model: str):
    """Recompute the capacity table for the *current* model and text; never reuse a stale one."""
    try:
        capacities = core.parse_capacities(_capacity_text(run), len(run.validation_scores))
    except ValueError:
        return None, []
    if not capacities:
        return None, []
    y, s = _labels_scores(run, model)
    return core.capacity_scenarios(model, y, s, capacities), capacities


def render_run(run: core.LoadedRun, mode: str) -> None:
    synthetic = run.is_synthetic
    provenance, provenance_label, comparison = core.classify_provenance(run, HISTORICAL_DIR)
    frozen_label = core.frozen_test_label(provenance)
    if synthetic:
        st.warning(
            "**SYNTHETIC DATA.** This run was produced on a generated fixture. Its metrics are "
            "synthetic and say nothing about the historical experiment or real transactions.", icon="⚠️",
        )
    elif provenance == core.Provenance.DECLARED_SOURCE_RUN:
        st.warning(f"**Provenance:** {_t(provenance_label)}. Do not describe these results as the historical experiment.", icon="⚠️")
    else:
        st.info(f"**Provenance:** {_t(provenance_label)}")
    st.caption(
        f"Run {_t(run.run_id)} - declared source {_t(run.source)} - dataset sha256 {run.manifest['dataset']['sha256'][:16]}... "
        f"- required and displayed artifacts checksum-verified: {'yes' if run.verified else 'NO'} "
        "(integrity of this directory, not authenticity of its producer)"
    )
    if run.unverified_extra_files:
        st.caption(
            f"{len(run.unverified_extra_files)} extra file(s) in the run directory are not listed in the "
            f"manifest and are ignored (unverified): {', '.join(run.unverified_extra_files[:5])}"
            + (" ..." if len(run.unverified_extra_files) > 5 else "")
        )
    for note in run.compatibility_notes:
        st.info(note)

    tabs = st.tabs([
        "Overview", "Model comparison", "Threshold explorer", "Review queue", "Features",
        "Cost scenario", "Exports", "Manifest & limitations",
    ])

    with tabs[0]:
        _data_quality(run.data_quality)
        st.markdown("#### Splits (stratified, disjoint)")
        st.json(run.split_summary, expanded=False)
        st.markdown("#### Selection")
        st.write(
            f"Selected model **{run.selected_model}** (validation Average Precision "
            f"{run.manifest['selection']['validation_average_precision'][run.selected_model]:.4f}); "
            f"validation-selected threshold **{run.selected_threshold:.6f}**."
        )
        for key in ("class_imbalance", "amount_distribution", "fraud_rate_by_amount_band"):
            _figure(run, key, width=560)

    with tabs[1]:
        st.markdown(f"#### Test-set comparison: {frozen_label}")
        _comparison_table(run.model_comparison)
        _figure(run, "precision_recall_curves", caption="Test-set PR curves written by the run (re-verified on display)", width=560)
        _figure(run, "confusion_matrices", width=700)

    with tabs[2]:
        st.markdown("#### Threshold explorer - validation predictions only")
        models = run.scored_models
        if _model_key(run) not in st.session_state:
            st.session_state[_model_key(run)] = _active_model(run)
        model = st.selectbox("Model", models, key=_model_key(run))
        st.session_state[_model_store_key(run)] = model
        default_t = run.threshold_for(model)
        slider_key, store_key = _slider_key(run, model), _threshold_store_key(run, model)
        if slider_key not in st.session_state:
            st.session_state[slider_key] = _active_threshold(run, model)
        col_a, col_b = st.columns([3, 1])
        threshold = col_a.slider(
            "Threshold (score >= threshold is an alert)", 0.0, 1.0, step=0.0005, format="%.4f", key=slider_key
        )
        st.session_state[store_key] = threshold
        col_b.button(
            "Reset to validation-selected", key=f"reset::{run.run_id}::{model}",
            on_click=_reset_threshold, args=(slider_key, store_key, default_t),
        )

        y, s = _labels_scores(run, model)
        summary = core.threshold_summary(model, y, s, threshold)
        selected_summary = core.threshold_summary(model, y, s, default_t)
        m = st.columns(6)
        m[0].metric("Alerts", f"{summary.alerts:,}", f"{summary.alerts - selected_summary.alerts:+,} vs selected")
        m[1].metric("Precision", f"{summary.precision:.3f}", f"{summary.precision - selected_summary.precision:+.3f}")
        m[2].metric("Recall", f"{summary.recall:.3f}", f"{summary.recall - selected_summary.recall:+.3f}")
        m[3].metric("False positives", f"{summary.false_positives:,}", f"{summary.false_positives - selected_summary.false_positives:+,}", delta_color="inverse")
        m[4].metric("Missed fraud", f"{summary.missed_fraud:,}", f"{summary.missed_fraud - selected_summary.missed_fraud:+,}", delta_color="inverse")
        m[5].metric("Alert rate", f"{summary.alert_rate:.3%}")
        st.caption(
            f"{summary.rows:,} validation rows, {summary.fraud_total} fraud. Validation-selected threshold for "
            f"{model}: {default_t:.6f} ({_t(run.manifest['threshold_policy']['decisions'][model].get('rule', ''))}). "
            + core.SCORE_DISCLAIMER
        )
        st.pyplot(_pr_figure(run, model, summary), width=640)

        st.markdown("#### Review-capacity scenarios")
        if _capacity_key(run) not in st.session_state:
            st.session_state[_capacity_key(run)] = _capacity_text(run)
        cap_text = st.text_input("Review capacities (alerts per period, comma-separated)", key=_capacity_key(run))
        st.session_state[_capacity_store_key(run)] = cap_text
        try:
            capacities = core.parse_capacities(cap_text, len(y))
        except ValueError as exc:
            st.error(f"{exc}. Enter positive whole numbers separated by commas, e.g. 10, 25, 50.")
        else:
            if not capacities:
                st.info("Enter at least one review capacity (a positive whole number) to see scenarios.")
            else:
                table = core.capacity_scenarios(model, y, s, capacities)
                st.dataframe(table[core.CAPACITY_COLUMNS], width="stretch", hide_index=True)
            st.caption("Threshold = score of the N-th highest validation row; tied scores can produce more alerts than the capacity.")
        _frozen_test_result(run.model_comparison, model, frozen_label)

    with tabs[3]:
        st.markdown("#### Simulated review queue (validation rows)")
        model = _active_model(run)
        threshold = _active_threshold(run, model)
        st.write(f"Model **{model}**, threshold **{threshold:.4f}** (set in the Threshold explorer).")
        limit = st.slider("Queue length", 5, core.MAX_QUEUE_ROWS, 25, 5, key=f"queue_limit::{run.run_id}")
        queue = core.build_review_queue(run, model, threshold, limit)
        store = _annotations(run.run_id)
        reveal = st.checkbox("Reveal dataset ground-truth labels (kept separate from your annotations)", value=False, key=f"reveal::{run.run_id}")
        load_features = st.checkbox("Load transaction features from the checksum-verified source CSV", value=False, key=f"features::{run.run_id}")
        table = core.queue_with_annotations(queue, store)
        if not reveal:
            table = table.drop(columns=[core.GROUND_TRUTH_COLUMN])
        rows = None
        if load_features and len(queue):
            try:
                rows = _load_rows(str(run.run_dir), run.content_key, tuple(int(i) for i in queue["row_id"]))
                table = table.merge(rows.reset_index(), on="row_id", how="left")
            except core.UILoadError as exc:
                st.error(str(exc))
        st.dataframe(
            table, width="stretch", hide_index=True,
            column_config={"model_score": st.column_config.NumberColumn("model_score (uncalibrated)", format="%.6f")},
        )
        st.info(core.FEATURE_DISCLAIMER + " No retraining happens from annotations.", icon="ℹ️")

        if len(queue):
            st.markdown("##### Annotate a queued row")
            c1, c2, c3 = st.columns([1, 1, 2])
            row_id = c1.selectbox("row_id", queue["row_id"].tolist(), key=f"row::{run.run_id}")
            choice = c2.radio("Analyst annotation", core.ANNOTATION_CHOICES, index=core.ANNOTATION_CHOICES.index(store.get(row_id)), horizontal=True, key=f"choice::{run.run_id}::{row_id}")
            # Notes are keyed by run and row. The widget entry disappears when another row is
            # shown, so the draft is mirrored in a per-row store and re-seeded from the saved
            # annotation (or the draft) when the row is selected again.
            note_key = f"note::{run.run_id}::{int(row_id)}"
            draft_key = f"note_draft::{run.run_id}::{int(row_id)}"
            if note_key not in st.session_state:
                st.session_state[note_key] = st.session_state.get(
                    draft_key, store.records.get(int(row_id), {}).get("note", "")
                )
            note = c3.text_input("Note (optional, stored with the annotation)", key=note_key)
            st.session_state[draft_key] = note
            if st.button("Save annotation", key=f"save::{run.run_id}"):
                store.annotate(int(row_id), choice, note)
                st.success(f"Saved annotation for row {row_id}: {choice}")
            st.write("Annotation counts:", store.summary())
            if reveal:
                st.write("Agreement with ground truth (queued rows):", core.annotation_agreement(queue, store))
            st.download_button("Download annotations (CSV, separate from labels)", store.frame().to_csv(index=False),
                               f"{run.run_id}_annotations.csv", "text/csv", key=f"dl_ann::{run.run_id}")

            available, message = core.explanation_availability(run, model)
            with st.expander(f"Local explanation for {model} (row {row_id})"):
                if not available:
                    st.write(message)
                elif not load_features or rows is None:
                    st.write("Enable *Load transaction features* above to compute the additive terms for this row. "
                             "The explanation file is first checked against the stored scores using checksum-verified source rows.")
                else:
                    try:
                        spec, check = _verified_linear_explainer(str(run.run_dir), run.content_key, run.manifest["dataset"]["sha256"])
                        contributions = core.linear_contributions(spec, rows.loc[int(row_id)])
                        displayed = float(queue.loc[queue["row_id"] == row_id, "model_score"].iloc[0])
                        if abs(contributions.attrs["score"] - displayed) > core.LINEAR_SPEC_SCORE_TOLERANCE:
                            raise core.UILoadError(
                                f"Reconstructed score {contributions.attrs['score']:.6f} does not match the displayed score "
                                f"{displayed:.6f} for row {row_id}; explanation withheld."
                            )
                        st.write(
                            f"{message} Verified against {check['rows_checked']} stored scores "
                            f"(max |difference| {check['max_abs_difference']:.2e}). Row {row_id}: log-odds "
                            f"{contributions.attrs['log_odds']:.3f} (intercept {contributions.attrs['intercept']:.3f}); "
                            f"reconstructed score {contributions.attrs['score']:.6f} = displayed score {displayed:.6f}."
                        )
                        st.dataframe(contributions.head(10), width="stretch", hide_index=True)
                        st.caption("Computed from the validated numeric coefficients exported by the run, not from a pickled model. "
                                   "Features named V* are anonymized PCA components; the terms show model reliance, not a business cause.")
                    except core.UILoadError as exc:
                        st.error(f"Explanation disabled: {exc}")

    with tabs[4]:
        st.markdown("#### Global Random Forest feature importance")
        st.dataframe(run.feature_importance, width="stretch", hide_index=True)
        _figure(run, "feature_importance", width=560)
        st.warning(core.IMPORTANCE_DISCLAIMER)
        st.info(core.FEATURE_DISCLAIMER)

    with tabs[5]:
        st.markdown("#### Cost scenario (hypothetical)")
        st.write("Enter explicit hypothetical unit costs. The result is a scenario computed on validation counts, not realized savings.")
        ack = st.checkbox("I understand these costs are hypothetical inputs I am supplying", key=f"ack::{run.run_id}")
        c1, c2, c3, c4 = st.columns(4)
        review_cost = c1.number_input("Review cost per alert", min_value=0.0, value=0.0, step=1.0, key=f"cost_review::{run.run_id}")
        missed_loss = c2.number_input("Loss per missed fraud", min_value=0.0, value=0.0, step=10.0, key=f"cost_missed::{run.run_id}")
        recovered = c3.number_input("Recovered per caught fraud", min_value=0.0, value=0.0, step=10.0, key=f"cost_recovered::{run.run_id}")
        currency = c4.text_input("Unit label", "units", key=f"cost_unit::{run.run_id}")
        model = _active_model(run)
        threshold = _active_threshold(run, model)
        if ack and (review_cost or missed_loss or recovered):
            inputs = core.CostInputs(review_cost, missed_loss, recovered, currency)
            y, s = _labels_scores(run, model)
            summary = core.threshold_summary(model, y, s, threshold)
            scenario = core.cost_scenario(summary, inputs)
            st.warning(scenario["disclaimer"], icon="⚠️")
            k = st.columns(4)
            k[0].metric("Review cost total", f"{scenario['review_cost_total']:,.2f} {currency}")
            k[1].metric("Missed-fraud loss total", f"{scenario['missed_fraud_loss_total']:,.2f} {currency}")
            k[2].metric("Recovered total", f"{scenario['recovered_total']:,.2f} {currency}")
            k[3].metric("Net scenario value", f"{scenario['net_scenario_value']:,.2f} {currency}")
            st.caption(f"Based on {summary.alerts} alerts, {summary.true_positives} caught and {summary.missed_fraud} missed fraud cases at validation threshold {threshold:.4f} for {model}.")
        else:
            st.info("Tick the acknowledgement and enter at least one non-zero hypothetical cost to compute a scenario.")

    with tabs[6]:
        model = _active_model(run)
        threshold = _active_threshold(run, model)
        y, s = _labels_scores(run, model)
        summary = core.threshold_summary(model, y, s, threshold)
        capacity_table, capacities = _capacity_table_or_none(run, model)
        if capacity_table is None:
            st.caption("No valid review capacities are set, so the export contains no capacity table.")
        cost_inputs = None
        scenario = None
        if st.session_state.get(f"ack::{run.run_id}"):
            cost_inputs = core.CostInputs(
                float(st.session_state.get(f"cost_review::{run.run_id}", 0.0)),
                float(st.session_state.get(f"cost_missed::{run.run_id}", 0.0)),
                float(st.session_state.get(f"cost_recovered::{run.run_id}", 0.0)),
                st.session_state.get(f"cost_unit::{run.run_id}", "units"),
            )
            if any((cost_inputs.review_cost_per_alert, cost_inputs.loss_per_missed_fraud, cost_inputs.recovered_per_caught_fraud)):
                scenario = core.cost_scenario(summary, cost_inputs)
            else:
                cost_inputs = None
        st.write(f"Exporting model **{model}** at validation threshold **{threshold:.6f}**.")
        report = core.render_report(
            mode, run.data_quality.get("source_description", run.source), run.data_quality, run.model_comparison,
            run.selected_model, run.selected_threshold, explorer=summary, capacity_table=capacity_table,
            cost=scenario, annotations=_annotations(run.run_id), manifest=run.manifest,
            limitations=run.analysis_summary.get("limitations", []),
            provenance_label=provenance_label, frozen_label=frozen_label,
        )
        config = core.export_config(
            mode, run, model, threshold, capacities, cost_inputs, _annotations(run.run_id),
            provenance=provenance, provenance_label=provenance_label,
        )
        _download_exports(mode, run, report, config, run.run_id)

    with tabs[7]:
        st.markdown("#### Run manifest")
        st.write("Verification:", run.verification)
        st.write("Provenance:", provenance.value, "-", _t(provenance_label))
        if comparison:
            st.write("Comparison with frozen historical artifacts:", comparison)
        st.write("Cross-file consistency checks:", run.consistency_checks)
        st.caption(
            "Digests prove the directory is internally consistent (integrity). They do not prove who produced it "
            "(authenticity); the UI therefore reads only tables, JSON and images from a run and never unpickles a model."
        )
        st.json(run.manifest, expanded=False)
        st.markdown("#### Limitations")
        _limitations(run.analysis_summary.get("limitations", []))
        decision = run.manifest["threshold_policy"]["decisions"][run.selected_model]
        test_row = run.model_comparison[(run.model_comparison["model"] == run.selected_model)
                                        & (run.model_comparison["policy"] == core.OPERATING_POLICY)].iloc[0]
        st.markdown(
            f"- Validation recall at the selected threshold: {decision['validation_recall']:.1%}; "
            f"test recall: {test_row['recall']:.1%}. A stratified random split does not simulate drift."
        )


def render_synthetic() -> None:
    st.header("Synthetic Demo (no downloads, no keys)")
    st.warning("Everything in this mode is generated. Metrics are **synthetic** and unrelated to the report's results.", icon="⚠️")
    rows = st.sidebar.number_input("Synthetic rows", 500, 50_000, 6000, 500)
    seed = st.sidebar.number_input("Synthetic seed", 0, 10_000, 2026, 1)
    csv_path, run_dir = core.synthetic_demo_paths(DEMO_ROOT, int(rows), int(seed))
    exists = (run_dir / core.MANIFEST_NAME).is_file()
    if not exists:
        st.info(f"No demo run for rows={rows}, seed={seed} yet. It is trained once (a few seconds, 60 small trees) and reused afterwards; nothing retrains on interaction.")
        if not st.button("Create synthetic demo run"):
            st.stop()
        with st.spinner("Generating fixture and training small models once..."):
            core.ensure_synthetic_demo_run(DEMO_ROOT, int(rows), int(seed))
    try:
        run = _open_run(run_dir, reproduced=False)
    except core.UILoadError as exc:
        st.error(str(exc))
        st.stop()
    render_run(run, MODE_SYNTHETIC)


def render_reproduced() -> None:
    st.header("Reproduced Experiment (verified local run)")
    runs = core.list_runs(RUNS_ROOT)
    options = [p.name for p in runs]
    choice = st.sidebar.selectbox("Run under output/runs", options or ["(none found)"])
    custom = st.sidebar.text_input("...or path to a run directory", "")
    if custom:
        run_dir = Path(custom)
    elif runs:
        run_dir = RUNS_ROOT / choice
    else:
        st.error(
            f"No runs with a {core.MANIFEST_NAME} were found under `{RUNS_ROOT}`. Produce one locally with "
            "`fraud-pipeline reproduce-historical` (needs data/raw/creditcardfraud/creditcard.csv). "
            "This mode does not fall back to historical or synthetic data."
        )
        st.stop()
    try:
        run = _open_run(run_dir, reproduced=True)
    except (core.UILoadError, FileNotFoundError, OSError) as exc:
        st.error(str(exc))
        st.stop()
    st.success(f"Artifact checksums verified against `{run.run_dir / core.MANIFEST_NAME}` (integrity of the directory, not authenticity of its producer).")
    render_run(run, MODE_REPRODUCED)


def main() -> None:
    st.set_page_config(page_title="Explainable Fraud Detection - local UI", layout="wide")
    st.title("Explainable Fraud Detection - local analysis interface")
    st.caption("Offline academic prototype on the ULB credit-card dataset. Not a payment control. " + core.FEATURE_DISCLAIMER)
    mode = st.sidebar.radio("Mode", MODES, key="mode")
    st.sidebar.caption("Modes never fall back to one another; missing inputs are reported as errors.")
    if mode == MODE_HISTORICAL:
        render_historical()
    elif mode == MODE_SYNTHETIC:
        render_synthetic()
    else:
        render_reproduced()


main()

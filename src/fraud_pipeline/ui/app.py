"""Streamlit front end. Launch: ``streamlit run src/fraud_pipeline/ui/app.py``.

Layout only; every number is computed in :mod:`fraud_pipeline.ui.core`.
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

HISTORICAL_DIR = Path(os.environ.get("FRAUD_UI_HISTORICAL_DIR", core.HISTORICAL_DIR))
RUNS_ROOT = Path(os.environ.get("FRAUD_UI_RUNS_ROOT", core.RUNS_ROOT))
DEMO_ROOT = Path(os.environ.get("FRAUD_UI_DEMO_ROOT", core.DEMO_ROOT))
EXPORT_DIR = core.PROJECT_ROOT / "output" / "ui_exports"

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


@st.cache_resource(show_spinner="Verifying run artifacts...")
def _load_run(run_dir: str, manifest_mtime: float, reproduced: bool) -> core.LoadedRun:
    loader = core.load_reproduced_run if reproduced else core.load_run
    return loader(Path(run_dir))


@st.cache_resource(show_spinner="Verifying model bundle checksum...")
def _load_bundle(run_dir: str, manifest_mtime: float):
    return core.load_run_bundle(core.load_run(Path(run_dir)))


@st.cache_data(show_spinner="Reading source dataset (checksum-verified)...")
def _load_rows(run_dir: str, manifest_mtime: float, row_ids: tuple[int, ...]) -> pd.DataFrame:
    return core.load_dataset_rows(core.load_run(Path(run_dir)), list(row_ids))


def _manifest_mtime(run_dir: Path) -> float:
    return (run_dir / core.MANIFEST_NAME).stat().st_mtime


def _open_run(run_dir: Path, reproduced: bool) -> core.LoadedRun:
    return _load_run(str(run_dir), _manifest_mtime(run_dir), reproduced)


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
    st.write(
        f"Source: `{dq.get('source')}` - {dq.get('source_description', dq.get('source_note', ''))}"
    )
    if "time_column_note" in dq:
        st.write(dq["time_column_note"])
    st.write(f"Class counts after duplicate policy: {dq.get('cleaned_class_counts')}")


def _frozen_test_result(frame: pd.DataFrame, model: str, synthetic: bool) -> None:
    label = "synthetic test set" if synthetic else "untouched historical test set"
    st.markdown(f"#### Frozen test result ({label})")
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
        st.markdown(f"- {item}")
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
    with st.expander("Preview report"):
        st.markdown(report)


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
                st.image(str(hist.figures[key]), caption=f"Committed figure: {hist.figures[key].name}")
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
        )
        config = core.export_config(MODE_HISTORICAL, None, None, None, [], None, None)
        config["historical_dir"] = str(hist.directory)
        _download_exports(MODE_HISTORICAL, None, report, config, "historical")


# --------------------------------------------------------------------------- run-backed modes


def render_run(run: core.LoadedRun, mode: str) -> None:
    synthetic = run.is_synthetic
    if synthetic:
        st.warning(
            "**SYNTHETIC DATA.** This run was produced on a generated fixture. Its metrics are "
            "synthetic and say nothing about the historical experiment or real transactions.", icon="⚠️",
        )
    st.caption(
        f"Run `{run.run_id}` - source `{run.source}` - dataset sha256 `{run.manifest['dataset']['sha256'][:16]}...` "
        f"- artifacts verified: {'yes' if not any(run.verification[k] for k in ('missing', 'modified')) else 'NO'}"
    )
    for note in run.compatibility_notes:
        st.info(note)

    tabs = st.tabs([
        "Overview", "Model comparison", "Threshold explorer", "Review queue", "Features",
        "Cost scenario", "Exports", "Manifest & limitations",
    ])
    state = st.session_state.setdefault(f"explorer::{run.run_id}", {})

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
            if key in run.figures:
                st.image(str(run.figures[key]), width=560)

    with tabs[1]:
        st.markdown("#### Test-set comparison (frozen)" + (" - synthetic" if synthetic else ""))
        _comparison_table(run.model_comparison)
        if "precision_recall_curves" in run.figures:
            st.image(str(run.figures["precision_recall_curves"]), caption="Test-set PR curves written by the run", width=560)
        if "confusion_matrices" in run.figures:
            st.image(str(run.figures["confusion_matrices"]), width=700)

    with tabs[2]:
        st.markdown("#### Threshold explorer - validation predictions only")
        models = run.scored_models
        model = st.selectbox("Model", models, index=models.index(run.selected_model) if run.selected_model in models else 0)
        default_t = run.threshold_for(model)
        col_a, col_b = st.columns([3, 1])
        threshold = col_a.slider("Threshold (score >= threshold is an alert)", 0.0, 1.0,
                                 float(state.get("threshold", default_t)), 0.0005, format="%.4f",
                                 key=f"slider::{run.run_id}::{model}")
        if col_b.button("Reset to validation-selected"):
            threshold = default_t
        state.update(model=model, threshold=threshold)

        y = run.validation_scores["label"].to_numpy()
        s = run.validation_scores[core.score_column(model)].to_numpy()
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
            f"{model}: {default_t:.6f} ({run.manifest['threshold_policy']['decisions'][model]['rule']}). "
            + core.SCORE_DISCLAIMER
        )
        st.pyplot(_pr_figure(run, model, summary), width=640)

        st.markdown("#### Review-capacity scenarios")
        cap_text = st.text_input("Review capacities (alerts per period, comma-separated)", state.get("capacities", "10, 25, 50, 100"))
        try:
            capacities = core.parse_capacities(cap_text, len(y))
            state["capacities"] = cap_text
            table = core.capacity_scenarios(model, y, s, capacities)
            st.dataframe(table[["review_capacity", "threshold", "alerts", "precision", "recall", "false_positives", "missed_fraud"]],
                         width="stretch", hide_index=True)
            st.caption("Threshold = score of the N-th highest validation row; tied scores can produce more alerts than the capacity.")
            state["capacity_table"] = table
        except ValueError as exc:
            st.error(str(exc))
        _frozen_test_result(run.model_comparison, model, synthetic)

    with tabs[3]:
        st.markdown("#### Simulated review queue (validation rows)")
        model = state.get("model", run.selected_model)
        threshold = state.get("threshold", run.threshold_for(model))
        st.write(f"Model **{model}**, threshold **{threshold:.4f}** (set in the Threshold explorer).")
        limit = st.slider("Queue length", 5, core.MAX_QUEUE_ROWS, 25, 5)
        queue = core.build_review_queue(run, model, threshold, limit)
        store = _annotations(run.run_id)
        reveal = st.checkbox("Reveal dataset ground-truth labels (kept separate from your annotations)", value=False)
        load_features = st.checkbox("Load transaction features from the checksum-verified source CSV", value=False)
        table = core.queue_with_annotations(queue, store)
        if not reveal:
            table = table.drop(columns=[core.GROUND_TRUTH_COLUMN])
        if load_features and len(queue):
            try:
                rows = _load_rows(str(run.run_dir), _manifest_mtime(run.run_dir), tuple(int(i) for i in queue["row_id"]))
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
            row_id = c1.selectbox("row_id", queue["row_id"].tolist())
            choice = c2.radio("Analyst annotation", core.ANNOTATION_CHOICES, index=core.ANNOTATION_CHOICES.index(store.get(row_id)), horizontal=True)
            note = c3.text_input("Note (optional, stored with the annotation)")
            if st.button("Save annotation"):
                store.annotate(int(row_id), choice, note)
                st.success(f"Saved annotation for row {row_id}: {choice}")
            st.write("Annotation counts:", store.summary())
            if reveal:
                st.write("Agreement with ground truth (queued rows):", core.annotation_agreement(queue, store))
            st.download_button("Download annotations (CSV, separate from labels)", store.frame().to_csv(index=False),
                               f"{run.run_id}_annotations.csv", "text/csv")

            with st.expander("Local explanation (Logistic Regression pipeline only)"):
                try:
                    bundle = _load_bundle(str(run.run_dir), _manifest_mtime(run.run_dir))
                    if not load_features:
                        st.write("Enable *Load transaction features* above to compute contributions for the selected row.")
                    else:
                        rows = _load_rows(str(run.run_dir), _manifest_mtime(run.run_dir), tuple(int(i) for i in queue["row_id"]))
                        contributions = core.linear_contributions(bundle, rows.loc[int(row_id)])
                        st.write(f"Row {row_id}: log-odds {contributions.attrs['log_odds']:.3f} (intercept {contributions.attrs['intercept']:.3f}).")
                        st.dataframe(contributions.head(10), width="stretch", hide_index=True)
                        st.caption("Additive log-odds terms of the fitted linear model. Features named V* are anonymized PCA components; the terms show model reliance, not a business cause.")
                except NotImplementedError as exc:
                    st.write(str(exc))
                    st.write(core.IMPORTANCE_DISCLAIMER)
                except core.UILoadError as exc:
                    st.error(str(exc))

    with tabs[4]:
        st.markdown("#### Global Random Forest feature importance")
        st.dataframe(run.feature_importance, width="stretch", hide_index=True)
        if "feature_importance" in run.figures:
            st.image(str(run.figures["feature_importance"]), width=560)
        st.warning(core.IMPORTANCE_DISCLAIMER)
        st.info(core.FEATURE_DISCLAIMER)

    with tabs[5]:
        st.markdown("#### Cost scenario (hypothetical)")
        st.write("Enter explicit hypothetical unit costs. The result is a scenario computed on validation counts, not realized savings.")
        ack = st.checkbox("I understand these costs are hypothetical inputs I am supplying")
        c1, c2, c3, c4 = st.columns(4)
        review_cost = c1.number_input("Review cost per alert", min_value=0.0, value=0.0, step=1.0)
        missed_loss = c2.number_input("Loss per missed fraud", min_value=0.0, value=0.0, step=10.0)
        recovered = c3.number_input("Recovered per caught fraud", min_value=0.0, value=0.0, step=10.0)
        currency = c4.text_input("Unit label", "units")
        state["cost"] = None
        if ack and (review_cost or missed_loss or recovered):
            inputs = core.CostInputs(review_cost, missed_loss, recovered, currency)
            model = state.get("model", run.selected_model)
            threshold = state.get("threshold", run.threshold_for(model))
            summary = core.threshold_summary(model, run.validation_scores["label"].to_numpy(),
                                             run.validation_scores[core.score_column(model)].to_numpy(), threshold)
            scenario = core.cost_scenario(summary, inputs)
            state["cost"] = scenario
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
        model = state.get("model", run.selected_model)
        threshold = state.get("threshold", run.threshold_for(model))
        y = run.validation_scores["label"].to_numpy()
        s = run.validation_scores[core.score_column(model)].to_numpy()
        summary = core.threshold_summary(model, y, s, threshold)
        cost_inputs = core.CostInputs(**{k: v for k, v in state["cost"]["inputs"].items()}) if state.get("cost") else None
        capacities = core.parse_capacities(state.get("capacities", "10, 25, 50, 100"), len(y))
        report = core.render_report(
            mode, run.data_quality.get("source_description", run.source), run.data_quality, run.model_comparison,
            run.selected_model, run.selected_threshold, explorer=summary, capacity_table=state.get("capacity_table"),
            cost=state.get("cost"), annotations=_annotations(run.run_id), manifest=run.manifest,
            limitations=run.analysis_summary.get("limitations", []),
        )
        config = core.export_config(mode, run, model, threshold, capacities, cost_inputs, _annotations(run.run_id))
        _download_exports(mode, run, report, config, run.run_id)

    with tabs[7]:
        st.markdown("#### Run manifest")
        st.write("Verification:", run.verification)
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
    st.success(f"Artifact checksums verified against `{run.run_dir / core.MANIFEST_NAME}`.")
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

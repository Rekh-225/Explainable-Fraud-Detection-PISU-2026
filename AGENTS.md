# Repository notes for contributors and coding agents

## Layout

- `src/fraud_pipeline/` - the analysis pipeline as a package (`fraud-pipeline` CLI).
- `src/run_analysis.py` - compatibility wrapper; equals `fraud-pipeline reproduce-historical`.
- `src/fraud_pipeline/ui/` - local Streamlit UI: `core.py` holds all calculations/loaders (unit-tested), `app.py` layout only. Launch: `streamlit run src/fraud_pipeline/ui/app.py`.
- `docs/` - UI guide, model card, example synthetic report, screenshots.
- `tests/` - pytest suite on deterministic synthetic data; no network, no Kaggle file needed.
- `output/analysis/` - **frozen** historical results (July 2026) cited by the report. Do not regenerate or overwrite.
- `output/runs/<run_id>/` - new runs (git-ignored). Runs are never overwritten.
- `output/ui_demo/`, `output/ui_exports/` - UI synthetic demo run and exports (git-ignored).
- `report/` - the academic report; historical artifact, do not edit.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt -e .            # core + ui + pytest; or requirements.lock.txt for an exact replica
```

Python 3.12 or 3.13. Core pins live in both `pyproject.toml` and `requirements.txt` (kept identical by `tests/test_packaging.py`); `requirements-ui.txt` adds Streamlit (`[ui]` extra), `requirements-dev.txt` adds pytest (`[dev]` extra).

## Verification

```bash
pytest -q                                             # ~40 s, synthetic data only (includes headless Streamlit AppTest)
fraud-pipeline synthetic --output tmp/s.csv --rows 3000 --seed 7 --duplicates 20
fraud-pipeline run --dataset tmp/s.csv --source synthetic --output-root tmp/runs --run-id smoke --rf-estimators 20
fraud-pipeline verify-run tmp/runs/smoke
```

Full historical retraining (needs `data/raw/creditcardfraud/creditcard.csv`, ~5 min, not in CI):

```bash
fraud-pipeline reproduce-historical
```

## Rules

- Dataset input must be explicit (`--dataset` + `--source`). Never add a silent OpenML fallback: the OpenML mirror omits `Time`, a model feature.
- Fit learned preprocessing on the training split only; select models and thresholds on validation only; never tune against test metrics.
- Keep `average_precision` and `pr_auc_trapezoidal` as separate columns.
- The UI must never import joblib/pickle or deserialize a model; explanations use `models/logistic_regression_linear.json`. `load_model_bundle` requires `trusted_source=True` (checksums prove integrity, not authenticity).
- Every file the UI consumes must be listed in `run_manifest.json["artifacts"]` and verified; unlisted files are ignored, never read. Loads are cached by *content key* (`core.run_content_key`), never by mtime; anything displayed later goes through `core.read_verified_artifact`.
- Cross-file consistency (`core._validate_consistency`) and provenance classification (`core.classify_provenance`) are evidence-based; never add a manifest flag that promotes a run to "historical".
- Paths come from `fraud_pipeline.workspace` (explicit > `FRAUD_PIPELINE_ROOT` > cwd), never from `__file__`. Run ids go through `workspace.validate_run_id`; manifest artifact paths through `manifest.is_safe_artifact_path`.
- `duplicate_policy="keep"` with retained duplicates is refused for evaluated runs; do not add grouping that changes the default `drop_exact` experiment.
- Untrusted run metadata is rendered through `core.escape_markdown`; report previews use `st.code`, not `st.markdown`.
- Do not report synthetic metrics as results of the historical experiment.
- UI: modes never fall back to one another; the threshold explorer and review queue use validation predictions only; the test-set table is frozen; annotations stay separate from labels; no retraining from the UI; cost results are hypothetical scenarios.
- Put UI arithmetic in `ui/core.py` with tests, never inline in `app.py`.
- Widget state is keyed per run and per model (`threshold::<run>::<model>`, `model::<run>`, `capacities::<run>`); reset buttons use `on_click` callbacks.
- Launch the UI with `--server.address 127.0.0.1 --browser.gatherUsageStats false` (also set in `.streamlit/config.toml`).

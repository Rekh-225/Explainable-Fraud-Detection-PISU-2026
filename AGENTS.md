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
pip install -r requirements-dev.txt -e .            # or requirements.lock.txt for an exact replica
```

Python 3.12 or 3.13. Direct dependencies pinned in `requirements.txt`.

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
- Never `joblib.load` a user-supplied path; use `fraud_pipeline.artifacts.load_model_bundle` (checksum-verified).
- Do not report synthetic metrics as results of the historical experiment.
- UI: modes never fall back to one another; the threshold explorer and review queue use validation predictions only; the test-set table is frozen; annotations stay separate from labels; no retraining from the UI; cost results are hypothetical scenarios.
- Put UI arithmetic in `ui/core.py` with tests, never inline in `app.py`.

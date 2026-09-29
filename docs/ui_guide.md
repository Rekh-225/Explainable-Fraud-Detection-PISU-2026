# Local UI guide

A small Streamlit interface over the `fraud_pipeline` core and its verified artifacts. It is designed to run only on your machine: no accounts, no deployment, no external model or LLM calls, and Streamlit's usage statistics are switched off (see *Local-only configuration* for what is enforced and what was observed).

![Historical Results mode](screenshots/historical_mode.png)

## Launch (Windows PowerShell)

```powershell
# once
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt -e .        # core + UI + pytest; or: pip install -e ".[ui]"

# every time, from the repository root
streamlit run src/fraud_pipeline/ui/app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false --server.headless true
```

Then open <http://127.0.0.1:8501> in your browser. Nothing is trained at startup. Stop with `Ctrl+C`.

macOS/Linux: same command after `source .venv/bin/activate`.

Paths default to the *workspace*: `FRAUD_PIPELINE_ROOT` if set, otherwise the directory you launched from (never the installed package location). Individual overrides: `FRAUD_UI_HISTORICAL_DIR`, `FRAUD_UI_RUNS_ROOT`, `FRAUD_UI_DEMO_ROOT`.

## Local-only configuration

`.streamlit/config.toml` in the repository root sets `browser.gatherUsageStats = false`, `server.address = "127.0.0.1"`, `server.enableCORS = true`, `server.enableXsrfProtection = true`, `client.toolbarMode = "viewer"` (hides the Deploy menu) and `server.maxUploadSize = 1`.

Precedence, highest first: **command-line flags > `STREAMLIT_*` environment variables > `./.streamlit/config.toml` > `~/.streamlit/config.toml` > defaults**. The launch command above repeats the address and usage-statistics flags so a user-level config or an environment variable such as `STREAMLIT_SERVER_ADDRESS=0.0.0.0` cannot silently widen the binding. Check the effective values with `streamlit config show`.

What this does and does not establish:

- *Configuration-only:* usage statistics are disabled and the server binds to loopback. `pytest tests/test_packaging.py` checks the file.
- *Observed on 2026-09-28 (Windows 11, Streamlit 1.63.0):* `netstat` showed the server listening on `127.0.0.1:8501` only; during a scripted session through all three modes, four samples of the Streamlit process's connections showed only loopback sockets. This is a point-in-time observation with a coarse tool, not a packet capture, and it says nothing about the browser itself. Do not read it as a guarantee of zero network traffic.

## Trust boundary: no model deserialization in the UI

`joblib`/pickle files execute code when loaded. A digest in a sidecar or manifest detects accidental changes but does **not** authenticate the producer - whoever supplies a run directory can supply matching digests. Therefore:

- The UI (`ui/app.py`, `ui/core.py`) never imports `joblib` or `pickle`. Opening a run, changing tabs, exploring predictions, loading transaction features and expanding explanations read only JSON, CSV and PNG files. `tests/test_ui_app.py` runs the app with a `joblib.load` sentinel through all of these actions, including against an externally assembled bundle with consistent checksums.
- Local explanations come from `models/logistic_regression_linear.json`: feature order, scaler mean/scale, coefficients and intercept, validated field by field (types, lengths, finite values, positive scales) before use. The reconstructed log-odds are checked against the fitted model in `tests/test_ui_core.py`.
- `fraud_pipeline.artifacts.load_model_bundle` still exists for local CLI/notebook use and refuses to run without `trusted_source=True`, an explicit assertion by the caller that they produced the run themselves. A path under `output/runs`, a manifest or a matching checksum is not that assertion.
- Nothing in this repository makes loading an arbitrary pickle/joblib file safe.

## Artifact verification

A run opens only if `run_manifest.json` is structurally valid (required keys, `artifacts` map of safe relative paths - forward slashes only, no `..`, no drive/UNC prefixes - to 64-hex digests, dataset digest, feature list), its manifest version is supported, **every required file is present, listed in `artifacts` and matches its digest**, and every listed file matches. A required file without a checksum entry is rejected outright (it would otherwise escape verification). Files present on disk but not listed are never read; the UI lists them as unverified extras. Size caps apply before reading (manifest 8 MB, artifacts 256 MB, tables/JSON 16 MB, source dataset 600 MB) and `validation_scores.csv` is parsed with a 1,000,000-row cap enforced during parsing; its columns, 0/1 labels, unique non-negative integer row ids and finite scores in [0, 1] are validated before anything renders.

**Verified snapshot, not a one-time check.** Each consumed file is read once, its bytes are hashed against the manifest, and the table/JSON is parsed *from those bytes*. The loaded run is cached under a *content key* - a digest of the manifest bytes plus the digest of every file in the directory - that the app recomputes on every rerun. Changing any artifact afterwards (same size, preserved timestamps, unchanged manifest mtime included) changes the key, forces a reload and surfaces a verification error instead of stale "verified" data. Figures and the Logistic Regression coefficient file are re-read and re-hashed at the moment they are displayed. Note that switching tabs is handled in the browser and does not rerun the script; re-verification happens on the next widget interaction (slider, selectbox, checkbox, button), which is when a modified run is rejected.

**Cross-file consistency.** Before rendering, the loader also checks that the files describe the same run: the selected model is a known learned model with a score column, a threshold decision, both `model_comparison.csv` policies and an entry in `validation_summary.json`; every threshold is finite and in [0, 1]; claimed validation precision/recall are recomputed from the stored labels and scores (tolerance 1e-9); run id, dataset source and checksum, feature order and split counts agree between the manifest, `data_quality.json`, `analysis_summary.json`, `split_summary.json`, `feature_importance.csv` and `validation_scores.csv`. Failures are reported as a controlled error, never as an exception.

**Explanations are checked, not assumed.** Before a Logistic Regression explanation is shown, the coefficient file is re-verified against the manifest and then made to reproduce the stored validation scores for up to 200 checksum-verified source rows (tolerance 1e-6). A file with a valid checksum but the wrong coefficients disables the explanation with a message; if the source CSV is unavailable the report can still be viewed but no explanation is offered.

All of these establish *internal consistency*: the directory agrees with its own manifest and with itself. They do not authenticate who produced it or whether the declared dataset source is genuine - see *Provenance labels*.

## Provenance labels

The declared `--source` is a claim by whoever produced the run. A CSV with the Kaggle column layout is not automatically the historical dataset, so the UI classifies every run from evidence:

| Label | Meaning |
|---|---|
| frozen historical artifacts | the committed `output/analysis/` files (Historical Results mode) |
| historical reproduction | dataset SHA-256 equals the documented `76274b69...551a89`, features are `Time, V1-V28, Amount`, the historical configuration was used (seed 42, 64/16/20, recall target 0.8, drop_exact, RF 200/12/2/sqrt/balanced_subsample) **and** the comparison checks against the frozen artifacts pass (selected model, thresholds, validation AP, `model_comparison.csv` columns, feature importances, duplicate count, split sizes) |
| new run on the recognized dataset | documented dataset digest and schema, but different configuration or results |
| declared-source run | anything else declared `kaggle`/`openml`; the UI states that dataset identity is not verified |
| synthetic | `--source synthetic` fixtures |

The frozen test table is titled accordingly ("frozen test results for this run (declared source, dataset identity not verified)", "... historical reproduction (untouched test split)", ...). The same label appears in exported reports (`Provenance` line) and configurations (`provenance`, `provenance_label`). No manifest flag can promote a run to "historical reproduction"; the classification is recomputed from the files.

## Modes

Modes never fall back to one another. If a mode's inputs are missing or fail verification you see an error, not another mode's data.

| Mode | Data shown | Needs | What it cannot do |
|---|---|---|---|
| **Historical Results** | The committed July-2026 files in `output/analysis/` (metrics, figures, feature importance), exactly as cited by the report | Nothing beyond the repository | No threshold explorer or review queue: the historical export has no row-level validation predictions. The panel *Missing provenance* lists everything that was not recorded (no manifest, no dataset checksum, no environment record, model file not loadable). |
| **Synthetic Demo** | A run on a deterministic synthetic fixture created **once** (button) and reused afterwards; every screen carries a SYNTHETIC banner | Nothing; no downloads or keys | Says nothing about real transactions or the report's numbers |
| **Reproduced Experiment** | A run directory under `output/runs/` produced by `fraud-pipeline reproduce-historical` (or `fraud-pipeline run`), opened only after manifest, checksum and version checks pass | The Kaggle CSV and ~5 minutes to produce the run locally | Rejects synthetic runs, tampered artifacts, unknown manifest versions and directories without a manifest |

![Synthetic Demo - threshold explorer](screenshots/synthetic_threshold_explorer.png)

## Tabs (run-backed modes)

- **Overview** - dataset quality (rows, duplicates removed, class counts, prevalence, Time-column note), split summary with membership fingerprints, model selection basis.
- **Model comparison** - the frozen test-set table for Dummy Prior, Logistic Regression and Random Forest at the default 0.5 and the validation-selected operating point. `average_precision` and `pr_auc_trapezoidal` are separate columns on purpose.
- **Threshold explorer** - operates on **validation predictions only**. The slider keeps a separate value per run and per model; *Reset to validation-selected* restores that model's threshold and the value survives later interactions. The review queue, cost scenario and exports all read the same active (run, model, threshold). Metrics: alerts, precision, recall, false positives, missed fraud, alert rate, deltas versus the selected point; validation PR curves; review-capacity scenarios (threshold at the N-th highest score, realised alert count shown because ties can exceed N; a blank field shows an instruction, invalid tokens show an error naming the token, and exports omit the table until the input is valid again). Below it the **frozen test result** is repeated and locked: moving the slider never re-evaluates the test set, and an explored threshold is not a newly validated operating point.
- **Review queue** - the flagged validation rows at the explorer threshold for the *same model*, highest score first. Ground-truth labels are hidden by default and, when revealed, appear in a column named `ground_truth_label`; analyst annotations (`confirmed_fraud`, `legitimate`, `escalate`) live in a separate store and CSV export. Optionally loads the transaction features from the source CSV after verifying its SHA-256 against the manifest. No retraining happens. The local explanation panel is bound to the model whose score is displayed: for Logistic Regression it shows the additive log-odds terms from the run's numeric coefficient file (and the reconstructed score next to the displayed score); for Random Forest it states that no local explanation is available and points to the global importance tab with its non-causal disclaimer. The winning model is never substituted.
- **Features** - global Random Forest impurity importance, labelled as model reliance rather than individual causal explanation.
- **Cost scenario** - requires an explicit acknowledgement and your own hypothetical unit costs; the output is labelled a scenario, never realized savings.
- **Exports** - download the analytical report (Markdown) and the configuration (JSON), or save both under `output/ui_exports/`. See [`example_report_synthetic.md`](example_report_synthetic.md) and [`example_config_synthetic.json`](example_config_synthetic.json).
- **Manifest & limitations** - the full run manifest, the verification result, the limitations list and the validation-vs-test recall gap.

![Reproduced Experiment - manifest](screenshots/reproduced_manifest.png)

## Input handling and safety

- See *Artifact verification* and *Trust boundary* above.
- The UI does not accept bank statements or transaction exports. `V1`-`V28` are anonymized PCA components whose transformation is not public, so the model cannot score other data.
- Scores are uncalibrated model outputs, not fraud probabilities.

## Tests

```bash
pytest -q tests/test_ui_core.py tests/test_ui_app.py
```

`test_ui_core.py` checks every UI calculation against the core evaluation functions, loader error states, split separation, annotation separation, the linear explanation against the fitted model, and exports. `test_review_round2.py` covers stale-cache rejection (figure, report table, coefficient file changed after loading), cross-file validation, the substituted-coefficient case, provenance classification (generated Kaggle-layout fixtures are *declared-source*, never historical), workspace resolution, run-id confinement, duplicate handling under `keep`, CSV limits and Markdown escaping. `test_run_integrity.py` covers manifest coverage (missing checksum entry, missing file, changed file), malformed manifests/JSON/CSV and the byte/row caps. `test_ui_app.py` drives the app headlessly with `streamlit.testing.v1.AppTest`: all three modes and their error paths, a complete Reproduced Experiment workflow on a generated run with the Kaggle column layout (declared `source=kaggle` for layout only - it is not the ULB data and proves nothing about the historical reproduction), the joblib sentinel, threshold reset/state per model and run, capacity-field recovery, and explanation binding in both winner configurations.

## Manual checklist (after launching)

1. Historical Results opens without errors; the *Missing provenance* panel lists the gaps; no Threshold explorer tab.
2. Synthetic Demo: click *Create synthetic demo run* once; a second visit reuses it (no training).
3. Threshold explorer: move the slider, click *Reset to validation-selected* - the slider snaps back and Alerts returns to the "+0 vs selected" state; change the queue length - the threshold stays.
4. Switch model to Logistic Regression, move its slider, switch back - Random Forest keeps its own value.
5. Clear the capacity field (instruction appears), type `abc` (error names the token), type `10, 25` (table returns).
6. Review queue: header shows the same model and threshold as the explorer; enable feature loading; the explanation expander is titled with the displayed model and, for Random Forest, says no local explanation is available.
7. Exports: the preview states the same model and threshold; with an invalid capacity field it says no capacity table is included.
8. Reproduced Experiment with no runs shows an error and no data from other modes.
9. With a run open, edit any byte of `figures/04_precision_recall_curves.png` (keep the size) and move a slider: the run is rejected with a verification error and the "verified: yes" caption disappears (switching tabs alone does not rerun the script).
10. Review queue: type a note for one row without saving, switch rows - the other row's note field is empty; switch back - the draft is still there; saved notes reappear when the row is reselected.
11. The provenance banner for a generated Kaggle-layout run reads "dataset identity NOT verified"; only a run on the documented ULB CSV with the historical configuration and passing comparison is called a historical reproduction.

# Explainable Fraud Detection

A reproducible study of credit-card fraud detection: how to pick an alert threshold you can defend, how to prove the numbers weren't massaged, and a local dashboard that lets a non-programmer feel the trade-off between catching fraud and annoying customers.

[![CI](https://github.com/Rekh-225/Explainable-Fraud-Detection-PISU-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/Rekh-225/Explainable-Fraud-Detection-PISU-2026/actions/workflows/ci.yml)
[![Python 3.12 | 3.13](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)

Built for the *DX: Digital Transformation for Business* course at Prishtina International Summer University 2026 (University of Prishtina). The written report is in [`report/`](report/Rehan_Khaliq_Fraud_Detection_Report_FINAL.pdf); this repository is the code and evidence behind it.

---

## The problem

In the public [ULB credit-card dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), **492 of 284,807 payments are fraud – 0.17%.** A model that says "never fraud" is 99.8% accurate and completely useless.

So accuracy is the wrong question. The real questions are:

- If I flag the 100 most suspicious payments, how many are actually fraud? (*precision*)
- What share of all fraud did I catch? (*recall*)
- My review team can check 50 alerts a day. Where should I set the bar, and what will I miss?
- Can anyone check that these numbers are real, or do they have to take my word for it?

This project answers those questions for one well-known dataset, and packages the answer so it can be re-run, verified, and explored.

## What is in this repository

**1. The experiment.** Three models (a no-skill baseline, Logistic Regression, Random Forest) trained on 64% of the data, compared on a 16% validation slice, and scored *once* on a 20% test slice that was never used for any decision. The alert threshold is chosen on validation data only, with a stated rule: *catch at least 80% of fraud, then minimise false alarms.*

**2. The evidence.** Every run writes a `run_manifest.json`: a fingerprint (SHA-256) of the input file, which rows went into which split, all model settings, library versions, the Git commit, and a checksum of every output file. `fraud-pipeline verify-run` re-hashes a run folder and reports anything missing, changed, or unlisted. 245 automated tests run on generated data, so the code can be checked without downloading anything. The historical results in `output/analysis/` have been regenerated from scratch three times with every compared number exactly equal.

**3. The dashboard.** A Streamlit app that runs only on your machine (bound to `127.0.0.1`, telemetry off). Drag the threshold, watch alerts / caught fraud / false alarms move; enter your team's review capacity; work through a simulated review queue; export a report. It never loads pickled model files and it labels clearly whether you are looking at frozen historical results, synthetic demo data, or a checksum-verified new run.

<p align="center">
  <img src="docs/screenshots/reproduced_threshold_explorer.png" width="80%" alt="Threshold explorer: slider over validation predictions, with alerts, precision, recall, false positives and missed fraud updating">
</p>

## Results

Random Forest had the best validation Average Precision, so it was selected. Its threshold, **0.6735**, was chosen on validation data to reach at least 80% recall. On the untouched test set (56,746 payments, 95 fraud):

| Metric | Test result |
|---|---:|
| Alerts raised | 69 |
| ... of which real fraud (true positives) | 66 |
| ... false alarms (false positives) | 3 |
| Fraud missed (false negatives) | 29 |
| Precision | 95.7% |
| Recall | 69.5% |
| F1 | 0.805 |
| Average Precision | 0.805 |
| ROC-AUC | 0.965 |
| False alarms per 1,000 legitimate payments | 0.053 |

**Read the recall line carefully.** The threshold was tuned to catch 80% of fraud on the validation set, but it caught 69.5% on the test set. That gap is the most useful result in the project: it's what happens when a policy tuned on one sample meets fresh data, and it is exactly why the dashboard shows the frozen test result next to the slider instead of letting you re-tune on it.

<p align="center">
  <img src="output/analysis/figures/04_precision_recall_curves.png" width="48%" alt="Precision-recall curves">
  <img src="output/analysis/figures/05_confusion_matrices.png" width="48%" alt="Confusion matrices at the validation-selected thresholds">
</p>

Full tables: [`output/analysis/model_comparison.csv`](output/analysis/model_comparison.csv), [`analysis_summary.json`](output/analysis/analysis_summary.json), [`feature_importance.csv`](output/analysis/feature_importance.csv).

## Try it in five minutes

No dataset download needed for this part.

```bash
git clone https://github.com/Rekh-225/Explainable-Fraud-Detection-PISU-2026.git
cd Explainable-Fraud-Detection-PISU-2026
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell;  macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt -e .

pytest -q                           # ~2 min: 245 tests on generated data
streamlit run src/fraud_pipeline/ui/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

In the dashboard sidebar:

- **Historical Results** – the report's numbers exactly as committed, plus an honest list of what the original July run did *not* record (no manifest, no row-level predictions).
- **Synthetic Demo** – click *Create synthetic demo run*. A tiny model is trained on generated data so you can use every control. Everything is stamped SYNTHETIC.
- **Reproduced Experiment** – appears once you have run the real experiment (next section).

A walkthrough with a manual checklist is in [`docs/ui_guide.md`](docs/ui_guide.md); what the model can and cannot be used for is in [`docs/model_card.md`](docs/model_card.md).

## Reproduce the real experiment

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (free account required; the file is not redistributed here) and place it at `data/raw/creditcardfraud/creditcard.csv`. Then:

```bash
fraud-pipeline reproduce-historical          # ~5 minutes on one core
fraud-pipeline verify-run output/runs/<run_id>
```

This uses the original settings (seed 42, exact-duplicate removal, stratified 64/16/20 split, 200-tree Random Forest) and writes to a new folder under `output/runs/`. It never overwrites `output/analysis/`. Open the new run in the dashboard's *Reproduced Experiment* mode: the app checks the dataset fingerprint and compares the run against the frozen results before it will call it a historical reproduction.

**Reproduction status.** Re-run on 2026-09-27, 2026-09-28 and 2026-09-29 (Python 3.13.14, scikit-learn 1.9.0, NumPy 2.3.2, pandas 2.3.2) against a `creditcard.csv` with SHA-256 `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`. Exact equality with `output/analysis/` for: duplicates removed (1,081), cleaned class counts, split sizes, selected model, validation Average Precision of all three models, all thresholds, every historical column of `model_comparison.csv` including confusion counts, all feature importances, and the selected model's test metrics. New runs add a `pr_auc_trapezoidal` column and a numeric coefficient file; manifests necessarily differ in timestamps and paths. Run folders are git-ignored, so an independent check means running the command yourself.

## How the experiment works

| Step | What happens | Why |
|---|---|---|
| Load | The CSV path and its source (`kaggle`, `openml`, `synthetic`) must be stated explicitly. Missing columns, non-numeric values, or labels other than 0/1 are rejected. | No silent guessing or substitution. The OpenML mirror lacks the `Time` column, so its results are not comparable and it is never used as a fallback. |
| Clean | 1,081 exact duplicate rows are removed. | The same payment must not appear in both training and test. |
| Split | Stratified random 64/16/20 train / validation / test, seed 42. Membership fingerprints are recorded. | Training teaches the model; validation makes every decision; test is looked at once. |
| Train | Dummy Prior (baseline), Logistic Regression (standardised features, balanced class weights), Random Forest (200 trees, balanced subsampling, bounded depth). Scaling is fitted on training rows only. | Two genuinely different model families plus a floor to beat. |
| Select | The learned model with the highest **validation** Average Precision wins. | AP is the right summary when positives are rare. |
| Threshold | On validation scores, keep thresholds reaching ≥ 80% recall, pick the one with the highest precision (ties: highest threshold). | Turns a score into a policy with a stated business rule. |
| Evaluate | The selected model and threshold are applied once to the test split. | The honest number. Any re-tuning here would be cheating. |

## What the safety checks do and don't prove

The dashboard and CLI go further than most student projects on integrity, so it's worth being precise about what that buys you.

- **Content, not timestamps.** The UI re-hashes every file it displays on every interaction. Editing a figure or coefficient file – even keeping the size and modification time identical – makes the run show a controlled error instead of stale "verified" data.
- **Cross-file consistency.** The selected model must exist in every table; thresholds must be in [0, 1]; the claimed validation precision/recall are recomputed from the stored row-level scores; Logistic Regression explanations are only shown if the coefficient file actually reproduces the stored scores.
- **No code execution from data.** The UI never unpickles a model. Explanations come from a plain JSON file of coefficients. Loading the `.joblib` bundle requires an explicit `trusted_source=True` from *you*.
- **Provenance from evidence, not flags.** A run is labelled *historical reproduction* only if its dataset SHA-256 matches the documented value, its configuration matches, and its results match `output/analysis/`. A Kaggle-shaped CSV declared with `--source kaggle` is labelled a *declared-source run*; synthetic runs are always labelled synthetic.

**The limit:** these checks establish that a run folder is internally consistent and unchanged since it was written. They cannot prove *who* wrote it or that its input was genuine. Checksums are integrity, not authenticity.

<details>
<summary><strong>Command reference</strong></summary>

```bash
fraud-pipeline run --dataset PATH --source kaggle|openml|synthetic
    [--seed 42] [--minimum-recall 0.8] [--duplicate-policy drop_exact|keep] [--rf-estimators 200]
    [--run-id ID] [--no-figures] [--workspace-root DIR] [--output-root DIR]
    [--max-dataset-bytes N] [--max-dataset-rows N]
fraud-pipeline synthetic --output data/synthetic.csv --rows 4000 --seed 0 --duplicates 25 [--no-time]
fraud-pipeline verify-run output/runs/<run_id>
fraud-pipeline reproduce-historical              # = python src/run_analysis.py
python -m fraud_pipeline ...                     # same as fraud-pipeline ...
```

- **Paths.** Defaults resolve from the workspace root: `--workspace-root`, else `$FRAUD_PIPELINE_ROOT`, else the current directory. An installed wheel run from another folder writes to `<that folder>/output/runs`, never into `site-packages`.
- **Run ids** must be a single filename-safe component (`A-Za-z0-9._-`). Traversal, separators, drive letters and reserved names are rejected; the resolved folder must stay inside `--output-root`; a non-empty folder is never reused.
- **Size limits.** Datasets above `--max-dataset-bytes` (default 2 GiB) are refused before parsing; above `--max-dataset-rows` (default 5,000,000) during parsing.
- **Duplicates.** `--duplicate-policy keep` is accepted only when the file contains no exact duplicates; otherwise identical rows would land in different splits and be evaluated as independent, so the run is refused with a pointer to `drop_exact` (the historical default). `split_summary.json` reports `content_overlap` for every run.
- **Run folder contents.** `run_manifest.json`, the same tables and figures as `output/analysis/`, `validation_scores.csv` (row-level validation scores; test scores are deliberately not exported), `models/selected_model.joblib` + schema, and `models/logistic_regression_linear.json` (feature order, scaler parameters, coefficients, intercept).
- **Installing.** `pip install .` gives the CLI; `pip install ".[ui]"` adds Streamlit; `pip install ".[dev]"` adds pytest. `requirements.lock.txt` is the exact frozen environment the results were verified with.

</details>

## Repository layout

```text
src/fraud_pipeline/      the pipeline as a package (data, splitting, modeling, thresholds, evaluation, manifest, cli)
src/fraud_pipeline/ui/   Streamlit dashboard: core.py = all calculations (tested), app.py = layout
src/run_analysis.py      original entry point, now a thin wrapper around reproduce-historical
tests/                   245 tests on deterministic synthetic data, incl. headless UI tests
docs/                    UI guide, model card, example synthetic report, screenshots
output/analysis/         frozen historical results (July 2026) – never regenerated in place
output/runs/<run_id>/    new runs (git-ignored)
report/                  the written report (PDF, DOCX)
.github/workflows/       CI: tests + synthetic smoke run on Python 3.12 and 3.13
```

## Limitations

This is an offline academic study, not a payment-control system.

- The data is two days of European transactions from 2013. Fraud patterns drift; nothing here measures that.
- Features `V1`–`V28` are anonymised principal components. The model cannot score anyone's real bank statement, and "importance" of `V14` explains nothing a human can act on.
- A random split does not simulate the future; a time-based split would be the next honest step.
- The dataset has no cost information. The dashboard's cost scenario uses numbers *you* type in and says so.
- Random Forest has no per-transaction explanation here; only Logistic Regression does.
- Good numbers on this benchmark say nothing about fairness, robustness, or production readiness. A real deployment needs human review, monitoring, audit logs, access control, and regulatory assessment.

## The report

The [report](report/Rehan_Khaliq_Fraud_Detection_Report_FINAL.pdf) places the experiment in a wider design: an Observe–Orient–Decide–Act loop with specialised agents for data collection, prediction, explanation, response, and human review. That architecture is a proposal discussed on paper; only the offline analysis and the local dashboard are implemented here.

## Citation

```bibtex
@misc{khaliq2026explainablefraud,
  author       = {Rehan Khaliq},
  title        = {Explainable AI-Powered Fraud Detection and Risk Analytics for Digital Payment Transactions},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/Rekh-225/Explainable-Fraud-Detection-PISU-2026}
}
```

## Acknowledgments

Completed during PISU 2026 at the University of Prishtina, course *DX: Digital Transformation for Business – MCP-Based RAG*. Thanks to Prof. Kohei Arai, Prof. Arbnor Pajaziti, the PISU organisers and fellow students for their teaching and feedback. The dataset was made available by the ULB Machine Learning Group via Kaggle (also OpenML data ID 1597).

## Author and licence

**Rehan Khaliq** – [github.com/Rekh-225](https://github.com/Rekh-225)

Public for academic review and reproducibility. No open-source licence has been added yet, so standard copyright applies; if one is added later it should state whether it covers the report and generated artifacts as well as the code.

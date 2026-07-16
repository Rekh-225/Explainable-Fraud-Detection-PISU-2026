# Explainable Fraud Detection and Risk Analytics

This repository contains the reproducible analysis for my Prishtina International Summer University 2026 student report, **"Explainable AI-Powered Fraud Detection and Risk Analytics for Digital Payment Transactions."**

The project uses the real, anonymized ULB credit-card fraud dataset and treats fraud detection as a rare-event classification problem. It compares a no-skill baseline, class-weighted Logistic Regression, and Random Forest, then selects an operating threshold using validation data only.

## Verified result

The selected Random Forest achieved the following results on the untouched test set:

| Metric | Result |
|---|---:|
| Precision | 95.7% |
| Recall | 69.5% |
| F1 score | 0.805 |
| Average Precision | 0.805 |
| True positives | 66 |
| False positives | 3 |
| False negatives | 29 |
| True negatives | 56,648 |

These results are for an offline academic prototype and should not be interpreted as production deployment performance.

## Method

- Validate data quality and remove exact duplicate rows before splitting.
- Preserve the natural fraud prevalence with a stratified 64/16/20 train-validation-test split.
- Verify that the three partitions have no row-index overlap.
- Compare Dummy Prior, class-weighted Logistic Regression, and Random Forest.
- Select the alert threshold on validation data only.
- Evaluate precision, recall, F1, Average Precision, ROC-AUC, confusion matrices, false positives per 1,000 legitimate transactions, and alert rate.
- Generate report-ready figures and machine-readable result files.

## Data

The preferred source is the [ULB credit-card fraud dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Place `creditcard.csv` at:

```text
data/raw/creditcardfraud/creditcard.csv
```

If the file is not present, the script attempts to retrieve OpenML data ID 1597. Raw data is intentionally excluded from GitHub.

`Time` records elapsed seconds since the first transaction; it is not a calendar timestamp. The dataset source does not specify a currency for `Amount`.

## Reproduce the analysis

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/run_analysis.py
```

The main program is [`src/run_analysis.py`](src/run_analysis.py). It writes the trained model to `output/analysis/models/` locally and regenerates the tables, JSON summaries, and figures under `output/analysis/`.

## Repository contents

```text
src/run_analysis.py          Reproducible training and evaluation pipeline
output/analysis/             Verified metrics, summaries, and six figures
report/                      Final student report in DOCX format
requirements.txt             Python dependencies
```

## Report

The final submission document is available at [`report/Rehan_Khaliq_Fraud_Detection_Report_FINAL.docx`](report/Rehan_Khaliq_Fraud_Detection_Report_FINAL.docx).

## Limitations

- The transactions cover only two days in September 2013.
- Features `V1`-`V28` are anonymized PCA components without business-semantic labels.
- A stratified random split does not simulate future fraud drift.
- The dataset does not provide a reliable investigation-cost or fraud-loss matrix.
- This work is an offline academic prototype, not a production payment control.

# Model card - PISU 2026 fraud-detection prototype

## Model

- **Selected model:** Random Forest (200 trees, max depth 12, min leaf 2, `sqrt` features, `balanced_subsample` class weight, seed 42), chosen over a class-weighted Logistic Regression by validation Average Precision (0.871 vs 0.814). A Dummy Prior classifier is the no-skill reference.
- **Operating threshold:** 0.6735 on the Random Forest score, selected on the validation split as the highest-precision threshold with validation recall >= 80% (ties -> highest threshold). Decision rule: `score >= threshold`.
- **Outputs:** uncalibrated scores in [0, 1]. They are not probabilities of fraud and were not calibrated.
- **Code:** `fraud_pipeline` (this repository). Historical run July 2026; bit-identical reproduction 2026-09-27 (see README, *Reproduction status*).

## Data

- **Source:** ULB Credit Card Fraud dataset, Kaggle release `creditcard.csv` (SHA-256 `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`). 284,807 transactions by European cardholders over **two days in September 2013**; 492 fraud cases (0.173%).
- **Data age:** the data are 13 years old at the time of writing. Payment behaviour, fraud tactics and merchant mixes have changed; nothing here has been validated on recent data.
- **Features:** `Time` (seconds since the first transaction), `V1`-`V28` (anonymized PCA components - the transformation is not public) and `Amount`. The OpenML mirror omits `Time` and is therefore not equivalent.
- **Cleaning:** 1,081 exact-duplicate rows removed (283,726 rows, 473 fraud remain). No missing values.
- **Split:** stratified random 64/16/20 train/validation/test, seed 42, disjoint by construction. Learned preprocessing (scaler) is fit on the training split only.

## Performance (historical, frozen)

| Split | Threshold | Precision | Recall | Notes |
|---|---:|---:|---:|---|
| Validation (45,396 rows, 76 fraud) | 0.6735 | 0.969 | 0.816 | threshold selected here |
| Test (56,746 rows, 95 fraud) | 0.6735 | 0.957 | 0.695 | untouched; AP 0.805, ROC-AUC 0.965; 66 TP, 3 FP, 29 FN |

**Validation-test recall gap:** the policy targeted >= 80% recall and achieved 81.6% on validation, but only **69.5% on the test set** (29 of 95 fraud cases missed). With 76 and 95 positives respectively the estimates are noisy, and the selection procedure is optimistic for the split it was selected on. The gap is reported rather than hidden; it is the main reason the operating point should be re-validated on new data before any operational use.

## Intended use

Offline academic study of rare-event classification, threshold selection and honest evaluation on a public benchmark. The local UI supports exploring validation trade-offs, simulating a review queue and exporting reports for discussion.

## Out-of-scope uses

- Approving, declining, blocking or investigating real transactions.
- Scoring bank statements, transaction exports or any data other than this dataset's feature layout. `V1`-`V28` cannot be computed for new data.
- Treating scores as calibrated probabilities, or treating global feature importance as an explanation of an individual decision.
- Claiming a new validated threshold from the UI's explorer: it evaluates validation predictions only and never re-scores the test set.

## Limitations and risks

- **Random split, no time dimension.** A stratified random split does not simulate future drift, new fraud patterns or the delay between transaction and label. Out-of-time validation was not performed.
- **Two days of 2013 data.** Seasonality, long-run behaviour and current fraud tactics are absent.
- **Anonymized features.** No business semantics, so fairness across customer groups cannot be assessed and explanations are limited to model reliance.
- **No cost information.** The dataset has no investigation-cost or loss data; the UI's cost scenario uses only user-supplied hypothetical numbers and labels the result a scenario.
- **Small positive counts.** 76 validation and 95 test fraud cases make every recall figure uncertain by several percentage points.
- **Duplicates.** Exact duplicates were removed; near-duplicates were not analysed.

## Provenance

- Historical artifacts (`output/analysis/`) have no run manifest, dataset checksum, environment record or row-level predictions; the UI's Historical mode lists these gaps.
- Runs produced with `fraud-pipeline reproduce-historical` carry a full manifest (dataset SHA-256, feature order, split fingerprints, model parameters, package versions, code commit, artifact checksums) and can be opened in Reproduced Experiment mode after verification.

## Pending full-data checks

- Out-of-time evaluation and probability calibration have not been run.
- The reproduction run directory is local and git-ignored; anyone auditing must re-run it against the Kaggle CSV with the pinned environment.

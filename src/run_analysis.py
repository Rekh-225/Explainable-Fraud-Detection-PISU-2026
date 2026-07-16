"""Reproducible fraud-detection analysis for the PISU 2026 student report.

The implementation intentionally stays small and auditable:
- real-world anonymized ULB transaction data from the authoritative Kaggle release;
- exact-duplicate removal before splitting;
- stratified train/validation/test sets;
- Dummy, Logistic Regression, and Random Forest models;
- validation-only operating-threshold selection;
- business-facing rare-event metrics and report-ready figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE = PROJECT_ROOT / "data" / "cache"
KAGGLE_CSV = PROJECT_ROOT / "data" / "raw" / "creditcardfraud" / "creditcard.csv"
OPENML_CSV = PROJECT_ROOT / "data" / "raw" / "creditcard_openml_1597.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

ULB_DATASET_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
OPENML_DATA_ID = 1597


def prepare_directories() -> None:
    for path in (DATA_CACHE, KAGGLE_CSV.parent, OUTPUT_DIR, FIGURE_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """Prefer the authoritative Kaggle CSV; fall back to OpenML if needed."""
    if KAGGLE_CSV.exists():
        frame = pd.read_csv(KAGGLE_CSV)
    elif OPENML_CSV.exists():
        frame = pd.read_csv(OPENML_CSV)
    else:
        dataset = fetch_openml(
            data_id=OPENML_DATA_ID,
            as_frame=True,
            data_home=DATA_CACHE,
            parser="pandas",
        )
        frame = dataset.frame.copy()
        if "Class" not in frame.columns:
            frame["Class"] = dataset.target
        frame.to_csv(OPENML_CSV, index=False)

    expected_features = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    missing = sorted(set(expected_features) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    ordered_columns = (["Time"] if "Time" in frame.columns else []) + expected_features
    frame = frame[ordered_columns].copy()
    frame["Class"] = frame["Class"].astype(int)
    for column in frame.columns.drop("Class"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def select_operating_threshold(
    y_true: pd.Series,
    scores: np.ndarray,
    minimum_recall: float = 0.80,
) -> tuple[float, dict[str, Any]]:
    """Select the highest-precision validation threshold reaching target recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision_at_threshold = precision[:-1]
    recall_at_threshold = recall[:-1]

    feasible = np.flatnonzero(recall_at_threshold >= minimum_recall)
    if feasible.size:
        best_precision = precision_at_threshold[feasible].max()
        candidates = feasible[precision_at_threshold[feasible] == best_precision]
        # If precision ties, use the highest threshold to reduce review volume.
        index = candidates[np.argmax(thresholds[candidates])]
        rule = f"highest precision with validation recall >= {minimum_recall:.0%}"
    else:
        f1_values = np.divide(
            2 * precision_at_threshold * recall_at_threshold,
            precision_at_threshold + recall_at_threshold,
            out=np.zeros_like(precision_at_threshold),
            where=(precision_at_threshold + recall_at_threshold) > 0,
        )
        index = int(np.argmax(f1_values))
        rule = "maximum validation F1 because recall target was not reached"

    threshold = float(thresholds[index])
    details = {
        "threshold": threshold,
        "rule": rule,
        "validation_precision": float(precision_at_threshold[index]),
        "validation_recall": float(recall_at_threshold[index]),
    }
    return threshold, details


def evaluate_scores(
    model_name: str,
    policy_name: str,
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    legitimate_count = tn + fp
    alerts = tp + fp
    return {
        "model": model_name,
        "policy": policy_name,
        "threshold": float(threshold),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "average_precision": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positives_per_1000_legitimate": float(fp / legitimate_count * 1000),
        "alert_rate": float(alerts / len(y_true)),
        "test_rows": int(len(y_true)),
        "test_fraud_cases": int((y_true == 1).sum()),
    }


def plot_class_imbalance(frame: pd.DataFrame) -> None:
    counts = frame["Class"].value_counts().sort_index()
    labels = ["Legitimate", "Fraud"]
    colors = ["#4C78A8", "#D1495B"]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    bars = ax.bar(labels, counts.values, color=colors, width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("Number of transactions (log scale)")
    ax.set_title("Extreme Class Imbalance in the Transaction Dataset", pad=18)
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count * 1.10,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_class_imbalance.png", dpi=220)
    plt.close(fig)


def plot_amount_distribution(frame: pd.DataFrame) -> None:
    plot_frame = frame[["Amount", "Class"]].copy()
    plot_frame["Transaction type"] = plot_frame["Class"].map(
        {0: "Legitimate", 1: "Fraud"}
    )
    plot_frame["log1p_amount"] = np.log1p(plot_frame["Amount"].clip(lower=0))
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    sns.histplot(
        data=plot_frame,
        x="log1p_amount",
        hue="Transaction type",
        stat="density",
        common_norm=False,
        bins=60,
        element="step",
        fill=False,
        palette={"Legitimate": "#4C78A8", "Fraud": "#D1495B"},
        ax=ax,
    )
    ax.set_xlabel("log(1 + transaction amount)")
    ax.set_ylabel("Density within each class")
    ax.set_title("Transaction Amount Distributions by Class")
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_amount_distribution.png", dpi=220)
    plt.close(fig)


def plot_fraud_rate_by_amount_band(frame: pd.DataFrame) -> None:
    bins = [-np.inf, 0, 10, 50, 100, 500, np.inf]
    labels = ["0", "0-10", "10-50", "50-100", "100-500", "500+"]
    band = pd.cut(frame["Amount"], bins=bins, labels=labels, include_lowest=True)
    grouped = frame.assign(amount_band=band).groupby(
        "amount_band", observed=False
    )["Class"].agg(["count", "sum", "mean"])

    fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(grouped))
    ax1.bar(x, grouped["count"], color="#B7C9E2", label="Transactions")
    ax1.set_yscale("log")
    ax1.set_ylabel("Transactions (log scale)", color="#34526F")
    ax1.set_xticks(x, grouped.index.astype(str))
    ax1.set_xlabel("Transaction amount band")

    ax2 = ax1.twinx()
    ax2.plot(x, grouped["mean"] * 100, color="#D1495B", marker="o", linewidth=2)
    ax2.set_ylabel("Fraud rate (%)", color="#A32638")
    ax1.set_title("Transaction Volume and Observed Fraud Rate by Amount Band")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_fraud_rate_by_amount_band.png", dpi=220)
    plt.close(fig)


def plot_precision_recall_curves(
    y_test: pd.Series,
    model_scores: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    palette = {"Logistic Regression": "#4C78A8", "Random Forest": "#D1495B"}
    for model_name, scores in model_scores.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax.plot(
            recall,
            precision,
            linewidth=2.1,
            color=palette.get(model_name),
            label=f"{model_name} (AP={ap:.3f})",
        )
    prevalence = float(y_test.mean())
    ax.axhline(
        prevalence,
        color="#777777",
        linestyle="--",
        linewidth=1.2,
        label=f"No-skill baseline ({prevalence:.4f})",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Performance on the Unseen Test Set")
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.04), frameon=False)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_precision_recall_curves.png", dpi=220)
    plt.close(fig)


def plot_confusion_matrices(
    y_test: pd.Series,
    scores_by_model: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for ax, model_name in zip(axes, ("Logistic Regression", "Random Forest")):
        predictions = (scores_by_model[model_name] >= thresholds[model_name]).astype(int)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        sns.heatmap(
            matrix,
            annot=True,
            fmt=",d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=ax,
        )
        ax.set_title(f"{model_name}\nvalidation-selected threshold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Test-Set Confusion Matrices", y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_confusion_matrices.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    top = importance.head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.barh(top["feature"], top["importance"], color="#4C78A8")
    ax.set_xlabel("Random Forest impurity-based importance")
    ax.set_title("Most Influential Anonymized Model Features")
    ax.text(
        0,
        -0.16,
        "V1-V28 are PCA components; importance does not reveal original business meaning.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
    )
    sns.despine()
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.22)
    fig.savefig(
        FIGURE_DIR / "06_random_forest_feature_importance.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    return importance


def main() -> None:
    prepare_directories()
    sns.set_theme(style="whitegrid", context="notebook")

    original = load_dataset()
    original_rows = len(original)
    original_class_counts = original["Class"].value_counts().sort_index().to_dict()
    missing_values = int(original.isna().sum().sum())
    duplicate_rows = int(original.duplicated().sum())

    frame = original.drop_duplicates().reset_index(drop=True)
    cleaned_rows = len(frame)
    cleaned_class_counts = frame["Class"].value_counts().sort_index().to_dict()
    fraud_prevalence = float(frame["Class"].mean())

    data_quality = {
        "source": ULB_DATASET_URL,
        "openml_data_id": OPENML_DATA_ID,
        "source_note": (
            "Authoritative Kaggle CSV with Time, V1-V28, Amount and Class"
            if "Time" in original.columns
            else "OpenML mirror with V1-V28, Amount and Class; Time is omitted"
        ),
        "original_rows": original_rows,
        "original_columns": int(original.shape[1]),
        "original_class_counts": original_class_counts,
        "missing_values": missing_values,
        "exact_duplicate_rows_removed": duplicate_rows,
        "cleaned_rows": cleaned_rows,
        "cleaned_class_counts": cleaned_class_counts,
        "cleaned_fraud_prevalence": fraud_prevalence,
    }
    save_json(OUTPUT_DIR / "data_quality.json", data_quality)

    plot_class_imbalance(frame)
    plot_amount_distribution(frame)
    plot_fraud_rate_by_amount_band(frame)

    X = frame.drop(columns="Class")
    y = frame["Class"]

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.20,
        stratify=y_development,
        random_state=SEED,
    )

    split_summary = {
        "random_state": SEED,
        "split_method": "stratified random 64/16/20 train-validation-test",
        "train": {"rows": len(y_train), "fraud": int(y_train.sum()), "rate": y_train.mean()},
        "validation": {
            "rows": len(y_validation),
            "fraud": int(y_validation.sum()),
            "rate": y_validation.mean(),
        },
        "test": {"rows": len(y_test), "fraud": int(y_test.sum()), "rate": y_test.mean()},
        "index_overlap": {
            "train_validation": len(set(X_train.index) & set(X_validation.index)),
            "train_test": len(set(X_train.index) & set(X_test.index)),
            "validation_test": len(set(X_validation.index) & set(X_test.index)),
        },
    }
    save_json(OUTPUT_DIR / "split_summary.json", split_summary)

    models: dict[str, Any] = {
        "Dummy Prior": DummyClassifier(strategy="prior", random_state=SEED),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=SEED,
        ),
    }

    validation_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}
    validation_ap: dict[str, float] = {}
    operating_thresholds: dict[str, float] = {}
    threshold_details: dict[str, dict[str, Any]] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        validation_scores[model_name] = model.predict_proba(X_validation)[:, 1]
        test_scores[model_name] = model.predict_proba(X_test)[:, 1]
        validation_ap[model_name] = average_precision_score(
            y_validation, validation_scores[model_name]
        )
        if model_name == "Dummy Prior":
            operating_thresholds[model_name] = 0.5
            threshold_details[model_name] = {
                "threshold": 0.5,
                "rule": "fixed non-learning reference threshold",
            }
        else:
            threshold, details = select_operating_threshold(
                y_validation, validation_scores[model_name]
            )
            operating_thresholds[model_name] = threshold
            threshold_details[model_name] = details

    # Select the learned model with the highest validation Average Precision.
    learned_names = ["Logistic Regression", "Random Forest"]
    selected_model_name = max(learned_names, key=lambda name: validation_ap[name])

    result_rows: list[dict[str, Any]] = []
    for model_name in models:
        result_rows.append(
            evaluate_scores(
                model_name,
                "default_0.5",
                y_test,
                test_scores[model_name],
                0.5,
            )
        )
        if model_name != "Dummy Prior":
            result_rows.append(
                evaluate_scores(
                    model_name,
                    "validation_operating_point",
                    y_test,
                    test_scores[model_name],
                    operating_thresholds[model_name],
                )
            )

    results = pd.DataFrame(result_rows)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    validation_payload = {
        "validation_average_precision": validation_ap,
        "threshold_selection": threshold_details,
        "selected_model": selected_model_name,
        "selection_rule": "highest validation Average Precision among learned models",
    }
    save_json(OUTPUT_DIR / "validation_summary.json", validation_payload)

    plot_precision_recall_curves(
        y_test,
        {name: test_scores[name] for name in learned_names},
    )
    plot_confusion_matrices(
        y_test,
        {name: test_scores[name] for name in learned_names},
        {name: operating_thresholds[name] for name in learned_names},
    )
    feature_importance = plot_feature_importance(
        models["Random Forest"], list(X.columns)
    )

    selected_row = results[
        (results["model"] == selected_model_name)
        & (results["policy"] == "validation_operating_point")
    ].iloc[0].to_dict()

    summary = {
        "project_title": "Explainable AI-Powered Fraud Detection and Risk Analytics for Digital Payment Transactions",
        "data_quality": data_quality,
        "splits": split_summary,
        "selected_model": selected_model_name,
        "selected_threshold": operating_thresholds[selected_model_name],
        "selected_model_test_metrics": selected_row,
        "top_random_forest_features": feature_importance.head(10).to_dict("records"),
        "model_selection_basis": "validation Average Precision",
        "threshold_policy": threshold_details[selected_model_name],
        "limitations": [
            "The transactions cover only two days in September 2013.",
            "V1-V28 are anonymized PCA components without business-semantic labels.",
            (
                "Time records elapsed seconds from the first transaction, not calendar timestamps or local time."
                if "Time" in original.columns
                else "The OpenML mirror omits the original Time field."
            ),
            "A stratified random split does not simulate future fraud drift.",
            "The dataset contains no reliable investigation-cost or fraud-loss matrix.",
            "This is an offline academic prototype, not a production payment control.",
        ],
    }
    save_json(OUTPUT_DIR / "analysis_summary.json", summary)

    joblib.dump(
        {
            "model": models[selected_model_name],
            "threshold": operating_thresholds[selected_model_name],
            "features": list(X.columns),
            "selected_model_name": selected_model_name,
        },
        MODEL_DIR / "selected_model.joblib",
        compress=3,
    )

    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()








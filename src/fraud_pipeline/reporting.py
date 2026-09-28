"""Report-ready figures and tabular exports (moved from the original script)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

from .data import LABEL_COLUMN
from .modeling import FOREST_MODEL, LOGISTIC_MODEL

PALETTE = {LOGISTIC_MODEL: "#4C78A8", FOREST_MODEL: "#D1495B"}
VALIDATION_SCORES_FILENAME = "validation_scores.csv"


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")


def plot_class_imbalance(frame: pd.DataFrame, figure_dir: Path) -> None:
    counts = frame[LABEL_COLUMN].value_counts().sort_index()
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
    fig.savefig(figure_dir / "01_class_imbalance.png", dpi=220)
    plt.close(fig)


def plot_amount_distribution(frame: pd.DataFrame, figure_dir: Path) -> None:
    plot_frame = frame[["Amount", LABEL_COLUMN]].copy()
    plot_frame["Transaction type"] = plot_frame[LABEL_COLUMN].map({0: "Legitimate", 1: "Fraud"})
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
    fig.savefig(figure_dir / "02_amount_distribution.png", dpi=220)
    plt.close(fig)


def plot_fraud_rate_by_amount_band(frame: pd.DataFrame, figure_dir: Path) -> None:
    bins = [-np.inf, 0, 10, 50, 100, 500, np.inf]
    labels = ["0", "0-10", "10-50", "50-100", "100-500", "500+"]
    band = pd.cut(frame["Amount"], bins=bins, labels=labels, include_lowest=True)
    grouped = frame.assign(amount_band=band).groupby("amount_band", observed=False)[
        LABEL_COLUMN
    ].agg(["count", "sum", "mean"])

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
    fig.savefig(figure_dir / "03_fraud_rate_by_amount_band.png", dpi=220)
    plt.close(fig)


def plot_precision_recall_curves(
    y_test: pd.Series, model_scores: dict[str, np.ndarray], figure_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for model_name, scores in model_scores.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax.plot(
            recall,
            precision,
            linewidth=2.1,
            color=PALETTE.get(model_name),
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
    fig.savefig(figure_dir / "04_precision_recall_curves.png", dpi=220)
    plt.close(fig)


def plot_confusion_matrices(
    y_test: pd.Series,
    scores_by_model: dict[str, np.ndarray],
    thresholds: dict[str, float],
    figure_dir: Path,
) -> None:
    model_names = list(scores_by_model)
    fig, axes = plt.subplots(1, len(model_names), figsize=(4.7 * len(model_names), 4.2))
    for ax, model_name in zip(np.atleast_1d(axes), model_names):
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
    fig.savefig(figure_dir / "05_confusion_matrices.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def feature_importance_table(model: RandomForestClassifier, feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)


def plot_feature_importance(importance: pd.DataFrame, figure_dir: Path) -> None:
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
        figure_dir / "06_random_forest_feature_importance.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)


def export_validation_scores(
    run_dir: Path,
    row_ids: list[int],
    labels: pd.Series,
    scores_by_model: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> Path:
    """Row-level validation scores for the next (UI) milestone.

    Columns: ``row_id`` (position in the source CSV), ``split`` (always
    ``validation``), ``label`` and, per model, ``score__<slug>`` and
    ``flag__<slug>`` (``score >= threshold``). Test-set scores are
    deliberately not exported so the untouched test set stays untouched.
    """
    table = pd.DataFrame({"row_id": row_ids, "split": "validation", "label": labels.to_numpy()})
    for model_name, scores in scores_by_model.items():
        slug = model_name.lower().replace(" ", "_")
        table[f"score__{slug}"] = scores
        table[f"flag__{slug}"] = (scores >= thresholds[model_name]).astype(int)
    path = run_dir / VALIDATION_SCORES_FILENAME
    table.to_csv(path, index=False)
    return path

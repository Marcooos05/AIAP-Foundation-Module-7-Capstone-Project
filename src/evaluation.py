"""Classification metrics, threshold selection and evaluation plots for fitted model pipelines."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use('Agg')  # render to files only, so the pipeline runs without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def predict_with_threshold(model: Pipeline, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """
    Predicts the positive class (1) when its predicted probability is at least `threshold`.

    Args:
        model (Pipeline): A fitted pipeline exposing `predict_proba`.
        X (pd.DataFrame): Features.
        threshold (float): Decision threshold on the positive-class probability.
    Returns:
        np.ndarray: Predicted labels (0/1).
    """
    return (model.predict_proba(X)[:, 1] >= threshold).astype(int)


def metrics_from_scores(y_true: pd.Series, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes classification metrics from positive-class probabilities.

    Recall (sensitivity) is the primary metric: a false negative is a missed
    person at risk (README §5.1, §5.3). Accuracy is reported for completeness
    only and is never used to choose a model.

    Args:
        y_true (pd.Series): True labels (0/1).
        y_score (np.ndarray): Predicted positive-class probabilities.
        threshold (float): Decision threshold on the positive-class probability.
    Returns:
        Dict[str, Any]: Threshold, recall, specificity, precision, F1, F2,
            ROC-AUC, PR-AUC (average precision), Brier score, accuracy and the
            confusion matrix counts.
    """
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'threshold': float(threshold),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) else 0.0,
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'f2': float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_score)),
        'average_precision': float(average_precision_score(y_true, y_score)),
        'brier_score': float(brier_score_loss(y_true, y_score)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }


def compute_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes classification metrics for a fitted model (see `metrics_from_scores`).

    Args:
        model (Pipeline): A fitted pipeline exposing `predict_proba`.
        X (pd.DataFrame): Evaluation features.
        y (pd.Series): Evaluation target variable.
        threshold (float): Decision threshold on the positive-class probability.
    Returns:
        Dict[str, Any]: Classification metrics.
    """
    return metrics_from_scores(y, model.predict_proba(X)[:, 1], threshold)


def threshold_for_target_recall(y_true: pd.Series, y_score: np.ndarray, target_recall: float) -> Dict[str, float]:
    """
    Finds the highest decision threshold whose recall is still at least
    `target_recall`. A higher threshold flags fewer people, so this is the
    threshold with the fewest false positives that still meets the recall target.

    Args:
        y_true (pd.Series): True labels (0/1).
        y_score (np.ndarray): Predicted positive-class probabilities.
        target_recall (float): Minimum recall to keep, in (0, 1].
    Returns:
        Dict[str, float]: The threshold, and the recall and precision it gives.
    Raises:
        ValueError: If `y_true` has no positive cases.
    """
    if int(np.sum(y_true)) == 0:
        raise ValueError("Cannot tune a recall threshold without positive cases.")

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall have one more entry than thresholds; drop the final (recall 0) point.
    # Recall falls as the threshold rises, so the last qualifying index is the highest threshold.
    meets_target = np.flatnonzero(recall[:-1] >= target_recall)
    best = int(meets_target[-1])
    return {
        'threshold': float(thresholds[best]),
        'recall': float(recall[best]),
        'precision': float(precision[best]),
    }


def plot_model_comparison(cv_results: Dict[str, Dict[str, Any]], metric: str, path: Path) -> None:
    """Horizontal bar chart of each model's mean CV score (with ±1 SD error bars), best first."""
    rows = sorted(cv_results.items(), key=lambda item: item[1]['cv_mean'])
    names = [name for name, _ in rows]
    means = [result['cv_mean'] for _, result in rows]
    stds = [result['cv_std'] for _, result in rows]

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.5))
    ax.barh(names, means, xerr=stds, color='tab:blue', alpha=0.8, capsize=3)
    for y_pos, mean in enumerate(means):
        ax.text(mean + 0.01, y_pos, f"{mean:.3f}", va='center', fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"Mean CV {metric} (±1 SD)")
    ax.set_title(f"Model comparison: {metric} (stratified CV on training set)")
    _save_figure(fig, path)


def plot_roc_curves(models: Dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, path: Path) -> None:
    """ROC curves for one or more fitted models on the same data."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X, y, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], ls='--', color='grey', label='Random')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title("ROC curve")
    _save_figure(fig, path)


def plot_precision_recall_curves(models: Dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, path: Path) -> None:
    """Precision-recall curves for one or more fitted models, with the no-skill (prevalence) line."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        PrecisionRecallDisplay.from_estimator(model, X, y, name=name, ax=ax)
    ax.axhline(float(np.mean(y)), ls='--', color='grey', label=f"No skill ({np.mean(y):.2f})")
    ax.legend(loc='lower left', fontsize=8)
    ax.set_title("Precision-recall curve")
    _save_figure(fig, path)


def plot_confusion_matrix(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    path: Path,
    labels: List[str],
    threshold: float = 0.5,
    title: str = "Confusion matrix",
) -> None:
    """Confusion matrix of a fitted model at the given decision threshold."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y, predict_with_threshold(model, X, threshold), display_labels=labels, cmap='Blues', ax=ax, colorbar=False,
    )
    ax.set_title(title, fontsize=10)
    _save_figure(fig, path)


def plot_calibration_curves(models: Dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, path: Path) -> None:
    """
    Calibration (reliability) curves: predicted probability vs observed
    positive rate. Points on the diagonal mean the predicted risks can be
    taken at face value (lit. review #7). class_weight='balanced' pushes
    probabilities upward, so some over-prediction is expected.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        CalibrationDisplay.from_estimator(model, X, y, n_bins=5, strategy='quantile', name=name, ax=ax)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title("Calibration curve")
    _save_figure(fig, path)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved figure to %s", path)

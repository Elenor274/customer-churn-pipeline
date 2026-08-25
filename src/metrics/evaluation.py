"""Threshold-agnostic probability evaluation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)


def _as_binary_target(y_true: Any, pos_label: Any) -> np.ndarray:
    """Convert labels to a binary array using the requested positive label."""
    y_true_array = np.asarray(y_true)
    return (y_true_array == pos_label).astype(int)


def _validate_probabilities(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    y_true_array = np.asarray(y_true)
    y_prob_array = np.asarray(y_prob, dtype=float)

    if y_true_array.ndim != 1 or y_prob_array.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional sequences.")

    if y_true_array.shape[0] != y_prob_array.shape[0]:
        raise ValueError("y_true and y_prob must have the same number of samples.")

    return y_true_array, y_prob_array


def expected_calibration_error(
    y_true: Any,
    y_prob: Any,
    n_bins: int = 10,
    pos_label: Any = 1,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute expected calibration error and the calibration curve values."""
    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")

    y_true_array, y_prob_array = _validate_probabilities(y_true, y_prob)
    y_true_binary = _as_binary_target(y_true_array, pos_label)

    prob_true, prob_pred = calibration_curve(
        y_true_binary,
        y_prob_array,
        n_bins=n_bins,
        strategy="uniform",
    )

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_array, bin_edges[1:-1], right=True)
    bin_counts = np.bincount(bin_ids, minlength=n_bins)
    non_empty_counts = bin_counts[bin_counts > 0]
    weights = non_empty_counts / y_prob_array.shape[0]

    ece = float(np.sum(np.abs(prob_true - prob_pred) * weights))
    return ece, prob_true, prob_pred


def evaluate_probabilities(
    y_true: Any,
    y_prob: Any,
    pos_label: Any = 1,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Return threshold-free ranking and calibration metrics for probabilities."""
    y_true_array, y_prob_array = _validate_probabilities(y_true, y_prob)
    y_true_binary = _as_binary_target(y_true_array, pos_label)

    precision, recall, pr_thresholds = precision_recall_curve(
        y_true_binary,
        y_prob_array,
        pos_label=1,
    )
    average_precision = float(average_precision_score(y_true_binary, y_prob_array))
    brier_score = float(brier_score_loss(y_true_binary, y_prob_array))
    ece, prob_true, prob_pred = expected_calibration_error(
        y_true_array,
        y_prob_array,
        n_bins=n_bins,
        pos_label=pos_label,
    )

    return {
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
        "average_precision": average_precision,
        "brier_score": brier_score,
        "ece": ece,
        "calibration": {
            "prob_true": prob_true,
            "prob_pred": prob_pred,
            "n_bins": n_bins,
        },
    }

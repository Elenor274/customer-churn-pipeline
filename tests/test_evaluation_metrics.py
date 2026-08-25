import numpy as np
import pytest
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss

from src.metrics.evaluation import evaluate_probabilities, expected_calibration_error


def test_pr_metrics_basic() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])

    metrics = evaluate_probabilities(y_true, y_prob)

    assert len(metrics["precision"]) == len(metrics["recall"])
    assert len(metrics["pr_thresholds"]) == len(metrics["precision"]) - 1
    assert metrics["average_precision"] == pytest.approx(
        average_precision_score(y_true, y_prob)
    )
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["brier_score"] == pytest.approx(brier_score_loss(y_true, y_prob))


def test_calibration_perfect_groups() -> None:
    y_prob = np.array([0.2] * 10 + [0.8] * 10)
    y_true = np.array([1, 1] + [0] * 8 + [1] * 8 + [0, 0])

    ece, prob_true, prob_pred = expected_calibration_error(y_true, y_prob, n_bins=2)
    calibration = calibration_curve(y_true, y_prob, n_bins=2, strategy="uniform")
    metrics = evaluate_probabilities(y_true, y_prob, n_bins=2)

    assert ece == pytest.approx(0.0, abs=1e-12)
    assert prob_true == pytest.approx(np.array([0.2, 0.8]))
    assert prob_pred == pytest.approx(np.array([0.2, 0.8]))
    assert calibration[0] == pytest.approx(np.array([0.2, 0.8]))
    assert calibration[1] == pytest.approx(np.array([0.2, 0.8]))
    assert metrics["calibration"]["n_bins"] == 2
    assert metrics["ece"] == pytest.approx(0.0, abs=1e-12)

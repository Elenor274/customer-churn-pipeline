"""Compatibility package exposing evaluation helpers at the repository root."""

from src.metrics.evaluation import evaluate_probabilities, expected_calibration_error

__all__ = ["evaluate_probabilities", "expected_calibration_error"]

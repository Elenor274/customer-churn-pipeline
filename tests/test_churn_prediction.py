import pandas as pd
import pytest

from src.churn_prediction import prepare_features, train_and_evaluate


def sample_data(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": [f"C-{index:03d}" for index in range(rows)],
            "tenure": list(range(rows)),
            "MonthlyCharges": [45.0 + (index % 8) * 7 for index in range(rows)],
            "TotalCharges": [str(80 + index * 35) for index in range(rows)],
            "Contract": [
                "Month-to-month" if index % 2 else "Two year" for index in range(rows)
            ],
            "InternetService": [
                "Fiber optic" if index % 3 else "DSL" for index in range(rows)
            ],
            "Churn": ["Yes" if index % 2 else "No" for index in range(rows)],
        }
    )


def test_prepare_features_removes_identifier_and_encodes_target() -> None:
    features, target = prepare_features(sample_data())

    assert "customerID" not in features.columns
    assert "Churn" not in features.columns
    assert set(target.unique()) == {0, 1}


def test_train_and_evaluate_reports_all_metrics() -> None:
    models, metrics = train_and_evaluate(sample_data())

    assert set(models) == {"logistic_regression", "random_forest"}
    assert set(metrics["logistic_regression"]) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }
    assert all(
        0 <= value <= 1 for result in metrics.values() for value in result.values()
    )


def test_prepare_features_rejects_missing_target() -> None:
    with pytest.raises(ValueError, match="Churn"):
        prepare_features(sample_data().drop(columns="Churn"))

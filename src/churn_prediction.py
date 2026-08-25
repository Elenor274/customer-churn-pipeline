"""Train and evaluate reproducible customer-churn classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a churn CSV and validate the minimum required schema."""
    data = pd.read_csv(path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")
    return data


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean the dataset and split it into features and a binary target."""
    frame = data.copy()

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")

    if ID_COLUMN in frame.columns:
        frame = frame.drop(columns=ID_COLUMN)

    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(
            frame["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True),
            errors="coerce",
        )

    frame = frame.dropna(subset=[TARGET_COLUMN])
    target = frame.pop(TARGET_COLUMN).astype(str).str.strip().str.lower()
    target = target.map({"yes": 1, "no": 0, "1": 1, "0": 0})

    if target.isna().any() or target.nunique() != 2:
        raise ValueError("Churn must be a binary column using Yes/No or 1/0 values.")

    return frame, target.astype(int)


def build_model(features: pd.DataFrame, estimator: Any) -> Pipeline:
    """Build a leakage-safe preprocessing and modeling pipeline."""
    categorical_columns = features.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric_columns = [
        column for column in features.columns if column not in categorical_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def train_and_evaluate(
    data: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict[str, Pipeline], dict[str, dict[str, float]]]:
    """Train two baseline models and return fitted pipelines and test metrics."""
    features, target = prepare_features(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    estimators = {
        "logistic_regression": LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    models: dict[str, Pipeline] = {}
    results: dict[str, dict[str, float]] = {}

    for name, estimator in estimators.items():
        pipeline = build_model(x_train, estimator)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]

        models[name] = pipeline
        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(
                float(precision_score(y_test, predictions, zero_division=0)), 4
            ),
            "recall": round(
                float(recall_score(y_test, predictions, zero_division=0)), 4
            ),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        }

    return models, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare customer-churn classifiers."
    )
    parser.add_argument("--data", required=True, type=Path, help="Path to the CSV file")
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--random-state", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(args.data)
    _, metrics = train_and_evaluate(
        data,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

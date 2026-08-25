# Customer Churn Prediction

[![CI](https://github.com/Elenor274/customer-churn-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Elenor274/customer-churn-pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)

A reproducible machine-learning workflow for predicting telecom customer churn. The project separates exploratory work from a reusable Python pipeline and evaluates models with metrics that are meaningful for an imbalanced classification problem.

## Highlights

- Leakage-safe preprocessing fitted only on training data
- Missing-value handling for numeric and categorical features
- One-hot encoding with support for previously unseen categories
- Stratified train/test split
- Logistic Regression and Random Forest baselines
- Accuracy, precision, recall, F1, and ROC-AUC reporting
- Synthetic unit tests and automated CI on Python 3.11 and 3.12

## Project structure

```text
.
├── customer_churn.ipynb       # Original exploratory analysis
├── src/
│   └── churn_prediction.py    # Reusable training and evaluation pipeline
├── tests/                     # Tests using synthetic customer data
├── requirements.txt
└── requirements-dev.txt
```

## Dataset

The pipeline expects the commonly used Telco Customer Churn schema, including a binary `Churn` target. The dataset is intentionally not committed to this repository; place your CSV under `data/` or pass any local path with `--data`.

## Quick start

```bash
git clone https://github.com/Elenor274/customer-churn-pipeline.git
cd customer-churn-pipeline

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.churn_prediction \
  --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The command prints accuracy, precision, recall, F1, and ROC-AUC as JSON for each model. Actual values depend on the dataset and train/test split.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```

## Modeling notes

Accuracy alone can hide poor churn detection when non-churning customers are the majority. This project therefore reports recall and F1 for the churn class alongside ROC-AUC. Both baseline estimators use class balancing, and preprocessing lives inside each scikit-learn pipeline to prevent information from the test set leaking into training.

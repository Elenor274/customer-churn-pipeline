# Customer Churn Prediction

This project aims to predict customer churn using machine learning models. The dataset comes from a telecom company and contains information about customers' demographics and services.

## Project Structure

- `customer_churn.ipynb`: Main Jupyter Notebook used during development and exploratory analysis.
- `src/churn_prediction.py`: Python script that contains the final pipeline including data preprocessing, model training, and evaluation.
- `.gitignore`: Standard gitignore file to exclude unnecessary files from version control.

## Dataset

The dataset includes the following columns:
- Customer information (e.g., gender, tenure, contract type, monthly charges)
- Services subscribed (e.g., InternetService, OnlineSecurity)
- Target variable: `Churn` (Yes/No)

## Models Used

- **Logistic Regression**
- **Random Forest Classifier**

## Results

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression| 0.82     |
| Random Forest      | 0.80     |

### Classification Reports

**Logistic Regression**
- Precision (Yes): 0.68
- Recall (Yes): 0.58
- F1-score (Yes): 0.63

**Random Forest**
- Precision (Yes): 0.66
- Recall (Yes): 0.47
- F1-score (Yes): 0.55

## How to Run

```bash
python src/churn_prediction.py

# Customer Churn Prediction 📉

This project aims to predict whether a customer will churn (i.e., stop using the service) based on various features from a telecom dataset.

---

## 📁 Dataset

The dataset contains information about 7,043 telecom customers, with features like:
- Customer demographics
- Services subscribed (Internet, Phone, Streaming)
- Contract type and tenure
- Monthly and total charges
- Whether the customer churned

---

## 🛠️ Steps Performed

1. **Data Cleaning:**
   - Converted `TotalCharges` to numeric
   - Handled missing values by filling with the column mean

2. **Feature Engineering:**
   - Encoded categorical variables using `LabelEncoder` and `get_dummies`

3. **Splitting Data:**
   - Used `train_test_split` to separate training and testing data

4. **Scaling:**
   - Standardized numerical features using `StandardScaler`

5. **Model Training:**
   - Logistic Regression
   - Random Forest Classifier

6. **Evaluation:**
   - Accuracy Score
   - Classification Report (Precision, Recall, F1-Score)

---

## 📊 Results

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression | 81.7%    |
| Random Forest       | 79.2%    |

---

## 🔍 Insights

- Customers with **month-to-month contracts**, **low tenure**, and **high monthly charges** are more likely to churn.
- Long-term contracts (e.g., one-year, two-year) reduce the churn rate significantly.
- Features such as `Contract`, `PaymentMethod`, and `InternetService` showed strong influence.

---

## 💡 Future Improvements

- Hyperparameter tuning (e.g., GridSearchCV)
- Try advanced models like XGBoost or LightGBM
- Perform feature importance visualization

---

## 🧪 Run the Code

To run the notebook:

```bash
jupyter notebook customer_churn.ipynb
# customer-churn-pipeline

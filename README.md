# 📉 Customer Churn Prediction System

## 📌 Problem Statement
Customer churn leads to significant revenue loss. This project predicts whether a customer is likely to churn based on historical customer behavior, enabling proactive retention strategies.

---

## 📊 Dataset
- Telco Customer Churn Dataset (7,000+ customers)
- Includes demographics, service usage, billing details

---

## ⚙️ Approach
- Data cleaning & preprocessing
- Feature engineering (tenure groups, average monthly spend)
- Model comparison:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Evaluation using ROC-AUC and F1-score

---

## 🏆 Results
| Model | ROC-AUC | F1-score |
|------|--------|---------|
| Logistic Regression | **0.83** | **0.58** |
| Random Forest | 0.82 | 0.56 |
| XGBoost | 0.83 | 0.55 |

Logistic Regression performed best and was selected for deployment.

---

## 🔍 Key Insights
- Customers with short tenure are more likely to churn
- Month-to-month contracts increase churn risk
- High monthly charges correlate with churn
- Long-term contracts reduce churn significantly

---

## 🚀 Deployment
The model is deployed using **Streamlit**.

👉 **Live App:** https://<your-streamlit-link>

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit

---

## 📌 Author
Soumyadeep Das

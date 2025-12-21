import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("../notebooks/churn_logistic_model.pkl")
scaler = joblib.load("../notebooks/scaler.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📉 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn based on key attributes.")

# ---- USER INPUTS ----
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 150.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 600.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

# ---- FEATURE VECTOR (MATCH TRAINING ORDER) ----
input_data = np.zeros((1, scaler.mean_.shape[0]))

# Manual mapping (based on your model)
input_data[0][3] = tenure
input_data[0][6] = monthly_charges
input_data[0][7] = total_charges
input_data[0][9] = total_charges / max(tenure, 1)

if contract == "One year":
    input_data[0][26] = 1
elif contract == "Two year":
    input_data[0][27] = 1

if internet == "Fiber optic":
    input_data[0][13] = 1
elif internet == "No":
    input_data[0][14] = 1

if paperless == "Yes":
    input_data[0][5] = 1

# Scale input
input_scaled = scaler.transform(input_data)

# ---- PREDICTION ----
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Churn Risk (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Churn Risk (Probability: {probability:.2f})")

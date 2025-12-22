import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

BASE_DIR = Path(__file__).resolve().parent  # app/
MODEL_PATH = BASE_DIR.parent / "notebooks" / "churn_logistic_model.pkl"
SCALER_PATH = BASE_DIR.parent / "notebooks" / "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

CHURN_THRESHOLD = 0.35  # ✅ Correct threshold for churn models

# ---------------- UI ----------------
st.title("📉 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn based on key attributes.")

# ---------------- USER INPUTS ----------------
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 600.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

# ---------------- FEATURE VECTOR ----------------
input_data = np.zeros((1, scaler.mean_.shape[0]))

# Numeric features
input_data[0][3] = tenure
input_data[0][6] = monthly_charges
input_data[0][7] = total_charges
input_data[0][9] = total_charges / max(tenure, 1)

# Contract type
if contract == "One year":
    input_data[0][26] = 1
elif contract == "Two year":
    input_data[0][27] = 1
# Month-to-month is baseline (all zeros) ✔️

# Internet service
if internet == "Fiber optic":
    input_data[0][13] = 1
elif internet == "No":
    input_data[0][14] = 1

# Paperless billing
if paperless == "Yes":
    input_data[0][5] = 1

# ---------------- SCALE ----------------
input_scaled = scaler.transform(input_data)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Churn"):
    churn_probability = model.predict_proba(input_scaled)[0][1]

    st.metric(
        label="Churn Probability",
        value=f"{churn_probability:.2%}"
    )

    if churn_probability >= CHURN_THRESHOLD:
        st.error("⚠️ High Churn Risk")
        st.caption("Customer should be targeted for retention actions.")
    elif churn_probability >= 0.25:
        st.warning("🟡 Medium Churn Risk")
        st.caption("Customer shows early signs of churn.")
    else:
        st.success("✅ Low Churn Risk")
        st.caption("Customer is unlikely to churn.")

    st.divider()
    st.caption(f"Decision threshold used: {CHURN_THRESHOLD}")

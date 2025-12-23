import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "notebooks" / "churn_logistic_model.pkl"
SCALER_PATH = BASE_DIR.parent / "notebooks" / "scaler.pkl"
FEATURES_PATH = BASE_DIR.parent / "notebooks" / "feature_names.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

CHURN_THRESHOLD = 0.35

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

# NEW IMPORTANT FEATURES
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Credit card (automatic)",
    "Bank transfer (automatic)"
])

# ---------------- BUILD INPUT DATAFRAME ----------------
input_df = pd.DataFrame(0, columns=feature_names, index=[0])

# Numeric features
input_df["tenure"] = tenure
input_df["MonthlyCharges"] = monthly_charges
input_df["TotalCharges"] = total_charges

# Tenure group (auto-generated)
if 12 <= tenure < 24 and "tenure_group_1-2yr" in input_df.columns:
    input_df["tenure_group_1-2yr"] = 1
elif 24 <= tenure < 48 and "tenure_group_2-4yr" in input_df.columns:
    input_df["tenure_group_2-4yr"] = 1
elif 48 <= tenure < 72 and "tenure_group_4-6yr" in input_df.columns:
    input_df["tenure_group_4-6yr"] = 1

# Contract
if contract == "One year":
    input_df["Contract_One year"] = 1
elif contract == "Two year":
    input_df["Contract_Two year"] = 1

# Internet service
if internet == "Fiber optic":
    input_df["InternetService_Fiber optic"] = 1
elif internet == "No":
    input_df["InternetService_No"] = 1

# Paperless Billing
input_df["PaperlessBilling"] = 1 if paperless == "Yes" else 0

# Tech Support
if tech_support == "Yes" and "TechSupport_Yes" in input_df.columns:
    input_df["TechSupport_Yes"] = 1
else:
    if "TechSupport_No" in input_df.columns:
        input_df["TechSupport_No"] = 1

# Online Security
if online_security == "Yes" and "OnlineSecurity_Yes" in input_df.columns:
    input_df["OnlineSecurity_Yes"] = 1
else:
    if "OnlineSecurity_No" in input_df.columns:
        input_df["OnlineSecurity_No"] = 1

# Payment Method
if payment_method == "Electronic check":
    input_df["PaymentMethod_Electronic check"] = 1
elif payment_method == "Mailed check":
    input_df["PaymentMethod_Mailed check"] = 1
elif payment_method == "Credit card (automatic)":
    input_df["PaymentMethod_Credit card (automatic)"] = 1
else:
    # Bank transfer (automatic)
    input_df["PaymentMethod_Bank transfer (automatic)"] = 1

# ---------------- SCALE ----------------
input_scaled = scaler.transform(input_df)

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


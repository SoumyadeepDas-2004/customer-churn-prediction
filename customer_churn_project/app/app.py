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

# ---------------- BUILD INPUT (CRITICAL FIX) ----------------
input_df = pd.DataFrame(0, columns=feature_names, index=[0])

# Numeric features
if "tenure" in input_df.columns:
    input_df["tenure"] = tenure

if "MonthlyCharges" in input_df.columns:
    input_df["MonthlyCharges"] = monthly_charges

if "TotalCharges" in input_df.columns:
    input_df["TotalCharges"] = total_charges

# Derived feature (only if it existed in training)
if "ChargesPerMonth" in input_df.columns:
    input_df["ChargesPerMonth"] = total_charges / max(tenure, 1)

# Contract (baseline = Month-to-month)
if contract == "One year" and "Contract_One year" in input_df.columns:
    input_df["Contract_One year"] = 1

if contract == "Two year" and "Contract_Two year" in input_df.columns:
    input_df["Contract_Two year"] = 1

# Internet service
if internet == "Fiber optic" and "InternetService_Fiber optic" in input_df.columns:
    input_df["InternetService_Fiber optic"] = 1

if internet == "No" and "InternetService_No" in input_df.columns:
    input_df["InternetService_No"] = 1

# Paperless billing
if paperless == "Yes" and "PaperlessBilling_Yes" in input_df.columns:
    input_df["PaperlessBilling_Yes"] = 1

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

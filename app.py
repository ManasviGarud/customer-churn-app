# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# load model + scaler
with open('rf_churn_model.pkl','rb') as f:
    model = pickle.load(f)
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

st.title("Telecom Customer Churn Predictor")

# Inputs
tenure = st.number_input("tenure (months)", min_value=0, max_value=100, value=12)
MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, value=70.0)
TotalCharges = st.number_input("TotalCharges", min_value=0.0, value=840.0)

# Create input dataframe
input_df = pd.DataFrame({
    'tenure':[tenure],
    'MonthlyCharges':[MonthlyCharges],
    'TotalCharges':[TotalCharges]
})

# scale numeric
input_df[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(
    input_df[['tenure','MonthlyCharges','TotalCharges']]
)

# ⭐ IMPORTANT FIX
# This line auto-adds all missing columns
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

if st.button("Predict Churn"):
    pred_proba = model.predict_proba(input_df)[:,1][0]
    pred = model.predict(input_df)[0]
    st.write(f"Churn Probability: {pred_proba:.2f}")
    st.write("Predicted Churn:", "Yes" if pred==1 else "No")

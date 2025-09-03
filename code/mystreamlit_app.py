import streamlit as st
import pandas as pd
import pickle
from huggingface_hub import InferenceClient

# Load trained ML model and columns
with open("code/RF_model.pkl", "rb") as file:
    loan_model = pickle.load(file)

with open("code/columns.pkl", "rb") as file:
    training_columns = pickle.load(file)

# Streamlit UI Header
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🏦 Loan Eligibility Checker</h1>
    <p style='text-align: center; color: #555;'>Find out if you're eligible for a loan with a few simple inputs!</p>
    <hr>
""", unsafe_allow_html=True)

# Inputs Section with Columns (for better layout)
col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("👨‍👩‍👧‍👦 No of Dependents", min_value=0, max_value=5)
    education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("💼 Self Employed", ["Yes", "No"])
    income_annum = st.number_input("💰 Annual Income", min_value=0)
    loan_amount = st.number_input("🏦 Loan Amount", min_value=0)

with col2:
    loan_term = st.number_input("📅 Loan Term (Months)", min_value=2, max_value=20)
    cibil_score = st.number_input("📊 CIBIL Score", min_value=300, max_value=900)
    residential_assets_value = st.number_input("🏠 Residential Assets Value", min_value=0)
    commercial_assets_value = st.number_input("🏢 Commercial Assets Value", min_value=0)
    luxury_assets_value = st.number_input("🚗 Luxury Assets Value", min_value=0)
    bank_asset_value = st.number_input("🏦 Bank Asset Value", min_value=0)

# Prepare input DataFrame
input_data = pd.DataFrame([[no_of_dependents, education, self_employed, income_annum,
                            loan_amount, loan_term, cibil_score, residential_assets_value,
                            commercial_assets_value, luxury_assets_value, bank_asset_value]],
                          columns=["no_of_dependents", "education", "self_employed", "income_annum",
                                   "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
                                   "commercial_assets_value", "luxury_assets_value", "bank_asset_value"])

# One-hot encode categorical columns and align columns
loan_dummies = pd.get_dummies(input_data)
for col in training_columns:
    if col not in loan_dummies.columns:
        loan_dummies[col] = 0
loan_dummies = loan_dummies[training_columns]

# Prediction Logic
if st.button("🔎 Check Eligibility"):
    prediction = loan_model.predict(loan_dummies)
    result = "✅ Approved" if prediction[0] == 1 else "❌ Rejected"

    st.markdown(f"""
        <div style="padding: 15px; background-color: #f1f8f4; border-radius: 12px; border: 2px solid {'#4CAF50' if prediction[0] == 1 else '#FF5252'}; text-align: center; font-size: 18px;">
            <b>Loan Eligibility Prediction:</b> <span style="color: {'#4CAF50' if prediction[0] == 1 else '#FF5252'}">{result}</span>
        </div>
    """, unsafe_allow_html=True)

    
    

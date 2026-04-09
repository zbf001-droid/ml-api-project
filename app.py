import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Bucknell Lending Club App", layout="wide")

log_model = joblib.load("loan_status_log_model.pkl")
lin_model = joblib.load("ret_pess_lin_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Bucknell Lending Club Loan Evaluation App")
st.subheader("Predict Fully Paid Probability and Pessimistic Return")

loan_amnt = st.number_input("Loan Amount", min_value=1000, max_value=40000, value=10000, step=500)
term_num = st.selectbox("Term (months)", [36, 60])
int_rate = st.number_input("Interest Rate", min_value=5.0, max_value=31.0, value=12.0, step=0.1)
grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
emp_length = st.selectbox(
    "Employment Length",
    ["Unknown", "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
     "6 years", "7 years", "8 years", "9 years", "10+ years"]
)
home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER", "NONE", "ANY"])
annual_inc = st.number_input("Annual Income", min_value=0.0, value=60000.0, step=1000.0)
verification_status = st.selectbox("Verification Status", ["Verified", "Source Verified", "Not Verified"])
purpose = st.selectbox(
    "Purpose",
    ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
     "medical", "small_business", "car", "vacation", "moving", "house", "wedding",
     "renewable_energy", "educational"]
)
dti = st.number_input("DTI", min_value=0.0, value=15.0, step=0.1)
delinq_2yrs = st.number_input("Delinquencies in Past 2 Years", min_value=0.0, value=0.0, step=1.0)
open_acc = st.number_input("Open Accounts", min_value=0.0, value=10.0, step=1.0)
pub_rec = st.number_input("Public Records", min_value=0.0, value=0.0, step=1.0)
fico_range_high = st.number_input("FICO Range High", min_value=664, max_value=850, value=700, step=1)
revol_bal = st.number_input("Revolving Balance", min_value=0.0, value=15000.0, step=500.0)
revol_util = st.number_input("Revolving Utilization", min_value=0.0, max_value=152.0, value=50.0, step=1.0)
credit_age_years = st.number_input("Credit Age (years)", min_value=0.0, value=20.0, step=1.0)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "loan_amnt": loan_amnt,
        "term_num": term_num,
        "int_rate": int_rate,
        "grade": grade,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "purpose": purpose,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "fico_range_high": fico_range_high,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "credit_age_years": credit_age_years
    }])

    input_dum = pd.get_dummies(input_df, drop_first=True)
    input_dum = input_dum.reindex(columns=model_columns, fill_value=0)

    fully_paid_prob = log_model.predict_proba(input_dum)[0][1]
    pred_ret_pess = lin_model.predict(input_dum)[0]

    if fully_paid_prob >= 0.85 and pred_ret_pess >= 2:
        recommendation = "Recommend Funding"
    elif fully_paid_prob >= 0.75 and pred_ret_pess >= 0:
        recommendation = "Review Carefully"
    else:
        recommendation = "Do Not Recommend Funding"

    st.markdown("## Results")
    st.write(f"**Probability Loan is Fully Paid:** {fully_paid_prob:.2%}")
    st.write(f"**Predicted Pessimistic Return:** {pred_ret_pess:.2f}")
    st.write(f"**Recommended Action:** {recommendation}")

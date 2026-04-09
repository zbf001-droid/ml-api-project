import streamlit as st
import pandas as pd
import pickle

# Open the file and load the model
file_to_load = 'lend_logistic_model.pkl'
with open(file_to_load, 'rb') as file:
    loaded_model = pickle.load(file)

# --- Streamlit UI ---
st.set_page_config(page_title="💰 Loan Default ML Predictor", layout="centered")
st.title("💰 Loan Default Prediction")
st.markdown("Enter a potential loan customer's details to predict their risk of default.")

# User input
ownhome = st.checkbox("Own Their Home?")
income = st.slider("Family Income", 20000, 1000000, 80000)
dti = st.slider("Debt-to-Income Ratio", 0, 40, 10)
fico = st.slider("FICO Score", 300, 850, 650)

# Prediction
if st.button("Predict Loan Default"):
    new_customer = pd.DataFrame({
    'home_ownership': [ownhome],
    'income': [income],
    'dti': [dti],
    'fico': [fico]
    })

    # Use the model to find the predicted probability of default
    predicted_prob = loaded_model.predict_proba(new_customer)[:, 0]
    # Use the model to find the predicted class
    predicted_class = loaded_model.predict(new_customer)
    
    # Format the predicted probability with two decimals and a leading zero
    formatted_prob = f"{predicted_prob[0]:.2f}"
    
    # Display the predicted probability and class in Streamlit
    st.write(f"Predicted Probability of Default: **{formatted_prob}**")
    if predicted_class[0] == 0:
        st.success("Predicted Class: **Default**")
    else:
        st.success("Predicted Class: **Not Default**")
    
    # Show the default probability and not default as a pie chart
    probabilities = [predicted_prob[0], 1 - predicted_prob[0]]
    labels = ['Default', 'Not Default']
    chart_data = pd.DataFrame({'Probability': probabilities}, index=labels)
    st.write("Prediction Breakdown:")
    # VERY simple chart
    st.bar_chart(chart_data)
    
st.markdown("---")
st.markdown("**Developed by Matt Bailey as simple ML Web App Demo** | Powered by Streamlit")

import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration (Enterprise Branding)
st.set_page_config(page_title="Credit Risk AI", layout="centered")

# 2. Load the Model
# We use st.cache_resource so it doesn't reload the model every time you click a button
@st.cache_resource
def load_model():
    return joblib.load('models/credit_risk_pipeline.joblib')

model = load_model()

# 3. Sidebar Information
st.sidebar.title("System Info")
st.sidebar.info("This model predicts the probability of a 90-day delinquency (NPA) based on financial behavior.")

# 4. Main UI
st.title("🏦 Credit Risk NPA Prediction")
st.markdown("Enter customer details below to assess default probability.")

# Input fields organized in columns for a cleaner look
col1, col2 = st.columns(2)

with col1:
    revolving = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=10.0) / 100
    age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
    past_due_30 = st.number_input("Times 30-59 Days Past Due", min_value=0, value=0)
    debt_ratio = st.number_input("Debt Ratio (Monthly Debt / Income)", min_value=0.0, value=0.1)

with col2:
    income = st.number_input("Monthly Income", min_value=0, value=5000)
    open_lines = st.number_input("Open Credit Lines/Loans", min_value=0, value=5)
    late_90 = st.number_input("Times 90+ Days Late", min_value=0, value=0)
    real_estate = st.number_input("Real Estate Loans", min_value=0, value=0)

# Optional fields
late_60 = st.number_input("Times 60-89 Days Past Due", min_value=0, value=0)
dependents = st.number_input("Number of Dependents", min_value=0, value=0)

# 5. Prediction Logic
if st.button("Assess Risk"):
    # Prepare the input data into a DataFrame with exact column names from training
    input_data = pd.DataFrame([[
        revolving, age, past_due_30, debt_ratio, income, 
        open_lines, late_90, real_estate, late_60, dependents
    ]], columns=[
        'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
    ])
    
    # Make Prediction
    prediction_proba = model.predict_proba(input_data)[0][1]
    
    # Display Result
    st.subheader(f"Default Probability: {prediction_proba:.2%}")
    
    if prediction_proba > 0.5:
        st.error("⚠️ HIGH RISK: High probability of delinquency.")
    else:
        st.success("✅ LOW RISK: Customer likely to remain current.")
# app/streamlit_app.py

import streamlit as st
import pandas as pd
import pickle

# Load hospital readmission model
with open("models/readmission_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Patient Readmission Risk Predictor")

# User inputs
age = st.number_input("Age", 0, 120)
gender = st.selectbox("Gender", ["Male", "Female"])
prior_admissions = st.number_input("Number of prior admissions", 0, 50)
chronic_conditions = st.number_input("Number of chronic conditions", 0, 10)

# Encode gender
gender_encoded = 0 if gender == "Male" else 1

# Predict button
if st.button("Predict Risk"):
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender_encoded],
        "prior_admissions": [prior_admissions],
        "chronic_conditions": [chronic_conditions]
    })
    risk = model.predict(input_data)[0]
    st.success(f"Predicted readmission risk: {'High' if risk==1 else 'Low'}")

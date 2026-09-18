import streamlit as st
import joblib
import numpy as np

st.title("Electric Bill Prediction")

st.write("Enter the number of AC Units to predict the Electric Bill.")

# Load model
model = joblib.load("electricity_bill_prediction_model.pkl")

# Input
ac_units = st.number_input(
    "Enter AC Units",
    min_value=1,
    max_value=149,
    value=1,
    step=1
)

# Prediction
if st.button("Predict Electric Bill"):

    input_data = np.array([[ac_units]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Electric Bill: ₹{prediction[0]:.2f}")

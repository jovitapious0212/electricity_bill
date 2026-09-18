import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Page title
st.title("Electric Bill Prediction")

st.write("Enter the number of AC Units to predict the Electric Bill.")

# Load the saved model
model = joblib.load("electricity_bill_prediction_model.pkl")

# Create polynomial transformer
poly = PolynomialFeatures(degree=2)

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

    # Convert input to 2D array
    input_data = np.array([[ac_units]])

    # Convert input into polynomial features
    input_data_poly = poly.fit_transform(input_data)

    # Predict electric bill
    prediction = model.predict(input_data_poly)

    # Display result
    st.success(f"Predicted Electric Bill: ₹{prediction[0]:.2f}")

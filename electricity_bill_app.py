import streamlit as st
import pickle
import numpy as np

# Page title
st.title("Electric Bill Prediction")

st.write("Enter the number of AC Units to predict the Electric Bill.")

# Load the saved model
with open("electricity_bill_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

# Input: greater than 0 and less than 150
ac_units = st.number_input(
    "Enter AC Units",
    min_value=1,
    max_value=149,
    value=1,
    step=1
)

# Prediction button
if st.button("Predict Electric Bill"):

    # Convert input into 2D array
    input_data = np.array([[ac_units]])

    # Predict
    prediction = model.predict(input_data)

    # Display result
    st.success(f"Predicted Electric Bill: ₹{prediction[0]:.2f}")

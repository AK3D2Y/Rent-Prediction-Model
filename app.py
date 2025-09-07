import streamlit as st
import pandas as pd
import joblib

# Load trained model (save your pipeline after training with joblib.dump)
model = joblib.load("rent-model.pkl")

st.title("🏠 House Rent Prediction App")

# User inputs
bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
size = st.number_input("Size (sqft)", min_value=100, max_value=10000, value=1000)
bathroom = st.number_input("Bathroom", min_value=1, max_value=10, value=2)

city = st.selectbox("City", ["Mumbai", "Bangalore", "Delhi", "Chennai", "Hyderabad"])
furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
floor = st.text_input("Floor (e.g., Ground, 1st out of 5)", "1st out of 5")
contact = st.text_input("Point of Contact", "Contact Agent")

# Prepare input
input_df = pd.DataFrame([{
    "BHK": bhk,
    "Size": size,
    "Bathroom": bathroom,
    "City": city,
    "Furnishing Status": furnishing,
    "Floor": floor,
    "Point of Contact": contact
}])

# Predict
if st.button("Predict Rent"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Rent: ₹ {prediction:,.0f}")

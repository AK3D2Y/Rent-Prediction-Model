import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Page configuration
st.set_page_config(
    page_title="House Rent Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 House Rent Prediction App")
st.markdown("---")
st.write("Enter the details of your house to get an estimated rent prediction.")


# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("rent-model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file 'rent-model.pkl' not found. Please ensure the model is trained and saved.")
        return None


model = load_model()

if model is not None:
    # Create input form
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Property Details")

        # BHK input
        bhk = st.selectbox(
            "Number of BHK (Bedrooms, Hall, Kitchen)",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            index=1  # default to 2 BHK
        )

        # Size input
        size = st.number_input(
            "Size (in sq ft)",
            min_value=100,
            max_value=10000,
            value=1000,
            step=50
        )

        # Bathroom input
        bathroom = st.selectbox(
            "Number of Bathrooms",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            index=1  # default to 2 bathrooms
        )

    with col2:
        st.subheader("Additional Information")

        # Point of Contact
        point_of_contact = st.selectbox(
            "Point of Contact",
            options=["Contact Owner", "Contact Agent", "Contact Builder"]
        )

        # Floor
        floor_options = ["Ground Floor", "1st Floor", "2nd Floor", "3rd Floor",
                         "4th Floor", "5th Floor", "6th Floor", "7th Floor",
                         "8th Floor", "9th Floor", "10th Floor", "Upper Basement",
                         "Lower Basement"]
        floor = st.selectbox(
            "Floor",
            options=floor_options
        )

        # City
        city_options = ["Mumbai", "Chennai", "Bangalore", "Hyderabad", "Delhi",
                        "Kolkata", "Pune", "Ahmedabad", "Surat", "Jaipur"]
        city = st.selectbox(
            "City",
            options=city_options,
            index=2  # default to Bangalore
        )

        # Furnishing Status
        furnishing_status = st.selectbox(
            "Furnishing Status",
            options=["Unfurnished", "Semi-Furnished", "Furnished"]
        )

    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🔮 Predict Rent", type="primary", use_container_width=True):
            # Create input dataframe
            input_data = pd.DataFrame({
                'BHK': [bhk],
                'Size': [size],
                'Bathroom': [bathroom],
                'Point of Contact': [point_of_contact],
                'Floor': [floor],
                'City': [city],
                'Furnishing Status': [furnishing_status]
            })

            # Make prediction
            try:
                prediction = model.predict(input_data)
                predicted_rent = prediction[0]

                # Display result
                st.success("✅ Prediction Complete!")

                # Create result display
                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    st.metric(
                        label="Predicted Monthly Rent",
                        value=f"₹{predicted_rent:,.0f}",
                        delta=None
                    )

                with result_col2:
                    st.metric(
                        label="Per Sq Ft Rate",
                        value=f"₹{predicted_rent / size:.2f}",
                        delta=None
                    )

                # Display input summary
                st.markdown("---")
                st.subheader("📋 Input Summary")

                summary_data = {
                    "Property Feature": ["BHK", "Size (sq ft)", "Bathrooms", "Floor",
                                         "City", "Furnishing", "Contact"],
                    "Value": [bhk, f"{size:,}", bathroom, floor, city,
                              furnishing_status, point_of_contact]
                }

                summary_df = pd.DataFrame(summary_data)
                st.table(summary_df)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("Please check that all inputs are valid and try again.")

else:
    st.error("Unable to load the prediction model. Please ensure 'rent-model.pkl' exists in the same directory.")

# Sidebar with additional information
st.sidebar.title("ℹ️ About")
st.sidebar.write("""
This app predicts house rent based on various features like:
- Number of BHK (Bedrooms, Hall, Kitchen)
- Size in square feet
- Number of bathrooms
- Floor level
- City location
- Furnishing status
- Point of contact

The model uses a Decision Tree Regressor with the following features:
- Standard scaling for numeric features
- One-hot encoding for categorical features
- Optimized hyperparameters for better accuracy
""")

st.sidebar.markdown("---")
st.sidebar.write("**Model Performance Metrics:**")
st.sidebar.write("- Uses RMSE and R² score for evaluation")
st.sidebar.write("- Trained on house rental dataset")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>🏠 House Rent Prediction App | Built with Streamlit</p>",
    unsafe_allow_html=True
)

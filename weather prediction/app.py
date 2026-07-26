import streamlit as st
import joblib
import pandas as pd


# Load saved model and columns
model = joblib.load("weather_model.pkl")
columns = joblib.load("columns.pkl")


# Page configuration
st.set_page_config(
    page_title="Weather Rain Prediction",
    page_icon="🌧️",
    layout="wide"
)


# Title
st.title("🌧️ Weather Rain Prediction")
st.write("Enter weather details to predict whether it will rain tomorrow.")


# Create input fields
input_data = {}

col1, col2, col3 = st.columns(3)

for i, feature in enumerate(columns):

    if i % 3 == 0:
        with col1:
            input_data[feature] = st.number_input(
                feature,
                value=0.0
            )

    elif i % 3 == 1:
        with col2:
            input_data[feature] = st.number_input(
                feature,
                value=0.0
            )

    else:
        with col3:
            input_data[feature] = st.number_input(
                feature,
                value=0.0
            )


# Prediction button
if st.button("🔍 Predict Weather"):

    # Convert input into dataframe
    input_df = pd.DataFrame([input_data])

    # Prediction
    prediction = model.predict(input_df)


    if prediction[0] == 1:
        st.error("🌧️ Prediction: Rain Tomorrow")
    else:
        st.success("☀️ Prediction: No Rain Tomorrow")


# Footer
st.write("---")
st.caption("Weather Prediction using Logistic Regression Machine Learning Model")
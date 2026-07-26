# Import Libraries
import streamlit as st
import pandas as pd
import joblib
# Load Saved Files
model = joblib.load("models/LR_ford_car.pkl")
scaler = joblib.load("models/scaler.pkl")
encoded_columns = joblib.load("models/columns.pkl")
# Configure Page
st.set_page_config(
 page_title="Ford Car Price Prediction",
 page_icon="🚗",
 layout="wide"
)
# Sidebar
st.sidebar.title("Project Details")
st.sidebar.write("Algorithm : Linear Regression")
st.sidebar.write("Model Accuracy : 84.02%")
st.sidebar.write("Developer : Shruti Darwatkar")
# Title
st.title("🚗Ford Car Price Prediction System")
st.write("Predict the selling price of a Ford car using Machine Learning.")
# Create Two Columns
col1, col2 = st.columns(2)
# Left Column
with col1:
 model_name = st.text_input(
 "Car Model",
 "Fiesta"
 )
 year = st.number_input(
 "Manufacturing Year",
 1996,
 2025,
 2018
 )
 mileage = st.number_input(
 "Mileage",
 0,
 300000, 30000
 )
 tax = st.number_input(
 "Road Tax",
 0,
 600,
 150
 )
# Right Column
with col2:
 transmission = st.selectbox(
 "Transmission",
 ["Manual", "Automatic", "Semi-Auto"]
 )
 fuelType = st.selectbox(
 "Fuel Type",
 ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
 )
 mpg = st.number_input(
 "MPG",
 0.0,
 100.0,
 55.4
 )
 engineSize = st.number_input(
    "Engine Size",
    1.0,
    5.0,
    1.5
)
# Predict Button
predict = st.button("Predict Price")
# Perform Prediction
if predict:
 # Create Input DataFrame
 input_data = pd.DataFrame({
 "model": [model_name],
 "year": [year],
 "transmission": [transmission],
 "mileage": [mileage],
 "fuelType": [fuelType],
 "tax": [tax],
 "mpg": [mpg],
 "engineSize": [engineSize]
 })
 # One-Hot Encoding
 input_data = pd.get_dummies(input_data)
 # Match Training Columns
 input_data = input_data.reindex(
 columns=encoded_columns, fill_value=0
 )
 # Numerical Columns
 numerical_columns = [
 "year",
 "mileage",
 "tax",
 "mpg",
 "engineSize"
 ]
 # Feature Scaling
 input_data[numerical_columns] = scaler.transform(
 input_data[numerical_columns]
 )
 # Predict Price
 prediction = model.predict(input_data)
 # Display Output
 st.success("Prediction Completed Successfully!")
 st.metric(
 label="Estimated Car Price",
 value=f"£{prediction[0]:,.2f}"
 )
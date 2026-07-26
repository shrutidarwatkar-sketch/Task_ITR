import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("Heart Disease Prediction")

age = st.number_input("Age")
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", ["ATA","NAP","ASY","TA"])
bp = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar", [0,1])
ecg = st.selectbox("Resting ECG", ["Normal","ST","LVH"])
hr = st.number_input("Maximum Heart Rate")
angina = st.selectbox("Exercise Angina", ["Y","N"])
oldpeak = st.number_input("Old Peak")
slope = st.selectbox("ST Slope", ["Up","Flat","Down"])

if st.button("Predict"):

    data = pd.DataFrame([{
        "Age":age,
        "Sex":sex,
        "ChestPainType":cp,
        "RestingBP":bp,
        "Cholesterol":chol,
        "FastingBS":fbs,
        "RestingECG":ecg,
        "MaxHR":hr,
        "ExerciseAngina":angina,
        "Oldpeak":oldpeak,
        "ST_Slope":slope
    }])

    for col in data.columns:
        if col in encoders:
            data[col] = encoders[col].transform(data[col])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Heart Disease: Yes")
    else:
        st.success("Heart Disease: No")
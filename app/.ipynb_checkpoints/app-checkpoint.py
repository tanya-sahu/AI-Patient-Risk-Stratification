import streamlit as st
import joblib
import numpy as np

model = joblib.load("heart_model.pkl")

st.title("AI Risk Predictor")


st.write("Enter patient details to predict heart disease risk")
age = st.number_input("Age")
sex = st.selectbox("Sex" , [0,1])
trestbps = st.number_input("Resting BP")
chol = st.number_input("Cholestrol")
thalch = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Angine" , [0,1])
oldpeak = st.number_input("Oldpeak")
ca = st.selectbox("Major Vessels" , [0,1,2,3])
cp_atypical = st.selectbox("CP Atypical Angina", [0,1])
cp_non = st.selectbox("CP Non Anginal", [0,1])
thal_normal = st.selectbox("Thal Normal", [0,1])

 if st.button("Predict"):

     patient = np.array([[age , sex , trestbps , chol , thalch , exang  , oldpeak , ca , cp_atypical , cp_non , thal_normal]])

      prediction = model.predict(patient)
    prob = model.predict_proba(patient)[0][1]

    st.write("Disease Probability:", prob)

    if prob < 0.3:
        st.success("Low Risk")
    elif prob < 0.5:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import uuid
from datetime import datetime
import json
import os

#########################################################################
## JSON File storage permanently
FILE_NAME = "patient_log.json"

def load_log():
    if os.path.exists(FILE_NAME):
        with open(FILE_NAME , "r") as f:
            return json.load(f)
    return []


def save_log(data):
    with open(FILE_NAME , "w") as f:
        json.dump(data , f , indent=4)

#########################################################################################################################


model = pickle.load(open("model.pkl","rb"))
columns = pickle.load(open("feature_columns.pkl","rb"))

st.title("AI Heart Risk Stratification System")
st.write("AI tool to prioritize patients based on heart risk")

st.set_page_config(
    page_title="AI Heart Risk System",
    page_icon="❤️",
    layout="wide"
)

st.sidebar.header("Patient Information")

#########################################################################################################################
## Patient Form

name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age")
sex = st.sidebar.selectbox("Sex",["Male","Female"])
trestbps = st.sidebar.number_input("Resting BP")
chol = st.sidebar.number_input("Cholesterol")
thalach = st.sidebar.number_input("Max Heart Rate")
exang = st.sidebar.selectbox("Exercise Angina",["No","Yes"])
oldpeak = st.sidebar.number_input("Oldpeak")
ca = st.sidebar.selectbox("Major Vessels",[0,1,2,3])

cp_non_anginal = st.sidebar.selectbox("CP Non Anginal",["No","Yes"])
cp_typical = st.sidebar.selectbox("CP Typical",["No","Yes"])
thal_normal = st.sidebar.selectbox("Thal Normal",["No","Yes"])


## Patient Data 
patient_id = str(uuid.uuid4())[:8]

patient_data = {
        "Patient ID": patient_id,
        "Patient Name": name,
        "Age": age,
        "Sex": sex,
        "BP": trestbps,
        "Cholesterol": chol,
        "Max HR": thalach,
        "Exercise Angina": exang,
        "Major Vessels": ca,
        "CP Non Anginal": cp_non_anginal,
        "CP Typical": cp_typical,
        "Thal Normal": thal_normal,
        "Time": datetime.now().strftime("%H:%M:%S")
    }


sex = 1 if sex == "Male" else  0
exang = 1 if exang == "Yes" else  0
cp_non_anginal = 1 if cp_non_anginal == "Yes" else  0
cp_typical = 1 if cp_typical == "Yes" else  0
thal_normal = 1 if thal_normal == "Yes" else  0


#########################################################################################################################

## Risk calculation

patients = pd.DataFrame([[age,sex,trestbps,chol,thalach,exang,oldpeak,ca,
                          cp_non_anginal,cp_typical,thal_normal]],
                        columns=columns)

prediction = model.predict(patients)
prob = model.predict_proba(patients)


st.subheader("Patient Risk Level")

risk = prob[0][1]

def get_priority(risk):
    if risk < 0.3:
       st.success("LOW RISK 🟢")
    elif risk < 0.7:
       st.warning("MEDIUM RISK 🟡")
    else:
       st.error("HIGH RISK 🔴")

get_priority(risk)
st.write("Risk Probability:",round(risk*100 , 2) , "%")



#########################################################################################################################
## Risk Bar graph
import matplotlib.pyplot as plt

risk_prob = prob[0][1]
low_prob = prob[0][0]

labels = ["Low Risk", "High Risk"]
sizes = [low_prob, risk_prob]

colors = ["#2ecc71", "#e74c3c"]

fig, ax = plt.subplots()

ax.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    wedgeprops={'width':0.4}
)

centre_circle = plt.Circle((0,0),0.70,fc='white')
fig.gca().add_artist(centre_circle)

ax.text(0,0,f"{risk_prob*100:.1f}%",ha='center',va='center',fontsize=18)

ax.set_title("Heart Disease Risk")

st.pyplot(fig)

#########################################################################################################################


## 1️⃣ Patient Storage System

if "patients" not in st.session_state:
    st.session_state.patients = []


if "log" not in st.session_state:
    st.session_state.log = load_log()



risk = prob[0][1]*100

if risk < 30:
    level = "Low"
elif risk < 70:
    level = "Medium"
else:
    level = "High"

patient_data["Risk Score"] = round(risk , 2)
patient_data["Priority"] = level




if st.button("Add Patient"):

    if name == "":
            st.error("Please enter patient name")
    else:
            st.session_state.patients.append(patient_data)

            st.session_state.log.append(patient_data)
            st.success("Patient Added Successfully")




#########################################################################################################################


## Dividing Patients

df_patients = pd.DataFrame(st.session_state.patients)

if not df_patients.empty:
    df_patients = df_patients.sort_values(
        by="Risk Score",
        ascending=False
    )

tab1, tab2, tab3 = st.tabs([
    "🔴 High Risk",
    "🟡 Medium Risk",
    "🟢 Low Risk"
])


with tab1:

    st.subheader("High Risk Patients")

    if not df_patients.empty and "Priority" in df_patients.columns:
        high = df_patients[df_patients["Priority"] == "High"]


        for i , row in high.iterrows():

            st.write(row)

            if st.button(f"Delete {row["Patient ID"]}" , key = row["Patient ID"]):

                st.session_state.patients = [
                    p for p in st.session_state.patients
                    if p["Patient ID"] != row["Patient ID"]
                ]


                st.rerun()
    else:
        st.info("No patients added yet")


with tab2:

    st.subheader("Medium Risk Patients")

    if not df_patients.empty and "Priority" in df_patients.columns:
        medium = df_patients[df_patients["Priority"] == "Medium"]
        for i, row in medium.iterrows():

            st.write(row)

            if st.button(f"Delete {row['Patient ID']}", key=row['Patient ID']+"m"):

                st.session_state.patients = [
                    p for p in st.session_state.patients
                    if p["Patient ID"] != row["Patient ID"]
                ]

                st.rerun()
    else:
        st.info("No patients added yet")


with tab3:

    st.subheader("Low Risk Patients")

    if not df_patients.empty and "Priority" in df_patients.columns:
        low = df_patients[df_patients["Priority"] == "Low"]
        
        for i, row in low.iterrows():

            st.write(row)

            if st.button(f"Delete {row['Patient ID']}", key=row['Patient ID']+"l"):

                st.session_state.patients = [
                    p for p in st.session_state.patients
                    if p["Patient ID"] != row["Patient ID"]
                ]

                st.rerun()
    else:
        st.info("No patients added yet")






#########################################################################################################################

## Feature Importance Graph
st.subheader("Feature Importance")

importances = model.feature_importances_

importance_df = pd.DataFrame({
    "feature":columns,
    "importance":importances
})

importance_df = importance_df.sort_values("importance",ascending=False)

st.bar_chart(importance_df.set_index("feature"))

#########################################################################################################################

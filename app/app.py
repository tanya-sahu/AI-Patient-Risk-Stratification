# =============================
# 1. CONFIG 
# =============================
import streamlit as st

st.set_page_config(
    page_title="AI Heart Risk System",
    page_icon="❤️",
    layout="wide"
)




# =============================
# 2. IMPORTS
# =============================
import pickle
import numpy as np
import pandas as pd
import uuid
from datetime import datetime
import json
import os
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go





# =============================
# 3. STYLE
# =============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}
.block-container{
    background: rgba(255,255,255,0.05);
    padding:20px;
    border-radius:15px;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
.high { border-left: 6px solid red; }
.medium { border-left: 6px solid orange; }
.low { border-left: 6px solid green; }
</style>
""", unsafe_allow_html=True)




# =============================
# 3. JSON STORAGE FUNCTIONS
# =============================
FILE_NAME = "patient_log.json"

def load_log():
    if os.path.exists(FILE_NAME):
        with open(FILE_NAME , "r") as f:
            return json.load(f)
    return []

def save_log(data):
    with open(FILE_NAME , "w") as f:
        json.dump(data , f , indent=4)


# =============================
# 4. LOAD MODEL
# =============================
heart_model = pickle.load(open("heart_model.pkl", "rb"))
heart_features = pickle.load(open("heart_features.pkl", "rb"))


diabetes_model = pickle.load(open("diabetes_model.pkl", "rb"))
diabetes_features = pickle.load(open("diabetes_features.pkl", "rb"))





# =============================
# 6. CARD FUNCTION
# =============================
def show_card(title, prob):
    percent = round(prob * 100)

    if prob < 0.3:
        cls = "low"
        level = "LOW 🟢"
    elif prob < 0.7:
        cls = "medium"
        level = "MEDIUM 🟡"
    else:
        cls = "high"
        level = "HIGH  🔴"

    st.markdown(f"""
    <div class="card {cls}">
        <h3>{title}</h3>
        <h2>{level} ({percent}%)</h2>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
## Patient Card Function
# ==========================================
def patient_card(row, key_suffix=""):

    # color decide
    if row["Priority"] == "Low":
        color = "#2ecc71"
    elif row["Priority"] == "Medium":
        color = "#f39c12"
    else:
        color = "#e74c3c"

    # main card
    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.08);
        padding:15px;
        border-radius:12px;
        margin-bottom:10px;
        border-left: 6px solid {color};
    ">
        <h4>{row['Patient Name']} ({row['Disease']})</h4>
        <p><b>Risk:</b> {row['Risk Score']}%</p>
    </div>
    """, unsafe_allow_html=True)

    # progress bar 🔥
    st.progress(int(row["Risk Score"]))

    # expand details 👇
    with st.expander("🔍 View Details"):

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Age:** {row['Age']}")
            st.write(f"**Sex:** {row['Sex']}")
            st.write(f"**Time:** {row['Time']}")

        with col2:
            st.write(f"**Disease:** {row['Disease']}")
            st.write(f"**Priority:** {row['Priority']}")

        st.json(row.to_dict())

    # delete button
    if st.button("❌ Delete", key=f"del_{row['Patient ID']}_{key_suffix}"):

        st.session_state.patients = [
            p for p in st.session_state.patients
            if p["Patient ID"] != row["Patient ID"]
        ]

        st.rerun()

# =============================
# 7. TITLE
# =============================
st.title("AI Patient Risk Stratification System")

disease = st.sidebar.selectbox(
    "Select Disease",
    ["Heart Disease ❤️", "Diabetes 🩸"]
)



# =============================
# 8. SIDEBAR INPUTS
# =============================
st.sidebar.header("Patient Info")


name = st.sidebar.text_input("Patient Name", value="John Doe")
age = st.sidebar.number_input(
    "Age (years)",
    min_value=1,
    max_value=120,
    value=30
)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])




#===============================
# Heart Disease Inputs
if disease == "Heart Disease ❤️":

    st.sidebar.markdown("### Heart Disease Risk Factors")  
    trestbps = st.sidebar.number_input(
       "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=200,
        value=120
    )

    chol = st.sidebar.number_input(
        "Cholesterol (mg/dL)",
        min_value=100,
        max_value=400,
        value=180
    )

    thalach = st.sidebar.number_input(
        "Max Heart Rate (bpm)",
        min_value=60,
        max_value=220,
        value=150
    )

    exang = st.sidebar.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"]
    )

    oldpeak = st.sidebar.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=6.0,
        value=1.0,
        step=0.1
    )

    ca = st.sidebar.selectbox(
        "Major Vessels (0–3)",
        [0,1,2,3]
    )

    cp_non_anginal = st.sidebar.selectbox(
        "Chest Pain Type: Non-Anginal",
        ["No", "Yes"]
    )

    cp_typical = st.sidebar.selectbox(
        "Chest Pain Type: Typical Angina",
        ["No", "Yes"]
    )

    thal_normal = st.sidebar.selectbox(
        "Thalassemia: Normal",
        ["No", "Yes"]
    )



   # encode
    sex = 1 if sex == "Male" else  0
    exang = 1 if exang == "Yes" else  0
    cp_non_anginal = 1 if cp_non_anginal == "Yes" else  0
    cp_typical = 1 if cp_typical == "Yes" else  0
    thal_normal = 1 if thal_normal == "Yes" else  0



elif disease == "Diabetes 🩸":
    st.sidebar.markdown("### Diabetes Risk Factors")  

       

    preg = st.sidebar.number_input(
        "Pregnancies (count)",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.sidebar.number_input(
        "Glucose Level (mg/dL)",
        min_value=70,
        max_value=200,
        value=100
    )

    bp = st.sidebar.number_input(
        "Blood Pressure (mm Hg)",
        min_value=60,
        max_value=140,
        value=80
    )

    skin = st.sidebar.number_input(
        "Skin Thickness (mm)",
        min_value=10,
        max_value=60,
        value=20
    )

    insulin = st.sidebar.number_input(
        "Insulin (mu U/ml)",
        min_value=15,
        max_value=276,
        value=80
    )

    bmi = st.sidebar.number_input(
         "BMI (kg/m²)",
         min_value=15.0,
         max_value=50.0,
         value=22.0,
         step=0.1
     )

    dpf = st.sidebar.number_input(
         "Diabetes Pedigree Function",
         min_value=0.0,
         max_value=2.5,
         value=0.3,
         step=0.01
     )
    





 

# =============================
# 9. MODEL INPUT
# =============================
if disease == "Heart Disease ❤️":

    input_data = [[age, sex, trestbps, chol, thalach,
                   exang, oldpeak, ca,
                   cp_non_anginal, cp_typical, thal_normal]]

    df_input = pd.DataFrame(input_data, columns=heart_features)

    model = heart_model
    features = heart_features

elif disease == "Diabetes 🩸":

    input_data = [[preg, glucose, bp, skin,
                   insulin, bmi, dpf, age]]

    df_input = pd.DataFrame(input_data, columns=diabetes_features)

    model = diabetes_model
    features = diabetes_features



prediction = model.predict(df_input)
prob = model.predict_proba(df_input)

risk_prob = prob[0][1]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df_input)




# Handle both formats
if isinstance(shap_values, list):
    shap_val = shap_values[: , 1]   # class 1
    base_val = explainer.expected_value[1]
else:
    shap_val = shap_values
    base_val = explainer.expected_value




# =============================
# 10. DISPLAY CARD
# =============================
col1, col2 = st.columns(2)

with col1:
   
    if disease == "Heart Disease ❤️":
        show_card("❤️ Heart Risk", risk_prob)
    else:
        show_card("🩸 Diabetes Risk", risk_prob)    







#===================================
## Patient Data for Logging
#===================================
patient_id = str(uuid.uuid4())[:8]

patient_data = {
        "Patient ID": patient_id,
        "Patient Name": name,
        "Risk Score" : round(risk_prob*100 , 2),
        "Age": age,
        "Sex": sex,
        "Time": datetime.now().strftime("%H:%M:%S")
    }


if disease == "Heart Disease ❤️":
    patient_data["Disease"] = "Heart Disease"
    patient_data["Resting Blood Pressure"] = trestbps
    patient_data["Cholesterol"] = chol,
    patient_data["Max Heart Rate"] = thalach
    patient_data["Exercise Induced Angina"] = exang
    patient_data["ST Depression (Oldpeak)"] = oldpeak
    patient_data["Major Vessels"] = ca
    patient_data["Chest Pain Type: Non-Anginal"] = cp_non_anginal
    patient_data["Chest Pain Type: Typical Angina"] = cp_typical
    patient_data["Thalassemia: Normal"] = thal_normal



elif disease == "Diabetes 🩸":
    patient_data["Disease"] = "Diabetes"
    patient_data["Pregnancies"] = preg
    patient_data["Glucose Level"] = glucose
    patient_data["Blood Pressure"] = bp
    patient_data["Skin Thickness"] = skin
    patient_data["Insulin"] = insulin
    patient_data["BMI"] = bmi
    patient_data["Diabetes Pedigree Function"] = dpf



risk = risk_prob*100

if risk < 30:
    level = "Low"
elif risk < 70:
    level = "Medium"
else:
    level = "High"

patient_data["Priority"] = level

# =============================
# 11. GAUGE
# =============================
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_prob*100,
    title={f'text': f"${patient_data["Disease"]} Risk %"},
    gauge={
        'axis': {'range': [0,100]},
        'steps': [
            {'range':[0,30],'color':"green"},
            {'range':[30,70],'color':"yellow"},
            {'range':[70,100],'color':"red"},
        ]
    }
))


with col2:
    st.plotly_chart(fig)









# =============================
# 12. ADD PATIENT
# =============================


if "patients" not in st.session_state:
    st.session_state.patients = []


if "log" not in st.session_state:
    st.session_state.log = load_log()



if st.sidebar.button("Add Patient"):

    if name == "":
            st.error("Please enter patient name")
    else:
            st.session_state.patients.append(patient_data)

            st.session_state.log.append(patient_data)
            save_log(st.session_state.log)
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

            ##st.write(row)
            patient_card(row , key_suffix="h")

            ##if st.button(f"Delete {row["Patient ID"]}" , key = row["Patient ID"]):

                ###st.session_state.patients = [
                  ##  p for p in st.session_state.patients
                  ##  if p["Patient ID"] != row["Patient ID"]
                ##]


                ##st.rerun()
    else:
        st.info("No patients added yet")


with tab2:

    st.subheader("Medium Risk Patients")

    if not df_patients.empty and "Priority" in df_patients.columns:
        medium = df_patients[df_patients["Priority"] == "Medium"]
        for i, row in medium.iterrows():

            
            ##st.write(row)
            patient_card(row , key_suffix="m")

             ##if st.button(f"Delete {row["Patient ID"]}" , key = row["Patient ID"]):

                ###st.session_state.patients = [
                  ##  p for p in st.session_state.patients
                  ##  if p["Patient ID"] != row["Patient ID"]
                ##]


                ##st.rerun()
    else:
        st.info("No patients added yet")


with tab3:

    st.subheader("Low Risk Patients")

    if not df_patients.empty and "Priority" in df_patients.columns:
        low = df_patients[df_patients["Priority"] == "Low"]
        
        for i, row in low.iterrows():

            
            ##st.write(row)
            patient_card(row , key_suffix="l")

             ##if st.button(f"Delete {row["Patient ID"]}" , key = row["Patient ID"]):

                ###st.session_state.patients = [
                  ##  p for p in st.session_state.patients
                  ##  if p["Patient ID"] != row["Patient ID"]
                ##]


                ##st.rerun()
    else:
        st.info("No patients added yet")






# =============================
# 14. FEATURE IMPORTANCE
# =============================
st.subheader("Feature Importance")

imp = model.feature_importances_

df_imp = pd.DataFrame({
    "Feature": features,
    "Importance": imp
}).sort_values("Importance", ascending=False)

st.bar_chart(df_imp.set_index("Feature"))








# =============================
# 15. SHAP
# =============================
st.subheader("SHAP Explanation")



fig = plt.figure()

shap.plots.waterfall(
    shap.Explanation(
        values=shap_val[0, 0],
        base_values=explainer.expected_value[1],
        data=df_input.iloc[0],
        feature_names=features
    )
)

st.pyplot(fig)

##st.markdown("## 📊 Dashboard Summary")

##total = len(df_patients)
##high = len(df_patients[df_patients["Priority"] == "High"])
##m##e##dium = len(df_patients[df_patients["Priority"] == "Medium"])
##low = len(df_patients[df_patients["Priority"] == "Low"])

##c1, c2, c3, c4 = st.columns(4)

##c##1.metric("Total Patients", total)
##c2.metric("🔴 High Risk", high)
##c3.metric("🟡 Medium Risk", medium)
##c4.metric("🟢 Low Risk", low)
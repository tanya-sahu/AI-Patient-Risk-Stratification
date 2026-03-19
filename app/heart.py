# =============================
# 1. IMPORTS
# =============================
import streamlit as st
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





shap.initjs()

# =============================
# 2. PAGE CONFIG + STYLING
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
    backdrop-filter: blur(10px);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
    text-align: center;
}

.high { border-left: 6px solid red; }
.medium { border-left: 6px solid orange; }
.low { border-left: 6px solid green; }

.title {
    font-size: 20px;
    font-weight: bold;
}

.value {
    font-size: 28px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)


def show_card(title, prob):
    
    percent = round(prob * 100)

    if prob > 0.7:
        level = "HIGH"
        cls = "high"
    elif prob > 0.4:
        level = "MEDIUM"
        cls = "medium"
    else:
        level = "LOW"
        cls = "low"

    st.markdown(f"""
    <div class="card {cls}">
        <div class="title">{title}</div>
        <div class="value">{level} ({percent}%)</div>
    </div>
    """, unsafe_allow_html=True)


import pickle

# load heart
heart_model = pickle.load(open("heart_model.pkl", "rb"))
heart_features = pickle.load(open("heart_features.pkl", "rb"))



# ###############################################################





st.title("AI Patient Risk Stratification System")

st.set_page_config(
    page_title="AI Heart Risk System",
    page_icon="❤️",
    layout="wide"
)



#########################################################################################################################
# =============================
# 5. SIDEBAR INPUTS
# =============================

st.sidebar.header("Patient Information")

##disease_type = st.sidebar.selectbox(
   ## "Select Analysis Type",
   ## ["Full Body Analysis", "Heart Disease", "Diabetes", "Stroke"]
##)

# ============================================================
# Commom Features
name = st.sidebar.text_input("Patient Name", value="John Doe")
age = st.sidebar.number_input(
    "Age (years)",
    min_value=1,
    max_value=120,
    value=30
)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

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




sex = 1 if sex == "Male" else  0
exang = 1 if exang == "Yes" else  0
cp_non_anginal = 1 if cp_non_anginal == "Yes" else  0
cp_typical = 1 if cp_typical == "Yes" else  0
thal_normal = 1 if thal_normal == "Yes" else  0




patient_id = str(uuid.uuid4())[:8]

patient_data = {
        "Patient ID": patient_id,
        "Patient Name": name,
        "Age": age,
        "Sex": sex,
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Exercise Induced Angina": exang,
        "Chest Pain Type: Non-Anginal" : cp_non_anginal,
        "Chest Pain Type: Typical Angina" : cp_typical,
        "Thalassemia: Normal" : thal_normal
    }




   
    

patients = pd.DataFrame([[age,sex,trestbps,chol,thalach,exang,oldpeak,ca,
                          cp_non_anginal,cp_typical,thal_normal]],
                    )

prediction = heart_model.predict(patients)
prob = heart_model.predict_proba(patients)
explainer = shap.TreeExplainer(heart_model)
shap_values = explainer.shap_values(patients)



# Handle both formats
if isinstance(shap_values, list):
    shap_val = shap_values[: , 1]   # class 1
    base_val = explainer.expected_value[1]
else:
    shap_val = shap_values
    base_val = explainer.expected_value



heart_prob = prob[0][1]
def get_priority(heart_prob):

    if heart_prob < 0.3:
        st.success("LOW RISK 🟢")

        if st.button("Recommendations"):   

            st.info("""
            • Maintain healthy lifestyle\n
            • Regular exercise\n
            • Routine checkup after 6 months\n
            """)

    elif heart_prob < 0.7:
        st.warning("MEDIUM RISK 🟡")

        if st.button("Recommendations"):
            st.warning("""
            • Recommend ECG test\n
            • Lipid profile monitoring\n
            • Cardiologist consultation\n
            • Lifestyle modification\n
            """)

    else:
        st.error("HIGH RISK 🔴")

        if st.button("Recommendations"):
            st.error("""
            • Immediate cardiologist consultation\n
            • Stress test / ECG\n
            • Possible angiography\n
            • Hospital observation recommended\n
            """)



col1, col2, col3 = st.columns(3)

with col1:
    show_card("❤️ Heart Disease", heart_prob)





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


## SHAP Explaination Graph
tab1, tab2 = st.tabs(["📊 Feature Importance", "🧠 SHAP Explanation"])

#########################################################################################################################
with tab1:

    st.subheader("Feature Importance")

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": columns,
        "importance": importances
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index("feature"),
        use_container_width=True
    )

#########################################################################################################################
with tab2:

    st.subheader("AI Explanation (SHAP)")

    fig = plt.figure()

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_val[0,0],
            base_values=explainer.expected_value[1],
            data=patients.iloc[0],
            feature_names=columns
        )
    )

    st.pyplot(fig)

































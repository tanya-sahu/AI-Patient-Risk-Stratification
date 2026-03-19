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



# =============================
# 7. TITLE
# =============================
st.title("AI Patient Risk Stratification System")





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







# =============================
# 9. MODEL INPUT
# =============================
input_data = [[age,sex,trestbps,chol,thalach,exang,
               oldpeak,ca,cp_non_anginal,cp_typical,thal_normal]]

df_input = pd.DataFrame(input_data, columns=heart_features)

prediction = heart_model.predict(df_input)
prob = heart_model.predict_proba(df_input)

explainer = shap.TreeExplainer(heart_model)
shap_values = explainer.shap_values(df_input)


# Handle both formats
if isinstance(shap_values, list):
    shap_val = shap_values[: , 1]   # class 1
    base_val = explainer.expected_value[1]
else:
    shap_val = shap_values
    base_val = explainer.expected_value

heart_prob = prob[0][1]






#===================================
## Patient Data for Logging
#===================================
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
        "Thalassemia: Normal" : thal_normal,
        "Patient Risk Score" : round(heart_prob*100 , 2)
    }


risk = heart_prob*100

if risk < 30:
    level = "Low"
elif risk < 70:
    level = "Medium"
else:
    level = "High"

patient_data["Risk Score"] = round(risk , 2)
patient_data["Priority"] = level






# =============================
# 10. DISPLAY CARD
# =============================
col1, col2 = st.columns(2)

with col1:
    show_card("❤️ Heart Risk", heart_prob)





# =============================
# 11. GAUGE
# =============================
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=heart_prob*100,
    title={'text': "Heart Risk %"},
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






# =============================
# 14. FEATURE IMPORTANCE
# =============================
st.subheader("Feature Importance")

imp = heart_model.feature_importances_

df_imp = pd.DataFrame({
    "Feature": heart_features,
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
        feature_names=heart_features
    )
)

st.pyplot(fig)


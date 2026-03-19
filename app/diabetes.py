# =============================
# 1. CONFIG 
# =============================
import streamlit as st

st.set_page_config(
    page_title="AI Patient Risk Stratification System",
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

sex = 1 if sex == "Male" else  0



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


input_data = [[preg , glucose , bp , skin , insulin , bmi , dpf , age]]
df_input = pd.DataFrame(input_data, columns=diabetes_features)

prediction = diabetes_model.predict(df_input)
prob = diabetes_model.predict_proba(df_input)

explainer = shap.TreeExplainer(diabetes_model)
shap_values = explainer.shap_values(df_input)


# Handle both formats
if isinstance(shap_values, list):
    shap_val = shap_values[: , 1]   # class 1
    base_val = explainer.expected_value[1]
else:
    shap_val = shap_values
    base_val = explainer.expected_value

diabetes_prob = prob[0][1]





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
        "Pregnancies": preg,
        "Glucose Level": glucose,
        "Blood Pressure": bp,
        "Skin Thickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "Diabetes Pedigree Function": dpf,
    }


risk = diabetes_prob*100

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
    show_card("❤️ Diabetes Risk", diabetes_prob)





# =============================
# 11. GAUGE
# =============================
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=diabetes_prob*100,
    title={'text': "Diabetes Risk %"},
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

imp = diabetes_model.feature_importances_

df_imp = pd.DataFrame({
    "Feature": diabetes_features,
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
        feature_names=diabetes_features
    )
)

st.pyplot(fig)







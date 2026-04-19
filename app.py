import streamlit as st
import pickle
import pandas as pd
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Heart Risk AI",
    page_icon="❤",
    layout="centered"
)

# ------------------ STYLING ------------------
st.markdown("""
<style>
.stApp {
 background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
 color: white;
}
h1, h2, h3, h4 {
 text-align: center;
}
.heart {
 font-size: 60px;
 text-align: center;
 animation: pulse 1.5s infinite;
 color: #ff4b4b;
}
@keyframes pulse {
 0% { transform: scale(1); }
 50% { transform: scale(1.15); }
 100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="heart">❤</div>', unsafe_allow_html=True)
st.markdown("<h1>Heart Disease Risk Assessment</h1>", unsafe_allow_html=True)

st.divider()

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("model.pkl", "rb"))

# ------------------ INPUT ------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    )
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)

with col2:
    fbs = st.selectbox("Fasting Sugar", ["<=120", ">120"])
    restecg = st.selectbox("Rest ECG",
        ["Normal", "ST-T", "LVH"]
    )
    exang = st.selectbox("Exercise Angina", ["No", "Yes"])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", ["Up", "Flat", "Down"])
    ca = st.selectbox("Major Vessels", [0,1,2,3,4])
    thal = st.selectbox("Thal", ["Normal", "Fixed", "Reversible"])

# ------------------ MAPPING (CRITICAL FIX) ------------------
sex = 1 if sex == "Male" else 0

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

fbs = 1 if fbs == ">120" else 0

restecg_map = {
    "Normal": 0,
    "ST-T": 1,
    "LVH": 2
}

exang = 1 if exang == "Yes" else 0

slope_map = {
    "Up": 0,
    "Flat": 1,
    "Down": 2
}

thal_map = {
    "Normal": 1,
    "Fixed": 2,
    "Reversible": 3
}

# ------------------ PREDICT ------------------
if st.button("Analyze"):

    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp_map[cp],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg_map[restecg],
        "thalch": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope_map[slope],
        "ca": ca,
        "thal": thal_map[thal]
    }])

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.metric("Risk Probability", f"{prob*100:.2f}%")

    if prediction == 1:
        st.error("High Risk")
    else:
        st.success("Low Risk")
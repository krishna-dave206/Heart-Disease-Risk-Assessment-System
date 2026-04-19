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
.stButton>button {
 border-radius: 8px;
 height: 3em;
 font-size: 16px;
}
.block-container {
 padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="heart">❤</div>', unsafe_allow_html=True)
st.markdown("<h1>Intelligent Heart Disease Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown("<h4>AI-Powered Clinical Decision Support System</h4>", unsafe_allow_html=True)

st.divider()

# ------------------ LOAD MODEL ------------------
model_path = os.path.join(os.getcwd(), "model.pkl")
model = pickle.load(open(model_path, "rb"))

# ------------------ INPUT ------------------
st.markdown("## Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])

    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar", ["≤120", ">120"])
    restecg = st.selectbox("ECG",
        ["Normal", "ST-T Abnormality", "LV Hypertrophy"])

    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", ["No", "Yes"])

    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Major Vessels", [0,1,2,3,4])
    thal = st.selectbox("Thal", ["Normal", "Fixed Defect", "Reversible Defect"])

# ------------------ ENCODING ------------------
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
    "ST-T Abnormality": 1,
    "LV Hypertrophy": 2
}

exang = 1 if exang == "Yes" else 0

slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

thal_map = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

# ------------------ PREDICTION ------------------
st.divider()

if st.button("Analyze Risk"):

    input_dict = {
        "age": age,
        "sex": sex,
        "cp": cp_map[cp],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg_map[restecg],
        "thalch": thalach,   # FIXED NAME
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope_map[slope],
        "ca": ca,
        "thal": thal_map[thal]
    }

    input_df = pd.DataFrame([input_dict])
    

    with st.spinner("Analyzing..."):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

    st.divider()
    st.markdown("## Result")

    st.metric("Risk Probability", f"{probability*100:.2f}%")
    st.progress(int(probability * 100))

    if prediction == 1:
        st.error("High Risk Detected")
    else:
        st.success("Low Risk")
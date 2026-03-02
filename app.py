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

# ------------------ BASIC INFO ------------------
st.markdown("## Basic Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 1, 100, 50)

with col2:
    sex = st.selectbox("Sex", ["Male", "Female"])

st.divider()

# ------------------ CLINICAL PARAMETERS ------------------
st.markdown("## Clinical Parameters")

col1, col2 = st.columns(2)

with col1:
    cp = st.selectbox("Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    )

    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar",
        ["≤ 120 mg/dl", "> 120 mg/dl"]
    )

    restecg = st.selectbox("Resting ECG Results",
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    )

    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

    slope = st.selectbox("Slope of ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )

    ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])

    thal = st.selectbox("Thalassemia",
        ["Normal", "Fixed Defect", "Reversible Defect", "Other"]
    )

# ------------------ PREDICTION ------------------
st.divider()

if st.button("Analyze Risk"):

    # IMPORTANT: Send raw values exactly like training data
    input_dict = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    input_df = pd.DataFrame([input_dict])

    with st.spinner("Analyzing patient data..."):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

    st.divider()
    st.markdown("## Risk Assessment Result")

    st.metric(
        label="Predicted Probability",
        value=f"{probability*100:.2f}%"
    )

    st.progress(int(probability * 100))

    if prediction == 1:
        st.error("High Risk of Heart Disease Detected")
        st.write("Clinical recommendation: Further medical evaluation is advised.")
    else:
        st.success("Low Risk of Heart Disease")
        st.write("Clinical recommendation: Maintain preventive healthcare practices.")
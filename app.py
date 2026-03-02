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

# ------------------ CUSTOM STYLING ------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Center headings */
h1, h2, h3, h4 {
    text-align: center;
}

/* Animated Heart */
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

/* Button styling */
.stButton>button {
    border-radius: 8px;
    height: 3em;
    font-size: 16px;
}

/* Padding */
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
# scaler_path = os.path.join(os.getcwd(), "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
# scaler = pickle.load(open(scaler_path, "rb"))

# ------------------ BASIC INFORMATION ------------------
st.markdown("## Basic Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 1, 100, 50)

with col2:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

sex_value = 1 if gender == "Male" else 0

st.divider()

# ------------------ CLINICAL PARAMETERS ------------------
st.markdown("## Clinical Parameters")

col1, col2 = st.columns(2)

with col1:
    cp = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        80, 200, 120
    )

    chol = st.number_input(
        "Cholesterol (mg/dl)",
        100, 400, 200
    )

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        60, 220, 150
    )

with col2:
    fbs_option = st.selectbox(
        "Fasting Blood Sugar",
        ["≤ 120 mg/dl (Normal)", "> 120 mg/dl (High)"]
    )

    restecg = st.selectbox(
        "Resting ECG Results",
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    )

    exang_option = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"]
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        0.0, 6.0, 1.0
    )

    slope = st.selectbox(
        "Slope of ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )

    ca = st.selectbox(
        "Number of Major Vessels (0–4)",
        [0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Thalassemia",
        ["Normal", "Fixed Defect", "Reversible Defect", "Other"]
    )

# ------------------ VALUE MAPPING ------------------

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

restecg_map = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

thal_map = {
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversible Defect": 2,
    "Other": 3
}

fbs_value = 1 if "> 120" in fbs_option else 0
exang_value = 1 if exang_option == "Yes" else 0

# ------------------ PREDICTION ------------------

st.divider()

if st.button("Analyze Risk"):

    input_dict = {
        "age": int(age),
        "sex": str(sex_value),
        "cp": str(cp_map[cp]),
        "trestbps": float(trestbps),
        "chol": float(chol),
        "fbs": str(fbs_value),
        "restecg": str(restecg_map[restecg]),
        "thalch": float(thalach),
        "exang": str(exang_value),
        "oldpeak": float(oldpeak),
        "slope": str(slope_map[slope]),
        "ca": float(ca),
        "thal": str(thal_map[thal])
    }

    input_df = pd.DataFrame([input_dict])

    with st.spinner("Analyzing patient data..."):
        # input_scaled = scaler.transform(input_df)
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
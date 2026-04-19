import streamlit as st
import pickle
import pandas as pd
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Heart Risk AI", page_icon="❤️", layout="centered")

# ------------------ OPENAI CLIENT (BUG FIX #1) ------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None  # was never instantiated before

# ------------------ LOAD MODEL (BUG FIX #2) ------------------
@st.cache_resource
def load_model():
    try:
        return pickle.load(open("model.pkl", "rb"))
    except FileNotFoundError:
        st.error("❌ model.pkl not found. Make sure it's in the same directory as app.py.")
        st.stop()

model = load_model()

# ------------------ CUSTOM STYLES ------------------
st.markdown("""
    <style>
        body { font-family: 'Segoe UI', sans-serif; }
        .risk-box {
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .high-risk  { background: #fdecea; color: #c0392b; border-left: 5px solid #c0392b; }
        .low-risk   { background: #eafaf1; color: #1e8449; border-left: 5px solid #1e8449; }
        .med-risk   { background: #fef9e7; color: #b7770d; border-left: 5px solid #f39c12; }
        .section-header { font-size: 0.85rem; text-transform: uppercase;
                          letter-spacing: 0.08em; color: #888; margin-top: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

# ------------------ UI ------------------
st.title("❤️ Heart Disease Risk Assessment")
st.caption("Enter patient details below and click **Analyze** for an AI-powered risk evaluation.")

# ------------------ INPUT ------------------
st.markdown('<p class="section-header">Patient Demographics</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 1, 100, 50)
with col2:
    sex = st.selectbox("Sex", ["Male", "Female"])

st.markdown('<p class="section-header">Cardiovascular Indicators</p>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar", ["≤120", ">120"])
with col4:
    restecg = st.selectbox("ECG Result", ["Normal", "ST-T", "LVH"])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

st.markdown('<p class="section-header">Additional Clinical Findings</p>', unsafe_allow_html=True)
col5, col6, col7 = st.columns(3)
with col5:
    slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
with col6:
    ca = st.selectbox("Major Vessels (Fluoroscopy)", [0, 1, 2, 3, 4])
with col7:
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed", "Reversible", "Other"])

# ------------------ ANALYZE ------------------
if st.button("🔍 Analyze Risk", use_container_width=True):

    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalach,      # exact column name the model was trained on
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # Force categorical columns to string (prevents dtype mismatch errors)
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    input_df[cat_cols] = input_df[cat_cols].astype(str)

    # ------------------ PREDICTION ------------------
    try:
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    risk_pct = prob * 100

    # ------------------ RISK DISPLAY ------------------
    st.markdown("---")
    st.subheader("📊 Assessment Result")

    # Colour-coded risk band
    if risk_pct >= 65:
        css_class, label = "high-risk", "⚠️ High Risk"
    elif risk_pct >= 35:
        css_class, label = "med-risk",  "🟡 Moderate Risk"
    else:
        css_class, label = "low-risk",  "✅ Low Risk"

    st.markdown(
        f'<div class="risk-box {css_class}">{label} — {risk_pct:.1f}% probability of heart disease</div>',
        unsafe_allow_html=True
    )

    st.progress(min(prob, 1.0))

    # ------------------ AI EXPLANATION (BUG FIX #3) ------------------
    if client is None:
        st.warning("⚠️ OPENAI_API_KEY not set in .env — skipping AI explanation.")
    else:
        with st.spinner("Generating clinical insight..."):
            prompt = f"""
You are a clinical decision-support assistant. A patient's heart disease risk model returned the following:

Patient Data:
{input_df.to_dict(orient='records')[0]}

Predicted Risk: {risk_pct:.1f}%
Classification: {"High Risk" if prediction == 1 else "Low Risk"}

In 3–4 sentences, provide:
1. A brief clinical interpretation of the key risk factors.
2. Specific, actionable prevention or follow-up advice.
Keep language clear and suitable for a clinician reading quickly.
"""
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                ai_text = response.choices[0].message.content
                st.markdown("### 🤖 AI Clinical Recommendation")
                st.info(ai_text)
            except Exception as e:
                st.error(f"❌ OpenAI call failed: {e}")

    st.caption("⚠️ This tool is for decision support only and does not replace professional medical diagnosis.")
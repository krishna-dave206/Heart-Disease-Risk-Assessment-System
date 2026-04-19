import streamlit as st
import pickle
import pandas as pd
import os
from openai import OpenAI

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Heart Risk AI", page_icon="❤")

# ------------------ OPENAI ------------------
OPENAI_API_KEY = os.getenv("REMOVED_API_KEY")

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("model.pkl", "rb"))

# ------------------ UI ------------------
st.title("❤ Heart Disease Risk Assessment")

# ------------------ INPUT ------------------
age = st.slider("Age", 1, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain", ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"])
trestbps = st.number_input("BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Sugar", ["≤120"," >120"])
restecg = st.selectbox("ECG", ["Normal","ST-T","LVH"])
thalach = st.number_input("Max HR", 60, 220, 150)
exang = st.selectbox("Exercise Angina", ["No","Yes"])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", ["Upsloping","Flat","Downsloping"])
ca = st.selectbox("Vessels", [0,1,2,3,4])
thal = st.selectbox("Thal", ["Normal","Fixed","Reversible","Other"])

# ------------------ FORMAT INPUT (IMPORTANT) ------------------
if st.button("Analyze"):

    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalach,   # EXACT NAME MATCH
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # 🔥 FORCE STRING FOR CATEGORICAL (THIS FIXES YOUR ERROR)
    cat_cols = ["sex","cp","fbs","restecg","exang","slope","thal"]
    input_df[cat_cols] = input_df[cat_cols].astype(str)

    # ------------------ PREDICTION ------------------
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.write(f"### Risk: {prob*100:.2f}%")

    # ------------------ AI (RAG STYLE EXPLANATION) ------------------
    prompt = f"""
    Patient Data:
    {input_df.to_dict()}

    Risk: {prob*100:.2f}%

    Give short clinical explanation and prevention advice.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    st.write("### AI Recommendation")
    st.write(response.choices[0].message.content)
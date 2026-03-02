
# Intelligent Heart Disease Risk Assessment System

## Overview
The Intelligent Heart Disease Risk Assessment System is an AI‑powered clinical decision support application that predicts the likelihood of heart disease using patient medical parameters. The system combines machine learning and an interactive web interface to provide real‑time risk analysis and interpretable probability output.

The objective of this project is to demonstrate an end‑to‑end machine learning deployment pipeline — from data preprocessing and model training to a publicly accessible web application.

This application is designed as an educational healthcare support tool and not as a substitute for professional medical diagnosis.

---

## Objectives
• Predict heart disease risk using clinical features  
• Provide probability‑based interpretation rather than only binary output  
• Implement a full machine learning pipeline  
• Deploy the model as an interactive web application  
• Demonstrate practical AI usage in healthcare analytics

---

## Technology Stack

| Component | Technology Used |
|----------|----------|
| Programming Language | Python |
| Machine Learning | Scikit‑Learn |
| Data Processing | Pandas, NumPy |
| User Interface | Streamlit |
| Model Storage | Pickle |
| Deployment | Streamlit Cloud |

---

## System Workflow

1. The user enters patient clinical data in the web interface.
2. Inputs are validated and structured into a dataset format.
3. Data is converted into a Pandas DataFrame matching the training schema.
4. The saved machine learning pipeline processes the input.
5. The model predicts:
   - Disease classification (Risk / No Risk)
   - Probability of heart disease
6. The result is visualized in the interface with risk interpretation and recommendation.

---
## Input Features

The prediction model uses the following medical parameters:

- Age
- Gender
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol Level
- Fasting Blood Sugar
- Resting ECG Results
- Maximum Heart Rate Achieved
- Exercise‑Induced Angina
- ST Depression (Oldpeak)
- Slope of ST Segment
- Number of Major Vessels
- Thalassemia Type

---


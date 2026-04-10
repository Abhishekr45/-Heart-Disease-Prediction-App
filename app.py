import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: red;'>❤️ Heart Disease Prediction App</h1>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar for inputs
st.sidebar.header("🧾 Patient Details")

age = st.sidebar.slider("Age", 18, 100, 40)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
chest_pain = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.sidebar.number_input("Resting BP", 80, 200, 120)
cholesterol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar", [0, 1])
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.sidebar.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Input Summary")
    st.write({
        "Age": age,
        "Sex": sex,
        "Cholesterol": cholesterol,
        "BP": resting_bp
    })

with col2:
    st.subheader("💡 Prediction Result")

# Predict button
if st.button("🔍 Predict Now"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
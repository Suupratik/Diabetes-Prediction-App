import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
st.title("🩺 Diabetes Prediction System")
st.markdown("Enter the patient’s medical details to predict diabetes.")

# Sidebar Info
st.sidebar.title("About")
st.sidebar.info("This app uses a machine learning model (SVM) trained on the PIMA Diabetes Dataset.")

# Input Form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
        glucose = st.number_input("Glucose", min_value=0)
        blood_pressure = st.number_input("Blood Pressure", min_value=0)

    with col2:
        skin_thickness = st.number_input("Skin Thickness", min_value=0)
        insulin = st.number_input("Insulin", min_value=0)
        bmi = st.number_input("BMI", min_value=0.0, format="%.2f")

    with col3:
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
        age = st.number_input("Age", min_value=0)

    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    input_data = np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree, age
    ]])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.error("🚨 The person is **diabetic**.")
    else:
        st.success("✅ The person is **not diabetic**.")

# Visualizations
st.subheader("📊 Data Visualizations")
if st.checkbox("Show sample data & plots"):
    df = pd.read_csv("diabetes.csv")

    st.write("### Sample of Dataset")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Diabetes Count")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x="Outcome", palette="Set2", ax=ax1)
        ax1.set_xticklabels(["Not Diabetic", "Diabetic"])
        st.pyplot(fig1)

    with col2:
        st.write("### Age Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df["Age"], bins=30, kde=True, ax=ax2, color="skyblue")
        st.pyplot(fig2)

    st.write("### Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

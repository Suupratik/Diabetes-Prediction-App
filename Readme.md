# 🧪 Diabetes Prediction App

A **machine learning web app** built using **Streamlit** that predicts whether a person is diabetic based on medical parameters. Powered by an SVM classifier and enriched with visual insights using Matplotlib and Seaborn.

---

## 🚀 Features

- 🔍 Predict diabetes using:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
- 📈 Displays:
  - Model accuracy (training & testing)
  - Confusion matrix plot
  - Data distribution visualization
- 🧠 Backend: Trained with Support Vector Machine (SVM)
- 💡 User-friendly UI with Streamlit

---

## 🧰 Tech Stack

- Python 🐍
- Streamlit 📊
- scikit-learn 🤖
- pandas, numpy 🧮
- seaborn & matplotlib 📈

---

## 📁 Folder Structure

Diabetes/
│
├── app.py # Streamlit web app
├── train_model.py # Model training and saving
├── diabetes.csv # Dataset used
├── model.pkl # Trained ML model
├── scaler.pkl # StandardScaler object
├── confusion_matrix.png # Plot image auto-generated
├── README.md # You're reading this!
## 📌 Credits

- Developed by [Supratik Mitra](https://github.com/Suupratik)
- Dataset: [PIMA Indian Diabetes Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

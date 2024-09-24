import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
voting_clf = joblib.load('voting_clf (3).pkl')  # Adjust the model filename if necessary
scaler = joblib.load('scaler (1).pkl')

# Streamlit UI for user input
st.title("Heart Disease Prediction App")

# Collect numerical feature inputs
age = st.number_input("Age (years)", 0, 120)
sysBP = st.number_input("Systolic Blood Pressure (mm Hg)", 0, 300)
diaBP = st.number_input("Diastolic Blood Pressure (mm Hg)", 0, 200)
glucose = st.number_input("Glucose Level (mg/dL)", 0, 300)
BMI = st.number_input("Body Mass Index (kg/m²)", 0.0, 50.0)
heartRate = st.number_input("Heart Rate (bpm)", 0, 200)
cigsPerDay = st.number_input("Cigarettes per Day", 0, 50)
totChol = st.number_input("Total Cholesterol (mg/dL)", 0, 400)

# Collect categorical feature inputs
gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")  # 0 = Female, 1 = Male
education = st.selectbox("Education Level", [1, 2, 3, 4])  # Example categories
currentSmoker = st.selectbox("Current Smoker", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")  # 0 = No, 1 = Yes
BPMeds = st.selectbox("On BP Meds", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")  # 0 = No, 1 = Yes
prevalentStroke = st.selectbox("Prevalent Stroke", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")  # 0 = No, 1 = Yes
prevalentHyp = st.selectbox("Prevalent Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")  # 0 = No, 1 = Yes
diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")  # 0 = No, 1 = Yes

# Combine all inputs into a single array (now 15 features including gender)
input_data = np.array([[age, sysBP, diaBP, glucose, BMI, heartRate, cigsPerDay, totChol,
                        education, currentSmoker, BPMeds, prevalentStroke, prevalentHyp, diabetes, gender]])

# Scale the numerical features (first 8 columns)
input_data_scaled = scaler.transform(input_data[:, :8])  # Scale only the numerical features

# Combine scaled numerical features and categorical features for final input
final_input = np.hstack([input_data_scaled, input_data[:, 8:]])  # Concatenate scaled numerical and categorical data

# Make predictions
prediction = voting_clf.predict(final_input)
probabilities = voting_clf.predict_proba(final_input)

# Display results
st.write("Prediction:", "Heart Disease" if prediction[0] == 1 else "No Heart Disease")
st.write("Probability of Heart Disease: {:.2f}%".format(probabilities[0][1] * 100))

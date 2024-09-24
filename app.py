import streamlit as st
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the best model and imputer
with open('best_voting_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('imputer.pkl', 'rb') as imputer_file:
    imputer = pickle.load(imputer_file)

# Load feature importance data
feature_importance_df = pd.read_csv('feature_importance.csv')

# Streamlit UI
st.title("Heart Disease Prediction App")

# User input for features
age = st.number_input("Age", min_value=0, max_value=120)
sex = st.selectbox("Sex", options=["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
resting_ecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2])
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250)
exercise_angina = st.selectbox("Exercise Angina", options=[0, 1])
oldpeak = st.number_input("Oldpeak (depression induced by exercise)", min_value=0.0, max_value=10.0)
st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])

# Prepare input data
input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [1 if sex == "Male" else 0],
    'ChestPainType': [chest_pain_type],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope],
})

# Impute missing values
input_data_imputed = imputer.transform(input_data)

# Make predictions
prediction = model.predict(input_data_imputed)

if prediction[0] == 0:
    st.write("The model predicts: Healthy Heart")
else:
    st.write("The model predicts: Defective Heart")

# Feature Importance Visualization
st.subheader("Feature Importance")
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance for Heart Disease Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
st.pyplot(plt)

# Display Test Set Evaluation Metrics
st.subheader("Model Evaluation Metrics")
test_metrics_df = pd.read_csv('test_metrics.csv')
st.write(test_metrics_df)

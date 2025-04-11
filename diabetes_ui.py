import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('diabetes_python.csv')

# Preprocessing
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Train model
clf = LogisticRegression(class_weight='balanced', max_iter=1000)
clf.fit(X_train, y_train)

# Streamlit UI
st.title("Diabetes Prediction")

# Input sliders
pregnancies = st.slider("Pregnancies", 0, 17, 1)
glucose = st.slider("Glucose", 0, 200, 120)
blood_pressure = st.slider("Blood Pressure", 0, 122, 70)
skin_thickness = st.slider("Skin Thickness", 0, 99, 20)
insulin = st.slider("Insulin", 0, 846, 79)
bmi = st.slider("BMI", 0.0, 67.1, 32.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.42, 0.5)
age = st.slider("Age", 21, 81, 33)

# Create input array
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)

# Adversarial attack simulation
if st.checkbox("Simulate Adversarial Attack"):
    noise = np.random.normal(0, 0.1, input_scaled.shape)
    input_scaled += noise

# Prediction
prediction = clf.predict(input_scaled)[0]
probability = clf.predict_proba(input_scaled)[0][1]

# Display result
st.write(f"Prediction: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
st.write(f"Probability: {probability:.2f}")

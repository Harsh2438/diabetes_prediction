import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------
# CUSTOM STYLING
# ----------------------------------------------------------------------------
st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stApp {
        background: linear-gradient(180deg, #0f1117 0%, #151823 100%);
    }
    .title-text {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0rem;
    }
    .subtitle-text {
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-positive {
        background: linear-gradient(135deg, #3f1720 0%, #2a1015 100%);
        border: 1px solid #7f1d1d;
    }
    .result-negative {
        background: linear-gradient(135deg, #0f2e1f 0%, #0a2016 100%);
        border: 1px solid #14532d;
    }
    .result-label {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-sub {
        font-size: 0.95rem;
        color: #9ca3af;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e5e7eb;
        margin-top: 1.2rem;
        margin-bottom: 0.3rem;
        border-bottom: 1px solid #2d2f3a;
        padding-bottom: 0.4rem;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# LOAD DATA + TRAIN MODEL (cached so it only runs once, not on every interaction)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv('diabetes_python.xls')
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    scaler.transform(X_test_raw)  # kept for parity with original pipeline

    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train, y_train)
    return clf, scaler

clf, scaler = load_model()

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown('<p class="title-text">🩺 Diabetes Risk Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle-text">Logistic Regression model trained on the Pima Indians Diabetes dataset. '
    'Adjust the values below to estimate diabetes risk.</p>',
    unsafe_allow_html=True
)

# ----------------------------------------------------------------------------
# SIDEBAR — About / Info
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About this app")
    st.write(
        "This tool uses a **Logistic Regression** model trained on the "
        "Pima Indians Diabetes dataset (768 patient records) to estimate "
        "the likelihood of diabetes based on 8 medical measurements."
    )
    st.divider()
    st.caption("⚠️ For educational/demo purposes only. Not a medical diagnosis tool.")
    st.divider()
    st.write("**Model details**")
    st.write("- Algorithm: Logistic Regression")
    st.write("- Class balancing: enabled")
    st.write("- Features scaled with StandardScaler")

# ----------------------------------------------------------------------------
# INPUT FORM — organized into logical groups with columns
# ----------------------------------------------------------------------------
st.markdown('<p class="section-header">Patient Information</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.slider("Pregnancies", 0, 17, 1)
    blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 122, 70)
    insulin = st.slider("Insulin (mu U/ml)", 0, 846, 79)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.42, 0.5)

with col2:
    glucose = st.slider("Glucose (mg/dL)", 0, 200, 120)
    skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20)
    bmi = st.slider("BMI", 0.0, 67.1, 32.0)
    age = st.slider("Age", 21, 81, 33)

st.markdown('<p class="section-header">Model Testing</p>', unsafe_allow_html=True)
adversarial = st.checkbox("🧪 Simulate adversarial noise (test model robustness)")

# ----------------------------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------------------------
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                         insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)

if adversarial:
    noise = np.random.normal(0, 0.1, input_scaled.shape)
    input_scaled = input_scaled + noise

predict_clicked = st.button("🔍 Predict", use_container_width=True, type="primary")

if predict_clicked:
    prediction = clf.predict(input_scaled)[0]
    probability = clf.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.markdown(f"""
            <div class="result-card result-positive">
                <div class="result-label">⚠️ Diabetes Risk Detected</div>
                <div class="result-sub">Estimated probability: {probability:.0%}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-card result-negative">
                <div class="result-label">✅ Low Diabetes Risk</div>
                <div class="result-sub">Estimated probability: {probability:.0%}</div>
            </div>
        """, unsafe_allow_html=True)

    st.progress(float(probability))

    m1, m2, m3 = st.columns(3)
    m1.metric("Glucose", f"{glucose} mg/dL")
    m2.metric("BMI", f"{bmi:.1f}")
    m3.metric("Age", f"{age} yrs")

    if adversarial:
        st.caption("🧪 Noise was added to the scaled input before this prediction — "
                   "useful for testing how sensitive the model is to small perturbations.")
else:
    st.info("Adjust the sliders above and click **Predict** to see the result.")

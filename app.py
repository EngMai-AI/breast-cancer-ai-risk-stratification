import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Breast Cancer AI System",
    page_icon="🩺",
    layout="wide"
)

st.title("🧠 AI-Powered Breast Cancer Risk Stratification System")

st.markdown("""
⚠️ Educational tool only — Not a diagnostic system
""")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv("cancer_classification.csv")

model, scaler = load_assets()
df = load_data()
features = df.columns[:-1]

# ===============================
# INPUTS
# ===============================
st.sidebar.header("🧬 Patient Parameters")

input_data = []
for f in features:
    input_data.append(
        st.sidebar.number_input(
            f,
            float(df[f].min()),
            float(df[f].max()),
            float(df[f].mean())
        )
    )

mode = st.sidebar.selectbox(
    "⚖️ Decision Mode",
    ["Screening (0.3)", "Balanced (0.5)", "Strict (0.7)"]
)

threshold = {
    "Screening (0.3)": 0.3,
    "Balanced (0.5)": 0.5,
    "Strict (0.7)": 0.7
}[mode]

# ===============================
# PREDICTION
# ===============================
if st.sidebar.button("🔍 Run Analysis"):

    X = np.array(input_data).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0][1]
    benign = 1 - prob

    risk = "Low"
    if prob >= 0.7:
        risk = "High"
    elif prob >= threshold:
        risk = "Moderate"

    confidence = max(prob, benign)
    uncertainty = 1 - confidence

    # ===============================
    # OUTPUT
    # ===============================
    st.subheader("🏥 System Output")

    if risk == "High":
        st.error("🔴 HIGH RISK")
    elif risk == "Moderate":
        st.warning("🟡 MODERATE RISK")
    else:
        st.success("🟢 LOW RISK")

    col1, col2, col3 = st.columns(3)
    col1.metric("Malignant Probability", f"{prob:.2%}")
    col2.metric("Benign Probability", f"{benign:.2%}")
    col3.metric("Confidence", f"{confidence:.2%}")

    st.metric("Uncertainty", f"{uncertainty:.2%}")

    st.subheader("📋 Patient Data")
    st.dataframe(pd.DataFrame([input_data], columns=features))

    # ===============================
    # SAFE FEATURE IMPORTANCE (NO SHAP)
    # ===============================
    st.subheader("📊 Feature Contribution (Approx)")

    try:
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:10]

        fig, ax = plt.subplots()
        ax.barh(range(10), importances[idx][::-1])
        ax.set_yticks(range(10))
        ax.set_yticklabels(features[idx][::-1])
        ax.set_title("Top 10 Features")

        st.pyplot(fig)

    except:
        st.info("Model does not support feature importance.")

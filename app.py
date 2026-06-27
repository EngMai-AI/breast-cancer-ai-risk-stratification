import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import shap

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Breast Cancer AI System",
    page_icon="🩺",
    layout="wide"
)

# ===============================
# HEADER
# ===============================
st.markdown("""
# 🧠 AI-Powered Breast Cancer Risk Stratification System
""")

st.success("🧠 AI System for Breast Cancer Risk Stratification using Deep Learning + Explainable AI (SHAP)")
st.warning("⚠️ Educational tool only — Not intended for clinical diagnosis")

st.markdown("---")

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_assets():
    model = load_model("cancer_prediction_model (2).keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv("cancer_classification.csv")

model, scaler = load_assets()
df = load_data()
features = df.columns[:-1]

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.header("🧬 Patient Parameters")

input_data = []
for f in features:
    input_data.append(
        st.sidebar.number_input(
            f,
            float(df[f].min()),
            float(df[f].max()),
            float(df[f].mean()),
            format="%.5f"
        )
    )

mode = st.sidebar.selectbox(
    "⚖️ Decision Mode",
    ["Screening (0.3)", "Balanced (0.5)", "Strict (0.7)"]
)

threshold = {"Screening (0.3)":0.3,"Balanced (0.5)":0.5,"Strict (0.7)":0.7}[mode]

# ===============================
# RUN MODEL
# ===============================
if st.sidebar.button("🔍 Run Analysis"):

    with st.spinner("🧠 AI Model Processing..."):

        X = np.array(input_data).reshape(1, -1)
        X_scaled = scaler.transform(X)

        prob = float(model.predict(X_scaled, verbose=0)[0][0])
        benign = 1 - prob

        prediction = int(prob >= threshold)

        confidence = max(prob, benign)
        uncertainty = 1 - confidence

        if prob >= 0.7:
            risk = "High"
        elif prob >= threshold:
            risk = "Moderate"
        else:
            risk = "Low"

    # ===============================
    # SYSTEM HEADER OUTPUT
    # ===============================
    st.markdown("## 🏥 System Output")

    st.markdown("### 🧠 Risk Assessment")

    if risk == "High":
        st.markdown("🔴 HIGH RISK DETECTED")
        st.error("Immediate clinical attention recommended")
    elif risk == "Moderate":
        st.markdown("🟡 MODERATE RISK")
        st.warning("Further diagnostic evaluation recommended")
    else:
        st.markdown("🟢 LOW RISK")
        st.success("No strong malignancy indicators detected")

    st.info(f"🧠 Risk Level: {risk} | Mode: {mode}")

    # ===============================
    # METRICS
    # ===============================
    col1, col2, col3 = st.columns(3)

    col1.metric("Malignant Probability", f"{prob:.2%}")
    col2.metric("Benign Probability", f"{benign:.2%}")
    col3.metric("Confidence", f"{confidence:.2%}")

    st.metric("Uncertainty", f"{uncertainty:.2%}")

    # ===============================
    # PROBABILITY VISUALIZATION
    # ===============================
    st.markdown("### 📊 Probability Distribution")

    st.progress(prob)
    st.caption(f"Malignant Probability: {prob:.2%}")

    st.progress(benign)
    st.caption(f"Benign Probability: {benign:.2%}")

    st.markdown("---")

    # ===============================
    # PATIENT DATA
    # ===============================
    st.subheader("📋 Patient Summary")
    st.dataframe(pd.DataFrame([input_data], columns=features), use_container_width=True)

    st.markdown("---")

    # ===============================
    # CLINICAL INTERPRETATION
    # ===============================
    st.subheader("🧠 Clinical Interpretation")

    if risk == "High":
        st.error("Immediate clinical attention recommended.")
    elif risk == "Moderate":
        st.warning("Recommend diagnostic imaging and follow-up.")
    else:
        st.success("Routine follow-up suggested.")

    st.markdown("---")

    # ===============================
    # SHAP EXPLANATION
    # ===============================
    st.subheader("🧠 Explainable AI (Top Contributing Features)")

    @st.cache_resource
    def get_explainer(bg):
        return shap.KernelExplainer(model.predict, bg)

    bg = scaler.transform(df.sample(30, random_state=42).iloc[:, :-1])
    explainer = get_explainer(bg)

    shap_values = explainer.shap_values(X_scaled, nsamples=10)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.array(shap_values).reshape(-1)

    shap_df = pd.DataFrame({
        "Feature": features,
        "Impact": shap_values
    })

    shap_df = shap_df.reindex(
        shap_df["Impact"].abs().sort_values(ascending=False).index
    )

    top_shap = shap_df.head(10)

    st.bar_chart(top_shap.set_index("Feature"))

    st.caption("📌 SHAP-based feature importance ranking")

    st.markdown("---")

    # ===============================
    # ROC CURVE
    # ===============================
    st.subheader("📊 Model Performance (ROC Curve)")

    roc_data = joblib.load("roc_data.pkl")

    if isinstance(roc_data, dict):
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        roc_auc = roc_data["auc"]
    else:
        fpr, tpr, roc_auc = roc_data

    st.metric("AUC Score", f"{roc_auc:.3f}")

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic Curve")
    ax.legend()

    st.pyplot(fig)

    st.markdown("---")

    # ===============================
    # AI SUMMARY
    # ===============================
    st.subheader("🧠 AI Decision Summary")

    st.info(f"""
🧠 The model classified this case as **{risk} risk**.

- Malignant Probability: {prob:.2%}  
- Confidence: {confidence:.2%}  
- Model AUC: {roc_auc:.3f}  

SHAP analysis highlights the most influential features driving this prediction.
""")

    # ===============================
    # DISCLAIMER
    # ===============================
    st.markdown("---")

    st.warning("""
⚠️ Disclaimer:
This system is intended for educational and research purposes only.  
It does not replace professional medical diagnosis or clinical judgment.
""")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc

# ---------------------- Page Config ----------------------
st.set_page_config(page_title="🍷 Wine Quality Prediction", layout="centered")
st.title("🍷 Wine Quality Prediction Web App")
st.markdown("### Logistic Regression Model (Vidusahan Perera - FC211032)")
st.markdown("---")

# ---------------------- Load Model ----------------------
@st.cache_resource
def load_model():
    return joblib.load("models/FC211032_VidusahanPerera/logreg_wine_quality_pipeline.joblib")

model = load_model()
st.success("Model loaded successfully ✅")

# ---------------------- Input Section ----------------------
st.sidebar.header("Enter Wine Properties")

feature_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

feature_ranges = {
    "fixed acidity": (4.6, 15.9),
    "volatile acidity": (0.12, 1.58),
    "citric acid": (0.0, 1.0),
    "residual sugar": (0.9, 15.5),
    "chlorides": (0.012, 0.611),
    "free sulfur dioxide": (1, 72),
    "total sulfur dioxide": (6, 289),
    "density": (0.990, 1.004),
    "pH": (2.74, 4.01),
    "sulphates": (0.33, 2.0),
    "alcohol": (8.4, 14.9)
}

user_input = {}
for feature in feature_names:
    low, high = feature_ranges[feature]
    default = (low + high) / 2
    user_input[feature] = st.sidebar.slider(feature, float(low), float(high), float(default))

input_df = pd.DataFrame([user_input])

st.subheader("Entered Wine Properties")
st.dataframe(input_df, use_container_width=True)

# ---------------------- Prediction ----------------------
if st.button("🔮 Predict Wine Quality"):
    proba = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]
    
    st.markdown("### 🧾 Prediction Result")
    st.write(f"**Predicted Probability of Good Quality (≥7): {proba:.2f}**")
    
    if pred == 1:
        st.success("✅ The model predicts this wine is **Good (quality ≥ 7)**")
    else:
        st.error("❌ The model predicts this wine is **Not Good (quality < 7)**")
    
    # Progress bar as a probability gauge
    st.progress(float(proba))

    # Show probability explanation
    st.info("A probability closer to 1.0 indicates higher confidence that the wine is of good quality.")

# ---------------------- Feature Importance ----------------------
st.markdown("---")
st.subheader("📊 Feature Importance (Standardized Coefficients)")

try:
    coefs = model.named_steps['clf'].coef_.flatten()
    features = model.named_steps['scaler'].get_feature_names_out(feature_names)
except AttributeError:
    coefs = model.named_steps['clf'].coef_.flatten()
    features = feature_names

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefs
}).sort_values("Coefficient", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=coef_df, x="Coefficient", y="Feature", palette="coolwarm", ax=ax)
ax.set_title("Feature Importance (Positive → Higher Quality)")
st.pyplot(fig)

# ---------------------- ROC Curve Demo ----------------------
st.markdown("---")
st.subheader("📈 ROC Curve Example")

# Generate simulated ROC for visualization
fpr, tpr, thresholds = roc_curve(
    [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
)
roc_auc = auc(fpr, tpr)

fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
ax_roc.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("Receiver Operating Characteristic (Demo)")
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

# ---------------------- Footer ----------------------
st.markdown("---")
st.caption("Developed by **Vidusahan Perera (FC211032)** | Machine Learning Group Project")


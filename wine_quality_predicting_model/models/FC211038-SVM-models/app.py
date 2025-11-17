import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="Wine Quality AI | SVM",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------- Custom CSS ----------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .good-wine {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        border: 3px solid #2e7d32;
    }
    .bad-wine {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        border: 3px solid #c62828;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stSlider > div > div > div {
        background: #667eea;
    }
    .quality-badge {
        font-size: 1.2rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- Header ----------------------
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 3rem;">🍷 Wine Quality AI</h1>
    <h3 style="margin:0; font-weight:300;">Support Vector Machine Powered Quality Prediction</h3>
    <p style="margin:0; opacity: 0.9;">By Kavindi Hewawasam | FC211038</p>
</div>
""", unsafe_allow_html=True)

# ---------------------- Load Model ----------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "wine_quality_svm_model.pkl"
    
    if not model_path.exists():
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()
    
    artifact = joblib.load(model_path)
    return {
        "model": artifact.get('model'),
        "scaler": artifact.get('scaler'),
        "feature_names": artifact.get('feature_names'),
        "threshold": artifact.get('optimal_threshold')
    }

mdl = load_model()

# ---------------------- Main Layout ----------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🎛️ Wine Characteristics Panel")
    
    # Organize features into categories
    st.markdown("#### 🧪 Basic Properties")
    col1a, col1b = st.columns(2)
    with col1a:
        fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.32, 0.1, 
                                 help="Non-volatile acids in wine")
        volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.53, 0.01,
                                   help="Acids that vaporize easily")
        citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.27, 0.01,
                              help="Contributes to freshness")
    
    with col1b:
        residual_sugar = st.slider("Residual Sugar", 0.9, 16.0, 2.54, 0.1,
                                 help="Sugar remaining after fermentation")
        chlorides = st.slider("Chlorides", 0.01, 0.61, 0.09, 0.001,
                            help="Salt content")
    
    st.markdown("#### 🧬 Chemical Composition")
    col2a, col2b = st.columns(2)
    with col2a:
        free_sulfur_dioxide = st.slider("Free SO₂", 1.0, 72.0, 15.9, 1.0,
                                      help="Free sulfur dioxide")
        total_sulfur_dioxide = st.slider("Total SO₂", 6.0, 289.0, 46.5, 1.0,
                                       help="Total sulfur dioxide")
    
    with col2b:
        density = st.slider("Density", 0.990, 1.004, 0.9967, 0.0001,
                          help="Density of wine")
        pH = st.slider("pH Level", 2.7, 4.0, 3.31, 0.01,
                     help="Acidity/basicity measure")
    
    st.markdown("#### 🍇 Final Properties")
    col3a, col3b = st.columns(2)
    with col3a:
        sulphates = st.slider("Sulphates", 0.3, 2.0, 0.66, 0.01,
                            help="Wine additive")
    with col3b:
        alcohol = st.slider("Alcohol %", 8.0, 15.0, 10.42, 0.1,
                          help="Alcohol content by volume")

with col2:
    st.markdown("### 📊 Analysis & Prediction")
    
    # Create input array
    features = [
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
    ]
    
    input_df = pd.DataFrame([features], columns=mdl["feature_names"])
    
    # Display current values in a nice card
    st.markdown("#### 📋 Current Wine Profile")
    profile_cols = st.columns(3)
    with profile_cols[0]:
        st.metric("Alcohol", f"{alcohol}%")
        st.metric("pH", f"{pH}")
    with profile_cols[1]:
        st.metric("Acidity", f"{volatile_acidity}")
        st.metric("Sugar", f"{residual_sugar}g/L")
    with profile_cols[2]:
        st.metric("SO₂ Total", f"{total_sulfur_dioxide}")
        st.metric("Density", f"{density:.3f}")
    
    # Prediction Button
    if st.button("🚀 Analyze Wine Quality", use_container_width=True):
        # Scale and predict
        input_scaled = mdl["scaler"].transform(input_df)
        proba = mdl["model"].predict_proba(input_scaled)[0, 1]
        pred = int(proba >= mdl["threshold"])
        
        # Display results
        st.markdown("---")
        
        if pred == 1:
            st.markdown(f"""
            <div class="good-wine">
                <h2 style="margin:0; color: white;">✅ GOOD QUALITY WINE</h2>
                <div class="quality-badge" style="background: #2e7d32; color: white;">Quality ≥ 6</div>
                <h1 style="margin:0; font-size: 4rem; color: white;">{(proba*100):.1f}%</h1>
                <p style="margin:0; font-size: 1.2rem;">Confidence Score</p>
                <p style="margin-top: 1rem;">🎉 This wine is predicted to be of <b>GOOD quality</b> according to SVM model</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bad-wine">
                <h2 style="margin:0; color: white;">❌ BAD QUALITY WINE</h2>
                <div class="quality-badge" style="background: #c62828; color: white;">Quality < 6</div>
                <h1 style="margin:0; font-size: 4rem; color: white;">{(proba*100):.1f}%</h1>
                <p style="margin:0; font-size: 1.2rem;">Confidence Score</p>
                <p style="margin-top: 1rem;">⚠️ This wine is predicted to be of <b>BAD quality</b> according to SVM model</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence gauge with color coding
        st.markdown("#### 📈 Prediction Confidence")
        
        # Color code the progress bar based on prediction
        if pred == 1:
            st.markdown(f'<div style="color: #2e7d32; font-weight: bold;">🟢 High confidence for GOOD quality: {proba*100:.1f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color: #c62828; font-weight: bold;">🔴 High confidence for BAD quality: {(1-proba)*100:.1f}%</div>', unsafe_allow_html=True)
        
        st.progress(float(proba))
        
        # Show threshold information
        st.markdown("---")
        st.info(f"**Model Details:** Using optimal threshold of **{mdl['threshold']:.3f}** - Probability ≥ {mdl['threshold']:.3f} = GOOD, < {mdl['threshold']:.3f} = BAD")

# ---------------------- Model Insights ----------------------
st.markdown("---")
st.markdown("## 🔬 SVM Model Insights")

col3, col4 = st.columns([2, 1])

with col3:
    st.markdown("### 📊 Feature Impact Analysis")
    
    # Feature importance visualization
    importance_data = {
        'Feature': ['Alcohol', 'Sulphates', 'Total SO₂', 'Volatile Acidity', 'Free SO₂',
                   'Chlorides', 'Density', 'Fixed Acidity', 'pH', 'Citric Acid', 'Residual Sugar'],
        'Impact': [9.7, 6.5, 3.4, 3.2, 1.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(importance_data['Feature']))
    
    bars = ax.barh(y_pos, importance_data['Impact'], color=plt.cm.viridis(np.linspace(0, 1, len(y_pos))))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_data['Feature'])
    ax.set_xlabel('Feature Importance Score')
    ax.set_title('SVM Feature Impact on Wine Quality Prediction')
    
    # Add value labels
    for i, v in enumerate(importance_data['Impact']):
        ax.text(v + 0.1, i, f'{v}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

with col4:
    st.markdown("### 🎯 Model Performance")
    
    metrics = [
        ("Accuracy", "76.5%", "📊"),
        ("Precision", "76.3%", "🎯"),
        ("Recall", "80.6%", "🔍"),
        ("F1-Score", "78.4%", "⭐")
    ]
    
    for metric, value, icon in metrics:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color: #667eea;">{icon}</h3>
            <h4 style="margin:0.5rem 0; color: #333;">{metric}</h4>
            <h2 style="margin:0; color: #764ba2;">{value}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Specs")
    st.write("**Algorithm:** SVM RBF")
    st.write("**Kernel:** Radial Basis Function")
    st.write("**Training Samples:** 1,087")
    st.write(f"**Optimal Threshold:** {mdl['threshold']:.3f}")
    st.write("**CV Score:** 0.756 ± 0.034")
    
    # Quality definition
    st.markdown("### 🎯 Quality Definition")
    st.write("**GOOD Wine:** Quality score ≥ 6")
    st.write("**BAD Wine:** Quality score < 6")

# ---------------------- Technical Details ----------------------
with st.expander("🔍 Technical Details & Methodology"):
    tab1, tab2, tab3 = st.tabs(["🧠 Model Architecture", "📈 Training Process", "🎯 Performance"])
    
    with tab1:
        st.markdown("""
        **Support Vector Machine Configuration:**
        - **Kernel**: RBF (Radial Basis Function)
        - **Regularization (C)**: 1.0
        - **Class Weight**: Balanced
        - **Gamma**: Scale
        
        **Binary Classification:**
        - **Class 1 (GOOD)**: Wine quality ≥ 6
        - **Class 0 (BAD)**: Wine quality < 6
        
        **Preprocessing Pipeline:**
        1. Outlier detection and clipping using IQR method
        2. StandardScaler for feature normalization
        3. Class weight balancing
        4. Probability calibration
        """)
    
    with tab2:
        st.markdown("""
        **Training Methodology:**
        - **Dataset**: 1,359 unique wine samples
        - **Train/Test Split**: 80/20 stratified split
        - **Cross-Validation**: 5-fold stratified CV
        - **Hyperparameter Tuning**: GridSearchCV
        
        **Optimal Parameters Found:**
        ```python
        {
            'C': 1,
            'class_weight': 'balanced', 
            'gamma': 'scale',
            'kernel': 'rbf'
        }
        ```
        """)
    
    with tab3:
        st.markdown("""
        **Performance Metrics:**
        | Class | Precision | Recall | F1-Score | Support |
        |-------|-----------|--------|----------|---------|
        | BAD (0) | 0.767 | 0.719 | 0.742 | 128 |
        | GOOD (1) | 0.763 | 0.806 | 0.784 | 144 |
        
        **Key Achievements:**
        - Excellent recall for GOOD quality wines (80.6%)
        - Balanced precision across both classes
        - Robust cross-validation performance
        """)

# ---------------------- Footer ----------------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])

with footer_col1:
    st.markdown("""
    **🍷 Wine Quality AI** | *Support Vector Machine Implementation*  
    Developed by **Kavindi Hewawasam (FC211038)**  
    Machine Learning Project | Binary Classification: GOOD vs BAD Wine
    """)

with footer_col2:
    st.markdown("""
    **Model Type:** SVM RBF  
    **Classification:** Binary  
    **Accuracy:** 76.5%
    """)

with footer_col3:
    st.markdown("""
    **Dataset:** Wine Quality Red  
    **Samples:** 1,359  
    **Classes:** GOOD/BAD
    """)
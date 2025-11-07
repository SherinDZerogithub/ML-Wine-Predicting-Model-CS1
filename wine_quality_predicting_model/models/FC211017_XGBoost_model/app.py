import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Vinología - Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    h1 {
        font-family: 'Georgia', serif;
        color: #d4a574;
        text-align: center;
        font-size: 4em;
        text-shadow: 0 0 30px rgba(212, 165, 116, 0.3);
    }
    
    .subtitle {
        text-align: center;
        color: #c4b5a0;
        font-size: 1.2em;
        letter-spacing: 2px;
        margin-bottom: 2em;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #8b4557 0%, #6b2142 100%);
        color: white;
        font-size: 1.2em;
        font-weight: 600;
        padding: 20px;
        border-radius: 15px;
        border: none;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(139, 69, 87, 0.6);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, rgba(139, 69, 87, 0.3) 0%, rgba(107, 33, 66, 0.3) 100%);
        border: 2px solid rgba(212, 165, 116, 0.4);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    
    .info-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #d4a574;
        border-radius: 20px;
        padding: 30px;
        margin-top: 40px;
        color: #c4b5a0;
    }
    
    .stNumberInput>div>div>input {
        background: rgba(255, 255, 255, 0.08);
        border: 2px solid rgba(212, 165, 116, 0.3);
        border-radius: 12px;
        color: white;
    }
    
    label {
        color: #d4a574 !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('xgb_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, scaler = load_model()

# Header
st.markdown("<h1>🍷 Vinología</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Wine Quality Analysis</p>", unsafe_allow_html=True)

# Main content
if model is not None and scaler is not None:
    # Create columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fixed_acidity = st.number_input(
            "Fixed Acidity",
            min_value=0.0,
            max_value=20.0,
            value=7.0,
            step=0.1,
            help="Typical range: 6.0 - 10.0"
        )
        
        volatile_acidity = st.number_input(
            "Volatile Acidity",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.01,
            help="Typical range: 0.2 - 0.8"
        )
        
        citric_acid = st.number_input(
            "Citric Acid",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            help="Typical range: 0.0 - 0.6"
        )
        
        residual_sugar = st.number_input(
            "Residual Sugar",
            min_value=0.0,
            max_value=20.0,
            value=2.5,
            step=0.1,
            help="Typical range: 1.0 - 5.0"
        )
    
    with col2:
        chlorides = st.number_input(
            "Chlorides",
            min_value=0.0,
            max_value=1.0,
            value=0.06,
            step=0.001,
            help="Typical range: 0.03 - 0.10"
        )
        
        free_sulfur_dioxide = st.number_input(
            "Free Sulfur Dioxide",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
            help="Typical range: 10 - 40"
        )
        
        total_sulfur_dioxide = st.number_input(
            "Total Sulfur Dioxide",
            min_value=0.0,
            max_value=300.0,
            value=100.0,
            step=1.0,
            help="Typical range: 50 - 150"
        )
        
        density = st.number_input(
            "Density",
            min_value=0.98,
            max_value=1.01,
            value=0.995,
            step=0.0001,
            format="%.4f",
            help="Typical range: 0.990 - 1.000"
        )
    
    with col3:
        pH = st.number_input(
            "pH Level",
            min_value=2.0,
            max_value=5.0,
            value=3.3,
            step=0.01,
            help="Typical range: 3.0 - 3.6"
        )
        
        sulphates = st.number_input(
            "Sulphates",
            min_value=0.0,
            max_value=2.0,
            value=0.6,
            step=0.01,
            help="Typical range: 0.4 - 0.8"
        )
        
        alcohol = st.number_input(
            "Alcohol (%)",
            min_value=0.0,
            max_value=20.0,
            value=10.5,
            step=0.1,
            help="Typical range: 9.0 - 14.0"
        )
    
    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🍷 Analyze Wine Quality"):
        try:
            # Prepare features
            features = np.array([[
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                pH, sulphates, alcohol
            ]])
            
            # Scale features
            input_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display result
            if prediction == 1:
                st.markdown("""
                    <div class='prediction-box'>
                        <h2 style='color: #d4a574; margin: 0;'>Wine is: Good Quality 🍷</h2>
                        <p style='color: #c4b5a0; margin-top: 10px;'>This wine exhibits excellent characteristics!</p>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown("""
                    <div class='prediction-box'>
                        <h2 style='color: #d4a574; margin: 0;'>Wine is: Bad Quality 🍷</h2>
                        <p style='color: #c4b5a0; margin-top: 10px;'>This wine may need improvement in some areas.</p>
                    </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    
    # Information section
    st.markdown("""
        <div class='info-box'>
            <h3 style='color: #d4a574; margin-bottom: 15px;'>About Our Analysis</h3>
            <p style='line-height: 1.8;'>
                Our advanced machine learning model analyzes 11 key physicochemical properties 
                of wine to predict its quality. Using XGBoost algorithm trained on thousands of 
                wine samples, we provide accurate quality assessments to help winemakers, 
                distributors, and enthusiasts make informed decisions.
            </p>
        </div>
    """, unsafe_allow_html=True)

else:
    st.error("⚠️ Could not load the model. Please ensure 'xgb_model.pkl' and 'scaler.pkl' are in the same directory.")
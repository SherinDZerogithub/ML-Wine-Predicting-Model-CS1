# wine_quality_predicting_model/models/FC211044_KNN_Wine_Quality/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model package
@st.cache_resource
def load_model():
    """Load the trained KNN model package"""
    try:
        model_path = "wine_quality_binary_knn_k1.joblib"
        model_package = joblib.load(model_path)
        return model_package
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_wine_quality(features, model_package):
    """Predict wine quality using the loaded model"""
    try:
        model = model_package['model']
        scaler = model_package['scaler']
        feature_names = model_package['feature_names']
        
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([features], columns=feature_names)
        
        # Scale features
        scaled_features = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def get_quality_recommendations():
    """Return recommended values for good quality wine"""
    return {
        'alcohol': {'min': 11.0, 'max': 13.5, 'unit': '%', 'importance': 'High'},
        'volatile acidity': {'min': 0.2, 'max': 0.4, 'unit': 'g/dm³', 'importance': 'High'},
        'sulphates': {'min': 0.5, 'max': 0.8, 'unit': 'g/dm³', 'importance': 'High'},
        'total sulfur dioxide': {'min': 30, 'max': 120, 'unit': 'mg/dm³', 'importance': 'Medium'},
        'chlorides': {'min': 0.04, 'max': 0.08, 'unit': 'g/dm³', 'importance': 'Medium'}
    }

def main():
    # Header
    st.title("🍷 Wine Quality Predictor")
    st.markdown("""
    This app predicts whether a wine is **Good** (quality ≥ 6) or **Bad** (quality < 6) 
    based on its chemical properties using a K-Nearest Neighbors machine learning model.
    """)
    
    # Load model
    model_package = load_model()
    
    if model_package is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        return
    
    # Get quality recommendations
    recommendations = get_quality_recommendations()
    
    # Display model info in sidebar
    with st.sidebar:
        st.header("Model Information")
        st.write(f"**Algorithm:** {model_package['model_name']}")
        st.write(f"**Number of Neighbors (k):** {model_package['best_k']}")
        st.write(f"**Accuracy:** {model_package['accuracy']:.2%}")
        st.write(f"**Features Used:** {len(model_package['feature_names'])}")
        st.write(f"**Training Date:** {model_package['training_date']}")
        
        st.markdown("---")
        st.header("Top 5 Features")
        for i, feature in enumerate(model_package['feature_names'], 1):
            st.write(f"{i}. {feature}")
        
        st.markdown("---")
        st.info("""
        **How to use:**
        1. Adjust the chemical properties using sliders
        2. Click 'Predict Wine Quality'
        3. View the prediction and confidence
        """)
    
    # Main content - Input form
    col1, col2 = st.columns([2, 2])
    
    with col1:
        st.header("Wine Chemical Properties")
        st.markdown("Adjust the sliders to match your wine's characteristics:")
        
        # Create input sliders based on the top 5 features
        features = {}
        
        # Alcohol content (most important feature)
        features['alcohol'] = st.slider(
            "Alcohol Content (%)",
            min_value=8.0,
            max_value=15.0,
            value=10.5,
            step=0.1,
            help="Percentage of alcohol in the wine. Higher values generally indicate better quality."
        )
        
        # Volatile acidity
        features['volatile acidity'] = st.slider(
            "Volatile Acidity (g/dm³)",
            min_value=0.1,
            max_value=1.2,
            value=0.5,
            step=0.01,
            help="Amount of acetic acid in wine. Lower values generally indicate better quality."
        )
        
        # Sulphates
        features['sulphates'] = st.slider(
            "Sulphates (g/dm³)",
            min_value=0.3,
            max_value=1.5,
            value=0.6,
            step=0.01,
            help="Wine additive. Higher values generally indicate better quality."
        )
        
        # Total sulfur dioxide
        features['total sulfur dioxide'] = st.slider(
            "Total Sulfur Dioxide (mg/dm³)",
            min_value=10.0,
            max_value=200.0,
            value=100.0,
            step=1.0,
            help="Amount of free and bound forms of SO₂. Moderate values are best."
        )
        
        # Chlorides
        features['chlorides'] = st.slider(
            "Chlorides (g/dm³)",
            min_value=0.01,
            max_value=0.2,
            value=0.08,
            step=0.001,
            help="Amount of salt in the wine. Lower values generally indicate better quality."
        )
        
        # Prediction button
        predict_button = st.button("🍷 Predict Wine Quality", type="primary", use_container_width=True)
    
    with col2:
        st.header("📊 Quick Reference Ranges")
        st.markdown("""
        **Typical Value Ranges:**
        - **Alcohol**: 9-14%
        - **Volatile Acidity**: 0.2-0.6 g/dm³
        - **Sulphates**: 0.4-0.8 g/dm³  
        - **Total SO₂**: 50-150 mg/dm³
        - **Chlorides**: 0.04-0.12 g/dm³
        
        **Quality Indicators:**
        - ✅ **Higher values** = Better quality:
          - Alcohol
          - Sulphates
        
        - ✅ **Lower values** = Better quality:
          - Volatile Acidity
          - Chlorides
          - Total Sulfur Dioxide
        """)
    
  
        # Add current values assessment
        if predict_button:
            st.markdown("---")
            st.subheader("🔍 Your Values Assessment")
            for feature, value in features.items():
                rec = recommendations[feature]
                if feature in ['alcohol', 'sulphates']:
                    # Higher is better
                    if value >= rec['min']:
                        st.write(f"✅ **{feature.title()}**: {value}{rec['unit']}")
                    else:
                        st.write(f"❌ **{feature.title()}**: {value}{rec['unit']} (low)")
                else:
                    # Lower is better
                    if value <= rec['max']:
                        st.write(f"✅ **{feature.title()}**: {value}{rec['unit']}")
                    else:
                        st.write(f"❌ **{feature.title()}**: {value}{rec['unit']} (high)")
    
    # Make prediction when button is clicked
    if predict_button:
        st.markdown("---")
        st.header("Prediction Results")
        
        # Display input values
        with st.expander("View Input Values"):
            input_df = pd.DataFrame([features], index=["Values"])
            st.dataframe(input_df.T.rename(columns={0: "Value"}))
        
        # Make prediction
        prediction, probability = predict_wine_quality(features, model_package)
        
        if prediction is not None:
            # Display results
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                if prediction == 1:
                    st.success("## 🎉 Prediction: GOOD WINE")
                    
                else:
                    st.error("## ⚠️ Prediction: BAD WINE")
            
            with col_result2:
                # Probability gauge
                good_prob = probability[1] * 100
                st.metric(
                    label="Confidence Score",
                    value=f"{good_prob:.1f}%",
                    delta="Good Wine Probability" if prediction == 1 else "Bad Wine Probability"
                )
            
            with col_result3:
                # Probability breakdown
                st.write("**Probability Breakdown:**")
                st.write(f"🍷 Bad Wine: {probability[0]:.2%}")
                st.write(f"🏆 Good Wine: {probability[1]:.2%}")
            
            # Quality Improvement Suggestions
            st.markdown("---")
            st.subheader("💡 Quality Improvement Suggestions")
            
            if prediction == 0:  # Bad wine
                improvement_col1, improvement_col2 = st.columns(2)
                
                with improvement_col1:
                    st.write("**To Improve Your Wine:**")
                    issues = []
                    
                    # Check each feature and provide specific advice
                    if features['alcohol'] < 11.0:
                        issues.append(f"• **Increase alcohol** content (current: {features['alcohol']}%, target: 11.0-13.5%)")
                    
                    if features['volatile acidity'] > 0.4:
                        issues.append(f"• **Reduce volatile acidity** (current: {features['volatile acidity']}g/dm³, target: 0.2-0.4g/dm³)")
                    
                    if features['sulphates'] < 0.5:
                        issues.append(f"• **Increase sulphates** (current: {features['sulphates']}g/dm³, target: 0.5-0.8g/dm³)")
                    
                    if features['total sulfur dioxide'] > 120:
                        issues.append(f"• **Reduce sulfur dioxide** (current: {features['total sulfur dioxide']}mg/dm³, target: 30-120mg/dm³)")
                    
                    if features['chlorides'] > 0.08:
                        issues.append(f"• **Reduce chlorides** (current: {features['chlorides']}g/dm³, target: 0.04-0.08g/dm³)")
                    
                    if issues:
                        for issue in issues:
                            st.write(issue)
                    else:
                        st.write("• All parameters are within good ranges. Consider other quality factors.")
                
                with improvement_col2:
                    st.write("**Winemaking Tips:**")
                    st.write("• **Fermentation control** for alcohol content")
                    st.write("• **Proper sanitation** to reduce volatile acidity")
                    st.write("• **Balanced sulfite addition** for preservation")
                    st.write("• **Quality water sources** to minimize chlorides")
            
            # Interpretation
            st.markdown("---")
            st.subheader("📊 Interpretation")
            
            if prediction == 1:
                st.success("""
                ✅ **This wine is predicted to be GOOD quality (≥6/10)**
                - Likely to have pleasant characteristics
                - Good balance of chemical properties
                - Higher probability of consumer satisfaction
                - Well-balanced chemical composition
                """)
            else:
                st.warning("""
                ⚠️ **This wine is predicted to be BAD quality (<6/10)**
                - May have unbalanced chemical properties
                - Could exhibit undesirable characteristics
                - Lower probability of consumer satisfaction
                - Consider adjusting the chemical parameters
                """)
            
            # Feature importance visualization
            st.markdown("---")
            st.subheader("🔬 Feature Impact Analysis")
            
            # Create simple bar chart showing feature importance
            importance_data = {
                'Feature': model_package['feature_names'],
                'Importance': [0.14, 0.11, 0.12, 0.11, 0.06]  # From your permutation importance
            }
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
            bars = ax.barh(importance_data['Feature'], importance_data['Importance'], 
                          color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Importance Score')
            ax.set_title('Feature Importance in Prediction')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, importance in zip(bars, importance_data['Importance']):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                       f'{importance:.3f}', ha='left', va='center', fontweight='bold')
            
            st.pyplot(fig)
            
            # Disclaimer
            st.markdown("---")
            st.info("""
            **📝 Disclaimer:** This prediction is based on chemical properties and machine learning analysis. 
            Actual wine quality can be subjective and influenced by factors beyond chemical composition.
            **Model accuracy: 78.12%**
            
            **Note:** The model uses the top 5 most important features identified through feature importance analysis.
            """)

if __name__ == "__main__":
    main()
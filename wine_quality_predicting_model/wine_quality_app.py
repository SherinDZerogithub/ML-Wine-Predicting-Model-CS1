import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #8B0000;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-good {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .prediction-bad {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .prediction-numeric {
        background-color: #e7f3ff;
        border: 2px solid #b8d4ff;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .model-card {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #8B0000;
    }
    .quality-score {
        font-size: 2.5rem;
        font-weight: bold;
        color: #8B0000;
    }
    .threshold-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class WineQualityPredictor:
    def __init__(self, model_path, model_type):
        self.model_type = model_type
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model and its components"""
        try:
            model_data = joblib.load(model_path)
            
            if self.model_type == "SVM":
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.threshold = model_data.get('optimal_threshold', 0.5)
                self.accuracy = "N/A"
                accuracy_value = model_data.get('accuracy')
                if accuracy_value is not None:
                    if isinstance(accuracy_value, (int, float)):
                        self.accuracy = f"{accuracy_value:.2%}" 
                    else:
                        self.accuracy = accuracy_value
                else:
                    self.accuracy = "0.7647"
                self.is_classification = True
                    
            elif self.model_type == "Logistic Regression":
                self.model = model_data['pipeline']
                self.scaler = self.model.named_steps['scaler']
                self.feature_names = model_data['feature_names']
                # Use the stored threshold from the model training
                self.threshold = model_data.get('threshold', 0.393)  # Default to 0.393 from your training
                self.accuracy = model_data.get('roc_data', {}).get('auc', 'N/A')
                self.is_classification = True
                
            elif self.model_type == "KNN":
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.best_k = model_data.get('best_k', 'N/A')
                accuracy_value = model_data.get('accuracy')
                if accuracy_value is not None:
                    if isinstance(accuracy_value, (int, float)):
                        self.accuracy = f"{accuracy_value:.2%}" 
                    else:
                        self.accuracy = accuracy_value
                else:
                    self.accuracy = "0.7812"
                self.is_classification = True
            
            elif self.model_type == "Random Forest":
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.accuracy = model_data.get('accuracy', 'N/A')
                # Random Forest is regression - predict numeric quality scores
                self.is_classification = False
                
            elif self.model_type == "XG-Boost":
                self.model = model_data['pipeline']
                if hasattr(self.model, 'named_steps') and 'scaler' in self.model.named_steps:
                    self.scaler = self.model.named_steps['scaler']
                else:
                    self.scaler = model_data.get('scaler', None)
                self.feature_names = model_data['feature_names']
                self.accuracy = model_data.get('accuracy', 'N/A')
                self.threshold = model_data.get('threshold', 0.5)
                self.is_classification = True
                
            st.success(f"✅ {self.model_type} model loaded successfully!")
            st.sidebar.info(f"📊 Model threshold: {getattr(self, 'threshold', 0.5):.3f}")
                
        except Exception as e: 
            st.error(f"❌ Error loading model: {str(e)}")
            raise e
    
    def predict_quality(self, feature_values):
        """Predict wine quality based on input features"""
        try:
            # Convert to numpy array and reshape
            features_array = np.array(feature_values).reshape(1, -1)
            
            # Create DataFrame with feature names to avoid warnings
            features_df = pd.DataFrame(features_array, columns=self.feature_names)
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Make prediction based on model type
            if not self.is_classification:
                # Regression model (Random Forest) - predict numeric quality
                quality_score = float(self.model.predict(features_scaled)[0])
                # Convert numeric score to binary classification for consistency
                binary_prediction = 1 if quality_score >= 6 else 0
                # Create artificial probabilities based on distance from threshold
                if quality_score >= 6:
                    prob_good = min(0.5 + (quality_score - 6) / 4, 0.95)  # Scale to reasonable probability
                    probabilities = np.array([1 - prob_good, prob_good])
                else:
                    prob_bad = min(0.5 + (6 - quality_score) / 4, 0.95)
                    probabilities = np.array([prob_bad, 1 - prob_bad])
                
                return binary_prediction, probabilities, quality_score
            else:
                # Classification models
                if hasattr(self.model, 'predict_proba'):
                    # Get probabilities for both classes
                    probabilities = self.model.predict_proba(features_scaled)[0]
                    
                    # DEBUG: Show raw probabilities and classes
                    debug_info = {
                        'raw_probabilities': probabilities,
                        'model_classes': getattr(self.model, 'classes_', None),
                        'threshold': getattr(self, 'threshold', 0.5)
                    }
                    
                    # Handle different class label ordering - CRITICAL FIX
                    if hasattr(self.model, 'classes_'):
                        classes = list(self.model.classes_)
                        # Map probabilities to correct classes
                        if len(classes) == 2:
                            class_0_idx = list(classes).index(0)
                            class_1_idx = list(classes).index(1)
                            prob_bad = probabilities[class_0_idx]
                            prob_good = probabilities[class_1_idx]
                        else:
                            # Fallback if unexpected number of classes
                            prob_bad = probabilities[0]
                            prob_good = probabilities[1]
                    else:
                        # Default ordering if no classes attribute
                        prob_bad = probabilities[0]
                        prob_good = probabilities[1]
                    
                    # Apply threshold for prediction - use the tuned threshold
                    threshold = getattr(self, 'threshold', 0.5)
                    prediction = 1 if prob_good >= threshold else 0
                    
                    # Return probabilities in consistent order: [bad, good]
                    probabilities = np.array([prob_bad, prob_good])
                    
                    return prediction, probabilities, debug_info
                    
                else:
                    # For models without probability (use predict)
                    raw_prediction = self.model.predict(features_scaled)[0]
                    # Map raw prediction to our binary classes
                    if raw_prediction >= 6:  # If it's predicting quality scores
                        prediction = 1
                        probabilities = np.array([0.2, 0.8])
                    else:
                        prediction = 0
                        probabilities = np.array([0.8, 0.2])
                    
                    return prediction, probabilities, None
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            return None, None, None
    
    def _is_good_value(self, feature, value):
        """Helper function to determine if feature value is in good range"""
        good_ranges = {
            'alcohol': (11.5, 14.0),
            'volatile acidity': (0.1, 0.5),
            'sulphates': (0.6, 1.0),
            'chlorides': (0.01, 0.08),
            'total sulfur dioxide': (20, 50),
            'citric acid': (0.2, 0.5),
            'fixed acidity': (6.5, 8.0),
            'density': (0.992, 0.997),
            'free sulfur dioxide': (10, 30),
            'residual sugar': (1.5, 3.0),
            'ph': (3.2, 3.4)
        }
        
        feature_lower = feature.lower()
        for key, (low, high) in good_ranges.items():
            if key in feature_lower:
                return low <= value <= high
        return True

def load_available_models():
    """Load all available trained models"""
    models_dir = Path("models")
    
    available_models = {
        "SVM": {
            "path": models_dir / "FC211038-SVM-models" / "wine_quality_svm_model.joblib",
            "description": "Support Vector Machine - good for complex boundaries",
            "type": "SVM"
        },
        "Logistic Regression": {
            "path": models_dir / "FC211032_VidusahanPerera" / "logreg_wine_quality_pipeline.joblib",
            "description": "Linear model with probability outputs - highly interpretable",
            "type": "Logistic Regression"
        },
        "KNN": {
            "path": models_dir / "FC211044_KNN_Wine_Quality" / "wine_quality_binary_knn_k1.joblib",
            "description": "K-Nearest Neighbors - simple and interpretable",
            "type": "KNN"
        },
        "Random Forest": {
            "path": models_dir / "FC211008-classification-tree-models" / "wine_quality_best_model_random_forest.joblib",
            "description": "Ensemble of decision trees - predicts numeric quality scores (3-8)",
            "type": "Random Forest"
        },
        "XG-Boost": {
            "path": models_dir / "FC211017_XGBoost_model" / "wine-quality-model-xgboost.joblib",
            "description": "Extreme Gradient Boosting - powerful and efficient",
            "type": "XG-Boost"
        }
    }
    
    # Check which models actually exist
    valid_models = {}
    for name, info in available_models.items():
        if info["path"].exists():
            valid_models[name] = info
        else:
            st.warning(f"⚠️ Model not found: {info['path']}")
    
    return valid_models

def display_classification_result(prediction, probabilities, features, predictor, debug_info=None):
    """Display results for classification models"""
    st.markdown("---")
    st.subheader("Prediction Results")
    
    # Show model threshold information
    threshold = getattr(predictor, 'threshold', 0.5)
    prob_bad, prob_good = probabilities
    
    # Check if prediction matches threshold logic
    threshold_correct = (prediction == 1 and prob_good >= threshold) or (prediction == 0 and prob_good < threshold)
    
    if not threshold_correct:
        st.error("🚨 PREDICTION LOGIC ERROR: Prediction doesn't match threshold!")
        st.write(f"Prediction: {prediction}, Good prob: {prob_good:.3f}, Threshold: {threshold:.3f}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if prediction == 1:  # Good quality (>=6)
            st.markdown("""
            <div class="prediction-good">
                <h2>🍷 GOOD QUALITY</h2>
                <p>This wine is predicted to be of good quality!</p>
                <p><strong>Quality >= 6</strong></p>
            </div>
            """, unsafe_allow_html=True)
            confidence = prob_good * 100
        else:  # Bad quality (<6)
            st.markdown("""
            <div class="prediction-bad">
                <h2>❌ BAD QUALITY</h2>
                <p>This wine is predicted to be of poor quality</p>
                <p><strong>Quality < 6</strong></p>
            </div>
            """, unsafe_allow_html=True)
            confidence = prob_bad * 100
        
        st.metric("Prediction Confidence", f"{confidence:.1f}%")
        
        # Show threshold comparison
        if prob_good >= threshold:
            st.success(f"✅ Good quality probability ({prob_good:.1%}) ≥ threshold ({threshold:.1%})")
        else:
            st.error(f"❌ Good quality probability ({prob_good:.1%}) < threshold ({threshold:.1%})")
        
        st.info(f"""
        **Probability Breakdown:**
        - Good Quality: {prob_good:.1%}
        - Bad Quality: {prob_bad:.1%}
        """)

    with col2:
        # Create probability chart
        prob_bad, prob_good = probabilities
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=['Bad Quality', 'Good Quality'],
            x=[prob_bad, prob_good],
            orientation='h',
            marker_color=['#dc3545', '#28a745'],
            text=[f'{prob_bad:.1%}', f'{prob_good:.1%}'],
            textposition='auto',
        ))
        
        # Add threshold line
        fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                     annotation_text=f"Threshold: {threshold:.3f}",
                     annotation_position="top right")
        
        fig.update_layout(
            title="Prediction Probabilities",
            xaxis_title="Probability",
            yaxis_title="Quality Category",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

def display_regression_result(quality_score, probabilities, features, predictor):
    """Display results for regression models (Random Forest)"""
    st.markdown("---")
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display numeric quality score
        st.markdown(f"""
        <div class="prediction-numeric">
            <h2>📊 QUALITY SCORE</h2>
            <div class="quality-score">{quality_score:.1f}/8</div>
            <p>Predicted Wine Quality Rating</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quality interpretation
        if quality_score >= 7:
            quality_label = "Excellent"
            quality_color = "#28a745"
        elif quality_score >= 6:
            quality_label = "Good"
            quality_color = "#20c997"
        elif quality_score >= 5:
            quality_label = "Average"
            quality_color = "#ffc107"
        else:
            quality_label = "Poor"
            quality_color = "#dc3545"
        
        st.metric("Quality Rating", quality_label)
        
        # Binary classification based on score
        binary_prediction = 1 if quality_score >= 6 else 0
        prob_bad, prob_good = probabilities
        st.info(f"""
        **Binary Classification:**
        - {'🍷 GOOD (≥6)' if binary_prediction == 1 else '❌ BAD (<6)'}
        - Confidence: {prob_good if binary_prediction == 1 else prob_bad:.1%}
        """)

    with col2:
        # Create quality gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = quality_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Wine Quality Score"},
            gauge = {
                'axis': {'range': [3, 8]},
                'bar': {'color': "#8B0000"},
                'steps': [
                    {'range': [3, 5], 'color': "lightgray"},
                    {'range': [5, 6], 'color': "lightyellow"},
                    {'range': [6, 8], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 6
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Probability breakdown
        prob_bad, prob_good = probabilities
        st.metric("Probability of Good Quality", f"{prob_good:.1%}")

def display_feature_analysis(features, predictor):
    """Display feature analysis for both model types"""
    st.subheader("🔍 Input Features Analysis")
    
    # Count problematic features
    problematic_features = []
    for feature, value in features.items():
        if not predictor._is_good_value(feature, value):
            problematic_features.append(feature)
    
    if problematic_features:
        st.warning(f"⚠️ {len(problematic_features)} features are outside ideal ranges for good quality wine")
    
    # Create columns for features
    feature_cols = st.columns(3)
    
    # Define ideal ranges for each feature
    ideal_ranges = {
        'alcohol': {'good': '11.5-14.0%', 'bad': '<10.0% or >14.0%'},
        'volatile acidity': {'good': '0.1-0.5', 'bad': '>0.8'},
        'sulphates': {'good': '0.6-1.0', 'bad': '<0.5'},
        'total sulfur dioxide': {'good': '20-50', 'bad': '>80'},
        'chlorides': {'good': '0.01-0.08', 'bad': '>0.1'},
        'fixed acidity': {'good': '6.5-8.0', 'bad': '<4.0 or >9.0'},
        'citric acid': {'good': '0.2-0.5', 'bad': '<0.1'},
        'density': {'good': '0.992-0.997', 'bad': '<0.990 or >1.000'},
        'free sulfur dioxide': {'good': '10-30', 'bad': '<5 or >50'},
        'residual sugar': {'good': '1.5-3.0', 'bad': '>4.0'},
        'ph': {'good': '3.2-3.4', 'bad': '<3.0 or >3.6'}
    }
    
    for i, feature in enumerate(features):
        with feature_cols[i % 3]:
            value = features[feature]
            is_good = predictor._is_good_value(feature, value)
            emoji = "✅" if is_good else "⚠️"
            
            st.metric(f"{emoji} {feature.title()}", f"{value:.3f}")
            
            # Show ideal range if available
            feature_lower = feature.lower()
            range_found = False
            for key, ranges in ideal_ranges.items():
                if key in feature_lower:
                    st.caption(f"Good range: {ranges['good']}")
                    if not is_good:
                        st.caption(f"❌ Outside ideal range")
                    range_found = True
                    break
            
            if not range_found:
                st.caption("No specific range defined")

def main():
    # Header
    st.markdown('<h1 class="main-header">🍷 Wine Quality Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar for model selection
    st.sidebar.title("Model Configuration")
    
    # Load available models
    available_models = load_available_models()
    
    if not available_models:
        st.error("❌ No trained models found. Please check your model paths.")
        return
    
    # Model selection
    selected_model_name = st.sidebar.selectbox(
        "Choose Prediction Model:",
        options=list(available_models.keys()),
        help="Select which machine learning model to use for prediction"
    )
    
    # Load selected model
    model_info = available_models[selected_model_name]
    
    try:
        predictor = WineQualityPredictor(model_info["path"], model_info["type"])
        
        # Display model information
        st.sidebar.markdown("---")
        st.sidebar.subheader("Model Information")
        
        model_type_info = "Classification" if predictor.is_classification else "Regression"
        
        st.sidebar.markdown(f"""
        <div class="model-card">
            <strong>Model Type:</strong> {selected_model_name}<br>
            <strong>Prediction Type:</strong> {model_type_info}<br>
            <strong>Description:</strong> {model_info['description']}<br>
            <strong>Accuracy:</strong> {predictor.accuracy}<br>
            <strong>Features Used:</strong> {len(predictor.feature_names)}<br>
            {f'<strong>Best K:</strong> {predictor.best_k}' if hasattr(predictor, 'best_k') else ''}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return
    
    # Main content area
    st.header("Wine Characteristics Input")
    
    # Create feature input form with more reasonable defaults
    col1, col2 = st.columns(2)
    
    features = {}
    
    with col1:
        st.subheader("Basic Properties")
        features['fixed acidity'] = st.slider("Fixed Acidity", 4.0, 16.0, 7.4, 0.1,
                                            help="Ideal range: 6.5-8.0 g/dm³")
        features['volatile acidity'] = st.slider("Volatile Acidity", 0.1, 1.6, 0.5, 0.01,
                                               help="Ideal range: 0.1-0.5 g/dm³ (lower is better)")
        features['citric acid'] = st.slider("Citric Acid", 0.0, 1.0, 0.25, 0.01,
                                          help="Ideal range: 0.2-0.5 g/dm³")
        features['residual sugar'] = st.slider("Residual Sugar", 0.9, 16.0, 2.5, 0.1,
                                             help="Ideal range: 1.5-3.0 g/dm³")
        features['chlorides'] = st.slider("Chlorides", 0.01, 0.6, 0.08, 0.001,
                                        help="Ideal range: 0.01-0.08 g/dm³ (lower is better)")
        
    with col2:
        st.subheader("Chemical Properties")
        features['free sulfur dioxide'] = st.slider("Free Sulfur Dioxide", 1.0, 72.0, 15.0, 1.0,
                                                  help="Ideal range: 10-30 mg/dm³")
        features['total sulfur dioxide'] = st.slider("Total Sulfur Dioxide", 6.0, 289.0, 45.0, 1.0,
                                                   help="Ideal range: 20-50 mg/dm³")
        features['density'] = st.slider("Density", 0.990, 1.004, 0.996, 0.001,
                                      help="Ideal range: 0.992-0.997 g/cm³")
        features['pH'] = st.slider("pH", 2.7, 4.0, 3.3, 0.1,
                                 help="Ideal range: 3.2-3.4")
        features['sulphates'] = st.slider("Sulphates", 0.3, 2.0, 0.65, 0.01,
                                        help="Ideal range: 0.6-1.0 g/dm³")
        features['alcohol'] = st.slider("Alcohol (%)", 8.0, 15.0, 11.0, 0.1,
                                      help="Ideal range: 11.5-14.0% (higher is generally better)")
    
    # Add interpretation help
    with st.expander("💡 How to interpret these values"):
        st.markdown("""
        **Feature Guidelines for Good Quality Wine:**
        - **Alcohol**: Higher alcohol content (11.5-14%) generally indicates better quality.
        - **Volatile Acidity**: Lower values (0.1-0.5) are better – high values can indicate a vinegar taste.
        - **Sulphates**: Moderate to high levels (0.6–1.0) help preservation and quality.
        - **Chlorides**: Lower salt content (0.01-0.08) indicates better quality.
        - **Total Sulfur Dioxide**: Moderate levels (20–50) are ideal for preservation.
        - **Citric Acid**: Moderate levels (0.2–0.5) add freshness.
        - **Fixed Acidity**: Balanced levels (6.5–8.0) contribute to structure.
        - **Free Sulfur Dioxide**: Moderate levels (10-30) prevent oxidation.
        - **Residual Sugar**: Balanced levels (1.5-3.0) for taste.
        - **pH**: Balanced acidity (3.2-3.4) for stability.
        - **Density**: Proper range (0.992-0.997) indicates correct composition.
        """)

    # Prediction button
    submitted = st.button("🍷 Predict Wine Quality", type="primary", use_container_width=True)
    
    # Save current feature values to session state
    st.session_state.features = features.copy()
    
    # Prediction logic
    if submitted:
        with st.spinner("Analyzing wine characteristics..."):
            feature_list = [features[feature] for feature in predictor.feature_names]
            prediction, probabilities, debug_info = predictor.predict_quality(feature_list)
        
        if prediction is not None and probabilities is not None:
            # Debug information (can be removed in production)
            with st.sidebar:
                st.markdown("---")
                st.subheader("Debug Info")
                st.write(f"Prediction: {prediction}")
                st.write(f"Probabilities: {probabilities}")
                
                # Handle debug info based on model type
                if not predictor.is_classification:
                    # For regression models, debug_info is the quality score
                    st.write(f"Quality Score: {debug_info:.3f}")
                else:
                    # For classification models, debug_info is a dictionary
                    if debug_info and isinstance(debug_info, dict):
                        st.write(f"Raw probabilities: {debug_info.get('raw_probabilities', 'N/A')}")
                        st.write(f"Model classes: {debug_info.get('model_classes', 'N/A')}")
                    st.write(f"Model threshold: {getattr(predictor, 'threshold', 0.5):.3f}")
            
            # Display appropriate result based on model type
            if not predictor.is_classification:
                # Random Forest - regression model
                display_regression_result(debug_info, probabilities, features, predictor)
            else:
                # Classification models
                display_classification_result(prediction, probabilities, features, predictor, debug_info)
            
            # Display feature analysis for both model types
            display_feature_analysis(features, predictor)
        else:
            st.error("❌ Prediction failed. Please check the model and input values.")

if __name__ == "__main__":
    main()
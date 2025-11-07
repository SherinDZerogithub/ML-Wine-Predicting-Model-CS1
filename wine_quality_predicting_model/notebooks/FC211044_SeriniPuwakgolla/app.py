import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Add the root directory to Python path to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Page configuration
st.set_page_config(
    page_title="🍷 Wine Quality Predictor",
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
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #8B0000;
        margin-bottom: 1rem;
    }
    .prediction-good {
        background-color: #d4edda;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #28a745;
        text-align: center;
    }
    .prediction-bad {
        background-color: #f8d7da;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #dc3545;
        text-align: center;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class WineQualityPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            current_dir = os.path.dirname(__file__)
            # Updated model path for k=1
            model_path = os.path.join(current_dir, '../../../models/FC211044_KNN_Wine_Quality/wine_quality_binary_knn_k1.joblib')
            
            if not os.path.exists(model_path):
                model_path = 'wine_quality_predicting_model/models/FC211044_KNN_Wine_Quality/wine_quality_binary_knn_k1.joblib'
            
            st.info(f"Looking for model at: {model_path}")
            
            if os.path.exists(model_path):
                model_package = joblib.load(model_path)
                self.model = model_package['model']
                self.scaler = model_package['scaler']
                self.feature_names = model_package['feature_names']
                self.model_info = model_package
                st.success("✅ Model loaded successfully!")
                st.info(f"Model uses {len(self.feature_names)} features: {', '.join(self.feature_names)}")
                
                # Updated label mapping for new threshold (>=6)
                st.info(f"Model classes: {self.model.classes_} (0=Bad, 1=Good)")
                st.info("📊 Quality Threshold: Good (quality >= 6), Bad (quality < 6)")
                
                return True
            else:
                st.error(f"❌ Model file not found at: {model_path}")
                # List available files for debugging
                model_dir = os.path.dirname(model_path)
                if os.path.exists(model_dir):
                    st.error(f"Available files in directory: {os.listdir(model_dir)}")
                return False
                
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def predict_quality(self, features):
        """Predict wine quality"""
        try:
            input_df = pd.DataFrame([features], columns=self.feature_names)
            scaled_features = self.scaler.transform(input_df)
            prediction = self.model.predict(scaled_features)[0]
            probability = self.model.predict_proba(scaled_features)[0]
            
            return prediction, probability
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None
    
    def _is_good_value(self, feature, value):
        """Helper function to determine if a feature value is in the 'good' range"""
        good_ranges = {
            'alcohol': (11.5, 15.0),
            'volatile acidity': (0.1, 0.5),
            'sulphates': (0.6, 2.0),
            'chlorides': (0.01, 0.08),
            'total sulfur dioxide': (20, 50),
            'citric acid': (0.2, 0.5)
        }
        
        if feature in good_ranges:
            min_val, max_val = good_ranges[feature]
            return min_val <= value <= max_val
        return True  # Default to True for features without defined ranges

def main():
    predictor = WineQualityPredictor()
    
    st.markdown('<h1 class="main-header">🍷 Red Wine Quality Predictor</h1>', unsafe_allow_html=True)
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2405/2405477.png", width=100)
    st.sidebar.title("Navigation")
    
    if predictor.model:
        app_mode = st.sidebar.selectbox(
            "Choose App Mode",
            ["Home", "Quality Prediction", "Data Analysis", "Model Info"]
        )
    else:
        app_mode = st.sidebar.selectbox(
            "Choose App Mode",
            ["Home", "Data Analysis"]
        )
        st.warning("⚠️ Model not loaded. Some features are disabled.")
    
    if app_mode == "Home":
        show_home_page(predictor)
    elif app_mode == "Quality Prediction" and predictor.model:
        show_prediction_page(predictor)
    elif app_mode == "Data Analysis":
        show_analysis_page(predictor)
    elif app_mode == "Model Info" and predictor.model:
        show_model_info_page(predictor)

def show_home_page(predictor):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ## Welcome to the Red Wine Quality Predictor!
        This application uses machine learning to predict whether a red wine is 
        **Good** or **Bad** based on its chemical properties.
        
        **🎯 Quality Threshold:** Wines with quality >= 6 are considered **Good**, others are **Bad**.
        """)
        if predictor.model:
            st.markdown("### Key Features Used:")
            for i, feature in enumerate(predictor.feature_names):
                st.markdown(f"""
                <div class="feature-card">
                    <h4>{"⭐" * min(i+1, 3)} {feature.title()}</h4>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("""
            ### ⚠️ Model Not Loaded
            Please ensure the model file exists at:
            `wine_quality_predicting_model/models/FC211044_KNN_Binary_Wine_Quality/wine_quality_binary_knn_k1.joblib`
            """)
    with col2:
        st.image("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=400", 
                 caption="Wine Quality Analysis")
        st.markdown("### Quick Stats")
        if predictor.model:
            st.metric("Model Accuracy", f"{predictor.model_info['accuracy']:.1%}")
            st.metric("Best K Value", predictor.model_info['best_k'])
            st.metric("Features Used", len(predictor.feature_names))
        else:
            st.metric("Status", "Model Missing")
            st.metric("Action Required", "Check Path")

def show_prediction_page(predictor):
    st.header("Wine Quality Prediction")
    st.info(f"🎯 This model uses {len(predictor.feature_names)} features: **{', '.join(predictor.feature_names)}**")
    st.info("📝 **Label Mapping:** 0 = Bad (Quality < 6), 1 = Good (Quality >= 6)")

    # Example buttons (outside the form)
    st.markdown("---")
    example_col1, example_col2, example_col3 = st.columns(3)

    with example_col1:
        if st.button("🍷 Example: Good Wine", use_container_width=True, key="good_example"):
            st.session_state.example_type = "good"

    with example_col2:
        if st.button("❌ Example: Bad Wine", use_container_width=True, key="bad_example"):
            st.session_state.example_type = "bad"

    with example_col3:
        if st.button("🔄 Reset Values", use_container_width=True, key="reset_example"):
            st.session_state.example_type = "reset"

    # Initialize session state for features if not exists
    if 'features' not in st.session_state:
        st.session_state.features = {}

    # Form for input features
    with st.form("wine_features_form", clear_on_submit=False):
        st.subheader("Enter Wine Characteristics")
        col1, col2 = st.columns(2)
        
        # Initialize features dictionary
        features = {}

        with col1:
            if 'alcohol' in predictor.feature_names:
                default_val = st.session_state.features.get('alcohol', 10.5)
                features['alcohol'] = st.slider("Alcohol (%)", 8.0, 15.0, default_val, 0.1,
                                              help="Higher alcohol content generally indicates better quality")
            
            if 'volatile acidity' in predictor.feature_names:
                default_val = st.session_state.features.get('volatile acidity', 0.5)
                features['volatile acidity'] = st.slider("Volatile Acidity", 0.1, 1.5, default_val, 0.01,
                                                       help="Lower volatile acidity is better for wine quality")
            
            if 'sulphates' in predictor.feature_names:
                default_val = st.session_state.features.get('sulphates', 0.6)
                features['sulphates'] = st.slider("Sulphates", 0.3, 2.0, default_val, 0.01,
                                                help="Higher sulphates can indicate better quality and preservation")
            
            if 'total sulfur dioxide' in predictor.feature_names:
                default_val = st.session_state.features.get('total sulfur dioxide', 35.0)
                features['total sulfur dioxide'] = st.slider("Total Sulfur Dioxide", 10.0, 200.0, default_val, 1.0,
                                                           help="Moderate levels are best; too high can affect taste")

        with col2:
            if 'chlorides' in predictor.feature_names:
                default_val = st.session_state.features.get('chlorides', 0.08)
                features['chlorides'] = st.slider("Chlorides", 0.01, 0.2, default_val, 0.001,
                                                help="Lower chloride levels generally indicate better quality")
            
            if 'fixed acidity' in predictor.feature_names:
                default_val = st.session_state.features.get('fixed acidity', 7.0)
                features['fixed acidity'] = st.slider("Fixed Acidity", 4.0, 16.0, default_val, 0.1,
                                                    help="Moderate fixed acidity contributes to wine structure")
            
            if 'citric acid' in predictor.feature_names:
                default_val = st.session_state.features.get('citric acid', 0.3)
                features['citric acid'] = st.slider("Citric Acid", 0.0, 1.0, default_val, 0.01,
                                                  help="Adds freshness to wine; moderate levels are best")
            
            if 'density' in predictor.feature_names:
                default_val = st.session_state.features.get('density', 0.996)
                features['density'] = st.slider("Density", 0.990, 1.005, default_val, 0.001,
                                              help="Related to alcohol and sugar content")

        submitted = st.form_submit_button("🎯 Predict Quality", use_container_width=True)

    # Apply example values
    if 'example_type' in st.session_state:
        if st.session_state.example_type == "good":
            # Good wine characteristics (quality >= 6)
            if 'alcohol' in predictor.feature_names: features['alcohol'] = 12.5
            if 'volatile acidity' in predictor.feature_names: features['volatile acidity'] = 0.3
            if 'sulphates' in predictor.feature_names: features['sulphates'] = 0.8
            if 'total sulfur dioxide' in predictor.feature_names: features['total sulfur dioxide'] = 30.0
            if 'chlorides' in predictor.feature_names: features['chlorides'] = 0.05
            if 'fixed acidity' in predictor.feature_names: features['fixed acidity'] = 7.5
            if 'citric acid' in predictor.feature_names: features['citric acid'] = 0.4
            if 'density' in predictor.feature_names: features['density'] = 0.995
            st.success("✅ Good wine example values loaded! (High alcohol, low volatile acidity)")
            
        elif st.session_state.example_type == "bad":
            # Bad wine characteristics (quality < 6)
            if 'alcohol' in predictor.feature_names: features['alcohol'] = 9.0
            if 'volatile acidity' in predictor.feature_names: features['volatile acidity'] = 1.2
            if 'sulphates' in predictor.feature_names: features['sulphates'] = 0.4
            if 'total sulfur dioxide' in predictor.feature_names: features['total sulfur dioxide'] = 120.0
            if 'chlorides' in predictor.feature_names: features['chlorides'] = 0.15
            if 'fixed acidity' in predictor.feature_names: features['fixed acidity'] = 6.5
            if 'citric acid' in predictor.feature_names: features['citric acid'] = 0.1
            if 'density' in predictor.feature_names: features['density'] = 0.998
            st.warning("❌ Bad wine example values loaded! (Low alcohol, high volatile acidity)")
        
        elif st.session_state.example_type == "reset":
            # Reset to default values
            for feature in features:
                features[feature] = st.session_state.features.get(feature, features[feature])
            st.info("🔄 Values reset to defaults")

    # Save current feature values to session state
    st.session_state.features = features.copy()

    # Prediction logic
    if submitted:
        with st.spinner("Analyzing wine characteristics..."):
            feature_list = [features[feature] for feature in predictor.feature_names]
            prediction, probabilities = predictor.predict_quality(feature_list)

        if prediction is not None:
            display_prediction_result(prediction, probabilities, features, predictor)
            
def display_prediction_result(prediction, probabilities, features, predictor):
    st.markdown("---")
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Updated label mapping for new threshold (>=6)
        if prediction == 1:  # 1 = Good quality (>=6)
            st.markdown("""
            <div class="prediction-good">
                <h2>🍷 GOOD QUALITY</h2>
                <p>This wine is predicted to be of good quality!</p>
                <p><strong>Quality >= 6</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:  # 0 = Bad quality (<6)
            st.markdown("""
            <div class="prediction-bad">
                <h2>❌ BAD QUALITY</h2>
                <p>This wine is predicted to be of poor quality</p>
                <p><strong>Quality < 6</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        confidence = probabilities[prediction] * 100
        st.metric("Prediction Confidence", f"{confidence:.1f}%")
        
        # Show probability breakdown
        st.info(f"""
        **Probability Breakdown:**
        - Good Quality: {probabilities[1]:.1%}
        - Bad Quality: {probabilities[0]:.1%}
        """)

    with col2:
        # Create probability chart with correct labels
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=['Bad Quality', 'Good Quality'],
            x=[probabilities[0], probabilities[1]],
            orientation='h',
            marker_color=['#dc3545', '#28a745'],
            text=[f'{probabilities[0]:.1%}', f'{probabilities[1]:.1%}'],
            textposition='auto',
        ))
        fig.update_layout(
            title="Prediction Probabilities",
            xaxis_title="Probability",
            yaxis_title="Quality Category",
            showlegend=False,
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)

    # Show feature values used with interpretation
    st.subheader("Input Features Analysis")
    
    # Create columns for features
    feature_cols = st.columns(3)
    
    # Define ideal ranges for each feature (for context)
    ideal_ranges = {
        'alcohol': {'good': '>11.5%', 'bad': '<10.0%'},
        'volatile acidity': {'good': '<0.5', 'bad': '>0.8'},
        'sulphates': {'good': '>0.6', 'bad': '<0.5'},
        'total sulfur dioxide': {'good': '20-50', 'bad': '>80'},
        'chlorides': {'good': '<0.08', 'bad': '>0.1'},
        'fixed acidity': {'good': '6.5-8.0', 'bad': 'Extremes'},
        'citric acid': {'good': '0.2-0.5', 'bad': '<0.1'},
        'density': {'good': '0.992-0.997', 'bad': 'Extremes'}
    }
    
    for i, feature in enumerate(features):
        with feature_cols[i % 3]:
            value = features[feature]
            emoji = "✅" if predictor._is_good_value(feature, value) else "⚠️"
            st.metric(f"{emoji} {feature.title()}", f"{value:.3f}")
            
            # Show ideal range if available
            if feature in ideal_ranges:
                st.caption(f"Good: {ideal_ranges[feature]['good']}")

    # Add interpretation help
    with st.expander("💡 How to interpret these values"):
        st.markdown("""
        **Feature Guidelines for Good Quality Wine:**
        - **Alcohol**: Higher alcohol content (11.5-14%) generally indicates better quality
        - **Volatile Acidity**: Lower values (<0.5) are better - high values can indicate vinegar taste
        - **Sulphates**: Moderate to high levels (0.6-1.0) help preservation and quality
        - **Chlorides**: Lower salt content (<0.08) indicates better quality
        - **Total Sulfur Dioxide**: Moderate levels (20-50) are ideal for preservation
        - **Citric Acid**: Moderate levels (0.2-0.5) add freshness
        - **Fixed Acidity**: Balanced levels (6.5-8.0) contribute to structure
        """)

def show_analysis_page(predictor):
    """Display data analysis visualizations"""
    st.header("Wine Quality Dataset Analysis")
    
    # Load dataset with error handling
    try:
        # Try multiple possible paths
        data_paths = [
            os.path.join(os.path.dirname(__file__), '../../../data/raw/winequality-red.csv'),
            'wine_quality_predicting_model/data/raw/winequality-red.csv',
            'data/raw/winequality-red.csv'
        ]
        
        wine_data = None
        used_path = ""
        
        for path in data_paths:
            if os.path.exists(path):
                wine_data = pd.read_csv(path, delimiter=',')
                used_path = path
                break
        
        if wine_data is not None:
            st.success(f"✅ Dataset loaded from: {used_path}")
            # Updated threshold to >=6
            wine_data['quality_label'] = (wine_data['quality'] >= 6).astype(int)
            st.info(f"📊 Dataset: {wine_data.shape[0]} samples, {wine_data.shape[1]-1} features")
            st.info(f"🎯 Quality distribution: {sum(wine_data['quality_label']==1)} Good (>=6), {sum(wine_data['quality_label']==0)} Bad (<6)")
        else:
            st.error("❌ Could not load dataset from any known path.")
            st.info("Please ensure the CSV file exists in one of these locations:")
            for path in data_paths:
                st.write(f" - {path}")
            return
            
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return
    
    # Analysis options
    analysis_type = st.selectbox(
        "Choose Analysis",
        ["Dataset Overview", "Quality Distribution", "Feature Relationships", "Correlation Analysis"]
    )
    
    if analysis_type == "Dataset Overview":
        show_dataset_overview(wine_data)
    elif analysis_type == "Quality Distribution":
        show_quality_distribution(wine_data)
    elif analysis_type == "Feature Relationships":
        show_feature_relationships(wine_data)
    elif analysis_type == "Correlation Analysis":
        show_correlation_analysis(wine_data)

def show_dataset_overview(wine_data):
    """Display dataset overview"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(wine_data))
    with col2:
        st.metric("Features", len(wine_data.columns) - 1)
    with col3:
        good_wines = (wine_data['quality'] >= 6).sum()  # Updated threshold
        st.metric("Good Wines (>=6)", good_wines)
    with col4:
        st.metric("Bad Wines (<6)", len(wine_data) - good_wines)
    
    # Display data sample
    st.subheader("Dataset Sample")
    st.dataframe(wine_data.head(10), use_container_width=True)
    
    # Basic statistics
    st.subheader("Statistical Summary")
    st.dataframe(wine_data.describe(), use_container_width=True)

def show_quality_distribution(wine_data):
    """Display quality distribution visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Original quality distribution
        fig = px.histogram(
            wine_data, x='quality', 
            title="Original Quality Score Distribution",
            color_discrete_sequence=['lightcoral']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Binary classification distribution with updated threshold
        quality_binary = (wine_data['quality'] >= 6).astype(int)  # Updated threshold
        fig = px.pie(
            values=quality_binary.value_counts().values,
            names=['Bad (<6)', 'Good (>=6)'],  # Updated labels
            title="Binary Quality Distribution",
            color_discrete_sequence=['lightcoral', 'lightgreen']
        )
        st.plotly_chart(fig, use_container_width=True)

def show_feature_relationships(wine_data):
    """Display feature relationships"""
    feature_x = st.selectbox("Select X Feature", wine_data.columns[:-1])
    feature_y = st.selectbox("Select Y Feature", wine_data.columns[:-1])
    
    fig = px.scatter(
        wine_data, x=feature_x, y=feature_y, color='quality',
        title=f"{feature_x} vs {feature_y} by Quality",
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_correlation_analysis(wine_data):
    """Display correlation analysis"""
    corr_matrix = wine_data.corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Heatmap",
        color_continuous_scale='RdBu_r',
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_model_info_page(predictor):
    """Display model information"""
    st.header("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Details")
        st.metric("Algorithm", "K-Nearest Neighbors")
        st.metric("Best K Value", predictor.model_info['best_k'])
        st.metric("Accuracy", f"{predictor.model_info['accuracy']:.1%}")
        st.metric("Quality Threshold", "Good (>=6), Bad (<6)")
    
    with col2:
        st.subheader("Feature Information")
        for feature in predictor.feature_names:
            st.write(f"✅ {feature}")
    
    # Model parameters
    st.subheader("Model Parameters")
    st.json(predictor.model_info['parameters'])

if __name__ == "__main__":
    main()
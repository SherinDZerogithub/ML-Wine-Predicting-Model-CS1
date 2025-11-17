import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class WineQualityPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.decision_tree = None
        self.random_forest = None
        self.knn_recommender = None
        self.wine_quality_data = None
        self.red_wine_data = None
        self.feature_columns = None
        self.best_params = None
        self.quality_clusters = None
        self.kmeans_model = None

    def clean_wine_quality_data(self):
        """Clean the wine quality dataset (no feature drop)"""
        if self.wine_quality_data is None:
            st.error("Wine quality data not loaded.")
            return False

        df = self.wine_quality_data.copy()

        # Handle missing values (fill with median)
        missing_before = df.isnull().sum()
        df.fillna(df.median(numeric_only=True), inplace=True)
        missing_after = df.isnull().sum()

        # Remove duplicates
        duplicates_before = df.duplicated().sum()
        df = df.drop_duplicates()
        duplicates_after = df.duplicated().sum()

        # Detect outliers using z-score (only mark, don't drop)
        from scipy.stats import zscore
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        z_scores = np.abs(zscore(df[numeric_cols]))
        outlier_flags = (z_scores > 3).sum(axis=1)
        outlier_count = (outlier_flags > 0).sum()

        self.wine_quality_data = df

        # Show cleaning summary
        st.info("✅ Data Cleaning Summary:")
        if missing_before.sum() > 0:
            st.write("**Missing Values (Before Cleaning):**")
            st.write(missing_before[missing_before > 0])
        else:
            st.write("- No missing values detected.")

        st.write(f"- Duplicate rows removed: {duplicates_before - duplicates_after}")
        st.write(f"- Potential outliers detected (z-score > 3): {outlier_count}")

        return True


        
    def load_data(self, quality_file, red_wine_file):
        """Load the wine quality and red wine datasets"""
        try:
            self.wine_quality_data = pd.read_csv(quality_file)
            self.red_wine_data = pd.read_csv(red_wine_file)
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Preprocess the wine quality data for training"""
        # Define feature columns based on the wine quality dataset
        self.feature_columns = [
            'volatile acidity', 'citric acid', 'residual sugar', 
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
            'density', 'pH', 'sulphates', 'alcohol'
        ]
        
        # Check if all columns exist in the dataset
        missing_cols = [col for col in self.feature_columns if col not in self.wine_quality_data.columns]
        if missing_cols:
            st.error(f"Missing columns in wine quality dataset: {missing_cols}")
            return None, None
        
        # Prepare features and target
        X = self.wine_quality_data[self.feature_columns]
        y = self.wine_quality_data['quality']
        
        # Handle missing values if any
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_models(self, X, y):
        """Train Decision Tree and Random Forest models with hyperparameter optimization"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Decision Tree with hyperparameter optimization
        dt_param_grid = {
            'max_depth': [3, 5, 7, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        
        dt_grid = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            dt_param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        dt_grid.fit(X_train, y_train)
        self.decision_tree = dt_grid.best_estimator_
        self.best_params = dt_grid.best_params_
        
        # Random Forest with hyperparameter optimization
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        rf_grid.fit(X_train, y_train)
        self.random_forest = rf_grid.best_estimator_
        
        # Evaluate models
        dt_pred = self.decision_tree.predict(X_test)
        rf_pred = self.random_forest.predict(X_test)
        
        dt_accuracy = accuracy_score(y_test, dt_pred)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        return {
            'dt_accuracy': dt_accuracy,
            'rf_accuracy': rf_accuracy,
            'best_dt_params': self.best_params,
            'best_rf_params': rf_grid.best_params_,
            'X_test': X_test,
            'y_test': y_test,
            'dt_pred': dt_pred,
            'rf_pred': rf_pred
        }
    
    def setup_recommender(self):
        """Setup recommendation system based on quality clusters"""
        if self.red_wine_data is None:
            return None
            
        # Prepare red wine features that match our model
        red_wine_feature_mapping = {
            'volatile acidity': 'volatile acidity',
            'citric acid': 'citric acid', 
            'residual sugar': 'residual sugar',
            'pH': 'pH',
            'alcohol': 'alcohol'
        }
        
        # Create feature matrix for red wines
        red_wine_features = pd.DataFrame()
        for model_feature in self.feature_columns:
            if model_feature in red_wine_feature_mapping:
                red_wine_features[model_feature] = self.red_wine_data[red_wine_feature_mapping[model_feature]]
            else:
                # Fill missing features with median values from training data
                red_wine_features[model_feature] = self.wine_quality_data[model_feature].median()
        
        # Handle missing values
        red_wine_features = red_wine_features.fillna(red_wine_features.mean())
        
        # Create quality-based clusters using K-means
        self.kmeans_model = KMeans(n_clusters=6, random_state=42)  # 6 clusters for qualities 3-8
        
        # Fit k-means on wine quality data
        quality_features = self.wine_quality_data[self.feature_columns].fillna(self.wine_quality_data[self.feature_columns].mean())
        quality_features_scaled = self.scaler.transform(quality_features)
        self.quality_clusters = self.kmeans_model.fit_predict(quality_features_scaled)
        
        # Create quality-cluster mapping
        self.quality_cluster_map = {}
        for i, quality in enumerate(self.wine_quality_data['quality']):
            cluster = self.quality_clusters[i]
            if quality not in self.quality_cluster_map:
                self.quality_cluster_map[quality] = []
            self.quality_cluster_map[quality].append(cluster)
        
        # Get most common cluster for each quality
        for quality in self.quality_cluster_map:
            clusters = self.quality_cluster_map[quality]
            self.quality_cluster_map[quality] = max(set(clusters), key=clusters.count)
        
        return red_wine_features
    
    def predict_quality(self, features, model_type='random_forest'):
        """Predict wine quality for given features"""
        # Ensure features match the expected number
        if len(features) != len(self.feature_columns):
            st.error(f"Expected {len(self.feature_columns)} features, got {len(features)}")
            return None, None
            
        features_scaled = self.scaler.transform([features])
        
        if model_type == 'decision_tree':
            prediction = self.decision_tree.predict(features_scaled)[0]
            probability = self.decision_tree.predict_proba(features_scaled)[0]
        else:
            prediction = self.random_forest.predict(features_scaled)[0]
            probability = self.random_forest.predict_proba(features_scaled)[0]
        
        return prediction, probability
    
    def get_quality_based_recommendations(self, predicted_quality, top_n=5):
        """Get wine recommendations based on predicted quality using multiple algorithms"""
        if self.red_wine_data is None:
            return []
        
        recommendations = []
        
        # Method 1: Direct quality-based filtering
        quality_range = [predicted_quality - 1, predicted_quality, predicted_quality + 1]
        
        # Method 2: Price-based recommendations (higher quality = higher price generally)
        price_ranges = {
            3: (0, 30),
            4: (20, 50),
            5: (30, 70),
            6: (50, 100),
            7: (70, 150),
            8: (100, 500)
        }
        
        min_price, max_price = price_ranges.get(predicted_quality, (0, 100))
        
        # Method 3: Alcohol content correlation (higher quality wines often have specific alcohol ranges)
        alcohol_ranges = {
            3: (8, 11),
            4: (9, 12),
            5: (10, 13),
            6: (11, 14),
            7: (12, 15),
            8: (13, 16)
        }
        
        min_alcohol, max_alcohol = alcohol_ranges.get(predicted_quality, (9, 15))
        
        # Method 4: User rating correlation
        rating_ranges = {
            3: (1, 5),
            4: (3, 6),
            5: (5, 7),
            6: (6, 8),
            7: (7, 9),
            8: (8, 10)
        }
        
        min_rating, max_rating = rating_ranges.get(predicted_quality, (5, 8))
        
        # Filter wines based on multiple criteria
        filtered_wines = self.red_wine_data[
            (self.red_wine_data['price'] >= min_price) & 
            (self.red_wine_data['price'] <= max_price) &
            (self.red_wine_data['alcohol'] >= min_alcohol) & 
            (self.red_wine_data['alcohol'] <= max_alcohol) &
            (self.red_wine_data['feedback rate by users'] >= min_rating) &
            (self.red_wine_data['feedback rate by users'] <= max_rating)
        ]
        
        # If no wines match strict criteria, relax constraints
        if len(filtered_wines) == 0:
            filtered_wines = self.red_wine_data[
                (self.red_wine_data['price'] >= min_price * 0.7) & 
                (self.red_wine_data['price'] <= max_price * 1.3) &
                (self.red_wine_data['feedback rate by users'] >= min_rating - 1)
            ]
        
        # Sort by multiple criteria (rating, price appropriateness, alcohol match)
        if len(filtered_wines) > 0:
            # Calculate recommendation scores
            filtered_wines = filtered_wines.copy()
            filtered_wines['price_score'] = 1 - abs(filtered_wines['price'] - (min_price + max_price) / 2) / max_price
            filtered_wines['alcohol_score'] = 1 - abs(filtered_wines['alcohol'] - (min_alcohol + max_alcohol) / 2) / max_alcohol
            filtered_wines['rating_score'] = filtered_wines['feedback rate by users'] / 10
            
            # Combined score
            filtered_wines['total_score'] = (
                filtered_wines['price_score'] * 0.3 +
                filtered_wines['alcohol_score'] * 0.3 +
                filtered_wines['rating_score'] * 0.4
            )
            
            # Sort by total score
            filtered_wines = filtered_wines.sort_values('total_score', ascending=False)
            
            # Get top recommendations
            for idx, wine in filtered_wines.head(top_n).iterrows():
                recommendation = {
                    'name': wine['Name of Red-wine'],
                    'country': wine['country'],
                    'price': wine['price'],
                    'alcohol': wine['alcohol'],
                    'rating': wine['feedback rate by users'],
                    'confidence': wine['total_score'],
                    'quality_match': predicted_quality,
                    'price_range': f"${min_price}-${max_price}",
                    'alcohol_range': f"{min_alcohol}-{max_alcohol}%"
                }
                recommendations.append(recommendation)
        
        # If still no recommendations, get top-rated wines in general
        if len(recommendations) == 0:
            top_wines = self.red_wine_data.nlargest(top_n, 'feedback rate by users')
            for idx, wine in top_wines.iterrows():
                recommendation = {
                    'name': wine['Name of Red-wine'],
                    'country': wine['country'],
                    'price': wine['price'],
                    'alcohol': wine['alcohol'],
                    'rating': wine['feedback rate by users'],
                    'confidence': 0.5,  # Lower confidence for general recommendations
                    'quality_match': predicted_quality,
                    'price_range': f"${min_price}-${max_price}",
                    'alcohol_range': f"{min_alcohol}-{max_alcohol}%"
                }
                recommendations.append(recommendation)
        
        return recommendations
    
    def get_country_analysis(self, predicted_quality):
        """Analyze wine recommendations by country"""
        if self.red_wine_data is None:
            return {}
            
        country_stats = {}
        
        # Get country-wise statistics
        for country in self.red_wine_data['country'].unique():
            country_wines = self.red_wine_data[self.red_wine_data['country'] == country]
            
            country_stats[country] = {
                'count': len(country_wines),
                'avg_price': country_wines['price'].mean(),
                'avg_rating': country_wines['feedback rate by users'].mean(),
                'avg_alcohol': country_wines['alcohol'].mean(),
                'price_range': f"${country_wines['price'].min():.0f}-${country_wines['price'].max():.0f}"
            }
        
        # Sort by average rating
        country_stats = dict(sorted(country_stats.items(), 
                                  key=lambda x: x[1]['avg_rating'], 
                                  reverse=True))
        
        return country_stats
    
    def save_model(self, filename='wine_quality_model.pkl'):
        """Save the trained models and preprocessor"""
        model_data = {
            'scaler': self.scaler,
            'decision_tree': self.decision_tree,
            'random_forest': self.random_forest,
            'knn_recommender': self.knn_recommender,
            'feature_columns': self.feature_columns,
            'best_params': self.best_params,
            'quality_clusters': self.quality_clusters,
            'kmeans_model': self.kmeans_model,
            'quality_cluster_map': getattr(self, 'quality_cluster_map', {})
        }
        joblib.dump(model_data, filename)
        return filename
    
    def load_model(self, filename='wine_quality_model.pkl'):
        """Load trained models and preprocessor"""
        try:
            model_data = joblib.load(filename)
            self.scaler = model_data['scaler']
            self.decision_tree = model_data['decision_tree']
            self.random_forest = model_data['random_forest']
            self.knn_recommender = model_data.get('knn_recommender')
            self.feature_columns = model_data['feature_columns']
            self.best_params = model_data['best_params']
            self.quality_clusters = model_data.get('quality_clusters')
            self.kmeans_model = model_data.get('kmeans_model')
            self.quality_cluster_map = model_data.get('quality_cluster_map', {})
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False

def main():
    st.set_page_config(page_title="Wine Quality Predictor", layout="wide")
    
    st.title("🍷 Wine Quality Prediction & Recommendation System")
    st.markdown("---")
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = WineQualityPredictor()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Model Training", "Wine Quality Prediction", "Model Performance", "Country Analysis"])
    
    if page == "Model Training":
        st.header("📊 Model Training")
        
        # File upload
        st.subheader("Upload Datasets")
        col1, col2 = st.columns(2)
        
        with col1:
            quality_file = st.file_uploader("Upload Wine Quality Dataset", type=['csv'], key="quality")
        
        with col2:
            red_wine_file = st.file_uploader("Upload Red Wine Dataset", type=['csv'], key="red_wine")
        
        if quality_file and red_wine_file:
            # Load data
            quality_df = pd.read_csv(quality_file)
            red_wine_df = pd.read_csv(red_wine_file)
            
            st.session_state.predictor.wine_quality_data = quality_df
            st.session_state.predictor.red_wine_data = red_wine_df
            
            # Clean the wine quality dataset
            if st.session_state.predictor.clean_wine_quality_data():
                st.success("✅ Datasets loaded and cleaned successfully!")
            else:
                st.warning("⚠️ Cleaning process incomplete.")

            
            # Display data info
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Wine Quality Dataset")
                st.write(f"Shape: {quality_df.shape}")
                st.write(quality_df.head())
            
            with col2:
                st.subheader("Red Wine Dataset")
                st.write(f"Shape: {red_wine_df.shape}")
                st.write(red_wine_df.head())
            
            # Train models
            if st.button("🚀 Train Models"):
                with st.spinner("Training models... This may take a few minutes."):
                    # Preprocess data
                    X, y = st.session_state.predictor.preprocess_data()
                    
                    if X is not None and y is not None:
                        # Train models
                        results = st.session_state.predictor.train_models(X, y)
                        
                        # Setup recommender
                        st.session_state.predictor.setup_recommender()
                        
                        # Save model
                        model_file = st.session_state.predictor.save_model()
                        
                        # Display results
                        st.success("🎉 Models trained successfully!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Decision Tree Accuracy", f"{results['dt_accuracy']:.3f}")
                            st.write("**Best Parameters:**")
                            st.json(results['best_dt_params'])
                        
                        with col2:
                            st.metric("Random Forest Accuracy", f"{results['rf_accuracy']:.3f}")
                            st.write("**Best Parameters:**")
                            st.json(results['best_rf_params'])
                        
                        st.session_state.training_results = results
                        
                        # Download model
                        with open(model_file, 'rb') as f:
                            st.download_button(
                                label="💾 Download Trained Model",
                                data=f.read(),
                                file_name=model_file,
                                mime="application/octet-stream"
                            )
    
    elif page == "Wine Quality Prediction":
        st.header("🔮 Wine Quality Prediction")
        
        # Check if model is trained
        if st.session_state.predictor.decision_tree is None:
            st.warning("⚠️ Please train the model first or load a pre-trained model.")
            
            # Option to load pre-trained model
            uploaded_model = st.file_uploader("Upload Pre-trained Model", type=['pkl'])
            if uploaded_model:
                try:
                    model_data = joblib.load(uploaded_model)
                    st.session_state.predictor.scaler = model_data['scaler']
                    st.session_state.predictor.decision_tree = model_data['decision_tree']
                    st.session_state.predictor.random_forest = model_data['random_forest']
                    st.session_state.predictor.knn_recommender = model_data.get('knn_recommender')
                    st.session_state.predictor.feature_columns = model_data['feature_columns']
                    st.session_state.predictor.best_params = model_data['best_params']
                    st.session_state.predictor.quality_clusters = model_data.get('quality_clusters')
                    st.session_state.predictor.kmeans_model = model_data.get('kmeans_model')
                    st.session_state.predictor.quality_cluster_map = model_data.get('quality_cluster_map', {})
                    st.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        
        else:
            # Input features
            st.subheader("🍇 Enter Wine Characteristics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
                citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
                residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
                chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.08, step=0.001)
                free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
            
            with col2:
                total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=45.0, step=1.0)
                density = st.number_input("Density", min_value=0.99, max_value=1.01, value=0.996, step=0.0001)
                pH = st.number_input("pH", min_value=2.0, max_value=5.0, value=3.3, step=0.01)
                sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.65, step=0.01)
                alcohol = st.number_input("Alcohol", min_value=8.0, max_value=16.0, value=10.0, step=0.1)
            
            # Model selection
            model_type = st.selectbox("Select Model", ["Random Forest", "Decision Tree"])
            
            if st.button("🎯 Predict Quality"):
                features = [
                    volatile_acidity, citric_acid, residual_sugar,
                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                    density, pH, sulphates, alcohol
                ]
                
                # Predict quality
                model_key = 'random_forest' if model_type == "Random Forest" else 'decision_tree'
                prediction, probability = st.session_state.predictor.predict_quality(features, model_key)
                
                if prediction is not None:
                    # Display prediction
                    st.success(f"🍷 Predicted Wine Quality: **{prediction}**")
                    
                    # Display probability distribution
                    if hasattr(st.session_state.predictor.random_forest, 'classes_'):
                        classes = st.session_state.predictor.random_forest.classes_
                        prob_df = pd.DataFrame({
                            'Quality': classes,
                            'Probability': probability
                        })
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(prob_df['Quality'], prob_df['Probability'])
                        ax.set_xlabel('Wine Quality')
                        ax.set_ylabel('Probability')
                        ax.set_title('Quality Prediction Probability Distribution')
                        
                        # Highlight predicted quality
                        for i, bar in enumerate(bars):
                            if classes[i] == prediction:
                                bar.set_color('red')
                        
                        st.pyplot(fig)
                    
                    # Get quality-based recommendations
                    if st.session_state.predictor.red_wine_data is not None:
                        st.subheader("🥂 Recommended Wines Based on Predicted Quality")
                        
                        recommendations = st.session_state.predictor.get_quality_based_recommendations(prediction, top_n=5)
                        
                        if recommendations:
                            # Display recommendation summary
                            st.info(f"**Recommendation Strategy for Quality {prediction}:**\n"
                                   f"- Expected Price Range: {recommendations[0]['price_range']}\n"
                                   f"- Expected Alcohol Range: {recommendations[0]['alcohol_range']}\n"
                                   f"- Target Rating: {7 + prediction - 5:.1f}+/10")
                            
                            # Display individual recommendations
                            for i, wine in enumerate(recommendations, 1):
                                with st.expander(f"🍷 #{i} {wine['name']} (Confidence: {wine['confidence']:.2f})"):
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.write(f"**🌍 Country:** {wine['country']}")
                                        st.write(f"**💰 Price:** ${wine['price']:.2f}")
                                    
                                    with col2:
                                        st.write(f"**🍺 Alcohol:** {wine['alcohol']:.1f}%")
                                        st.write(f"**⭐ Rating:** {wine['rating']:.1f}/10")
                                    
                                    with col3:
                                        st.write(f"**🎯 Quality Match:** {wine['quality_match']}")
                                        st.write(f"**📊 Confidence:** {wine['confidence']:.2f}")
                                        
                                        # Recommendation reason
                                        if wine['confidence'] > 0.7:
                                            st.success("🎯 Perfect Match!")
                                        elif wine['confidence'] > 0.5:
                                            st.warning("👍 Good Match")
                                        else:
                                            st.info("💡 General Recommendation")
                        else:
                            st.warning("No specific recommendations found for this quality level.")
                    else:
                        st.warning("Red wine dataset not available for recommendations.")
    
    elif page == "Model Performance":
        st.header("📈 Model Performance Analysis")
        
        if 'training_results' not in st.session_state:
            st.warning("⚠️ No training results available. Please train the model first.")
        else:
            results = st.session_state.training_results
            
            # Performance metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Decision Tree Performance")
                st.metric("Accuracy", f"{results['dt_accuracy']:.3f}")
                
                # Confusion matrix
                cm_dt = confusion_matrix(results['y_test'], results['dt_pred'])
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title('Decision Tree Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Random Forest Performance")
                st.metric("Accuracy", f"{results['rf_accuracy']:.3f}")
                
                # Confusion matrix
                cm_rf = confusion_matrix(results['y_test'], results['rf_pred'])
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax)
                ax.set_title('Random Forest Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
            
            # Feature importance (Random Forest)
            if st.session_state.predictor.random_forest is not None:
                st.subheader("🔍 Feature Importance")
                feature_importance = pd.DataFrame({
                    'feature': st.session_state.predictor.feature_columns,
                    'importance': st.session_state.predictor.random_forest.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(range(len(feature_importance)), feature_importance['importance'])
                ax.set_xlabel('Features')
                ax.set_ylabel('Importance')
                ax.set_title('Feature Importance (Random Forest)')
                ax.set_xticks(range(len(feature_importance)))
                ax.set_xticklabels(feature_importance['feature'], rotation=45, ha='right')
                
                st.pyplot(fig)
                
                # Display feature importance table
                st.write("**Feature Importance Ranking:**")
                st.dataframe(feature_importance)
    
    elif page == "Country Analysis":
        st.header("🌍 Wine Country Analysis")
        
        if st.session_state.predictor.red_wine_data is not None:
            # Get country statistics
            country_stats = st.session_state.predictor.get_country_analysis(5)  # Default quality 5
            
            if country_stats:
                st.subheader("📊 Country-wise Wine Statistics")
                
                # Convert to DataFrame for display
                stats_df = pd.DataFrame(country_stats).T
                stats_df = stats_df.round(2)
                
                st.dataframe(stats_df)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Average price by country
                    fig, ax = plt.subplots(figsize=(10, 6))




if __name__ == "__main__":
    main()


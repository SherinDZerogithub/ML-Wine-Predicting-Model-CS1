# app.py
from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

app = Flask(__name__)

# Load model and scaler
model = XGBClassifier()
model.load_model('xgb_model.json')
scaler = joblib.load('scaler.pkl')

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from form
        features = [float(request.form[col]) for col in [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]]

        # Scale
        input_scaled = scaler.transform([features])

        # Predict
        prediction = model.predict(input_scaled)[0]
        result = "Good Quality 🍷" if prediction == 1 else "Bad Quality 🍷"

        return render_template('index.html', prediction_text=f'Wine is: {result}')
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)

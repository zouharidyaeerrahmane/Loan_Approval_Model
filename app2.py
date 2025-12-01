from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')

# --- LOAD MODEL ---
try:
    with open(MODEL_PATH, 'rb') as file:
        artifacts = pickle.load(file)
    W = artifacts["W"]
    b = artifacts["b"]
    scaler = artifacts["scaler"]
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'models/model.pkl' not found.")
    exit()

# --- HELPER FUNCTIONS ---
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predict_logic(X, W, b):
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    return (A >= 0.5).astype(int), A

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get data from Form
        # We must convert strings to numbers manually here
        form_data = request.form
        
        # CATEGORICAL ENCODING (Adjust these to match your training!)
        # Assuming: Graduate=1, Not Graduate=0 / Yes=1, No=0
        education_val = 1 if form_data['Education'] == 'Graduate' else 0
        self_employed_val = 1 if form_data['Self_Employed'] == 'Yes' else 0
        
        # NUMERICAL FEATURES extraction
        features = [
            int(form_data['Dependents']),
            education_val,
            self_employed_val,
            float(form_data['Annual_Income']),
            float(form_data['Loan_Amount']),
            float(form_data['Loan_Period']),
            float(form_data['Credit_Score']),
            float(form_data['Residential_Assets']),
            float(form_data['Commercial_Assets']),
            float(form_data['Luxury_Assets']),
            float(form_data['Bank_Assets'])
        ]

        # 2. Prepare for Model
        features_array = np.array(features).reshape(1, -1)
        
        # 3. Normalize
        features_scaled = scaler.transform(features_array)
        
        # 4. Predict
        pred_class, probability = predict_logic(features_scaled, W, b)
        
        result_text = "Approved" if pred_class[0] == 1 else "Rejected"
        confidence_score = f"{float(probability[0]) * 100:.2f}%"

        # 5. Render Result Page with Data
        return render_template('results.html', 
                             result=result_text, 
                             confidence=confidence_score,
                             applicant_data=form_data) # Pass original data to display

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
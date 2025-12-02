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
        # 1. Get data from Form and store for display
        form_data = request.form
        
        # Create a mutable dictionary for display purposes
        applicant_data_display = dict(form_data)

        # 2. Get values directly from the form, matching the model's feature names
        
        # Handle Dependents: Convert '3+' to 3
        dependents_str = form_data.get('Dependants_namber', '0')
        dependants_number = 3 if dependents_str == '3+' else int(dependents_str)

        # Encode Education: Graduate=1, Not Graduate=0
        education_val = 1 if form_data.get('Education') == 'Graduate' else 0
        
        # Encode Self-Employed: Yes=1, No=0
        self_employed_val = 1 if form_data.get('Self_Employed') == 'Yes' else 0

        # Get numerical values directly
        annual_income = float(form_data.get('Annula_Income', 0))
        loan_amount = float(form_data.get('Loan_Amount', 0))
        loan_period_months = float(form_data.get('Loan_Period_Months', 0))
        credit_score = float(form_data.get('Credit_Score', 0))
        residential_assets = float(form_data.get('Residential_Assets', 0))
        commercial_assets = float(form_data.get('Commercial_Assets', 0))
        luxury_assets = float(form_data.get('Luxury_Assets', 0))
        bank_assets = float(form_data.get('Bank_Assets', 0))

        # 3. Create the feature array in the EXACT order the model expects
        # Order: ['Dependants_namber', 'Education', 'Self_Employed', 'Annula_Income', 'Loan_Amount', 
        #         'Loan_Period_Months', 'Credit_Score', 'Residential_Assets', 'Commercial_Assets', 
        #         'Luxury_Assets', 'Bank_Assets']
        features = [
            dependants_number,
            education_val,
            self_employed_val,
            annual_income,
            loan_amount,
            loan_period_months,
            credit_score,
            residential_assets,
            commercial_assets,
            luxury_assets,
            bank_assets
        ]

        # 4. Prepare for Model
        features_array = np.array(features).reshape(1, -1)
        
        # 5. Normalize
        features_scaled = scaler.transform(features_array)
        
        # 6. Predict
        pred_class, probability = predict_logic(features_scaled, W, b)
        
        # The model predicts 0 for 'Approved' and 1 for 'Rejected'
        result_text = "Approved" if pred_class[0] == 0 else "Rejected"
        
        # Adjust confidence score based on prediction
        if result_text == "Approved":
            confidence_score = f"{float(1 - probability[0]) * 100:.2f}%"
        else:
            confidence_score = f"{float(probability[0]) * 100:.2f}%"

        # 7. Render Result Page with Data
        return render_template('result.html', 
                             result=result_text, 
                             confidence=confidence_score,
                             applicant_data=applicant_data_display)

    except Exception as e:
        # Log the full error for debugging
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
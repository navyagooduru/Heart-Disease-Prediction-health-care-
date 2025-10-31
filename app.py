from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
model = None
scaler = None
feature_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# ---------------------- MODEL FUNCTIONS ----------------------

def load_and_train_model():
    """Train model if no saved model exists"""
    global model, scaler

    try:
        print("Loading dataset...")
        df = pd.read_csv('heart.csv')
        print(f"Dataset loaded with {len(df)} records")

        X = df.drop('target', axis=1)
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained successfully! Accuracy: {accuracy:.4f}")

        joblib.dump(model, 'heart_disease_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')

        return True

    except Exception as e:
        print(f"Error training model: {e}")
        print(traceback.format_exc())
        return False


def load_saved_model():
    """Load model and scaler from disk"""
    global model, scaler

    try:
        if os.path.exists('heart_disease_model.pkl') and os.path.exists('scaler.pkl'):
            model = joblib.load('heart_disease_model.pkl')
            scaler = joblib.load('scaler.pkl')
            print("Model loaded successfully from saved files!")
            return True
        else:
            print("Saved model not found.")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def initialize_model():
    """Initialize model once at app startup"""
    global model, scaler
    print("Initializing model...")
    if not load_saved_model():
        print("Training new model...")
        load_and_train_model()

# ---------------------- ROUTES ----------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form
        print("Received data:", data)

        # Validate all required fields
        missing_fields, features = [], []
        for feature in feature_names:
            if feature not in data or data[feature] == '':
                missing_fields.append(feature)
            else:
                try:
                    features.append(float(data[feature]))
                except ValueError:
                    return jsonify({'error': f'Invalid value for {feature}'}), 400

        if missing_fields:
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400

        if model is None or scaler is None:
            initialize_model()

        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        result = {
            'prediction': int(prediction),
            'probability': float(probability[1]),
            'message': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected',
            'risk_level': 'High' if prediction == 1 else 'Low'
        }
        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/get_sample_data', methods=['GET'])
def get_sample_data():
    try:
        sample_data = {
            'age': 52, 'sex': 1, 'cp': 0, 'trestbps': 125, 'chol': 212,
            'fbs': 0, 'restecg': 1, 'thalach': 168, 'exang': 0,
            'oldpeak': 1.0, 'slope': 2, 'ca': 2, 'thal': 3
        }
        return jsonify(sample_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })


@app.route('/get_dataset_info', methods=['GET'])
def get_dataset_info():
    try:
        df = pd.read_csv('heart.csv')
        info = {
            'total_records': len(df),
            'features': list(df.columns),
            'target_distribution': df['target'].value_counts().to_dict()
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------- MAIN ----------------------

if __name__ == '__main__':
    print("Starting Heart Disease Prediction Server...")
    initialize_model()
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

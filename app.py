"""
Breast Cancer Prediction Web Application
Student: Jimoh-Alabi Islamiat Modupe
Matric No: 250000033

A Flask-based web application for breast cancer classification
using machine learning (Logistic Regression).

Algorithm: Logistic Regression
Model Persistence: Pickle
Dataset: Wisconsin Breast Cancer Dataset (UCI Repository)
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import json
import os
import sys

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'breast_cancer_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')

print("="*80)
print("BREAST CANCER PREDICTION WEB APPLICATION")
print("="*80)

# Load model and resources
try:
    print("\n[1] Loading trained model...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("    ✅ Model loaded successfully!")
    
    print("[2] Loading feature scaler...")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("    ✅ Scaler loaded successfully!")
    
    print("[3] Loading metadata...")
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    print("    ✅ Metadata loaded successfully!")
    
    print("\n" + "="*80)
    print("✅ APPLICATION READY")
    print("="*80)
    print(f"\nModel Type: {metadata['model_type']}")
    print(f"Accuracy: {metadata['accuracy']*100:.2f}%")
    print(f"F1-Score: {metadata['f1_score']:.4f}")
    print("="*80)
    
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Required file not found!")
    print(f"   {e}")
    print("\n   Please ensure all model files are in the 'model/' directory:")
    print("   - breast_cancer_model.pkl")
    print("   - scaler.pkl")
    print("   - model_metadata.json")
    exit(1)
except Exception as e:
    print(f"\n❌ ERROR loading resources: {e}")
    exit(1)

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', 
                         feature_names=metadata['feature_names'],
                         model_accuracy=metadata['accuracy'],
                         model_type=metadata['model_type'])

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests
    
    Expected JSON format:
    {
        "features": [array of 30 float values]
    }
    
    Returns:
    {
        "success": true/false,
        "prediction": "benign" or "malignant",
        "confidence": float (0-1),
        "probabilities": {
            "benign": float,
            "malignant": float
        }
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data received'
            }), 400
        
        features = data.get('features', [])
        
        # Validate input
        if not features:
            return jsonify({
                'success': False,
                'error': 'No features provided'
            }), 400
        
        if len(features) != 30:
            return jsonify({
                'success': False,
                'error': f'Expected 30 features, received {len(features)}'
            }), 400
        
        # Convert to numpy array and reshape
        input_data = np.array(features, dtype=np.float64).reshape(1, -1)
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = int(model.predict(input_scaled)[0])
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Get confidence (probability of the predicted class)
        prob_malignant = float(probabilities[0])
        prob_benign = float(probabilities[1])
        confidence = prob_benign if prediction == 1 else prob_malignant
        
        # Prepare response
        result = {
            'success': True,
            'prediction': metadata['target_names'][prediction],
            'confidence': confidence,
            'probabilities': {
                'malignant': prob_malignant,
                'benign': prob_benign
            },
            'interpretation': get_interpretation(prediction, confidence)
        }
        
        # Log prediction
        print(f"\n✅ Prediction: {result['prediction']} "
              f"(confidence: {result['confidence']:.4f})")
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid input data: {str(e)}'
        }), 400
    
    except Exception as e:
        print(f"\n❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'An error occurred during prediction'
        }), 500

def get_interpretation(prediction, confidence):
    """
    Generate human-readable interpretation of prediction
    
    Args:
        prediction (int): 0 for malignant, 1 for benign
        confidence (float): Probability of predicted class
    
    Returns:
        str: Interpretation message
    """
    if prediction == 1:  # Benign
        if confidence > 0.9:
            return "High confidence: The tumor is likely BENIGN (non-cancerous)."
        elif confidence > 0.75:
            return "Moderate confidence: The tumor appears to be BENIGN."
        else:
            return "Low confidence: Further medical evaluation is strongly recommended."
    else:  # Malignant
        if confidence > 0.9:
            return "High confidence: The tumor is likely MALIGNANT (cancerous). Immediate medical consultation is advised."
        elif confidence > 0.75:
            return "Moderate confidence: The tumor appears to be MALIGNANT. Please consult a healthcare professional."
        else:
            return "Low confidence: Results are inconclusive. Professional medical evaluation is necessary."

@app.route('/model_info')
def model_info():
    """Return model metadata"""
    return jsonify(metadata)

@app.route('/health')
def health():
    """Health check endpoint for deployment monitoring"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'scaler_loaded': True
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Get port from environment variable (for deployment) or use 8080
    port = int(os.environ.get('PORT', 8080))
    
    print("\n" + "="*80)
    print("🚀 STARTING WEB APPLICATION")
    print("="*80)
    print(f"\n📱 Local URL: http://127.0.0.1:{port}")
    print("\n⚠️  EDUCATIONAL PURPOSE ONLY")
    print("   This application is for educational demonstration.")
    print("   NOT intended for actual medical diagnosis.")
    print("   Always consult qualified healthcare professionals.")
    print("\n" + "="*80 + "\n")
    
    # Run the application
    # debug=False for production deployment
    app.run(host='0.0.0.0', port=port, debug=False)
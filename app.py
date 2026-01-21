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

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'breast_cancer_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')

# Define the 5 features you selected for your model
# IMPORTANT: These must match the features you used in model_building.ipynb
SELECTED_FEATURES = [
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'concavity_mean'
]

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
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        print("    ✅ Metadata loaded successfully!")
    else:
        # Create default metadata if file doesn't exist
        metadata = {
            'model_type': 'Logistic Regression',
            'feature_names': SELECTED_FEATURES,
            'target_names': ['malignant', 'benign'],
            'accuracy': 0.95,
            'f1_score': 0.95
        }
        print("    ⚠️  Metadata file not found, using defaults")
    
    print("\n" + "="*80)
    print("✅ APPLICATION READY")
    print("="*80)
    print(f"\nModel Type: {metadata['model_type']}")
    print(f"Selected Features: {', '.join(SELECTED_FEATURES)}")
    print(f"Number of Features: {len(SELECTED_FEATURES)}")
    print("="*80)
    
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Required file not found!")
    print(f"   {e}")
    print("\n   Please ensure all model files are in the 'model/' directory:")
    print("   - breast_cancer_model.pkl")
    print("   - scaler.pkl")
    exit(1)
except Exception as e:
    print(f"\n❌ ERROR loading resources: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', 
                         feature_names=SELECTED_FEATURES,
                         model_accuracy=metadata.get('accuracy', 0.95),
                         model_type=metadata.get('model_type', 'Logistic Regression'))

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests
    
    Expected JSON format:
    {
        "features": [array of 5 float values]
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
        
        expected_features = len(SELECTED_FEATURES)
        if len(features) != expected_features:
            return jsonify({
                'success': False,
                'error': f'Expected {expected_features} features, received {len(features)}'
            }), 400
        
        # Check for NaN or invalid values
        try:
            features_float = [float(f) for f in features]
            if any(np.isnan(features_float)) or any(np.isinf(features_float)):
                return jsonify({
                    'success': False,
                    'error': 'Invalid feature values (NaN or Inf detected)'
                }), 400
        except (ValueError, TypeError) as e:
            return jsonify({
                'success': False,
                'error': f'Invalid feature values: {str(e)}'
            }), 400
        
        # Convert to numpy array and reshape
        input_data = np.array(features_float, dtype=np.float64).reshape(1, -1)
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = int(model.predict(input_scaled)[0])
        probabilities = model.predict_proba(input_scaled)[0]
        
        # FIXED: Correct probability mapping
        # For sklearn LogisticRegression with binary classification:
        # probabilities[0] = probability of class 0 (malignant)
        # probabilities[1] = probability of class 1 (benign)
        prob_malignant = float(probabilities[0])
        prob_benign = float(probabilities[1])
        
        # Get confidence (probability of the predicted class)
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
        print(f"\n✅ Prediction: {result['prediction'].upper()} "
              f"(confidence: {result['confidence']*100:.2f}%)")
        print(f"   Probabilities - Malignant: {prob_malignant:.4f}, Benign: {prob_benign:.4f}")
        
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
        'scaler_loaded': True,
        'features': SELECTED_FEATURES
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
    app.run(host='0.0.0.0', port=port, debug=False)
"""
Breast Cancer Prediction Web Application
Student: Jimoh-Alabi Islamiat Modupe
Matric No: 250000033

A Streamlit-based web application for breast cancer classification
using machine learning (Logistic Regression).

Algorithm: Logistic Regression
Model Persistence: Pickle
Dataset: Wisconsin Breast Cancer Dataset (UCI Repository)
"""

import streamlit as st
import numpy as np
import pickle
import json
import os

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🏥",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 20px 0;
        border-radius: 5px;
    }
    .result-benign {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .result-malignant {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 15px;
        margin: 20px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'breast_cancer_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')

# Load model and resources
@st.cache_resource
def load_model_resources():
    """Load model, scaler, and metadata"""
    try:
        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load metadata
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                'model_type': 'Logistic Regression',
                'feature_names': ['radius_mean', 'texture_mean', 'perimeter_mean', 
                                'area_mean', 'concavity_mean'],
                'target_names': ['malignant', 'benign'],
                'accuracy': 0.95
            }
        
        return model, scaler, metadata
    
    except FileNotFoundError as e:
        st.error(f"❌ Error: Required model files not found in '{MODEL_DIR}/' directory")
        st.error(f"Please ensure the following files exist:")
        st.error("- breast_cancer_model.pkl")
        st.error("- scaler.pkl")
        st.error("- model_metadata.json (optional)")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load resources
model, scaler, metadata = load_model_resources()

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 Breast Cancer Prediction System</h1>
    <p>Machine Learning Based Tumor Classification</p>
    <p style="font-size: 12px; margin-top: 10px;">
        Student: Jimoh-Alabi Islamiat Modupe | Matric: 250000033
    </p>
</div>
""", unsafe_allow_html=True)

# Warning message
st.markdown("""
<div class="warning-box">
    <h3>⚠️ Educational Purpose Only</h3>
    <p>This system is a machine learning demonstration and should NOT be used for actual 
    medical diagnosis. Always consult qualified healthcare professionals for medical decisions.</p>
</div>
""", unsafe_allow_html=True)

# Model information
st.markdown(f"""
<div class="info-box">
    <p><strong>Model Information:</strong> {metadata['model_type']} trained on Wisconsin 
    Breast Cancer Dataset with {len(metadata['feature_names'])} selected features.</p>
    <p><strong>Model Accuracy:</strong> {metadata.get('accuracy', 0.95)*100:.2f}%</p>
</div>
""", unsafe_allow_html=True)

# Input form
st.subheader("📊 Enter Tumor Feature Values")

col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input(
        "Radius Mean",
        min_value=0.0,
        max_value=50.0,
        value=14.0,
        step=0.01,
        help="Typical range: 6.98 - 28.11"
    )
    
    texture_mean = st.number_input(
        "Texture Mean",
        min_value=0.0,
        max_value=50.0,
        value=20.0,
        step=0.01,
        help="Typical range: 9.71 - 39.28"
    )
    
    perimeter_mean = st.number_input(
        "Perimeter Mean",
        min_value=0.0,
        max_value=200.0,
        value=90.0,
        step=0.01,
        help="Typical range: 43.79 - 188.50"
    )

with col2:
    area_mean = st.number_input(
        "Area Mean",
        min_value=0.0,
        max_value=3000.0,
        value=600.0,
        step=1.0,
        help="Typical range: 143.5 - 2501.0"
    )
    
    concavity_mean = st.number_input(
        "Concavity Mean",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.001,
        help="Typical range: 0.0 - 0.43",
        format="%.4f"
    )

# Prediction button
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button("🔬 Predict", use_container_width=True, type="primary")

# Make prediction
if predict_button:
    # Collect features
    features = np.array([
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        concavity_mean
    ]).reshape(1, -1)
    
    # Validate inputs
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        st.error("❌ Invalid input values detected. Please check your inputs.")
    else:
        with st.spinner("Analyzing tumor features..."):
            try:
                # Scale features
                features_scaled = scaler.transform(features)
                
                # Make prediction
                prediction = int(model.predict(features_scaled)[0])
                probabilities = model.predict_proba(features_scaled)[0]
                
                # Get probabilities
                prob_malignant = float(probabilities[0])
                prob_benign = float(probabilities[1])
                
                # Get confidence
                confidence = prob_benign if prediction == 1 else prob_malignant
                
                # Display result
                if prediction == 1:  # Benign
                    st.markdown("""
                    <div class="result-benign">
                        <h2 style="color: #28a745;">✅ Prediction: BENIGN (Non-cancerous)</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if confidence > 0.9:
                        interpretation = "High confidence: The tumor is likely BENIGN (non-cancerous)."
                    elif confidence > 0.75:
                        interpretation = "Moderate confidence: The tumor appears to be BENIGN."
                    else:
                        interpretation = "Low confidence: Further medical evaluation is strongly recommended."
                else:  # Malignant
                    st.markdown("""
                    <div class="result-malignant">
                        <h2 style="color: #dc3545;">⚠️ Prediction: MALIGNANT (Cancerous)</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if confidence > 0.9:
                        interpretation = "High confidence: The tumor is likely MALIGNANT. Immediate medical consultation is advised."
                    elif confidence > 0.75:
                        interpretation = "Moderate confidence: The tumor appears to be MALIGNANT. Please consult a healthcare professional."
                    else:
                        interpretation = "Low confidence: Results are inconclusive. Professional medical evaluation is necessary."
                
                # Show details
                st.markdown("### 📊 Prediction Details")
                st.write(f"**Interpretation:** {interpretation}")
                st.write(f"**Confidence:** {confidence*100:.2f}%")
                
                # Progress bar for confidence
                st.progress(confidence)
                
                # Probabilities
                st.markdown("### 📈 Class Probabilities")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Malignant", f"{prob_malignant*100:.2f}%")
                with col2:
                    st.metric("Benign", f"{prob_benign*100:.2f}%")
                
                # Disclaimer
                st.warning("⚠️ Remember: This is an educational tool. Always consult medical professionals for diagnosis.")
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.error("Please check your input values and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>© 2026 Breast Cancer Prediction System | Educational Project</p>
    <p>Powered by Logistic Regression & Streamlit</p>
</div>
""", unsafe_allow_html=True)
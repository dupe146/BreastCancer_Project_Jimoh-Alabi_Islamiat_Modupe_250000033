# 🔬 Breast Cancer Classification Web Application

A machine learning-based web application for breast cancer tumor classification using the Wisconsin Breast Cancer Dataset.
---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running Locally](#running-locally)
- [Deployment (Render.com)](#deployment-rendercom)
- [API Documentation](#api-documentation)
- [Model Information](#model-information)
- [Disclaimer](#disclaimer)

---

## 📊 Project Overview

This project implements a complete machine learning pipeline for breast cancer classification:

1. **Data Processing**: Loading and preprocessing the Wisconsin Breast Cancer Dataset
2. **Model Training**: Training a Logistic Regression classifier
3. **Web Application**: Flask-based interactive web interface
4. **Deployment**: Production-ready deployment on Render.com

live link: 

### Key Achievements
- ✅ **Accuracy**: 97%+
- ✅ **F1-Score**: 0.97
- ✅ **Real-time Predictions**: Instant classification results
- ✅ **User-Friendly Interface**: Clean, intuitive web UI

---

## ✨ Features

- 🔍 **Tumor Classification**: Binary classification (Benign/Malignant)
- 📊 **30 Feature Analysis**: Comprehensive tumor characteristic evaluation
- 💯 **Confidence Scores**: Probability estimates for predictions
- 🎨 **Modern UI**: Clean, responsive web interface
- 🚀 **Fast Predictions**: Near-instant results
- 📱 **Mobile Friendly**: Responsive design for all devices
- 🔒 **API Endpoints**: RESTful API for programmatic access

---

## 🛠️ Technologies Used

### Backend
- **Python 3.10+**
- **Flask 3.0.0** - Web framework
- **scikit-learn 1.3.2** - Machine learning
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Gunicorn** - Production WSGI server

### Frontend
- **HTML5**
- **CSS3** (with custom styling)
- **JavaScript (Vanilla)**

### Deployment
- **Render.com** - Cloud platform
- **Git/GitHub** - Version control

---

## 📁 Project Structure

```
BreastCancer_Project_ModupeJimoh_MatricNo/
│
├── app.py                              # Main Flask application
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
│
├── model/                             # Model artifacts
│   ├── model_building.ipynb          # Training notebook
│   ├── breast_cancer_model.pkl       # Trained model (~50KB)
│   ├── scaler.pkl                    # Feature scaler (~5KB)
│   └── model_metadata.json           # Model information
│
├── templates/                         # HTML templates
│   └── index.html                    # Main web interface
│
└── static/                            # Static files (optional)
    └── style.css                     # Custom styles
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git

### Step 1: Clone Repository

```bash
git clone BreastCancer_Project_Jimoh-Alabi_Islamiat_Modupe_250000033.git
cd BreastCancer_Project_Jimoh-Alabi_Islamiat_Modupe_250000033
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Model Files

Ensure these files exist in the `model/` directory:
- ✅ `breast_cancer_model.pkl`
- ✅ `scaler.pkl`
- ✅ `model_metadata.json`

---

## 💻 Running Locally

### Start the Application

```bash
python app.py
```

### Access the Application

Open your browser and navigate to:
```
http://127.0.0.1:8080
```

### Testing the Application

1. Click **"Load Benign Example"** or **"Load Malignant Example"**
2. Click **"Predict with Machine Learning"**
3. View the prediction results with confidence scores

### Stopping the Application

Press `Ctrl + C` in the terminal

---

## 🌐 Deployment (Render.com)

### Prerequisites
- GitHub account
- Render.com account (free tier available)
- Project pushed to GitHub

### Step-by-Step Deployment

#### 1. Prepare for Deployment

Ensure your project has:
- ✅ `requirements.txt`
- ✅ All model files in `model/` directory
- ✅ `.gitignore` file (to avoid uploading unnecessary files)

#### 2. Push to GitHub

```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

#### 3. Deploy on Render.com

1. **Go to Render.com** and sign in
2. **Click "New +" → "Web Service"**
3. **Connect your GitHub repository**
4. **Configure the service:**

   ```
   Name: breast-cancer-predictor-yourname
   Region: Choose closest to you
   Branch: main
   Root Directory: (leave empty)
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app:app
   ```

5. **Set Environment Variables** (if needed):
   ```
   PORT=10000
   ```

6. **Click "Create Web Service"**

7. **Wait for deployment** (takes 2-5 minutes)

8. **Access your app** at:
   ```
   https://breast-cancer-predictor-yourname.onrender.com
   ```

### Render.com Settings Summary

| Setting | Value |
|---------|-------|
| **Environment** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app` |
| **Port** | Auto-detected (10000) |
| **Instance Type** | Free |

### Troubleshooting Deployment

**Issue: Build fails**
- Check `requirements.txt` for typos
- Ensure Python version compatibility

**Issue: App crashes on start**
- Check that all model files are committed to Git
- Verify `app.py` has no syntax errors
- Check Render logs: Dashboard → Logs

**Issue: 404 errors**
- Ensure `templates/index.html` exists
- Check file paths are correct

---

## 📡 API Documentation

### Endpoints

#### 1. Home Page
```
GET /
```
Returns the main web interface.

#### 2. Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

#### 3. Get Model Information
```
GET /model_info
```

**Response:**
```json
{
  "model_type": "Logistic Regression",
  "accuracy": 0.9737,
  "f1_score": 0.9712,
  "feature_names": [...],
  "target_names": ["malignant", "benign"]
}
```

#### 4. Make Prediction
```
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "features": [
    17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419,
    0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373,
    0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622,
    0.6656, 0.7119, 0.2654, 0.4601, 0.1189
  ]
}
```

**Response:**
```json
{
  "success": true,
  "prediction": "malignant",
  "confidence": 0.9856,
  "probabilities": {
    "malignant": 0.9856,
    "benign": 0.0144
  },
  "interpretation": "High confidence: The tumor is likely MALIGNANT..."
}
```

### Using the API with cURL

```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [17.99, 10.38, 122.8, ...]
  }'
```

---

## 🧠 Model Information

### Dataset
- **Source**: Wisconsin Breast Cancer Dataset (UCI Repository)
- **Samples**: 569 (357 benign, 212 malignant)
- **Features**: 30 numerical features computed from digitized images

### Feature Categories
1. **Mean values** (10 features): radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
2. **Standard error** (10 features): SE of above measurements
3. **Worst values** (10 features): Largest values of above measurements

### Model Architecture
- **Algorithm**: Logistic Regression
- **Solver**: lbfgs
- **Max Iterations**: 10,000
- **Random State**: 42 (for reproducibility)

### Performance Metrics
- **Accuracy**: 97.37%
- **Precision**: 96.88%
- **Recall**: 98.46%
- **F1-Score**: 97.12%

### Data Preprocessing
1. Train-test split: 80-20
2. Feature scaling: StandardScaler (zero mean, unit variance)
3. Stratified sampling: Maintains class distribution

---

## ⚠️ Disclaimer

**IMPORTANT: This application is for EDUCATIONAL PURPOSES ONLY.**

- ❌ NOT intended for actual medical diagnosis
- ❌ NOT a substitute for professional medical advice
- ❌ NOT validated for clinical use

**Always consult qualified healthcare professionals for medical concerns.**

This project demonstrates:
- Machine learning classification techniques
- Web application development
- Model deployment practices

It should **not** be used to make real medical decisions.

---

## 📝 Development Notes

### Local Development

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Run in debug mode (for development)
export FLASK_ENV=development  # macOS/Linux
set FLASK_ENV=development     # Windows
python app.py
```

### Testing

Test the health endpoint:
```bash
curl http://127.0.0.1:8080/health
```

---

## 🤝 Contributing

This is an educational project. Contributions for learning purposes are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is created for educational purposes as part of academic coursework.

---
---

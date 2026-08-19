# ❤️ Heart Disease Prediction App

A machine learning-powered web application that predicts the risk of heart disease based on patient health metrics using the K-Nearest Neighbors (KNN) algorithm.

## 📋 About the Project

This project addresses the critical need for early heart disease detection by leveraging machine learning to analyze patient health data and provide risk predictions.

**What the project does:**

- Analyzes patient health parameters (age, blood pressure, cholesterol, heart rate, etc.)
- Predicts the probability of heart disease presence
- Provides an interactive web interface for easy use by healthcare professionals and patients

**What problem it solves:**

- Early detection of heart disease risk can save lives
- Reduces the burden on medical professionals for initial screening
- Provides quick, data-driven predictions based on established health metrics

**Why it was built:**

- Heart disease is one of the leading causes of death worldwide
- Machine learning can provide rapid, consistent analysis of health data
- User-friendly interfaces make predictions accessible to non-technical users

## ✨ Features

- 🔍 **Real-time Prediction** - Instant heart disease risk assessment
- 📊 **Interactive Input Form** - Easy-to-use sidebar interface with sliders and dropdowns
- 💡 **Visual Results** - Clear, color-coded prediction output (High Risk/Low Risk)
- 🧾 **Patient Summary** - Display of input parameters for verification
- 📱 **Responsive Design** - Works seamlessly on desktop and mobile devices
- 🎯 **Multiple Input Parameters** - Analyzes 11 different health metrics for comprehensive assessment

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Streamlit** - Interactive web application framework
- **Scikit-learn** - Machine learning library (KNN model)
- **Pandas** - Data manipulation and processing
- **NumPy** - Numerical computing
- **Joblib** - Model and data serialization

## 📂 Project Structure

```
Heart_Disease/
│
├── app.py                      # Main Streamlit application
├── knn_heart_model.pkl         # Trained KNN model
├── scaler.pkl                  # Data scaler for normalization
├── columns.pkl                 # Expected feature columns
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .git/                       # Version control

```

## ⚙️ Installation

Follow these steps to set up the project on your local machine:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Heart_Disease.git
   cd Heart_Disease
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 How to Run

1. **Ensure you're in the project directory:**

   ```bash
   cd path/to/Heart_Disease
   ```

2. **Activate the virtual environment** (if you created one)

3. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

4. **Access the application:**
   - The app will automatically open in your default browser
   - If not, navigate to: `http://localhost:8501`

5. **Make a prediction:**
   - Enter patient details using the sidebar controls
   - Click the "🔍 Predict Now" button
   - View the prediction result in the main panel

## 📊 Project Architecture

### Data Flow

```
Patient Input (Sidebar)
         ↓
Data Validation & Formatting
         ↓
Feature Encoding (One-hot encoding)
         ↓
Data Normalization (Scaler)
         ↓
KNN Model Prediction
         ↓
Result Interpretation & Display
```

### Key Components

**1. Input Processing:**

- Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar
- Resting ECG, Max Heart Rate, Exercise Angina, Oldpeak, ST Slope
- All inputs are validated and formatted according to training data specifications

**2. Feature Engineering:**

- Categorical variables (Sex, Chest Pain Type, etc.) are one-hot encoded
- Missing columns are filled with 0
- Features are reordered to match training data structure

**3. Data Normalization:**

- Input features are scaled using the pre-trained scaler
- Ensures consistency with training data normalization

**4. Prediction Model:**

- K-Nearest Neighbors (KNN) algorithm
- Trained on standardized heart disease dataset
- Binary classification: 0 (Low Risk) or 1 (High Risk)

**5. Output Display:**

- Color-coded results (Green for Low Risk, Red for High Risk)
- Summary of input parameters for user verification

### Model Details

- **Algorithm:** K-Nearest Neighbors (KNN)
- **Input Features:** 11 health metrics
- **Output:** Binary classification (Heart Disease: Yes/No)
- **Model File:** `knn_heart_model.pkl`

### Live Demo

https://abhishekr45--heart-disease-prediction-app-app-3vfev9.streamlit.app/

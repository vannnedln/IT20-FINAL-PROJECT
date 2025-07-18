# 🏥 Stroke Risk Prediction System

A machine learning project to predict stroke risk based on patient health and lifestyle factors. Includes data analysis, model training, evaluation, and an interactive web application for real-time predictions.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Limitations](#limitations)
- [License](#license)

## 🎯 Project Overview
This project implements a binary classification system to predict whether a patient is at risk of stroke using commonly available health metrics. The focus is on maximizing recall and F1-score for clinical relevance, using advanced techniques to handle class imbalance.

## ✨ Features
- Data preprocessing with SMOTE and class weights
- Threshold tuning for improved recall
- Logistic Regression and Random Forest models
- Model evaluation with accuracy, precision, recall, F1-score, and ROC-AUC
- Interactive Streamlit web app for predictions
- Visualization scripts for model metrics

## 📊 Dataset
- **Source:** Stroke Prediction Dataset (~5,110 records)
- **Target:** `stroke` (0: No stroke, 1: Stroke)
- **Features:** Demographics, medical history, lifestyle factors

## 🛠️ Installation
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stroke-prediction-system
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Verify dataset**
   Ensure `healthcare-dataset-stroke-data.csv` is in the project root.

## 🚀 Usage
### 1. Run Complete Analysis
```bash
python stroke_prediction_analysis.py
```
- Loads and preprocesses the dataset
- Trains both models with SMOTE and class weights
- Evaluates and prints metrics

### 2. Launch Web Application
```bash
streamlit run streamlit_app.py
```
- Open your browser to `http://localhost:8501`
- Input patient data and get real-time risk predictions

### 3. Visualize Model Metrics
```bash
python single_model_metrics_visualization.py
```
- Shows bar charts of accuracy, precision, recall, F1-score, and ROC-AUC for each model (test set only)

## 📁 Project Structure
```
stroke-prediction-system/
│
├── healthcare-dataset-stroke-data.csv    # Dataset file
├── stroke_prediction_analysis.py         # Main analysis script
├── streamlit_app.py                      # Web application
├── single_model_metrics_visualization.py # Model metrics visualization (test set)
├── requirements.txt                      # Python dependencies
├── README.md                             # Project documentation
```

## 📈 Model Performance (Test Set)
| Metric      | Logistic Regression | Random Forest |
|-------------|--------------------|--------------|
| Accuracy    | 0.56               | 0.78         |
| Precision   | 0.09               | 0.13         |
| Recall      | 0.86               | 0.64         |
| F1-Score    | 0.16               | 0.22         |
| ROC-AUC     | 0.84               | 0.78         |

- **Random Forest**: Best F1-score and accuracy, good recall
- **Logistic Regression**: Highest recall, lower precision and accuracy

## ⚠️ Limitations
- Dataset size and class imbalance
- Model performance may vary on new data
- Not intended for clinical decision-making

## 📄 License
MIT License

---

**Disclaimer**: This tool is for educational and research purposes only. Always consult healthcare professionals for medical decisions. The predictions should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
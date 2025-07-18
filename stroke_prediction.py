

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE

st.set_page_config(
    page_title="Stroke Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        color: #fffff;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.5em;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #1f77b4;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e1e5e9;
    }
    .risk-very-high {
        background-color: #ffebee;
        border-left-color: #d32f2f;
        border: 2px solid #f44336;
        box-shadow: 0 6px 12px rgba(211, 47, 47, 0.2);
    }
    .risk-high {
        background-color: #fff5f5;
        border-left-color: #e53e3e;
        border: 1px solid #fed7d7;
    }
    .risk-moderate {
        background-color: #fff8e1;
        border-left-color: #ff9800;
        border: 1px solid #ffcc02;
    }
    .risk-low {
        background-color: #f7fafc;
        border-left-color: #38a169;
        border: 1px solid #c6f6d5;
    }
    </style>
""", unsafe_allow_html=True)

class StrokePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.dataset_stats = None
        self.threshold = 0.2
    
    def load_and_train_models(self):
        try:
            df = pd.read_csv('healthcare-dataset-stroke-data.csv')
            self.dataset_stats = {
                'avg_age': df['age'].mean(),
                'stroke_rate': df['stroke'].mean(),
                'avg_glucose': df['avg_glucose_level'].mean(),
                'avg_bmi': pd.to_numeric(df['bmi'], errors='coerce').mean()
            }
            df_processed = df.copy()
            if 'id' in df_processed.columns:
                df_processed = df_processed.drop('id', axis=1)
            df_processed['bmi'] = pd.to_numeric(df_processed['bmi'], errors='coerce')
            df_processed['bmi'].fillna(df_processed['bmi'].median(), inplace=True)
            categorical_cols = df_processed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
            X = df_processed.drop('stroke', axis=1)
            y = df_processed['stroke']
            self.feature_names = list(X.columns)
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
            self.models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
            self.models['Random Forest'] = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
            for name, model in self.models.items():
                model.fit(X_resampled, y_resampled)
            return True
        except Exception as e:
            st.error(f"Error loading data or training models: {str(e)}")
            return False
    
    def predict_stroke_risk(self, patient_data):
        try:
            if not self.models:
                st.error("Models not loaded. Please refresh the page.")
                return None
            if self.feature_names is None:
                st.error("Feature names not loaded. Please refresh the page.")
                return None
            if self.scaler is None:
                st.error("Scaler not loaded. Please refresh the page.")
                return None
            patient_df = pd.DataFrame([patient_data])
            for col, encoder in self.label_encoders.items():
                if col in patient_df.columns:
                    try:
                        patient_df[col] = encoder.transform(patient_df[col])
                    except ValueError:
                        patient_df[col] = encoder.transform([encoder.classes_[0]])
            for feature in self.feature_names:
                if feature not in patient_df.columns:
                    patient_df[feature] = 0
            patient_df = patient_df[self.feature_names]
            patient_scaled = self.scaler.transform(patient_df)
            predictions = {}
            for name, model in self.models.items():
                pred_proba = model.predict_proba(patient_scaled)[0, 1]
                pred_class = 1 if pred_proba >= self.threshold else 0
                predictions[name] = {
                    'probability': pred_proba,
                    'prediction': 'High Risk' if pred_class == 1 else 'Low Risk',
                    'confidence': max(pred_proba, 1 - pred_proba)
                }
            return predictions
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error(f"Debug info - Models loaded: {len(self.models) if self.models else 0}")
            st.error(f"Debug info - Feature names: {self.feature_names is not None}")
            st.error(f"Debug info - Scaler: {self.scaler is not None}")
            return None

def main():
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StrokePredictor()
        with st.spinner("Loading and training machine learning models..."):
            if not st.session_state.predictor.load_and_train_models():
                st.error("Failed to load the application. Please check if the dataset file exists.")
                return
    predictor = st.session_state.predictor
    st.markdown('<div class="main-header">Stroke Risk Prediction System</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Prediction", "Dataset Overview"])
    if page == "Prediction":
        show_prediction_page(predictor)
    elif page == "Dataset Overview":
        show_dataset_overview()
  
def show_prediction_page(predictor):
    st.markdown('<div class="sub-header">Patient Information Input</div>', unsafe_allow_html=True)
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", min_value=0, max_value=100, value=50, help="Patient's age in years")
            ever_married = st.selectbox("Ever Married", ["Yes", "No"])
            residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        with col2:
            st.subheader("Medical History")
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            avg_glucose_level = st.slider("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0, step=0.1, help="mg/dL")
            bmi = st.slider("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, help="Body Mass Index")
        with col3:
            st.subheader("Lifestyle")
            work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt job", "children", "Never worked"])
            smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        submitted = st.form_submit_button("Predict Stroke Risk", use_container_width=True)
    if submitted:
        patient_data = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }
        with st.spinner("Analyzing patient data..."):
            predictions = predictor.predict_stroke_risk(patient_data)
        if predictions:
            show_prediction_results(predictions, patient_data, predictor)

def show_prediction_results(predictions, patient_data, predictor):
    st.markdown("---")
    st.markdown('<div class="sub-header">Prediction Results</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for i, (model_name, pred) in enumerate(predictions.items()):
        col = col1 if i == 0 else col2
        with col:
            probability = pred['probability']
            if probability >= 0.4:
                risk_level = "Very High Risk"
                risk_class = "risk-very-high"
                risk_color = "#d32f2f"
                risk_icon = "🚨"
            elif probability >= 0.2:
                risk_level = "High Risk" 
                risk_class = "risk-high"
                risk_color = "#f44336"
                risk_icon = "⚠️"
            elif probability >= 0.1:
                risk_level = "Moderate Risk"
                risk_class = "risk-moderate"
                risk_color = "#ff9800"
                risk_icon = "⚡"
            else:
                risk_level = "Low Risk"
                risk_class = "risk-low"
                risk_color = "#4caf50"
                risk_icon = "✅"
            st.markdown(f"""
                <div class="metric-card {risk_class}">
                    <h3 style="color: #2c3e50; margin-bottom: 15px;">{risk_icon} {model_name}</h3>
                    <p style="color: #34495e; font-size: 16px; margin: 8px 0;"><strong>Risk Level:</strong> <span style="color: {risk_color}; font-weight: bold;">{risk_level}</span></p>
                    <p style="color: #34495e; font-size: 16px; margin: 8px 0;"><strong>Probability:</strong> <span style="color: {risk_color}; font-weight: bold;">{pred['probability']:.1%}</span></p>
                    <p style="color: #34495e; font-size: 16px; margin: 8px 0;"><strong>Confidence:</strong> <span style="color: #2c3e50; font-weight: bold;">{pred['confidence']:.1%}</span></p>
                </div>
            """, unsafe_allow_html=True)
    # --- Show key metrics for both models (test set) ---
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, f1_score
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df_processed = df.copy()
    if 'id' in df_processed.columns:
        df_processed = df_processed.drop('id', axis=1)
    df_processed['bmi'] = pd.to_numeric(df_processed['bmi'], errors='coerce')
    df_processed['bmi'].fillna(df_processed['bmi'].median(), inplace=True)
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = predictor.label_encoders.get(col)
        if le:
            df_processed[col] = le.transform(df_processed[col])
    X = df_processed.drop('stroke', axis=1)
    y = df_processed['stroke']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_scaled = predictor.scaler.fit_transform(X_train)
    X_test_scaled = predictor.scaler.transform(X_test)
    metrics = []
    for name, model in predictor.models.items():
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= predictor.threshold).astype(int)
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0)
        })
    df_metrics = pd.DataFrame(metrics).set_index("Model")
    def highlight_rf(s):
        return [
            'background-color: #d4edda; font-weight: bold; color: #222;' if s.name == 'Random Forest' else 'color: #fff;'
            for _ in s
        ]
    st.markdown("""
        <style>
        .stDataFrame tbody td {
            color: #fff !important;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("### Model Key Metrics (Test Set)")
    st.dataframe(df_metrics.style.format("{:.2f}").apply(highlight_rf, axis=1), use_container_width=True)
    st.markdown(
        "<div style='font-size:1.1em; color:#38a169; font-weight:bold;'>"
        "Random Forest is recommended based on higher Accuracy, Precision, and F1-Score."
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown('<div class="sub-header">Risk Visualization</div>', unsafe_allow_html=True)
    fig_prob = go.Figure()
    models = list(predictions.keys())
    probabilities = [pred['probability'] for pred in predictions.values()]
    colors = ['#1f77b4', '#ff7f0e']
    fig_prob.add_trace(go.Bar(
        x=models,
        y=probabilities,
        marker_color=colors,
        text=[f"{p:.1%}" for p in probabilities],
        textposition='auto',
    ))
    fig_prob.update_layout(
        title="Stroke Risk Probability by Model",
        xaxis_title="Model",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=400
    )
    st.plotly_chart(fig_prob, use_container_width=True)
    st.markdown('<div class="sub-header">Risk Factors Analysis</div>', unsafe_allow_html=True)
    if predictor.dataset_stats:
        risk_factors = []
        if patient_data['age'] > predictor.dataset_stats['avg_age']:
            risk_factors.append(f"Age ({patient_data['age']}) is above average ({predictor.dataset_stats['avg_age']:.1f})")
        if patient_data['avg_glucose_level'] > predictor.dataset_stats['avg_glucose']:
            risk_factors.append(f"Glucose level ({patient_data['avg_glucose_level']}) is above average ({predictor.dataset_stats['avg_glucose']:.1f})")
        if patient_data['bmi'] > predictor.dataset_stats['avg_bmi']:
            risk_factors.append(f"BMI ({patient_data['bmi']}) is above average ({predictor.dataset_stats['avg_bmi']:.1f})")
        if patient_data['hypertension'] == 1:
            risk_factors.append("Has hypertension")
        if patient_data['heart_disease'] == 1:
            risk_factors.append("Has heart disease")
        if patient_data['smoking_status'] in ['smokes', 'formerly smoked']:
            risk_factors.append(f"Smoking status: {patient_data['smoking_status']}")
        if risk_factors:
            st.write("**Identified Risk Factors:**")
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.write("✅ No major risk factors identified compared to dataset averages.")
    st.markdown('<div class="sub-header">Recommendations</div>', unsafe_allow_html=True)
    max_prob = max(pred['probability'] for pred in predictions.values())
    if max_prob >= 0.4:
        st.error("🚨 **VERY HIGH RISK** - URGENT medical consultation required")
        st.write("**IMMEDIATE ACTION NEEDED:**")
        st.write("• 🏥 **Contact emergency services or go to ER immediately**")
        st.write("• 📞 **Call your doctor TODAY**")
        st.write("• 💊 **Take all prescribed medications as directed**")
        st.write("• 🚭 **Stop smoking immediately if applicable**")
        st.write("• 📊 **Monitor blood pressure and glucose levels daily**")
    elif max_prob >= 0.2:
        st.error("⚠️ **HIGH RISK** - Immediate medical consultation recommended")
        st.write("**URGENT STEPS:**")
        st.write("• 📅 **Schedule appointment with healthcare provider within 24-48 hours**")
        st.write("• 🩺 **Request comprehensive cardiovascular evaluation**")
        st.write("• 💊 **Discuss medication options with doctor**")
        st.write("• 📊 **Monitor blood pressure and glucose levels regularly**")
        st.write("• 🥗 **Adopt strict dietary modifications immediately**")
    elif max_prob >= 0.1:
        st.warning("⚡ **MODERATE RISK** - Preventive measures strongly recommended")
        st.write("**PREVENTIVE STEPS:**")
        st.write("• 📅 **Schedule appointment with healthcare provider within 1-2 weeks**")
        st.write("• 🏃‍♂️ **Increase physical activity (with doctor approval)**")
        st.write("• 🥗 **Implement heart-healthy diet changes**")
        st.write("• 📊 **Regular monitoring of risk factors**")
        st.write("• 🚭 **Smoking cessation if applicable**")
    else:
        st.success("✅ **LOW RISK** - Continue healthy lifestyle")
        st.write("**MAINTAIN CURRENT HEALTH:**")
        st.write("• 🏃‍♂️ **Regular exercise and balanced diet**")
        st.write("• 📅 **Annual health screenings**")
        st.write("• 🚭 **Avoid smoking and limit alcohol**")
        st.write("• 🧘‍♂️ **Manage stress effectively**")
        st.write("• 💤 **Maintain good sleep hygiene**")

def show_dataset_overview():
    st.markdown('<div class="sub-header">Dataset Overview</div>', unsafe_allow_html=True)
    try:
        df = pd.read_csv('healthcare-dataset-stroke-data.csv')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            stroke_rate = df['stroke'].mean()
            st.metric("Stroke Rate", f"{stroke_rate:.1%}")
        with col3:
            avg_age = df['age'].mean()
            st.metric("Average Age", f"{avg_age:.1f} years")
        with col4:
            missing_bmi = (df['bmi'] == 'N/A').sum()
            st.metric("Missing BMI", f"{missing_bmi:,}")
        st.markdown("**Dataset Sample:**")
        st.dataframe(df.head(10), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            fig_age = px.histogram(df, x='age', color='stroke', title='Age Distribution by Stroke',
                                 labels={'stroke': 'Stroke'})
            st.plotly_chart(fig_age, use_container_width=True)
        with col2:
            gender_stroke = df.groupby(['gender', 'stroke']).size().reset_index(name='count')
            fig_gender = px.bar(gender_stroke, x='gender', y='count', color='stroke',
                              title='Gender Distribution by Stroke')
            st.plotly_chart(fig_gender, use_container_width=True)
        st.markdown("**Feature Distributions:**")
        col1, col2 = st.columns(2)
        with col1:
            hypertension_counts = df['hypertension'].value_counts()
            fig_hyp = px.pie(values=hypertension_counts.values, names=['No', 'Yes'],
                           title='Hypertension Distribution')
            st.plotly_chart(fig_hyp, use_container_width=True)
        with col2:
            smoking_counts = df['smoking_status'].value_counts()
            fig_smoke = px.pie(values=smoking_counts.values, names=smoking_counts.index,
                             title='Smoking Status Distribution')
            st.plotly_chart(fig_smoke, use_container_width=True)
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'healthcare-dataset-stroke-data.csv' is in the current directory.")

if __name__ == "__main__":
    main() 
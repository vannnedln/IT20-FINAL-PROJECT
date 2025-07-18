import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

THRESHOLD = 0.2
DATA_PATH = 'healthcare-dataset-stroke-data.csv'

df = pd.read_csv(DATA_PATH)
df_processed = df.copy()
if 'id' in df_processed.columns:
    df_processed = df_processed.drop('id', axis=1)
df_processed['bmi'] = pd.to_numeric(df_processed['bmi'], errors='coerce')
df_processed['bmi'].fillna(df_processed['bmi'].median(), inplace=True)

label_encoders = {}
categorical_cols = df_processed.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

X = df_processed.drop('stroke', axis=1)
y = df_processed['stroke']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
}
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

for model_name, model in models.items():
    model.fit(X_resampled, y_resampled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)
    values = [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred, zero_division=0),
        roc_auc_score(y_test, y_proba)
    ]
  
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(metrics, values, color='salmon' if model_name == 'Random Forest' else 'skyblue')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.set_title(f'{model_name} Model Metrics (Test Set)')
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show() 
"""
Save trained Random Forest model for deployment
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
logger.info("Loading data...")
hourly_df = pd.read_csv('data/processed/eicu_hourly_all_features.csv')
outcomes_df = pd.read_csv('data/processed/eicu_outcomes.csv')

# Extract features
feature_cols = [col for col in hourly_df.columns if col not in ['patientunitstayid', 'hour']]

X_features = []
y_labels = []
patient_ids = outcomes_df['patientunitstayid'].values

for patient_id in patient_ids:
    patient_data = hourly_df[hourly_df['patientunitstayid'] == patient_id]

    if len(patient_data) == 0:
        continue

    patient_hourly = patient_data[feature_cols].ffill().bfill()

    if len(patient_hourly) == 0:
        continue

    patient_features = []
    for col in feature_cols:
        values = patient_hourly[col].values
        values = values[~np.isnan(values)]

        if len(values) > 0:
            patient_features.extend([
                np.mean(values),
                np.std(values) if len(values) > 1 else 0,
                np.min(values),
                np.max(values),
                np.max(values) - np.min(values),
            ])
        else:
            patient_features.extend([0, 0, 0, 0, 0])

    X_features.append(patient_features)

    patient_outcome = outcomes_df[outcomes_df['patientunitstayid'] == patient_id]
    if len(patient_outcome) > 0:
        y_labels.append(patient_outcome['mortality'].values[0])

X = np.array(X_features)
y = np.array(y_labels)

logger.info(f"Data prepared: X={X.shape}, y={y.shape}")

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
logger.info("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200, max_depth=15, random_state=42,
    class_weight='balanced', n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# Evaluate
from sklearn.metrics import roc_auc_score, f1_score
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)

logger.info(f"Test Results: AUC={auc:.4f}, F1={f1:.4f}")

# Save model and scaler
output_dir = Path('results/dl_models')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open(output_dir / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

logger.info(f"Model and scaler saved to {output_dir}")
logger.info("Ready for deployment!")

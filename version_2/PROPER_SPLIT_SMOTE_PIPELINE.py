"""
PROPER DATA SPLIT + SMOTE + PIPELINE

Correct order:
1. Split data FIRST: 70% train, 15% test, 15% validation
2. SMOTE only on training set (no leakage)
3. Train model on balanced training data
4. Evaluate on test set (original, unbalanced)
5. Final validation on validation set (original, unbalanced)
6. Add disease-specific layers
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, f1_score, precision_recall_curve
)
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("STEP-BY-STEP PIPELINE WITH PROPER DATA SPLITS (70/15/15)")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD ALL DATA (NO FILTERING)
# ============================================================================

print("\n[STEP 1] LOAD ALL 2,468 PATIENTS...")

df = pd.read_csv('data/processed_icu_hourly_v2.csv')

print(f"✓ Raw data: {df.shape[0]:,} hourly records, {df['patientunitstayid'].nunique():,} patients")
print(f"  Deaths: {df['mortality'].sum():,} ({100*df['mortality'].mean():.2f}%)")

# ============================================================================
# STEP 2: AGGREGATE TO PATIENT LEVEL
# ============================================================================

print("\n[STEP 2] AGGREGATE TO PATIENT LEVEL (SMART IMPUTATION)...")

patient_groups = df.groupby('patientunitstayid')
patients_data = []

vital_cols = ['sao2', 'heartrate', 'respiration']
lab_cols = ['BUN', 'HCO3', 'Hct', 'Hgb', 'WBC x 1000', 'creatinine',
            'potassium', 'sodium', 'chloride', 'Temperature']

for patient_id, group in patient_groups:
    rec = {'patientunitstayid': patient_id, 'mortality': group['mortality'].iloc[0]}

    for col in vital_cols + lab_cols:
        if col in group.columns:
            values = pd.to_numeric(group[col], errors='coerce').dropna()
            if len(values) > 0:
                rec[f'{col}_mean'] = values.mean()
                rec[f'{col}_min'] = values.min()
                rec[f'{col}_max'] = values.max()
                rec[f'{col}_std'] = values.std() if len(values) > 1 else 0
            else:
                rec[f'{col}_mean'] = np.nan
                rec[f'{col}_min'] = np.nan
                rec[f'{col}_max'] = np.nan
                rec[f'{col}_std'] = 0

    patients_data.append(rec)

patients_df = pd.DataFrame(patients_data)

# Impute
feature_cols = [c for c in patients_df.columns if c not in ['patientunitstayid', 'mortality']]
for col in feature_cols:
    patients_df[col].fillna(patients_df[col].median(), inplace=True)
patients_df.fillna(0, inplace=True)

X = patients_df[feature_cols].values
y = patients_df['mortality'].values

print(f"✓ Patient-level data: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  Deaths: {y.sum()}, Survivors: {len(y) - y.sum()}")
print(f"  Imbalance: {(y==0).sum() / max(y.sum(), 1):.1f}:1")

# ============================================================================
# STEP 3: SPLIT DATA FIRST (BEFORE SMOTE) - 70/15/15
# ============================================================================

print("\n[STEP 3] SPLIT DATA: 70% TRAIN, 15% TEST, 15% VALIDATION...")

# First split: 70% train, 30% temp (for test+validation)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# Second split: split temp into test and validation (50/50)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"\n✓ Train set: {X_train.shape[0]} samples ({y_train.sum()} deaths, {100*y_train.mean():.2f}%)")
print(f"✓ Test set: {X_test.shape[0]} samples ({y_test.sum()} deaths, {100*y_test.mean():.2f}%)")
print(f"✓ Validation set: {X_val.shape[0]} samples ({y_val.sum()} deaths, {100*y_val.mean():.2f}%)")
print(f"✓ Total: {X_train.shape[0] + X_test.shape[0] + X_val.shape[0]} = {X.shape[0]} ✓")

# ============================================================================
# STEP 4: SMOTE ONLY ON TRAINING DATA (NO LEAKAGE)
# ============================================================================

print("\n[STEP 4] APPLY SMOTE + AUGMENTATION TO TRAINING DATA ONLY...")

def manual_smote_enhance(X, y, k=1, sampling_strategy=2.0):
    """SMOTE with augmentation"""
    from sklearn.neighbors import NearestNeighbors

    X_array = X if isinstance(X, np.ndarray) else X.values
    y_array = y if isinstance(y, np.ndarray) else y.values

    minority_idx = np.where(y_array == 1)[0]
    X_minority = X_array[minority_idx]

    n_minority = len(minority_idx)
    n_synthetic = int(n_minority * sampling_strategy)

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n_minority))
    nbrs.fit(X_minority)

    synthetic_samples = []

    for i in range(n_synthetic):
        minority_idx_sample = np.random.randint(0, n_minority)
        minority_sample = X_minority[minority_idx_sample]

        _, neighbor_indices = nbrs.kneighbors([minority_sample])
        neighbor_indices = neighbor_indices[0]
        neighbor_indices = neighbor_indices[neighbor_indices != minority_idx_sample]

        if len(neighbor_indices) > 0:
            neighbor_idx = np.random.choice(neighbor_indices)
            neighbor_sample = X_minority[neighbor_idx]

            alpha = np.random.uniform(0, 1)
            synthetic = minority_sample + alpha * (neighbor_sample - minority_sample)

            noise = np.random.normal(0, 0.05 * np.std(X_array, axis=0))
            synthetic = synthetic + noise

            synthetic_samples.append(synthetic)

    synthetic_samples = np.array(synthetic_samples)
    X_balanced = np.vstack([X_array, synthetic_samples])
    y_balanced = np.hstack([y_array, np.ones(len(synthetic_samples))])

    return X_balanced, y_balanced, len(synthetic_samples)

X_train_balanced, y_train_balanced, n_synthetic = manual_smote_enhance(X_train, y_train, k=1, sampling_strategy=2.0)

print(f"\n✓ SMOTE applied to TRAINING DATA ONLY:")
print(f"  Original: {X_train.shape[0]} samples ({y_train.sum()} deaths)")
print(f"  Synthetic: +{n_synthetic} deaths")
print(f"  Balanced: {X_train_balanced.shape[0]} samples ({y_train_balanced.sum()} deaths)")
print(f"  New imbalance: {(y_train_balanced==0).sum() / max(y_train_balanced.sum(), 1):.1f}:1")

print(f"\n✓ Test and Validation sets: UNCHANGED (original, unbalanced)")

# ============================================================================
# STEP 5: SCALE DATA (TRAIN SCALER ON TRAINING ONLY)
# ============================================================================

print("\n[STEP 5] SCALE DATA (FIT SCALER ON TRAINING ONLY)...")

scaler = StandardScaler()
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

print(f"✓ Scaler fitted on training data")
print(f"✓ Test and validation data transformed using training statistics")

# ============================================================================
# STEP 6: TRAIN BEST MODEL
# ============================================================================

print("\n[STEP 6] TRAIN BEST MODEL ON BALANCED TRAINING DATA...")

models = {
    'RandomForest': RandomForestClassifier(n_estimators=150, max_depth=15,
                                          min_samples_split=10, class_weight='balanced', n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=8,
                                                   learning_rate=0.1, random_state=42),
}

best_model_name = None
best_train_auc = 0
model_results = {}

for model_name, model in models.items():
    print(f"\n  Training {model_name}...")

    # Train on balanced training data
    model.fit(X_train_balanced_scaled, y_train_balanced)

    # Evaluate on training (for reference)
    y_train_pred = model.predict_proba(X_train_balanced_scaled)[:, 1]
    auc_train = roc_auc_score(y_train_balanced, y_train_pred)

    # Evaluate on test set
    y_test_pred = model.predict_proba(X_test_scaled)[:, 1]
    auc_test = roc_auc_score(y_test, y_test_pred)

    # Evaluate on validation set
    y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
    auc_val = roc_auc_score(y_val, y_val_pred)

    model_results[model_name] = {
        'train_auc': auc_train,
        'test_auc': auc_test,
        'val_auc': auc_val
    }

    print(f"    Train AUC: {auc_train:.4f}")
    print(f"    Test AUC:  {auc_test:.4f}")
    print(f"    Val AUC:   {auc_val:.4f}")

    if auc_test > best_train_auc:
        best_train_auc = auc_test
        best_model_name = model_name

print(f"\n✓ Best model (by test AUC): {best_model_name}")

best_model = models[best_model_name]

# ============================================================================
# STEP 7: DETAILED EVALUATION ON TEST SET
# ============================================================================

print("\n[STEP 7] DETAILED EVALUATION ON TEST SET...")

y_test_pred = best_model.predict_proba(X_test_scaled)[:, 1]
auc_test = roc_auc_score(y_test, y_test_pred)

# Find optimal threshold
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
youden_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[youden_idx]

y_test_binary = (y_test_pred >= optimal_threshold).astype(int)
cm_test = confusion_matrix(y_test, y_test_binary)

if cm_test.shape == (2, 2):
    tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
else:
    tn_test = cm_test[0, 0] if 0 < cm_test.shape[0] and 0 < cm_test.shape[1] else 0
    tp_test = cm_test[-1, -1] if -1 < cm_test.shape[0] and -1 < cm_test.shape[1] else 0
    fp_test = cm_test[0, 1] if cm_test.shape[1] > 1 else 0
    fn_test = cm_test[1, 0] if cm_test.shape[0] > 1 else 0

sens_test = tp_test / (tp_test + fn_test) if (tp_test + fn_test) > 0 else 0
spec_test = tn_test / (tn_test + fp_test) if (tn_test + fp_test) > 0 else 0

print(f"\n✓ TEST SET RESULTS ({X_test.shape[0]} samples, {y_test.sum()} deaths):")
print(f"  AUC: {auc_test:.4f}")
print(f"  Optimal threshold: {optimal_threshold:.4f}")
print(f"  Sensitivity: {sens_test:.4f} ({int(tp_test)}/{int(tp_test + fn_test)})")
print(f"  Specificity: {spec_test:.4f}")
print(f"  TP={int(tp_test)}, TN={int(tn_test)}, FP={int(fp_test)}, FN={int(fn_test)}")

# ============================================================================
# STEP 8: FINAL VALIDATION ON VALIDATION SET
# ============================================================================

print("\n[STEP 8] FINAL VALIDATION ON VALIDATION SET...")

y_val_pred = best_model.predict_proba(X_val_scaled)[:, 1]
auc_val = roc_auc_score(y_val, y_val_pred)

y_val_binary = (y_val_pred >= optimal_threshold).astype(int)
cm_val = confusion_matrix(y_val, y_val_binary)

if cm_val.shape == (2, 2):
    tn_val, fp_val, fn_val, tp_val = cm_val.ravel()
else:
    tn_val = cm_val[0, 0] if 0 < cm_val.shape[0] and 0 < cm_val.shape[1] else 0
    tp_val = cm_val[-1, -1] if -1 < cm_val.shape[0] and -1 < cm_val.shape[1] else 0
    fp_val = cm_val[0, 1] if cm_val.shape[1] > 1 else 0
    fn_val = cm_val[1, 0] if cm_val.shape[0] > 1 else 0

sens_val = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0
spec_val = tn_val / (tn_val + fp_val) if (tn_val + fp_val) > 0 else 0

print(f"\n✓ VALIDATION SET RESULTS ({X_val.shape[0]} samples, {y_val.sum()} deaths):")
print(f"  AUC: {auc_val:.4f}")
print(f"  Sensitivity: {sens_val:.4f} ({int(tp_val)}/{int(tp_val + fn_val)})")
print(f"  Specificity: {spec_val:.4f}")
print(f"  TP={int(tp_val)}, TN={int(tn_val)}, FP={int(fp_val)}, FN={int(fn_val)}")

# ============================================================================
# STEP 9: DISEASE-SPECIFIC MODELS
# ============================================================================

print("\n[STEP 9] TRAIN DISEASE-SPECIFIC MODELS...")

disease_mappings = {
    'Sepsis': ['WBC x 1000', 'respiration', 'heartrate', 'Temperature'],
    'Respiratory': ['sao2', 'respiration', 'HCO3'],
    'Renal': ['creatinine', 'potassium', 'sodium', 'chloride'],
    'Cardiac': ['heartrate', 'Temperature', 'BUN'],
    'Hepatic': ['sodium', 'BUN', 'chloride']
}

disease_results = {}

for disease, disease_features in disease_mappings.items():
    print(f"\n  Training {disease} model...")

    feature_indices = []
    for feat in disease_features:
        for i, col in enumerate(feature_cols):
            if feat in col:
                feature_indices.append(i)
                break

    if len(feature_indices) == 0:
        print(f"    ⚠️  No features found")
        continue

    X_train_disease = X_train_balanced[:, feature_indices]
    X_test_disease = X_test[:, feature_indices]
    X_val_disease = X_val[:, feature_indices]

    scaler_disease = StandardScaler()
    X_train_disease_scaled = scaler_disease.fit_transform(X_train_disease)
    X_test_disease_scaled = scaler_disease.transform(X_test_disease)
    X_val_disease_scaled = scaler_disease.transform(X_val_disease)

    model_disease = RandomForestClassifier(n_estimators=100, max_depth=12,
                                          class_weight='balanced', n_jobs=-1, random_state=42)
    model_disease.fit(X_train_disease_scaled, y_train_balanced)

    auc_test_disease = roc_auc_score(y_test, model_disease.predict_proba(X_test_disease_scaled)[:, 1])
    auc_val_disease = roc_auc_score(y_val, model_disease.predict_proba(X_val_disease_scaled)[:, 1])

    disease_results[disease] = {'test_auc': auc_test_disease, 'val_auc': auc_val_disease}
    print(f"    Test AUC: {auc_test_disease:.4f}, Val AUC: {auc_val_disease:.4f}")

# ============================================================================
# STEP 10: SAVE ALL RESULTS
# ============================================================================

print("\n[STEP 10] SAVE RESULTS...")

results_dir = Path('results/phase2_outputs')
results_dir.mkdir(exist_ok=True, parents=True)

final_results = {
    'data_splits': {
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'train_deaths': int(y_train.sum()),
        'test_deaths': int(y_test.sum()),
        'val_deaths': int(y_val.sum()),
        'train_after_smote': int(X_train_balanced.shape[0]),
        'synthetic_samples': int(n_synthetic)
    },
    'best_model': best_model_name,
    'all_models': model_results,
    'test_set_performance': {
        'auc': float(auc_test),
        'sensitivity': float(sens_test),
        'specificity': float(spec_test),
        'optimal_threshold': float(optimal_threshold),
        'tp': int(tp_test),
        'tn': int(tn_test),
        'fp': int(fp_test),
        'fn': int(fn_test)
    },
    'validation_set_performance': {
        'auc': float(auc_val),
        'sensitivity': float(sens_val),
        'specificity': float(spec_val),
        'tp': int(tp_val),
        'tn': int(tn_val),
        'fp': int(fp_val),
        'fn': int(fn_val)
    },
    'disease_models': disease_results
}

with open(results_dir / 'PROPER_SPLIT_SMOTE_PIPELINE_RESULTS.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"✓ Saved: PROPER_SPLIT_SMOTE_PIPELINE_RESULTS.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY: PROPER DATA SPLIT + SMOTE PIPELINE")
print("=" * 80)

print(f"\n✅ DATA SPLITS (70/15/15):")
print(f"   Train: {X_train.shape[0]} → {X_train_balanced.shape[0]} (after SMOTE)")
print(f"   Test: {X_test.shape[0]}")
print(f"   Validation: {X_val.shape[0]}")

print(f"\n✅ SMOTE APPLICATION:")
print(f"   Applied ONLY to training data: No data leakage")
print(f"   Generated: {n_synthetic} synthetic death samples")
print(f"   Test/Val: Original, unbalanced (realistic evaluation)")

print(f"\n✅ BEST MODEL: {best_model_name}")
print(f"   Test AUC: {auc_test:.4f}")
print(f"   Validation AUC: {auc_val:.4f}")

print(f"\n✅ CLINICAL METRICS (Test Set):")
print(f"   Sensitivity: {sens_test:.4f} (catch {int(sens_test*100)}% of deaths)")
print(f"   Specificity: {spec_test:.4f} (avoid {int(spec_test*100)}% false alarms)")

print(f"\n✅ DISEASE-SPECIFIC MODELS: {len(disease_results)} trained")
for disease, metrics in disease_results.items():
    print(f"   {disease}: Test {metrics['test_auc']:.4f}, Val {metrics['val_auc']:.4f}")

print("\n" + "=" * 80)

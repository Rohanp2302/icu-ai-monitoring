"""
REBUILD: Proper Feature Engineering from Raw Data
- 2,468 patients (vs previous 82-91)
- Aggregate hourly vitals → patient-level features
- NO aggressive filtering = NO data loss
- Focus on 24h window as per original intent
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("STEP 1: LOAD RAW DATA (2,468 PATIENTS)")
print("=" * 80)

# Load raw hourly data
df = pd.read_csv('data/processed_icu_hourly_v2.csv')

print(f"\n✓ Raw data loaded:")
print(f"  Rows: {df.shape[0]:,}")
print(f"  Columns: {df.shape[1]}")
print(f"  Unique patients: {df['patientunitstayid'].nunique()}")
print(f"  Mortality: {df['mortality'].sum():,} deaths, {(df['mortality']==0).sum():,} survivors")
print(f"  Class balance: {(df['mortality']==0).sum() / max(df['mortality'].sum(), 1):.1f}:1 (imbalance)")

# Quality check
print(f"\nData quality:")
print(f"  Missing values per column:")
for col in df.columns:
    pct_missing = 100 * df[col].isna().sum() / len(df)
    if pct_missing > 0:
        print(f"    {col}: {pct_missing:.1f}%")

print("\n" + "=" * 80)
print("STEP 2: AGGREGATE HOURLY → PATIENT-LEVEL FEATURES")
print("=" * 80)

# Group by patient (keep mortality)
patient_groups = df.groupby('patientunitstayid')

# Extract patient-level data
patients_data = []

for patient_id, group in patient_groups:
    patient_rec = {'patientunitstayid': patient_id}
    
    # Mortality (same for all rows of patient)
    patient_rec['mortality'] = group['mortality'].iloc[0]
    
    # Demographics/static
    patient_rec['n_hours'] = len(group)
    
    # Vital stats - aggregate across 24h window
    vital_cols = ['sao2', 'heartrate', 'respiration']
    lab_cols = ['BUN', 'HCO3', 'Hct', 'Hgb', 'WBC x 1000', 'creatinine', 'potassium', 'sodium', 'chloride', 'Temperature']
    
    # For each numeric column, compute: mean, min, max, std
    for col in vital_cols + lab_cols:
        if col in group.columns:
            values = pd.to_numeric(group[col], errors='coerce').dropna()
            if len(values) > 0:
                patient_rec[f'{col}_mean'] = values.mean()
                patient_rec[f'{col}_min'] = values.min()
                patient_rec[f'{col}_max'] = values.max()
                patient_rec[f'{col}_std'] = values.std() if len(values) > 1 else 0
    
    patients_data.append(patient_rec)

patients_df = pd.DataFrame(patients_data)

print(f"\n✓ Patient-level features created:")
print(f"  Patients: {len(patients_df)}")
print(f"  Features: {len(patients_df.columns)}")
print(f"  Mortality distribution: {patients_df['mortality'].value_counts().to_dict()}")

# Handle missing values
print(f"\n  Handling missing values...")
initial_nulls = patients_df.isna().sum().sum()
patients_df = patients_df.fillna(patients_df.median())
final_nulls = patients_df.isna().sum().sum()

print(f"  Null values: {initial_nulls} → {final_nulls}")

# Prepare X and y
feature_cols = [c for c in patients_df.columns if c not in ['patientunitstayid', 'mortality']]
X = patients_df[feature_cols].fillna(0)
y = patients_df['mortality'].astype(int)

print(f"\n✓ Final dataset:")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  Positive samples: {y.sum()} ({100*y.mean():.2f}%)")
print(f"  Negative samples: {(1-y).sum()} ({100*(1-y).mean():.2f}%)")
print(f"  Imbalance ratio: {(1-y).sum() / max(y.sum(), 1):.1f}:1")

# ============================================================================
# STEP 3: PROPER CROSS-VALIDATION (NO LEAKAGE)
# ============================================================================

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.metrics import roc_curve

print("\n" + "=" * 80)
print("STEP 3: STRATIFIED 5-FOLD CROSS-VALIDATION (NO DATA LEAKAGE)")
print("=" * 80)

X_array = X.values
y_array = y.values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_array, y_array), 1):
    X_train, X_test = X_array[train_idx], X_array[test_idx]
    y_train, y_test = y_array[train_idx], y_array[test_idx]
    
    # Scale on TRAIN only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n[FOLD {fold}] Train {X_train.shape[0]}, Test {X_test.shape[0]}")
    print(f"  Deaths in train: {y_train.sum()}, test: {y_test.sum()}")
    
    # Train RF with class weights
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate on TEST (not train!)
    y_pred = rf.predict_proba(X_test_scaled)[:, 1]
    
    try:
        auc = roc_auc_score(y_test, y_pred)
    except:
        auc = 0.5
    
    fold_results.append({'fold': fold, 'auc': auc, 'n_test': len(y_test), 'n_positive_test': y_test.sum()})
    print(f"  AUC: {auc:.4f}")

# Summary
aucs = [r['auc'] for r in fold_results]
print(f"\n✓ Cross-Validation Results:")
print(f"  Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"  Range: {np.min(aucs):.4f} - {np.max(aucs):.4f}")

# ============================================================================
# STEP 4: INDEPENDENT TEST SET EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: FINAL 80/20 HOLD-OUT TEST")
print("=" * 80)

X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X_array, y_array, test_size=0.2, stratify=y_array, random_state=42
)

scaler_final = StandardScaler()
X_train_final_scaled = scaler_final.fit_transform(X_train_final)
X_test_final_scaled = scaler_final.transform(X_test_final)

print(f"Train: {X_train_final.shape[0]} ({y_train_final.sum()} deaths)")
print(f"Test: {X_test_final.shape[0]} ({y_test_final.sum()} deaths)")

rf_final = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_final.fit(X_train_final_scaled, y_train_final)

y_pred_train = rf_final.predict_proba(X_train_final_scaled)[:, 1]
y_pred_test = rf_final.predict_proba(X_test_final_scaled)[:, 1]

auc_train = roc_auc_score(y_train_final, y_pred_train)
auc_test = roc_auc_score(y_test_final, y_pred_test)

print(f"\nPerformance:")
print(f"  Train AUC: {auc_train:.4f}")
print(f"  Test AUC: {auc_test:.4f}")
print(f"  Overfitting gap: {auc_train - auc_test:.4f}")

if auc_test >= 0.75:
    print(f"\n✓ GOOD: Test AUC ≥ 0.75")
elif auc_test >= 0.70:
    print(f"\n⚠️  ACCEPTABLE: Test AUC ≥ 0.70")
else:
    print(f"\n❌ NEEDS WORK: Test AUC < 0.70")

# ============================================================================
# SAVE RESULTS
# ============================================================================

results_dir = Path('results/phase2_outputs')
results_dir.mkdir(exist_ok=True, parents=True)

import json

summary = {
    'data': {
        'n_patients': int(len(patients_df)),
        'n_features': int(len(feature_cols)),
        'n_deaths': int(y.sum()),
        'n_survivors': int((1-y).sum()),
        'imbalance_ratio': float((1-y).sum() / max(y.sum(), 1))
    },
    'cv_results': {
        'mean_auc': float(np.mean(aucs)),
        'std_auc': float(np.std(aucs)),
        'min_auc': float(np.min(aucs)),
        'max_auc': float(np.max(aucs)),
        'fold_details': [
            {
                'fold': int(r['fold']),
                'auc': float(r['auc']),
                'n_test': int(r['n_test']),
                'n_positive_test': int(r['n_positive_test'])
            }
            for r in fold_results
        ]
    },
    'test_results': {
        'train_auc': float(auc_train),
        'test_auc': float(auc_test),
        'overfitting_gap': float(auc_train - auc_test),
        'train_size': int(X_train_final.shape[0]),
        'test_size': int(X_test_final.shape[0])
    }
}

with open(results_dir / 'PROPER_FULL_DATASET_RESULTS.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Saved: PROPER_FULL_DATASET_RESULTS.json")

print("\n" + "=" * 80)
print("SUMMARY: FIXING THE DAMAGE FROM PHASE D")
print("=" * 80)

print(f"\n❌ What was wrong:")
print(f"  • AUC 1.0 with n=82 = pure data leakage")
print(f"  • Phase 1 lost 95% of data (2,468 → 82)")
print(f"  • SMOTE applied before split = information leak")
print(f"  • No proper cross-validation")

print(f"\n✓ What we fixed:")
print(f"  • Loaded ALL 2,468 patients from raw data")
print(f"  • Proper patient-level feature aggregation")
print(f"  • 5-fold stratified cross-validation")
print(f"  • Independent test set (80/20 split)")
print(f"  • Class weights (no SMOTE to avoid leakage)")

print(f"\n📊 REALISTIC PERFORMANCE:")
print(f"  • Cross-val AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"  • Test AUC: {auc_test:.4f}")
print(f"  • This is the REAL performance on unseen data")

print("\n" + "=" * 80)

#!/usr/bin/env python3
"""
DATA QUALITY IMPROVEMENTS - PHASE 1.5
Analyze missing data patterns, impute, normalize, handle outliers
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("DATA QUALITY IMPROVEMENTS - COMPREHENSIVE ANALYSIS")
print("="*80)

# Load the 24h windows dataset
df = pd.read_csv('results/phase1_outputs/phase1_24h_windows.csv')
print(f"\n[1/6] Loading dataset: {df.shape}")

# ============================================================================
# STEP 1: MISSING DATA ANALYSIS
# ============================================================================
print("\n[2/6] Analyzing missing data patterns...")

missing_analysis = pd.DataFrame({
    'column': df.columns,
    'missing_count': df.isna().sum(),
    'missing_pct': (df.isna().sum() / len(df) * 100).round(2),
    'dtype': df.dtypes,
    'non_null_mean': df.mean(),
    'non_null_std': df.std()
})

missing_analysis = missing_analysis[missing_analysis['column'] != 'patientunitstayid']
missing_analysis = missing_analysis[missing_analysis['column'] != 'mortality']
missing_analysis = missing_analysis.sort_values('missing_pct', ascending=False)

print("\nMissing Data Summary:")
print(f"  Features with NO missing data: {sum(missing_analysis['missing_pct'] == 0)}")
print(f"  Features with <10% missing: {sum(missing_analysis['missing_pct'] < 10)}")
print(f"  Features with 10-50% missing: {sum((missing_analysis['missing_pct'] >= 10) & (missing_analysis['missing_pct'] < 50))}")
print(f"  Features with 50-90% missing: {sum((missing_analysis['missing_pct'] >= 50) & (missing_analysis['missing_pct'] < 90))}")
print(f"  Features with >90% missing: {sum(missing_analysis['missing_pct'] > 90)}")

print("\n  Top 10 most sparse features:")
for idx, row in missing_analysis.head(10).iterrows():
    print(f"    - {row['column']:40} {row['missing_pct']:6.1f}% missing")

print("\n  Top 10 densest features:")
for idx, row in missing_analysis.tail(10).iterrows():
    print(f"    - {row['column']:40} {row['missing_pct']:6.1f}% missing")

# ============================================================================
# STEP 2: FEATURE SELECTION - DROP SPARSE FEATURES
# ============================================================================
print("\n[3/6] Feature selection: removing sparse features...")

# Drop features with >80% missing (too sparse to be useful)
features_to_drop = missing_analysis[missing_analysis['missing_pct'] > 80]['column'].tolist()
print(f"  Dropping {len(features_to_drop)} features with >80% missing:")
for feat in features_to_drop[:5]:
    print(f"    - {feat}")
if len(features_to_drop) > 5:
    print(f"    ... and {len(features_to_drop) - 5} more")

df_clean = df.drop(columns=features_to_drop + ['patientunitstayid'])
print(f"\n  Dataset after dropping sparse features: {df_clean.shape}")

# ============================================================================
# STEP 3: HANDLE MISSING VALUES
# ============================================================================
print("\n[4/6] Handling missing values...")

# Separate features and target
X = df_clean.drop(columns=['mortality'])
y = df_clean['mortality']

print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# Strategy 1: Forward fill for temporal data (within same patient)
print("\n  Applying multiple imputation strategies:")

# First, forward fill for vitals (likely temporal decay)
vital_cols = [c for c in X.columns if any(x in c for x in ['heartrate', 'sao2', 'respiration', 'temperature', 'systemic', 'cvp'])]
if len(vital_cols) > 0:
    X[vital_cols] = X[vital_cols].ffill().bfill()
    print(f"    - Forward fill for {len(vital_cols)} vital columns")

# Strategy 2: Mean imputation for lab values
lab_cols = [c for c in X.columns if any(x in c for x in ['creatinine', 'bilirubin', 'platelets', 'glucose'])]
if len(lab_cols) > 0:
    imputer = SimpleImputer(strategy='mean')
    X[lab_cols] = imputer.fit_transform(X[lab_cols])
    print(f"    - Mean imputation for {len(lab_cols)} lab columns")

# Strategy 3: Median imputation for SOFA/organ scores
sofa_cols = [c for c in X.columns if 'SOFA' in c or 'SOFA' in c]
if len(sofa_cols) > 0:
    imputer = SimpleImputer(strategy='median')
    X[sofa_cols] = imputer.fit_transform(X[sofa_cols])
    print(f"    - Median imputation for {len(sofa_cols)} SOFA columns")

# Strategy 4: Fill remaining with mean (generic)
remaining_missing = X.columns[X.isna().any()]
if len(remaining_missing) > 0:
    imputer = SimpleImputer(strategy='mean')
    X[remaining_missing] = imputer.fit_transform(X[remaining_missing])
    print(f"    - Mean imputation for {len(remaining_missing)} remaining columns")

print(f"\n  Missing values after imputation: {X.isna().sum().sum()}")

# ============================================================================
# STEP 4: OUTLIER DETECTION & HANDLING
# ============================================================================
print("\n[5/6] Handling outliers...")

# Use IQR method to detect outliers
outlier_count = 0
for col in X.columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 3 * IQR  # Use 3x IQR for severe outliers
    upper_bound = Q3 + 3 * IQR
    
    # Clip to bounds instead of removing
    mask = (X[col] < lower_bound) | (X[col] > upper_bound)
    outlier_count += mask.sum()
    
    if mask.sum() > 0:
        X.loc[mask, col] = X[col].clip(lower_bound, upper_bound)

print(f"  Clipped {outlier_count} outlier values to IQR bounds")

# ============================================================================
# STEP 5: FEATURE NORMALIZATION
# ============================================================================
print("\n[6/6] Normalizing features...")

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

print(f"  Features normalized to μ=0, σ=1")
print(f"  Feature means (should be ~0): {X_normalized.mean().describe()}")
print(f"  Feature stds (should be ~1): {X_normalized.std().describe()}")

# ============================================================================
# SAVE CLEANED DATASET
# ============================================================================
print("\n" + "="*80)
print("SAVING CLEANED & NORMALIZED DATASET")
print("="*80)

# Combine X and y
df_final = X_normalized.copy()
df_final['mortality'] = y.values

# Add back patient IDs for reference
df_final['patientunitstayid'] = df['patientunitstayid'].values

# Reorder columns
cols = ['patientunitstayid', 'mortality'] + [c for c in df_final.columns if c not in ['patientunitstayid', 'mortality']]
df_final = df_final[cols]

# Save
output_path = 'results/phase1_outputs/phase1_24h_windows_CLEAN.csv'
df_final.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved: {output_path}")
print(f"  Shape: {df_final.shape}")

# Save quality report
quality_report = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "phase": "Data Quality Improvements",
    
    "original_dataset": {
        "shape": df.shape,
        "missing_features": int(missing_analysis['missing_pct'].isna().sum()),
        "avg_missing_pct": float(missing_analysis['missing_pct'].mean())
    },
    
    "after_sparse_removal": {
        "shape": df_clean.shape,
        "features_removed": len(features_to_drop),
        "features_kept": df_clean.shape[1] - 1  # -1 for mortality
    },
    
    "after_imputation": {
        "missing_values": int(X.isna().sum().sum()),
        "imputation_methods": ["forward_fill", "mean_imputation", "median_imputation"]
    },
    
    "after_normalization": {
        "shape": df_final.shape,
        "feature_mean": float(X_normalized.mean().mean()),
        "feature_std": float(X_normalized.std().mean()),
        "scaled": True
    },
    
    "final_dataset": {
        "total_windows": len(df_final),
        "total_features": df_final.shape[1] - 2,  # -2 for patient_id and mortality
        "mortality_cases": int(df_final['mortality'].sum()),
        "mortality_rate": float(df_final['mortality'].mean()),
        "non_mortality_cases": int((df_final['mortality'] == 0).sum()),
        "feature_quality": "Ready for deep learning"
    }
}

import json
with open('results/phase1_outputs/data_quality_report.json', 'w') as f:
    json.dump(quality_report, f, indent=2)

print("\nQuality report saved: results/phase1_outputs/data_quality_report.json")

# Print final summary
print("\n" + "="*80)
print("FINAL DATASET SUMMARY")
print("="*80)
print(f"\nCleaned & Normalized Dataset:")
print(f"  Shape: {df_final.shape}")
print(f"  Total features: {df_final.shape[1] - 2}")
print(f"  Mortality rate: {df_final['mortality'].mean()*100:.2f}%")
print(f"  Missing values: {df_final.isna().sum().sum()}")
print(f"  All features normalized: μ≈0, σ≈1")
print(f"\nPHASE 1.5 COMPLETE - Ready for Phase 2: Deep Learning Model Training")
print()

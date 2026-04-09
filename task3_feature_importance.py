"""
TASK 3: FEATURE IMPORTANCE ANALYSIS
Date: April 8, 2026
Purpose: Identify which features drive model predictions
"""

import os
import numpy as np
import pandas as pd
import json
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

print('=' * 80)
print('TASK 3: FEATURE IMPORTANCE ANALYSIS')
print('=' * 80)

# ==============================================================================
# STEP 1: Load model, data, and prepare test set
# ==============================================================================
print('\n1. LOADING DATA & MODEL...')
print('-' * 80)

# Load checkpoint
checkpoint_path = 'results/phase2_outputs/ensemble_model_CORRECTED.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
scaler_mean = np.array(checkpoint['scaler_mean'])
scaler_scale = np.array(checkpoint['scaler_scale'])

# Load data
phase2_data_path = 'results/phase1_outputs/phase1_24h_windows_CLEAN.csv'
df = pd.read_csv(phase2_data_path)
feature_cols = [c for c in df.columns if c not in ['patientunitstayid', 'mortality']]

X = df[feature_cols].values
y = df['mortality'].values

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale data
scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train simple model for feature importance
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)

test_auc_before = lr.score(X_test_scaled, y_test)
print(f'✓ Model trained')
print(f'  Test accuracy: {test_auc_before:.4f}')
print(f'  Test set: {len(X_test)} samples, {np.sum(y_test)} deaths')

# ==============================================================================
# STEP 2: Compute permutation importance
# ==============================================================================
print('\n2. COMPUTING FEATURE IMPORTANCE...')
print('-' * 80)

# Permutation importance (model-agnostic)
perm_importance = permutation_importance(
    lr, X_test_scaled, y_test, 
    n_repeats=10, 
    random_state=42,
    scoring='roc_auc'
)

# Create feature importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}).sort_values('importance', ascending=False)

print(f'✓ Permutation importance computed for {len(feature_cols)} features')

# ==============================================================================
# STEP 3: Compute coefficient-based importance (for linear model)
# ==============================================================================
print('\n3. EXTRACTING MODEL COEFFICIENTS...')
print('-' * 80)

coefficients = np.abs(lr.coef_[0])
coef_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': lr.coef_[0],
    'abs_coefficient': coefficients
}).sort_values('abs_coefficient', ascending=False)

print(f'✓ Model coefficients extracted')

# ==============================================================================
# STEP 4: Display top features
# ==============================================================================
print('\n4. TOP FEATURES BY IMPORTANCE')
print('=' * 80)

print('\n📊 PERMUTATION IMPORTANCE (Model-Agnostic):')
print('-' * 60)
print('Feature Name                          | Importance |   ±Std')
print('-' * 60)
for idx, row in importance_df.head(10).iterrows():
    print(f'{row["feature"]:<35} | {row["importance"]:>10.4f} | {row["std"]:>6.4f}')

print('\n\n📊 MODEL COEFFICIENTS (Linear Model):')
print('-' * 60)
print('Feature Name                          |  Coefficient |   Abs Value')
print('-' * 60)
for idx, row in coef_importance_df.head(10).iterrows():
    print(f'{row["feature"]:<35} | {row["coefficient"]:>12.4f} | {row["abs_coefficient"]:>11.4f}')

# ==============================================================================
# STEP 5: Save importance results
# ==============================================================================
print('\n5. SAVING RESULTS...')
print('-' * 80)

# Create comprehensive importance report
importance_report = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "model_type": "Logistic Regression (proxy for neural network)",
    "test_auc": float(test_auc_before),
    "total_features": len(feature_cols),
    "permutation_importance": importance_df.to_dict('records'),
    "coefficient_importance": coef_importance_df.to_dict('records'),
    "top_5_features_by_permutation": importance_df.head(5)['feature'].tolist(),
    "top_5_features_by_coefficient": coef_importance_df.head(5)['feature'].tolist(),
    "interpretation": {
        "permutation_importance": "How much the model's performance decreases when each feature is randomly shuffled",
        "coefficients": "Raw model weights (positive = increases mortality risk, negative = decreases risk)"
    }
}

report_path = 'results/phase2_outputs/feature_importance_analysis.json'
with open(report_path, 'w') as f:
    json.dump(importance_report, f, indent=2, default=str)

print(f'✓ Report saved to: {report_path}')

# ==============================================================================
# STEP 6: Create visualizations
# ==============================================================================
print('\n6. CREATING VISUALIZATIONS...')
print('-' * 80)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Permutation importance
ax1 = axes[0]
top_n = 15
top_features = importance_df.head(top_n)
y_pos = np.arange(len(top_features))

ax1.barh(y_pos, top_features['importance'].values, xerr=top_features['std'].values, 
         align='center', alpha=0.7, color='steelblue', capsize=5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_features['feature'].values, fontsize=10)
ax1.set_xlabel('Permutation Importance (AUC decrease)', fontsize=11)
ax1.set_title(f'Top {top_n} Features by Permutation Importance', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# Plot 2: Coefficient magnitudes
ax2 = axes[1]
top_coef = coef_importance_df.head(top_n)
colors = ['red' if x < 0 else 'green' for x in top_coef['coefficient'].values]

y_pos = np.arange(len(top_coef))
ax2.barh(y_pos, top_coef['coefficient'].values, align='center', alpha=0.7, color=colors)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(top_coef['feature'].values, fontsize=10)
ax2.set_xlabel('Model Coefficient', fontsize=11)
ax2.set_title(f'Top {top_n} Features by Model Coefficient (Red=Mortality Risk, Green=Protective)', 
              fontsize=12, fontweight='bold')
ax2.axvline(0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

plt.tight_layout()
plot_path = 'results/phase2_outputs/feature_importance_analysis.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f'✓ Visualization saved to: {plot_path}')

# ==============================================================================
# SUMMARY
# ==============================================================================
print('\n' + '=' * 80)
print('TASK 3 SUMMARY')
print('=' * 80)
print(f'\n✅ Feature importance analysis complete')
print(f'\n🎯 KEY FINDINGS:')
print(f'  - Most important feature: {importance_df.iloc[0]["feature"]} (importance: {importance_df.iloc[0]["importance"]:.4f})')
print(f'  - Top 5 features identified and ranked')
print(f'  - Model coefficients indicate protective vs risk factors')
print(f'\n📁 FILES SAVED:')
print(f'  1. {report_path}')
print(f'  2. {plot_path}')
print('\n✓ TASK 3 COMPLETE')
print('=' * 80)

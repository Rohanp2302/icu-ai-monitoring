"""
TASK 5: ERROR ANALYSIS
Date: April 8, 2026
Purpose: Understand model mistakes - false positives and false negatives
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
from sklearn.metrics import confusion_matrix

print('=' * 80)
print('TASK 5: ERROR ANALYSIS')
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

# Train model
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)

# Get predictions
y_pred = lr.predict(X_test_scaled)
y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]

print(f'✓ Model trained')
print(f'  Test set: {len(X_test)} samples, {np.sum(y_test)} deaths')

# ==============================================================================
# STEP 2: Analyze prediction errors
# ==============================================================================
print('\n2. ANALYZING PREDICTION ERRORS...')
print('-' * 80)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f'✓ Confusion Matrix:')
print(f'  True Negatives (TN):  {tn}')
print(f'  False Positives (FP): {fp}')
print(f'  False Negatives (FN): {fn}')
print(f'  True Positives (TP):  {tp}')

# Error types
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0  # Clinical: dangerous
false_alarm_rate = fp / (tn + fp) if (tn + fp) > 0 else 0

print(f'\n✓ Error Rates:')
print(f'  Sensitivity (catch deaths):  {sensitivity:.4f}')
print(f'  Miss Rate (miss deaths):     {miss_rate:.4f} ⚠️  CRITICAL')
print(f'  False Alarm Rate:            {false_alarm_rate:.4f}')

# ==============================================================================
# STEP 3: Identify misclassified samples
# ==============================================================================
print('\n3. IDENTIFYING MISCLASSIFIED SAMPLES...')
print('-' * 80)

# False Negatives (most critical)
fn_mask = (y_test == 1) & (y_pred == 0)
fn_indices = np.where(fn_mask)[0]

print(f'\n📍 FALSE NEGATIVES (Model missed {len(fn_indices)} deaths):')
if len(fn_indices) > 0:
    for i, idx in enumerate(fn_indices):
        print(f'  FN {i+1}:')
        print(f'    - Predicted probability: {y_pred_proba[idx]:.4f}')
        print(f'    - Actual: Death (1)')
        print(f'    - Top 3 feature values:')
        
        # Get scaled feature values
        sample_scaled = X_test_scaled[idx]
        
        # Sort by absolute deviation from mean (std)
        abs_devs = np.abs(sample_scaled)
        top_3_indices = np.argsort(-abs_devs)[:3]
        
        for feat_idx in top_3_indices:
            feat_name = feature_cols[feat_idx]
            feat_value = X_test[idx, feat_idx]
            scaled_value = sample_scaled[feat_idx]
            print(f'      * {feat_name}: {feat_value:.3f} (scaled: {scaled_value:.3f})')
else:
    print('  ✓ No false negatives! (Perfect recall)')

# False Positives (less critical but increases false alarms)
fp_mask = (y_test == 0) & (y_pred == 1)
fp_indices = np.where(fp_mask)[0]

print(f'\n📍 FALSE POSITIVES (Model incorrectly predicted {len(fp_indices)} deaths):')
if len(fp_indices) > 0:
    for i, idx in enumerate(fp_indices[:3]):  # Show first 3
        print(f'  FP {i+1}:')
        print(f'    - Predicted probability: {y_pred_proba[idx]:.4f}')
        print(f'    - Actual: Survived (0)')
        print(f'    - Top 3 risk factors:')
        
        sample_scaled = X_test_scaled[idx]
        abs_devs = np.abs(sample_scaled)
        top_3_indices = np.argsort(-abs_devs)[:3]
        
        for feat_idx in top_3_indices:
            feat_name = feature_cols[feat_idx]
            feat_value = X_test[idx, feat_idx]
            scaled_value = sample_scaled[feat_idx]
            print(f'      * {feat_name}: {feat_value:.3f} (scaled: {scaled_value:.3f})')
    
    if len(fp_indices) > 3:
        print(f'  ... and {len(fp_indices) - 3} more false positives')
else:
    print('  ✓ No false positives! (Perfect precision)')

# ==============================================================================
# STEP 4: Feature analysis for errors
# ==============================================================================
print('\n4. FEATURE ANALYSIS FOR ERRORS...')
print('-' * 80)

if len(fn_indices) > 0:
    fn_features = X_test[fn_indices]
    fn_mean = fn_features.mean(axis=0)
    test_mean = X_test.mean(axis=0)
    
    # Compare FN cases to overall test set
    feature_diffs = fn_mean - test_mean
    feature_diffs_std = feature_diffs / (X_test.std(axis=0) + 1e-8)
    
    print(f'\nFeature patterns in FALSE NEGATIVES vs overall test set:')
    sorted_indices = np.argsort(-np.abs(feature_diffs_std))[:5]
    for j, feat_idx in enumerate(sorted_indices):
        print(f'  {j+1}. {feature_cols[feat_idx]:40} diff={feature_diffs[feat_idx]:+7.3f} std')

# ==============================================================================
# STEP 5: Create error distribution visualization
# ==============================================================================
print('\n5. CREATING ERROR VISUALIZATIONS...')
print('-' * 80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Confusion Matrix
ax1 = axes[0, 0]
cm_array = np.array([[tn, fp], [fn, tp]])
im1 = ax1.imshow(cm_array, cmap='Blues', aspect='auto')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Predicted Negative', 'Predicted Positive'])
ax1.set_yticklabels(['Actual Negative', 'Actual Positive'])
ax1.set_xlabel('Predicted Label', fontsize=11)
ax1.set_ylabel('True Label', fontsize=11)
ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

for i in range(2):
    for j in range(2):
        text = ax1.text(j, i, cm_array[i, j], ha="center", va="center", 
                       color="white" if cm_array[i, j] > cm_array.max()/2 else "black",
                       fontsize=14, fontweight='bold')

plt.colorbar(im1, ax=ax1)

# Plot 2: Prediction probability distribution by true label
ax2 = axes[0, 1]
ax2.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.6, label='Actual Negatives', color='blue', edgecolor='black')
ax2.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.6, label='Actual Deaths', color='red', edgecolor='black')
ax2.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Decision Threshold')
ax2.set_xlabel('Predicted Probability', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Probability Distribution by True Label', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3, axis='y')

# Plot 3: Error type breakdown
ax3 = axes[1, 0]
error_categories = ['True\nNegatives', 'False\nPositives', 'False\nNegatives', 'True\nPositives']
error_counts = [tn, fp, fn, tp]
colors_errors = ['green', 'orange', 'red', 'darkgreen']
bars = ax3.bar(error_categories, error_counts, color=colors_errors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('Confusion Matrix Breakdown', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3, axis='y')

for bar, count in zip(bars, error_counts):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: Metrics comparison
ax4 = axes[1, 1]
metrics = ['Sensitivity\n(Catch Deaths)', 'Miss Rate\n(Dangerous)', 'Specificity\n(No False Alarms)', 'False Alarm\nRate']
values = [sensitivity, miss_rate, specificity, false_alarm_rate]
colors_metrics = ['green', 'red', 'blue', 'orange']
bars = ax4.barh(metrics, values, color=colors_metrics, alpha=0.7, edgecolor='black')
ax4.set_xlabel('Rate', fontsize=11)
ax4.set_title('Key Performance Metrics', fontsize=12, fontweight='bold')
ax4.set_xlim([0, 1])
ax4.grid(alpha=0.3, axis='x')

for bar, value in zip(bars, values):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2.,
            f' {value:.1%}', ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plot_path = 'results/phase2_outputs/error_analysis.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f'✓ Error analysis visualization saved to: {plot_path}')

# ==============================================================================
# STEP 6: Save error analysis report
# ==============================================================================
print('\n6. SAVING ERROR ANALYSIS REPORT...')
print('-' * 80)

error_analysis = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "test_set_size": len(X_test),
    "confusion_matrix": {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    },
    "metrics": {
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "miss_rate": float(miss_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0
    },
    "error_analysis": {
        "false_negatives_count": int(len(fn_indices)),
        "false_negatives_critical": "YES - Misses detecting deaths" if fn > 0 else "NO",
        "false_positives_count": int(len(fp_indices)),
        "false_positives_issue": "May cause unnecessary interventions"
    },
    "clinical_interpretation": [
        f"The model correctly identifies {sensitivity*100:.1f}% of ICU deaths",
        f"The model misses {miss_rate*100:.1f}% of deaths (DANGEROUS)",
        f"False alarm rate is {false_alarm_rate*100:.1f}% (acceptable if low)",
        "Focus on reducing false negatives for patient safety"
    ]
}

report_path = 'results/phase2_outputs/error_analysis.json'
with open(report_path, 'w') as f:
    json.dump(error_analysis, f, indent=2)

print(f'✓ Error analysis report saved to: {report_path}')

# ==============================================================================
# SUMMARY
# ==============================================================================
print('\n' + '=' * 80)
print('TASK 5 SUMMARY')
print('=' * 80)
print(f'\n✅ Error analysis complete')
print(f'\n🎯 KEY FINDINGS:')
print(f'  - Sensitivity: {sensitivity*100:.1f}% (catches deaths)')
print(f'  - Miss Rate: {miss_rate*100:.1f}% ⚠️  (CRITICAL - misses deaths)')
print(f'  - False Alarm Rate: {false_alarm_rate*100:.1f}%')
print(f'  - False Negatives: {fn} cases')
print(f'  - False Positives: {fp} cases')
print(f'\n📁 FILES SAVED:')
print(f'  1. {report_path}')
print(f'  2. {plot_path}')
print('\n✓ TASK 5 COMPLETE')
print('=' * 80)

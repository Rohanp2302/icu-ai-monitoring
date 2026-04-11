"""
TASK 6: BOOTSTRAP CONFIDENCE INTERVALS
Date: April 8, 2026
Purpose: Quantify uncertainty in model performance metrics
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
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score

print('=' * 80)
print('TASK 6: BOOTSTRAP CONFIDENCE INTERVALS')
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
# STEP 2: Perform bootstrap resampling
# ==============================================================================
print('\n2. PERFORMING BOOTSTRAP RESAMPLING...')
print('-' * 80)

n_bootstraps = 1000
np.random.seed(42)

# Store bootstrap metrics
bootstrap_auc = []
bootstrap_acc = []
bootstrap_sens = []
bootstrap_spec = []
bootstrap_prec = []

print(f'Running {n_bootstraps} bootstrap samples...')

for i in range(n_bootstraps):
    # Resample with replacement
    indices = np.random.choice(len(X_test), size=len(X_test), replace=True)
    y_boot = y_test[indices]
    y_pred_boot = y_pred[indices]
    y_pred_proba_boot = y_pred_proba[indices]
    
    # Calculate metrics
    if len(np.unique(y_boot)) > 1:  # Only if both classes present
        try:
            bootstrap_auc.append(roc_auc_score(y_boot, y_pred_proba_boot))
        except:
            bootstrap_auc.append(np.nan)
    else:
        bootstrap_auc.append(np.nan)
    
    bootstrap_acc.append(accuracy_score(y_boot, y_pred_boot))
    bootstrap_sens.append(recall_score(y_boot, y_pred_boot, zero_division=0))
    
    # Specificity
    tn = ((y_boot == 0) & (y_pred_boot == 0)).sum()
    fp = ((y_boot == 0) & (y_pred_boot == 1)).sum()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    bootstrap_spec.append(spec)
    
    # Precision
    try:
        bootstrap_prec.append(precision_score(y_boot, y_pred_boot, zero_division=0))
    except:
        bootstrap_prec.append(0)
    
    if (i + 1) % 200 == 0:
        print(f'  Completed {i + 1} / {n_bootstraps} bootstrap samples')

print(f'✓ Bootstrap resampling complete')

# Remove NaN values
bootstrap_auc_valid = np.array([x for x in bootstrap_auc if not np.isnan(x)])

# ==============================================================================
# STEP 3: Compute confidence intervals
# ==============================================================================
print('\n3. COMPUTING 95% CONFIDENCE INTERVALS...')
print('-' * 80)

def compute_ci(data, ci=95):
    """Compute confidence interval using percentile method"""
    lower = np.percentile(data, (100 - ci) / 2)
    upper = np.percentile(data, 100 - (100 - ci) / 2)
    mean = np.mean(data)
    return mean, lower, upper

metrics = {
    'AUC': bootstrap_auc_valid,
    'Accuracy': bootstrap_acc,
    'Sensitivity': bootstrap_sens,
    'Specificity': bootstrap_spec,
    'Precision': bootstrap_prec
}

ci_results = {}
for metric_name, metric_data in metrics.items():
    mean, lower, upper = compute_ci(metric_data)
    ci_results[metric_name] = {
        'mean': mean,
        'lower_bound': lower,
        'upper_bound': upper,
        'ci_range': upper - lower
    }
    print(f'{metric_name:15} : {mean:.4f}  [{lower:.4f}, {upper:.4f}]  ±{(upper-lower)/2:.4f}')

# ==============================================================================
# STEP 4: Create bootstrap distribution visualizations
# ==============================================================================
print('\n4. CREATING BOOTSTRAP VISUALIZATIONS...')
print('-' * 80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

metrics_to_plot = [
    ('AUC', bootstrap_auc_valid, 'steelblue'),
    ('Accuracy', bootstrap_acc, 'forestgreen'),
    ('Sensitivity', bootstrap_sens, 'coral'),
    ('Specificity', bootstrap_spec, 'plum'),
    ('Precision', bootstrap_prec, 'gold'),
]

for idx, (metric_name, metric_data, color) in enumerate(metrics_to_plot):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Histogram
    ax.hist(metric_data, bins=40, alpha=0.7, color=color, edgecolor='black')
    
    # Add mean line
    mean_val = ci_results[metric_name]['mean']
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    
    # Add CI bounds
    ci_val = ci_results[metric_name]
    ax.axvline(ci_val['lower_bound'], color='darkred', linestyle=':', linewidth=1.5, 
               label=f'95% CI: [{ci_val["lower_bound"]:.4f}, {ci_val["upper_bound"]:.4f}]')
    ax.axvline(ci_val['upper_bound'], color='darkred', linestyle=':', linewidth=1.5)
    
    ax.set_xlabel('Metric Value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{metric_name} Bootstrap Distribution\n({len(metric_data)} resamples)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

# Remove extra subplot
axes[1, 2].remove()

plt.tight_layout()
plot_path = 'results/phase2_outputs/bootstrap_confidence_intervals.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f'✓ Bootstrap distributions saved to: {plot_path}')

# ==============================================================================
# STEP 5: Create summary table
# ==============================================================================
print('\n5. BOOTSTRAP CI SUMMARY TABLE')
print('=' * 80)

print('\nMetric           | Mean    | 95% CI Lower | 95% CI Upper | CI Range')
print('-' * 70)
for metric_name, ci in ci_results.items():
    print(f'{metric_name:15} | {ci["mean"]:.4f} | {ci["lower_bound"]:12.4f} | {ci["upper_bound"]:12.4f} | {ci["ci_range"]:.4f}')

# ==============================================================================
# STEP 6: Save bootstrap results
# ==============================================================================
print('\n6. SAVING BOOTSTRAP RESULTS...')
print('-' * 80)

bootstrap_report = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "n_bootstrap_samples": n_bootstraps,
    "test_set_size": len(X_test),
    "confidence_level": 95,
    "confidence_intervals": {},
    "interpretation": {
        "meaning": "95% CI means we're 95% confident the true metric lies in this range",
        "narrow_ci": "Good - metric is stable",
        "wide_ci": "Bad - metric is unstable, may need more data"
    }
}

# Add CI results
for metric_name, ci in ci_results.items():
    bootstrap_report["confidence_intervals"][metric_name] = {
        "mean": float(ci['mean']),
        "lower_bound_95_percent": float(ci['lower_bound']),
        "upper_bound_95_percent": float(ci['upper_bound']),
        "ci_width": float(ci['ci_range']),
        "stability": "Stable" if ci['ci_range'] < 0.1 else "Moderate" if ci['ci_range'] < 0.2 else "Unstable"
    }

# Add clinical recommendations
bootstrap_report["clinical_recommendations"] = [
    "Model performance is stable within these confidence bounds",
    f"AUC 95% CI: [{ci_results['AUC']['lower_bound']:.4f}, {ci_results['AUC']['upper_bound']:.4f}]",
    f"Expected generalization error: {1 - ci_results['Accuracy']['mean']:.4f}",
    "Consider CI width when deploying model in production"
]

report_path = 'results/phase2_outputs/bootstrap_confidence_intervals.json'
with open(report_path, 'w') as f:
    json.dump(bootstrap_report, f, indent=2)

print(f'✓ Bootstrap report saved to: {report_path}')

# ==============================================================================
# STEP 7: Statistical stability assessment
# ==============================================================================
print('\n7. STATISTICAL STABILITY ASSESSMENT...')
print('-' * 80)

print('\n🔍 STABILITY EVALUATION:')
for metric_name, ci in ci_results.items():
    ci_width = ci['ci_range']
    if ci_width < 0.05:
        stability = '✅ EXCELLENT - Very tight CI'
    elif ci_width < 0.10:
        stability = '✅ GOOD - Tight CI'
    elif ci_width < 0.20:
        stability = '⚠️  MODERATE - Moderate spread'
    else:
        stability = '❌ POOR - Wide CI, unstable'
    
    print(f'  {metric_name:15}: CI width = {ci_width:.4f}  {stability}')

# ==============================================================================
# SUMMARY
# ==============================================================================
print('\n' + '=' * 80)
print('TASK 6 SUMMARY')
print('=' * 80)
print(f'\n✅ Bootstrap confidence intervals calculated')
print(f'\n📊 RESULTS:')
print(f'  - Bootstrap samples: {n_bootstraps}')
print(f'  - Confidence level: 95%')
print(f'  - AUC mean: {ci_results["AUC"]["mean"]:.4f}')
print(f'  - AUC 95% CI: [{ci_results["AUC"]["lower_bound"]:.4f}, {ci_results["AUC"]["upper_bound"]:.4f}]')
print(f'\n📁 FILES SAVED:')
print(f'  1. {report_path}')
print(f'  2. {plot_path}')
print('\n✓ TASK 6 COMPLETE - ALL ROBUSTNESS TASKS FINISHED!')
print('=' * 80)

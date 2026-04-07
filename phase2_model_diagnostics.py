"""
PHASE 2 - MODEL DIAGNOSTICS & COMPREHENSIVE ANALYSIS
Generates detailed performance metrics WITHOUT plotting to save disk space
"""

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, f1_score, brier_score_loss, log_loss,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score
)
from sklearn.preprocessing import StandardScaler
import json
import os

print("="*80)
print("PHASE 2 - COMPREHENSIVE MODEL DIAGNOSTICS")
print("="*80)

# [1/7] Load data
print("\n[1/7] Loading data...")
df = pd.read_csv('results/phase1_outputs/phase1_24h_windows_CLEAN.csv')
X = df.drop(['patientunitstayid', 'mortality'], axis=1).values
y = df['mortality'].values

print(f"  Dataset shape: {X.shape}")
print(f"  Mortality rate: {y.mean():.2%}")
print(f"  Positive samples: {y.sum()}, Negative samples: {(1-y).sum()}")

# [2/7] Load trained model
print("\n[2/7] Loading trained model...")
from phase2_ensemble_model import MultiArchitectureEnsemble

model = MultiArchitectureEnsemble()
checkpoint = torch.load('results/phase2_outputs/ensemble_model.pth')

# Handle wrapped checkpoint
if isinstance(checkpoint, dict):
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        print(f"  Model loaded from wrapped checkpoint")
    else:
        model.load_state_dict(checkpoint)
        print(f"  Model loaded from state dict")
else:
    model.load_state_dict(checkpoint)
    print(f"  Model loaded")

model.eval()

print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# [3/7] Generate predictions on test set
print("\n[3/7] Generating predictions...")

# Split data same way as training
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to torch
X_test_torch = torch.from_numpy(X_test_scaled).float().unsqueeze(1)

# Get predictions
with torch.no_grad():
    logits = model(X_test_torch)
    probs = torch.sigmoid(logits).numpy().flatten()
    preds = (probs >= 0.5).astype(int)

print(f"  Predictions generated: {len(probs)} samples")
print(f"  Predicted positives: {preds.sum()}")
print(f"  Actual positives: {y_test.sum()}")

# [4/7] Compute comprehensive metrics
print("\n[4/7] Computing metrics...")

metrics = {
    'overall': {},
    'by_threshold': {},
    'robustness': {}
}

# Basic metrics
metrics['overall']['auc_roc'] = float(roc_auc_score(y_test, probs))
fpr, tpr, thresholds = roc_curve(y_test, probs)

# Precision-recall
precision, recall, pr_thresholds = precision_recall_curve(y_test, probs)
metrics['overall']['pr_auc'] = float(auc(recall, precision))

# At threshold 0.5
cm = confusion_matrix(y_test, preds)
tn, fp, fn, tp = cm.ravel()
metrics['overall']['tp'] = int(tp)
metrics['overall']['fp'] = int(fp)
metrics['overall']['tn'] = int(tn)
metrics['overall']['fn'] = int(fn)

# Performance metrics
metrics['overall']['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
metrics['overall']['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
metrics['overall']['precision_value'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0
metrics['overall']['npv'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0
metrics['overall']['f1_score'] = float(f1_score(y_test, preds))
metrics['overall']['brier_score'] = float(brier_score_loss(y_test, probs))
metrics['overall']['log_loss'] = float(log_loss(y_test, probs))
metrics['overall']['matthews_cc'] = float(matthews_corrcoef(y_test, preds))
metrics['overall']['cohens_kappa'] = float(cohen_kappa_score(y_test, preds))

print(f"  AUC-ROC: {metrics['overall']['auc_roc']:.4f}")
print(f"  PR-AUC: {metrics['overall']['pr_auc']:.4f}")
print(f"  F1 Score: {metrics['overall']['f1_score']:.4f}")
print(f"  Sensitivity: {metrics['overall']['sensitivity']:.4f}")
print(f"  Specificity: {metrics['overall']['specificity']:.4f}")

# [5/7] Error analysis
print("\n[5/7] Analyzing errors...")

errors = {
    'false_positives': {},
    'false_negatives': {},
    'correct_predictions': {}
}

# False positives
fp_mask = (preds == 1) & (y_test == 0)
fp_indices = np.where(fp_mask)[0]
if len(fp_indices) > 0:
    fp_probs = probs[fp_indices]
    errors['false_positives']['count'] = int(len(fp_indices))
    errors['false_positives']['mean_prob'] = float(fp_probs.mean())

# False negatives
fn_mask = (preds == 0) & (y_test == 1)
fn_indices = np.where(fn_mask)[0]
if len(fn_indices) > 0:
    fn_probs = probs[fn_indices]
    errors['false_negatives']['count'] = int(len(fn_indices))
    errors['false_negatives']['mean_prob'] = float(fn_probs.mean())

# Correct predictions
correct_mask = preds == y_test
errors['correct_predictions']['count'] = int(correct_mask.sum())

print(f"  False Positives: {errors['false_positives'].get('count', 0)}")
print(f"  False Negatives: {errors['false_negatives'].get('count', 0)}")
print(f"  Correct Predictions: {errors['correct_predictions']['count']}")

# [6/7] Threshold analysis
print("\n[6/7] Threshold analysis...")

threshold_metrics = []
for thresh in np.arange(0.1, 0.95, 0.05):
    preds_thresh = (probs >= thresh).astype(int)
    if preds_thresh.sum() == 0 or (1 - preds_thresh).sum() == 0:
        continue
    
    try:
        cm_thresh = confusion_matrix(y_test, preds_thresh, labels=[0, 1])
        if cm_thresh.shape == (2, 2):
            tn, fp, fn, tp = cm_thresh.ravel()
        else:
            continue
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0
        
        threshold_metrics.append({
            'threshold': float(thresh),
            'sensitivity': float(sens),
            'specificity': float(spec),
            'precision': float(prec),
            'f1': float(f1),
            'predicted_positives': int(preds_thresh.sum())
        })
    except:
        continue

metrics['by_threshold'] = sorted(threshold_metrics, key=lambda x: x['f1'], reverse=True)[:5]

print(f"  Best thresholds (by F1):")
for i, m in enumerate(metrics['by_threshold'][:3]):
    print(f"    {i+1}. Threshold {m['threshold']:.2f}: F1={m['f1']:.4f}, Sens={m['sensitivity']:.4f}, Spec={m['specificity']:.4f}")

# [7/7] Robustness metrics
print("\n[7/7] Computing robustness metrics...")

metrics['robustness']['prob_mean'] = float(probs.mean())
metrics['robustness']['prob_std'] = float(probs.std())

# Score distribution
metrics['robustness']['score_distribution'] = {
    'positives': {
        'mean': float(probs[y_test == 1].mean()),
        'std': float(probs[y_test == 1].std())
    },
    'negatives': {
        'mean': float(probs[y_test == 0].mean()),
        'std': float(probs[y_test == 0].std())
    }
}

score_sep = abs(metrics['robustness']['score_distribution']['positives']['mean'] - 
                metrics['robustness']['score_distribution']['negatives']['mean'])

print(f"  Probability mean: {metrics['robustness']['prob_mean']:.4f}")
print(f"  Probability std: {metrics['robustness']['prob_std']:.4f}")
print(f"  Score separation: {score_sep:.4f}")

# Save outputs
print("\n[SAVING] Generating outputs...")
os.makedirs('results/phase2_outputs', exist_ok=True)

with open('results/phase2_outputs/diagnostics_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  ✓ Saved: diagnostics_metrics.json")

with open('results/phase2_outputs/error_analysis.json', 'w') as f:
    json.dump(errors, f, indent=2)
print(f"  ✓ Saved: error_analysis.json")

# Summary report
print("\n" + "="*80)
print("PHASE 2 DIAGNOSTICS SUMMARY")
print("="*80)
print(f"""
PERFORMANCE METRICS:
  AUC-ROC:                {metrics['overall']['auc_roc']:.4f}
  PR-AUC:                 {metrics['overall']['pr_auc']:.4f}
  F1 Score:               {metrics['overall']['f1_score']:.4f}
  
CLINICAL METRICS:
  Sensitivity (Recall):   {metrics['overall']['sensitivity']:.4f} (catches {metrics['overall']['tp']} of {metrics['overall']['tp']+metrics['overall']['fn']} deaths)
  Specificity:            {metrics['overall']['specificity']:.4f}
  Precision:              {metrics['overall']['precision_value']:.4f}
  NPV:                    {metrics['overall']['npv']:.4f}

CALIBRATION:
  Brier Score:            {metrics['overall']['brier_score']:.4f} (lower=better)
  Log Loss:               {metrics['overall']['log_loss']:.4f}

ROBUSTNESS:
  Matthews CC:            {metrics['overall']['matthews_cc']:.4f}
  Cohen's Kappa:          {metrics['overall']['cohens_kappa']:.4f}

ERROR ANALYSIS:
  False Positives:        {errors['false_positives'].get('count', 0)} (want low)
  False Negatives:        {errors['false_negatives'].get('count', 0)} (want very low)
  Correct Predictions:    {errors['correct_predictions']['count']}

SCORE SEPARATION:
  Positive Class μ:       {metrics['robustness']['score_distribution']['positives']['mean']:.4f}
  Negative Class μ:       {metrics['robustness']['score_distribution']['negatives']['mean']:.4f}
  Separation (|Δμ|):      {score_sep:.4f}
""")

print("="*80)
print("✅ PHASE 2 DIAGNOSTICS COMPLETE")
print("="*80)

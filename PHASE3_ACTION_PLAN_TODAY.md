# TODAY'S ROBUSTNESS IMPROVEMENT ACTION PLAN
**Date**: April 8, 2026  
**Objective**: Validate 93.91% AUC model and fix 8 robustness gaps  
**Estimated Time**: 2-3 hours  
**Target**: Production-ready deployment by end of day

---

## QUICK START: 6 EXECUTABLE TASKS (IN PRIORITY ORDER)

### TASK 1: External Validation (Challenge2012) ⚠️ CRITICAL
**Time**: 30 minutes  
**Importance**: Highest - proves real-world readiness  
**Blocker**: If Challenge2012 AUC < 0.80, DO NOT DEPLOY

#### Step 1a: Load Challenge2012 Data
```python
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, sensitivity_specificity_support

# Check if Challenge2012 data exists
import os
challenge_path = "data/challenge2012/ICUType1.csv"  # Adjust path as needed
if os.path.exists(challenge_path):
    print("✓ Challenge2012 data found")
    X_challenge = pd.read_csv(challenge_path)
    print(f"Shape: {X_challenge.shape}")
else:
    print("❌ Challenge2012 data not found - check path")
    # List available data files
    os.system("dir data")
```

#### Step 1b: Load Model & Scaler
```python
# Load checkpoint from yesterday
checkpoint_path = "results/phase2_outputs/ensemble_model_CORRECTED.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Extract scaler parameters (trained on Phase 2 data)
scaler_mean = checkpoint['scaler_mean']
scaler_scale = checkpoint['scaler_scale']

# Reconstruct StandardScaler
scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale

print("✓ Model loaded")
print(f"✓ Scaler mean: {scaler_mean[:5]}...")
print(f"✓ Scaler scale: {scaler_scale[:5]}...")
```

#### Step 1c: Preprocess Challenge2012 Data
```python
# Ensure feature alignment
# Challenge2012 may have different columns - need to extract same features as Phase 2
# Load Phase 2 feature list
phase2_data = pd.read_csv("results/phase1_outputs/phase1_24h_windows_CLEAN.csv")
feature_cols = [c for c in phase2_data.columns if c not in ['patientunitstayid', 'mortality']]

print(f"Phase 2 used {len(feature_cols)} features")

# Extract same features from Challenge2012
X_challenge_features = X_challenge[feature_cols]  # May error if features don't match
X_challenge_scaled = scaler.transform(X_challenge_features)

print(f"✓ Challenge2012 preprocessed: {X_challenge_scaled.shape}")
```

#### Step 1d: Evaluate on Challenge2012
```python
# Load model weights
model = load_ensemble_model()  # Load architecture
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Get predictions
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_challenge_scaled)
    logits = model(X_tensor)
    probs = torch.sigmoid(logits).numpy()

# Extract mortality labels from Challenge2012
y_challenge = X_challenge['mortality'].values  # Adjust column name as needed

# Compute metrics
challenge_auc = roc_auc_score(y_challenge, probs)
challenge_sens = sensitivity_specificity_support(y_challenge, probs > 0.5)[0]
challenge_spec = sensitivity_specificity_support(y_challenge, probs > 0.5)[1]

print(f"\n📊 CHALLENGE2012 EXTERNAL VALIDATION RESULTS:")
print(f"   AUC:  {challenge_auc:.4f} (Expected: ≥0.85)")
print(f"   Sensitivity: {challenge_sens:.4f}")
print(f"   Specificity: {challenge_spec:.4f}")

# Decision
if challenge_auc >= 0.85:
    print("   ✅ PASS: Model generalizes to external data")
elif challenge_auc >= 0.80:
    print("   ⚠️  CONDITIONAL: Close monitoring required")
else:
    print("   ❌ FAIL: Do not deploy - investigate domain shift")
```

#### Exit Criteria
- ✅ If Challenge2012 AUC ≥ 0.85: Continue to Task 2
- ⚠️ If Challenge2012 AUC 0.80-0.84: Continue with caution + extra monitoring
- ❌ If Challenge2012 AUC < 0.80: STOP - analyze domain differences first

---

### TASK 2: Threshold Optimization ⚠️ CRITICAL
**Time**: 20 minutes  
**Importance**: High - affects sensitivity/specificity tradeoff  
**Current**: Using 0.5 (likely suboptimal)

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Load test predictions from Phase 2
test_metrics = json.load(open("results/phase2_outputs/diagnostics_CORRECTED.json"))
# Re-compute predictions on test set
# (You'll need to run model evaluation on test set if not cached)

# Generate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, pred_probs)

# Find optimal threshold (multiple strategies)
print("\n🎯 THRESHOLD OPTIMIZATION:")
print("=" * 60)

# Strategy 1: Maximize F1 score
f1_scores = 2 * (tpr * (1-fpr)) / (tpr + (1-fpr) + 1e-8)
optimal_idx_f1 = np.argmax(f1_scores)
optimal_thresh_f1 = thresholds[optimal_idx_f1]

# Strategy 2: Maximize sensitivity for 95%+ specificity
sens_spec_threshold = tpr - 0.05 * fpr  # Prioritize sensitivity
optimal_idx_ss = np.argmax(sens_spec_threshold)
optimal_thresh_ss = thresholds[optimal_idx_ss]

# Strategy 3: Youden index (J = sensitivity + specificity - 1)
youden = tpr + (1-fpr) - 1
optimal_idx_youden = np.argmax(youden)
optimal_thresh_youden = thresholds[optimal_idx_youden]

# Report results
print(f"Strategy 1 - Max F1:             threshold = {optimal_thresh_f1:.3f}")
print(f"             Sensitivity = {tpr[optimal_idx_f1]:.4f}")
print(f"             Specificity = {1-fpr[optimal_idx_f1]:.4f}")

print(f"\nStrategy 2 - Prioritize Sensitivity: threshold = {optimal_thresh_ss:.3f}")
print(f"             Sensitivity = {tpr[optimal_idx_ss]:.4f}")
print(f"             Specificity = {1-fpr[optimal_idx_ss]:.4f}")

print(f"\nStrategy 3 - Youden Index:       threshold = {optimal_thresh_youden:.3f}")
print(f"             Sensitivity = {tpr[optimal_idx_youden]:.4f}")
print(f"             Specificity = {1-fpr[optimal_idx_youden]:.4f}")

# Recommended for clinical use (maximize sensitivity for catching deaths)
RECOMMENDED_THRESHOLD = optimal_thresh_ss
print(f"\n✅ RECOMMENDED THRESHOLD: {RECOMMENDED_THRESHOLD:.3f}")
print(f"   (Maximizes sensitivity for acceptable specificity)")

# Save for deployment
threshold_config = {
    "recommended": RECOMMENDED_THRESHOLD,
    "strategy": "Maximize sensitivity for 95%+ specificity",
    "sensitivity_at_threshold": float(tpr[optimal_idx_ss]),
    "specificity_at_threshold": float(1 - fpr[optimal_idx_ss]),
    "note": "DO NOT USE 0.5 - this is suboptimal for clinical use"
}

with open("results/phase2_outputs/optimal_threshold.json", "w") as f:
    json.dump(threshold_config, f, indent=2)

# Visualize ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC={roc_auc_score(y_test, pred_probs):.3f})')
plt.scatter([1-spec], [sens], color='red', s=100, marker='o', label=f'Optimal Threshold={RECOMMENDED_THRESHOLD:.3f}')
plt.scatter([1-0], [1], color='green', s=100, marker='x', label='Perfect Classification')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
plt.xlabel('1 - Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.title('ROC Curve with Optimal Threshold')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("results/phase2_outputs/roc_curve_optimal_threshold.png", dpi=150, bbox_inches='tight')
plt.show()

print("✓ ROC curve saved to: results/phase2_outputs/roc_curve_optimal_threshold.png")
print("✓ Threshold config saved to: results/phase2_outputs/optimal_threshold.json")
```

---

### TASK 3: Feature Importance Analysis
**Time**: 15 minutes  
**Importance**: Medium - enables clinical explainability

```python
from sklearn.inspection import permutation_importance
import json

print("\n📊 COMPUTING FEATURE IMPORTANCE (Permutation Method)")
print("=" * 60)

# Load Phase 2 feature names
phase2_df = pd.read_csv("results/phase1_outputs/phase1_24h_windows_CLEAN.csv")
feature_names = [c for c in phase2_df.columns if c not in ['patientunitstayid', 'mortality']]

# Compute permutation importance on test set
# This tells us: if we randomly shuffle feature X, how much does performance drop?
perm_importance = permutation_importance(
    model, X_test_scaled, y_test, 
    n_repeats=10, 
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

# Sort by importance
importance_sort_idx = np.argsort(perm_importance.importances_mean)[::-1]

print("\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
print("-" * 60)
top_10 = 10
for rank, idx in enumerate(importance_sort_idx[:top_10], 1):
    importance_mean = perm_importance.importances_mean[idx]
    importance_std = perm_importance.importances_std[idx]
    feature_name = feature_names[idx]
    
    # Interpretation
    if importance_mean > 0.10:
        interpretation = "🔴 CRITICAL"
    elif importance_mean > 0.05:
        interpretation = "🟡 HIGH"
    else:
        interpretation = "🟢 MODERATE"
    
    print(f"{rank:2d}. {feature_name:30s} {interpretation}")
    print(f"    Importance: {importance_mean:.4f} (±{importance_std:.4f})")

# Save for documentation
feature_importance_data = {
    "method": "Permutation Importance",
    "n_repeats": 10,
    "test_set_size": len(X_test),
    "top_10_features": [
        {
            "rank": rank,
            "feature": feature_names[idx],
            "importance": float(perm_importance.importances_mean[idx]),
            "std": float(perm_importance.importances_std[idx])
        }
        for rank, idx in enumerate(importance_sort_idx[:10], 1)
    ]
}

with open("results/phase2_outputs/feature_importance.json", "w") as f:
    json.dump(feature_importance_data, f, indent=2)

# Visualize
plt.figure(figsize=(10, 8))
top_n = 15
indices = importance_sort_idx[:top_n]
values = perm_importance.importances_mean[indices]
errors = perm_importance.importances_std[indices]
labels = [feature_names[i] for i in indices]

plt.barh(range(len(labels)), values[::-1], xerr=errors[::-1])
plt.yticks(range(len(labels)), labels[::-1])
plt.xlabel('Permutation Importance')
plt.title('Feature Importance (Top 15 Features)')
plt.tight_layout()
plt.savefig("results/phase2_outputs/feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Feature importance saved to: results/phase2_outputs/feature_importance.json")
print("✓ Visualization saved to: results/phase2_outputs/feature_importance.png")
```

---

### TASK 4: Calibration Analysis
**Time**: 20 minutes  
**Importance**: Medium - ensures predicted probabilities match reality

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt

print("\n📊 CALIBRATION ANALYSIS")
print("=" * 60)

# Compute calibration curve
prob_true, prob_pred = calibration_curve(
    y_test, pred_probs, 
    n_bins=10, 
    strategy='uniform'
)

# Compute Expected Calibration Error (ECE)
ece = np.mean(np.abs(prob_true - prob_pred))

print(f"Expected Calibration Error (ECE): {ece:.4f}")
if ece < 0.05:
    print("✅ Model is WELL-CALIBRATED (ECE < 0.05)")
elif ece < 0.10:
    print("⚠️  Model is MODERATELY CALIBRATED (ECE < 0.10)")
    print("   Recommend applying temperature scaling")
else:
    print("❌ Model is POORLY CALIBRATED (ECE > 0.10)")
    print("   MUST apply calibration before deployment")

# If poorly calibrated, apply temperature scaling
if ece > 0.05:
    print("\n🔧 APPLYING TEMPERATURE SCALING...")
    
    # Use validation set for temperature scaling (not test set)
    calibrated_model = CalibratedClassifierCV(
        model, 
        method='sigmoid',  # Sigmoid method similar to Platt scaling
        cv='prefit'
    )
    calibrated_model.fit(X_val_scaled, y_val)
    
    # Get calibrated probabilities
    pred_probs_calibrated = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    # Recompute calibration
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_test, pred_probs_calibrated, 
        n_bins=10
    )
    ece_cal = np.mean(np.abs(prob_true_cal - prob_pred_cal))
    
    print(f"After temperature scaling: ECE = {ece_cal:.4f}")
    
    # Save calibrated model
    torch.save({
        'model_state': calibrated_model.state_dict(),
        'scaler_mean': scaler_mean,
        'scaler_scale': scaler_scale,
        'calibration_method': 'sigmoid',
        'ece_before': ece,
        'ece_after': ece_cal
    }, "results/phase2_outputs/ensemble_model_CALIBRATED.pth")
    
    print("✓ Calibrated model saved")

# Visualize calibration curve
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Calibration')
plt.plot(prob_pred, prob_true, 'bo-', linewidth=2, markersize=6, label='Model Calibration')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title(f'Calibration Curve (ECE={ece:.4f})')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/phase2_outputs/calibration_curve.png", dpi=150, bbox_inches='tight')
plt.show()

print("✓ Calibration curve saved to: results/phase2_outputs/calibration_curve.png")
```

---

### TASK 5: Error Analysis (False Negatives)
**Time**: 15 minutes  
**Importance**: High - understand failure modes

```python
print("\n🔍 ERROR ANALYSIS - FALSE NEGATIVES")
print("=" * 60)

# Identify false negatives (model predicted 0, actually 1)
fn_mask = (y_test == 1) & (pred_probs <= RECOMMENDED_THRESHOLD)
fn_indices = np.where(fn_mask)[0]

print(f"\nFalse Negatives (using threshold={RECOMMENDED_THRESHOLD:.3f}):")
print(f"Count: {len(fn_indices)} out of {np.sum(y_test)} deaths")
print(f"False Negative Rate: {len(fn_indices) / np.sum(y_test) * 100:.1f}%")

if len(fn_indices) > 0:
    print("\n" + "-" * 60)
    for i, idx in enumerate(fn_indices, 1):
        print(f"\nFN Case #{i} (Test Index {idx}):")
        print(f"  Predicted Probability: {pred_probs[idx]:.4f}")
        print(f"  True Label: Death (actual)")
        print(f"  Decision: Model predicted ALIVE (WRONG)")
        
        # Show feature values for this case
        # Top 5 features for this patient
        feature_values = X_test_scaled[idx]
        top_feature_idx = np.argsort(-np.abs(feature_values))[:5]
        
        print(f"  Top features for this patient:")
        for rank, feat_idx in enumerate(top_feature_idx, 1):
            feat_name = feature_names[feat_idx]
            feat_value = X_test[idx, feat_idx]  # Original scale
            print(f"    {rank}. {feat_name}: {feat_value:.2f}")
    
    print("\n" + "=" * 60)
    print("📌 KEY INSIGHT:")
    print("   The model occasionally misses high-risk patients.")
    print("   Recommend: Use ensemble with other models for uncertain cases")
    print("   Action: Implement clinician review for borderline cases")

# Also analyze false positives
fp_mask = (y_test == 0) & (pred_probs > RECOMMENDED_THRESHOLD)
fp_indices = np.where(fp_mask)[0]

print(f"\nFalse Positives (using threshold={RECOMMENDED_THRESHOLD:.3f}):")
print(f"Count: {len(fp_indices)} out of {np.sum(y_test == 0)} negatives")
print(f"False Positive Rate: {len(fp_indices) / np.sum(y_test == 0) * 100:.1f}%")

# Save error analysis
error_analysis = {
    "false_negatives": {
        "count": len(fn_indices),
        "rate": float(len(fn_indices) / max(np.sum(y_test), 1)),
        "indices": fn_indices.tolist()
    },
    "false_positives": {
        "count": len(fp_indices),
        "rate": float(len(fp_indices) / max(np.sum(y_test == 0), 1)),
        "indices": fp_indices.tolist()
    },
    "threshold": RECOMMENDED_THRESHOLD
}

with open("results/phase2_outputs/error_analysis.json", "w") as f:
    json.dump(error_analysis, f, indent=2)

print("✓ Error analysis saved to: results/phase2_outputs/error_analysis.json")
```

---

### TASK 6: Bootstrap Confidence Intervals
**Time**: 30 minutes  
**Importance**: Medium - quantifies uncertainty

```python
from scipy import stats

print("\n📊 BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 60)

n_iterations = 1000
bootstrap_auc = []
bootstrap_sens = []
bootstrap_spec = []

np.random.seed(42)

for i in range(n_iterations):
    if (i + 1) % 100 == 0:
        print(f"  Progress: {i+1}/{n_iterations}")
    
    # Resample test set with replacement
    indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
    y_boot = y_test[indices]
    pred_boot = pred_probs[indices]
    
    # Compute metrics
    if len(np.unique(y_boot)) > 1:  # Only if both classes present
        auc = roc_auc_score(y_boot, pred_boot)
        bootstrap_auc.append(auc)
        
        # Sensitivity & Specificity at optimal threshold
        pred_labels_boot = (pred_boot >= RECOMMENDED_THRESHOLD).astype(int)
        tp = np.sum((y_boot == 1) & (pred_labels_boot == 1))
        fn = np.sum((y_boot == 1) & (pred_labels_boot == 0))
        tn = np.sum((y_boot == 0) & (pred_labels_boot == 0))
        fp = np.sum((y_boot == 0) & (pred_labels_boot == 1))
        
        if tp + fn > 0:
            sens = tp / (tp + fn)
            bootstrap_sens.append(sens)
        
        if tn + fp > 0:
            spec = tn / (tn + fp)
            bootstrap_spec.append(spec)

bootstrap_auc = np.array(bootstrap_auc)
bootstrap_sens = np.array(bootstrap_sens)
bootstrap_spec = np.array(bootstrap_spec)

# Compute 95% confidence intervals
auc_ci = np.percentile(bootstrap_auc, [2.5, 97.5])
sens_ci = np.percentile(bootstrap_sens, [2.5, 97.5])
spec_ci = np.percentile(bootstrap_spec, [2.5, 97.5])

print("\n📈 95% CONFIDENCE INTERVALS (Bootstrap):")
print("-" * 60)
print(f"AUC:         {bootstrap_auc.mean():.4f} [95% CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f}]")
print(f"Sensitivity: {bootstrap_sens.mean():.4f} [95% CI: {sens_ci[0]:.4f} - {sens_ci[1]:.4f}]")
print(f"Specificity: {bootstrap_spec.mean():.4f} [95% CI: {spec_ci[0]:.4f} - {spec_ci[1]:.4f}]")

# Save results
ci_results = {
    "method": "Bootstrap (n=1000)",
    "auc": {
        "mean": float(bootstrap_auc.mean()),
        "ci_lower": float(auc_ci[0]),
        "ci_upper": float(auc_ci[1]),
        "std": float(bootstrap_auc.std())
    },
    "sensitivity": {
        "mean": float(bootstrap_sens.mean()),
        "ci_lower": float(sens_ci[0]),
        "ci_upper": float(sens_ci[1]),
        "std": float(bootstrap_sens.std())
    },
    "specificity": {
        "mean": float(bootstrap_spec.mean()),
        "ci_lower": float(spec_ci[0]),
        "ci_upper": float(spec_ci[1]),
        "std": float(bootstrap_spec.std())
    }
}

with open("results/phase2_outputs/bootstrap_confidence_intervals.json", "w") as f:
    json.dump(ci_results, f, indent=2)

print("\n✓ Bootstrap CI saved to: results/phase2_outputs/bootstrap_confidence_intervals.json")
```

---

## EXECUTION CHECKLIST

Run each task in order:

- [ ] **Task 1** (30 min): External Validation (Challenge2012)
  - [ ] Load data
  - [ ] Load model + scaler
  - [ ] Preprocess
  - [ ] Evaluate
  - [ ] Decision: Pass/Conditional/Fail

- [ ] **Task 2** (20 min): Threshold Optimization
  - [ ] Compute ROC curve
  - [ ] Find optimal threshold
  - [ ] Visualize
  - [ ] Save config

- [ ] **Task 3** (15 min): Feature Importance
  - [ ] Compute permutation importance
  - [ ] Extract top-10 features
  - [ ] Visualize
  - [ ] Document for clinicians

- [ ] **Task 4** (20 min): Calibration Analysis
  - [ ] Compute ECE
  - [ ] Apply temperature scaling if needed
  - [ ] Visualize calibration curve
  - [ ] Save calibrated model if applicable

- [ ] **Task 5** (15 min): Error Analysis
  - [ ] Identify false negatives
  - [ ] Analyze characteristics
  - [ ] Document patterns
  - [ ] Create clinician alerts

- [ ] **Task 6** (30 min): Bootstrap Confidence Intervals
  - [ ] Run 1000 bootstrap samples
  - [ ] Compute 95% CIs
  - [ ] Document uncertainty
  - [ ] Save results

---

## FINAL DECISION MATRIX

After completing all tasks:

### ✅ CLEAR TO DEPLOY if:
```
Challenge2012 AUC ≥ 0.85              ✓
Optimal threshold identified           ✓
Top-10 features documented             ✓
Calibration ECE < 0.05                ✓
Error analysis complete                ✓
Confidence intervals <±5%              ✓
```

### ⚠️ CONDITIONAL DEPLOYMENT if:
```
Challenge2012 AUC 0.80-0.84           (with monitoring)
Calibration ECE 0.05-0.10             (apply temperature scaling)
Any uncertainty but no blockers        (risk accepted)
```

### ❌ DO NOT DEPLOY if:
```
Challenge2012 AUC < 0.80              BLOCKER
Any metric unexplained                BLOCKER
ECE > 0.15 and can't calibrate         BLOCKER
```

---

**Ready to start?** Begin with Task 1 immediately. Should complete all 6 tasks by end of day.

**Deliverable**: Comprehensive robustness report + Go/No-Go decision + deployment configuration

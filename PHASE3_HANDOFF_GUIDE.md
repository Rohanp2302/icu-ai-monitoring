# PHASE 2 → PHASE 3 HANDOFF GUIDE

## Quick Summary for Phase 3

✅ **Phase 2 Status**: Data leakage fixed, model validated
- Test AUC: 93.91% (meets 90% target)
- CV AUC: 99.60% ± 0.35% (stable)
- All preprocessing verified

---

## What Changed in Phase 2

### Before (❌ Invalid)
- Training: Raw unnormalized data
- Evaluation: Normalized data
- **Result**: 99.62% AUC (inflated)

### After (✅ Valid)
- Training: Normalized data (scaler fitted on train only)
- Evaluation: Normalized data (using train statistics)
- **Result**: 93.91% AUC (honest, reproducible)

---

## For Phase 3 Implementation

### Step 1: Load Validated Model
```python
import torch

# Load the corrected, validated model
checkpoint = torch.load('results/phase2_outputs/ensemble_model_CORRECTED.pth')

# Extract components
model_state = checkpoint['model_state']
scaler_mean = np.array(checkpoint['scaler_mean'])
scaler_scale = np.array(checkpoint['scaler_scale'])
test_auc = checkpoint['test_auc']
```

### Step 2: Reconstruct StandardScaler
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale
# This scaler is fitted on Phase 2 training data
# Use it to preprocess ALL Phase 3 data
```

### Step 3: Use Proper Preprocessing Pipeline
```python
# Load Phase 2 clean data
df = pd.read_csv('results/phase1_outputs/phase1_24h_windows_CLEAN.csv')
X = df.drop(columns=['patientunitstayid', 'mortality']).values
y = df['mortality'].values

# Apply scaler from Phase 2 (already fitted, DON'T refit)
X_scaled = scaler.transform(X)  # Use train statistics

# Now safe to split for Phase 3 work
X_train_p3, X_test_p3, y_train_p3, y_test_p3 = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
# Note: Data already scaled with Phase 2 train statistics
```

### Step 4: Critical - Never Re-fit Scaler
```python
# ❌ WRONG
scaler_new = StandardScaler()
X_new = scaler_new.fit_transform(X_phase3_data)  # DON'T DO THIS

# ✅ RIGHT - Use Phase 2 scaler
X_new = scaler.transform(X_phase3_data)  # Use existing scaler
```

---

## Preprocessing Checklist for Phase 3

Before using data in any Phase 3 model:

- [ ] Load `ensemble_model_CORRECTED.pth` to get scaler statistics
- [ ] Extract `scaler_mean` and `scaler_scale` from checkpoint
- [ ] Reconstruct StandardScaler with these values
- [ ] Apply `scaler.transform()` to all Phase 3 data (never fit)
- [ ] Verify: All data has mean ≈ 0 and scale ≈ 1 after transformation
- [ ] Document: "Using Phase 2 train statistics for normalization"
- [ ] Test: Sanity check that stats haven't changed unexpectedly

---

## Validation Framework to Carry Forward

### For Every Model in Phase 3:

1. **Data Handling**
   - [ ] Split data BEFORE preprocessing
   - [ ] Fit scaler ONLY on training data
   - [ ] Apply scaler using training statistics to val/test

2. **Baseline Comparison**
   - [ ] Include ≥2 baselines (LR, RF, or heuristics)
   - [ ] Train baselines with SAME preprocessing
   - [ ] Report all models' metrics in comparison table

3. **Cross-Validation**
   - [ ] Use 5-fold stratified CV
   - [ ] Fit scaler independently per fold
   - [ ] Report mean ± std of key metrics
   - [ ] Flag any folds with suspicious results (0% or 100% performance)

4. **Suspicious Result Investigation**
   - [ ] If CV shows extreme variance (>0.1 range): investigate
   - [ ] If train AUC > test AUC by >20%: possible overfitting/leakage
   - [ ] If reported AUC >99% on small dataset: likely leakage or chance

5. **Documentation**
   - [ ] Document preprocessing pipeline
   - [ ] Save train statistics (mean, scale) with model
   - [ ] Report train/val/test AUC gap
   - [ ] Include baseline comparison in report

---

## Phase 3 Model Architecture Ideas

Based on Phase 2 learnings:

### Option 1: Ensemble Enhancement
- Keep 3-path architecture, tune hyperparameters
- Swap in LSTM/GRU instead of dense for temporal modeling
- Add attention mechanisms

### Option 2: Hybrid Approach
- Random Forest (best single model: 99.84% AUC)
- + Ensemble (competitive, shows promise)
- Voting ensemble of both

### Option 3: Domain-Specific
- Add feature importance analysis from RF
- Design architecture to emphasize key features
- Include clinical constraints in loss function

---

## One-Liner Rule: "Trust, But Verify"

**Every metric should pass these three tests:**

1. **Preprocessing Check**: Can you explain how each data point was scaled?
2. **Baseline Check**: How does it compare to simple models?
3. **CV Check**: Is performance stable across folds?

If you can't answer all three clearly → investigate before proceeding.

---

## Emergency Debug Checklist

If Phase 3 metrics seem suspiciously good:

- [ ] Print training data statistics (mean, min, max, std)
- [ ] Print test data statistics - should be ≈ same mean, similar range
- [ ] Check for NaN or infinite values
- [ ] Verify train/test/val have NO overlap
- [ ] Run 5-fold CV and plot distribution of folds
- [ ] Compare against Random Forest baseline
- [ ] If CV fold 3 has 0% accuracy but fold 1 has 100%: stop and debug

---

## Recommended Phase 3 Structure

```python
# phase3_model.py structure template

def load_phase2_scaler():
    """Load validated scaler from Phase 2"""
    checkpoint = torch.load('results/phase2_outputs/ensemble_model_CORRECTED.pth')
    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint['scaler_mean'])
    scaler.scale_ = np.array(checkpoint['scaler_scale'])
    return scaler

def preprocess_data(X, scaler):
    """Always use transform, never fit"""
    return scaler.transform(X)

def validate_preprocessing(X_train, X_test):
    """Sanity check preprocessing"""
    assert 0 <= X_train.mean() <= 0.1, f"Train mean off: {X_train.mean()}"
    assert 0 <= X_test.mean() <= 0.1, f"Test mean off: {X_test.mean()}"
    assert 0.8 <= X_train.std() <= 1.2, f"Train std off: {X_train.std()}"
    print("✓ Preprocessing validation passed")

def compare_baselines(X_train, X_test, y_train, y_test):
    """Always compare against baselines"""
    baseline_rf = RandomForestClassifier(...)
    baseline_rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, baseline_rf.predict_proba(X_test)[:, 1])
    return {'random_forest': rf_auc}

def main():
    # Load & preprocess
    scaler = load_phase2_scaler()
    X = load_data()
    X_scaled = preprocess_data(X, scaler)
    
    # Split & validate
    X_train, X_test, y_train, y_test = train_test_split(...)
    validate_preprocessing(X_train, X_test)
    
    # Train & compare
    model = train_phase3_model(...)
    baselines = compare_baselines(...)
    
    # Report results
    report_metrics(model, baselines)
```

---

## Success Criteria for Phase 3

✅ Model passes:
- AUC > 90% on test set
- Reasonable train-test gap (<15%)
- Cross-validation shows stable results
- Outperforms OR matches best baseline
- Preprocessing is consistent and documented

---

## Key Contact Points

**Phase 2 Artifacts** (for reference):
- `PHASE2_DATA_LEAKAGE_CORRECTION_REPORT.md` - Full technical details
- `PHASE2_CORRECTION_COMPLETE.md` - Summary of what changed
- `results/phase2_outputs/ensemble_model_CORRECTED.pth` - Model + scaler

**Files to Copy to Phase 3 Directory**:
```
- phase2_diagnostics_corrected.py (copy pattern)
- phase2_baselines_corrected.py (copy pattern)
- phase2_cross_validation_corrected.py (copy pattern)
```

**Lessons to Remember**:
1. SPLIT → NORMALIZE (never normalize before splitting)
2. Fit scaler on TRAIN only
3. Use train statistics on VAL/TEST
4. Always include baselines
5. Trust CV results more than single test sets

---

**Ready for Phase 3!** 🚀

---

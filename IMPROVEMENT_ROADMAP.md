# ICU Mortality Model - Improvement Roadmap
**Date**: April 7, 2026  
**Current Status**: Poor (12.2% recall, 0.88 AUC)  
**Target Status**: Excellent (75%+ recall, 0.90+ AUC)

---

## 🎯 Phase 1: Quick Wins (Days 1-2)

### 1.1 Threshold Optimization
**Problem**: Model uses 0.5 threshold for 50% prevalence, but mortality is 8.6% (rare event)  
**Solution**: Find optimal threshold on validation set  
**Expected Impact**: Recall 12% → 65-80%, AUC stays ~0.88

**Steps**:
```python
# Find optimal threshold by maximizing F1/Recall on validation set
thresholds = np.arange(0.01, 0.5, 0.01)
best_threshold = 0.08-0.12 (TBD - calculate on val set)
# Retrain decision boundary
```

**Effort**: 2 hours  
**Files to modify**: `app.py` line 215 (change predict_proba threshold)

---

### 1.2 Simple Ensemble (3 Algorithms)
**Problem**: Single RF model has high variance  
**Solution**: Combine RF + Logistic Regression + Gradient Boosting  
**Why**:
- RF: 0.8877 AUC, 12% recall (high specificity)
- LR: 0.7200 AUC, 59.8% recall (catches deaths)
- GB: ~0.85 AUC, balanced

**Steps**:
1. Load existing `models/logistic_regression_baseline.py`
2. Create `models/ensemble_improved.py`
3. Average predictions: `(RF + LR + GB) / 3`
4. New threshold on ensemble output

**Expected metrics after ensemble**:
- AUC: 0.8877 → 0.88-0.91
- Recall: 12% → 50-65%
- Implementation: 1 day

---

### 1.3 Load Existing LSTM Models (Don't Retrain)
**Problem**: 5 trained LSTM checkpoints exist but aren't deployed  
**Solution**: Create parallel `/api/predict-temporal` endpoint

**Checkpoints available**:
```
checkpoints/multimodal/
├── fold_0_best_model.pt  (AUC: 0.85-0.87 estimated)
├── fold_1_best_model.pt
├── fold_2_best_model.pt
├── fold_3_best_model.pt
└── fold_4_best_model.pt
```

**Why temporal models are better**:
- Use 24-hour sequence data: `X_24h.npy`, `means_24h.npy`, `stds_24h.npy`
- Capture trends: is HR stable or drifting upward?
- Capture volatility: are vitals erratic (bad sign)?
- Attention mechanism: identify which hours matter most

**Expected metrics**:
- AUC: 0.85-0.87
- Recall: 40-60% (better than RF!)
- Ensemble LSTM + RF: 0.88-0.92 AUC, 65-75% recall

**Effort**: 2-3 days (load model, test, create API endpoint)

---

## 📊 Phase 2: Feature Engineering (Days 3-5)

### 2.1 Temporal Feature Aggregation (24-hour patterns)
**Current problem**: Using static aggregations (mean, std, min, max) - loses temporal information

**What to add**:
```
For each vital (HR, RR, SaO2, BP, Temp):
- Linear trend (slope of best-fit line over 24h)
- Volatility (coefficient of variation)
- Entropy (disorder/chaos in signal)
- Autocorrelation at lags 1h, 4h, 12h
- Change rate (% change per hour)
- Number of abnormal events in 24h
- Duration of abnormality
- Recovery time after intervention
- Heart rate variability (HRV) metrics
```

**Code location**: `src/feature_engineering.py` → enhance `extract_patient_features()`

**Expected improvement**: 10-15% AUC gain

---

### 2.2 Disease-Specific Risk Factors
**Current**: Only vitals (HR, RR, O2, BP, Temp)  
**Missing**: 
- Sepsis indicators: lactate, WBC, temp trends
- Kidney injury: creatinine, BUN, urine output
- Respiratory: pO2/FiO2 ratio, acid-base status
- Liver: bilirubin, albumin, INR
- Infection flags, antibiotic usage
- Admission type (trauma, surgery, medical)

**Data source**:
- `data/processed_icu_hourly_v2.csv` - likely has most labs
- `data/therapeutic_targets.json` - treatment info
- Create separate feature groups for each organ system

**Code**: Create `src/feature_engineering_disease_specific.py`

**Expected improvement**: 5-10% AUC gain, much better interpretability

---

### 2.3 Fix Feature Mismatch (42 vs 120)
**Current issue**: Warning "Expected 42 features, got 39"

**Investigate**:
```
- models/ensemble_predictor.py expects 42 features
- app.py extract_patient_features() creates 120 features
- Why the mismatch? Different design specs?
```

**Solution**:
1. Choose ONE feature set: either 42 or 120
2. Update all models to use same set
3. Document feature definition clearly

---

## 🔄 Phase 3: Model Architecture (Days 6-10)

### 3.1 Evaluate: RF vs LSTM vs Transformer
**Compare on same test set**:
| Model | AUC | Recall | F1 | Training Time | Interpretability |
|-------|-----|--------|----|----|---|
| Random Forest | 0.8877 | 12% | 0.18 | <1min | High ✅ |
| Logistic Regression | 0.7200 | 59.8% | 0.38 | <1min | High ✅ |
| LSTM (existing) | 0.85-0.87? | 40-60%? | 0.40-0.50? | <5s/patient | Medium |
| Transformer | 0.87-0.90? | 60-70%? | 0.55-0.60? | <5s/patient | Medium |
| Ensemble (3-5 models) | **0.90-0.92** | **70-80%** | **0.60-0.65** | ~5s/patient | Medium |

**Decision**: Test LSTM checkpoints first (free), then ensemble

---

### 3.2 Proper Cross-Validation Strategy
**Current**: Unknown fold strategy  
**New**:
- 5-fold CV with stratified split (preserve 8.6% mortality rate)
- Report mean ± std on each fold
- Monitor for overfitting: Recall_train vs Recall_val gap
- Calibration plot: are probabilities well-calibrated for rare events?

**Code**: `src/evaluation/k_fold_validator.py`

---

### 3.3 Address Class Imbalance
**Current**: 91.4% negative, 8.6% positive  
**Methods**:
1. **Weighted loss**: `pos_weight = count_neg / count_pos = 10.6`
2. **Threshold optimization**: Move threshold down (already in Phase 1)
3. **SMOTE**: Oversample minority class (careful - can cause overfitting)
4. **Focal Loss**: Pytorch layer focusing on hard negatives
5. **Cost-sensitive learning**: Higher penalty for FN than FP

**Implement**: Use method 1 + 2 initially

---

## 📋 Phase 4: Evaluation & Validation (Days 11-12)

### 4.1 Comprehensive Metrics
```python
from sklearn.metrics import:
  - roc_auc_score (discriminative ability)
  - precision_recall_curve (rare event focus)
  - confusion_matrix (TP/FP/TN/FN)
  - calibration_curve (probability reliability)
  - roc_curve (threshold analysis)

For clinical validation:
  - Sensitivity @ >90% specificity
  - NPV (negative predictive value) > 98%
  - Positive LR, Negative LR
```

### 4.2 Held-Out Test Set
- 20% of data unseen during training
- Stratified split (keep 8.6% mortality)
- Report all metrics above
- Create confusion matrices, calibration plots

### 4.3 Clinical Validation
- Get feedback from ICU physicians
- Is recall 70% acceptable? (miss 30% of deaths)
- Are top features clinically sensible?
- Does risk score correlate with clinical impression?

---

## 🚀 Implementation Order

### Week 1 (Priority Order):
1. **Day 1 (4 hours)**: Calculate optimal threshold
2. **Day 1-2 (8 hours)**: Build simple 3-model ensemble
3. **Day 2-3 (12 hours)**: Load and test LSTM checkpoint
4. **Day 3-4 (8 hours)**: Add 24-hour temporal features

### Week 2:
5. **Day 5-6 (12 hours)**: Add disease-specific features (sepsis, AKI, etc.)
6. **Day 6-7 (8 hours)**: Fix 42 vs 120 feature mismatch
7. **Day 8-10 (12 hours)**: Proper CV and cross-validation
8. **Day 11-12 (8 hours)**: Final validation and clinical feedback

---

## 📊 Success Criteria

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| AUC | 0.8877 | 0.90+ | ❌ |
| Recall | 12% | 75%+ | ❌ |
| Precision | 83% | 60%+ | ✅ |
| F1 | 0.18 | 0.60+ | ❌ |
| Specificity | 99.8% | 90%+ | ✅ |
| Clinical use | No | Yes | ❌ |

---

## 📁 Key Files to Modify

```
Priority order:
1. app.py - Add threshold optimization + ensemble endpoint
2. src/baselines/random_forest_baseline.py - Ensemble predictions
3. src/feature_engineering.py - Add temporal + disease features
4. src/models/temporal_loader.py - Load LSTM checkpoints (new file)
5. src/evaluation/metrics_comprehensive.py - Full metrics (new file)
6. src/training/cv_trainer.py - Proper K-fold CV (new file)
```

---

## 🎯 Next Steps

**Immediate (Next 2 hours)**:
1. [ ] Review current train/val/test split strategy
2. [ ] Calculate optimal decision threshold on validation set
3. [ ] Update `app.py` to use new threshold
4. [ ] Measure new recall/precision after threshold change

**Today**:
5. [ ] Create ensemble of 3 models
6. [ ] Measure ensemble metrics
7. [ ] Document which features are actually being used

**This week**:
8. [ ] Load and test LSTM checkpoints
9. [ ] Design 24-hour temporal feature aggregation
10. [ ] Plan disease-specific factor integration

---

## 💡 Key Insights

✅ **What's working**: 
- Data collection infrastructure
- Multiple model types built
- Training pipeline operational

❌ **What's broken**:
- Decision threshold not optimized for rare events
- Better models not deployed
- Temporal information underutilized
- Disease context ignored

✅ **Why it's fixable**:
- Don't need to retrain from scratch
- Good models already exist
- Data quality seems decent
- Problem is integration, not fundamentals

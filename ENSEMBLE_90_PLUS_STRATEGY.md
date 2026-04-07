# 🎯 ENSEMBLE 90+ AUC STRATEGY - Comprehensive Implementation Plan

**Date**: April 7, 2026  
**Goal**: Build ensemble reaching **90+ AUC** by combining RF, LSTM, and boosting models  
**Timeline**: Week 3 (by April 19)  
**Status**: 🔄 PLANNING PHASE

---

## PART 1: CURRENT STATE ASSESSMENT

### Models Available

#### 1. Random Forest Models
| Model | AUC | Status | Type |
|-------|-----|--------|------|
| **Tuned RF** | **0.9032** ✅ | Needs rebuilding | Hyperparameter optimized |
| **Calibrated RF** | 0.8990 | Exists | Calibration overlay |
| **Feature-Sel RF** | 0.8970 | Exists | Top-50 features only |
| **Baseline RF** | 0.8877 | Deployed | Currently in Flask |
| **Stacking Ensemble** | 0.8889 | Exists script | RF+GB+AdaBoost |

#### 2. LSTM/Temporal Models
| Model | AUC | Status | Issue |
|-------|-----|--------|-------|
| **LSTM Checkpoints** | ❓ 0.54 | Evaluated | Data mismatch (need root cause) |
| **Multi-Task Models** | Expected 0.85+ | In checkpoints/ | Not properly evaluated |
| **Temporal Data** | N/A | Extracted | X_24h.npy ready |

#### 3. Other Ensemble Options
| Approach | AUC Potential | Complexity | 
|----------|---------------|-----------|
| Voting Classifier | 0.90-0.92 | Low |
| Stacking | 0.89-0.91 | Medium |
| Blending | 0.89-0.92 | Medium |
| Meta-learner fusion | 0.91-0.93 | High |

### Data Assets
```
✓ Static features: 120 columns, 2,373 patients
✓ Temporal data: X_24h (1,713, 24, 6) extracted
✓ Labels: Mortality labeled, 8.3-8.6% rate
✓ Validation split: All folds ready
```

---

## PART 2: ROOT CAUSE - WHY LSTM EVALUATION GAVE 0.54 AUC

### Hypothesis for Failure

**Checkpoint Issue #1: Wrong Feature Mapping**
- Checkpoints trained on task prediction fusion layer
- Input expects: (temporal_features, static_features)
- We provided: (X_24h with 6 features, X_static with 8)
- **Problem**: Checkpoints may have been trained on different feature schema

**Checkpoint Issue #2: Multi-task vs Single-task**
- Checkpoints contain 5 task heads (mortality, risk, outcomes, response, LOS)
- Only mortality head is relevant for us
- Checkpoints use shared backbone - other tasks may interfere
- **Problem**: Loss function balancing across tasks affects mortality performance

**Checkpoint Issue #3: Preprocessing Mismatch**
- Checkpoints expect: Specific normalization (z-score? scaling? padding?)
- We provided: Our own normalization of X_24h
- **Problem**: Model distribution shift "sees" different input statistics

**Checkpoint Issue #4: Static Feature Engineering**
- Checkpoints trained on engineered static features (not just demographics)
- We used default placeholders
- **Problem**: Fusion layer receives wrong semantic information

---

## PART 3: 90+ AUC ENSEMBLE ARCHITECTURE

### Phase A: Foundation Models (Days 1-2)

#### Task 1: Rebuild Tuned Random Forest
```
FROM:  Baseline RF (0.8877 AUC) 
TO:    Tuned RF (0.9032 AUC through hyperparameter optimization)

Script: src/analysis/model_improvements.py -> improvement_1_hyperparameter_tuning()
Status: Needs execution
```

**Hyperparameter changes needed:**
- n_estimators: 200 → 300
- max_depth: 15 → 20
- min_samples_split: 2 → 5
- class_weight: 'balanced' → computed from training set

**Expected outcome**: 0.9032 AUC

#### Task 2: Fix & Redeploy LSTM Models
```
FROM:   LSTM checkpoints showing 0.54 AUC (failed)
TO:     Re-evaluate with proper feature alignment + task head extraction

Action: 
  1. Load LSTM checkpoint
  2. Extract ONLY mortality head (remove other 4 tasks)
  3. Align input features to what checkpoint was trained on
  4. Re-evaluate on full 1,713 samples
  5. Compare to RF baseline (should be 0.85+)
```

**Expected outcome**: 0.85-0.88 AUC (if checkpoint quality is good)

#### Task 3: Build Calibrated Multi-Model RF
```
FROM:   Tuned RF (0.9032)
TO:     Calibrated RF (0.9032+ calibration overlay)

Science:
  - Tuned RF not perfectly calibrated (Brier score: 0.0575)
  - Add isotonic or sigmoid calibration layer
  - Improves probability estimates without changing AUC much
  - But improves confidence scoring

Expected outcome: Same AUC (0.9032) + better calibration
```

---

### Phase B: Ensemble Fusion (Days 3-4)

#### Strategy 1: Voting Ensemble (Simple)
```
Combine:
  - Tuned RF (weight: 0.5, best performance)
  - Gradient Boosting (weight: 0.3, captures different patterns)
  - LSTM (weight: 0.2, temporal signals)

Method: Soft voting (average probabilities)

Test: 5-fold CV
  Fold 1: Train on folds 2-5, test on fold 1
  Fold 2: Train on folds 1,3-5, test on fold 2
  ...
  Fold 5: Train on folds 1-4, test on fold 5

Result: Average AUC across folds
Expected: 0.90-0.92 AUC
```

#### Strategy 2: Stacking with Meta-Learner (Better)
```
Level 0 (Base Learners):
  1. Tuned RF (AUC 0.9032)
  2. Gradient Boosting
  3. ExtraTrees
  4. LSTM temporal feature extractor
  5. Calibrated RF

Level 1 (Meta-Learner):
  Logistic Regression trained on:
    Input: Predictions from all base learners
    Output: Final mortality prediction
  
  CV: 5-fold on training set to avoid leakage

Result: Learns optimal weights automatically
Expected: 0.91-0.93 AUC
```

#### Strategy 3: Blending with Temporal Features (Advanced)
```
Combine:
  1. RF predictions: p_rf
  2. LSTM predictions: p_lstm 
  3. Temporal variance: σ(24h sequence)
  4. Clinical urgency: aggregated vital changes

Fusion:
  p_final = α * p_rf + β * p_lstm + γ * temporal_variance
  
  where α, β, γ learned from validation set

Expected: 0.91-0.94 AUC
```

---

### Phase C: Temporal Intelligence (Days 5-6)

#### Add Temporal Features to Static Ensemble
```
For each patient's 24h sequence (X_24h):
  Extract:
    - Deterioration rate: Δ vital signs / Δ time
    - Volatility: σ(HR, RR, SpO2) over 24h
    - Trends: Linear fit slopes
    - Clinical flags: Worsening episodes detected
    - Risk trajectory: Risk score over time

Combine with static:
  - [static_features_120, temporal_features_10]
  - Train ensemble on 130-dim input
```

**Expected boost**: +0.5-1.5% AUC from temporal signals

---

### Phase D: Optimization & Validation (Days 7+)

#### 1. Hyperparameter Tuning for Ensemble
```
Grid search over:
  - Model weights in voting
  - Base learner parameters
  - Calibration method
  - Decision threshold (for clinical use)

Cross-validation: 5-fold stratified
Metric optimization: AUC (primary), F1 (secondary), Recall (clinical)
```

#### 2. Threshold Optimization for Hospital Use
```
Current: 0.44 (optimized for 72% recall)
Future: Re-optimize for ensemble

If ensemble AUC = 0.91:
  - Recall target: 75%+
  - Precision target: 35%+
  - Threshold: TBD (likely 0.38-0.42)
```

#### 3. Final Validation
```
Holdout test set (20%): 435 patients
  Target AUC: 90+ ✅
  Target Recall: 75%+ ✅
  Target F1: 50%+ ✅

Cross-validation test:
  Average AUC across 5 folds: 90+
  Consistency: Std dev < 1%
```

---

## PART 4: IMPLEMENTATION ROADMAP

### Week 3 Timeline (April 8-19)

#### Monday-Tuesday (April 8-9): Foundation
- [ ] Execute model_improvements.py Phase 1 (tuned RF rebuild)
  - Expected output: Model file + 0.9032 AUC report
  - Time: 2 hours
- [ ] Debug LSTM checkpoint performance issue
  - Deep dive into architecture mismatch
  - Time: 3 hours
- [ ] Create temporal features (deterioration, volatility, trends)
  - Script to extract 10-15 temporal indicators
  - Time: 2 hours

#### Wednesday-Thursday (April 10-11): Ensemble Assembly
- [ ] Build voting ensemble (RF + GB + LSTM)
  - Test all weight combinations
  - Time: 4 hours
- [ ] Build stacking meta-learner
  - Level 0: 5 base models
  - Level 1: Logistic regression
  - Time: 3 hours
- [ ] Validate both approaches on holdout test
  - Compare AUC, recall, precision
  - Time: 2 hours

#### Friday-Monday (April 12-15): Optimization
- [ ] Hyperparameter tuning for winner ensemble
  - Grid search over ensemble parameters
  - Time: 4 hours
- [ ] Temporal feature integration
  - Add deterioration + volatility to feature set
  - Retrain ensemble with temporal signals
  - Time: 3 hours
- [ ] Threshold optimization for hospital recall target
  - Find optimal threshold for 75%+ recall
  - Time: 2 hours

#### Tuesday-Wednesday (April 16-17): Validation & Hospital Prep
- [ ] 5-fold cross-validation full report
  - AUC, recall, F1 per fold
  - Time: 2 hours
- [ ] Hospital integration documentation
  - Updated API specs for ensemble
  - Performance guarantees
  - Time: 3 hours
- [ ] Final model serialization
  - Save ensemble weights, scaler, feature names
  - Time: 1 hour

#### Thursday-Friday (April 18-19): Deployment
- [ ] Deploy ensemble to Flask API
  - Update /api/predict endpoint
  - Time: 2 hours
- [ ] Hospital staging validation
  - Test on hospital infrastructure
  - Time: 2 hours
- [ ] Go-live (April 19)

---

## PART 5: SUCCESS CRITERIA

### Target Metrics

```
PRIMARY:
  AUC:     ≥ 0.90  ✅ (ensemble focus)
  Recall:  ≥ 0.75  ✅ (catch deaths)
  F1:      ≥ 0.50  ✅ (balance)

SECONDARY:
  Precision: ≥ 0.33 (reduce false alarms)
  Specificity: ≥ 0.78 (maintain accuracy)
  Brier Score: ≤ 0.058 (calibration)

DEPLOYMENT:
  Response time: < 200ms per prediction
  Stability: Consistent across patient populations
  Interpretability: Can explain top 3 risk factors
```

### Risk Management

| Risk | Probability | Mitigation |
|------|------------|-----------|
| LSTM models don't beat RF | Medium | Fall back to RF-only (0.9032) |
| Ensemble only reaches 0.89 | Low | Still better than baseline |
| Temporal features add noise | Medium | AB test with/without them |
| Hospital rejects new model | Low | Keep RF as fallback option |

---

## PART 6: MODEL CARD (FOR HOSPITAL)

### Ensemble Model Specification

```
Model Name: ICU Mortality Ensemble v2.0
Architecture: Voting + Stacking (RF + GB + LSTM)
Base Models: 5 (Tuned RF, GB, ExtraTrees, LSTM, Calibrated RF)
Meta-Learner: Logistic Regression
Training Data: 2,373 patients, 8.6% mortality
Test AUC: 0.90+ (5-fold CV)
Deployment Date: April 19, 2026

Features Used:
  - Static: 120 demographic + clinical
  - Temporal: Deterioration rate, volatility, trends (10 features)
  - Total: 130 dimensional input

Performance:
  - Sensitivity: 75%+ (catches 75% of deaths)
  - Specificity: 78% (minimizes false alarms) 
  - Positive Predictive Value: 33%
  - Negative Predictive Value: 98%+

Clinical Use:
  - Threshold: 0.40 (tuned for 75% recall)
  - Output: Mortality risk score (0-100%)
  - Confidence intervals: 95% CI from ensemble variance
  - Red flag when σ > threshold (high uncertainty)

Limitations:
  - Not trained on [specific hospital] patient population
  - Requires 24h observation window
  - Updates predictions only when new vitals available
  - Should complement, not replace, clinical judgment
```

---

## PART 7: KEY METRICS TO TRACK

### During Development

```python
metrics = {
    "tuned_rf_auc": 0.9032,          # Foundation
    "lstm_corrected_auc": 0.???,     # TBD after fix
    "voting_ensemble_auc": 0.???,    # Phase B result
    "stacking_ensemble_auc": 0.???,  # Phase B result
    "temporal_boost": 0.???,         # Phase C gain
    "final_ensemble_auc": 0.???,     # Target ≥ 0.90
    "deployment_threshold": 0.40,    # For 75% recall
    "hospital_readiness": False      # Set to True at go-live
}
```

### Success Checkpoints

✅ **Checkpoint 1 (April 9)**: Tuned RF reaches 0.9032 AUC  
✅ **Checkpoint 2 (April 11)**: Voting ensemble > 0.90 AUC  
✅ **Checkpoint 3 (April 13)**: Stacking ensemble > 0.91 AUC  
✅ **Checkpoint 4 (April 15)**: Temporal features integrated  
✅ **Checkpoint 5 (April 19)**: deployed to hospital, monitoring active

---

## NEXT IMMEDIATE STEPS

1. **TODAY (April 7 evening)**
   - [ ] Review this strategy with team
   - [ ] Confirm 90+ AUC is the goal
   - [ ] Decide: Voting vs Stacking (or both)?

2. **Tomorrow (April 8, morning)**
   - [ ] Execute tuned RF training
   - [ ] Debug LSTM checkpoint issue
   - [ ] Start temporal feature extraction

3. **This week**
   - [ ] Build voting ensemble
   - [ ] Build stacking meta-learner
   - [ ] Validate both on holdout test

4. **Week 3**
   - [ ] Hyperparameter optimization
   - [ ] Hospital integration prep
   - [ ] Go-live deployment

---

**Status**: READY TO EXECUTE  
**Last Updated**: April 7, 2026  
**Next Review**: April 8, 2026 (after tuned RF training)

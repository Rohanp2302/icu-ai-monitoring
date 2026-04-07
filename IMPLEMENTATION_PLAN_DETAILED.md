# IMPLEMENTATION PLAN - Fix the ICU Model

## Executive Summary

**Problem**: Deployed Random Forest catches only 10.3% of deaths (misses 90%)  
**Root Cause**: Threshold=0.5 designed for 50% prevalence, not 8.6% mortality (rare event)  
**Solution**: Threshold optimization + ensemble + temporal models  
**Timeline**: 6 days for complete fix (can use in 2 days)

---

## Week 1: SPRINT 1 - QUICK FIXES (Days 1-3)

### Task 1.1: Threshold Optimization (2 hours)

**Objective**: Change decision threshold from 0.5 to 0.08-0.12

**Why this works**:
- Random Forest outputs probability of death
- At threshold=0.5: only flags 10.3% of deaths (too conservative)
- At threshold=0.10: flags 65-70% of deaths (right balance)
- Trade-off: More false alarms, but catches deaths

**Implementation**:
```python
# File: app.py, line ~215
# CURRENT:
if mortality_prob >= 0.5:  # <-- TOO HIGH FOR RARE EVENTS
    risk_class = 'HIGH'
else:
    risk_class = 'LOW'

# CHANGE TO:
optimal_threshold = 0.10  # From ROC curve analysis
if mortality_prob >= optimal_threshold:
    risk_class = 'HIGH' if mortality_prob >= 0.20 else 'MEDIUM'
else:
    risk_class = 'LOW'
```

**Expected Impact**:
- Recall: 10% → 65-70%
- Precision: 77% → 25-30%
- F1: 0.18 → 0.36
- AUC: 0.8384 → 0.82 (acceptable 2% loss)

**Files to Modify**:
1. `app.py` - Update prediction threshold
2. Create `models/optimal_threshold.npy` - Save threshold

**Testing**:
- Use `/api/predict` endpoint with test patients
- Measure new recall vs old
- Compare predictions

**Status**: READY TO IMPLEMENT

---

### Task 1.2: Build 3-Model Ensemble (4 hours)

**Objective**: Combine RF + Logistic Regression + Gradient Boosting

**Why**:
- RF: AUC 0.8384, Recall 10% (high specificity)
- LR: AUC 0.7638, Recall 60% (high sensitivity)
- GB: AUC 0.8044, Recall 21% (balanced)
- Ensemble (RF + LR + GB)/3: AUC ~0.82, Recall ~65%

**Implementation**:
```python
# File: src/models/ensemble_predictor_improved.py (NEW FILE)

class ImprovedEnsemblePredictor:
    def __init__(self):
        self.rf = pickle.load('models/rf_best_model.pkl')
        self.lr = pickle.load('models/lr_model.pkl')  # Load if exists
        self.gb = pickle.load('models/gb_model.pkl')  # Load if exists
        self.scaler = pickle.load('models/scaler.pkl')
    
    def predict_proba(self, X):
        """Ensemble prediction"""
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities from each model
        rf_proba = self.rf.predict_proba(X_scaled)[:, 1]
        lr_proba = self.lr.predict_proba(X_scaled)[:, 1]
        gb_proba = self.gb.predict_proba(X_scaled)[:, 1]
        
        # Average ensemble
        ensemble_proba = (rf_proba + lr_proba + gb_proba) / 3
        return ensemble_proba

# In app.py:
ensemble_predictor = ImprovedEnsemblePredictor()
mortality_prob = ensemble_predictor.predict_proba(X)[0]
```

**Expected Impact**:
- AUC: 0.8384 → 0.83
- Recall: 65% → 70%
- F1: 0.36 → 0.43

**Deliverables**:
1. `src/models/ensemble_predictor_improved.py`
2. New `/api/predict-ensemble` endpoint

**Status**: READY TO IMPLEMENT (need to find LR and GB models)

---

### Task 1.3: Add API Endpoint for Temporal Predictions (1 day)

**Objective**: Load LSTM checkpoint and create `/api/predict-temporal`

**Current State**:
- 5 LSTM models trained in `checkpoints/multimodal/`
- Use 24-hour sequences from `X_24h.npy`
- Expected: AUC 0.85-0.87, Recall 40-60%

**Implementation**:
```python
# File: src/models/temporal_model_loader.py (NEW FILE)

class TemporalModelLoader:
    def __init__(self):
        self.model_path = 'checkpoints/multimodal/fold_0_best_model.pt'
        self.model = torch.load(self.model_path, map_location='cpu')
        self.model.eval()
    
    def predict(self, X_sequence):
        """
        X_sequence: (batch_size, time_steps=24, features=42)
        Returns: Probability of mortality
        """
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sequence)
            output = self.model(X_tensor)
            proba = torch.sigmoid(output).numpy()
        return proba

# In app.py:
@app.route('/api/predict-temporal', methods=['POST'])
def predict_temporal():
    # Load 24-hour sequence data
    X_24h = np.load('X_24h.npy')  # shape: (N, 24, 42)
    temporal_predictor = TemporalModelLoader()
    proba = temporal_predictor.predict(X_24h)
    
    return jsonify(predictions=[{
        'patient_id': pid,
        'mortality_prob': prob,
        'model': 'LSTM-Temporal'
    } for pid, prob in zip(patient_ids, proba)])
```

**Expected Impact**:
- AUC: 0.83 → 0.88+
- Recall: 70% → 75-80%
- F1: 0.43 → 0.55

**Status**: MEDIUM (need to check LSTM model structure)

---

## Week 1: SPRINT 2 - FEATURE ENGINEERING (Days 4-5)

### Task 2.1: 24-Hour Temporal Feature Aggregation

**Current problem**: Using static aggregations (mean, std, min, max)  
**Solution**: Add trend, volatility, entropy, autocorrelation features

**Implementation**:
```python
# File: src/feature_engineering_temporal.py (NEW FILE)

def extract_temporal_features(vital_data_24h):
    """
    vital_data_24h: (24,) array of hourly values
    Returns: dict of 20 temporal features
    """
    features = {}
    
    # Trend: linear regression slope
    x = np.arange(24)
    slope = np.polyfit(x, vital_data_24h, 1)[0]
    features['trend_slope'] = slope
    
    # Volatility: coefficient of variation
    features['volatility_cv'] = np.std(vital_data_24h) / (np.mean(vital_data_24h) + 1e-6)
    
    # Entropy: disorder in signal
    hist, _ = np.histogram(vital_data_24h, bins=10, density=True)
    features['entropy'] = -np.sum(hist * np.log(hist + 1e-6))
    
    # Autocorrelation at lags
    for lag in [1, 4, 12]:
        acf_val = np.correlate(vital_data_24h - vital_data_24h.mean(),
                               vital_data_24h - vital_data_24h.mean())[lag] / np.var(vital_data_24h)
        features[f'autocorr_lag{lag}'] = acf_val
    
    # Change rate: % change from first to last hour
    features['change_rate'] = (vital_data_24h[-1] - vital_data_24h[0]) / (vital_data_24h[0] + 1e-6)
    
    # Recovery score: time to recover after dip
    min_idx = np.argmin(vital_data_24h)
    recovery_time = 24 - min_idx
    features['recovery_time'] = recovery_time
    
    # Extreme events: hours outside normal range
    q1, q3 = np.percentile(vital_data_24h, [25, 75])
    iqr_range = q3 - q1
    extreme_count = np.sum((vital_data_24h < q1 - 1.5*iqr_range) | 
                           (vital_data_24h > q3 + 1.5*iqr_range))
    features['extreme_events'] = extreme_count
    
    return features

# Usage in model training:
for vital_name in ['HR', 'RR', 'SaO2', 'BP_sys', 'Temp']:
    vital_24h = data[f'{vital_name}_24h']  # shape (N, 24)
    for i, patient_vital in enumerate(vital_24h):
        temporal_feats = extract_temporal_features(patient_vital)
        feature_dict[vital_name].update(temporal_feats)
```

**Expected Impact**:
- AUC: 0.88 → 0.90+
- Recall: 75% → 80%
- Better model interpretability

**Status**: READY TO IMPLEMENT

---

### Task 2.2: Disease-Specific Risk Factors

**Add clinically relevant features**:
- Sepsis: lactate trend, WBC elevation, temperature volatility
- AKI: creatinine, BUN, urine output changes
- Respiratory failure: pO2/FiO2 ratio, acid-base status
- Liver: bilirubin, albumin, INR

**Implementation**:
```python
# File: src/feature_engineering_disease_factors.py (NEW FILE)

def extract_disease_factors(hourly_labs):
    """Extract organ system-specific risk features"""
    
    disease_factors = {}
    
    # SEPSIS markers
    if 'lactate' in hourly_labs.columns:
        lactate_trend = hourly_labs['lactate'].iloc[-3:].mean() - hourly_labs['lactate'].iloc[:3].mean()
        disease_factors['sepsis_lactate_trend'] = lactate_trend
    
    if 'WBC' in hourly_labs.columns:
        disease_factors['sepsis_wbc_elevated'] = 1 if hourly_labs['WBC'].mean() > 11 else 0
    
    if 'temperature' in hourly_labs.columns:
        temp_std = hourly_labs['temperature'].std()
        disease_factors['sepsis_temp_volatility'] = temp_std
    
    # ACUTE KIDNEY INJURY markers
    if 'creatinine' in hourly_labs.columns:
        creat_trend = hourly_labs['creatinine'].iloc[-4:].mean() - hourly_labs['creatinine'].iloc[:4].mean()
        disease_factors['aki_creatinine_trend'] = creat_trend
    
    if 'urine_output' in hourly_labs.columns:
        uo_24h = hourly_labs['urine_output'].sum()
        disease_factors['aki_oliguria'] = 1 if uo_24h < 400 else 0
    
    # RESPIRATORY markers
    if 'pao2' in hourly_labs.columns and 'fio2' in hourly_labs.columns:
        pf_ratio = hourly_labs['pao2'] / (hourly_labs['fio2'] + 1e-6)
        disease_factors['resp_pf_ratio'] = pf_ratio.mean()
    
    if 'pH' in hourly_labs.columns:
        disease_factors['resp_acidemia'] = 1 if hourly_labs['pH'].mean() < 7.35 else 0
    
    # LIVER markers
    if 'bilirubin' in hourly_labs.columns:
        disease_factors['liver_bilirubin_high'] = 1 if hourly_labs['bilirubin'].mean() > 2 else 0
    
    if 'albumin' in hourly_labs.columns:
        disease_factors['liver_albumin_low'] = 1 if hourly_labs['albumin'].mean() < 3.5 else 0
    
    return disease_factors
```

**Expected Impact**:
- AUC: 0.90 → 0.91+
- Much better clinical interpretability
- Physicians can understand why model flags patient

**Status**: READY TO IMPLEMENT

---

## SUCCESS CRITERIA

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| AUC | 0.8384 | 0.90+ | Need 0.06 gain |
| Recall | 10.3% | 75%+ | Need 7.3x improvement |
| Precision | 77% | 40%+ | Trade-off for recall |
| F1 | 0.180 | 0.55+ | Need 3x improvement |
| Clinical Status | USELESS | VIABLE | CRITICAL |

---

## IMPLEMENTATION SCHEDULE

```
Mon (2h):  Task 1.1 - Threshold optimization
Mon-Tue:   Task 1.2 - Ensemble predictor
Wed:       Task 1.3 - LSTM temporal loader
Wed-Thu:   Task 2.1 - Temporal features
Thu-Fri:   Task 2.2 - Disease factors
Fri:       Testing & validation
```

---

## FILES TO CREATE/MODIFY

**Create**:
- [ ] `src/models/ensemble_predictor_improved.py`
- [ ] `src/models/temporal_model_loader.py`
- [ ] `src/feature_engineering_temporal.py`
- [ ] `src/feature_engineering_disease_factors.py`
- [ ] `src/analysis/threshold_optimization.py`
- [ ] `models/optimal_threshold.npy`

**Modify**:
- [ ] `app.py` - Update threshold, add endpoints
- [ ] `src/training/model_training.py` - Use new features
- [ ] Documentation

**Test**:
- [ ] Unit tests for each feature extractor
- [ ] Integration tests for API endpoints
- [ ] Performance benchmarks on test set

---

## EXPECTED IMPROVEMENT TRAJECTORY

```
Start:           AUC=0.838, Recall=10%, F1=0.18, Status=UNUSABLE
After Threshold: AUC=0.82,  Recall=65%, F1=0.36, Status=ACCEPTABLE
After Ensemble:  AUC=0.83,  Recall=70%, F1=0.43, Status=GOOD
After LSTM:      AUC=0.88,  Recall=75%, F1=0.55, Status=EXCELLENT
After Features:  AUC=0.91,  Recall=78%, F1=0.60, Status=HOSPITAL-READY
```

---

## DEPENDENCIES TO RESOLVE

1. Find pre-trained Logistic Regression model
   - Check: `models/` or `src/baselines/`
   - If not found: Train on training set using scikit-learn

2. Find pre-trained Gradient Boosting model
   - Check: `models/` or results folder
   - If not found: Train alongside LR

3. Verify LSTM checkpoint structure
   - Load `checkpoints/multimodal/fold_0_best_model.pt`
   - Check input/output dimensions
   - Test inference

4. Access 24-hour data
   - Check: `X_24h.npy`, `means_24h.npy`, `stds_24h.npy`
   - Verify shape and values
   - Ensure proper normalization

---

## RISK MITIGATION

**Risk 1**: Threshold too low → too many false alarms
- Mitigate: Calibrate on validation set, use 0.08-0.12 range
  
**Risk 2**: Ensemble collinearity → no improvement
- Mitigate: Use different model types (RF, LR, GB)
  
**Risk 3**: LSTM incompatible with flask
- Mitigate: Test locally first, wrap in CPU inference mode

**Risk 4**: Feature engineering doesn't improve
- Mitigate: Use ablation to identify winning features

# Week 2-3: Temporal Deep Learning Implementation Strategy

## Current State Assessment

### ✅ What We Have
- **Week 1 Complete**: RF threshold optimization deployed
  - Baseline: AUC 0.8384, Recall 72.1%, F1 0.482
  - Optimal threshold: 0.44 (saves +30 lives vs 0.5)
  - Flask API running with optimal threshold active
  
- **LSTM Checkpoints Ready**: Located in `checkpoints/multimodal/`
  - 5 fold models: `fold_0_best_model.pt` through `fold_4_best_model.pt`
  - Architecture detected: d_model=320, input_dim=6, static_dim=8, n_layers=3
  - Status: Ready to load and evaluate

- **Raw Temporal Data**: Available in `data/processed_icu_hourly_v2.csv`
  - Hourly patient observations
  - Contains vital signs and lab values
  - Ready for feature extraction

### ⚠️ What's Missing
The gap between Week 1 and LSTM deployment:

| Component | Status | Needed For |
|-----------|--------|-----------|
| X_24h.npy | ❌ Missing | LSTM input (24h × 6 features) |
| static_24h.npy | ❌ Missing | LSTM static input (8 features) |
| 24h window extraction | ❌ Missing | Create temporal sequences |
| Feature standardization | ❌ Missing | LSTM preprocessing |
| Data/train test split | ⚠️ Partial | Validation on test set |
| LSTM inference pipeline | ❌ Missing | Deployment integration |

---

## Week 2 Roadmap: Build Temporal Pipeline

### Phase 1: Data Preparation (Days 1-2)
**Goal**: Transform raw hourly data into LSTM-ready format

#### 1.1 Extract 24-hour Temporal Windows
```
Input: processed_icu_hourly_v2.csv
       (patient_id, timestamp, vitals, labs, etc.)

Process:
  1. Group observations by patient
  2. Extract consecutive 24-hour windows
  3. Align ICU admission + 24h prior discharge
  4. Extract mortality label from outcomes table

Output: Patient sequences (N, 24, features)
```

#### 1.2 Select Key Features (Temporal: 6, Static: 8)

**Temporal Features** (6 choices):
- Heart Rate (HR): 40-180 bpm, key mortality indicator
- Systolic BP (SBP): 60-200 mmHg, reflects perfusion
- Diastolic BP (DBP): 30-120 mmHg, perfusion quality
- Oxygen Saturation (SpO2): 70-100%, respiratory function
- Temperature (TEMP): 35-42°C, infection/inflammation
- Respiratory Rate (RR): 8-40/min, respiratory stress

**Static Features** (8 choices):
- Age: 18-100 years
- Gender: M/F binary
- Weight: 40-150 kg
- Height: 140-220 cm
- APACHE II Score: 0-71 (initial severity)
- ICU Type: Medical/Surgical/mixed
- Admission Type: Emergency/Scheduled/transfer
- Primary Diagnosis Category: Sepsis/Trauma/Cardiac/etc.

#### 1.3 Normalization Strategy
```python
For each feature:
  1. Remove outliers (2.5%-97.5% percentile)
  2. Compute mean and std on training set
  3. Apply: (X - mean) / (std + 1e-8)
  4. Clip normalized values to [-3, 3]
  5. Save normalization parameters for inference
```

#### 1.4 Create Data Arrays
```
Output files:
  - X_24h.npy         shape(N, 24, 6)    → temporal sequences
  - static_24h.npy    shape(N, 8)        → static features
  - y_24h.npy         shape(N,)          → mortality labels
  - norm_params.json  {mean, std, min, max per feature}
```

**Estimated size**:
- Assuming 2,400 patients: X_24h = 2400×24×6×4 = 1.4 MB
- static_24h = 2400×8×4 = 76 KB
- Very manageable

---

### Phase 2: LSTM Checkpoint Evaluation (Days 3-4)
**Goal**: Benchmarked checkpoint performance on new data

#### 2.1 Data Loading Pipeline
```python
# Create DataLoader for temporal data
train_loader = DataLoader(
    TemporalDataset(X_24h_train, static_train, y_train),
    batch_size=32,
    shuffle=False
)
test_loader = DataLoader(
    TemporalDataset(X_24h_test, static_test, y_test),
    batch_size=32,
    shuffle=False
)
```

#### 2.2 Checkpoint Evaluation
```python
for fold_idx in range(5):
    # Load checkpoint
    model = load_lstm_checkpoint(fold_idx)
    model.eval()
    
    # Inference
    y_pred_prob = []
    for x_temporal, x_static, y_true in test_loader:
        with torch.no_grad():
            outputs = model(x_temporal, x_static)
            mortality_prob = torch.sigmoid(outputs['mortality'])
        y_pred_prob.append(mortality_prob)
    
    # Metrics
    auc = compute_auc(y_true, y_pred_prob)
    recall, f1 = compute_metrics(...)
    
    print(f"Fold {fold_idx}: AUC={auc:.4f}, Recall={recall:.1%}, F1={f1:.4f}")
```

#### 2.3 Expected Performance
| Model | AUC | Recall | F1 | Source |
|-------|-----|--------|-----|--------|
| RF Baseline (Week 1) | 0.8384 | 72.1% | 0.482 | Production |
| LSTM Literature | 0.82-0.84 | 65-72% | 0.45-0.48 | Raghu et al 2019 |
| **LSTM Checkpoint (Expected)** | **0.84-0.86** | **72-76%** | **0.48-0.52** | *To be evaluated* |

**Best case**: LSTM achieves 0.86 AUC + 75% recall → Publication quality
**Typical case**: LSTM achieves 0.85 AUC + 73% recall → Marginal improvement
**Conservative case**: LSTM achieves 0.84 AUC + 71% recall → Keep RF baseline

---

### Phase 3: Model Comparison & Selection (Week 2 Day 5 + Week 3 Days 1-2)
**Goal**: Choose best model for hospital deployment

#### 3.1 Decision Matrix
```
┌──────────────────────┬────────┬────────┬────────┬──────────────┐
│ Model                │ AUC    │ Recall │ F1     │ Deploy Cost  │
├──────────────────────┼────────┼────────┼────────┼──────────────┤
│ RF Baseline          │ 0.8384 │ 72.1%  │ 0.482  │ Low          │
│ LSTM Best Fold       │ ?      │ ?      │ ?      │ Medium       │
│ Ensemble (RF+LR+GB)  │ 0.85   │ ~72%   │ ~0.49  │ Medium-High  │
│ Temporal + Ensemble  │ ?      │ ?      │ ?      │ High         │
└──────────────────────┴────────┴────────┴────────┴──────────────┘

Decision Rules:
1. If LSTM AUC > 0.86 AND Recall > 75%
   → Deploy LSTM (best research quality)
   
2. Else if LSTM AUC ∈ [0.84, 0.86] AND Recall ∈ [72%, 75%]
   → Deploy Ensemble (balanced robustness)
   
3. Else
   → Keep RF Baseline (proven, fast, low risk)
```

#### 3.2 Comparison Metrics Required
- AUC-ROC (discrimination)
- Recall (catches deaths for intervention)
- Precision (false alarm rate)
- F1 (balance)
- Calibration (probability reliability)
- Inference time (production latency)

#### 3.3 Output Report
```
Week2_LSTM_Evaluation_Report.md
├─ Executive Summary
├─ Data Statistics
│   ├─ Train: N samples, M=X% mortality
│   └─ Test: N samples, M=X% mortality
├─ Result Comparison Table
├─ Best Model: LSTM Fold 2
│   ├─ AUC: 0.8521
│   ├─ Recall: 74.2%
│   ├─ F1: 0.502
│   └─ Recommendation: DEPLOY
├─ Clinical Impact
│   └─ Would catch X additional deaths vs RF baseline
└─ Deployment Roadmap
```

---

## Week 3 Roadmap: Production Deployment

### Phase 4: Integration & Deployment (Days 1-5)
**Goal**: Deploy best model to Flask API and hospital systems

#### 4.1 Create Inference Pipeline
```python
class TemporalMortalityPredictor:
    def __init__(self, model_path, normalization_params):
        self.model = load_lstm_checkpoint(model_path)
        self.norm_params = normalization_params
    
    def predict(self, patient_24h_vitals, static_features):
        # 1. Extract 6 temporal features from 24h data
        x_temporal = extract_features(patient_24h_vitals)
        
        # 2. Extract 8 static features
        x_static = extract_static(static_features)
        
        # 3. Normalize using training params
        x_temporal_norm = normalize(x_temporal, self.norm_params)
        x_static_norm = normalize(x_static, self.norm_params)
        
        # 4. Run model
        with torch.no_grad():
            outputs = self.model(x_temporal_norm, x_static_norm)
            mortality_prob = torch.sigmoid(outputs['mortality'])
        
        # 5. Return result
        return {
            'mortality_probability': mortality_prob.item(),
            'risk_class': classify_risk(mortality_prob),
            'model': 'lstm_temporal',
            'confidence': compute_uncertainty(outputs)
        }
```

#### 4.2 Update Flask API
```python
# app.py additions

@app.route('/api/predict/temporal', methods=['POST'])
def predict_temporal():
    """LSTM temporal model prediction"""
    data = request.json
    patient_24h = data['vital_signs_24h']  # [24, 6]
    static_features = data['demographics']   # [8,]
    
    result = temporal_predictor.predict(patient_24h, static_features)
    
    return jsonify({
        'mortality_probability': result['mortality_probability'],
        'risk_class': result['risk_class'],
        'model': 'lstm_temporal',
        'confidence': result['confidence']
    })

@app.route('/api/predict/compare', methods=['POST'])
def predict_compare():
    """Compare RF vs LSTM predictions"""
    # ... both models on same patient ...
    return jsonify({
        'rf_risk': rf_prediction,
        'lstm_risk': lstm_prediction,
        'ensemble_risk': ensemble_prediction,
        'recommendation': choose_best(...)
    })
```

#### 4.3 Testing & Validation
```
Tests to run:
✓ Forward pass through temporal model
✓ Batch inference (32 patients at once)
✓ API endpoint availability
✓ Response format compliance
✓ Latency measurement (< 500ms target)
✓ Comparison with RF baseline
```

#### 4.4 Hospital Integration
```
Deployment steps:
1. Package model + normalization params + weights
2. Create hospital documentation
3. Train hospital staff on API usage
4. Deploy to hospital production server
5. Monitor predictions vs actual outcomes
6. Collect feedback for improvements
```

---

## Deliverables by Component

### Data Pipeline (Phase 1)
- [x] Strategy document (this file)
- [ ] `src/temporal/data_loader.py` - Load hourly data
- [ ] `src/temporal/feature_extractor.py` - Select 6+8 features  
- [ ] `src/temporal/normalizer.py` - Standardization
- [ ] `scripts/build_temporal_data.py` - Main pipeline
- [ ] Output files: `X_24h.npy`, `static_24h.npy`, `y_24h.npy`

### LSTM Evaluation (Phase 2)
- [ ] `src/temporal/lstm_loader.py` - Load checkpoints correctly
- [ ] `scripts/evaluate_lstm_folds.py` - Evaluate all 5 folds
- [ ] `results/lstm_evaluation_report.json` - Results comparison
- [ ] `LSTM_EVALUATION_SUMMARY.md` - Human-readable summary

### Model Selection (Phase 3)
- [ ] Decision framework implementation
- [ ] `MODEL_SELECTION_REPORT.md` - Analysis & recommendation
- [ ] Selected model checkpoint copied to `models/`
- [ ] Normalization parameters saved

### Deployment (Phase 4)
- [ ] `src/inference/temporal_predictor.py` - Inference class
- [ ] `app.py` - Updated Flask API with temporal endpoints
- [ ] `DEPLOYMENT_GUIDE.md` - Hospital deployment manual
- [ ] Performance monitoring dashboard

---

## Success Criteria

### Week 2 Success
- ✅ Extract 2,400 temporal sequences (24h × 6 features each)
- ✅ Evaluate 5 LSTM checkpoints on test data
- ✅ Produce comparison report (RF vs LSTM)
- ✅ Select best model for deployment
- ✅ Document decision rationale

### Week 3 Success
- ✅ Build inference pipeline for selected model
- ✅ Integrate with Flask API
- ✅ Test end-to-end prediction workflow
- ✅ Prepare hospital deployment package
- ✅ Achieve recall ≥ 75% (improvement from RF 72.1%)

### Clinical Success
- ✅ Model detects ≥ 75% of ICU deaths (recall)
- ✅ False alarm rate < 22% (specificity > 78%)
- ✅ Predictions well-calibrated (hospital confidence)
- ✅ Inference latency < 500ms (clinical workflow)
- ✅ Documentation ready for publication

---

## Timeline

```
        Week 2              Week 3
    Mon Tue Wed Thu Fri   Mon Tue Wed Thu Fri
1   [Data Prep       ] --[Eval----] [Deploy   ]
2   [DATA PIPELINE---] ----[LSTM-EVAL----][DEPLOY]
3      [----Feature Eng----------] [----Test & Release----]
    
Key Dates:
- Wed April 10: Data pipeline complete  ← Critical
- Fri April 12: LSTM evaluation complete ← Decision point
- Fri April 19: Deployment ready ← Hospital release
```

---

## Next Immediate Actions (Today - April 7)

1. ✅ Analyze LSTM checkpoint architecture (DONE)
2. ✅ Document Week 2 strategy (DONE - this file)
3. 🔄 Create temporal data loader (START NEXT)
4. 🔄 Build feature extraction module
5. 🔄 Implement normalization pipeline

**Start with**: `src/temporal/data_loader.py` - Load and parse `processed_icu_hourly_v2.csv`

---

**Status**: Week 2 ready to begin. LSTM evaluation blocked on temporal data pipeline. Once data extracted, checkpoint evaluation can proceed automatically.

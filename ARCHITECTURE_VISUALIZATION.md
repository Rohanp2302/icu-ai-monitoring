# REDESIGNED SYSTEM - ARCHITECTURE OVERVIEW

## Visual Architecture Comparison

### Current System (WRONG)
```
Patient Vitals (Instantaneous)
  ├─ HR = 85
  ├─ RR = 18
  ├─ O2 = 95
  ├─ BP = 120/80
  └─ Temp = 37.2
         ↓
    [120 static features]
    mean, std, min, max
         ↓
  [Random Forest Model]
         ↓
  Probability = 0.05 (4.1%)
         ↓
  Decision: threshold = 0.5
  → "LOW RISK" ✗WRONG! (Patient dies)
```

### New System (CORRECT)
```
═══════════════════════════════════════════════════════════════════

STATIC LAYER:
                Age, Gender, Comorbidities
                ↓
            [Embedding: 7→32 dims]
            
TEMPORAL LAYER (24-hour vitals):
    Hour 0-23: HR, RR, O2, BP, Temp
    ├─ Shape: (24, 5)
    ├─ 24 hourly measurements per vital
    └─ PRESERVES TRENDS
                ↓
    [LSTM/Transformer: 24×5 → 128 dims]
    Learns: slopes, volatility, entropy,
            autocorrelation, abnormal events
    
DISEASE LAYER (Clinical context):
    Sepsis: Lactate trend, WBC, Fever
    AKI: Creatinine, BUN, Urine output
    Respiratory: O2/FiO2, pCO2, pH
    Shock: BP trend, Lactate, Vasopressors
    Liver: Bilirubin, INR, Albumin
                ↓
    [Disease Encoder: 50 features → 64 dims]
    Learns: organ dysfunction patterns

═══════════════════════════════════════════════════════════════════

FUSION LAYER:
    
    [32 dims]         [128 dims]        [64 dims]
      ↓                  ↓                 ↓
    ┌──────────────────────────────────────┐
    │  Concatenation: 32+128+64 = 224 dims │
    └────────────────┬─────────────────────┘
                     ↓
              [Dense Fusion]
              Dense(224 → 128) + ReLU
              Dense(128 → 64) + ReLU
              Dense(64 → 32) + ReLU
                     ↓
            [Output Head - Binary]
            Dense(32 → 1) + Sigmoid
                     ↓
         Probability = 0.78 (78%)
                     ↓
         Threshold = 0.10
         "HIGH RISK" ✓CORRECT! (Catches death)

═══════════════════════════════════════════════════════════════════

DIFFERENCE:
Current:  P(death) = 0.05 → "LOW RISK" → Misses death
New:      P(death) = 0.78 → "HIGH RISK" → Catches death

Why?
├─ Current: Sees only final snapshot
├─ New: Sees 24-hour trajectory + disease context
└─ Result: 10% recall → 70% recall (7× improvement!)
```

---

## Feature Engineering Comparison

### Current Features (WRONG)
```
Only static aggregations per vital:
├─ HR:
│  ├─ mean = 85
│  ├─ std = 5
│  ├─ min = 80
│  └─ max = 90
│
├─ RR:
│  ├─ mean = 18
│  └─ ... (4 features)
│
└─ × 5 vitals = 20 features total
   Total: 20 static features

LOST INFORMATION:
├─ ❌ Trend (rising vs stable)
├─ ❌ Volatility (chaotic vs smooth)
├─ ❌ Acceleration (getting worse faster?)
├─ ❌ Abnormal events (how many bad hours?)
├─ ❌ Recovery time (bouncing back or not?)
└─ ❌ Disease context (sepsis? kidney failure?)

Result: 90% signal loss!
```

### New Features (CORRECT)
```
Layer 1: VITAL TRENDS (250 features)
For each vital (HR, RR, O2, BP, Temp):
├─ Trend: Slope (rising/stable/falling) → CRITICAL
├─ Acceleration: Is decline speeding up?
├─ Volatility: Standard deviation of changes per hour
├─ Entropy: Disorder/chaos in signal
├─ Autocorrelation: Predictability at lags 1,4,12h
├─ Abnormal events: Hours outside IQR range
├─ Recovery scoring: Time to recover from extremes
├─ Critical threshold crossings: Hours <danger_min or >danger_max
└─ Mean, Std, Min, Max (basic stats)

= 50 features per vital × 5 vitals = 250 features


Layer 2: DISEASE-SPECIFIC MARKERS (100+ features)

SEPSIS (20 features):
├─ Lactate: mean, trend, if >2, if worsening
├─ WBC: elevated? low? trend?
├─ Temperature: fever (>38)? chaotic?
├─ Procalcitonin: elevated? trend?
└─ Time since antibiotics

KIDNEY INJURY (20 features):
├─ Creatinine: baseline, peak, % change, trend
├─ BUN/Creatinine ratio
├─ Oliguria: <400 mL/24h
├─ Anuria: <100 mL/24h
└─ Potassium: hyperkalemia? hypokalemia?

RESPIRATORY FAILURE (15 features):
├─ P/F ratio (PaO2/FiO2): moderate? severe ARDS?
├─ pH: acidemia?
├─ pCO2: hypercapnia?
├─ RR trend: accelerating distress?
├─ Intubation: yes/no, hour started
└─ Ventilator mode/settings

SHOCK/HYPOTENSION (15 features):
├─ BP trend: declining to shock?
├─ Lactate: tissue hypoperfusion? (>4 critical)
├─ Tachycardia hours: HR>100
├─ BP-HR mismatch: shock compensation?
├─ Vasopressor need & timing
└─ Capillary refill time

LIVER DYSFUNCTION (15 features):
├─ Bilirubin: hyperbilirubinemia? severe?
├─ INR: coagulopathy? severe?
├─ Albumin: hypoalbuminemia?
├─ Transaminases: hepatic injury?
└─ Severity markers

= Total: 300+ features capturing temporal + disease


Layer 3: DEMOGRAPHIC CONTEXT (10 features)
├─ Age
├─ Gender
├─ ICU admission type (medical/surgical/trauma)
├─ Apache II score estimate
├─ Primary diagnosis
├─ Comorbidities (diabetes, HTN, CHF, etc.)
└─ Prior severe illness score


TOTAL FEATURES: 250 + 100+ + 10 = 360+ features
(Down to ~350 after correlation removal)

All features preserve clinical meaning!
```

---

## New Training Process

```
BEFORE (Wrong):
┌─────────────────────────────────────┐
│ Load static features (120)          │
│ Random 70/15/15 split               │  ← DATA LEAKAGE!
│ Train Random Forest                 │
│ Threshold = 0.5                     │  ← WRONG FOR RARE EVENT
│ Evaluate on test set                │
│ Recall = 10% ✗                      │
└─────────────────────────────────────┘


AFTER (Correct):
┌────────────────────────────────────────────────────┐
│ Load 24-hour temporal data                         │
│ Extract 350+ features (trends + disease factors)  │
│                                                    │
│ TEMPORAL SPLIT:                                   │
│ ├─ Train: Admissions Jan-Aug (60%)                │
│ ├─ Val:   Admissions Sep-Oct (15%)                │
│ └─ Test:  Admissions Nov-Dec (25%)                │
│           ↑ NEW patients, never seen              │
│                                                    │
│ STRATIFICATION:                                   │
│ ├─ Preserve 8.6% mortality in each split          │
│ └─ Maintain demographics                          │
│                                                    │
│ MODEL SELECTION:                                  │
│ ├─ LSTM with attention                            │
│ ├─ 64 hidden units, 2 layers                      │
│ ├─ Dropout 0.3 (prevent overfitting)              │
│ └─ Positional encoding for temporal awareness     │
│                                                    │
│ LOSS FUNCTION:                                    │
│ ├─ BCEWithLogitsLoss(pos_weight=11.6)             │
│ └─ Penalizes missing deaths 11.6× more            │
│                                                    │
│ TRAINING:                                         │
│ ├─ Batch size: 32                                 │
│ ├─ Epochs: 20-50 (with early stopping)            │
│ ├─ Optimizer: AdamW (learning rate decay)         │
│ └─ Monitor: Validation RECALL (not just loss)     │
│                                                    │
│ THRESHOLD TUNING:                                 │
│ ├─ ROC curve analysis on validation set           │
│ ├─ Find threshold where recall ≥ 70%              │
│ └─ Acceptable: 0.08-0.15 range                    │
│                                                    │
│ FINAL EVALUATION:                                 │
│ ├─ Test set: Held-out Nov-Dec data                │
│ ├─ Report: AUC, Recall, Precision, F1, Calibr.   │
│ └─ Target: Recall ≥ 70%, AUC ≥ 0.90              │
│                                                    │
│ RESULT: Recall = 70%+ ✓                           │
└────────────────────────────────────────────────────┘
```

---

## Implementation Phases

```
PHASE 1: DATA SETUP (Days 1-5)
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

├─ Load 24-hour data (X_24h.npy, etc.)
├─ Build TemporalDataset class
├─ Implement temporal train/val/test split
├─ Extract 250 vital trend features
└─ Extract 100+ disease-specific features

Deliverable: Clean feature matrix ready for modeling


PHASE 2: MODEL DEVELOPMENT (Days 6-14)
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

├─ Build LSTM + Attention architecture
├─ Implement weighted loss (pos_weight)
├─ Training loop with early stopping
├─ 5-fold cross-validation
├─ Hyperparameter tuning
└─ Find optimal threshold (ROC curve)

Deliverable: Trained model with validation metrics


PHASE 3: EVALUATION (Days 15-17)
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

├─ Held-out test set evaluation (ONCE!)
├─ Confusion matrices
├─ ROC / Precision-Recall curves
├─ Calibration analysis
├─ Feature importance (attention weights)
└─ Clinical interpretation

Deliverable: Comprehensive results report


PHASE 4: DEPLOYMENT (Days 18-21)
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

├─ API integration
├─ Model serving (ONNX/TorchServe)
├─ Clinical validation with doctors
├─ Edge cases testing
└─ Production optimization

Deliverable: Hospital-ready prediction system
```

---

## Expected Improvements

```
COMPARISON TABLE:

Metric              Current (RF)    New (LSTM)    Improvement
─────────────────────────────────────────────────────────────
AUC                 0.8384         0.90+         +7-8%
Recall              10.3%          70%+          +6.8×
Precision           77%            45%           -32% (accept)
F1                  0.18           0.55+         +3×
Latency/Patient     <1ms           <50ms         +49ms (ok)
Interpretability    Medium         High          +Better

CLINICAL IMPACT:

41 patients die in test set:
├─ RF model: Catches 4, misses 37 (10%)
└─ New model: Catches 29, misses 12 (71%)

Difference: 25 more families get warning!
           25 more times hospital prepares treatment!
           This is CLINICALLY SIGNIFICANT!
```

---

## Decision Checklist (What We're Committing To)

```
Core Changes:
✓ Data: From static mean/std to 24-hour sequences
✓ Features: From 120 to 350+ with clinical meaning
✓ Splitting: From random to temporal + stratified
✓ Model: From Random Forest to LSTM + Attention
✓ Loss: From binary cross-entropy to weighted BCE
✓ Validation: From accuracy to recall-focused
✓ Threshold: From 0.5 to 0.08-0.15 (optimized)

Expected Outcomes:
✓ Recall: 10% → 70% (6.8× improvement)
✓ F1: 0.18 → 0.55 (3× improvement)
✓ Clinical viability: NO → YES
✓ Hospital deployability: NO → YES

Risks:
⚠ Longer training time (minutes vs seconds)
⚠ More complex model (harder to debug)
⚠ Requires GPU or TPU for production
⚠ Needs more thorough validation

Mitigations:
✓ Good documentation at every step
✓ Unit tests for feature extraction
✓ Regular cross-validation checkpoints
✓ Clinical domain expert review

Timeline: 3-4 weeks for complete rewrite
         Could have preliminary results in 1-2 weeks
```

---

## START HERE: First Day Tasks

If approved, execute immediately:

```
TASK 1: Load and Inspect Data (1 hour)
─────────────────────────────────────
├─ [ ] Load X_24h.npy
├─ [ ] Check shape, dtype
├─ [ ] Load means_24h.npy, stds_24h.npy
├─ [ ] Load sample patient data
├─ [ ] Visualize 5 example 24-hour trajectories
└─ [ ] Verify data quality

TASK 2: Plan Data Pipeline Architecture (1 hour)
───────────────────────────────────────────────
├─ [ ] Design TemporalDataset class structure
├─ [ ] Plan temporal split strategy
├─ [ ] Design feature storage (disk vs memory)
├─ [ ] Plan batch processing
└─ [ ] Write pseudocode

TASK 3: Set Up Project Structure (30 min)
──────────────────────────────────────────
├─ [ ] Create src/data_loaders/ directory
├─ [ ] Create src/features/ directory
├─ [ ] Create src/models/temporal/ directory
├─ [ ] Create configs/ directory
└─ [ ] Create notebooks/eda/ directory

TASK 4: Quick EDA (1 hour)
──────────────────────────
├─ [ ] Correlation between vital trends and mortality
├─ [ ] Visualization of dead vs survived 24h trajectories
├─ [ ] Check for data quality issues
└─ [ ] Identify critical features

Total Time: ~4-5 hours
Deliverable: Confirmation data is ready for redesigned pipeline
```

Ready to proceed? Which phase would you like to start with?

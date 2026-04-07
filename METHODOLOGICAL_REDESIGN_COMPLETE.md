# COMPLETE METHODOLOGICAL REDESIGN
## From Instantaneous Vitals to 24-Hour Temporal + Disease-Specific Prediction

**Status**: Major Architectural Overhaul  
**Timeline**: 3-4 weeks  
**Scope**: Complete rewrite of data pipeline, features, models, validation  
**Target**: Hospital-grade ICU mortality prediction system

---

## PART 1: WHY INSTANTANEOUS VITALS FAIL

### The Fundamental Problem

```
Current Approach:
  Patient arrives at ICU
  ↓ Take one measurement
  ├─ HR = 85 bpm
  ├─ RR = 18 breaths/min
  ├─ O2 = 95%
  ├─ BP = 120/80
  └─ Temp = 37.2°C
  
  Model says: "This looks normal! Low risk"
  
  Reality:
  Hour 0:   HR=85 (normal reading)
  Hour 1:   HR=88 (rising)
  Hour 4:   HR=95 (rising more)
  Hour 12:  HR=110 (alarm! deteriorating!)
  Hour 24:  HR=125, patient dies
  
  PROBLEM: Single snapshot missed the TREND
```

### Why Trends Matter (Clinical Context)

```
Same vital value (HR=90), different meanings:

Scenario A: Patient stable
  Hour 0:   HR = 90
  Hour 6:   HR = 89
  Hour 12:  HR = 91
  Hour 24:  HR = 89
  Trend: STABLE → Prognosis: GOOD

Scenario B: Patient deteriorating  
  Hour 0:   HR = 85
  Hour 6:   HR = 87
  Hour 12:  HR = 90
  Hour 24:  HR = 95
  Trend: RISING → Prognosis: BAD

Same final value (HR=90), opposite outcomes!
Machine must SEE THE TRAJECTORY, not just endpoint.
```

### Medical Reality

Doctors make decisions based on:
1. **Where is vital been for 24 hours?** (trend)
2. **Are vitals erratic?** (chaos = bad)
3. **Who is this patient?** (age, comorbidities, why admitted)
4. **Do they have infection?** (lactate, WBC trends)
5. **Are organs failing?** (kidney, lung, liver markers)
6. **Response to treatment?** (inotropes, antibiotics, ventilator settings)

Current model uses: None of this!

---

## PART 2: NEW ARCHITECTURE (24-HOUR TEMPORAL + DISEASE-SPECIFIC)

### 2.1 New Data Structure

```
OLD:
┌──────────────────┐
│ Patient record   │
├──────────────────┤
│ patient_id       │
│ age              │
│ gender           │
│ HR_mean: 85      │ ← Single value
│ RR_mean: 18      │    loses 90% of info
│ O2_mean: 95      │
│ BP: 120/80       │
└──────────────────┘

NEW:
┌────────────────────────────────────────────────┐
│ Patient Temporal Profile (24-hour)             │
├────────────────────────────────────────────────┤
│ Demographics:                                  │
│  - patient_id, age, gender, admission_type    │
│  - Comorbidities: diabetes, hypertension, ... │
│                                                │
│ Hourly Vital Signs (Hour 0→24):              │
│  - Heart Rate: [80, 82, 84, 85, 87, 90...]   │
│  - Respiration: [16, 17, 18, 18, 19, 20...]  │
│  - O2 Sat:     [96, 96, 95, 94, 93, 92...]   │
│  - BP Systolic: [118, 119, 120, 121, 123...] │
│  - Temperature: [37.0, 37.1, 37.3, 37.5...]  │
│                                                │
│ Hourly Labs (if available):                   │
│  - Lactate:     [2.0, 2.1, 2.2, 2.5, ...]    │
│  - WBC:         [12, 13, 14, 15, 16, ...]    │
│  - Creatinine:  [1.0, 1.1, 1.2, 1.3, ...]    │
│  - pH:          [7.40, 7.38, 7.36, 7.35...]  │
│  - LactateBase: [-2, -3, -4, -5, ...]        │
│                                                │
│ Treatment Info:                                │
│  - Antibiotics started: Hour 2                │
│  - Vasopressors started: Hour 8               │
│  - Ventilation: Not intubated → Intubated hr12│
│  - Fluids given: 2L, 3L, 2L, ... (per hour)  │
│                                                │
│ Outcome (24h later):                          │
│  - Survived: Yes/No                           │
│  - ICU stay: 14 days                          │
│  - Complications: Septic shock, AKI, ...      │
└────────────────────────────────────────────────┘
```

### 2.2 Three Layers of Features

```
Layer 1: STATIC DEMOGRAPHICS
─────────────────────────────
├─ Age (years)
├─ Gender (M/F)
├─ Admission type (medical/surgical/trauma)
├─ Comorbidities (diabetes, HTN, CHF, etc)
├─ Severity score (APACHE II estimate)
└─ Primary diagnosis

Layer 2: DYNAMIC VITAL TRENDS (24-hour patterns)
──────────────────────────────────────────────
For each vital (HR, RR, O2, BP, Temp):
├─ Mean (average over 24h)
├─ Std Dev (variability)
├─ Min/Max (extremes)
├─ Trend (linear slope: improving vs deteriorating)
├─ Volatility (coefficient of variation)
├─ Entropy (chaos in signal)
├─ Autocorrelation (predictability)
├─ Number of abnormal events
├─ Duration of abnormality
└─ Recovery time after events

Layer 3: DISEASE-SPECIFIC FACTORS (Clinical context)
─────────────────────────────────────────────────
SEPSIS MARKERS:
├─ Lactate (trend + absolute)
├─ WBC (elevated? trend?)
├─ Temperature (fever? erratic?)
├─ Systemic inflammatory markers
└─ Time since antibiotics started

KIDNEY INJURY:
├─ Creatinine (trend)
├─ BUN (trend)
├─ Urine output (oliguria? anuria?)
├─ Electrolytes (K, Na, Cl)
└─ Acid-base status

RESPIRATORY FAILURE:
├─ O2/FiO2 ratio
├─ pH trend
├─ pCO2 trend
├─ Ventilation mode
└─ Intubation timing

CARDIAC/SHOCK:
├─ Vasopressor requirement
├─ BP trend
├─ HR variability (HRV)
├─ Lactate/pyruvate ratio
└─ Capillary refill time

LIVER DYSFUNCTION:
├─ Total bilirubin
├─ INR
├─ Albumin
└─ Transaminases (AST, ALT)
```

### 2.3 Model Architecture

```
CURRENT ARCHITECTURE (WRONG):
┌────────────────────┐
│ 120 static features│ ← Mean/std/min/max only
└────────────────────┘
         ↓
┌────────────────────┐
│ Random Forest      │ ← No temporal reasoning
└────────────────────┘
         ↓
┌────────────────────┐
│ Mortality score    │
└────────────────────┘


NEW ARCHITECTURE (CORRECT):

Static Branch:                Temporal Branch:               Disease Branch:
┌─────────────┐              ┌──────────────┐               ┌─────────────┐
│Demographics │              │Hourly vitals │               │Labs & Labs  │
│(7 features) │              │(24×5 = 120)  │               │(Disease fac)│
└──────┬──────┘              └───────┬──────┘               └──────┬──────┘
       │                             │                             │
       ▼                             ▼                             ▼
┌──────────────┐            ┌──────────────────┐         ┌──────────────┐
│Embedding     │            │LSTM / Transformer│         │Disease       │
│(32 dims)     │            │Attention (128 d) │         │Encoder       │
└──────┬───────┘            └────────┬─────────┘         │(64 dims)     │
       │                             │                   └──────┬───────┘
       │        ┌────────────────────┴────────────────────┐     │
       │        │                                          │     │
       │        ▼                                          ▼     │
       │   ┌──────────────────────────────────────────────────┐  │
       │   │         Concatenation Layer                      │  │
       │   │    (32 + 128 + 64 = 224 dimensions)             │  │
       │   └──────────────┬───────────────────────────────────┘  │
       │                  │                                       │
       └──────────────────┼───────────────────────────────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ Dense Fusion │
                   │ Layers (3x)  │
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │Output Heads: │
                   ├──────────────┤
                   │1. Mortality  │ ← Primary output
                   │2. Risk Class │
                   │3. LOS Pred   │
                   │4. Organ Fail │
                   └──────────────┘
```

---

## PART 3: NEW DATA PIPELINE

### 3.1 Data Collection & Preprocessing

```
Step 1: Load Raw Data
─────────────────────
Source: eICU / PhysioNet / Hospital records
├─ Hourly vital signs (24 hours)
├─ Hourly laboratory values (if available)
├─ Medication records
├─ Admission details
└─ Outcome (died or survived)

Files needed:
├─ hourly_vitals_24h.csv (N×24×5) - HR, RR, O2, BP, Temp per hour
├─ hourly_labs_24h.csv (N×24×10) - Lactate, WBC, Creatinine, etc
├─ patient_demographics.csv (N×7) - Age, gender, comorbidities
├─ medication_log.csv - Antibiotics start, Vasopressors, etc
└─ outcomes.csv - (N×2) - patient_id, mortality (0/1)


Step 2: Handle Missing Data
────────────────────────────
Problem: Not all patients have all measurements

Strategies:
├─ FORWARD FILL: Use last known value
├─ INTERPOLATION: Linear interpolation between measurements
├─ PATTERN: Use hospital patterns (assume 6-hour intervals)
└─ INDICATOR: Create "was_missing" feature for downstream model

Example:
  HR readings:  [80, _, _, 85, _, _, 90, ...]
  After fill:   [80, 81, 82, 85, 87, 89, 90, ...]
  Indicator:    [0,  1,  1,  0,  1,  1,  0, ...]

Decision Rule:
├─ <10% missing → interpolate
├─ 10-50% missing → forward fill + indicator
├─ >50% missing → exclude patient or use last reading


Step 3: Outlier Detection & Handling
────────────────────────────────────
Problem: Measurement errors (e.g., HR=300)

Method: Statistical outlier detection per vital
├─ HR: Keep 40-200 (exclude <40 or >200)
├─ RR: Keep 8-60 (exclude <8 or >60)
├─ O2: Keep 50-100% (exclude <50 or >100)
├─ Temp: Keep 35-41°C (exclude <35 or >41)
├─ BP: Use clinical judgment

Action: Mark as missing, then apply Step 2


Step 4: Normalization (Per-Patient, Per-Hospital)
───────────────────────────────────────────────────
Problem: Different hospitals use different measurement scales

Method: Z-score normalization WITHIN patient's 24-hour window
├─ Calculate mean & std for each patient-vital pair
├─ X_norm = (X - patient_mean) / (patient_std + eps)
├─ Preserves TRENDS while handling scale differences
└─ Alternative: Min-max scaling if needed


Step 5: Temporal Alignment
──────────────────────────
Problem: Admissions at different times

Standard timeline:
  Hour 0: ICU admission
  Hour 1-23: Monitoring period
  Hour 24: Outcome measurement
  
Result: Every patient has 24 equally-spaced observations
```

### 3.2 Feature Engineering (NEW)

```
VITAL TREND FEATURES (per vital, for all 5 vitals):
───────────────────────────────────────────────

def extract_vital_trends(vital_24h_data):
    """vital_24h: numpy array of shape (24,)"""
    
    features = {}
    
    # 1. Basic statistics
    features['mean'] = np.mean(vital_24h)
    features['std'] = np.std(vital_24h)
    features['min'] = np.min(vital_24h)
    features['max'] = np.max(vital_24h)
    features['range'] = features['max'] - features['min']
    
    # 2. TREND (most important for ICU)
    x = np.arange(24)
    coeffs = np.polyfit(x, vital_24h, 1)
    features['trend_slope'] = coeffs[0]  # Positive = getting worse
    
    # 3. ACCELERATION (is decline speeding up?)
    slopes_early = np.polyfit(x[:12], vital_24h[:12], 1)[0]
    slopes_late = np.polyfit(x[12:], vital_24h[12:], 1)[0]
    features['acceleration'] = slopes_late - slopes_early
    
    # 4. VOLATILITY (chaos in measurement)
    diffs = np.diff(vital_24h)
    features['volatility_hourly_change'] = np.std(diffs)
    features['cv'] = np.std(vital_24h) / (np.mean(vital_24h) + 1e-6)
    
    # 5. ENTROPY (disorder / predictability)
    # Higher entropy = more chaotic/unpredictable
    hist, _ = np.histogram(vital_24h, bins=10, density=True)
    hist = hist[hist > 0]
    features['entropy'] = -np.sum(hist * np.log(hist + 1e-10))
    
    # 6. AUTOCORRELATION (how predictable is next value?)
    acf_vals = []
    for lag in [1, 2, 4, 6, 12]:
        if len(vital_24h) > lag:
            acf = np.correlate(vital_24h - vital_24h.mean(),
                               vital_24h - vital_24h.mean())[lag]
            acf = acf / (np.var(vital_24h) * len(vital_24h))
            acf_vals.append(acf)
    features['autocorr_lag1'] = acf_vals[0] if len(acf_vals) > 0 else 0
    features['autocorr_lag4'] = acf_vals[2] if len(acf_vals) > 2 else 0
    features['autocorr_lag12'] = acf_vals[4] if len(acf_vals) > 4 else 0
    
    # 7. ABNORMAL EVENTS (hours outside typical range)
    q1, q3 = np.percentile(vital_24h, [25, 75])
    iqr = q3 - q1
    abnormal = np.sum((vital_24h < q1 - 1.5*iqr) | (vital_24h > q3 + 1.5*iqr))
    features['n_abnormal_hours'] = abnormal
    features['proportion_abnormal'] = abnormal / 24.0
    
    # 8. RECOVERY (time to recover from extremes)
    min_idx = np.argmin(vital_24h)
    max_idx = np.argmax(vital_24h)
    features['hours_since_min'] = 24 - min_idx
    features['hours_since_max'] = 24 - max_idx
    
    # 9. CRITICAL THRESHOLDS
    # Vital-specific danger zones
    features['time_critical_low'] = count_critical_low(vital_24h)
    features['time_critical_high'] = count_critical_high(vital_24h)
    
    return features

# VITAL-SPECIFIC CONFIG:
vital_configs = {
    'HR': {
        'critical_low': 40,
        'critical_high': 130,
        'danger_slope': 2.5  # bpm/hour
    },
    'RR': {
        'critical_low': 8,
        'critical_high': 50,
        'danger_slope': 1.0  # breaths/hour
    },
    'O2': {
        'critical_low': 85,
        'critical_high': 100,
        'danger_slope': -1.0  # negative = declining
    },
    'Temp': {
        'critical_low': 35,
        'critical_high': 40,
        'danger_slope': 0.5  # degrees/hour
    },
    'BP_sys': {
        'critical_low': 90,
        'critical_high': 200,
        'danger_slope': -2.0  # negative = declining to shock
    }
}

# Result: ~50 features per vital × 5 vitals = 250 trend features
```

### 3.3 Disease-Specific Feature Engineering

```
SEPSIS DETECTION:
─────────────────

def extract_sepsis_features(labs_24h, vitals_24h):
    features = {}
    
    # Lactate (tissue hypoxia marker)
    if 'lactate' in labs_24h:
        lactate = labs_24h['lactate']
        features['lactate_mean'] = np.mean(lactate[~np.isnan(lactate)])
        features['lactate_trend'] = np.polyfit(np.arange(len(lactate)), lactate, 1)[0]
        features['lactate_high'] = 1 if np.mean(lactate) > 2 else 0
        features['lactate_worsening'] = 1 if features['lactate_trend'] > 0.1 else 0
    
    # WBC (infection marker)
    if 'wbc' in labs_24h:
        wbc = labs_24h['wbc']
        features['wbc_elevated'] = 1 if np.mean(wbc) > 11 else 0
        features['wbc_low'] = 1 if np.mean(wbc) < 4 else 0  # Also bad!
        features['wbc_trend'] = np.polyfit(np.arange(len(wbc)), wbc, 1)[0]
    
    # Temperature (fever + volatility)
    if 'temperature' in vitals_24h:
        temp = vitals_24h['temperature']
        features['temp_fever'] = 1 if np.mean(temp) > 38 else 0
        features['temp_hypothermia'] = 1 if np.mean(temp) < 36 else 0
        features['temp_volatile'] = np.std(temp)  # Erratic = bad
    
    # Procalcitonin (if available - very sepsis-specific)
    if 'procalcitonin' in labs_24h:
        pct = labs_24h['procalcitonin']
        features['pct_elevated'] = 1 if np.mean(pct) > 0.5 else 0
    
    return features


ACUTE KIDNEY INJURY:
────────────────────

def extract_aki_features(labs_24h):
    features = {}
    
    # Creatinine (primary kidney marker)
    if 'creatinine' in labs_24h:
        creat = labs_24h['creatinine']
        features['creat_baseline'] = creat[0]
        features['creat_peak'] = np.max(creat)
        features['creat_change'] = features['creat_peak'] - features['creat_baseline']
        features['creat_increase_pct'] = (features['creat_change'] / 
                                          (features['creat_baseline'] + 0.1)) * 100
        features['creat_trend'] = np.polyfit(np.arange(len(creat)), creat, 1)[0]
    
    # BUN (urea, kidney function)
    if 'bun' in labs_24h:
        bun = labs_24h['bun']
        features['bun_high'] = 1 if np.mean(bun) > 20 else 0
        features['bun_creat_ratio'] = np.mean(bun) / (np.mean(creat) + 0.1)
    
    # Urine output (critical for AKI diagnosis)
    if 'urine_output_24h' in labs_24h:
        uo = labs_24h['urine_output_24h']
        features['oliguria'] = 1 if uo < 400 else 0  # <0.5 mL/kg/hr
        features['anuria'] = 1 if uo < 100 else 0    # <0.1 mL/kg/hr
        features['urine_output_ml'] = uo
    
    # Potassium (kidney regulation marker)
    if 'potassium' in labs_24h:
        k = labs_24h['potassium']
        features['hyperkalemia'] = 1 if np.mean(k) > 5.5 else 0
        features['hypokalemia'] = 1 if np.mean(k) < 3.5 else 0
    
    return features


RESPIRATORY FAILURE:
────────────────────

def extract_respiratory_features(labs_24h, vitals_24h):
    features = {}
    
    # Oxygenation (PaO2/FiO2 ratio)
    if 'pao2' in labs_24h and 'fio2' in labs_24h:
        pao2 = labs_24h['pao2']
        fio2 = labs_24h['fio2']
        pf_ratio = np.mean(pao2 / (fio2 + 1e-6))
        features['pf_ratio'] = pf_ratio
        features['pf_severe'] = 1 if pf_ratio < 100 else 0  # ARDS criterion
        features['pf_moderate'] = 1 if pf_ratio < 200 else 0
    
    # Acid-base (pH, pCO2)
    if 'ph' in labs_24h:
        ph = labs_24h['ph']
        features['ph_mean'] = np.mean(ph)
        features['acidemia'] = 1 if np.mean(ph) < 7.35 else 0
        features['alkalemia'] = 1 if np.mean(ph) > 7.45 else 0
    
    if 'pco2' in labs_24h:
        pco2 = labs_24h['pco2']
        features['hypercapnia'] = 1 if np.mean(pco2) > 45 else 0
    
    # RR trend (indicator of respiratory distress)
    if 'rr' in vitals_24h:
        rr = vitals_24h['rr']
        features['rr_trend'] = np.polyfit(np.arange(len(rr)), rr, 1)[0]
        features['rr_high_hours'] = np.sum(rr > 30)
    
    # Intubation flag
    if 'intubated' in labs_24h:
        features['intubated'] = labs_24h['intubated']
        features['intubation_hour'] = labs_24h.get('intubation_hour', 24)
    
    return features


SHOCK / HYPOTENSION:
────────────────────

def extract_shock_features(labs_24h, vitals_24h):
    features = {}
    
    # Lactate (tissue perfusion marker - most important for shock)
    if 'lactate' in labs_24h:
        lactate = labs_24h['lactate']
        features['lactate_high'] = 1 if np.mean(lactate) > 4 else 0
        features['lactate_critical'] = 1 if np.mean(lactate) > 5 else 0
    
    # Blood pressure trend
    if 'bp_systolic' in vitals_24h:
        bp = vitals_24h['bp_systolic']
        features['bp_trend'] = np.polyfit(np.arange(len(bp)), bp, 1)[0]
        features['hypotension_hours'] = np.sum(bp < 90)
        features['shock_risk'] = 1 if features['bp_trend'] < -2 else 0
    
    # Heart rate (tachycardia compensating for low BP)
    if 'hr' in vitals_24h:
        hr = vitals_24h['hr']
        features['tachycardia_hours'] = np.sum(hr > 100)
        features['hr_bp_mismatch'] = (np.sum(hr > 100) > 12 and 
                                       features.get('hypotension_hours', 0) > 0)
    
    # Vasopressor requirement
    if 'vasopressor_started' in labs_24h:
        features['needs_vasopressor'] = labs_24h['vasopressor_started']
        features['vasopressor_hour'] = labs_24h.get('vasopressor_hour', 24)
    
    return features


LIVER DYSFUNCTION:
──────────────────

def extract_liver_features(labs_24h):
    features = {}
    
    # Bilirubin (jaundice marker)
    if 'bilirubin_total' in labs_24h:
        bili = labs_24h['bilirubin_total']
        features['hyperbilirubinemia'] = 1 if np.mean(bili) > 2 else 0
        features['severe_hyperbilirubinemia'] = 1 if np.mean(bili) > 5 else 0
    
    # INR (coagulation / synthetic function)
    if 'inr' in labs_24h:
        inr = labs_24h['inr']
        features['coagulopathy'] = 1 if np.mean(inr) > 1.5 else 0
        features['severe_coagulopathy'] = 1 if np.mean(inr) > 2.0 else 0
    
    # Albumin (protein/synthetic function)
    if 'albumin' in labs_24h:
        alb = labs_24h['albumin']
        features['hypoalbuminemia'] = 1 if np.mean(alb) < 3.5 else 0
        features['severe_hypoalbuminemia'] = 1 if np.mean(alb) < 2.5 else 0
    
    # Transaminases (hepatocellular injury)
    if 'ast' in labs_24h or 'alt' in labs_24h:
        ast = labs_24h.get('ast', [0])
        alt = labs_24h.get('alt', [0])
        features['hepatic_injury'] = 1 if (np.mean(ast) > 40 or np.mean(alt) > 40) else 0
    
    return features
```

---

## PART 4: NEW MODEL ARCHITECTURE

### 4.1 Why We Need Different Models

```
OLD (WRONG):
Random Forest 
├─ Assumes all features equally independent
├─ Can't model sequences
├─ Loses temporal information
└─ Result: 10% recall (misses 90% of deaths)


NEW (CORRECT):
"Multi-Stream Fusion Network"

Components:
1. LSTM / GRU Stream: Learns temporal patterns
   Input: 24×5 hourly vitals
   Output: 128-dim context vector
   
2. Transformer Stream: Learns dependencies across vitals
   Input: 24×5 hourly vitals with attention
   Output: 128-dim context vector
   
3. Dense Stream: Processes disease features
   Input: 50-100 disease-specific features
   Output: 64-dim context vector
   
4. Fusion Layer: Combines all streams
   Input: (128 + 128 + 64 = 320) dims
   Hidden: 256 → 128 → 64
   Output: Mortality probability
```

### 4.2 Three Model Options

```
OPTION A: LSTM-based (Recommended for Speed)
────────────────────────────────────────────
Architecture:
  Input: (batch, 24, 5) ← Hours × Vitals
  ↓
  Embedding: (batch, 24, 32)
  ↓
  2-layer LSTM: 64 hidden units, dropout=0.3
  ↓
  Attention: 8 heads, 32 dims
  ↓
  Output: (batch, 128)
  
Advantages:
✓ Fast training (<2 minutes on modern GPU)
✓ Fast inference (<10ms per patient)
✓ Good for temporal sequences
✓ Less data required

Disadvantages:
✗ Can't capture very long-range dependencies
✗ Single pass (can't revisit early data)

Use when: Fast deployment needed


OPTION B: TRANSFORMER-based (Best Performance)
──────────────────────────────────────────────
Architecture:
  Input: (batch, 24, 5) ← Hours × Vitals + positional encoding
  ↓
  Multi-head self-attention: 8 heads, 64 dims
  ↓
  Feed-forward: 256 hidden
  ↓
  3 transformer blocks
  ↓
  Output: (batch, 128)

Advantages:
✓ Can see all time points at once
✓ Better for modeling complex dependencies
✓ Better extrapolation to new patterns
✓ Very interpretable (attention weights show which hours matter)

Disadvantages:
✗ Slower training (5-10 minutes on GPU)
✗ More parameters, needs more data
✗ Inference still fast (<50ms per patient)

Use when: Best performance priority


OPTION C: HYBRID CNN-LSTM (Balanced)
────────────────────────────────────
Architecture:
  Input: (batch, 24, 5)
  ↓
  1D Conv: 32 filters, kernel=3, stride=1 ← Captures local patterns
  ↓
  BiLSTM: 64 hidden ← Bidirectional learning
  ↓
  Self-attention: 4 heads ← Focus on important hours
  ↓
  Output: (batch, 128)

Advantages:
✓ Fast AND good performance
✓ CNNs good at local temporal patterns
✓ LSTMs good at sequence
✓ Balanced performance-speed
✓ Fewer parameters than Transformer

Use when: Production deployment needed
```

---

## PART 5: NEW TRAINING PIPELINE

### 5.1 Data Splitting Strategy

```
CRITICAL: Temporal Data Requires Special Splitting!

WRONG (Current):
├─ Random 70/15/15 split
├─ Some Hours 0-24 in train, some in test
└─ Result: Model learns from test data implicitly


CORRECT (What We Need):
├─ Temporal Split: 
│  ├─ Train: Admissions from Jan-Aug (chronological)
│  ├─ Val:   Admissions from Sep-Oct
│  └─ Test:  Admissions from Nov-Dec (never seen before)
│
├─ Stratification:
│  ├─ Preserve 8.6% mortality rate in each split
│  └─ Maintain demographics distribution
│
└─ What we're protecting against:
   ├─ Data leakage from future admissions
   ├─ Trend changes (seasons, new protocols)
   └─ Covariate shift (patient population changes)

Implementation:
```python
# Temporal split
all_admissions = load_all_admissions()  # sorted by date
n = len(all_admissions)

train_idx = int(0.6 * n)   # Jan-Aug
val_idx = int(0.75 * n)    # Sep-Oct

train_data = all_admissions[:train_idx]
val_data = all_admissions[train_idx:val_idx]
test_data = all_admissions[val_idx:]

# Stratify by outcome
for dataset in [train_data, val_data, test_data]:
    n_deaths = np.sum(dataset.mortality)
    pct = n_deaths / len(dataset)
    print(f"Mortality rate: {pct:.1%}")  # Should all be ~8.6%
```

### 5.2 Training Procedure

```
Hyperparameters:
────────────────
Model: LSTM (chosen for speed)
  Hidden units: 64
  Dropout: 0.3
  Attention heads: 4
  
Loss function: BCEWithLogitsLoss(pos_weight=11.6)
  ↑ pos_weight = count_negatives / count_positives
  ↑ Penalizes false negatives more
  
Optimizer: AdamW
  Learning rate: 1e-3 (decay by 0.5 every 5 epochs)
  Weight decay: 1e-4
  Batch size: 32
  
Regularization:
  ├─ Dropout: 0.3
  ├─ L2 penalty: 1e-4
  ├─ Early stopping: patience=5, monitor="val_recall"
  └─ Gradient clipping: max_norm=1.0

Training Loop:
──────────────
for epoch in range(max_epochs):
    for batch_idx, (X_vitals, X_disease, y) in enumerate(train_loader):
        # Forward pass
        logits = model(X_vitals, X_disease)
        loss = criterion(logits, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    # Validation
    val_loss, val_recall, val_precision = evaluate(model, val_loader)
    
    # Save best model (monitor recall, not just loss!)
    if val_recall > best_recall:
        best_recall = val_recall
        save_checkpoint(model)
    
    # Early stopping
    if no_improvement_for(5_epochs):
        break

Stopping Criteria:
──────────────────
✓ Recall on validation set >= 70%
✓ AUC >= 0.90
✓ F1 >= 0.55
✓ No improvement for 5 epochs on val_recall
```

### 5.3 New Validation Strategy

```
HELD-OUT TEST SET (Can only be used ONCE at the end):
──────────────────────────────────────────────────────
For next 2 weeks:
├─ Train on labeled admissions Jan-Oct
├─ Validate on Sep-Oct
├─ NEVER TOUCH Nov-Dec test set
│
Final evaluation:
  Load best model from validation
  ↓
  Evaluate ONCE on Nov-Dec test set
  ↓
  Report metrics
  ↓
  If metrics bad: Go back to training
  (Don't tune on test set!)


Cross-Validation for Development:
─────────────────────────────────
5-fold with stratification:
├─ Fold 1: Train on 60-80%, val on 80-100% (early)
├─ Fold 2: Train on 40-60%, val on 60-80%
├─ Fold 3: Train on 20-40%, val on 40-60%
├─ Fold 4: Train on 0-20%, val on 20-40%
└─ Fold 5: Shuffle all, random splits

Report: Mean ± Std across folds


Metrics to Monitor:
──────────────────
Primary (for mortality prediction):
├─ AUC (area under ROC curve) → should be 0.90+
├─ Recall @ 90% specificity → should be 70%+
├─ F1 score → should be 0.55+
└─ Recall @ different thresholds

Secondary (clinical usefulness):
├─ Calibration curve (are probabilities trustworthy?)
├─ Confusion matrix at optimal threshold
├─ Sensitivity & specificity
├─ Positive/Negative predictive values
└─ Decision curve analysis

Avoid optimizing:
✗ Accuracy (misleading for imbalanced data)
✗ Precision alone (low recall unacceptable)
✗ AUC alone (doesn't guarantee good recall)
```

---

## PART 6: COMPLETE IMPLEMENTATION ROADMAP

### Week 1: Data Pipeline Setup

```
Day 1: Data Exploration & Validation
─────────────────────────────────────
Files: data/processed_icu_hourly_v2.csv, X_24h.npy, means_24h.npy
├─ [ ] Load and inspect current data structure
├─ [ ] Check for missing values per vital
├─ [ ] Visualize 10 patient 24-hour trajectories
├─ [ ] Identify data quality issues
└─ [ ] Document findings

Day 2: Build Data Loader
────────────────────────
├─ [ ] Create TemporalDataset class
│   ├─ Loads (X_24h, X_disease, y)
│   ├─ Handles batch processing
│   └─ On-the-fly normalization
├─ [ ] Implement temporal train/val/test split
├─ [ ] Create DataLoader with stratification
└─ [ ] Unit tests for data loading

Day 3-4: Feature Engineering - Vitals
──────────────────────────────────────
├─ [ ] Implement extract_vital_trends() for all vitals
│   ├─ Trend (slope)
│   ├─ Volatility (std, CV)
│   ├─ Entropy (disorder)
│   ├─ Autocorrelation
│   └─ Abnormal events
├─ [ ] Test on sample patients
├─ [ ] Save features to disk
└─ [ ] Visualize important features vs mortality

Day 5-6: Feature Engineering - Disease
───────────────────────────────────────
├─ [ ] Implement extract_sepsis_features()
├─ [ ] Implement extract_aki_features()
├─ [ ] Implement extract_respiratory_features()
├─ [ ] Implement extract_shock_features()
├─ [ ] Aggregate all disease factors
└─ [ ] Validate feature importance
```

### Week 2: Model Development

```
Day 7-8: Baseline Model
──────────────────────
├─ [ ] Build simple LSTM model
│   ├─ 1 LSTM layer
│   ├─ 64 hidden units
│   ├─ Dropout 0.3
│   └─ Binary output
├─ [ ] Implement training loop
├─ [ ] Train on 50% sample to debug
└─ [ ] Measure baseline metrics

Day 9-10: Improved Model
─────────────────────────
├─ [ ] Add attention mechanism
├─ [ ] Process disease features separately
├─ [ ] Implement fusion layer
├─ [ ] Add weighted loss (pos_weight)
├─ [ ] Train on full dataset
└─ [ ] Compare with baseline

Day 11-12: Validation & Hyperparameter Tuning
──────────────────────────────────────────────
├─ [ ] Conduct 5-fold cross-validation
├─ [ ] Tune learning rate
├─ [ ] Tune dropout rate
├─ [ ] Tune batch size
├─ [ ] Find optimal threshold (ROC curve)
└─ [ ] Save best model
```

### Week 3-4: Evaluation & Refinement

```
Day 13-14: Comprehensive Evaluation
────────────────────────────────────
├─ [ ] Generate confusion matrices
├─ [ ] Plot ROC curve
├─ [ ] Plot Precision-Recall curve
├─ [ ] Calibration curves
├─ [ ] Decision curve analysis
├─ [ ] Feature importance (attention weights)
└─ [ ] Clinical interpretation

Day 15-16: Final Validation
──────────────────────────
├─ [ ] Evaluate on held-out test set (ONCE ONLY)
├─ [ ] Compare with RF baseline
├─ [ ] Document results
├─ [ ] Prepare visualizations
└─ [ ] Write technical report

Day 17-20: Refinement & Production
──────────────────────────────────
├─ [ ] API integration
├─ [ ] Model serving (ONNX/TorchServe)
├─ [ ] 95% confidence interval on predictions
├─ [ ] Clinical validation with doctors
├─ [ ] Final optimization
└─ [ ] Deployment readiness
```

---

## PART 7: SUCCESS METRICS (Clear Targets)

```
Current System (RF):         New System (LSTM + Temporal):
AUC: 0.8384                  AUC: 0.90+                (requirement)
Recall: 10.3%                Recall: 70%+              (CRITICAL)
Precision: 77%               Precision: 45%+           (acceptable trade)
F1: 0.18                     F1: 0.55+                 (7.7× improvement!)
Status: USELESS              Status: HOSPITAL-READY

Clinical Impact:
Current: Misses 36 of 41 deaths
New:     Catches 28-29 of 41 deaths (improvement!)
```

---

## NEXT IMMEDIATE ACTIONS

### This Week (Days 1-3):
1. [ ] Accept/finalize this architectural redesign
2. [ ] Set up new directory structure
3. [ ] Load and inspect current 24-hour data (X_24h.npy)
4. [ ] Build TemporalDataset class
5. [ ] Verify temporal split strategy

Should I proceed with this complete redesign? Which week would you like to start?

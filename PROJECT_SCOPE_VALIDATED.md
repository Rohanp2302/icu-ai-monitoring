# INTERPRETABLE ML SYSTEM FOR INDIAN HOSPITALS
## Project Scope - Validated Against Actual Data

**Status**: ✅ VALIDATED AGAINST RAW DATA  
**Last Updated**: April 7, 2026  
**Primary Dataset**: eICU-CRD (2,520 patients, 1.6M vitals, 434K labs)  
**Secondary Dataset**: Challenge2012 (12,000 patients, validation)

---

## 1️⃣ PROJECT OBJECTIVES (CORE)

### Primary Goal
Build **Interpretable ML system** that surpasses SOFA/APACHE scores in:
- ✅ Mortality prediction accuracy (90+ AUC)
- ✅ Clinical organ dysfunction detection
- ✅ Medicine response tracking
- ✅ **Explainability to patient families**

### System Functions
| Function | Input | Output | Technology |
|----------|-------|--------|-----------|
| **Risk Prediction** | 24h patient data | Mortality risk + risk drivers | Deep Learning |
| **Organ Health** | Vitals + Labs + SOFA | Per-organ dysfunction scores | AIML + Rules |
| **Medicine Tracking** | Medications + vital response | Treatment effectiveness | Time-series Analysis |
| **Progress vs Expected** | Timeline data | Actual vs predicted recovery trajectory | Forecasting |
| **Patient Explanation** | Risk drivers + organ status | Easy-to-understand visual report | Interpretability |

### Clinical Setting
- **Target**: Indian hospitals (SOFA/APACHE adapted for local context)
- **Users**: Doctors, ICU staff, AND patient families
- **Use Case**: Early warning, treatment optimization, family communication

---

## 2️⃣ DATA SOURCES (VALIDATED INVENTORY)

### Dataset 1: eICU-CRD ⭐ PRIMARY
**Location**: `e:/icu_project/data/raw/eicu/`

#### Vital Signs Data
- **Table**: `vitalPeriodic.csv`
- **Records**: 1,634,960 (dense, ~2-3 min intervals)
- **Features**: Heart rate, MAP, BP, SpO2, temperature, respiratory rate, etc.
- **Time Coverage**: Minutes from ICU admission

#### Laboratory Tests
- **Table**: `lab.csv`
- **Records**: 434,660 tests from 147 unique types
- **Coverage**: Comprehensive organ function markers:
  - **Renal**: Creatinine, BUN, urine output
  - **Liver**: Bilirubin, AST, ALT, albumin
  - **Coagulation**: INR, platelets, PTT, PT
  - **Metabolic**: pH, HCO3, lactate, K+, Na+, Ca2+, Mg2+
  - **Hematologic**: WBC, RBC, Hgb, Hct
  - **Others**: Glucose, CRP, procalcitonin

#### Medications & Treatments
- **Table 1**: `medication.csv` (75,604 records)
  - Drug names, dosage, route (IV/oral/NG), frequency
  - Start/stop times in minutes from admission
- **Table 2**: `admissiondrug.csv` (7,417 pre-admission meds)
- **Table 3**: `infusiondrug.csv` (infusion-specific)
- **Table 4**: `treatment.csv` (procedures, ventilation, dialysis)

#### Organ Dysfunction Markers
- **Table**: `apacheApsVar.csv`
- **SOFA Components**:
  - CNS: Eyes, Motor, Verbal (Glasgow Coma Scale components)
  - Respiratory: Ventilation requirement (yes/no)
  - Renal: Dialysis requirement (yes/no)
  - Hematologic: Platelets, WBC
  - Hepatic: Bilirubin, PT/INR
- **Plus 15+ APS variables** for additional context

#### Clinical Assessments
- **Nursing Notes**: `nurseCharting.csv` (1.4M entries)
- **Physical Exam**: `physicalExam.csv`
- **Intake/Output**: `intakeOutput.csv`

#### Outcomes & Demographics
- **Patient Info**: `patient.csv` (2,520 patients)
  - Age, gender, ethnicity, height, weight
  - Admission reason, discharge status (died/alive)
- **Hospital Info**: `hospital.csv`
- **Outcome**: HOSPITALIZATION MORTALITY (5.0% event rate)

**Total Records**: 4.6M rows  
**Time Span Per Patient**: Full ICU stay (usually 24-72 hours, up to 200+ hours)

### Dataset 2: Challenge2012 (VALIDATION)
**Location**: `e:/icu_project/data/raw/challenge2012/`

#### Structure
- **3 Sets**: set-a, set-b, set-c
- **Patients**: 12,000 total (4K per set)
- **Format**: Individual .txt files per patient
- **Outcomes**: Separate files (Outcomes-a/b/c.txt)

#### Available Data
- Vital signs (sparse, ~12 readings per patient)
- SOFA aggregate score
- SAPS-I score
- In-hospital mortality label (0/1)

#### Limitation
- ❌ No lab tests
- ❌ No medication data
- ❌ Sparse vitals (7-37 min intervals, irregular)
- ✅ Used for external validation only

---

## 3️⃣ FEATURE ENGINEERING PIPELINE

### Phase 1: Raw Data Extraction

#### From eICU - Vital Signs
Extract from `vitalPeriodic.csv`:
```
Features: HR, MAP, SBP, DBP, SpO2, Temp, RR, CVP, IAP, 
          PEEP, FiO2, VT, MV, Peak_Pressure, Plateau_Pressure
Time aggregation: 1-hour windows (mean, std, min, max, median, trend)
Temporal feature: Time from admission, hour-of-day (circadian pattern)
```

#### From eICU - Laboratory Tests
Extract from `lab.csv`:
```
For each lab type (147 types), aggregate:
- Most recent value before hour
- Change from previous day  
- Trend (increasing/decreasing/stable)
- Days since measurement
- Normal range deviation

Priority labs:
- Renal (creatinine, BUN, K+, urine output)
- Liver (bilirubin, ALT, AST, albumin, PT/INR)
- Hematologic (WBC, platelets, Hgb)
- Metabolic (pH, HCO3, lactate, Na+, Ca2+, Mg2+, gluc)
- Sepsis markers (CRP, PCT, lactate)
```

#### From eICU - Medications
Extract from `medication.csv`, `admissiondrug.csv`:
```
For each medication:
- Is patient on vasopressor? (cumulative)
- Sedation level (type + dose combined)
- Antibiotic count
- Antifungal count
- Antiviral count
- Diuretics (yes/no, type)
- Insulin (yes/no, total units in past 24h)
- Paralytic agents (yes/no)
- Ventilator settings (if on respiratory support)
```

#### From eICU - Organ Dysfunction
Extract from `apacheApsVar.csv`, calculate:
```
SOFA COMPONENTS:
- Respiratory: FiO2/pO2 ratio, ventilation requirement
- Circulatory: MAP, vasopressor requirement
- Renal: Creatinine, urine output (0-4 scale)
- Coagulation: Platelets count (0-4 scale)
- Hepatic: Bilirubin level (0-4 scale)
- CNS: GCS (0-4 scale)

SOFA Total = Sum (0-24 scale) - SUPRA-FRAMEWORK

Multi-organ dysfunction markers:
- 2+ organ failure (yes/no)
- 3+ organ failure (yes/no)
- 4+ organ failure (yes/no)
```

### Phase 2: Organ Health Scoring

```python
ORGAN_HEALTH_SCORES = {
    "respiratory": {
        "markers": ["SpO2", "pO2", "pCO2", "pH", "RR", "FiO2", "ventilation_days"],
        "normal_range": "SpO2>92, pO2>60, pCO2 35-45, RR 12-20",
        "output": "Resp Health Score 0-10 + SOFA component"
    },
    
    "cardiovascular": {
        "markers": ["Map", "HR", "BP", "lactate", "vasopressor", "CVP"],
        "normal_range": "MAP 65-110, HR 60-100, lactate <2",
        "output": "CV Health Score 0-10 + SOFA component"
    },
    
    "renal": {
        "markers": ["creatinine", "BUN", "K+", "UOP", "dialysis"],
        "normal_range": "Cr <1.5, BUN <20, UOP >0.5ml/kg/h",
        "output": "Renal Health Score 0-10 + SOFA component"
    },
    
    "hepatic": {
        "markers": ["bilirubin", "AST", "ALT", "albumin", "PT", "INR"],
        "normal_range": "Bil <1.2, AST <40, albumin >3.5",
        "output": "Hepatic Health Score 0-10 + SOFA component"
    },
    
    "hematologic": {
        "markers": ["Hgb", "WBC", "platelets", "PT", "INR"],
        "normal_range": "Hgb 12-16, WBC 4-11, Plt >150",
        "output": "Hematologic Health Score 0-10 + SOFA component"
    },
    
    "neurologic": {
        "markers": ["GCS", "sedation_type", "sedation_dose", "paralytics"],
        "normal_range": "GCS 15, no sedation",
        "output": "Neuro Health Score 0-10 + SOFA component"
    }
}
```

### Phase 3: 24-Hour Temporal Windows

Create sliding windows:
```
For each patient:
  For each 24-hour window from admission (t → t+24h):
    Extract features aggregated over 24h
    Create sequential features: 
      - Mean value, std, min, max, trend
      - Deterioration rate (change from start to end)
      - Volatility (variation throughout window)
    Create target:
      - If window ends with mortality prediction → 1
      - Else → 0
    Create metadata:
      - Hours from admission
      - Window number (1st, 2nd, 3rd 24h)
```

**Output**: Windowed dataset with (N_patients × N_windows, 200+ features)

### Phase 4: Medicine Response Features

```
For each medication window (past 24h):
  Response features:
    - Vasopressor started? + time to response (BP increase)
    - Antibiotic started? + time to response (temp decrease, lactate decrease)
    - Insulin dose/hour + glucose trend response  
    - Diuretic dose + UOP response
    - Sedation changes + HR/RR response
    
Create features:
    - Treatment intensity (number of different drug types)
    - Medicine combination patterns
    - Time since last medication change
    - Response lag (how fast did patient respond to treatment)
```

---

## 4️⃣ DEEP LEARNING ARCHITECTURE (AIML)

### Model Architecture
```
INPUT LAYER (200+ features per 24h window)
│
├─ VITAL SIGN BRANCH
│  ├─ Dense layer (50 units, ReLU)
│  ├─ Batch Norm
│  ├─ Dropout (0.3)
│  └─ Output: 20 units
│
├─ LAB RESULTS BRANCH  
│  ├─ Dense layer (50 units, ReLU)
│  ├─ Batch Norm
│  ├─ Dropout (0.3)
│  └─ Output: 20 units
│
├─ MEDICATION BRANCH
│  ├─ Dense layer (30 units, ReLU)
│  └─ Output: 10 units
│
├─ ORGAN HEALTH BRANCH
│  ├─ Dense layer (30 units, ReLU)
│  └─ Output: 10 units
│
└─ TEMPORAL/SEQUENCE BRANCH
   ├─ If using multi-window approach:
   │  └─ LSTM (32 units, 512 units) over 3 × 24h windows
   └─ Output: 20 units

CONCATENATE ALL BRANCHES (60 units)
│
DENSE LAYERS:
├─ Dense (64, ReLU) + Batch Norm + Dropout(0.3)
├─ Dense (32, ReLU) + Batch Norm + Dropout(0.3)
├─ Dense (16, ReLU) + Dropout(0.2)
│
MULTI-TASK OUTPUT HEADS:
├─ Head 1: Mortality Prediction (1, Sigmoid) ← PRIMARY
├─ Head 2: Organ Dysfunction (6, Sigmoid) ← Respiratory, CV, Renal, Hepatic, Hematologic, CNS
├─ Head 3: Treatment Response (1, Linear) ← Will patient respond to current treatment?
└─ Head 4: Recovery Trajectory (1, Linear) ← Expected improvement 24h ahead

LOSS FUNCTION:
  Total = (0.5 × BCE_mortality) + (0.2 × BCE_organs) + (0.2 × MSE_response) + (0.1 × MSE_recovery)
```

### Training Strategy
- **Backbone**: TensorFlow/Keras
- **Optimizer**: Adam (learning rate scheduling)
- **Regularization**: L2 + Dropout + Batch Norm
- **Cross-validation**: 5-fold temporal split
- **Class weights**: Handle 5% mortality class imbalance
- **Early stopping**: Monitor validation AUC

---

## 5️⃣ EXPLAINABILITY LAYER

### Feature Importance (SHAP values)
```
For each prediction:
  1. Calculate SHAP values for all 200+ features
  2. Identify top 5 most important features
  3. Show direction: Is this feature pushing risk UP or DOWN?
  4. Show magnitude: How much does this contribute to risk?
```

### Organ-Specific Explanations
```
For each organ:
  - Which labs/vitals are abnormal?
  - How bad is the dysfunction (SOFA score)?
  - Which medications are targeting this organ?
  - Is it improving or deteriorating?
```

### Simple Visual Output
```
Example Output Tweet for Patient Family:
"Your mother's kidney health is DECLINING (SOFA 3/4).
Her creatinine is UP 50% from yesterday.
Doctors increased dialysis - expected to help next 24h.
Risk score: MEDIUM (62% mortality risk - higher than average 5%)"
```

---

## 6️⃣ PERFORMANCE TARGETS (MUST ACHIEVE)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Mortality Prediction AUC** | ≥0.90 | ? | 🔄 TO BUILD |
| **Mortality Sensitivity (Recall)** | ≥0.70 | ? | 🔄 TO BUILD |
| **SOFA Prediction RMSE** | <2.0 | ? | 🔄 TO BUILD |
| **Organ Dysfunction F1** | ≥0.75 | ? | 🔄 TO BUILD |
| **Treatment Response R²** | ≥0.60 | ? | 🔄 TO BUILD |
| **APACHE/SOFA Comparison** | Beat by 5% AUC | ? | 🔄 TO VALIDATE |

---

## 7️⃣ IMPLEMENTATION ROADMAP

### PHASE 1: Data Pipeline (Week 1)
- [ ] Load & parse eICU raw CSVs (1.6M vitals, 434K labs)
- [ ] Create feature extraction engine (vital aggregation, lab aggregations, med combinations)
- [ ] Build 24-hour windowing logic
- [ ] Create organ health calculator
- [ ] Generate feature matrix: (N_windows, 200+) from raw data
- [ ] Save processed data for modeling

### PHASE 2: Deep Learning Model (Week 2)
- [ ] Build multi-task architecture (mortality + organs + response + recovery)
- [ ] Implement training loop with 5-fold CV
- [ ] Train base model on eICU dataset
- [ ] Tune hyperparameters for 90+ AUC
- [ ] Achieve target performance (≥0.90 AUC)

### PHASE 3: AIML Components (Week 2-3)
- [ ] Implement SOFA scoring engine
- [ ] Create organ health rules (expert system)
- [ ] Build feature importance calculator (SHAP)
- [ ] Create decision tree interpretability layer

### PHASE 4: Medicine Response Tracking (Week 3)
- [ ] Extract medication effectiveness from historical data
- [ ] Build response prediction model
- [ ] Create treatment recommendation pathway

### PHASE 5: UI/UX (Week 4)
- [ ] Build dashboard showing:
  - **Mortality risk** with confidence interval
  - **Organ health scores** (6 organs visual)
  - **Medicine timeline** (what meds, when, effectiveness)
  - **Predicted trajectory** (actual vs expected recovery)
  - **Key risk drivers** (SHAP explanations, patient-family friendly)
- [ ] Mobile-friendly design for India
- [ ] Explanation mode for patient families

### PHASE 6: Validation & Deployment (Week 4-5)
- [ ] External validation on Challenge2012
- [ ] Compare vs SOFA/APACHE on same cohort
- [ ] Clinical validation with Indian hospital partners
- [ ] Deploy API + UI

---

## 8️⃣ VALIDATION CHECKLIST (USE BEFORE EVERY CODING SESSION)

**Before starting ANY implementation, answer these**:

- [ ] Is this feature engineering from RAW data?
- [ ] Are we using AT LEAST 24 HOURS of temporal data?
- [ ] Does our prediction output include organ-specific health scores?
- [ ] Is there medicine/treatment response tracking?
- [ ] Are we using AIML + Deep Learning (not just tree ensembles)?
- [ ] Does the output explain WHY risk is high (interpretability)?
- [ ] Are we extracting features from eICU labs/medications (not just vitals)?
- [ ] Is the model targeting 90+ AUC (not quick deployment)?
- [ ] Are we tracking progress vs predicted trajectory?
- [ ] Does the UI/output work for patient families?

---

## 9️⃣ RED FLAGS (STOP & FIX)

🚫 If you see this, BACKTRACK:
1. Model trained on pre-processed CSVs → GO TO RAW DATA
2. Instant prediction models → USE 24H WINDOWS
3. Only tree ensembles → ADD DEEP LEARNING
4. Generic feature extraction → USE EICU LABS + MEDS
5. Single metric focus → TRACK MULTIPLE ORGANS
6. Technical explainability only → MAKE IT FAMILY-FRIENDLY
7. High AUC but low recall → FIX THRESHOLD FOR CLINICAL USE
8. No medication tracking → ADD TREATMENT RESPONSE
9. Copied old docs without updating → VALIDATE AGAINST NEW ARCHITECTURE

---

## 📋 PROJECT STATUS DASHBOARD

| Component | Status | Owner | ETA |
|-----------|--------|-------|-----|
| Data Architecture | 🔄 STARTING | — | Week 1 |
| Raw Data Pipeline | 🔄 STARTING | — | Week 1 |
| Feature Engineering | 📋 PLANNED | — | Week 1-2 |
| Deep Learning Model | 📋 PLANNED | — | Week 2 |
| AIML Components | 📋 PLANNED | — | Week 2-3 |
| Medicine Tracking | 📋 PLANNED | — | Week 3 |
| UI/Dashboard | 📋 PLANNED | — | Week 4 |
| Validation & Deployment | 📋 PLANNED | — | Week 4-5 |

---

## 🎯 SUCCESS CRITERIA

✅ **MUST HAVE**:
1. Extract 200+ features from eICU raw data (vitals + labs + meds)
2. Build 24-hour temporal windows (not instant predictions)
3. 90+ AUC mortality prediction on eICU dataset
4. Multi-organ health scoring (6 organs tracked)
5. Medicine response tracking and visualization
6. SHAP-based explainability for predictions
7. Validation comparison vs SOFA/APACHE
8. UI accessible to Indian family members

✅ **NICE TO HAVE**:
- Disease-specific criteria added over time
- Challenge2012 external validation
- API + mobile app
- Real-time integration with hospital systems

---

**APPROVED FOR DEVELOPMENT**: This project scope has been validated against actual data available in your workspace. All features mentioned are extractable from eICU-CRD dataset.

**NEXT STEP**: Start with PHASE 1 - Build raw data pipeline to extract eICU features.

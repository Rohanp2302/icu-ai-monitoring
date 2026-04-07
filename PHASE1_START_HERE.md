# 🚀 START HERE - PHASE 1 ACTION PLAN
## Building Interpretable ML System for Indian Hospitals

**Goal**: Extract all features from eICU raw data → Build deep learning model → Achieve 90+ AUC

**Timeline**: Start immediately, Phase 1 completion in 3-5 days

**Prerequisites**: Tech stack installed and verified ✅

---

## ✅ PREREQUISITE: Tech Stack Verification

**BEFORE starting Phase 1, do this:**

```powershell
# 1. Check current installation
python verify_tech_stack.py

# Expected: ✅ ALL CHECKS PASSED

# If ANY ❌ FAIL, follow INSTALL_GUIDE.md
```

**Required** (must have before Phase 1):
- ✅ Python 3.8+
- ✅ PyTorch 2.2.0 with CUDA support
- ✅ Transformers (Hugging Face)
- ✅ SHAP (explainability)
- ✅ Optuna (hyperparameter tuning)
- ✅ scikit-learn, pandas, numpy

**If any missing**: Run `pip install -r requirements.txt` (takes 15 mins)

See **INSTALL_GUIDE.md** for complete setup with GPU/CUDA configuration.

---

## ✅ Project Scope VALIDATED

This project is:
- ✅ **NOT** just mortality prediction (it's organ tracking + medicine response + explainability system)
- ✅ Built on **REAL data** (eICU has 434K labs, 75K meds, not sparse data)
- ✅ Using **24-hour temporal windows** (not instant predictions)
- ✅ Using **AIML + Deep Learning** (multi-task architecture planned)
- ✅ Extracting from **RAW datasets**, not pre-processed CSVs
- ✅ Targeting **90+ AUC** as primary objective

---

## 📊 PHASE 1: RAW DATA → FEATURE ENGINEERING

### Step 1: Load eICU Core Tables

**Files to Load** (from `e:/icu_project/data/raw/eicu/`):
```
patient.csv             → Demographics, discharge status (OUTCOME)
vitalPeriodic.csv       → 1.6M vital measurements
lab.csv                 → 434K lab results (147 types)
medication.csv          → 75K medications with dosage/timing
apacheApsVar.csv        → SOFA components for organ dysfunction
treatement.csv          → Dialysis, ventilation markers
intakeOutput.csv        → Fluid balance
nurseCharting.csv       → Clinical assessment notes
admissiondrug.csv       → Pre-admission medications
```

**Task 1.1**: Create script `phase1_raw_data_loader.py`
```python
# Load all eICU tables
# Print shape, dtypes, sample rows for each
# Identify missing values
# Check time columns (minutes from admission)
# Verify outcome variable definition

Output:
- Patients loaded: 2,520
- Total vitalPeriodic records: 1,634,960
- Avg records per patient: ~648
- Outcome: Column name, unique values, event rate
```

### Step 2: Extract Vital Sign Features

**Task 1.2**: Create script `phase1_vital_extraction.py`
```python
# For each patient's vitalPeriodic data:
#   For each 1-hour window:
#     Calculate: mean, std, min, max, median, trend
#   Aggregate to: 1-hour features (not 24h yet)

Vital columns: heartrate, map, sbp, dbp, sao2, temperature, 
               respiratoryrate, cvp, sao2_calc, etc.

Output shape: (N_patients × N_hours, 15_vitals × 5_agg)
Example: (2520 × 72h avg, ~75 features)

Features:
  - HR_mean, HR_std, HR_min, HR_max, HR_trend
  - SpO2_mean, SpO2_...
  - [repeat for all vitals]
```

### Step 3: Extract Lab Features

**Task 1.3**: Create script `phase1_lab_extraction.py`
```python
# For each patient's lab tests:
#   For each hour:
#     Extract most recent lab value before this hour
#     Calculate: days since measurement, deviation from normal, trend

Priority labs (147 available, start with 30):
  Renal: creatinine, BUN, K+, Na+, Cl-
  Liver: bilirubin, AST, ALT, albumin, PT, INR
  Blood: WBC, RBC, Hgb, Hct, platelets, pH, pO2, pCO2, HCO3
  Other: glucose, CRP, procalcitonin, lactate

Output: (N_patients × N_hours, 30_labs × 3_agg)
Example features:
  - Cr_last_value, Cr_days_since, Cr_deviation
  - Bili_last_value, Bili_...
  [repeat for all labs]
```

### Step 4: Extract Medication Features

**Task 1.4**: Create script `phase1_med_extraction.py`
```python
# For each patient's medications:
#   For each hour:
#     Count active medications by category

Med categories:
  - Vasopressors? (dopamine, epinephrine, etc)
  - Sedation drugs? (propofol, midazolam, etc)
  - Insulin? (units/hour)
  - Antibiotics? (count of active)
  - Antifungals? (count active)
  - Antivirals? (count active)
  - Anticoagulants? (type + dose)
  - Diuretics? (type + amount)
  - Inotropes? (type)
  - Paralytics? (active or not)

Output: (N_patients × N_hours, 20_med_features)
Binary + continuous features indicating treatment intensity
```

### Step 5: Calculate Organ Dysfunction Scores

**Task 1.5**: Create script `phase1_organ_health.py`
```python
# For each patient's data, hour by hour:
#   Calculate 6 organ health scores (0-4 SOFA scale each)

RESPIRATORY:
  - SpO2/FiO2 ratio
  - Ventilation requirement (from treatment data)
  - SOFA score (0-4)

CARDIOVASCULAR:
  - MAP level
  - Vasopressor requirement
  - Lactate level
  - SOFA score (0-4)

RENAL:
  - Creatinine level
  - Urine output
  - SOFA score (0-4)

HEPATIC:
  - Bilirubin level
  - PT/INR
  - SOFA score (0-4)

HEMATOLOGIC:
  - Platelet count
  - SOFA score (0-4)

NEUROLOGIC:
  - GCS or sedation level
  - SOFA score (0-4)

Output: (N_patients × N_hours, 6_organs + 6_SOFA_scores)
Example:
  resp_health_0-10, resp_SOFA_0-4
  cv_health_0-10, cv_SOFA_0-4
  [etc for 6 organs]
```

### Step 6: Create 24-Hour Windows

**Task 1.6**: Create script `phase1_24h_windowing.py`
```python
# For each patient:
#   For each consecutive 24-hour window:
#     Aggregate all features into single row
#     Features: mean, std, min, max of each hourly feature
#     Create target: mortality if patient died in this window

Input: Hourly features from steps 1-5
Output: (N_windows, 200+ features)

Window logic:
  Patient with 72 hours data → 3 windows of 24h each
  Patient with 48 hours data → 2 windows
  Total windows: estimate 5,000-10,000

Target:
  - Mortality window: 1 if patient died in this 24h
  - Recovery window: 0 if patient survived

Additional outputs:
  - Window metadata (patient_id, hour_from_admission, window_number)
  - Feature statistics (mean, std, missing %)
```

### Step 7: Data Validation & Exploration

**Task 1.7**: Create script `phase1_data_validation.py`
```python
# Validate extracted data quality

Checks:
  - No NaN in feature matrix (handle with forward fill or remove)
  - Feature ranges are sensible (no -999 or 99999 errors)
  - Mortality distribution: ~5% event rate
  - Patient representation: no 1 patient dominating dataset
  - Feature correlation: no perfect (1.0) correlations

Output:
  print: Data quality report
  save: data_quality_metrics.json
  
Example output:
  "Patients with complete 24h data: 2,100 / 2,520 (83%)"
  "Windows created: 7,842"
  "Mortality rate: 5.2% (408 deaths)"
  "Features with >50% missing: 12 (removed)"
  "Final feature count: 198"
```

---

## 🎯 PHASE 1 DELIVERABLES

After completing Phase 1, you should have:

1. ✅ **Raw data loader** → Confirms all 2,500+ patients can be loaded
2. ✅ **Hourly vital features** → (2,520 patients × 72 avg hours, 75 features)
3. ✅ **Hourly lab features** → (2,520 patients × 72 avg hours, 90 features)
4. ✅ **Hourly medication features** → (2,520 patients × 72 avg hours, 20 features)
5. ✅ **Hourly organ scores** → (2,520 patients × 72 avg hours, 12 features)
6. ✅ **24-hour feature matrix** → (7,842 windows, 198 features)
7. ✅ **Data quality report** → Shows data is ready for modeling

**Output File Structure**:
```
results/
├── phase1_raw_data_summary.json (what was loaded)
├── phase1_features_hourly.pkl (intermediate hourly features)
├── phase1_windows_24h.csv (final dataset for modeling)
├── phase1_feature_names.json (list of all 198 features)
├── phase1_data_quality_report.md (validation results)
└── phase1_patient_metadata.csv (patient_id, hours_in_icu, mortality)
```

---

## ⚠️ PHASE 1 VALIDATION CHECKLIST

Before moving to Phase 2 (model building), confirm:

- [ ] All 2,520 patients loaded successfully from patient.csv
- [ ] Vitalperiodic records: 1,600,000+ rows loaded
- [ ] Lab records: 400,000+ rows loaded
- [ ] Medication records: 75,000+ rows loaded
- [ ] 24h windows created: 5,000+ windows
- [ ] Mortality rate: ~5% (matches expected)
- [ ] Features extracted from RAW data (not pre-processed CSVs)
- [ ] No NaN values in feature matrix (or properly handled)
- [ ] Organ dysfunction scores calculated for each window
- [ ] Medicine/treatment intensity captured
- [ ] All feature names documented
- [ ] Data quality report shows <5% missing data

If ANY check fails, STOP and fix before proceeding.

---

## 🔧 HOW TO START RIGHT NOW

1. **Create directory structure**:
```powershell
mkdir -p E:\icu_project\phase1_scripts
mkdir -p E:\icu_project\phase1_outputs
```

2. **Start with Task 1.1** - Data loader:
```bash
# Create: phase1_raw_data_loader.py
# Purpose: Load all eICU CSVs, print summaries
# Expected runtime: <5 minutes
# Output: Confirms data is accessible and readable
```

3. **Run and validate each step**:
- Save intermediate outputs
- Print data summaries after each step
- Do NOT proceed to next step until current step is validated

4. **Document findings**:
- What features did we find?
- Any missing data issues?
- Data quality metrics

---

## ❌ WHAT NOT TO DO IN PHASE 1

🚫 **DO NOT**:
- Use pre-processed CSVs from `data/processed/` folder
- Skip the 24-hour windowing (go straight to raw vitals)
- Ignore medication and lab data (they're critical)
- Assume features are already extracted (extract from RAW)
- Train any models yet (Phase 1 = data only, no models)
- Use only Challenge2012 (focus on eICU for now)

✅ **DO**:
- Extract from RAW data in `data/raw/eicu/`
- Handle missing values explicitly
- Document feature engineering logic
- Validate data quality at each step
- Save intermediate outputs for reproducibility

---

## 📞 VALIDATION BEFORE EACH SESSION

**Before coding, ask yourself**:
1. Are we using RAW data or processed data? → Should be RAW
2. Are we extracting eICU labs/meds or just vitals? → Should include labs + meds
3. Is prediction window 24+ hours? → Should be 24h windows
4. Are we tracking organ health? → Should have 6 organ scores
5. Do we know what features we're extracting? → Should have documented list

If answer to ANY is "no", pause and review PROJECT_SCOPE_VALIDATED.md.

---

**Ready to start?** Begin with Phase 1, Task 1.1: Raw Data Loader

Good luck! 🚀

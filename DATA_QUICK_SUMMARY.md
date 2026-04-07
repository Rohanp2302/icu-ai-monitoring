# ICU Data Exploration - QUICK REFERENCE

## Dataset Overview

| Aspect | Challenge2012 | eICU-CRD |
|--------|---------------|----------|
| **Total Patients** | 12,000 | 2,520 |
| **Mortality Rate** | ~4-5% | 5.0% (126/2520) |
| **Total Records** | 48,000 files | 4.6M rows |
| **Total Size** | ~200MB | 309 MB |
| **Primary Outcome** | `In-hospital_death` (0/1) | `unitdischargestatus` (Alive/Expired) |

## Challenge2012 Dataset Quick Facts

### Files
- **set-a**: 4,000 patients, 4,000 .txt files, Outcomes-a.txt
- **set-b**: 4,000 patients, 4,000 .txt files, Outcomes-b.txt  
- **set-c**: 4,000 patients, 4,000 .txt files, Outcomes-c.txt

### Time Granularity
- **Format**: HH:MM from ICU admission (e.g., "00:07", "00:37")
- **Typical Interval**: 7-37 minutes (IRREGULAR)
- **Data Quality**: SPARSE (many -1 missing values)

### Data Structure
```
Time,Parameter,Value
00:00,RecordID,132539
00:00,Age,54
00:07,HR,73
00:07,NIDiasABP,65
```

### Available Parameters
- Demographics: Age, Gender, Height, Weight, ICUType
- Vitals: HR, RespRate, Temp, SysABP, DiaABP, MAP, SAO2, GCS, Urine
- Scores: SAPS-I (6-21), SOFA (0-11)
- Other: Length_of_stay, Survival

### Limitations
❌ No lab values  
❌ No medications  
❌ No diagnoses  
❌ Only organ failure scores (not components)  
❌ Sparse vital measurements  

---

## eICU-CRD Dataset Quick Facts

### Key Tables Overview

| Table | Rows | Time Unit | Notes |
|-------|------|-----------|-------|
| **patient.csv** | 2,520 | - | Demographics, admission/discharge |
| **vitalPeriodic.csv** | 1,634,960 | Minutes | Dense vitals (mean 688 per pt) |
| **lab.csv** | 434,660 | Minutes | 147 unique lab types |
| **medication.csv** | 75,604 | Minutes | Drug administration records |
| **apacheApsVar.csv** | 2,205 | - | **SOFA components + APS** |
| **nurseCharting.csv** | 1,477,163 | Minutes | Nursing documentation |
| **diagnosis.csv** | 24,978 | Minutes | ICD-9 diagnoses |
| **admissiondrug.csv** | 7,417 | - | Pre-admission medications |

### Time Granularity

| Data Type | Granularity | Typical Interval | Records |
|-----------|-------------|------------------|---------|
| Vital Signs | Minutes from admission | **2-3 minutes** | 1.6M |
| Lab Tests | Minutes from admission | **Varies (4-24h)** | 435K |
| Medications | Start/Stop offsets | **Variable duration** | 75K |

### Available Lab Tests (147 Types)

**Chemistry**: BUN, Creatinine, Glucose, Albumin, Bilirubin, ALT, AST, CPK, Sodium, pH, Base Deficit/Excess, pao2, pco2  
**Hematology**: WBC (with differentials), Hematocrit, Carboxyhemoglobin  
**Cardiac**: BNP, Troponin, CRP, CPK-MB, Digoxin  
**Other**: 24h urine protein, Clostridium difficile, ESR

### SOFA Component Variables

| Organ | Variables | Values |
|-------|-----------|--------|
| **CNS** | eyes, motor, verbal | 1-15 (GCS) |
| **Respiratory** | vent | 0/1 |
| **Renal** | dialysis | 0/1 |
| **Hemodynamic** | heartrate, meanbp | Actual values |
| **Hematologic** | wbc, hematocrit | Lab values |
| **Metabolic** | glucose, sodium, creatinine, bilirubin, albumin, ph | Lab values |

### Sample Patient Flow

```
1. Patient admitted
   └─ patient.csv: Demographics, age, weight, discharge status

2. ICU monitoring starts
   └─ vitalPeriodic.csv: 688 vital records on average
      (every 2-3 min: HR, BP, Temp, RespRate, etc.)

3. Labs ordered throughout stay
   └─ lab.csv: ~170 lab tests spread over ICU stay
      (values like BUN=28, glucose=145, etc.)

4. Medications given
   └─ medication.csv + admissiondrug.csv
      (drug names, dosages, routes, start/stop times)

5. Organ dysfunction assessed
   └─ apacheApsVar.csv: SOFA scores, Apache variables

6. Discharge or death
   └─ patient.csv: unitdischargestatus = "Alive" or "Expired"
```

---

## Key Insights

### eICU vs Challenge2012

**eICU Advantages** ✅
- Dense vital measurements (1.6M records)
- Actual lab values (147 types)
- Medication administration details
- SOFA component breakdown
- Comprehensive organ dysfunction tracking
- Nursing assessments

**Challenge2012 Advantages** ✅
- 12K patients (bigger dataset)
- Simpler structure
- Established benchmark

**Recommendation**: 
🎯 **PRIMARY = eICU** (rich clinical data)  
🎯 **VALIDATION = Challenge2012** (external test set)

---

## File Paths

### Challenge2012
```
e:/icu_project/data/raw/challenge2012/
├── set-a/ (4,000 .txt files)
├── set-b/ (4,000 .txt files)
├── set-c/ (4,000 .txt files)
├── Outcomes-a.txt
├── Outcomes-b.txt
└── Outcomes-c.txt
```

### eICU
```
e:/icu_project/data/raw/eicu/
├── patient.csv
├── vitalPeriodic.csv
├── lab.csv
├── medication.csv
├── apacheApsVar.csv
├── nurseCharting.csv
├── diagnosis.csv
├── ... [25 more tables]
└── sqlite/ (alternative format)
```

---

## Time Granularity Summary

| Metric | Challenge2012 | eICU |
|--------|---------------|------|
| **Vital Time Unit** | HH:MM format | Minutes from admission |
| **Vital Frequency** | ~7-37 min intervals | ~2-3 min intervals |
| **Lab Time Unit** | ❌ Not available | Minutes from admission |
| **Medication Times** | ❌ Not available | Start/stop in minutes |
| **Total Time-series Records** | ~48,000 vital entries | ~3.0M records |

---

## Mortality Outcome

| Dataset | Outcome Field | Values | Rate |
|---------|---------------|--------|------|
| **Challenge2012** | In-hospital_death | 0=alive, 1=dead | 4-5% |
| **eICU** | unitdischargestatus | "Alive"/"Expired" | 5.0% |

---

## Feature Engineering Recommendations

### Tier 1: Vital Signs (Dense Data)
- Heart rate trends, std deviation
- Blood pressure stability
- Temperature dynamics
- Respiratory rate changes
- Oxygen saturation

### Tier 2: Lab Values (Organ Function)
- Creatinine (renal)
- Bilirubin (hepatic)
- WBC, bands (infection)
- Glucose control
- Electrolyte balance

### Tier 3: SOFA Components
- GCS (neurological)
- Mechanical ventilation requirement
- Renal replacement therapy

### Tier 4: Medications
- Vasopressor use
- Antibiotics
- Sedation intensity
- Number of active medications

### Tier 5: Diagnoses
- Sepsis presence
- ARDS status
- Trauma markers
- Comorbidity burden

---

## Answers to Key Questions

### Q: What features does eICU have that Challenge2012 doesn't?
✅ Lab chemistry (147 types)  
✅ Medication detail (names, dosages, routes, times)  
✅ Nursing assessments (1.4M records)  
✅ Diagnostic codes (ICD-9)  
✅ Dense vital measurements  
✅ Pre-admission history  
✅ Allergy information  

### Q: What's the time granularity?
- eICU vitals: **2-3 minutes** (1.6M records)
- eICU labs: **Variable** (~4-24h between tests)
- Challenge2012: **7-37 minutes** (irregular, sparse)

### Q: What's the mortality outcome?
- eICU: **5.0%** (126 deaths / 2,520 patients)
- Challenge2012: **~4-5%** across sets

### Q: Are organ dysfunction markers available?
✅ **YES** - apacheApsVar.csv has:
- SOFA components (CNS, respiratory, renal)
- Full Apache APS variables (26 columns)
- Organ-specific measurements

### Q: What medication data is available?
✅ Medication.csv: 75,604 administration records
✅ Fields: drug name, dosage, route, frequency, start/stop time
✅ admissiondrug.csv: 7,417 pre-admission medications

---

## Next Steps

1. **Load eICU data** and explore patient cohort
2. **Engineer time-series features** from vitalPeriodic.csv
3. **Create lab value panel** from 147 available tests
4. **Calculate SOFA scores** from components
5. **Build medication intensity score** from medication.csv
6. **Train initial model** on eICU
7. **Validate on Challenge2012** (set-c or external)

---

**For detailed analysis, see: DATA_EXPLORATION_REPORT.md**  
**For structured data, see: DATA_REFERENCE.json**

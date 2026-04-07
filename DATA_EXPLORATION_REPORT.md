# ICU Project Data Exploration Report
**Date:** April 7, 2026  
**Workspace:** e:/icu_project

---

## Executive Summary

This project contains two major ICU mortality prediction datasets:

1. **Challenge2012 Dataset**: 12,000 patients across 3 sets (training/validation/test), with SAPS-I and SOFA scores
2. **eICU-CRD (eICU Collaborative Research Database)**: 2,520 patients with comprehensive clinical data including vital signs, lab tests, medications, Apache scores, and nursing notes

eICU provides **significantly richer data** with time-series clinical information, lab tests, and medication records that Challenge2012 lacks.

---

## 1. CHALLENGE2012 DATASET

### 1.1 Structure Overview

```
├── set-a/        4,000 patient files
├── set-b/        4,000 patient files  
├── set-c/        4,000 patient files
├── Outcomes-a.txt
├── Outcomes-b.txt
└── Outcomes-c.txt
```

### 1.2 Patient Distribution

| Set | Patients | File Format | Outcome File |
|-----|----------|-------------|--------------|
| set-a | 4,000 | Individual .txt files | Outcomes-a.txt (4,000 rows) |
| set-b | 4,000 | Individual .txt files | Outcomes-b.txt (4,000 rows) |
| set-c | 4,000 | Individual .txt files | Outcomes-c.txt (4,000 rows) |
| **TOTAL** | **12,000** | - | - |

### 1.3 Individual Patient File Format

**File structure:** Each patient has a CSV-like file (e.g., `132539.txt`)  
**Header:** `Time,Parameter,Value`  
**Example content:**
```
Time,Parameter,Value
00:00,RecordID,132539
00:00,Age,54
00:00,Gender,0
00:00,Height,-1
00:00,ICUType,4
00:00,Weight,-1
00:07,GCS,15
00:07,HR,73
00:07,NIDiasABP,65
00:07,NIMAP,92.33
00:07,NISysABP,147
00:07,RespRate,19
00:07,Temp,35.1
00:07,Urine,900
```

**Length per patient:** ~274 lines per file (sparse time-series data)

### 1.4 Outcomes File Format

**Structure:** CSV with 6 columns and 4,000 rows per set

| Column | Description | Values | Unique |
|--------|-------------|--------|--------|
| RecordID | Patient ID | 132539-162999 | 4,000 per set |
| SAPS-I | Simplified Acute Physiology Score | 6-21 | ~35 |
| SOFA | Sequential Organ Failure Assessment | 0-11 | ~24 |
| Length_of_stay | ICU stay duration (hours) | 4-1637 | ~84 |
| Survival | Days survival after discharge | -1 to 918 | ~644 |
| In-hospital_death | Binary outcome | 0 (alive) or 1 (dead) | 2 |

**Example rows:**
```
RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death
132539,6,1,5,-1,0
132540,16,8,8,-1,0
132551,19,8,6,5,1         <- In-hospital death (outcome=1)
```

### 1.5 Data Quality Notes

- **Sparse data**: Many vital signs missing (-1 values)
- **Time granularity**: Measurements at irregular intervals (every 7-37 minutes shown in example)
- **Limited features**: Only vitals, demographics, and organ failure scores
- **No medication data**: No drug administration records
- **No lab results**: No blood work or chemistry data
- **Challenge-based**: Set-c designed as blind test set

---

## 2. eICU-CRD (eICU Collaborative Research Database)

### 2.1 Complete Data Structure (31 CSV files)

**Total size:** 308.9 MB across 4,605,753 records

| # | File Name | Rows | Cols | Size(MB) | Primary Content |
|---|-----------|------|------|----------|-----------------|
| 1 | patient.csv | **2,520** | 29 | 0.6 | Demographics, admission, discharge info |
| 2 | vitalPeriodic.csv | **1,634,960** | 19 | 83.2 | **Periodic vital signs (LARGEST)** |
| 3 | nurseCharting.csv | **1,477,163** | 8 | 106.7 | Nursing notes vital signs |
| 4 | lab.csv | **434,660** | 10 | 25.0 | **Lab test results (147 types)** |
| 5 | vitalAperiodic.csv | 274,088 | 13 | 10.2 | Non-periodic/manual vital entries |
| 6 | infusiondrug.csv | 38,256 | 9 | 2.1 | IV medication infusions |
| 7 | treatment.csv | 38,290 | 5 | 3.4 | Treatments administered |
| 8 | medication.csv | **75,604** | 15 | 6.1 | **Ordered medications** |
| 9 | physicalExam.csv | 84,058 | 6 | 11.9 | Physical exam findings |
| 10 | nurseAssessment.csv | 91,589 | 8 | 13.4 | Nursing assessments/scores |
| 11 | intakeOutput.csv | 100,466 | 12 | 15.0 | Fluid intake/output |
| 12 | respiratoryCare.csv | 5,436 | 34 | 0.5 | Mechanical ventilation parameters |
| 13 | respiratoryCharting.csv | 176,089 | 7 | 10.2 | Respiratory assessments |
| 14 | apacheApsVar.csv | **2,205** | 26 | 0.2 | **SOFA components + APS** |
| 15 | apachePatientResult.csv | 3,676 | 23 | 0.6 | Apache severity predictions |
| 16 | apachePredVar.csv | 2,205 | 51 | 0.3 | Apache predictor variables |
| 17 | diagnosis.csv | 24,978 | 7 | 2.5 | ICD-9 diagnoses |
| 18 | admissionDx.csv | 7,578 | 6 | 1.1 | Admission diagnoses |
| 19 | admissiondrug.csv | 7,417 | 14 | 2.5 | Pre-admission medications |
| 20 | note.csv | 24,758 | 8 | 3.4 | Clinical notes |
| 21 | pastHistory.csv | 12,109 | 8 | 2.0 | Patient past medical history |
| 22 | carePlanGeneral.csv | 33,148 | 6 | 1.9 | General care plans |
| 23 | carePlanGoal.csv | 3,633 | 7 | 0.2 | Care plan goals |
| 24 | carePlanCareProvider.csv | 5,627 | 8 | 0.3 | Assigned care providers |
| 25 | carePlanInfectiousDisease.csv | 112 | 8 | 0.0 | Infection-specific care plans |
| 26 | carePlanEOL.csv | 15 | 5 | 0.0 | End-of-life care plans |
| 27 | allergy.csv | 2,475 | 13 | 0.2 | Patient allergies |
| 28 | hospital.csv | 186 | 4 | 0.0 | Hospital metadata |
| 29 | microLab.csv | 342 | 7 | 0.0 | Microbiology lab results |
| 30 | customLab.csv | 30 | 7 | 0.0 | Custom lab tests |
| - | sqlite/ | - | - | - | SQLite database version |

### 2.2 Key Patient Demographics (patient.csv)

**Total Unique Patients:** 2,520  
**Patient Features (29 columns):**

| Demographics | Admission Info | Discharge Info | Clinical |
|---|---|---|---|
| patientunitstayid | hospitaladmittime24 | hospitaldischargetime24 | unittype |
| patienthealthsystemstayid | hospitaladmitsource | hospitaldischargelocation | unitstaytype |
| gender | unitadmitsource | hospitaldischargestatus | apacheadmissiondx |
| age | unitadmittime24 | unitdischargestatus | (outcome) |
| ethnicity | unitvisitnumber | unitdischargeoffset | - |
| admissionheight | - | unitdischargelocation | - |
| admissionweight | - | dischargeweight | - |
| dischargeweight | - | - | - |

### 2.3 MORTALITY & OUTCOME DEFINITION

| Metric | Value |
|--------|-------|
| **Total Patients** | 2,520 |
| **Deaths (Expired)** | 126 (5.0%) |
| **Alive** | 2,392 (94.9%) |
| **Unknown** | 2 |
| **Outcome Field** | `unitdischargestatus` in patient.csv |
| **Values** | "Alive", "Expired", NaN |

> **Challenge2012 uses:** `In-hospital_death` (0/1 binary)  
> **eICU uses:** `unitdischargestatus` (categorical: "Alive"/"Expired")

---

## 3. VITAL SIGNS & TIME-SERIES DATA

### 3.1 Vital Periodic (vitalPeriodic.csv) - PRIMARY VITALS

**Largest table: 1,634,960 records from 2,520 patients**

#### Time Granularity
- **Type:** Minutes from ICU admission
- **Column:** `observationoffset` (in minutes)
- **Records per patient:** Min=1, Max=13,150, **Mean=688.4**
- **Typical interval:** ~2.4 minutes between recordings
- **Quality:** Dense, regular time-series monitoring

#### Available Vital Parameters (19 columns)

| Cardiovascular | Respiratory | CNS | Other |
|---|---|---|---|
| heartrate | respiration | temperature | sao2 |
| systemicsystolic | etco2 | - | cvp |
| systemicdiastolic | - | - | - |
| systemicmean | - | - | - |
| pasystolic | - | - | - |
| padiastolic | - | - | - |
| pamean | - | - | - |
| st1, st2, st3 | - | - | icp |

**Example vital record:**
```
patientunitstayid=141765, observationoffset=1179 minutes
  temperature: NaN (not measured)
  heartrate: 82.0 bpm
  sao2: (measured)
  respiration: (measured)
```

### 3.2 Vital Aperiodic (vitalAperiodic.csv)

**274,088 records** - Manual/non-periodic vital entries  
Used for measurements taken outside routine monitoring

### 3.3 Nurse Charting (nurseCharting.csv)

**1,477,163 records** - Nursing documentation of vitals  
Contains nursing chart labels for captured vital signs

---

## 4. LAB TESTS & CHEMISTRY DATA

### 4.1 Lab Results (lab.csv)

**434,660 total lab records**

#### Unique Lab Tests Available: **147 types**

##### Blood Chemistry Tests
- BUN, Creatinine, Glucose, Albumin, Bilirubin
- ALT (SGPT), AST (SGOT), CPK, CPK-MB
- Sodium, PO2, PCO2, pH
- Base Deficit, Base Excess

##### Hematology Tests
- WBC (with differentials: -bands, -basos, -eos, -lymphs, -monos, -polys)
- Carboxyhemoglobin
- Hematocrit

##### Cardiac Tests
- BNP, Troponin, CRP, ESR
- Digoxin, Amikacin (peak/trough)

##### Microbiology
- Clostridium difficile toxin A+B
- Blood cultures (implied by microLab)

##### Drug Levels
- Carbamazepine, Digoxin
- Gentamicin, Vancomycin
- Phenytoin, Theophylline

##### Other
- 24-hour urine protein, urea nitrogen
- ANF/ANA (autoimmune markers)
- Fe (Iron), D-dimer

#### Time Granularity
- **Column:** `labresultoffset` (minutes from ICU admission)
- **Type:** Clinical lab draw times (less frequent than vitals)

**Sample lab record:**
```
labid=X, patientunitstayid=Y, labresultoffset=120 (minutes)
  labname: "BUN"
  labresult: 28.5
  labmeasurenamesystem: "mg/dL"
```

---

## 5. MEDICATION DATA

### 5.1 Medications Administered (medication.csv)

**75,604 medication records** - Drugs given during ICU stay

#### Columns (15):
- `drugname`: Name of medication
- `dosage`: Amount and unit
- `routeadmin`: Route (IV, oral, etc.)
- `frequency`: Dosing frequency
- `drugstartoffset`: When started (minutes from admission)
- `drugstopoffset`: When stopped
- `drugivadmixture`: IV mixture flag
- `drugordercancelled`: Cancelled flag
- `loadingdose`: Initial loading dose
- `prn`: As-needed flag
- `gtc`: Gastric tube compatible
- Plus: medicationid, patientunitstayid, drugorderoffset, drughiclseqno

#### Time Granularity
- `drugstartoffset` and `drugstopoffset` in minutes
- Allows tracking medication duration

### 5.2 Pre-Admission Medications (admissiondrug.csv)

**7,417 pre-admission medication records** - Home medications

#### Columns (14):
- Same structure as medication.csv
- `drugoffset`: Pre-admission drug entry time
- `specialtytype`: Medication specialty
- `usertype`: Type of provider entering data

> **Key Difference:** admission drugs recorded PRE-ICU, medications recorded IN ICU

---

## 6. ORGAN DYSFUNCTION SCORES

### 6.1 SOFA Components (apacheApsVar.csv)

**2,205 records** - SOFA scoring variables for 2,205 patients  

#### SOFA Component Variables (from 26 columns):
| Organ System | Variables | Values |
|---|---|---|
| **CNS** | eyes, motor, verbal | Glasgow Coma Scale components (1-15 scale) |
| **Respiratory** | vent | Binary: intubated/mechanically ventilated |
| **Renal** | dialysis | Binary: on renal replacement therapy |
| **Other APS** | intubated, meds, urine | Intubation status, medications used, urine output |

#### Complete APS Variables (26 columns):
- **Organ dysfunction:** intubated, vent, dialysis, eyes, motor, verbal
- **Hemodynamics:** meanbp, temperature, heartrate
- **Blood gas:** pH, pco2, pao2, fio2
- **Chemistry:** sodium, bun, creatinine, glucose
- **Heme:** wbc, hematocrit, bilirubin
- **Other:** meds, urine

**Example:**
```
patientunitstayid=X
  eyes: 4, motor: 6, verbal: 5 -> GCS = 15
  vent: 0 (not ventilated)
  dialysis: 0 (not on dialysis)
  pH: 7.35, po2: 95, pco2: 40
  glucose: 145, creatinine: 1.2, wbc: 8.5
```

### 6.2 Apache Prediction Scores

- **apachePatientResult.csv** (3,676 records): Predicted mortality risk scores
- **apachePredVar.csv** (2,205 records): 51 predictor variables

---

## 7. DIAGNOSTIC & CLINICAL INFORMATION

### 7.1 Diagnoses (diagnosis.csv)

**24,978 diagnosis records**

| Field | Type | Example |
|---|---|---|
| diagnosisstring | Text | Diagnosis description |
| icd9code | Code | ICD-9 code |
| diagnosispriority | Numeric | Priority ranking |
| activeupondischarge | Binary | Still active at discharge |
| diagnosisoffset | Minutes | When diagnosed |

### 7.2 Clinical Notes (note.csv)

**24,758 note records** - Clinical documentation  
Type field indicates note categories (assessments, plans, nursing notes, etc.)

---

## 8. DATA QUALITY & AVAILABILITY ANALYSIS

### 8.1 Time Granularity Summary

| Data Type | Time Unit | Interval | Density |
|---|---|---|---|
| **Vital Periodic** | Minutes from admission | ~2-3 min | DENSE |
| **Lab Tests** | Minutes from admission | Irregular (varies) | SPARSE |
| **Medications** | Minutes from admission (start/stop) | Variable duration | MODERATE |
| **Challenge2012** | HH:MM from admission | ~7-37 min gaps | SPARSE |

### 8.2 Feature Availability

#### eICU has but Challenge2012 lacks:
✅ Actual lab chemistry values (147 lab types)  
✅ Medication administration details (drug names, dosages, routes)  
✅ Detailed nursing assessments (1.4M records)  
✅ Organ dysfunction scores (SOFA components, Apache)  
✅ ICD-9 diagnoses with severity  
✅ Dense vital sign time-series (1.6M records, ~2-3 min intervals)  
✅ Pre-admission medication history  
✅ Patient allergy information  
✅ Clinical notes/documentation  

#### Challenge2012 unique features:
✅ SAPS-I score (pre-computed severity)  
✅ Simpler, more standardized structure  
✅ Larger dataset (12,000 vs 2,520 patients)  

### 8.3 Data Coverage Issues

**eICU (2,520 patients):**
- Patient data: 100% (2,520/2,520)
- Vital periodic: 100% patients, variable record counts
- Lab tests: 434,660/1,634,960 vitals (~26% as lab records)
- SOFA/Apache: 2,205/2,520 patients (87.5%)
- Some fields have NaN values

**Challenge2012 (12,000 patients):**
- Set-a: 4,000 patients with outcomes
- Set-b: 4,000 patients with outcomes
- Set-c: 4,000 patients with outcomes
- Many vital values are -1 (missing)

---

## 9. KEY QUESTIONS ANSWERED

### Q1: What features are available in eICU that Challenge2012 doesn't have?

**eICU Advantages:**
1. **Lab Chemistry** - 147 unique lab types (blood work, electrolytes, etc.)
2. **Medication Details** - Drug names, dosages, routes, frequencies, start/stop times
3. **Nursing Data** - 1.4M nursing chart entries with detailed assessments
4. **Diagnostic Data** - ICD-9 codes, diagnosis descriptions, active status
5. **Pre-Admission History** - Past medical history, home medications, allergies
6. **Clinical Notes** - Full documentation with timestamps
7. **Intensive Monitoring** - 1.6M vital sign records (dense time-series)
8. **Organ Failure Data** - SOFA component variables with values

**Challenge2012:**
- Only vitals and organ failure SCORES (not components)
- No actual medication records
- No lab values
- Sparse vital data (many -1 missing values)

### Q2: What's the time granularity?

| Dataset | Granularity | Typical Interval |
|---|---|---|
| **eICU Vitals** | Minutes from ICU admission | 2-3 minutes |
| **eICU Labs** | Minutes from ICU admission | Variable (hours) |
| **eICU Meds** | Minutes from start/stop | Duration tracked |
| **Challenge2012** | HH:MM from admission | 7-37 minutes (irregular) |

### Q3: What's the mortality rate/outcome definition?

| Dataset | Outcome Variable | Definition | Rate |
|---|---|---|---|
| **eICU** | `unitdischargestatus` | "Alive" / "Expired" | 5.0% mortality |
| **Challenge2012** | `In-hospital_death` | 0 (survived) / 1 (died) | ~4-5% |

### Q4: Are organ dysfunction markers available?

**YES - eICU has comprehensive organ dysfunction data:**

| Organ System | Variables | Location |
|---|---|---|
| **CNS** | eyes, motor, verbal | apacheApsVar.csv (SOFA) |
| **Respiratory** | vent, respiration | apacheApsVar.csv + vitalPeriodic |
| **Renal** | dialysis, creatinine, urine | apacheApsVar.csv + lab.csv |
| **Hepatic** | bilirubin | lab.csv |
| **Cardiovascular** | heartrate, meanbp | vitalPeriodic.csv |
| **Hematologic** | wbc, hematocrit, platelets | lab.csv |

**Challenge2012:**
- Only has SOFA aggregate score (0-11)
- No component breakdown

### Q5: What medicine/medication data is available?

| Aspect | eICU | Challenge2012 |
|---|---|---|
| **Administered Meds** | 75,604 records with full details | ❌ None |
| **Pre-admission Meds** | 7,417 records | ❌ None |
| **Drug Name** | ✅ Yes | ❌ No |
| **Dosage** | ✅ Yes | ❌ No |
| **Route** | ✅ Yes (IV, oral, etc.) | ❌ No |
| **Frequency** | ✅ Yes | ❌ No |
| **Start/Stop Time** | ✅ Yes (minutes) | ❌ No |
| **IV Infusions** | ✅ 38,256 records | ❌ No |

---

## 10. RECOMMENDATIONS FOR ICU PROJECT

### Dataset Choice
**Recommendation: PRIMARY = eICU, SECONDARY = Challenge2012**

**Rationale:**
- eICU has 1,634,960 vital measurements vs Challenge2012's sparse data
- 434,660 lab test results with actual values (not available in Challenge2012)
- 75,604 medication administration records with clinical context
- SOFA component scoring enables understanding organ dysfunction mechanisms
- Smaller patient count (2,520 vs 12,000) but MUCH richer clinical data

### Data Integration Strategy

1. **Use eICU as primary training data**
   - Rich time-series features (vitals every 2-3 min)
   - Lab tests every 4-24 hours
   - Complete medication profiles

2. **Use Challenge2012 for external validation**
   - Larger patient count provides different patient population
   - Tests generalization
   - Different data collection procedures/institutions

3. **Feature Engineering Priority**

| Priority | Feature Source | Type |
|---|---|---|
| 1 | Vital trends | Periodic + Apache |
| 2 | Lab values | lab.csv (147 types) |
| 3 | SOFA components | apacheApsVar.csv |
| 4 | Medication timing | medication.csv |
| 5 | Diagnosis severity | diagnosis.csv + ICD-9 |
| 6 | Nursing assessments | nurseAssessment.csv |

### Expected Model Performance Advantages

**With eICU data:**
- Dense temporal monitoring enables LSTM/RNN models
- 147 lab types provide comprehensive biochemical picture
- Medication timing captures treatment intensity
- SOFA components enable interpretability (which organs are failing?)

**Challenge2012 alone would be limited:**
- Sparse vital data
- No lab values
- No medication context
- Only aggregate failure scores

---

## 11. DATA FILES QUICK REFERENCE

### Challenge2012 Location
```
e:/icu_project/data/raw/challenge2012/
├── set-a/          (4,000 .txt files)
├── set-b/          (4,000 .txt files)
├── set-c/          (4,000 .txt files)
├── Outcomes-a.txt  (4,000 rows, 6 columns)
├── Outcomes-b.txt
└── Outcomes-c.txt
```

### eICU Location
```
e:/icu_project/data/raw/eicu/
├── patient.csv                      (2,520 patients)
├── vitalPeriodic.csv               (1,634,960 records)
├── lab.csv                         (434,660 lab tests)
├── medication.csv                  (75,604 medications)
├── admissiondrug.csv               (7,417 pre-admission meds)
├── apacheApsVar.csv                (2,205 SOFA + APS)
├── nurseCharting.csv               (1,477,163 nursing notes)
├── diagnosis.csv                   (24,978 diagnoses)
├── vitalAperiodic.csv              (274,088 aperiodic vitals)
├── intakeOutput.csv                (100,466 I/O records)
├── respiratoryCharting.csv         (176,089 respiratory)
├── physicalExam.csv                (84,058 exams)
├── nurseAssessment.csv             (91,589 assessments)
├── apachePatientResult.csv         (3,676 Apache scores)
├── apachePredVar.csv               (2,205 predictor vars)
├── pastHistory.csv                 (12,109 history)
├── note.csv                        (24,758 clinical notes)
├── carePlanGeneral.csv             (33,148 care plans)
├── admissionDx.csv                 (7,578 admission dx)
├── diagnosis.csv                   (diagnoses)
├── treatment.csv                   (38,290 treatments)
├── nurseCare.csv                   (42,080 care records)
├── infusiondrug.csv                (38,256 IV infusions)
├── allergy.csv                     (2,475 allergies)
├── respiratoryCare.csv             (5,436 vent parameters)
├── carePlanGoal.csv, carePlanEOL.csv, etc.
├── microLab.csv                    (342 microbiology)
├── customLab.csv                   (30 custom labs)
├── hospital.csv                    (186 hospitals)
└── sqlite/                         (SQLite version)
```

---

## 12. COLUMN INVENTORY

### Patient Demographics (from patient.csv)
patientunitstayid, patienthealthsystemstayid, gender, age, ethnicity, hospitalid, wardid, apacheadmissiondx, admissionheight, hospitaladmittime24, hospitaladmitoffset, hospitaladmitsource, hospitaldischargeyear, hospitaldischargetime24, hospitaldischargeoffset, hospitaldischargelocation, hospitaldischargestatus, unittype, unitadmittime24, unitadmitsource, unitvisitnumber, unitstaytype, admissionweight, dischargeweight, unitdischargetime24, unitdischargeoffset, unitdischargelocation, unitdischargestatus, uniquepid

### Vital Signs - Periodic (from vitalPeriodic.csv)
vitalperiodicid, patientunitstayid, observationoffset, temperature, sao2, heartrate, respiration, cvp, etco2, systemicsystolic, systemicdiastolic, systemicmean, pasystolic, padiastolic, pamean, st1, st2, st3, icp

### Lab Tests (from lab.csv)
labid, patientunitstayid, labresultoffset, labtypeid, labname, labresult, labresulttext, labmeasurenamesystem, labmeasurenameinterface, labresultrevisedoffset

### Medications (from medication.csv)
medicationid, patientunitstayid, drugorderoffset, drugstartoffset, drugivadmixture, drugordercancelled, drugname, drughiclseqno, dosage, routeadmin, frequency, loadingdose, prn, drugstopoffset, gtc

### SOFA & Apache APS (from apacheApsVar.csv)
apacheapsvarid, patientunitstayid, intubated, vent, dialysis, eyes, motor, verbal, meds, urine, wbc, temperature, respiratoryrate, sodium, heartrate, meanbp, ph, hematocrit, creatinine, albumin, pao2, pco2, bun, glucose, bilirubin, fio2

### Nursing Documentation (from nurseCharting.csv)
nursingchartid, patientunitstayid, nursingchartoffset, nursingchartentryoffset, nursingchartcelltypecat, nursingchartcelltypevallabel, nursingchartcelltypevalname, nursingchartvalue

### Diagnoses (from diagnosis.csv)
diagnosisid, patientunitstayid, activeupondischarge, diagnosisoffset, diagnosisstring, icd9code, diagnosispriority

---

## Conclusion

**eICU-CRD is the superior dataset for this ICU project** due to:
- 1.6M dense vital sign measurements
- 434K lab test results (147 types)
- 75K medication records with clinical context
- Comprehensive organ dysfunction scoring
- Complete patient outcomes

**Challenge2012 serves as valuable external validation** with a larger patient cohort but simpler data structure.

**Combined approach maximizes model robustness** by training on rich eICU data and validating on Challenge2012's independent patient population.

---

*End of Report*

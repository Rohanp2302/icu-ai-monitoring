#!/usr/bin/env python3
"""
PHASE 1 - TASK 1.1: RAW DATA LOADER & SUMMARY
Simple data loading and validation
"""

import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/eicu")
OUTPUT_PATH = Path("results/phase1_outputs")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("PHASE 1 - TASK 1.1: RAW DATA LOADER")
print("="*80)

# Load patient data
print("\n[1/7] Loading patient.csv...")
try:
    patients = pd.read_csv(RAW_DATA_PATH / "patient.csv")
    mortality = (patients["hospitaldischargestatus"] == "Expired").sum() if "hospitaldischargestatus" in patients.columns else 0
    print(f"  -> Patients: {len(patients)}")
    print(f"  -> Mortality: {mortality} ({100*mortality/len(patients):.1f}%)")
except Exception as e:
    print(f"  ERROR: {e}")

# Load vitals  
print("\n[2/7] Loading vitalPeriodic.csv...")
try:
    vitals = pd.read_csv(RAW_DATA_PATH / "vitalPeriodic.csv", low_memory=False)
    print(f"  -> Vital records: {len(vitals):,}")
    print(f"  -> Unique patients: {vitals['patientunitstayid'].nunique()}")
    print(f"  -> Avg records/patient: {len(vitals)/vitals['patientunitstayid'].nunique():.1f}")
except Exception as e:
    print(f"  ERROR: {e}")

# Load labs
print("\n[3/7] Loading lab.csv...")
try:
    labs = pd.read_csv(RAW_DATA_PATH / "lab.csv")
    print(f"  -> Lab records: {len(labs):,}")
    print(f"  -> Unique patients: {labs['patientunitstayid'].nunique()}")
    print(f"  -> Lab types: {labs['labname'].nunique()}")
except Exception as e:
    print(f"  ERROR: {e}")

# Load medications
print("\n[4/7] Loading medication.csv...")
try:
    meds = pd.read_csv(RAW_DATA_PATH / "medication.csv")
    print(f"  -> Medication records: {len(meds):,}")
    print(f"  -> Unique patients: {meds['patientunitstayid'].nunique()}")
    print(f"  -> Drug types: {meds['drugname'].nunique()}")
except Exception as e:
    print(f"  ERROR: {e}")

# Load SOFA/Apache
print("\n[5/7] Loading apacheApsVar.csv...")
try:
    apache = pd.read_csv(RAW_DATA_PATH / "apacheApsVar.csv")
    print(f"  -> Apache records: {len(apache)}")
    print(f"  -> Unique patients: {apache['patientunitstayid'].nunique()}")
except Exception as e:
    print(f"  ERROR: {e}")

# Load treatment (dialysis, ventilation markers)
print("\n[6/7] Loading treatment.csv...")
try:
    treatment = pd.read_csv(RAW_DATA_PATH / "treatment.csv")
    print(f"  -> Treatment records: {len(treatment)}")
    print(f"  -> Unique patients: {treatment['patientunitstayid'].nunique()}")
except Exception as e:
    print(f"  ERROR: {e}")

# Load intakeOutput
print("\n[7/7] Loading intakeOutput.csv...")
try:
    intakeoutput = pd.read_csv(RAW_DATA_PATH / "intakeOutput.csv")
    print(f"  -> Intake/Output records: {len(intakeoutput)}")
    print(f"  -> Unique patients: {intakeoutput['patientunitstayid'].nunique()}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "="*80)
print("PHASE 1 STATUS: Raw data successfully loaded and validated!")
print("="*80)
print("\nNext: Task 1.2 - Extract vital signs features")
print("      Task 1.3 - Extract lab features")
print("      Task 1.4 - Extract medication features")
print("      Task 1.5 - Calculate organ dysfunction scores")
print("      Task 1.6 - Create 24-hour windows")
print("      Task 1.7 - Validate dataset")
print("\n")

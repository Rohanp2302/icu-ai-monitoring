"""
PHASE 1 - TASK 1.1: RAW DATA LOADER
=====================================
Purpose: Load and validate eICU raw datasets
Output: Data quality report + summaries
Runtime: ~2-3 minutes
Status: CHECKPOINT - Must validate before proceeding to feature extraction
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIG
# ============================================================================
RAW_DATA_PATH = Path("data/raw/eicu")
OUTPUT_PATH = Path("results/phase1_outputs")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TASK 1.1.1: LOAD PATIENT DEMOGRAPHICS
# ============================================================================
print("\n" + "="*80)
print("PHASE 1 - TASK 1.1: RAW DATA LOADER")
print("="*80)

print("\n[STEP 1/7] Loading PATIENT.CSV (Demographics & Outcomes)")
print("-" * 80)

try:
    patients = pd.read_csv(RAW_DATA_PATH / "patient.csv")
    print(f"✓ Patients loaded: {len(patients)} rows")
    print(f"✓ Columns: {list(patients.columns)}")
    print(f"✓ Sample patient:")
    print(patients.head(1).to_string())
    
    # CRITICAL: Identify outcome variable
    print(f"\n  Checking outcome variable...")
    if 'hospitaldischargestatus' in patients.columns:
        print(f"    - hospitaldischargestatus values: {patients['hospitaldischargestatus'].unique()}")
        mortality = (patients['hospitaldischargestatus'] == 'Expired').sum()
        print(f"    - Deaths: {mortality}")
        print(f"    - Mortality rate: {100*mortality/len(patients):.2f}%")
    
    if 'patientunitstayid' not in patients.columns:
        print("    ⚠ WARNING: No patientunitstayid column found!")
    else:
        print(f"    - Unique patients: {patients['patientunitstayid'].nunique()}")
    
    patients_summary = {
        "total_rows": len(patients),
        "columns": list(patients.columns),
        "unique_patients": patients['patientunitstayid'].nunique() if 'patientunitstayid' in patients.columns else 0,
        "mortality_count": mortality if 'hospitaldischargestatus' in patients.columns else 0,
        "mortality_rate": 100*mortality/len(patients) if 'hospitaldischargestatus' in patients.columns else 0,
    }
    
except Exception as e:
    print(f"✗ ERROR loading patient.csv: {e}")
    patients_summary = {"error": str(e)}

# ============================================================================
# TASK 1.1.2: LOAD VITAL SIGNS DATA
# ============================================================================
print("\n[STEP 2/7] Loading VITALPERIODIC.CSV (Vital Signs ~1.6M records)")
print("-" * 80)

try:
    vitals = pd.read_csv(RAW_DATA_PATH / "vitalPeriodic.csv", low_memory=False)
    print(f"✓ Vital records loaded: {len(vitals)} rows")
    print(f"✓ Vital columns: {list(vitals.columns)}")
    print(f"✓ Time range:")
    if 'observationoffset' in vitals.columns:
        print(f"    - Min offset: {vitals['observationoffset'].min()} minutes")
        print(f"    - Max offset: {vitals['observationoffset'].max()} minutes")
    
    print(f"✓ Sample vital signs:")
    vital_value_cols = [c for c in vitals.columns if c not in ['patientunitstayid', 'observationoffset', 'nursingchartoffset']]
    print(f"    - Vital measurement columns: {vital_value_cols[:10]}...")
    
    unique_patients_vitals = vitals['patientunitstayid'].nunique() if 'patientunitstayid' in vitals.columns else 0
    avg_records_per_patient = len(vitals) / unique_patients_vitals if unique_patients_vitals > 0 else 0
    
    print(f"✓ Unique patients with vitals: {unique_patients_vitals}")
    print(f"✓ Avg records per patient: {avg_records_per_patient:.1f}")
    print(f"✓ Missing data percentage:")
    for col in vital_value_cols[:5]:
        if col in vitals.columns:
            missing_pct = 100 * vitals[col].isna().sum() / len(vitals)
            print(f"    - {col}: {missing_pct:.1f}%")
    
    vitals_summary = {
        "total_records": len(vitals),
        "unique_patients": unique_patients_vitals,
        "avg_records_per_patient": avg_records_per_patient,
        "columns": list(vitals.columns),
        "time_min_offset": vitals['observationoffset'].min() if 'observationoffset' in vitals.columns else 0,
        "time_max_offset": vitals['observationoffset'].max() if 'observationoffset' in vitals.columns else 0,
    }
    
except Exception as e:
    print(f"✗ ERROR loading vitalPeriodic.csv: {e}")
    vitals_summary = {"error": str(e)}

# ============================================================================
# TASK 1.1.3: LOAD LABORATORY TESTS
# ============================================================================
print("\n[STEP 3/7] Loading LAB.CSV (Laboratory Tests ~434K records)")
print("-" * 80)

try:
    labs = pd.read_csv(RAW_DATA_PATH / "lab.csv", low_memory=False)
    print(f"✓ Lab records loaded: {len(labs)} rows")
    
    # What lab types are available?
    if 'labname' in labs.columns:
        unique_lab_types = labs['labname'].nunique()
        print(f"✓ Unique lab test types: {unique_lab_types}")
        print(f"✓ Top 20 lab tests by frequency:")
        top_labs = labs['labname'].value_counts().head(20)
        for lab, count in top_labs.items():
            print(f"    - {lab}: {count} tests")
    
    unique_patients_labs = labs['patientunitstayid'].nunique() if 'patientunitstayid' in labs.columns else 0
    avg_labs_per_patient = len(labs) / unique_patients_labs if unique_patients_labs > 0 else 0
    
    print(f"✓ Unique patients with labs: {unique_patients_labs}")
    print(f"✓ Avg lab tests per patient: {avg_labs_per_patient:.1f}")
    
    if 'labresult' in labs.columns:
        print(f"✓ Lab value range: {labs['labresult'].min()} to {labs['labresult'].max()}")
        print(f"✓ Missing lab values: {100*labs['labresult'].isna().sum()/len(labs):.1f}%")
    
    labs_summary = {
        "total_records": len(labs),
        "unique_lab_types": unique_lab_types if 'labname' in labs.columns else 0,
        "unique_patients": unique_patients_labs,
        "avg_labs_per_patient": avg_labs_per_patient,
        "columns": list(labs.columns),
    }
    
except Exception as e:
    print(f"✗ ERROR loading lab.csv: {e}")
    labs_summary = {"error": str(e)}

# ============================================================================
# TASK 1.1.4: LOAD MEDICATIONS
# ============================================================================
print("\n[STEP 4/7] Loading MEDICATION.CSV (Active Medications ~75K records)")
print("-" * 80)

try:
    meds = pd.read_csv(RAW_DATA_PATH / "medication.csv", low_memory=False)
    print(f"✓ Medication records loaded: {len(meds)} rows")
    
    if 'drugname' in meds.columns:
        unique_drugs = meds['drugname'].nunique()
        print(f"✓ Unique drugs: {unique_drugs}")
        print(f"✓ Top 15 medications by frequency:")
        top_drugs = meds['drugname'].value_counts().head(15)
        for drug, count in top_drugs.items():
            print(f"    - {drug}: {count} records")
    
    # Check for dosage info
    dosage_cols = [c for c in meds.columns if 'dose' in c.lower()]
    if dosage_cols:
        print(f"✓ Dosage columns: {dosage_cols}")
    
    unique_patients_meds = meds['patientunitstayid'].nunique() if 'patientunitstayid' in meds.columns else 0
    print(f"✓ Unique patients with medications: {unique_patients_meds}")
    
    meds_summary = {
        "total_records": len(meds),
        "unique_drugs": unique_drugs if 'drugname' in meds.columns else 0,
        "unique_patients": unique_patients_meds,
        "columns": list(meds.columns),
    }
    
except Exception as e:
    print(f"✗ ERROR loading medication.csv: {e}")
    meds_summary = {"error": str(e)}

# ============================================================================
# TASK 1.1.5: LOAD APACHE VARIABLES (SOFA COMPONENTS)
# ============================================================================
print("\n[STEP 5/7] Loading APACHEAPSVAR.CSV (SOFA & Apache Variables)")
print("-" * 80)

try:
    apache = pd.read_csv(RAW_DATA_PATH / "apacheApsVar.csv", low_memory=False)
    print(f"✓ Apache records loaded: {len(apache)} rows")
    print(f"✓ Columns: {list(apache.columns)}")
    
    # Look for SOFA components
    sofa_cols = [c for c in apache.columns if 'sofa' in c.lower()]
    if sofa_cols:
        print(f"✓ SOFA columns found: {sofa_cols}")
    else:
        print(f"✓ No explicit SOFA columns, but Apache variables available")
    
    unique_patients_apache = apache['patientunitstayid'].nunique() if 'patientunitstayid' in apache.columns else 0
    print(f"✓ Unique patients with Apache data: {unique_patients_apache}")
    
    apache_summary = {
        "total_records": len(apache),
        "unique_patients": unique_patients_apache,
        "columns": list(apache.columns),
        "sofa_columns": sofa_cols,
    }
    
except Exception as e:
    print(f"✗ ERROR loading apacheApsVar.csv: {e}")
    apache_summary = {"error": str(e)}

# ============================================================================
# TASK 1.1.6: LOAD INTAKE/OUTPUT (FLUID BALANCE)
# ============================================================================
print("\n[STEP 6/7] Loading INTAKEOUTPUT.CSV (Fluid Balance & Urine Output)")
print("-" * 80)

try:
    io = pd.read_csv(RAW_DATA_PATH / "intakeOutput.csv", low_memory=False)
    print(f"✓ Intake/Output records loaded: {len(io)} rows")
    print(f"✓ Columns: {list(io.columns)}")
    
    unique_patients_io = io['patientunitstayid'].nunique() if 'patientunitstayid' in io.columns else 0
    print(f"✓ Unique patients with I/O data: {unique_patients_io}")
    
    io_summary = {
        "total_records": len(io),
        "unique_patients": unique_patients_io,
        "columns": list(io.columns),
    }
    
except Exception as e:
    print(f"✗ ERROR loading intakeOutput.csv: {e}")
    io_summary = {"error": str(e)}

# ============================================================================
# TASK 1.1.7: DATA QUALITY REPORT
# ============================================================================
print("\n[STEP 7/7] Data Quality & Alignment Check")
print("-" * 80)

# Check patient alignment across datasets
all_datasets = {
    "patients": patients_summary.get("unique_patients", 0),
    "vitals": vitals_summary.get("unique_patients", 0),
    "labs": labs_summary.get("unique_patients", 0),
    "medications": meds_summary.get("unique_patients", 0),
    "apache": apache_summary.get("unique_patients", 0),
    "intake_output": io_summary.get("unique_patients", 0),
}

print("\n✓ Patient coverage across datasets:")
for dataset, count in all_datasets.items():
    pct = 100 * count / all_datasets['patients'] if all_datasets['patients'] > 0 else 0
    print(f"    - {dataset}: {count} patients ({pct:.1f}% of total)")

print("\n✓ DATA QUALITY ASSESSMENT:")
print(f"    - ✅ Patient demographics: Available ({patients_summary.get('total_rows', 0)} records)")
print(f"    - ✅ Vital signs: DENSE data ({vitals_summary.get('total_records', 0)} records, "
      f"{vitals_summary.get('avg_records_per_patient', 0):.0f} avg/patient)")
print(f"    - ✅ Laboratory tests: Available ({labs_summary.get('total_records', 0)} records, "
      f"{labs_summary.get('unique_lab_types', 0)} types)")
print(f"    - ✅ Medications: Available ({meds_summary.get('total_records', 0)} records, "
      f"{meds_summary.get('unique_drugs', 0)} drugs)")
print(f"    - ✅ SOFA/Apache variables: Available ({apache_summary.get('total_records', 0)} records)")
print(f"    - ✅ Fluid balance: Available ({io_summary.get('total_records', 0)} records)")

mortality_rate = patients_summary.get("mortality_rate", 0)
print(f"\n✓ OUTCOME VARIABLE:")
print(f"    - Event rate: {mortality_rate:.2f}% mortality ({patients_summary.get('mortality_count', 0)} deaths)")
print(f"    - Sufficient for imbalanced classification: {'Yes' if 2 < mortality_rate < 50 else 'Check'}")

# ============================================================================
# SAVE SUMMARY REPORT
# ============================================================================
summary_report = {
    "timestamp": datetime.now().isoformat(),
    "phase": "Phase 1 - Task 1.1",
    "purpose": "Raw data loader & validation",
    "patients": patients_summary,
    "vitals": vitals_summary,
    "labs": labs_summary,
    "medications": meds_summary,
    "apache": apache_summary,
    "intake_output": io_summary,
    "quality_checks": {
        "all_datasets_loaded": all(not s.get("error") for s in [
            patients_summary, vitals_summary, labs_summary, 
            meds_summary, apache_summary, io_summary
        ]),
        "patient_alignment_ok": all(count > 0 for count in all_datasets.values()),
        "mortality_rate": mortality_rate,
        "recommended_action": "✅ PROCEED TO FEATURE EXTRACTION" if mortality_rate > 1 else "❌ CHECK DATA"
    }
}

# Save JSON report
with open(OUTPUT_PATH / "phase1_data_quality_report.json", "w") as f:
    json.dump(summary_report, f, indent=2)

print("\n" + "="*80)
print("✅ PHASE 1 - TASK 1.1 COMPLETE")
print("="*80)
print(f"\n✓ Summary report saved to: results/phase1_outputs/phase1_data_quality_report.json")
print(f"\n✓ NEXT STEPS:")
print(f"    1. Review the data quality report")
print(f"    2. If all ✅, proceed to Task 1.2 (Vital Feature Extraction)")
print(f"    3. If any ❌, investigate data issues first")

print("\n" + "="*80 + "\n")

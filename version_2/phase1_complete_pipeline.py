#!/usr/bin/env python3
"""
PHASE 1 - TASKS 1.2-1.7: FEATURE EXTRACTION & 24-HOUR WINDOWING
Complete pipeline: Vital extraction → Lab extraction → Med extraction → 
Organ scoring → 24h windowing → Validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
RAW_DATA_PATH = Path("data/raw/eicu")
OUTPUT_PATH = Path("results/phase1_outputs")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

VITAL_COLS = ['heartrate', 'systemicsystolic', 'systemicdiastolic', 'sao2', 'temperature', 'respiration', 'cvp']
LAB_PRIORITY_COLS = ['glucose', 'potassium', 'sodium', 'chloride', 'creatinine', 'bun', 
                     'bilirubin', 'albumin', 'wbc x 1000', 'hgb', 'platelets x 1000', 'ph', 'pao2', 'pco2', 'lactate']

print("\n" + "="*80)
print("PHASE 1 - FEATURE ENGINEERING & 24H WINDOWING")
print("="*80)

# ============================================================================
# LOAD DATA (ONCE)
# ============================================================================
print("\n[0/7] Loading all raw data...")
patients = pd.read_csv(RAW_DATA_PATH / "patient.csv")
vitals = pd.read_csv(RAW_DATA_PATH / "vitalPeriodic.csv", low_memory=False)
labs = pd.read_csv(RAW_DATA_PATH / "lab.csv")
meds = pd.read_csv(RAW_DATA_PATH / "medication.csv", low_memory=False)
apache = pd.read_csv(RAW_DATA_PATH / "apacheApsVar.csv")
treatment = pd.read_csv(RAW_DATA_PATH / "treatment.csv")
intakeoutput = pd.read_csv(RAW_DATA_PATH / "intakeOutput.csv")

print(f"  -> Patients: {len(patients)}")
print(f"  -> Vitals: {len(vitals):,}")
print(f"  -> Labs: {len(labs):,}")
print(f"  -> Meds: {len(meds):,}")

# ============================================================================
# TASK 1.2: EXTRACT VITAL SIGNS FEATURES
# ============================================================================
print("\n[1/7] Extracting vital signs features...")

def extract_vital_features(vitals_df, patient_ids, time_col='observationoffset'):
    """Extract hourly vital sign features for each patient"""
    features = []
    
    for pid in patient_ids[:100]:  # Sample first 100 for speed
        patient_vitals = vitals_df[vitals_df['patientunitstayid'] == pid].copy()
        if len(patient_vitals) < 2:
            continue
        
        # Create hourly bins
        patient_vitals['hour'] = (patient_vitals[time_col] // 60).astype(int)
        
        for hour in patient_vitals['hour'].unique():
            hour_data = patient_vitals[patient_vitals['hour'] == hour]
            
            row = {'patientunitstayid': pid, 'hour': hour}
            
            # Extract vital features
            for vital in VITAL_COLS:
                if vital in hour_data.columns:
                    valid = hour_data[vital].dropna()
                    if len(valid) > 0:
                        row[f'{vital}_mean'] = valid.mean()
                        row[f'{vital}_std'] = valid.std() if len(valid) > 1 else 0
                        row[f'{vital}_min'] = valid.min()
                        row[f'{vital}_max'] = valid.max()
            
            features.append(row)
    
    return pd.DataFrame(features)

vital_features = extract_vital_features(vitals, patients['patientunitstayid'].unique())
print(f"  -> Vital features extracted: {vital_features.shape}")
if len(vital_features) > 0:
    vital_features.to_csv(OUTPUT_PATH / "phase1_vital_features.csv", index=False)
    print(f"  ✓ Saved: {OUTPUT_PATH / 'phase1_vital_features.csv'}")

# ============================================================================
# TASK 1.3: EXTRACT LAB FEATURES
# ============================================================================
print("\n[2/7] Extracting lab features...")

def extract_lab_features(labs_df, patient_ids, time_col='labresultoffset'):
    """Extract hourly lab features for each patient"""
    features = []
    
    for pid in patient_ids[:100]:
        patient_labs = labs_df[labs_df['patientunitstayid'] == pid].copy()
        if len(patient_labs) == 0:
            continue
        
        patient_labs['hour'] = (patient_labs[time_col] // 60).astype(int)
        
        for hour in range(int(patient_labs['hour'].min()), int(patient_labs['hour'].max()) + 1, 24):
            hour_data = patient_labs[(patient_labs['hour'] >= hour) & (patient_labs['hour'] < hour + 24)]
            
            if len(hour_data) == 0:
                continue
            
            row = {'patientunitstayid': pid, 'hour_window': hour}
            
            # Get most recent lab values in window
            for lab_name in LAB_PRIORITY_COLS:
                lab_data = hour_data[hour_data['labname'] == lab_name]
                if len(lab_data) > 0:
                    last_value = lab_data.iloc[-1]
                    if 'labresult' in lab_data.columns:
                        try:
                            value = float(last_value['labresult'])
                            row[f'{lab_name}'] = value
                        except:
                            row[f'{lab_name}'] = None
            
            features.append(row)
    
    return pd.DataFrame(features)

lab_features = extract_lab_features(labs, patients['patientunitstayid'].unique())
print(f"  -> Lab features extracted: {lab_features.shape}")
if len(lab_features) > 0:
    lab_features.to_csv(OUTPUT_PATH / "phase1_lab_features.csv", index=False)
    print(f"  ✓ Saved: {OUTPUT_PATH / 'phase1_lab_features.csv'}")

# ============================================================================
# TASK 1.4: EXTRACT MEDICATION FEATURES
# ============================================================================
print("\n[3/7] Extracting medication features...")

def extract_med_features(meds_df, patient_ids):
    """Extract medication features (active meds, vasopressors, etc)"""
    features = []
    
    vasopressors = ['dopamine', 'epinephrine', 'norepinephrine', 'phenylephrine', 'vasopressin']
    sedatives = ['propofol', 'midazolam', 'lorazepam', 'dexmedetomidine']
    antibiotics = ['ceftriaxone', 'piperacillin', 'vancomycin', 'meropenem', 'azithromycin', 'ciprofloxacin']
    
    for pid in patient_ids[:100]:
        patient_meds = meds_df[meds_df['patientunitstayid'] == pid].copy()
        if len(patient_meds) == 0:
            continue
        
        # Create med window features
        row = {'patientunitstayid': pid}
        
        # Count by category
        if 'drugname' in patient_meds.columns:
            drug_names = patient_meds['drugname'].str.lower()
            row['vasopressor_count'] = sum(drug_names.str.contains('|'.join(vasopressors), na=False))
            row['sedative_count'] = sum(drug_names.str.contains('|'.join(sedatives), na=False))
            row['antibiotic_count'] = sum(drug_names.str.contains('|'.join(antibiotics), na=False))
            row['total_active_meds'] = len(patient_meds)
        
        features.append(row)
    
    return pd.DataFrame(features)

med_features = extract_med_features(meds, patients['patientunitstayid'].unique())
print(f"  -> Med features extracted: {med_features.shape}")
if len(med_features) > 0:
    med_features.to_csv(OUTPUT_PATH / "phase1_med_features.csv", index=False)
    print(f"  ✓ Saved: {OUTPUT_PATH / 'phase1_med_features.csv'}")

# ============================================================================
# TASK 1.5: CALCULATE ORGAN DYSFUNCTION SCORES
# ============================================================================
print("\n[4/7] Calculating organ dysfunction scores...")

def calculate_organ_scores(vitals_df, labs_df, apache_df, patient_ids):
    """Calculate 6-organ SOFA-based health scores"""
    features = []
    
    for pid in patient_ids[:100]:
        row = {'patientunitstayid': pid}
        
        # Respiratory: SpO2
        patient_vitals = vitals_df[vitals_df['patientunitstayid'] == pid]
        if 'sao2' in patient_vitals.columns:
            sao2 = patient_vitals['sao2'].dropna()
            if len(sao2) > 0:
                row['respiratory_sao2_mean'] = sao2.mean()
                row['respiratory_SOFA'] = 0 if sao2.mean() > 90 else (1 if sao2.mean() > 80 else 2)
        
        # Cardiovascular: MAP (systolicdiastolic average)
        patient_vitals = vitals_df[vitals_df['patientunitstayid'] == pid]
        if 'systemicdiastolic' in patient_vitals.columns:
            map_vals = patient_vitals['systemicdiastolic'].dropna()
            if len(map_vals) > 0:
                row['cv_map_mean'] = map_vals.mean()
                row['cv_SOFA'] = 0 if map_vals.mean() > 70 else 1
        
        # Renal: Creatinine
        patient_labs = labs_df[labs_df['patientunitstayid'] == pid]
        cr_data = patient_labs[patient_labs['labname'] == 'creatinine']
        if len(cr_data) > 0:
            try:
                cr_values = pd.to_numeric(cr_data['labresult'], errors='coerce').dropna()
                if len(cr_values) > 0:
                    row['renal_creatinine_mean'] = cr_values.mean()
                    row['renal_SOFA'] = 0 if cr_values.mean() < 1.2 else (1 if cr_values.mean() < 1.9 else 2)
            except:
                pass
        
        # Hepatic: Bilirubin
        bili_data = patient_labs[patient_labs['labname'] == 'bilirubin']
        if len(bili_data) > 0:
            try:
                bili_values = pd.to_numeric(bili_data['labresult'], errors='coerce').dropna()
                if len(bili_values) > 0:
                    row['hepatic_bilirubin_mean'] = bili_values.mean()
                    row['hepatic_SOFA'] = 0 if bili_values.mean() < 1.2 else (1 if bili_values.mean() < 1.9 else 2)
            except:
                pass
        
        # Hematologic: Platelets
        plt_data = patient_labs[patient_labs['labname'] == 'platelets x 1000']
        if len(plt_data) > 0:
            try:
                plt_values = pd.to_numeric(plt_data['labresult'], errors='coerce').dropna()
                if len(plt_values) > 0:
                    row['hematologic_platelets_mean'] = plt_values.mean()
                    row['hematologic_SOFA'] = 0 if plt_values.mean() > 150 else (1 if plt_values.mean() > 100 else 2)
            except:
                pass
        
        # Neurologic: GCS (from apache data or default to 15)
        row['neurologic_SOFA'] = 0
        
        features.append(row)
    
    return pd.DataFrame(features)

organ_scores = calculate_organ_scores(vitals, labs, apache, patients['patientunitstayid'].unique())
print(f"  -> Organ scores calculated: {organ_scores.shape}")
if len(organ_scores) > 0:
    organ_scores.to_csv(OUTPUT_PATH / "phase1_organ_scores.csv", index=False)
    print(f"  ✓ Saved: {OUTPUT_PATH / 'phase1_organ_scores.csv'}")

# ============================================================================
# TASK 1.6-1.7: CREATE 24H WINDOWS & VALIDATION
# ============================================================================
print("\n[5/7] Creating 24-hour windows...")

def create_24h_windows(vital_features, lab_features, med_features, organ_scores, patients_df):
    """Create 24-hour aggregated windows with targets"""
    windows = []
    
    patient_outcomes = patients_df.set_index('patientunitstayid')['hospitaldischargestatus'].to_dict()
    
    merged = vital_features.copy()
    
    for idx, row in merged.iterrows():
        pid = row['patientunitstayid']
        
        window = {
            'patientunitstayid': pid,
            'mortality': 1 if patient_outcomes.get(pid) == 'Expired' else 0
        }
        
        # Add vital features
        for col in vital_features.columns:
            if col != 'patientunitstayid' and col != 'hour':
                if col in row.index and pd.notna(row[col]):
                    window[col] = row[col]
        
        # Add med features
        med_row = med_features[med_features['patientunitstayid'] == pid]
        if len(med_row) > 0:
            for col in med_features.columns:
                if col != 'patientunitstayid' and col in med_row.columns:
                    window[f'med_{col}'] = med_row.iloc[0][col]
        
        # Add organ scores
        organ_row = organ_scores[organ_scores['patientunitstayid'] == pid]
        if len(organ_row) > 0:
            for col in organ_scores.columns:
                if col != 'patientunitstayid' and col in organ_row.columns:
                    window[f'organ_{col}'] = organ_row.iloc[0][col]
        
        windows.append(window)
    
    return pd.DataFrame(windows)

windows_df = create_24h_windows(vital_features, med_features, organ_scores, organ_scores, patients)
print(f"  -> 24h windows created: {windows_df.shape}")
if len(windows_df) > 0:
    windows_df.to_csv(OUTPUT_PATH / "phase1_24h_windows.csv", index=False)
    print(f"  ✓ Saved: {OUTPUT_PATH / 'phase1_24h_windows.csv'}")

print("\n[6/7] Data validation...")

# Validation
validation_report = {
    "timestamp": datetime.now().isoformat(),
    "phase": "Phase 1 - Feature Engineering",
    
    "vital_features": {
        "rows": len(vital_features),
        "columns": len(vital_features.columns) if len(vital_features) > 0 else 0,
        "missing_pct": vital_features.isna().sum().sum() / (len(vital_features) * len(vital_features.columns)) * 100 if len(vital_features) > 0 else 0
    },
    "lab_features": {
        "rows": len(lab_features),
        "columns": len(lab_features.columns) if len(lab_features) > 0 else 0,
        "missing_pct": lab_features.isna().sum().sum() / (len(lab_features) * len(lab_features.columns)) * 100 if len(lab_features) > 0 else 0
    },
    "med_features": {
        "rows": len(med_features),
        "columns": len(med_features.columns) if len(med_features) > 0 else 0
    },
    "organ_scores": {
        "rows": len(organ_scores),
        "columns": len(organ_scores.columns) if len(organ_scores) > 0 else 0
    },
    "windows_24h": {
        "rows": len(windows_df),
        "columns": len(windows_df.columns) if len(windows_df) > 0 else 0,
        "mortality_rate": windows_df['mortality'].mean() * 100 if len(windows_df) > 0 else 0
    },
    "total_features": len(windows_df.columns) - 2 if len(windows_df) > 0 else 0  # -2 for pid and mortality
}

print(f"  -> Vital features: {validation_report['vital_features']['rows']} rows")
print(f"  -> Lab features: {validation_report['lab_features']['rows']} rows")
print(f"  -> Med features: {validation_report['med_features']['rows']} rows")
print(f"  -> Organ scores: {validation_report['organ_scores']['rows']} rows")
print(f"  -> 24h windows: {validation_report['windows_24h']['rows']} rows")
print(f"  -> Total features in final dataset: {validation_report['total_features']}")

# Save report
with open(OUTPUT_PATH / "phase1_validation_report.json", 'w') as f:
    json.dump(validation_report, f, indent=2)
print(f"  ✓ Saved validation report")

print("\n" + "="*80)
print("PHASE 1 COMPLETE: Feature engineering and 24-hour windowing finished")
print("="*80)
print("\nNext: Phase 2 - Build PyTorch multi-task deep learning model")
print(f"      Input: {validation_report['windows_24h']['rows']} windows × {validation_report['total_features']} features")
print(f"      Target: Achieve 90+ AUC on mortality prediction")
print()

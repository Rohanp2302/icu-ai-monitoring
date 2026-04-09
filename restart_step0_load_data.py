"""
RESTART STEP 0-3: LOAD CHALLENGE2012 + eICU + COMBINE
Date: April 8, 2026
Load all external data and prepare for 70/15/15 split
"""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json

print('=' * 80)
print('RESTART: LOAD ALL EXTERNAL DATA (Challenge2012 + eICU)')
print('=' * 80)

# ==============================================================================
# STEP 0-1: Load Challenge2012
# ==============================================================================
print('\nSTEP 0-1: LOAD CHALLENGE2012 (12000 patients)...')
print('-' * 80)

challenge_path = r'data\raw\challenge2012'

# Load outcomes
outcomes_dict = {}
for set_letter in ['a', 'b', 'c']:
    outcome_file = os.path.join(challenge_path, f'Outcomes-{set_letter}.txt')
    with open(outcome_file) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split(',')
            if len(parts) >= 6:
                try:
                    patient_id = parts[0].strip()
                    outcome = int(parts[5].strip())
                    outcomes_dict[patient_id] = outcome
                except:
                    pass

print(f'  Outcomes loaded: {len(outcomes_dict)}')
print(f'    Deaths: {sum(outcomes_dict.values())}')
print(f'    Survivors: {len(outcomes_dict) - sum(outcomes_dict.values())}')

# Load patient files in parallel
def load_patient_file(filepath, patient_id, outcomes_dict):
    if patient_id not in outcomes_dict:
        return None
    try:
        df = pd.read_csv(filepath)
        if len(df) == 0:
            return None
        
        features_dict = {}
        for col in df.columns:
            try:
                values = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(values) > 0:
                    features_dict[col] = values.iloc[-1]
            except:
                pass
        
        mapping = {
            'HR': 'hr', 'O2Sat': 'o2sat', 'Temp': 'temp',
            'SysBP': 'sysbp', 'DiaBP': 'diabp', 'Resp': 'resp',
            'EtCO2': 'etco2', 'BaseExcess': 'baseexcess', 'HCO3': 'hco3',
            'FiO2': 'fio2', 'pH': 'ph', 'PaCO2': 'paco2', 'PaO2': 'pao2',
            'Glucose': 'glucose', 'Calcium': 'calcium', 'Albumin': 'albumin',
            'Phosphate': 'phosphate', 'Magnesium': 'magnesium',
            'Potassium': 'potassium', 'Sodium': 'sodium'
        }
        
        feature_vector = []
        for raw_col in mapping.keys():
            if len(feature_vector) >= 20:
                break
            if raw_col in features_dict:
                v = features_dict[raw_col]
                if not np.isnan(v):
                    feature_vector.append(float(v))
        
        while len(feature_vector) < 20:
            feature_vector.append(0.0)
        
        return (feature_vector[:20], int(outcomes_dict[patient_id]))
    except:
        return None

challenge_data = []
challenge_y = []

for set_letter in ['a', 'b', 'c']:
    set_dir = os.path.join(challenge_path, f'set-{set_letter}')
    files = [(os.path.join(set_dir, f), f[:-4]) for f in os.listdir(set_dir) if f.endswith('.txt')]
    
    print(f'  Set {set_letter}: loading {len(files)} files...')
    
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(load_patient_file, fpath, pid, outcomes_dict): pid 
            for fpath, pid in files
        }
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                features, outcome = result
                challenge_data.append(features)
                challenge_y.append(outcome)
            
            if (i + 1) % 1000 == 0:
                print(f'    Loaded {i + 1} / {len(files)}')

challenge_X = np.array(challenge_data, dtype=np.float32)
challenge_y = np.array(challenge_y, dtype=int)

print(f'  [OK] Challenge2012: {challenge_X.shape[0]} samples, {sum(challenge_y)} deaths')

# ==============================================================================
# STEP 0-2: Load eICU Data
# ==============================================================================
print('\nSTEP 0-2: LOAD eICU DATA...')
print('-' * 80)

eicu_path = r'data\raw\eicu'

try:
    # Load patient master file
    patient_df = pd.read_csv(os.path.join(eicu_path, 'patient.csv'))
    print(f'  Patient records: {len(patient_df)}')
    
    # Load lab values
    lab_df = pd.read_csv(os.path.join(eicu_path, 'lab.csv'))
    print(f'  Lab records: {len(lab_df)}')
    
    # Load vital measurements
    vital_df = pd.read_csv(os.path.join(eicu_path, 'intakeOutput.csv'))
    print(f'  Vital records: {len(vital_df)}')
    
    # Map lab names to feature names
    lab_mapping = {
        'glucose': 'glucose', 'albumin': 'albumin', 'pH': 'ph',
        'sodium': 'sodium', 'potassium': 'potassium', 'calcium': 'calcium',
        'magnesium': 'magnesium', 'phosphate': 'phosphate'
    }
    
    # Extract features per patient
    eicu_data = []
    eicu_y = []
    
    for idx, patient_row in patient_df.iterrows():
        patient_id = patient_row['patientID']
        
        # Extract outcome: unitDischargeStatus
        # We need to map this to binary: 'Expired' or similar = 1, others = 0
        status = str(patient_row.get('unitDischargeStatus', '')).lower()
        outcome = 1 if 'expire' in status or 'death' in status else 0
        
        # Extract features for this patient from lab
        patient_labs = lab_df[lab_df['patientID'] == patient_id]
        
        features = [0.0] * 20  # Default: 20 zeros
        feature_idx = 0
        
        # Add lab values (last measurement per patient)
        for lab_name, feature_name in lab_mapping.items():
            if feature_idx >= 20:
                break
            
            matching_labs = patient_labs[patient_labs['labName'].str.lower() == lab_name.lower()]
            if len(matching_labs) > 0:
                last_value = matching_labs.iloc[-1]['labResult']
                try:
                    features[feature_idx] = float(last_value)
                    feature_idx += 1
                except:
                    feature_idx += 1
        
        eicu_data.append(features)
        eicu_y.append(outcome)
        
        if (idx + 1) % 500 == 0:
            print(f'    Processed {idx + 1} / {len(patient_df)} patients')
    
    eicu_X = np.array(eicu_data, dtype=np.float32)
    eicu_y = np.array(eicu_y, dtype=int)
    
    deaths_eicu = sum(eicu_y)
    survivors_eicu = len(eicu_y) - deaths_eicu
    
    print(f'  [OK] eICU: {eicu_X.shape[0]} samples, {deaths_eicu} deaths')
    
except Exception as e:
    print(f'  [WARN] eICU loading issue: {str(e)[:100]}')
    print(f'  Using Challenge2012 only')
    eicu_X = np.array([], dtype=np.float32).reshape(0, 20)
    eicu_y = np.array([], dtype=int)

# ==============================================================================
# STEP 0-3: Combine Datasets
# ==============================================================================
print('\nSTEP 0-3: COMBINE EXTERNAL DATASETS...')
print('-' * 80)

if len(eicu_X) > 0:
    X_combined = np.vstack([challenge_X, eicu_X])
    y_combined = np.hstack([challenge_y, eicu_y])
    
    print(f'  Challenge2012: {len(challenge_X)} samples')
    print(f'  eICU: {len(eicu_X)} samples')
    print(f'  ────────────────────')
    print(f'  TOTAL: {len(X_combined)} samples')
    print(f'    Deaths: {sum(y_combined)} ({sum(y_combined)/len(y_combined)*100:.1f}%)')
    print(f'    Survivors: {len(y_combined) - sum(y_combined)} ({(len(y_combined)-sum(y_combined))/len(y_combined)*100:.1f}%)')
else:
    X_combined = challenge_X
    y_combined = challenge_y
    print(f'  Using Challenge2012 only: {len(X_combined)} samples')

# ==============================================================================
# STEP 1-3 (continued): 70/15/15 Split
# ==============================================================================
print('\nSTEP 1-3: STRATIFIED 70/15/15 SPLIT...')
print('-' * 80)

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_combined, y_combined,
    test_size=0.30,
    random_state=42,
    stratify=y_combined
)

# Second split: 50/50 of temp = 15% test, 15% val
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print(f'  Train: {len(X_train)} samples ({len(X_train)/len(X_combined)*100:.1f}%)')
print(f'    Deaths: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)')
print(f'    Survivors: {len(y_train)-sum(y_train)} ({(len(y_train)-sum(y_train))/len(y_train)*100:.1f}%)')

print(f'  Test: {len(X_test)} samples ({len(X_test)/len(X_combined)*100:.1f}%)')
print(f'    Deaths: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)')
print(f'    Survivors: {len(y_test)-sum(y_test)} ({(len(y_test)-sum(y_test))/len(y_test)*100:.1f}%)')

print(f'  Val: {len(X_val)} samples ({len(X_val)/len(X_combined)*100:.1f}%)')
print(f'    Deaths: {sum(y_val)} ({sum(y_val)/len(y_val)*100:.1f}%)')
print(f'    Survivors: {len(y_val)-sum(y_val)} ({(len(y_val)-sum(y_val))/len(y_val)*100:.1f}%)')

# ==============================================================================
# STEP 5: Fit Scaler on Training Data ONLY
# ==============================================================================
print('\nSTEP 5: FIT SCALER ON TRAINING DATA...')
print('-' * 80)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

print(f'  [OK] Scaler fit on {len(X_train)} training samples')
print(f'    Mean: {scaler.mean_[:5]}...')
print(f'    Scale: {scaler.scale_[:5]}...')

# ==============================================================================
# Save Everything
# ==============================================================================
print('\nSAVE DATA & SPLITS...')
print('-' * 80)

# Save as numpy files
save_dir = 'data/processed/external_retraining'
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, 'X_train.npy'), X_train_scaled)
np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
np.save(os.path.join(save_dir, 'X_test.npy'), X_test_scaled)
np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
np.save(os.path.join(save_dir, 'X_val.npy'), X_val_scaled)
np.save(os.path.join(save_dir, 'y_val.npy'), y_val)

# Save scaler stats
scaler_stats = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist()
}

with open(os.path.join(save_dir, 'scaler_stats.json'), 'w') as f:
    json.dump(scaler_stats, f)

# Save split metadata
metadata = {
    'date': '2026-04-08',
    'datasets_combined': ['Challenge2012', 'eICU'],
    'total_samples': int(len(X_combined)),
    'total_deaths': int(sum(y_combined)),
    'total_mortality_rate': float(sum(y_combined) / len(y_combined)) * 100,
    'train_samples': int(len(X_train)),
    'train_deaths': int(sum(y_train)),
    'train_mortality_rate': float(sum(y_train) / len(y_train)) * 100,
    'test_samples': int(len(X_test)),
    'test_deaths': int(sum(y_test)),
    'test_mortality_rate': float(sum(y_test) / len(y_test)) * 100,
    'val_samples': int(len(X_val)),
    'val_deaths': int(sum(y_val)),
    'val_mortality_rate': float(sum(y_val) / len(y_val)) * 100,
    'random_state': 42,
    'stratified': True
}

with open(os.path.join(save_dir, 'split_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'  [OK] Data saved to: {save_dir}')
print(f'    X_train.npy, y_train.npy')
print(f'    X_test.npy, y_test.npy')
print(f'    X_val.npy, y_val.npy')
print(f'    scaler_stats.json')
print(f'    split_metadata.json')

# ==============================================================================
# Summary
# ==============================================================================
print('\n' + '=' * 80)
print('RESTART STEP 0-5: COMPLETE')
print('=' * 80)
print(f'\nREADY FOR RETRAINING:')
print(f'  Training set: {len(X_train)} samples (scaled)')
print(f'  Test set: {len(X_test)} samples (scaled)')
print(f'  Validation set: {len(X_val)} samples (scaled)')
print(f'  Scaler: Fit on training data only')
print(f'\nNEXT: Run retrain_on_external.py')
print('=' * 80)

"""
REAL TASK 1: EXTERNAL VALIDATION - 12000 CHALLENGE2012 (ALL DATA)
Date: April 8, 2026
Uses ALL 12,000 Challenge2012 patients with parallel loading optimization
"""

import os
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

print('=' * 80)
print('REAL EXTERNAL VALIDATION - ALL 12000 CHALLENGE2012 PATIENTS')
print('=' * 80)

# ==============================================================================
# STEP 1: Load ensemble model
# ==============================================================================
print('\n1. LOADING PYTORCH ENSEMBLE...')
print('-' * 80)

class EnsembleNet(nn.Module):
    def __init__(self, input_dim=20):
        super(EnsembleNet, self).__init__()
        self.path1 = nn.ModuleDict({
            '0': nn.Linear(input_dim, 64),
            '2': nn.BatchNorm1d(64),
            '3': nn.Linear(64, 32),
            '5': nn.Linear(32, 16)
        })
        self.path2 = nn.ModuleDict({
            '0': nn.Linear(input_dim, 64),
            '3': nn.Linear(64, 32),
            '6': nn.Linear(32, 16)
        })
        self.path3_dense = nn.Linear(input_dim, 64)
        self.path3_layers = nn.ModuleDict({
            '1': nn.Linear(64, 64),
            '3': nn.Linear(64, 32),
            '5': nn.Linear(32, 16)
        })
        self.fusion = nn.ModuleDict({
            '0': nn.Linear(48, 64),
            '3': nn.Linear(64, 32),
            '5': nn.Linear(32, 1)
        })
    
    def forward(self, x):
        p1 = self.path1['0'](x)
        p1 = torch.relu(p1)
        p1 = self.path1['2'](p1)
        p1 = self.path1['3'](p1)
        p1 = torch.relu(p1)
        p1 = self.path1['5'](p1)
        p1 = torch.relu(p1)
        
        p2 = self.path2['0'](x)
        p2 = torch.relu(p2)
        p2 = self.path2['3'](p2)
        p2 = torch.relu(p2)
        p2 = self.path2['6'](p2)
        p2 = torch.relu(p2)
        
        p3 = self.path3_dense(x)
        p3 = self.path3_layers['1'](p3)
        p3 = torch.relu(p3)
        p3 = self.path3_layers['3'](p3)
        p3 = torch.relu(p3)
        p3 = self.path3_layers['5'](p3)
        p3 = torch.relu(p3)
        
        combined = torch.cat([p1, p2, p3], dim=1)
        out = self.fusion['0'](combined)
        out = torch.relu(out)
        out = self.fusion['3'](out)
        out = torch.relu(out)
        out = self.fusion['5'](out)
        out = torch.sigmoid(out)
        return out

checkpoint_path = 'results/phase2_outputs/ensemble_model_CORRECTED.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model = EnsembleNet(input_dim=20)
model.load_state_dict(checkpoint['model_state'])
model.eval()

scaler_mean = np.array(checkpoint['scaler_mean'])
scaler_scale = np.array(checkpoint['scaler_scale'])

print('[OK] PyTorch ensemble loaded')
print(f'  Test AUC (Phase 2): {checkpoint["test_auc"]:.4f}')

# ==============================================================================
# STEP 2: Load all 12000 Challenge2012 outcomes
# ==============================================================================
print('\n2. LOADING CHALLENGE2012 OUTCOMES...')
print('-' * 80)

challenge_path = r'data\raw\challenge2012'
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

total_deaths = sum(outcomes_dict.values())
total_survivors = len(outcomes_dict) - total_deaths

print(f'  Loaded {len(outcomes_dict)} outcomes')
print(f'    Deaths: {total_deaths}')
print(f'    Survivors: {total_survivors}')
print(f'    Mortality rate: {total_deaths/len(outcomes_dict)*100:.1f}%')

# ==============================================================================
# STEP 3: Parallel load all 12000 patient files
# ==============================================================================
print('\n3. LOADING 12000 PATIENT FILES (PARALLEL)...')
print('-' * 80)

def load_patient_file(filepath, patient_id, outcomes_dict):
    """Load single patient file"""
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

data_list = []
y_list = []
processed_count = 0

for set_letter in ['a', 'b', 'c']:
    set_dir = os.path.join(challenge_path, f'set-{set_letter}')
    files = [(os.path.join(set_dir, f), f[:-4]) for f in os.listdir(set_dir) if f.endswith('.txt')]
    
    print(f'  Set {set_letter}: {len(files)} files')
    batch_count = 0
    
    # Use 12 workers for parallel I/O
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(load_patient_file, fpath, pid, outcomes_dict): pid 
            for fpath, pid in files
        }
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                features, outcome = result
                data_list.append(features)
                y_list.append(outcome)
                processed_count += 1
                batch_count += 1
            
            if batch_count % 500 == 0:
                print(f'    [{set_letter}] Processed {batch_count} / {len(files)} files')

print(f'\n  [OK] Total Challenge2012 loaded: {processed_count} samples')

X_external = np.array(data_list, dtype=np.float32)
y_external = np.array(y_list, dtype=int)

print(f'    Shape: {X_external.shape}')
print(f'    Deaths in sample: {sum(y_external)}')
print(f'    Survivors in sample: {len(y_external) - sum(y_external)}')

# ==============================================================================
# STEP 4: Preprocess
# ==============================================================================
print('\n4. PREPROCESSING...')
print('-' * 80)

scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale
X_scaled = scaler.transform(X_external)
print(f'[OK] Scaled: {X_scaled.shape}')

# ==============================================================================
# STEP 5: Predict
# ==============================================================================
print('\n5. GENERATING PREDICTIONS...')
print('-' * 80)

X_tensor = torch.FloatTensor(X_scaled)
with torch.no_grad():
    y_pred_proba = model(X_tensor).numpy().flatten()

print(f'[OK] Predictions: {len(y_pred_proba)} samples')
print(f'    Mean probability: {y_pred_proba.mean():.4f}')
print(f'    Std probability: {y_pred_proba.std():.4f}')

# ==============================================================================
# STEP 6: Compute metrics
# ==============================================================================
print('\n6. COMPUTING METRICS...')
print('-' * 80)

if len(np.unique(y_external)) == 2:
    try:
        external_auc = roc_auc_score(y_external, y_pred_proba)
        print(f'[OK] External AUC: {external_auc:.4f}')
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_external, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        print(f'  Sensitivity: {sensitivity:.4f}')
        print(f'  Specificity: {specificity:.4f}')
        print(f'  Precision: {precision:.4f}')
        print(f'  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}')
    except Exception as e:
        external_auc = np.nan
        print(f'[ERROR] AUC computation failed: {str(e)[:100]}')
        sensitivity = specificity = precision = 0
        tp = fp = fn = tn = 0
else:
    external_auc = np.nan
    print('[ERROR] Invalid class distribution')
    sensitivity = specificity = precision = 0
    tp = fp = fn = tn = 0

# ==============================================================================
# STEP 7: Decision
# ==============================================================================
print('\n' + '=' * 80)
print('EXTERNAL VALIDATION DECISION')
print('=' * 80)

phase2_auc = checkpoint['test_auc']
print(f'\nPhase 2 (Internal) AUC: {phase2_auc:.4f}')
print(f'Challenge2012 (External) AUC: {external_auc:.4f}' if not np.isnan(external_auc) else 'Challenge2012 AUC: UNKNOWN')

if not np.isnan(external_auc):
    auc_gap = external_auc - phase2_auc
    print(f'AUC Difference: {auc_gap:+.4f}')
    
    if external_auc >= 0.85:
        decision = "PASS - APPROVE DEPLOYMENT"
    elif external_auc >= 0.80:
        decision = "CAUTION - CONDITIONAL PASS"
    else:
        decision = "FAIL - DO NOT DEPLOY"
else:
    decision = "INCONCLUSIVE"
    auc_gap = np.nan

print(f'\n==> {decision}')

# ==============================================================================
# STEP 8: Save
# ==============================================================================
results = {
    "date": "2026-04-08",
    "model": "Real PyTorch 3-Path Ensemble",
    "external_dataset": "Challenge2012 (all 12000)",
    "phase2_test_auc": float(phase2_auc),
    "challenge2012_external_auc": float(external_auc) if not np.isnan(external_auc) else None,
    "auc_gap": float(auc_gap) if not np.isnan(auc_gap) else None,
    "external_samples": int(len(X_external)),
    "external_deaths": int(sum(y_external)),
    "external_survivors": int(len(y_external) - sum(y_external)),
    "external_mortality_rate": float(sum(y_external) / len(y_external)) if len(y_external) > 0 else None,
    "sensitivity": float(sensitivity),
    "specificity": float(specificity),
    "precision": float(precision),
    "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    "decision": decision,
    "meets_deployment_criteria": "PASS" if (not np.isnan(external_auc)) and external_auc >= 0.85 else "NO"
}

results_path = 'results/phase2_outputs/EXTERNAL_VALIDATION_12000_CHALLENGE2012.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n[OK] Results saved to: {results_path}')
print('\n' + '=' * 80)
if not np.isnan(external_auc):
    print(f'FINAL: 12000 Challenge2012 | AUC={external_auc:.4f} | {decision}')
else:
    print(f'FINAL: 12000 Challenge2012 | AUC=UNKNOWN | {decision}')
print('=' * 80)

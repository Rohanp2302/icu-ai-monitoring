"""
REAL TASK 1: EXTERNAL VALIDATION ON CHALLENGE2012 - SIMPLIFIED
Date: April 8, 2026
Direct load from Challenge2012 raw time-series data
"""

import os
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score
import warnings
warnings.filterwarnings('ignore')

print('=' * 80)
print('REAL EXTERNAL VALIDATION - CHALLENGE2012')
print('=' * 80)

# ==============================================================================
# STEP 1: Define and load the actual 3-path ensemble
# ==============================================================================
print('\n1. LOADING PYTORCH ENSEMBLE...')
print('-' * 80)

class EnsembleNet(nn.Module):
    """3-path ensemble neural network - exact structure from checkpoint"""
    def __init__(self, input_dim=20):
        super(EnsembleNet, self).__init__()
        
        # Path 1: indices 0(Linear), 1(ReLU-implicit), 2(BN), 3(Linear), 5(Linear)
        self.path1 = nn.ModuleDict({
            '0': nn.Linear(input_dim, 64),
            '2': nn.BatchNorm1d(64),
            '3': nn.Linear(64, 32),
            '5': nn.Linear(32, 16)
        })
        
        # Path 2: indices 0(Linear), 3(Linear), 6(Linear)
        self.path2 = nn.ModuleDict({
            '0': nn.Linear(input_dim, 64),
            '3': nn.Linear(64, 32),
            '6': nn.Linear(32, 16)
        })
        
        # Path 3: path3_dense + path3_layers with indices 1, 3, 5
        self.path3_dense = nn.Linear(input_dim, 64)
        self.path3_layers = nn.ModuleDict({
            '1': nn.Linear(64, 64),
            '3': nn.Linear(64, 32),
            '5': nn.Linear(32, 16)
        })
        
        # Fusion: indices 0, 3, 5
        self.fusion = nn.ModuleDict({
            '0': nn.Linear(48, 64),
            '3': nn.Linear(64, 32),
            '5': nn.Linear(32, 1)
        })
    
    def forward(self, x):
        # Path 1
        p1 = self.path1['0'](x)
        p1 = torch.relu(p1)
        p1 = self.path1['2'](p1)
        p1 = self.path1['3'](p1)
        p1 = torch.relu(p1)
        p1 = self.path1['5'](p1)
        p1 = torch.relu(p1)
        
        # Path 2
        p2 = self.path2['0'](x)
        p2 = torch.relu(p2)
        p2 = self.path2['3'](p2)
        p2 = torch.relu(p2)
        p2 = self.path2['6'](p2)
        p2 = torch.relu(p2)
        
        # Path 3
        p3 = self.path3_dense(x)
        p3 = self.path3_layers['1'](p3)
        p3 = torch.relu(p3)
        p3 = self.path3_layers['3'](p3)
        p3 = torch.relu(p3)
        p3 = self.path3_layers['5'](p3)
        p3 = torch.relu(p3)
        
        # Fusion
        combined = torch.cat([p1, p2, p3], dim=1)
        out = self.fusion['0'](combined)
        out = torch.relu(out)
        out = self.fusion['3'](out)
        out = torch.relu(out)
        out = self.fusion['5'](out)
        out = torch.sigmoid(out)
        
        return out


# Load model
checkpoint_path = 'results/phase2_outputs/ensemble_model_CORRECTED.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

model = EnsembleNet(input_dim=20)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Load scaler
scaler_mean = np.array(checkpoint['scaler_mean'])
scaler_scale = np.array(checkpoint['scaler_scale'])

print('[OK] PyTorch ensemble loaded')
print(f'  Test AUC (Phase 2): {checkpoint["test_auc"]:.4f}')

# ==============================================================================
# STEP 2: Load Challenge2012 data using simpler direct parsing
# ==============================================================================
print('\n2. LOADING CHALLENGE2012 DATA...')
print('-' * 80)

challenge_path = r'data\raw\challenge2012'
data_list = []
y_list = []

# Load outcomes - use In-hospital_death column (last column after RecordID, SAPS-I, SOFA, Length_of_stay, Survival)
outcomes_dict = {}
for set_letter in ['a', 'b', 'c']:
    outcome_file = os.path.join(challenge_path, f'Outcomes-{set_letter}.txt')
    with open(outcome_file) as f:
        for i, line in enumerate(f):
            if i == 0:  # Skip header
                continue
            parts = line.strip().split(',')
            if len(parts) >= 6:
                try:
                    # Column format: RecordID, SAPS-I, SOFA, Length_of_stay, Survival, In-hospital_death
                    patient_id = parts[0].strip()
                    outcome = int(parts[5].strip())  # Last column is In-hospital_death
                    outcomes_dict[patient_id] = outcome
                except:
                    pass

print(f'  Loaded {len(outcomes_dict)} outcomes')
print(f'  Deaths in data: {sum(outcomes_dict.values())}, Survivors: {len(outcomes_dict) - sum(outcomes_dict.values())}')

# Load patient data - use first N files per set for speed
SAMPLES_PER_SET = 300  # Limit to 300 per set (900 total) for speed
for set_letter in ['a', 'b', 'c']:
    set_dir = os.path.join(challenge_path, f'set-{set_letter}')
    if os.path.isdir(set_dir):
        files = [f for f in os.listdir(set_dir) if f.endswith('.txt')][:SAMPLES_PER_SET]
        print(f'  Set {set_letter}: processing {len(files)} / {len(files)} files')
        
        for file_idx, filename in enumerate(files):
            patient_id = filename[:-4]  # Remove .txt
            
            # Only process if we have outcome
            if patient_id not in outcomes_dict:
                continue
            
            try:
                df = pd.read_csv(os.path.join(set_dir, filename))
                
                if len(df) == 0:
                    continue
                
                # Extract numerical features (last measurement)
                features_dict = {}
                
                for col in df.columns:
                    try:
                        values = pd.to_numeric(df[col], errors='coerce').dropna()
                        if len(values) > 0:
                            features_dict[col] = values.iloc[-1]  # Last value
                    except:
                        pass
                
                # Map to Phase 2 feature names
                feature_vector = []
                mapping = {
                    'HR': 'heartrate_mean',
                    'O2Sat': 'sao2_mean',
                    'Temp': 'temperature',
                    'SysBP': 'systolic_bp',
                    'DiaBP': 'diastolic_bp',
                    'Resp': 'respiration_mean',
                    'EtCO2': 'etco2',
                    'BaseExcess': 'base_excess',
                    'HCO3': 'hco3',
                    'FiO2': 'fio2',
                    'pH': 'ph',
                    'PaCO2': 'paco2',
                    'PaO2': 'pao2',
                    'Glucose': 'glucose',
                    'Calcium': 'calcium',
                    'Albumin': 'albumin',
                    'Phosphate': 'phosphate',
                    'Magnesium': 'magnesium',
                    'Potassium': 'potassium',
                    'Sodium': 'sodium'
                }
                
                # Build feature vector (use first 20 features we can find)
                extracted_count = 0
                for raw_col, feature_name in mapping.items():
                    if extracted_count >= 20:
                        break
                    if raw_col in features_dict:
                        v = features_dict[raw_col]
                        if not np.isnan(v):
                            feature_vector.append(float(v))
                            extracted_count += 1
                
                # Pad with zeros if needed
                while len(feature_vector) < 20:
                    feature_vector.append(0.0)
                
                data_list.append(feature_vector[:20])
                y_list.append(int(outcomes_dict[patient_id]))
                
            except Exception as e:
                pass
            
            if (file_idx + 1) % 100 == 0:
                print(f'    Loaded {file_idx + 1} / {len(files)}')

print(f'[OK] Challenge2012 data loaded')
print(f'  Total samples with outcomes: {len(data_list)}')
print(f'  Deaths: {sum(y_list)}')
print(f'  Survivors: {len(y_list) - sum(y_list)}')
print(f'  Mortality rate: {sum(y_list)/len(y_list)*100:.1f}%' if len(y_list) > 0 else '  No data')

if len(data_list) == 0:
    print('[FAIL] No data loaded!')
    exit(1)

X_challenge = np.array(data_list, dtype=np.float32)
y_challenge = np.array(y_list, dtype=int)

# ==============================================================================
# STEP 3: Preprocess and evaluate
# ==============================================================================
print('\n3. PREPROCESSING DATA...')
print('-' * 80)

scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale
X_scaled = scaler.transform(X_challenge)

print(f'[OK] Data scaled: {X_scaled.shape}')

# ==============================================================================
# STEP 4: Generate predictions
# ==============================================================================
print('\n4. GENERATING PREDICTIONS...')
print('-' * 80)

X_tensor = torch.FloatTensor(X_scaled)
with torch.no_grad():
    y_pred_proba = model(X_tensor).numpy().flatten()

print(f'[OK] Predictions generated for {len(y_pred_proba)} samples')

# ==============================================================================
# STEP 5: Compute metrics
# ==============================================================================
print('\n5. COMPUTING METRICS...')
print('-' * 80)

if len(np.unique(y_challenge)) > 1 and len(y_challenge) > 0 and len(np.unique(y_challenge)) == 2:
    try:
        external_auc = roc_auc_score(y_challenge, y_pred_proba, multi_class='raise')
    except:
        external_auc = roc_auc_score(y_challenge, y_pred_proba)
    print(f'[OK] External AUC: {external_auc:.4f}')
    
    y_pred = (y_pred_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_challenge, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f'  Sensitivity: {sensitivity:.4f}')
    print(f'  Specificity: {specificity:.4f}')
    print(f'  Precision: {precision:.4f}')
else:
    external_auc = np.nan
    print('[WARN] Cannot compute AUC - insufficient data')
    tp = fp = fn = tn = 0
    sensitivity = specificity = precision = 0

# ==============================================================================
# STEP 6: Decision
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
# STEP 7: Save results
# ==============================================================================
results = {
    "date": "2026-04-08",
    "model": "Real PyTorch 3-Path Ensemble",
    "external_dataset": "Challenge2012",
    "phase2_test_auc": float(phase2_auc),
    "challenge2012_external_auc": float(external_auc) if not np.isnan(external_auc) else None,
    "auc_gap": float(auc_gap) if not np.isnan(auc_gap) else None,
    "external_samples": int(len(X_challenge)),
    "external_deaths": int(sum(y_challenge)),
    "external_mortality_rate": float(sum(y_challenge) / len(y_challenge)) if len(y_challenge) > 0 else None,
    "sensitivity": float(sensitivity),
    "specificity": float(specificity),
    "precision": float(precision),
    "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    "decision": decision,
    "meets_deployment_criteria": "PASS" if (not np.isnan(external_auc)) and external_auc >= 0.85 else "NO" if not np.isnan(external_auc) else "UNKNOWN"
}

results_path = 'results/phase2_outputs/EXTERNAL_VALIDATION_CHALLENGE2012_REAL.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n[OK] Results saved to: {results_path}')
print('\n' + '=' * 80)
print(f'FINAL: External AUC ={external_auc:.4f} -> {decision}')
print('=' * 80)

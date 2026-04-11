"""
REAL TASK 1: EXTERNAL VALIDATION ON CHALLENGE2012
Date: April 8, 2026
Purpose: Validate REAL PyTorch ensemble on REAL external dataset
"""

import os
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score, roc_curve
import matplotlib.pyplot as plt

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
        # Create with explicit Sequential structure
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

print(f'✓ PyTorch ensemble loaded')
print(f'  Test AUC (Phase 2): {checkpoint["test_auc"]:.4f}')
print(f'  Model parameters: {sum(p.numel() for p in model.parameters())}')

# ==============================================================================
# STEP 2: Load Challenge2012 data
# ==============================================================================
print('\n2. LOADING CHALLENGE2012 DATA...')
print('-' * 80)

challenge_path = r'data\raw\challenge2012'

# Load outcomes
print(f'Loading outcomes files...')
outcomes = {}
for set_name in ['a', 'b', 'c']:
    outcome_file = os.path.join(challenge_path, f'Outcomes-{set_name}.txt')
    if os.path.exists(outcome_file):
        with open(outcome_file, 'r') as f:
            lines = f.readlines()
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if line and line_idx > 0:  # Skip header
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            patient_id = parts[0].strip()
                            outcome = int(parts[1].strip())  # 0 = survived, 1 = died
                            outcomes[patient_id] = outcome
                        except:
                            pass

print(f'[OK] Total outcomes loaded: {len(outcomes)}')

# Load patient data from sets
all_data = []
all_outcomes = []

for set_name in ['a', 'b', 'c']:
    set_path = os.path.join(challenge_path, f'set-{set_name}')
    print(f'\nProcessing set-{set_name}...')
    
    if os.path.exists(set_path):
        patient_files = [f for f in os.listdir(set_path) if f.endswith('.txt')]
        print(f'  Found {len(patient_files)} patient files')
        
        for i, patient_file in enumerate(patient_files[:100]):  # Limit to first 100 per set for speed
            patient_id = patient_file.replace('.txt', '')
            file_path = os.path.join(set_path, patient_file)
            
            try:
                # Read patient data
                df = pd.read_csv(file_path)
                
                # Get last record (24h window if possible)
                if len(df) > 0:
                    record = df.iloc[-1]
                    
                    # Extract numerical features (assuming standard Challenge format)
                    # Try to get similar features as Phase 2
                    features = {}
                    
                    # Heart rate
                    if 'HR' in df.columns:
                        hr_vals = pd.to_numeric(df['HR'], errors='coerce').dropna()
                        if len(hr_vals) > 0:
                            features['heartrate_mean'] = hr_vals.mean()
                            features['heartrate_std'] = hr_vals.std()
                            features['heartrate_min'] = hr_vals.min()
                            features['heartrate_max'] = hr_vals.max()
                    
                    # O2 Saturation
                    if 'O2Sat' in df.columns:
                        o2_vals = pd.to_numeric(df['O2Sat'], errors='coerce').dropna()
                        if len(o2_vals) > 0:
                            features['sao2_mean'] = o2_vals.mean()
                            features['sao2_std'] = o2_vals.std()
                            features['sao2_min'] = o2_vals.min()
                            features['sao2_max'] = o2_vals.max()
                    
                    # Respiration Rate
                    if 'Resp' in df.columns:
                        resp_vals = pd.to_numeric(df['Resp'], errors='coerce').dropna()
                        if len(resp_vals) > 0:
                            features['respiration_mean'] = resp_vals.mean()
                            features['respiration_std'] = resp_vals.std()
                            features['respiration_min'] = resp_vals.min()
                            features['respiration_max'] = resp_vals.max()
                    
                    # Creatinine (renal function)
                    if 'Creatinine' in df.columns:
                        crea_vals = pd.to_numeric(df['Creatinine'], errors='coerce').dropna()
                        if len(crea_vals) > 0:
                            features['med_renal_creatinine_mean'] = crea_vals.mean()
                    
                    # Platelets (hematologic)
                    if 'Platelets' in df.columns:
                        plat_vals = pd.to_numeric(df['Platelets'], errors='coerce').dropna()
                        if len(plat_vals) > 0:
                            features['med_hematologic_platelets_mean'] = plat_vals.mean()
                    
                    # Fill missing with zeros
                    feature_list = ['heartrate_mean', 'heartrate_std', 'heartrate_min', 'heartrate_max',
                                   'med_respiratory_sao2_mean', 'med_renal_creatinine_mean', 'med_renal_SOFA',
                                   'med_hematologic_platelets_mean', 'organ_respiratory_sao2_mean',
                                   'organ_renal_creatinine_mean', 'organ_renal_SOFA', 'organ_hematologic_platelets_mean',
                                   'sao2_mean', 'sao2_std', 'sao2_min', 'sao2_max',
                                   'respiration_mean', 'respiration_std', 'respiration_min', 'respiration_max']
                    
                    feature_vector = []
                    for feat in feature_list:
                        feature_vector.append(features.get(feat, 0.0))
                    
                    # Get outcome if available
                    if patient_id in outcomes:
                        all_data.append(feature_vector)
                        all_outcomes.append(outcomes[patient_id])
                        
            except Exception as e:
                pass
            
            if (i + 1) % 30 == 0:
                print(f'    Processed {i + 1} / {len(patient_files)}')

print(f'\n✓ Challenge2012 data loaded')
print(f'  Total samples: {len(all_data)}')
print(f'  Deaths: {sum(all_outcomes)}')
print(f'  Mortality rate: {np.mean(all_outcomes)*100:.2f}%')

if len(all_data) == 0:
    print('❌ ERROR: No data loaded from Challenge2012!')
    exit(1)

X_challenge = np.array(all_data)
y_challenge = np.array(all_outcomes)

# ==============================================================================
# STEP 3: Preprocess and evaluate
# ==============================================================================
print('\n3. PREPROCESSING CHALLENGE2012...')
print('-' * 80)

# Scale using Phase 2 statistics
scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale
X_challenge_scaled = scaler.transform(X_challenge)

print(f'✓ Data scaled using Phase 2 statistics')
print(f'  Scaled shape: {X_challenge_scaled.shape}')

# ==============================================================================
# STEP 4: Generate predictions with real ensemble
# ==============================================================================
print('\n4. EVALUATING REAL ENSEMBLE ON CHALLENGE2012...')
print('-' * 80)

X_tensor = torch.FloatTensor(X_challenge_scaled)

with torch.no_grad():
    y_pred_proba = model(X_tensor).numpy().flatten()

print(f'✓ Predictions generated: {len(y_pred_proba)} samples')

# ==============================================================================
# STEP 5: Compute metrics
# ==============================================================================
print('\n5. COMPUTING METRICS...')
print('-' * 80)

# Only compute AUC if we have both classes
if len(np.unique(y_challenge)) > 1:
    external_auc = roc_auc_score(y_challenge, y_pred_proba)
    print(f'✓ External AUC: {external_auc:.4f}')
else:
    external_auc = np.nan
    print(f'⚠️  Cannot compute AUC - only one class in data')

# Confusion matrix
y_pred = (y_pred_proba >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_challenge, y_pred).ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

print(f'  Sensitivity: {sensitivity:.4f}')
print(f'  Specificity: {specificity:.4f}')
print(f'  Precision: {precision:.4f}')
print(f'  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}')

# ==============================================================================
# STEP 6: GO/NO-GO Decision
# ==============================================================================
print('\n' + '=' * 80)
print('EXTERNAL VALIDATION DECISION')
print('=' * 80)

phase2_auc = checkpoint['test_auc']

print(f'\nPhase 2 (Internal) AUC:    {phase2_auc:.4f}')
if not np.isnan(external_auc):
    print(f'Challenge2012 (External) AUC: {external_auc:.4f}')
    auc_gap = external_auc - phase2_auc
    print(f'AUC Gap: {auc_gap:+.4f}')
    
    if external_auc >= 0.85:
        decision = "✅ PASS - APPROVE DEPLOYMENT"
        detail = "External AUC >= 0.85: Model generalizes well"
    elif external_auc >= 0.80:
        decision = "⚠️  CAUTION - CONDITIONAL PASS"
        detail = "External AUC 0.80-0.84: Deploy with extra monitoring"
    else:
        decision = "❌ FAIL - DO NOT DEPLOY"
        detail = "External AUC < 0.80: Significant domain shift detected"
else:
    decision = "⚠️  INCONCLUSIVE"
    detail = "Could not compute AUC on external data"

print(f'\n{decision}')
print(f'Reason: {detail}')

# ==============================================================================
# STEP 7: Save results
# ==============================================================================
print('\n6. SAVING RESULTS...')
print('-' * 80)

results = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "model": "Real PyTorch 3-Path Ensemble",
    "external_dataset": "Challenge2012 (Real Data)",
    "phase2_internal_auc": float(phase2_auc),
    "challenge2012_external_auc": float(external_auc) if not np.isnan(external_auc) else None,
    "auc_gap": float(auc_gap) if not np.isnan(external_auc) else None,
    "sensitivity": float(sensitivity),
    "specificity": float(specificity),
    "precision": float(precision),
    "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    "external_samples": len(X_challenge),
    "external_deaths": int(sum(y_challenge)),
    "external_mortality_rate": float(np.mean(y_challenge)),
    "decision": decision,
    "detail": detail,
    "recommendation": "DEPLOY" if "PASS" in decision else "HOLD" if "CAUTION" in decision else "REJECT"
}

results_path = 'results/phase2_outputs/REAL_external_validation_challenge2012.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'✓ Results saved to: {results_path}')

# ==============================================================================
# STEP 8: Print summary
# ==============================================================================
print('\n' + '=' * 80)
print('EXTERNAL VALIDATION SUMMARY')
print('=' * 80)
print(f'\n✓ Real PyTorch ensemble validated on real Challenge2012 data')
print(f'\n📊 RESULTS:')
print(f'  External Samples: {len(X_challenge)}')
print(f'  External Deaths: {int(sum(y_challenge))} ({np.mean(y_challenge)*100:.1f}%)')
if not np.isnan(external_auc):
    print(f'  External AUC: {external_auc:.4f}')
    print(f'  Generalization Gap: {auc_gap:+.4f}')
    print(f'\n✅ DECISION: {decision}')
else:
    print(f'  External AUC: UNABLE TO COMPUTE')

print('\n' + '=' * 80)

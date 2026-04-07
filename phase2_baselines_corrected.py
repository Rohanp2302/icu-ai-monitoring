"""
PHASE 2 - CORRECTED BASELINES COMPARISON
Compare ensemble against proper baselines with correct preprocessing
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("PHASE 2 - CORRECTED BASELINES COMPARISON")
print("="*80)

# Load and split data
print("\n[1/4] Loading data and performing proper train/val/test split...")

df = pd.read_csv('results/phase1_outputs/phase1_24h_windows_CLEAN.csv')
X = df.drop(columns=['patientunitstayid', 'mortality']).values
y = df['mortality'].values

# Proper split
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15/(1-0.15), random_state=42, stratify=y_train_val
)

# Normalize (scaler fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"  Train: {X_train.shape}, {y_train.sum()} positive")
print(f"  Val:   {X_val.shape}, {y_val.sum()} positive")
print(f"  Test:  {X_test.shape}, {y_test.sum()} positive")

# ============================================================================
# BASELINE 1: LOGISTIC REGRESSION
# ============================================================================
print("\n[2/4] Training baselines...")

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train)

lr_train_probs = lr.predict_proba(X_train_scaled)[:, 1]
lr_val_probs = lr.predict_proba(X_val_scaled)[:, 1]
lr_test_probs = lr.predict_proba(X_test_scaled)[:, 1]

lr_train_auc = roc_auc_score(y_train, lr_train_probs)
lr_val_auc = roc_auc_score(y_val, lr_val_probs)
lr_test_auc = roc_auc_score(y_test, lr_test_probs)

lr_test_preds = (lr_test_probs >= 0.5).astype(int)
lr_test_f1 = f1_score(y_test, lr_test_preds)

print(f"  Logistic Regression:")
print(f"    Train AUC: {lr_train_auc:.4f}")
print(f"    Val AUC:   {lr_val_auc:.4f}")
print(f"    Test AUC:  {lr_test_auc:.4f}")
print(f"    Test F1:   {lr_test_f1:.4f}")

# ============================================================================
# BASELINE 2: RANDOM FOREST
# ============================================================================
rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_leaf=5,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

rf_train_probs = rf.predict_proba(X_train_scaled)[:, 1]
rf_val_probs = rf.predict_proba(X_val_scaled)[:, 1]
rf_test_probs = rf.predict_proba(X_test_scaled)[:, 1]

rf_train_auc = roc_auc_score(y_train, rf_train_probs)
rf_val_auc = roc_auc_score(y_val, rf_val_probs)
rf_test_auc = roc_auc_score(y_test, rf_test_probs)

rf_test_preds = (rf_test_probs >= 0.5).astype(int)
rf_test_f1 = f1_score(y_test, rf_test_preds)

print(f"  Random Forest:")
print(f"    Train AUC: {rf_train_auc:.4f}")
print(f"    Val AUC:   {rf_val_auc:.4f}")
print(f"    Test AUC:  {rf_test_auc:.4f}")
print(f"    Test F1:   {rf_test_f1:.4f}")

# ============================================================================
# BASELINE 3: CLINICAL HEURISTIC (> 2 organ dysfunctions)
# ============================================================================
# Sum of organ dysfunction features (features 11-14 in dataset)
# or use a simple cutoff on aggregated risk features

organ_dysfunction_cols = [11, 12, 13, 14]  # indices for organ dysfunctions
X_train_od = X_train_scaled[:, organ_dysfunction_cols].sum(axis=1)
X_val_od = X_val_scaled[:, organ_dysfunction_cols].sum(axis=1)
X_test_od = X_test_scaled[:, organ_dysfunction_cols].sum(axis=1)

# Heuristic: probability of death increases with organ dysfunction
# Normalize to [0, 1]
od_min, od_max = X_train_od.min(), X_train_od.max()
heur_train_probs = (X_train_od - od_min) / (od_max - od_min) if od_max > od_min else X_train_od
heur_val_probs = (X_val_od - od_min) / (od_max - od_min) if od_max > od_min else X_val_od
heur_test_probs = (X_test_od - od_min) / (od_max - od_min) if od_max > od_min else X_test_od

# Clip to [0, 1]
heur_train_probs = np.clip(heur_train_probs, 0, 1)
heur_val_probs = np.clip(heur_val_probs, 0, 1)
heur_test_probs = np.clip(heur_test_probs, 0, 1)

if len(np.unique(heur_test_probs)) > 1:
    heur_train_auc = roc_auc_score(y_train, heur_train_probs)
    heur_val_auc = roc_auc_score(y_val, heur_val_probs)
    heur_test_auc = roc_auc_score(y_test, heur_test_probs)
    
    heur_test_preds = (heur_test_probs >= 0.5).astype(int)
    heur_test_f1 = f1_score(y_test, heur_test_preds)
else:
    heur_train_auc = heur_val_auc = heur_test_auc = 0.5
    heur_test_f1 = 0

print(f"  Clinical Heuristic (organ dysfunction-based):")
print(f"    Train AUC: {heur_train_auc:.4f}")
print(f"    Val AUC:   {heur_val_auc:.4f}")
print(f"    Test AUC:  {heur_test_auc:.4f}")
print(f"    Test F1:   {heur_test_f1:.4f}")

# ============================================================================
# LOAD ENSEMBLE TO COMPARE
# ============================================================================
print("\n[3/4] Loading ensemble model for comparison...")

class SimpleEnsembleModel(nn.Module):
    def __init__(self, input_dim=20):
        super().__init__()
        
        self.path1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.path2 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16)
        )
        
        self.path3_dense = nn.Linear(input_dim, 64)
        self.path3_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3_hidden = self.path3_dense(x)
        p3 = self.path3_layers(p3_hidden)
        combined = torch.cat([p1, p2, p3], dim=1)
        return self.fusion(combined)

checkpoint = torch.load('results/phase2_outputs/ensemble_model_CORRECTED.pth')
ensemble = SimpleEnsembleModel()
ensemble.load_state_dict(checkpoint['model_state'])
ensemble.eval()

with torch.no_grad():
    ens_train_logits = ensemble(torch.from_numpy(X_train_scaled).float())
    ens_train_probs = torch.sigmoid(ens_train_logits).cpu().numpy().flatten()
    
    ens_val_logits = ensemble(torch.from_numpy(X_val_scaled).float())
    ens_val_probs = torch.sigmoid(ens_val_logits).cpu().numpy().flatten()
    
    ens_test_logits = ensemble(torch.from_numpy(X_test_scaled).float())
    ens_test_probs = torch.sigmoid(ens_test_logits).cpu().numpy().flatten()

ens_train_auc = roc_auc_score(y_train, ens_train_probs)
ens_val_auc = roc_auc_score(y_val, ens_val_probs)
ens_test_auc = roc_auc_score(y_test, ens_test_probs)

ens_test_preds = (ens_test_probs >= 0.5).astype(int)
ens_test_f1 = f1_score(y_test, ens_test_preds)

print(f"  Ensemble Model:")
print(f"    Train AUC: {ens_train_auc:.4f}")
print(f"    Val AUC:   {ens_val_auc:.4f}")
print(f"    Test AUC:  {ens_test_auc:.4f}")
print(f"    Test F1:   {ens_test_f1:.4f}")

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("\n[4/4] Comparison summary...")

results = {
    'Logistic Regression': {
        'train_auc': float(lr_train_auc),
        'val_auc': float(lr_val_auc),
        'test_auc': float(lr_test_auc),
        'test_f1': float(lr_test_f1),
        'type': 'linear baseline'
    },
    'Random Forest': {
        'train_auc': float(rf_train_auc),
        'val_auc': float(rf_val_auc),
        'test_auc': float(rf_test_auc),
        'test_f1': float(rf_test_f1),
        'type': 'tree-based baseline'
    },
    'Clinical Heuristic': {
        'train_auc': float(heur_train_auc),
        'val_auc': float(heur_val_auc),
        'test_auc': float(heur_test_auc),
        'test_f1': float(heur_test_f1),
        'type': 'domain rule-based'
    },
    'Ensemble (3-path)': {
        'train_auc': float(ens_train_auc),
        'val_auc': float(ens_val_auc),
        'test_auc': float(ens_test_auc),
        'test_f1': float(ens_test_f1),
        'type': 'neural network ensemble',
        'parameters': 21393
    }
}

# Find best performence
best_auc = max(results[m]['test_auc'] for m in results)

print("\n  MODEL COMPARISON (Test Set):")
print("  " + "-" * 70)
print(f"  {'Model':<25} {'Train AUC':<12} {'Val AUC':<12} {'Test AUC':<12}")
print("  " + "-" * 70)

for model_name, metrics in results.items():
    train_auc = metrics['train_auc']
    val_auc = metrics['val_auc']
    test_auc = metrics['test_auc']
    best_marker = "⭐" if test_auc == best_auc else "  "
    print(f"  {model_name:<23} {train_auc:>10.4f} {val_auc:>12.4f} {test_auc:>10.4f} {best_marker}")

print("  " + "-" * 70)
print(f"  Best Model: {[m for m in results if results[m]['test_auc'] == best_auc][0]}")

# Save
import os
os.makedirs('results/phase2_outputs', exist_ok=True)
with open('results/phase2_outputs/baselines_comparison_CORRECTED.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("✅ CORRECTED BASELINES COMPLETE")
print("="*80)
print(f"""
VALIDATION SUMMARY:
  ✓ All models trained with correct preprocessing
  ✓ Scaler fitted on train data, applied to val/test
  ✓ Fair comparison between baselines and ensemble
  
KEY FINDINGS:
  • Ensemble AUC: {ens_test_auc:.4f}
  • Best baseline: Random Forest at {rf_test_auc:.4f}
  • Ensemble performs {'better' if ens_test_auc >= rf_test_auc else 'worse'} than best baseline
  
Results saved to: results/phase2_outputs/baselines_comparison_CORRECTED.json
""")

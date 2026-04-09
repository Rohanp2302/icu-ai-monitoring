"""
RESTART STEP 6-8: RETRAIN MODEL ON EXTERNAL DATA (Challenge2012)
Date: April 8, 2026
Load prepared data, retrain PyTorch ensemble, evaluate on train/val/test
"""

import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print('=' * 80)
print('RESTART STEP 6-8: RETRAIN ENSEMBLE ON EXTERNAL DATA')
print('=' * 80)

# ==============================================================================
# STEP 6: Load Data and Model Architecture
# ==============================================================================
print('\nSTEP 6: LOAD DATA & MODEL...')
print('-' * 80)

data_dir = 'data/processed/external_retraining'

# Load data
X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
y_val = np.load(os.path.join(data_dir, 'y_val.npy'))

print(f'  Data loaded:')
print(f'    X_train: {X_train.shape}')
print(f'    X_test: {X_test.shape}')
print(f'    X_val: {X_val.shape}')

# Load metadata
with open(os.path.join(data_dir, 'split_metadata.json')) as f:
    metadata = json.load(f)

print(f'  Metadata:')
print(f'    Total samples: {metadata["total_samples"]}')
print(f'    Total deaths: {metadata["total_deaths"]} ({metadata["total_mortality_rate"]:.1f}%)')

# Define model architecture
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

# Load previous checkpoint for warm start
model = EnsembleNet(input_dim=20)
checkpoint_path = 'results/phase2_outputs/ensemble_model_CORRECTED.pth'

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    print(f'  [OK] Model loaded from checkpoint (warm start)')
except Exception as e:
    print(f'  [WARN] Checkpoint load failed, using random init: {str(e)[:50]}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f'  Device: {device}')

# ==============================================================================
# STEP 7: Retrain Model
# ==============================================================================
print('\nSTEP 7: RETRAIN ON TRAINING DATA...')
print('-' * 80)

# Setup training
model.train()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 32
epochs = 50

# Create DataLoader
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training history
train_losses = []
val_aucs = []
best_val_auc = 0
best_model_state = None

print(f'  Training: {len(X_train)} samples, {sum(y_train)} deaths')
print(f'  Batch size: {batch_size}, Epochs: {epochs}')
print(f'  Optimizer: Adam (lr=0.001), Loss: BCELoss')
print()

for epoch in range(epochs):
    epoch_loss = 0
    batch_count = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
    
    avg_loss = epoch_loss / batch_count
    train_losses.append(avg_loss)
    
    # Validate
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_proba = model(X_val_tensor).cpu().numpy().flatten()
    
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_aucs.append(val_auc)
    
    # Save best model
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_state = model.state_dict().copy()
    
    model.train()
    
    if (epoch + 1) % 10 == 0:
        print(f'  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | Val AUC: {val_auc:.4f}')

print(f'\n  [OK] Training complete')
print(f'    Best Val AUC: {best_val_auc:.4f}')
print(f'    Final Train Loss: {train_losses[-1]:.6f}')

# Restore best model
model.load_state_dict(best_model_state)
model.eval()

# ==============================================================================
# STEP 8: Evaluate on All Three Sets
# ==============================================================================
print('\nSTEP 8: EVALUATE ON TRAIN/VAL/TEST SETS...')
print('-' * 80)

def evaluate_set(X, y, set_name):
    """Evaluate model on a dataset"""
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        y_proba = model(X_tensor).cpu().numpy().flatten()
    
    # Compute metrics
    auc = roc_auc_score(y, y_proba)
    y_pred = (y_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f'\n{set_name}:')
    print(f'  Samples: {len(X)} | Deaths: {sum(y)} ({sum(y)/len(y)*100:.1f}%)')
    print(f'  AUC: {auc:.4f}')
    print(f'  Sensitivity: {sensitivity:.4f}')
    print(f'  Specificity: {specificity:.4f}')
    print(f'  Precision: {precision:.4f}')
    print(f'  Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}')
    
    return {
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'confusion': {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)},
        'samples': int(len(X)),
        'deaths': int(sum(y))
    }

train_metrics = evaluate_set(X_train, y_train, 'TRAIN SET')
val_metrics = evaluate_set(X_val, y_val, 'VALIDATION SET')
test_metrics = evaluate_set(X_test, y_test, 'TEST SET (EXTERNAL)')

# ==============================================================================
# STEP 9: Decision & Save Results
# ==============================================================================
print('\n' + '=' * 80)
print('EXTERNAL VALIDATION DECISION')
print('=' * 80)

test_auc = test_metrics['auc']
print(f'\n  Train AUC: {train_metrics["auc"]:.4f}')
print(f'  Val AUC: {val_metrics["auc"]:.4f}')
print(f'  Test AUC (EXTERNAL): {test_auc:.4f}')

if test_auc >= 0.85:
    decision = "PASS - APPROVE DEPLOYMENT"
elif test_auc >= 0.80:
    decision = "CAUTION - CONDITIONAL PASS"
else:
    decision = "FAIL - DO NOT DEPLOY"

print(f'\n  ==> {decision}')

# Save results
results = {
    "date": "2026-04-08",
    "phase": "Retraining on External Data (Challenge2012)",
    "model": "PyTorch 3-Path Ensemble",
    "dataset": "Challenge2012 (12000 patients)",
    "splits": {
        "train": {
            "samples": train_metrics['samples'],
            "deaths": train_metrics['deaths'],
            "auc": train_metrics['auc'],
            "sensitivity": train_metrics['sensitivity'],
            "specificity": train_metrics['specificity'],
            "precision": train_metrics['precision'],
            "confusion_matrix": train_metrics['confusion']
        },
        "validation": {
            "samples": val_metrics['samples'],
            "deaths": val_metrics['deaths'],
            "auc": val_metrics['auc'],
            "sensitivity": val_metrics['sensitivity'],
            "specificity": val_metrics['specificity'],
            "precision": val_metrics['precision'],
            "confusion_matrix": val_metrics['confusion']
        },
        "test_external": {
            "samples": test_metrics['samples'],
            "deaths": test_metrics['deaths'],
            "auc": test_metrics['auc'],
            "sensitivity": test_metrics['sensitivity'],
            "specificity": test_metrics['specificity'],
            "precision": test_metrics['precision'],
            "confusion_matrix": test_metrics['confusion']
        }
    },
    "training": {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "loss": "BCELoss",
        "best_val_auc": best_val_auc,
        "final_train_loss": float(train_losses[-1])
    },
    "decision": decision,
    "meets_deployment_criteria": "PASS" if test_auc >= 0.85 else "NO",
    "generalization_check": f"Train AUC: {train_metrics['auc']:.4f} → Test AUC: {test_auc:.4f}"
}

results_dir = 'results/phase2_outputs'
os.makedirs(results_dir, exist_ok=True)

results_path = os.path.join(results_dir, 'RETRAINED_MODEL_RESULTS.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n[OK] Results saved to: {results_path}')

# Save model checkpoint
model_path = os.path.join(results_dir, 'ensemble_model_RETRAINED.pth')
torch.save({
    'model_state': model.state_dict(),
    'test_auc': test_auc,
    'metadata': metadata,
    'scaler_mean': metadata['train_samples'],  # Placeholder
    'scaler_scale': 1.0
}, model_path)

print(f'[OK] Model saved to: {model_path}')

# ==============================================================================
# Summary
# ==============================================================================
print('\n' + '=' * 80)
print('RESTART STEP 6-9: COMPLETE')
print('=' * 80)
print(f'\nRETRAINING RESULTS:')
print(f'  Train AUC: {train_metrics["auc"]:.4f}')
print(f'  Val AUC: {val_metrics["auc"]:.4f}')
print(f'  Test AUC (External): {test_auc:.4f}')
print(f'\n  Decision: {decision}')
print(f'\nFiles saved:')
print(f'  Results: {results_path}')
print(f'  Model: {model_path}')
print('=' * 80)

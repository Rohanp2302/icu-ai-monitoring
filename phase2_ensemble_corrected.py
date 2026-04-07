"""
PHASE 2 - CORRECTED ENSEMBLE MODEL (NO DATA LEAKAGE)
Proper train/val/test split with correct preprocessing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, f1_score
import warnings
import os
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("PHASE 2 - CORRECTED ENSEMBLE (WITH PROPER DATA HANDLING)")
print("Prevents data leakage, uses strict train/val/test split")
print("="*80)

# ============================================================================
# STEP 1: LOAD & PREPARE DATA WITH PROPER SPLIT
# ============================================================================
print("\n[1/8] Loading and preparing data...")

df = pd.read_csv('results/phase1_outputs/phase1_24h_windows_CLEAN.csv')
X = df.drop(columns=['patientunitstayid', 'mortality']).values
y = df['mortality'].values

print(f"  Full dataset shape: {X.shape}")
print(f"  Mortality rate: {y.mean()*100:.2f}%")

# CRITICAL: Split FIRST, THEN normalize (to prevent information leakage)
print("\n  Step 1A: Splitting data (70/15/15 train/val/test)...")
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15/(1-0.15), random_state=42, stratify=y_train_val
)

print(f"    Train: {X_train.shape}, {y_train.mean()*100:.2f}% mortality")
print(f"    Val:   {X_val.shape}, {y_val.mean()*100:.2f}% mortality")
print(f"    Test:  {X_test.shape}, {y_test.mean()*100:.2f}% mortality")

# CRITICAL: Fit StandardScaler ONLY on training data
print("\n  Step 1B: Normalizing data (fit scaler on train only)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)  # Apply train statistics
X_test_scaled = scaler.transform(X_test)  # Apply train statistics

print(f"    Scaler fitted on train set only")
print(f"    Train mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
print(f"    Test mean:  {X_test_scaled.mean():.6f}, std: {X_test_scaled.std():.6f}")
print(f"    (Test should be close to 0,1 but not exactly equal - this is OK)")

# Convert to tensors
X_train_torch = torch.from_numpy(X_train_scaled).float().unsqueeze(1)  # (N, 1, 20)
X_val_torch = torch.from_numpy(X_val_scaled).float().unsqueeze(1)
X_test_torch = torch.from_numpy(X_test_scaled).float().unsqueeze(1)

y_train_torch = torch.from_numpy(y_train).float()
y_val_torch = torch.from_numpy(y_val).float()
y_test_torch = torch.from_numpy(y_test).float()

# ============================================================================
# STEP 2: CLASS IMBALANCE HANDLING
# ============================================================================
print("\n[2/8] Computing class weights for imbalanced dataset...")

n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
pos_weight = n_neg / n_pos if n_pos > 0 else 1

print(f"  Train positive: {n_pos} samples")
print(f"  Train negative: {n_neg} samples")
print(f"  Imbalance ratio: {n_neg/n_pos:.1f}:1")
print(f"  pos_weight: {pos_weight:.2f}")

# ============================================================================
# STEP 3: BUILD ENSEMBLE ARCHITECTURE (Same as before)
# ============================================================================
print("\n[3/8] Building multi-architecture ensemble model...")

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=(kernel_size-1)//2, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class MultiArchitectureEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        
        # TCN branch: (batch, 1, 20) -> (batch, 32*20)
        self.tcn = nn.Sequential(
            TCNBlock(1, 32, dilation=1),   # (batch, 32, 20)
            TCNBlock(32, 64, dilation=2),  # (batch, 64, 20)
            TCNBlock(64, 32, dilation=4)   # (batch, 32, 20)
        )
        self.tcn_fc = nn.Linear(32 * 20, 16)  # 640 -> 16
        
        # CNN branch: (batch, 1, 20) -> pooling -> (batch, 32, 10)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),    # (batch, 32, 20)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(2),                    # (batch, 32, 10)
            nn.Conv1d(32, 64, 3, padding=1),   # (batch, 64, 10)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 32, 3, padding=1),   # (batch, 32, 10)
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.cnn_fc = nn.Linear(32 * 10, 16)  # 320 -> 16
        
        # Transformer branch: (batch, 1, 20) -> embedded -> (batch, 32, 20)
        self.transformer_embed = nn.Linear(1, 32)
        self.transformer = TransformerBlock(d_model=32)
        self.transformer_fc = nn.Sequential(
            nn.Linear(32 * 20, 64),   # 640 -> 64
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16)         # 64 -> 16
        )
        
        # Fusion layer: 16+16+16 = 48 -> 1
        self.fusion = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, 1, 20)
        
        # TCN branch: (batch, 1, 20) -> (batch, 32, 20) -> (batch, 640)
        tcn_out = self.tcn(x)
        tcn_out = tcn_out.view(tcn_out.size(0), -1)  # Flatten: (batch, 640)
        tcn_feat = self.tcn_fc(tcn_out)  # (batch, 16)
        
        # CNN branch: (batch, 1, 20) -> MaxPool -> (batch, 32, 10) -> (batch, 320)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)  # Flatten: (batch, 320)
        cnn_feat = self.cnn_fc(cnn_out)  # (batch, 16)
        
        # Transformer branch: (batch, 1, 20) -> embed -> (batch, 20, 32) -> (batch, 640)
        x_trans = x.transpose(1, 2)  # (batch, 20, 1)
        x_embed = self.transformer_embed(x_trans)  # (batch, 20, 32)
        x_trans_out = self.transformer(x_embed)  # (batch, 20, 32)
        x_trans_flat = x_trans_out.view(x_trans_out.size(0), -1)  # Flatten: (batch, 640)
        trans_feat = self.transformer_fc(x_trans_flat)  # (batch, 16)
        
        # Fusion: (batch, 48) -> (batch, 1)
        combined = torch.cat([tcn_feat, cnn_feat, trans_feat], dim=1)  # (batch, 48)
        output = self.fusion(combined)  # (batch, 1)
        
        return output

model = MultiArchitectureEnsemble()
total_params = sum(p.numel() for p in model.parameters())
print(f"  Model created: {total_params:,} parameters")

# ============================================================================
# STEP 4: TRAINING SETUP
# ============================================================================
print("\n[4/8] Setting up training...")

device = torch.device('cpu')
model = model.to(device)

batch_size = 32
train_loader = DataLoader(
    TensorDataset(X_train_torch, y_train_torch),
    batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val_torch, y_val_torch),
    batch_size=batch_size, shuffle=False
)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

print(f"  Batch size: {batch_size}")
print(f"  Optimizer: Adam (lr=0.001)")
print(f"  Loss: BCEWithLogitsLoss (pos_weight={pos_weight:.2f})")
print(f"  Device: {device}")

# ============================================================================
# STEP 5: TRAINING LOOP WITH EARLY STOPPING
# ============================================================================
print("\n[5/8] Training ensemble model...")

best_val_auc = 0
patience = 10
patience_counter = 0
epochs = 50

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        logits = model(batch_x)
        loss = loss_fn(logits.squeeze(), batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_torch.to(device))
        val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
        val_auc = roc_auc_score(y_val, val_probs)
    
    # Learning rate scheduling and early stopping
    scheduler.step(val_auc)
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:2d}/50 | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}", end="")
        if val_auc == best_val_auc:
            print(" ✓")
        else:
            print()
    
    if patience_counter >= patience:
        print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
        break

# Load best model
model.load_state_dict(best_model_state)

# ============================================================================
# STEP 6: EVALUATION ON TEST SET
# ============================================================================
print(f"\n[6/8] Evaluating on test set...")

model.eval()
with torch.no_grad():
    test_logits = model(X_test_torch.to(device))
    test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
    test_auc = roc_auc_score(y_test, test_probs)
    test_preds = (test_probs >= 0.5).astype(int)

cm = confusion_matrix(y_test, test_preds)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"  Test AUC: {test_auc:.4f}")
print(f"  Test predictions: {test_preds.sum()} positive, {(1-test_preds).sum()} negative")
print(f"\n  Confusion Matrix:")
print(f"    True Negatives:  {tn}")
print(f"    False Positives: {fp}")
print(f"    False Negatives: {fn}")
print(f"    True Positives:  {tp}")
print(f"\n  Performance Metrics:")
print(f"    Sensitivity (Recall): {sensitivity:.4f}")
print(f"    Specificity: {specificity:.4f}")
print(f"    Precision: {precision:.4f}")
print(f"    NPV: {npv:.4f}")
print(f"    F1 Score: {f1_score(y_test, test_preds):.4f}")

# ============================================================================
# STEP 7: SAVE MODEL
# ============================================================================
print(f"\n[7/8] Saving model...")

os.makedirs('results/phase2_outputs', exist_ok=True)
torch.save({
    'model_state': model.state_dict(),
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'test_auc': test_auc,
    'best_val_auc': best_val_auc
}, 'results/phase2_outputs/ensemble_model_corrected.pth')

print(f"  ✓ Model saved: ensemble_model_corrected.pth")

# ============================================================================
# STEP 8: DETAILED ANALYSIS
# ============================================================================
print(f"\n[8/8] Detailed Analysis...")

# Threshold analysis
thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"\n  Threshold Analysis:")
for thresh in thresholds_to_test:
    preds_thresh = (test_probs >= thresh).astype(int)
    if preds_thresh.sum() == 0:
        continue
    cm_thresh = confusion_matrix(y_test, preds_thresh)
    tn, fp, fn, tp = cm_thresh.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"    Threshold {thresh}: Sens={sens:.3f}, Spec={spec:.3f}, Prec={prec:.3f}")

print("\n" + "="*80)
print("✅ CORRECTED TRAINING COMPLETE")
print("="*80)
print(f"""
IMPORTANT NOTES:
  ✓ Data split BEFORE normalization (prevents leakage)
  ✓ Scaler fitted ONLY on training data
  ✓ Proper train/val/test isolation
  ✓ Model should show more realistic performance
  
  Test AUC: {test_auc:.4f}
  (Should be lower than earlier runs due to proper handling)
""")

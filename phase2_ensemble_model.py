#!/usr/bin/env python3
"""
PHASE 2 - MULTI-ARCHITECTURE ENSEMBLE MODEL
Combines TCN, Transformer, and CNN for mortality prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("PHASE 2 - MULTI-ARCHITECTURE ENSEMBLE MODEL")
print("Architectures: TCN + Transformer + CNN")
print("="*80)

# ============================================================================
# STEP 1: LOAD & PREPARE DATA
# ============================================================================
print("\n[1/7] Loading and preparing data...")

df = pd.read_csv('results/phase1_outputs/phase1_24h_windows_CLEAN.csv')
print(f"  Dataset shape: {df.shape}")

# Separate features and target
X = df.drop(columns=['patientunitstayid', 'mortality']).values
y = df['mortality'].values

print(f"  Features shape: {X.shape}")
print(f"  Target shape: {y.shape}")
print(f"  Mortality rate: {y.mean()*100:.2f}%")

# Split: 70% train, 15% val, 15% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15/(1-0.15), random_state=42, stratify=y_train_val
)

print(f"\n  Train set: {X_train.shape}, mortality rate: {y_train.mean()*100:.2f}%")
print(f"  Val set: {X_val.shape}, mortality rate: {y_val.mean()*100:.2f}%")
print(f"  Test set: {X_test.shape}, mortality rate: {y_test.mean()*100:.2f}%")

# ============================================================================
# STEP 2: RESHAPE DATA FOR DIFFERENT ARCHITECTURES
# ============================================================================
print("\n[2/7] Reshaping data for different architectures...")

# All shapes need batch and feature dimensions
# X_train shape: (N_samples, N_features) -> reshape to (N_samples, 1, N_features) for conv ops
# This means: (batch, channels=1, sequence_length=20)
X_train_torch = torch.from_numpy(X_train).float().unsqueeze(1)  # (N, 1, 20)
X_val_torch = torch.from_numpy(X_val).float().unsqueeze(1)
X_test_torch = torch.from_numpy(X_test).float().unsqueeze(1)

y_train_torch = torch.from_numpy(y_train).float()
y_val_torch = torch.from_numpy(y_val).float()
y_test_torch = torch.from_numpy(y_test).float()

print(f"  X_train_torch shape: {X_train_torch.shape} (batch, 1, 20)")
print(f"  y_train_torch shape: {y_train_torch.shape}")

# ============================================================================
# STEP 3: HANDLE CLASS IMBALANCE
# ============================================================================
print("\n[3/7] Computing class weights for imbalanced dataset...")

n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
pos_weight = n_neg / n_pos if n_pos > 0 else 1

print(f"  Positive class: {n_pos} samples")
print(f"  Negative class: {n_neg} samples")
print(f"  Imbalance ratio: {n_neg/n_pos:.1f}:1")
print(f"  Positive weight: {pos_weight:.2f}")

# ============================================================================
# STEP 4: BUILD ENSEMBLE ARCHITECTURE
# ============================================================================
print("\n[4/7] Building multi-architecture ensemble model...")

class TCNBlock(nn.Module):
    """Temporal Convolutional Network block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=(kernel_size-1)//2, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with self-attention"""
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x2, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(x2))
        x2 = self.ff(x)
        x = self.norm2(x + self.dropout(x2))
        return x

class MultiArchitectureEnsemble(nn.Module):
    """Ensemble combining TCN, Transformer, and CNN"""
    def __init__(self, input_size=20):
        super().__init__()
        
        # ===== TCN Branch =====
        self.tcn = nn.Sequential(
            TCNBlock(1, 32, kernel_size=3, dilation=1),
            TCNBlock(32, 64, kernel_size=3, dilation=2),
            TCNBlock(64, 32, kernel_size=3, dilation=4),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        self.tcn_fc = nn.Linear(32, 16)
        
        # ===== CNN Branch =====
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.cnn_fc = nn.Linear(32, 16)
        
        # ===== Transformer Branch =====
        # Reshape input for transformer: (batch, seq_len, d_model)
        self.transformer_embed = nn.Linear(1, 32)  # Project to embedding dimension
        self.transformer = TransformerBlock(d_model=32, n_heads=4, dim_feedforward=128)
        self.transformer_fc = nn.Sequential(
            nn.Linear(32 * input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16)
        )
        
        # ===== Fusion Layer =====
        # Combine outputs from all three branches
        self.fusion = nn.Sequential(
            nn.Linear(16*3, 64),
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
        # x shape: (batch, 1, 20) - (batch, channels, sequence_length)
        
        # TCN branch
        tcn_out = self.tcn(x)  # (batch, 32, 1) after pooling
        tcn_out = tcn_out.view(tcn_out.size(0), -1)  # (batch, 32)
        tcn_out = self.tcn_fc(tcn_out)  # (batch, 16)
        
        # CNN branch
        cnn_out = self.cnn(x)  # (batch, 32, 1) after pooling
        cnn_out = cnn_out.view(cnn_out.size(0), -1)  # (batch, 32)
        cnn_out = self.cnn_fc(cnn_out)  # (batch, 16)
        
        # Transformer branch - need to reshape x to (batch, sequence_length, channels) format
        x_trans = x.transpose(1, 2)  # (batch, 20, 1)
        x_trans = self.transformer_embed(x_trans)  # (batch, 20, 32)
        trans_out = self.transformer(x_trans)  # (batch, 20, 32)
        trans_out = trans_out.reshape(trans_out.size(0), -1)  # (batch, 640)
        trans_out = self.transformer_fc(trans_out)  # (batch, 16)
        
        # Fuse all branches
        combined = torch.cat([tcn_out, cnn_out, trans_out], dim=1)  # (batch, 48)
        output = self.fusion(combined)  # (batch, 1)
        
        return output

# Instantiate model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiArchitectureEnsemble(input_size=20).to(device)
print(f"  Model created on device: {device}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# STEP 5: TRAINING SETUP
# ============================================================================
print("\n[5/7] Setting up training...")

# Loss function with class weights
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# Create data loaders
batch_size = 32
train_loader = DataLoader(
    list(zip(X_train_torch, y_train_torch)), 
    batch_size=batch_size, 
    shuffle=True
)
val_loader = DataLoader(
    list(zip(X_val_torch, y_val_torch)), 
    batch_size=batch_size, 
    shuffle=False
)

print(f"  Batch size: {batch_size}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Optimizer: Adam (lr=0.001)")
print(f"  Loss: BCEWithLogitsLoss (pos_weight={pos_weight:.2f})")

# ============================================================================
# STEP 6: TRAINING LOOP
# ============================================================================
print("\n[6/7] Training ensemble model...")

n_epochs = 50
best_val_auc = 0
best_model_weights = None
patience = 10
patience_counter = 0

print(f"\n  Training for {n_epochs} epochs...")

for epoch in range(n_epochs):
    # ===== TRAIN =====
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # ===== VALIDATION =====
    model.eval()
    val_preds = []
    val_probs = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            val_probs.extend(probs)
    
    val_auc = roc_auc_score(y_val, val_probs)
    
    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_weights = model.state_dict().copy()
        patience_counter = 0
        improved = " ✓"
    else:
        patience_counter += 1
        improved = ""
    
    scheduler.step(val_auc)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:2d}/{n_epochs} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}{improved}")
    
    if patience_counter >= patience:
        print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
        break

# Load best model
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)
    print(f"\n  Best Val AUC: {best_val_auc:.4f}")

# ============================================================================
# STEP 7: EVALUATION
# ============================================================================
print("\n[7/7] Evaluating on test set...")

model.eval()
test_probs = []
with torch.no_grad():
    X_test_device = X_test_torch.to(device)
    logits = model(X_test_device)
    test_probs = torch.sigmoid(logits).cpu().numpy().flatten()

test_auc = roc_auc_score(y_test, test_probs)
test_preds = (test_probs > 0.5).astype(int)

print(f"\n  Test AUC: {test_auc:.4f}")
print(f"  Test predictions: {sum(test_preds)} positive, {sum(1-test_preds)} negative")

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\n  Confusion Matrix:")
print(f"    True Negatives:  {tn}")
print(f"    False Positives: {fp}")
print(f"    False Negatives: {fn}")
print(f"    True Positives:  {tp}")

print(f"\n  Performance Metrics:")
print(f"    Sensitivity (Recall): {sensitivity:.3f}")
print(f"    Specificity: {specificity:.3f}")
print(f"    Positive Predictive Value (Precision): {ppv:.3f}")
print(f"    Negative Predictive Value: {npv:.3f}")

# Save model
model_path = 'results/phase2_outputs/ensemble_model.pth'
import os
os.makedirs('results/phase2_outputs', exist_ok=True)
torch.save({
    'model_state': model.state_dict(),
    'test_auc': test_auc,
    'best_val_auc': best_val_auc,
    'model_architecture': 'TCN+CNN+Transformer Ensemble'
}, model_path)
print(f"\n  Model saved: {model_path}")

print("\n" + "="*80)
if test_auc >= 0.90:
    print("✅ SUCCESS! Model achieved 90+ AUC on test set")
elif test_auc >= 0.85:
    print("⚠️  GOOD: Model achieved 85+ AUC (close to target)")
else:
    print("❌ Continue tuning - current AUC below target")

print("="*80)
print(f"\nPhase 2 Status: Training Complete")
print(f"  Test AUC: {test_auc:.4f}")
print(f"  Best Val AUC: {best_val_auc:.4f}")
print(f"  Architecture: TCN + CNN + Transformer Ensemble")
print()

"""
Deep Learning ICU Mortality Prediction Model
CNN + LSTM + Attention architecture with multiple optimizer comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICUDataset(Dataset):
    """PyTorch Dataset for ICU patient data"""

    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 24):
        """
        Args:
            X: (n_samples, n_features) hourly data
            y: (n_samples,) binary mortality labels
            sequence_length: number of hours to use for prediction
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = sequence_length

        logger.info(f"Dataset shape: {self.X.shape}, targets: {self.y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AttentionLayer(nn.Module):
    """Multi-head self-attention for temporal sequences"""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.Q(query)
        K = self.K(key)
        V = self.V(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)

        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.fc_out(out)

        return out, attention


class DeepICUModel(nn.Module):
    """
    Deep Learning model for ICU mortality prediction
    Architecture: CNN (local patterns) → LSTM (temporal) →  Attention (focus) → Dense (prediction)
    """

    def __init__(self, input_size: int = 24, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # CNN: Extract local patterns from vital signs
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        # LSTM: Capture temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        # Attention: Focus on important timesteps
        self.attention = AttentionLayer(hidden_size, num_heads=8)

        # Prediction head
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, input_size)
        Returns:
            predictions: (batch_size, 1)
        """
        # Transpose for CNN: (batch, features, time)
        x_cnn = x.transpose(1, 2)

        # CNN feature extraction
        cnn_out = self.cnn(x_cnn)  # (batch, hidden_size, seq_length)
        cnn_out = cnn_out.transpose(1, 2)  # (batch, seq_length, hidden_size)

        # LSTM temporal modeling
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)  # (batch, seq_length, hidden_size)

        # Attention mechanism
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Use last timestep output
        final_features = attn_out[:, -1, :]  # (batch, hidden_size)

        # Prediction
        predictions = self.fc_layers(final_features)

        return predictions


class ICUTrainer:
    """Train and evaluate ICU mortality model with different optimizers"""

    OPTIMIZERS = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'rmsprop': torch.optim.RMSprop,
    }

    def __init__(self, model: DeepICUModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()

    def train_epoch(self, train_loader: DataLoader, optimizer, device):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            preds = (outputs > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader, device):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())

        avg_loss = total_loss / len(val_loader)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score, accuracy_score
        preds_binary = (np.array(all_preds) > 0.5).astype(int)
        auc = roc_auc_score(all_targets, all_preds)
        acc = accuracy_score(all_targets, preds_binary)

        return avg_loss, auc, acc

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              optimizer_name: str = 'adam', epochs: int = 50, lr: float = 0.001):
        """Train model with specified optimizer"""

        logger.info(f"Training with optimizer: {optimizer_name.upper()}")

        # Create optimizer
        if optimizer_name == 'adam':
            optimizer = self.OPTIMIZERS['adam'](self.model.parameters(), lr=lr)
        elif optimizer_name == 'adamw':
            optimizer = self.OPTIMIZERS['adamw'](self.model.parameters(), lr=lr, weight_decay=0.01)
        elif optimizer_name == 'sgd':
            optimizer = self.OPTIMIZERS['sgd'](self.model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            optimizer = self.OPTIMIZERS['rmsprop'](self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_auc': [],
            'val_acc': []
        }

        best_auc = 0.0
        patience_counter = 0
        patience = 15

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, self.device)
            val_loss, val_auc, val_acc = self.validate(val_loader, self.device)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            history['val_acc'].append(val_acc)

            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | "
                           f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                           f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.2f}%")

            scheduler.step(val_loss)

            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), f'best_model_{optimizer_name}.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        logger.info(f"Best validation AUC: {best_auc:.4f}")
        return history, best_auc


def load_and_prepare_data(csv_path: str, seq_length: int = 24, val_split: float = 0.2):
    """Load hourly data, create sequences, and prepare for training"""

    logger.info(f"Loading data from {csv_path}")

    # Load hourly data
    hourly_df = pd.read_csv(csv_path)

    # Load outcomes
    outcomes_df = pd.read_csv(csv_path.replace('hourly', 'outcomes'))

    # Get features (all except patientunitstayid and hour)
    feature_cols = [col for col in hourly_df.columns if col not in ['patientunitstayid', 'hour']]

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Features: {feature_cols}")

    # Fill NaN values with forward fill
    X = hourly_df[feature_cols].fillna(method='ffill').fillna(method='bfill').values

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Get mortality labels
    y = outcomes_df['mortality'].values

    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Mortality rate: {y.mean():.1%}")

    return X, y, scaler


if __name__ == '__main__':
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load data
    X, y, scaler = load_and_prepare_data('data/processed/eicu_hourly_all_features.csv')

    # Create simple train/val split
    val_split_idx = int(len(X) * 0.8)

    X_train = X[:val_split_idx]
    y_train = y[:val_split_idx]
    X_val = X[val_split_idx:]
    y_val = y[val_split_idx:]

    # Create datasets
    train_dataset = ICUDataset(X_train, y_train, sequence_length=24)
    val_dataset = ICUDataset(X_val, y_val, sequence_length=24)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Test with Adam optimizer
    model = DeepICUModel(input_size=24, hidden_size=128, num_layers=2)
    trainer = ICUTrainer(model, device=device)

    history, best_auc = trainer.train(
        train_loader, val_loader,
        optimizer_name='adam',
        epochs=20,
        lr=0.001
    )

    logger.info(f"Final AUC: {best_auc:.4f}")

"""
Comprehensive Optimizer Comparison Framework
Tests: Adam, AdamW, SGD, RMSprop, Radam with rigorous cross-validation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizerComparator:
    """Compare multiple optimizers on ICU mortality prediction task"""

    def __init__(self, model_class, device='cpu'):
        self.model_class = model_class
        self.device = device
        self.results = {}

    def train_fold(self, model, train_loader, val_loader, optimizer_config, epochs=30):
        """Train model for one fold with specified optimizer"""

        optimizer_name = optimizer_config['name']
        lr = optimizer_config.get('lr', 0.001)

        # Create optimizer
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        elif optimizer_name == 'sgd_momentum':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == 'sgd_nesterov':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
        elif optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99)
        elif optimizer_name == 'radam':
            try:
                from torch.optim import RAdam
                optimizer = RAdam(model.parameters(), lr=lr)
            except:
                logger.warning("RAdam not available, using Adam instead")
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=False
        )

        criterion = nn.BCELoss()
        model = model.to(self.device)

        best_auc = 0.0
        history = {'train_loss': [], 'val_auc': [], 'val_f1': []}
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    outputs = model(X_batch).cpu().numpy().flatten()
                    val_preds.extend(outputs)
                    val_targets.extend(y_batch.cpu().numpy())

            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)

            try:
                val_auc = roc_auc_score(val_targets, val_preds)
                val_f1 = f1_score(val_targets, (val_preds > 0.5).astype(int))
            except:
                val_auc = 0.0
                val_f1 = 0.0

            history['train_loss'].append(train_loss)
            history['val_auc'].append(val_auc)
            history['val_f1'].append(val_f1)

            scheduler.step(val_auc)

            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            if (epoch + 1) % 5 == 0:
                logger.info(f"  [{optimizer_name}] Epoch {epoch+1}: Loss={train_loss:.4f}, "
                           f"Val AUC={val_auc:.4f}, Val F1={val_f1:.4f}")

        return {
            'best_auc': best_auc,
            'final_auc': val_auc,
            'final_f1': val_f1,
            'epochs_trained': epoch + 1,
            'history': history
        }

    def compare_optimizers(self, X, y, optimizer_configs: List[Dict],
                          n_folds: int = 5, epochs_per_fold: int = 30):
        """Compare optimizers across k-folds"""

        logger.info("="*70)
        logger.info("OPTIMIZER COMPARISON - ICU MORTALITY PREDICTION")
        logger.info("="*70)
        logger.info(f"Dataset: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"Positive class: {y.mean():.1%}")
        logger.info(f"Cross-validation: {n_folds}-fold")
        logger.info(f"Optimizers: {[cfg['name'] for cfg in optimizer_configs]}")
        logger.info("="*70)

        # Import here to avoid circular dependency
        from dl_icu_model import ICUDataset

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        results_by_optimizer = {cfg['name']: [] for cfg in optimizer_configs}
        timing_by_optimizer = {cfg['name']: [] for cfg in optimizer_configs}

        fold_idx = 0
        for train_idx, val_idx in skf.split(X, y):
            fold_idx += 1
            logger.info(f"\nFold {fold_idx}/{n_folds}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_dataset = ICUDataset(X_train, y_train)
            val_dataset = ICUDataset(X_val, y_val)

            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            for opt_config in optimizer_configs:
                opt_name = opt_config['name']
                logger.info(f"  Testing {opt_name}...")

                # Create fresh model for each optimizer
                model = self.model_class(input_size=X.shape[1], hidden_size=128, num_layers=2)

                start_time = time.time()
                fold_result = self.train_fold(
                    model, train_loader, val_loader,
                    opt_config, epochs=epochs_per_fold
                )
                elapsed_time = time.time() - start_time

                results_by_optimizer[opt_name].append(fold_result)
                timing_by_optimizer[opt_name].append(elapsed_time)

                logger.info(f"    AUC: {fold_result['best_auc']:.4f}, "
                           f"F1: {fold_result['final_f1']:.4f}, "
                           f"Time: {elapsed_time:.1f}s")

        # Aggregate results
        logger.info("\n" + "="*70)
        logger.info("FINAL RESULTS")
        logger.info("="*70)

        summary = {}
        for opt_config in optimizer_configs:
            opt_name = opt_config['name']
            aucs = [r['best_auc'] for r in results_by_optimizer[opt_name]]
            f1s = [r['final_f1'] for r in results_by_optimizer[opt_name]]
            epochs = [r['epochs_trained'] for r in results_by_optimizer[opt_name]]
            times = timing_by_optimizer[opt_name]

            summary[opt_name] = {
                'auc_mean': np.mean(aucs),
                'auc_std': np.std(aucs),
                'auc_scores': aucs,
                'f1_mean': np.mean(f1s),
                'f1_std': np.std(f1s),
                'f1_scores': f1s,
                'avg_epochs': np.mean(epochs),
                'total_time': sum(times),
                'avg_time_per_fold': np.mean(times),
            }

            logger.info(f"\n{opt_name.upper()}:")
            logger.info(f"  AUC: {summary[opt_name]['auc_mean']:.4f} ± {summary[opt_name]['auc_std']:.4f}")
            logger.info(f"  F1:  {summary[opt_name]['f1_mean']:.4f} ± {summary[opt_name]['f1_std']:.4f}")
            logger.info(f"  Avg Epochs: {summary[opt_name]['avg_epochs']:.1f}")
            logger.info(f"  Total Time: {summary[opt_name]['total_time']:.1f}s")

        self.results = summary
        return summary, results_by_optimizer


def save_comparison_report(summary: Dict, output_path: str = 'results/dl_optimizer_comparison.json'):
    """Save comparison results to JSON"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'results': summary,
        'best_optimizer': max(summary.items(), key=lambda x: x[1]['auc_mean'])[0],
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\nComparison report saved to {output_path}")
    return report


if __name__ == '__main__':
    logger.info("Optimizer Comparison Framework loaded successfully")
    logger.info("This module will be used with dl_icu_model.py")

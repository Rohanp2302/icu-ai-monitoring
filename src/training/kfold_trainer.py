"""
Phase 3: K-Fold Cross-Validation Training Pipeline

Implements 5-fold stratified cross-validation with:
- Automatic model checkpointing
- Early stopping based on validation loss
- Learning rate scheduling (ReduceLROnPlateau)
- Per-fold training and evaluation
- Task-specific metrics tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import pandas as pd


class MultiTaskTensorDataset(torch.utils.data.Dataset):
    """Dataset for temporal/static features and multi-task targets."""

    def __init__(self, x_temporal: torch.Tensor, x_static: torch.Tensor, y_dict: Dict[str, torch.Tensor]):
        self.x_temporal = x_temporal
        self.x_static = x_static
        self.y_dict = y_dict

    def __len__(self):
        return self.x_temporal.size(0)

    def __getitem__(self, idx):
        y_item = {k: v[idx] for k, v in self.y_dict.items()}
        return self.x_temporal[idx], self.x_static[idx], y_item


class KFoldTrainer:
    """K-fold cross-validation trainer for multi-task ICU model"""

    def __init__(
        self,
        model_class,
        loss_fn_class,
        device: str = "cpu",
        n_splits: int = 5,
        random_state: int = 42,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        """
        Args:
            model_class: Model class (MultiTaskICUModel)
            loss_fn_class: Loss function class (MultiTaskLoss)
            device: 'cpu' or 'cuda'
            n_splits: Number of CV folds
            random_state: Random seed
            checkpoint_dir: Directory to save model weights
            log_dir: Directory to save logs
        """
        self.model_class = model_class
        self.loss_fn_class = loss_fn_class
        self.device = device
        self.n_splits = n_splits
        self.random_state = random_state

        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logger()

        self.fold_results = {}

    def _setup_logger(self) -> logging.Logger:
        """Setup logging to file and console"""
        logger = logging.getLogger("KFoldTrainer")
        logger.setLevel(logging.INFO)

        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"kfold_training_{timestamp}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    @staticmethod
    def _sigmoid_np(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        best_thr = 0.5
        best_j = -1.0
        thresholds = np.linspace(0.05, 0.95, 19)
        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tpr = tp / (tp + fn + 1e-8)
            tnr = tn / (tn + fp + 1e-8)
            j = tpr + tnr - 1.0
            if j > best_j:
                best_j = j
                best_thr = float(thr)
        return best_thr

    def create_fold_datasets(
        self,
        X: np.ndarray,
        X_static: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        fold_idx: int,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train/val/test dataloaders for a specific fold.

        Args:
            X: (N, T, F) temporal feature tensor
            X_static: (N, S) static feature tensor
            y_dict: Dict of label arrays
            fold_idx: Fold index (0 to n_splits-1)

        Returns:
            (train_loader, val_loader, test_loader)
        """
        N = X.shape[0]

        # Generate fold split
        np.random.seed(self.random_state + fold_idx)
        all_indices = np.arange(N)
        np.random.shuffle(all_indices)

        train_size = int(0.6 * N)
        val_size = int(0.2 * N)

        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size : train_size + val_size]
        test_indices = all_indices[train_size + val_size :]

        # Create datasets
        datasets = {}
        for split_name, indices in [
            ("train", train_indices),
            ("val", val_indices),
            ("test", test_indices),
        ]:
            X_split = torch.from_numpy(X[indices]).float()
            X_static_split = torch.from_numpy(X_static[indices]).float()

            y_split = {}
            for task_name, y in y_dict.items():
                y_split[task_name] = torch.from_numpy(y[indices]).float()

            datasets[split_name] = MultiTaskTensorDataset(X_split, X_static_split, y_split)

        # Create dataloaders
        train_loader = DataLoader(
            datasets["train"], batch_size=64, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            datasets["val"], batch_size=64, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            datasets["test"], batch_size=64, shuffle=False, num_workers=0
        )

        return train_loader, val_loader, test_loader

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            model: Model to train
            train_loader: Training dataloader
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            Dict of loss values
        """
        model.train()
        epoch_losses = {}

        for batch_idx, (X_batch, X_static_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            X_static_batch = X_static_batch.to(self.device)

            # Move targets to device
            y_batch_device = {}
            for task, y in y_batch.items():
                y_batch_device[task] = y.to(self.device)

            # Forward pass
            outputs = model(X_batch, X_static_batch)

            # Get task weights
            task_weights = torch.softmax(model.log_task_weights, dim=0)

            # Compute loss
            total_loss, loss_dict = criterion(outputs, y_batch_device, task_weights)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate losses
            for task_name, task_loss in loss_dict.items():
                if task_name not in epoch_losses:
                    epoch_losses[task_name] = []
                epoch_losses[task_name].append(task_loss.item())

            epoch_losses.setdefault("total", []).append(total_loss.item())

        # Average losses
        avg_losses = {task: np.mean(losses) for task, losses in epoch_losses.items()}
        return avg_losses

    def evaluate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion,
        mortality_threshold: Optional[float] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate model on validation or test set.

        Args:
            model: Model to evaluate
            val_loader: Validation/test dataloader
            criterion: Loss function

        Returns:
            (losses_dict, metrics_dict)
        """
        model.eval()
        epoch_losses = {}
        epoch_metrics = {}

        all_outputs = {}
        all_targets = {}

        with torch.no_grad():
            for X_batch, X_static_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                X_static_batch = X_static_batch.to(self.device)

                # Move targets to device
                y_batch_device = {}
                for task, y in y_batch.items():
                    y_batch_device[task] = y.to(self.device)

                # Forward pass
                outputs = model(X_batch, X_static_batch)

                # Get task weights
                task_weights = torch.softmax(model.log_task_weights, dim=0)

                # Compute loss
                total_loss, loss_dict = criterion(outputs, y_batch_device, task_weights)

                # Accumulate losses
                for task_name, task_loss in loss_dict.items():
                    if task_name not in epoch_losses:
                        epoch_losses[task_name] = []
                    epoch_losses[task_name].append(task_loss.item())

                epoch_losses.setdefault("total", []).append(total_loss.item())

                # Store outputs and targets for metrics
                for task_name, output in outputs.items():
                    if task_name in y_batch_device:
                        if task_name not in all_outputs:
                            all_outputs[task_name] = []
                        all_outputs[task_name].append(output.cpu().numpy())
                    elif task_name == "total_los" and "los" in y_batch_device:
                        all_outputs.setdefault("total_los", []).append(output.cpu().numpy())
                        all_targets.setdefault("total_los", []).append(
                            y_batch_device["los"][:, 0:1].cpu().numpy()
                        )
                    elif task_name == "remaining_los" and "los" in y_batch_device:
                        all_outputs.setdefault("remaining_los", []).append(output.cpu().numpy())
                        all_targets.setdefault("remaining_los", []).append(
                            y_batch_device["los"][:, 1:2].cpu().numpy()
                        )

                for task_name, target in y_batch_device.items():
                    if task_name == "los":
                        continue
                    if task_name not in all_targets:
                        all_targets[task_name] = []
                    all_targets[task_name].append(target.cpu().numpy())

        # Average losses
        avg_losses = {task: np.mean(losses) for task, losses in epoch_losses.items()}

        # Compute metrics
        epoch_metrics = self._compute_metrics(all_outputs, all_targets, mortality_threshold)

        return avg_losses, epoch_metrics

    def _compute_metrics(self, outputs: Dict, targets: Dict, mortality_threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Compute task-specific metrics.

        Args:
            outputs: Model outputs concatenated across batches
            targets: Ground truth labels concatenated across batches

        Returns:
            Dict of metrics per task
        """
        metrics = {}

        from sklearn.metrics import (
            roc_auc_score,
            f1_score,
            mean_squared_error,
            mean_absolute_error,
        )

        # Concatenate across batches
        for task_name in list(outputs.keys()):
            if task_name not in targets:
                continue
            outputs[task_name] = np.concatenate(outputs[task_name], axis=0)
            targets[task_name] = np.concatenate(targets[task_name], axis=0)

        # Mortality: AUC
        if "mortality" in outputs:
            try:
                y_true = targets["mortality"].reshape(-1).astype(int)
                mortality_probs = self._sigmoid_np(outputs["mortality"].reshape(-1))
                auc = roc_auc_score(y_true, mortality_probs)
                metrics["mortality_auc"] = auc

                best_thr = mortality_threshold if mortality_threshold is not None else self._find_best_threshold(y_true, mortality_probs)
                y_pred_best = (mortality_probs >= best_thr).astype(int)
                tp = np.sum((y_pred_best == 1) & (y_true == 1))
                tn = np.sum((y_pred_best == 0) & (y_true == 0))
                fp = np.sum((y_pred_best == 1) & (y_true == 0))
                fn = np.sum((y_pred_best == 0) & (y_true == 1))

                metrics["mortality_best_threshold"] = float(best_thr)
                metrics["mortality_sensitivity"] = float(tp / (tp + fn + 1e-8))
                metrics["mortality_specificity"] = float(tn / (tn + fp + 1e-8))
                metrics["mortality_f1"] = float(f1_score(y_true, y_pred_best, zero_division=0))
            except:
                metrics["mortality_auc"] = 0.0

        # Risk: F1 (macro)
        if "risk" in outputs:
            try:
                preds = np.argmax(outputs["risk"], axis=1)
                f1 = f1_score(targets["risk"].reshape(-1), preds, average="macro", zero_division=0)
                metrics["risk_f1"] = f1
            except:
                metrics["risk_f1"] = 0.0

        # LOS: MAE
        if "total_los" in outputs and "total_los" in targets:
            try:
                mae = mean_absolute_error(targets["total_los"].squeeze(), outputs["total_los"].squeeze())
                metrics["los_mae"] = mae
            except:
                metrics["los_mae"] = 0.0

        return metrics

    def train_fold(
        self,
        fold_idx: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 50,
        lr: float = 0.001,
        patience: int = 10,
    ) -> Dict:
        """
        Train model for a single fold.

        Args:
            fold_idx: Fold index
            train_loader: Training dataloader
            val_loader: Validation dataloader
            test_loader: Test dataloader
            epochs: Maximum epochs
            lr: Learning rate
            patience: Early stopping patience

        Returns:
            Dict with fold results
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"FOLD {fold_idx + 1}/{self.n_splits}")
        self.logger.info(f"{'='*80}")

        # Create model and optimizer
        model, criterion = self.model_class(), self.loss_fn_class(device=self.device)
        model = model.to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        best_epoch = 0
        no_improve_count = 0

        fold_history = {
            "train_losses": [],
            "val_losses": [],
            "val_metrics": [],
            "test_results": {},
        }

        for epoch in range(epochs):
            # Train
            train_losses = self.train_epoch(model, train_loader, optimizer, criterion)

            # Validate
            val_losses, val_metrics = self.evaluate(model, val_loader, criterion)

            # Log
            self.logger.info(
                f"Epoch {epoch+1:3d} | "
                f"Train Loss: {train_losses.get('total', 0):.4f} | "
                f"Val Loss: {val_losses.get('total', 0):.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

            fold_history["train_losses"].append(train_losses)
            fold_history["val_losses"].append(val_losses)
            fold_history["val_metrics"].append(val_metrics)

            # Early stopping
            val_loss_avg = np.mean(list(val_losses.values()))
            scheduler.step(val_loss_avg)

            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_epoch = epoch
                no_improve_count = 0

                # Save checkpoint
                checkpoint_path = (
                    self.checkpoint_dir / f"fold_{fold_idx}_best_model.pt"
                )
                torch.save(model.state_dict(), checkpoint_path)
                self.logger.info(f"[SAVE] Best model at epoch {epoch+1}")
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                self.logger.info(f"[EARLY STOP] No improvement for {patience} epochs")
                break

        # Evaluate on test set
        model.load_state_dict(
            torch.load(self.checkpoint_dir / f"fold_{fold_idx}_best_model.pt")
        )
        # Calibrate mortality threshold on validation set and carry to test.
        _, val_metrics_for_threshold = self.evaluate(model, val_loader, criterion)
        mortality_threshold = val_metrics_for_threshold.get("mortality_best_threshold", 0.5)

        test_losses, test_metrics = self.evaluate(
            model,
            test_loader,
            criterion,
            mortality_threshold=mortality_threshold,
        )

        fold_history["test_results"] = {
            "losses": test_losses,
            "metrics": test_metrics,
            "calibrated_threshold": float(mortality_threshold),
        }

        self.logger.info(f"\n[TEST] Fold {fold_idx+1} Results:")
        if "mortality_auc" in test_metrics:
            self.logger.info(f"  Mortality AUC: {test_metrics['mortality_auc']:.4f}")
        if "risk_f1" in test_metrics:
            self.logger.info(f"  Risk F1: {test_metrics['risk_f1']:.4f}")
        if "los_mae" in test_metrics:
            self.logger.info(f"  LOS MAE: {test_metrics['los_mae']:.4f}")

        return fold_history

    def run_kfold(
        self,
        X: np.ndarray,
        X_static: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        epochs: int = 50,
        lr: float = 0.001,
    ) -> Dict:
        """
        Run complete k-fold cross-validation.

        Args:
            X: (N, T, F) temporal feature tensor
            X_static: (N, S) static feature tensor
            y_dict: Dict of label arrays
            epochs: Max epochs per fold
            lr: Learning rate

        Returns:
            Dict with results from all folds
        """
        all_fold_results = {}

        for fold_idx in range(self.n_splits):
            # Create fold datasets
            train_loader, val_loader, test_loader = self.create_fold_datasets(
                X, X_static, y_dict, fold_idx
            )

            # Train fold
            fold_results = self.train_fold(
                fold_idx, train_loader, val_loader, test_loader, epochs, lr
            )

            all_fold_results[f"fold_{fold_idx}"] = fold_results

        # Summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info("K-FOLD SUMMARY")
        self.logger.info(f"{'='*80}")

        # Aggregate metrics
        mortality_aucs = []
        risk_f1s = []
        los_maes = []

        for fold_name, fold_data in all_fold_results.items():
            test_metrics = fold_data["test_results"]["metrics"]
            if "mortality_auc" in test_metrics:
                mortality_aucs.append(test_metrics["mortality_auc"])
            if "risk_f1" in test_metrics:
                risk_f1s.append(test_metrics["risk_f1"])
            if "los_mae" in test_metrics:
                los_maes.append(test_metrics["los_mae"])

        if mortality_aucs:
            self.logger.info(
                f"Mortality AUC: {np.mean(mortality_aucs):.4f} +/- {np.std(mortality_aucs):.4f}"
            )
        if risk_f1s:
            self.logger.info(
                f"Risk F1:       {np.mean(risk_f1s):.4f} +/- {np.std(risk_f1s):.4f}"
            )
        if los_maes:
            self.logger.info(
                f"LOS MAE:       {np.mean(los_maes):.4f} +/- {np.std(los_maes):.4f}"
            )

        # Save results
        results_file = self.log_dir / f"kfold_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    k: {
                        "metrics": v["test_results"]["metrics"],
                        "losses": {lk: float(lv) for lk, lv in v["test_results"]["losses"].items()},
                    }
                    for k, v in all_fold_results.items()
                },
                f,
                indent=2,
            )
        self.logger.info(f"\n[SAVE] Results saved to {results_file}")

        # Save fold-level summary table for quick comparison.
        summary_rows = []
        for fold_name, fold_data in all_fold_results.items():
            m = fold_data["test_results"]["metrics"]
            summary_rows.append(
                {
                    "fold": fold_name,
                    "mortality_auc": float(m.get("mortality_auc", np.nan)),
                    "mortality_f1": float(m.get("mortality_f1", np.nan)),
                    "mortality_sensitivity": float(m.get("mortality_sensitivity", np.nan)),
                    "mortality_specificity": float(m.get("mortality_specificity", np.nan)),
                    "mortality_threshold": float(fold_data["test_results"].get("calibrated_threshold", np.nan)),
                    "risk_f1": float(m.get("risk_f1", np.nan)),
                    "los_mae": float(m.get("los_mae", np.nan)),
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_file = self.log_dir / f"kfold_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(summary_file, index=False)
        self.logger.info(f"[SAVE] Fold summary saved to {summary_file}")

        return all_fold_results

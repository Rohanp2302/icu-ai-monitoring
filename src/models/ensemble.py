"""
Phase 3: Ensemble Learning System

Combines predictions from 6 models:
- 5 fold-specific models (from 5-fold cross-validation)
- 1 full-dataset model

Provides:
- Mean predictions across all models
- Uncertainty estimation (std across ensemble)
- Confidence intervals
- Model-specific predictions for analysis
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class ModelEnsemble:
    """Ensemble of multi-task models for robust predictions"""

    def __init__(
        self,
        model_class,
        device: str = "cpu",
        n_models: int = 6,
    ):
        """
        Args:
            model_class: Model class (MultiTaskICUModel)
            device: 'cpu' or 'cuda'
            n_models: Number of models in ensemble (typically 5 CV folds + 1 full)
        """
        self.model_class = model_class
        self.device = device
        self.n_models = n_models

        self.models = []
        self.model_names = []

    def add_model(self, weight_path: str, model_name: str = None):
        """
        Load and add model to ensemble.

        Args:
            weight_path: Path to model weights
            model_name: Optional name for model (default: model_{idx})
        """
        model = self.model_class().to(self.device)
        model.load_state_dict(torch.load(weight_path, map_location=self.device))
        model.eval()

        self.models.append(model)
        self.model_names.append(model_name or f"model_{len(self.models)-1}")

    def predict_single(
        self,
        model: nn.Module,
        X: torch.Tensor,
        X_static: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions from single model.

        Args:
            model: Model to predict with
            X: (batch_size, seq_len, input_dim) temporal features
            X_static: (batch_size, static_dim) static features

        Returns:
            Dict with predictions per task
        """
        model.eval()
        with torch.no_grad():
            outputs = model(X, X_static)

        # Convert to numpy
        predictions = {}
        for task_name, output in outputs.items():
            predictions[task_name] = output.cpu().numpy()

        return predictions

    def predict_ensemble(
        self,
        X: torch.Tensor,
        X_static: Optional[torch.Tensor] = None,
        return_individual: bool = False,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get ensemble predictions with uncertainty estimation.

        Args:
            X: (batch_size, seq_len, input_dim) temporal features
            X_static: (batch_size, static_dim) static features
            return_individual: Whether to return individual model predictions

        Returns:
            Dict with:
                'mean': Task predictions (ensemble mean)
                'std': Uncertainty estimates (ensemble std)
                'lower': 95% confidence lower bound
                'upper': 95% confidence upper bound
                'individual': Individual model predictions (if requested)
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble!")

        # Get predictions from all models
        all_predictions = []
        for model in self.models:
            preds = self.predict_single(model, X, X_static)
            all_predictions.append(preds)

        # Compute ensemble statistics
        ensemble_results = {"mean": {}, "std": {}, "lower": {}, "upper": {}}

        task_names = all_predictions[0].keys()

        for task_name in task_names:
            # Stack predictions from all models
            task_preds = np.stack(
                [preds[task_name] for preds in all_predictions], axis=0
            )  # (n_models, batch_size, output_shape)

            # Compute statistics
            mean = np.mean(task_preds, axis=0)
            std = np.std(task_preds, axis=0)

            # 95% confidence interval
            lower = np.percentile(task_preds, 2.5, axis=0)
            upper = np.percentile(task_preds, 97.5, axis=0)

            ensemble_results["mean"][task_name] = mean
            ensemble_results["std"][task_name] = std
            ensemble_results["lower"][task_name] = lower
            ensemble_results["upper"][task_name] = upper

        # Individual predictions for analysis
        if return_individual:
            ensemble_results["individual"] = {
                self.model_names[i]: preds for i, preds in enumerate(all_predictions)
            }

        return ensemble_results

    def save_ensemble(self, save_dir: str):
        """
        Save all ensemble models and metadata.

        Args:
            save_dir: Directory to save ensemble
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        for i, model in enumerate(self.models):
            model_file = save_path / f"model_{i}.pt"
            torch.save(model.state_dict(), model_file)

        # Save metadata
        metadata = {
            "n_models": len(self.models),
            "model_names": self.model_names,
            "timestamp": str(pd.Timestamp.now()),
        }

        metadata_file = save_path / "ensemble_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_ensemble(self, load_dir: str):
        """Load all ensemble models from directory"""
        load_path = Path(load_dir)

        # Load metadata
        metadata_file = load_path / "ensemble_metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Load models
        self.models = []
        self.model_names = metadata["model_names"]

        for i in range(metadata["n_models"]):
            model_file = load_path / f"model_{i}.pt"
            self.add_model(str(model_file), metadata["model_names"][i])


class UncertaintyTransformer:
    """Convert ensemble std to uncertainty metrics"""

    @staticmethod
    def calibrate_uncertainty(
        ensemble_std: Dict[str, np.ndarray],
        calibration_factor: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Calibrate uncertainty estimates.

        Args:
            ensemble_std: Standard deviations from ensemble
            calibration_factor: Scale factor for uncertainty

        Returns:
            Calibrated uncertainty estimates
        """
        calibrated = {}
        for task_name, std in ensemble_std.items():
            calibrated[task_name] = std * calibration_factor
        return calibrated

    @staticmethod
    def compute_confidence_scores(
        ensemble_mean: Dict[str, np.ndarray],
        ensemble_std: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Compute confidence scores for predictions.

        High confidence: Low uncertainty
        Low confidence: High uncertainty

        Args:
            ensemble_mean: Mean predictions
            ensemble_std: Standard deviations

        Returns:
            Confidence scores (0 to 1, higher = more confident)
        """
        confidence_scores = {}

        for task_name in ensemble_mean.keys():
            mean = ensemble_mean[task_name]
            std = ensemble_std[task_name]

            # Inverse sigmoid of std: confidence = 1 / (1 + std)
            confidence = 1.0 / (1.0 + np.abs(std))
            confidence_scores[task_name] = confidence

        return confidence_scores

    @staticmethod
    def flag_high_uncertainty_samples(
        ensemble_std: Dict[str, np.ndarray],
        uncertainty_threshold: float = 0.3,
    ) -> Dict[str, np.ndarray]:
        """
        Flag samples with high uncertainty that need review.

        Args:
            ensemble_std: Standard deviations from ensemble
            uncertainty_threshold: Threshold for flagging

        Returns:
            Boolean mask per task (True if uncertain)
        """
        uncertain_masks = {}

        for task_name, std in ensemble_std.items():
            # For probabilities, std > threshold is concerning
            if "mortality" in task_name or "risk" in task_name:
                uncertain_masks[task_name] = np.any(std > uncertainty_threshold, axis=-1)
            else:
                uncertain_masks[task_name] = np.any(std > uncertainty_threshold, axis=-1)

        return uncertain_masks


class EnsemblePredictor:
    """High-level interface for ensemble prediction"""

    def __init__(
        self,
        model_class,
        ensemble_weights_dir: str,
        device: str = "cpu",
    ):
        """
        Args:
            model_class: Model class
            ensemble_weights_dir: Directory containing model weights
            device: 'cpu' or 'cuda'
        """
        self.ensemble = ModelEnsemble(model_class, device=device)
        self.ensemble.load_ensemble(ensemble_weights_dir)

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        X_static: Optional[np.ndarray] = None,
        return_confidence: bool = True,
    ) -> Dict:
        """
        High-level prediction with uncertainty quantification.

        Args:
            X: (batch_size, seq_len, input_dim) features
            X_static: (batch_size, static_dim) static features
            return_confidence: Whether to compute confidence scores

        Returns:
            Dict with predictions, uncertainty, and confidence
        """
        # Convert to torch
        X_torch = torch.from_numpy(X).float()
        X_static_torch = (
            torch.from_numpy(X_static).float() if X_static is not None else None
        )

        # Get ensemble predictions
        ensemble_results = self.ensemble.predict_ensemble(
            X_torch, X_static_torch, return_individual=False
        )

        # Prepare output
        output = {
            "predictions": ensemble_results["mean"],
            "uncertainty": ensemble_results["std"],
            "confidence_lower": ensemble_results["lower"],
            "confidence_upper": ensemble_results["upper"],
        }

        # Confidence scores
        if return_confidence:
            confidence_scores = UncertaintyTransformer.compute_confidence_scores(
                ensemble_results["mean"], ensemble_results["std"]
            )
            output["confidence_scores"] = confidence_scores

            # Flag uncertain samples
            uncertain_masks = UncertaintyTransformer.flag_high_uncertainty_samples(
                ensemble_results["std"], uncertainty_threshold=0.2
            )
            output["needs_review"] = uncertain_masks

        return output

    def predict_batch(
        self,
        X_list: List[np.ndarray],
        batch_size: int = 64,
    ) -> Dict:
        """
        Predict for multiple patients in batches.

        Args:
            X_list: List of patient feature arrays
            batch_size: Prediction batch size

        Returns:
            Dict with predictions for all patients
        """
        all_predictions = {}

        for i in range(0, len(X_list), batch_size):
            batch_X = np.stack(X_list[i : i + batch_size])
            batch_results = self.predict_with_uncertainty(batch_X)

            all_predictions.update(
                {
                    f"patient_{i+j}": {
                        task: batch_results["predictions"][task][j]
                        for task in batch_results["predictions"].keys()
                    }
                    for j in range(len(batch_X))
                }
            )

        return all_predictions


if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 3: ENSEMBLE LEARNING SYSTEM")
    print("=" * 80)
    print("\nEnsemble components configured:")
    print("  - 5-fold CV models (from k-fold training)")
    print("  - 1 full-dataset model")
    print("  - Total: 6 model ensemble")
    print("\nEnsemble capabilities:")
    print("  - Mean predictions across all models")
    print("  - Uncertainty estimation (std, CI bounds)")
    print("  - Confidence scoring")
    print("  - High-uncertainty sample flagging")
    print("  - Batch prediction interface")
    print("\n[OK] Ensemble system ready for integration")

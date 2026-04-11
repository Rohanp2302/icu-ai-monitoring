"""
Logistic Regression Baseline Model

Simple baseline for comparison with multi-task ensemble.
Trained on 42 engineered features + 3 static demographic features.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import pickle
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class LogisticRegressionBaseline:
    """Simple LR baseline for ICU mortality prediction."""

    def __init__(self, max_iter: int = 1000, random_state: int = 42):
        """
        Initialize LR baseline.

        Args:
            max_iter: Max iterations for LR solver
            random_state: Random seed
        """
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        self.scaler = StandardScaler()
        self.n_features = None
        self.is_fitted = False

    def prepare_features(self, x_temporal: np.ndarray, x_static: np.ndarray) -> np.ndarray:
        """
        Prepare features for LR (flatten temporal, concatenate static).

        Args:
            x_temporal: (N, 24, 42) temporal features
            x_static: (N, 20) static features

        Returns:
            (N, 1008 + 20) flattened feature array
        """
        # Flatten temporal: (N, 24, 42) -> (N, 1008)
        x_temporal_flat = x_temporal.reshape(x_temporal.shape[0], -1)

        # Concatenate: (N, 1008 + 20) = (N, 1028)
        x_features = np.concatenate([x_temporal_flat, x_static], axis=1)

        return x_features

    def fit(self, x_temporal: np.ndarray, x_static: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train LR model.

        Args:
            x_temporal: (N, 24, 42) temporal features
            x_static: (N, 20) static features
            y: (N,) binary labels (mortality)

        Returns:
            Dict with training metrics
        """
        try:
            # Prepare features
            X = self.prepare_features(x_temporal, x_static)
            self.n_features = X.shape[1]

            # Remove NaN samples
            valid_mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]

            logger.info(f"Training LR on {len(X_clean)} valid samples ({len(X_clean)/len(X)*100:.1f}%)")

            # Scale features
            X_scaled = self.scaler.fit_transform(X_clean)

            # Train
            self.model.fit(X_scaled, y_clean)
            self.is_fitted = True

            # Evaluate
            y_pred = self.model.predict(X_scaled)
            y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]

            auc = roc_auc_score(y_clean, y_pred_proba)
            f1 = f1_score(y_clean, y_pred)
            acc = accuracy_score(y_clean, y_pred)

            return {
                'status': 'success',
                'n_samples': len(X_clean),
                'auc': float(auc),
                'f1': float(f1),
                'accuracy': float(acc)
            }

        except Exception as e:
            logger.error(f"LR training failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def predict(self, x_temporal: np.ndarray, x_static: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.

        Args:
            x_temporal: (N, 24, 42) temporal features
            x_static: (N, 20) static features

        Returns:
            (predictions, probabilities) of shape (N,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        X = self.prepare_features(x_temporal, x_static)
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]

        return y_pred, y_proba

    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        logger.info(f"Saved to {filepath}")

    def load(self, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
        self.is_fitted = True
        logger.info(f"Loaded from {filepath}")


if __name__ == '__main__':
    print("Logistic Regression Baseline module loaded")

"""
Logistic Regression Baseline Model

Simple baseline for comparison with multi-task ensemble.
Trained on 42 engineered features + 20 static features.
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
        """Initialize LR baseline."""
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def prepare_features(self, x_temporal: np.ndarray, x_static: np.ndarray) -> np.ndarray:
        """Flatten temporal + concatenate static features."""
        x_temporal_flat = x_temporal.reshape(x_temporal.shape[0], -1)
        x_features = np.concatenate([x_temporal_flat, x_static], axis=1)
        return x_features

    def fit(self, x_temporal: np.ndarray, x_static: np.ndarray, y: np.ndarray) -> Dict:
        """Train LR model."""
        try:
            X = self.prepare_features(x_temporal, x_static)
            valid_mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]

            logger.info(f"Training LR on {len(X_clean)} samples")

            X_scaled = self.scaler.fit_transform(X_clean)
            self.model.fit(X_scaled, y_clean)
            self.is_fitted = True

            y_pred = self.model.predict(X_scaled)
            y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]

            auc = roc_auc_score(y_clean, y_pred_proba)
            f1 = f1_score(y_clean, y_pred)
            acc = accuracy_score(y_clean, y_pred)

            return {'status': 'success', 'auc': float(auc), 'f1': float(f1), 'accuracy': float(acc)}
        except Exception as e:
            logger.error(f"LR training failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def predict(self, x_temporal: np.ndarray, x_static: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        X = self.prepare_features(x_temporal, x_static)
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]
        return y_pred, y_proba

    def save(self, filepath: str):
        """Save model."""
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        logger.info(f"Saved to {filepath}")

    def load(self, filepath: str):
        """Load model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
        self.is_fitted = True

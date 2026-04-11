"""
Ensemble Predictor - Combines multiple models for robust mortality prediction
Week 1 Day 2-3: Build ensemble with RF, LR, and GB models
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Combines predictions from multiple models
    - Random Forest (good AUC, moderate recall)
    - Logistic Regression (high recall)
    - Gradient Boosting (balanced performance)
    """
    
    def __init__(self, models_dir: Path = None, use_averaging=True):
        """
        Initialize ensemble predictor
        
        Args:
            models_dir: Directory containing model files
            use_averaging: If True, average probabilities; else use voting
        """
        self.models = {}
        self.scalers = {}
        self.use_averaging = use_averaging
        self.models_dir = models_dir or Path(__file__).parent.parent.parent / 'models'
        self.weights = {
            'random_forest': 0.4,  # Give RF moderate weight (good AUC)
            'logistic_regression': 0.35,  # Give LR good weight (high recall)
            'gradient_boosting': 0.25  # Give GB lower weight (balance)
        }
    
    def load_models(self) -> Dict[str, bool]:
        """Load individual models from disk"""
        load_status = {}
        
        # Try to load Random Forest
        try:
            rf_path = self.models_dir / 'best_model.pkl'
            if rf_path.exists():
                with open(rf_path, 'rb') as f:
                    self.models['random_forest'] = pickle.load(f)
                load_status['random_forest'] = True
                logger.info("✓ Loaded Random Forest model")
            else:
                load_status['random_forest'] = False
        except Exception as e:
            logger.error(f"Failed to load RF: {e}")
            load_status['random_forest'] = False
        
        # Try to load Logistic Regression
        try:
            lr_path = self.models_dir / 'logistic_regression.pkl'
            if lr_path.exists():
                with open(lr_path, 'rb') as f:
                    self.models['logistic_regression'] = pickle.load(f)
                load_status['logistic_regression'] = True
                logger.info("✓ Loaded Logistic Regression model")
            else:
                logger.warning(f"LR model not found at {lr_path}")
                load_status['logistic_regression'] = False
        except Exception as e:
            logger.error(f"Failed to load LR: {e}")
            load_status['logistic_regression'] = False
        
        # Try to load Gradient Boosting
        try:
            gb_path = self.models_dir / 'gradient_boosting.pkl'
            if gb_path.exists():
                with open(gb_path, 'rb') as f:
                    self.models['gradient_boosting'] = pickle.load(f)
                load_status['gradient_boosting'] = True
                logger.info("✓ Loaded Gradient Boosting model")
            else:
                logger.warning(f"GB model not found at {gb_path}")
                load_status['gradient_boosting'] = False
        except Exception as e:
            logger.error(f"Failed to load GB: {e}")
            load_status['gradient_boosting'] = False
        
        return load_status
    
    def predict_proba(self, X: np.ndarray, return_all=False) -> np.ndarray:
        """
        Generate ensemble predictions
        
        Args:
            X: Feature matrix (n_samples, n_features)
            return_all: If True, return all individual model predictions
        
        Returns:
            Ensemble probabilities, or tuple of (ensemble, individual) if return_all=True
        """
        
        # Check that we have at least one model
        if not self.models:
            raise ValueError("No models loaded. Call load_models() first.")
        
        ensemble_proba = np.zeros(len(X))
        individual_predictions = {}
        weights_sum = 0
        
        # RF predictions
        if 'random_forest' in self.models:
            try:
                rf_proba = self.models['random_forest'].predict_proba(X)[:, 1]
                weight = self.weights['random_forest']
                ensemble_proba += weight * rf_proba
                individual_predictions['random_forest'] = rf_proba
                weights_sum += weight
                logger.debug(f"RF prediction range: [{rf_proba.min():.3f}, {rf_proba.max():.3f}]")
            except Exception as e:
                logger.error(f"Error in RF prediction: {e}")
        
        # LR predictions
        if 'logistic_regression' in self.models:
            try:
                lr_proba = self.models['logistic_regression'].predict_proba(X)[:, 1]
                weight = self.weights['logistic_regression']
                ensemble_proba += weight * lr_proba
                individual_predictions['logistic_regression'] = lr_proba
                weights_sum += weight
                logger.debug(f"LR prediction range: [{lr_proba.min():.3f}, {lr_proba.max():.3f}]")
            except Exception as e:
                logger.error(f"Error in LR prediction: {e}")
        
        # GB predictions
        if 'gradient_boosting' in self.models:
            try:
                gb_proba = self.models['gradient_boosting'].predict_proba(X)[:, 1]
                weight = self.weights['gradient_boosting']
                ensemble_proba += weight * gb_proba
                individual_predictions['gradient_boosting'] = gb_proba
                weights_sum += weight
                logger.debug(f"GB prediction range: [{gb_proba.min():.3f}, {gb_proba.max():.3f}]")
            except Exception as e:
                logger.error(f"Error in GB prediction: {e}")
        
        # Normalize by weights
        if weights_sum > 0:
            ensemble_proba = ensemble_proba / weights_sum
        
        if return_all:
            return ensemble_proba, individual_predictions
        else:
            return ensemble_proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Binary predictions using threshold
        
        Args:
            X: Feature matrix
            threshold: Decision threshold
        
        Returns:
            Binary predictions (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_model_count(self) -> int:
        """Return number of loaded models"""
        return len(self.models)
    
    def get_model_status(self) -> Dict[str, bool]:
        """Return status of each model"""
        return {name: (model is not None) for name, model in self.models.items()}


def create_ensemble_predictor(models_dir: Path = None) -> EnsemblePredictor:
    """
    Factory function to create and initialize ensemble predictor
    
    Args:
        models_dir: Path to models directory
    
    Returns:
        Initialized EnsemblePredictor instance
    """
    ensemble = EnsemblePredictor(models_dir)
    status = ensemble.load_models()
    
    loaded_count = sum(1 for v in status.values() if v)
    logger.info(f"Ensemble initialized with {loaded_count} models")
    
    return ensemble


if __name__ == '__main__':
    # Test ensemble predictor
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    
    ensemble = create_ensemble_predictor()
    
    # Test with random features
    n_samples = 100
    n_features = 120
    X_test = np.random.randn(n_samples, n_features)
    
    try:
        proba, individual = ensemble.predict_proba(X_test, return_all=True)
        print(f"\nEnsemble predictions: min={proba.min():.3f}, max={proba.max():.3f}, mean={proba.mean():.3f}")
        print(f"Individual predictions:")
        for model_name, preds in individual.items():
            print(f"  {model_name}: mean={preds.mean():.3f}, range=[{preds.min():.3f}, {preds.max():.3f}]")
    except Exception as e:
        print(f"Error: {e}")

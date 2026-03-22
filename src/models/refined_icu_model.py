"""
Refined ICU Mortality Model - Using Full Feature Set
Sklearn-based gradient boosting with comprehensive feature engineering
Target: > 0.90 AUC with all available eICU data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score,
    recall_score, roc_curve, confusion_matrix, classification_report
)
import logging
import json
from pathlib import Path
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RefinedICUModel:
    """High-performance ICU mortality model using full feature set"""

    def __init__(self):
        self.scaler = RobustScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None

    def load_data(self, hourly_csv: str, outcomes_csv: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare data"""

        logger.info(f"Loading data from {hourly_csv}")

        hourly_df = pd.read_csv(hourly_csv)
        outcomes_df = pd.read_csv(outcomes_csv)

        # Get features
        feature_cols = [col for col in hourly_df.columns
                       if col not in ['patientunitstayid', 'hour']]

        logger.info(f"Features ({len(feature_cols)}): {feature_cols}")

        #Extract patient-level features
        X_features = []
        y_labels = []
        patient_ids = outcomes_df['patientunitstayid'].values

        for patient_id in patient_ids:
            patient_data = hourly_df[hourly_df['patientunitstayid'] == patient_id]

            if len(patient_data) == 0:
                continue

            # Aggregate hourly data for this patient
            # Use latest available values, fillforward for missing
            patient_hourly = patient_data[feature_cols].ffill().bfill()

            if len(patient_hourly) == 0:
                continue

            # Create patient features:
            # - Mean value
            # - Std dev (variability)
            # - Min/Max (range)
            # - Trend (linear regression slope)
            patient_features = []

            for col in feature_cols:
                values = patient_hourly[col].values
                values = values[~np.isnan(values)]

                if len(values) > 0:
                    patient_features.extend([
                        np.mean(values),         # Mean
                        np.std(values) if len(values) > 1 else 0,  # Std
                        np.min(values),          # Min
                        np.max(values),          # Max
                        np.max(values) - np.min(values),  # Range
                    ])
                else:
                    patient_features.extend([0, 0, 0, 0, 0])

            X_features.append(patient_features)

            # Get mortality label
            patient_outcome = outcomes_df[outcomes_df['patientunitstayid'] == patient_id]
            if len(patient_outcome) > 0:
                y_labels.append(patient_outcome['mortality'].values[0])

        X = np.array(X_features)
        y = np.array(y_labels)

        logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Mortality rate: {y.mean():.1%}")
        logger.info(f"Missing patients: {len(patient_ids) - len(y_labels)}")

        return X, y

    def create_models(self) -> Dict:
        """Create ensemble of models"""

        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=42, class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42,
                class_weight='balanced', n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=42, validation_fraction=0.1,
                n_iter_no_change=10
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200, max_depth=15, random_state=42,
                class_weight='balanced', n_jobs=-1
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=200, learning_rate=0.1, random_state=42
            ),
        }

        return models

    def evaluate_models(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> Dict:
        """Evaluate all models with cross-validation"""

        logger.info(f"\nEvaluating {len(self.models)} models with {n_folds}-fold CV...")

        results = {}

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        for model_name, model in self.models.items():
            logger.info(f"\n  Evaluating {model_name}...")

            scoring = {
                'roc_auc': 'roc_auc',
                'f1': 'f1',
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall'
            }

            cv_results = cross_validate(
                model, X, y, cv=skf, scoring=scoring, n_jobs=-1
            )

            results[model_name] = {
                'auc_mean': cv_results['test_roc_auc'].mean(),
                'auc_std': cv_results['test_roc_auc'].std(),
                'auc_scores': cv_results['test_roc_auc'].tolist(),
                'f1_mean': cv_results['test_f1'].mean(),
                'f1_std': cv_results['test_f1'].std(),
                'f1_scores': cv_results['test_f1'].tolist(),
                'accuracy_mean': cv_results['test_accuracy'].mean(),
                'accuracy_std': cv_results['test_accuracy'].std(),
                'precision_mean': cv_results['test_precision'].mean(),
                'recall_mean': cv_results['test_recall'].mean(),
            }

            logger.info(f"    AUC: {results[model_name]['auc_mean']:.4f} ± {results[model_name]['auc_std']:.4f}")
            logger.info(f"    F1:  {results[model_name]['f1_mean']:.4f}")
            logger.info(f"    Accuracy: {results[model_name]['accuracy_mean']:.4f}")

        return results

    def train_best_model(self, X: np.ndarray, y: np.ndarray, results: Dict):
        """Train best model on full data"""

        best_model_name = max(results.items(), key=lambda x: x[1]['auc_mean'])[0]
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]

        logger.info(f"\nTraining best model: {best_model_name}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train
        self.best_model.fit(X_scaled, y)

        logger.info(f"Best model trained")

        return best_model_name

    def run(self, hourly_csv: str = 'data/processed/eicu_hourly_all_features.csv',
            outcomes_csv: str = 'data/processed/eicu_outcomes.csv'):
        """Complete pipeline"""

        logger.info("="*70)
        logger.info("REFINED ICU MORTALITY MODEL")
        logger.info("="*70)

        # Load data
        X, y = self.load_data(hourly_csv, outcomes_csv)

        # Create models
        self.models = self.create_models()

        # Evaluate
        results = self.evaluate_models(X, y, n_folds=5)

        # Train best
        best_model_name = self.train_best_model(X, y, results)

        # Save results
        output_dir = Path('results/dl_models')
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / 'model_comparison.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\nResults saved to {output_dir}")
        logger.info("\n" + "="*70)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*70)

        for model_name, metrics in results.items():
            print(f"{model_name:25s}: AUC={metrics['auc_mean']:.4f}±{metrics['auc_std']:.4f}, "
                  f"F1={metrics['f1_mean']:.4f}, Acc={metrics['accuracy_mean']:.4f}")

        logger.info("="*70)

        return results


if __name__ == '__main__':
    model = RefinedICUModel()
    results = model.run()

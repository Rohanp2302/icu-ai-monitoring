"""
Comprehensive ROC Curve Analysis - All Models, All Data Splits
Training, Validation, and Test ROC curves with full metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ROCAnalyzer:
    """Generate ROC curves for all models across all data splits"""

    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced', n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
            'Extra Trees': ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced', n_jobs=-1),
            'AdaBoost': AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
        }
        self.roc_data = {}

    def load_data(self):
        """Load and prepare data"""
        logger.info("Loading data...")
        hourly_df = pd.read_csv('data/processed/eicu_hourly_all_features.csv')
        outcomes_df = pd.read_csv('data/processed/eicu_outcomes.csv')

        feature_cols = [col for col in hourly_df.columns if col not in ['patientunitstayid', 'hour']]

        X_features = []
        y_labels = []

        for patient_id in outcomes_df['patientunitstayid'].values:
            patient_data = hourly_df[hourly_df['patientunitstayid'] == patient_id]
            if len(patient_data) == 0:
                continue

            patient_hourly = patient_data[feature_cols].ffill().bfill()
            if len(patient_hourly) == 0:
                continue

            patient_features = []
            for col in feature_cols:
                values = patient_hourly[col].values
                values = values[~np.isnan(values)]
                if len(values) > 0:
                    patient_features.extend([
                        np.mean(values),
                        np.std(values) if len(values) > 1 else 0,
                        np.min(values),
                        np.max(values),
                        np.max(values) - np.min(values)
                    ])
                else:
                    patient_features.extend([0]*5)

            X_features.append(patient_features)
            patient_outcome = outcomes_df[outcomes_df['patientunitstayid'] == patient_id]
            if len(patient_outcome) > 0:
                y_labels.append(patient_outcome['mortality'].values[0])

        X = np.array(X_features)
        y = np.array(y_labels)

        logger.info(f"Data loaded: X={X.shape}, y={y.shape}, Mortality rate={y.mean():.1%}")
        return X, y

    def analyze_all_splits(self, X, y, n_splits=5):
        """Generate ROC curves for train/val/test splits"""

        results = {}

        # Create 80/20 train/test split
        logger.info("Creating train/test split (80/20)...")
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale
        logger.info("Scaling features...")
        scaler = RobustScaler()
        X_train_full_scaled = scaler.fit_transform(X_train_full)
        X_test_scaled = scaler.transform(X_test)

        # K-fold on training set for validation
        logger.info(f"Creating {n_splits}-fold validation splits...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for model_name, model in self.models.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"Analyzing {model_name}...")
            logger.info(f"{'='*70}")

            results[model_name] = {
                'train': {'fpr': [], 'tpr': [], 'auc': []},
                'validation': {'fpr': [], 'tpr': [], 'auc': []},
                'test': {'fpr': None, 'tpr': None, 'auc': None}
            }

            # Cross-validation on training set
            fold_num = 0
            for train_idx, val_idx in skf.split(X_train_full_scaled, y_train_full):
                fold_num += 1
                X_tr, X_val = X_train_full_scaled[train_idx], X_train_full_scaled[val_idx]
                y_tr, y_val = y_train_full[train_idx], y_train_full[val_idx]

                # Train fold model
                fold_model = model.__class__(**model.get_params())
                fold_model.fit(X_tr, y_tr)

                # Training ROC
                y_train_pred = fold_model.predict_proba(X_tr)[:, 1]
                fpr_tr, tpr_tr, _ = roc_curve(y_tr, y_train_pred)
                auc_tr = auc(fpr_tr, tpr_tr)
                results[model_name]['train']['fpr'].append(fpr_tr.tolist())
                results[model_name]['train']['tpr'].append(tpr_tr.tolist())
                results[model_name]['train']['auc'].append(auc_tr)

                # Validation ROC
                y_val_pred = fold_model.predict_proba(X_val)[:, 1]
                fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred)
                auc_val = auc(fpr_val, tpr_val)
                results[model_name]['validation']['fpr'].append(fpr_val.tolist())
                results[model_name]['validation']['tpr'].append(tpr_val.tolist())
                results[model_name]['validation']['auc'].append(auc_val)

                logger.info(f"  Fold {fold_num}: Train AUC={auc_tr:.4f}, Val AUC={auc_val:.4f}")

            # Final model on full training set for test evaluation
            logger.info(f"Training final model on full training set...")
            final_model = model.__class__(**model.get_params())
            final_model.fit(X_train_full_scaled, y_train_full)

            # Test ROC
            y_test_pred = final_model.predict_proba(X_test_scaled)[:, 1]
            fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
            auc_test = auc(fpr_test, tpr_test)
            results[model_name]['test']['fpr'] = fpr_test.tolist()
            results[model_name]['test']['tpr'] = tpr_test.tolist()
            results[model_name]['test']['auc'] = auc_test

            # Average metrics
            results[model_name]['train_auc_mean'] = np.mean(results[model_name]['train']['auc'])
            results[model_name]['train_auc_std'] = np.std(results[model_name]['train']['auc'])
            results[model_name]['val_auc_mean'] = np.mean(results[model_name]['validation']['auc'])
            results[model_name]['val_auc_std'] = np.std(results[model_name]['validation']['auc'])

            logger.info(f"\nAggregated Results:")
            logger.info(f"  Train AUC: {results[model_name]['train_auc_mean']:.4f} +/- {results[model_name]['train_auc_std']:.4f}")
            logger.info(f"  Val AUC:   {results[model_name]['val_auc_mean']:.4f} +/- {results[model_name]['val_auc_std']:.4f}")
            logger.info(f"  Test AUC:  {results[model_name]['test']['auc']:.4f}")
            logger.info(f"  Overfitting Gap: {results[model_name]['train_auc_mean'] - results[model_name]['test']['auc']:.4f}")

        return results

    def save_results(self, results, output_path='results/roc_analysis.json'):
        """Save results for visualization"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        summary = {}
        for model_name, data in results.items():
            summary[model_name] = {
                'train_auc_mean': float(data['train_auc_mean']),
                'train_auc_std': float(data['train_auc_std']),
                'val_auc_mean': float(data['val_auc_mean']),
                'val_auc_std': float(data['val_auc_std']),
                'test_auc': float(data['test']['auc']),
                'n_train_folds': len(data['train']['auc']),
                'test_fpr': data['test']['fpr'],
                'test_tpr': data['test']['tpr']
            }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results saved to {output_path}")
        return summary


if __name__ == '__main__':
    analyzer = ROCAnalyzer()
    X, y = analyzer.load_data()
    results = analyzer.analyze_all_splits(X, y, n_splits=5)
    summary = analyzer.save_results(results)

    print("\n" + "="*70)
    print("ROC CURVE ANALYSIS - FINAL SUMMARY")
    print("="*70)

    for model, metrics in sorted(summary.items(), key=lambda x: x[1]['test_auc'], reverse=True):
        print(f"\n{model}:")
        print(f"  Train AUC (5-fold):  {metrics['train_auc_mean']:.4f} +/- {metrics['train_auc_std']:.4f}")
        print(f"  Val AUC (5-fold):    {metrics['val_auc_mean']:.4f} +/- {metrics['val_auc_std']:.4f}")
        print(f"  Test AUC (20%):      {metrics['test_auc']:.4f}")
        gap = metrics['train_auc_mean'] - metrics['test_auc']
        status = "Good (no overfitting)" if gap < 0.1 else "Moderate" if gap < 0.15 else "High (overfitting)"
        print(f"  Overfitting Gap:     {gap:.4f} ({status})")

    print("\n" + "="*70)

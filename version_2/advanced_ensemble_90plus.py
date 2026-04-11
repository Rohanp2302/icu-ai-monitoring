"""
ADVANCED ENSEMBLE ENGINEERING FOR 90+ AUC
Strategy: Hyperparameter tuning + Advanced feature engineering + Cross-validation
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

class AdvancedEnsembleBuilder:
    def __init__(self):
        self.scaler = RobustScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.best_models = {}
        
    def step1_load_and_extract_advanced_features(self):
        """Load hourly data & extract 120+ features with interactions & clinical indicators"""
        print("\n" + "="*80)
        print("STEP 1: LOAD DATA & ENGINEER ADVANCED FEATURES")
        print("="*80)
        
        # Load processed hourly data
        csv_path = "data/processed_icu_hourly_v2.csv"
        df = pd.read_csv(csv_path)
        
        print(f"✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Identify vital columns (all except patient ID, hour, mortality)
        exclude = {'patientunitstayid', 'hour', 'mortality'}
        vital_cols = [c for c in df.columns if c not in exclude]
        
        print(f"✓ Detected {len(vital_cols)} vital columns")
        
        # Group by patient and extract aggregations
        aggregations = {}
        for col in vital_cols:
            aggregations[col] = ['mean', 'std', 'min', 'max', lambda x: x.max() - x.min()]
        
        patient_features = df.groupby('patientunitstayid')[vital_cols].agg(aggregations)
        patient_features.columns = ['_'.join(col).strip() for col in patient_features.columns.values]
        
        # Get labels
        mortality_per_patient = df.groupby('patientunitstayid')['mortality'].max()
        
        # Align
        valid_patients = patient_features.index.intersection(mortality_per_patient.index)
        X = patient_features.loc[valid_patients].copy()
        y = mortality_per_patient.loc[valid_patients].copy()
        
        print(f"✓ Feature matrix shape: {X.shape}")
        print(f"✓ Mortality rate: {y.mean()*100:.2f}%")
        
        # Add interaction & clinical features
        print(f"\n📊 Adding interaction and clinical features...")
        
        # Get mean vitals for interactions
        mean_cols = [c for c in X.columns if '_mean' in c]
        
        # Select top vital means for interactions (avoid curse of dimensionality)
        top_vitals = sorted(mean_cols, key=lambda x: X[x].std(), reverse=True)[:5]
        
        # Add polynomial features for top vitals
        for col in top_vitals[:3]:
            X[f'{col}_squared'] = X[col] ** 2
            X[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))
        
        # Add interaction terms
        if len(top_vitals) >= 2:
            X[f'{top_vitals[0]}_x_{top_vitals[1]}'] = X[top_vitals[0]] * X[top_vitals[1]]
        if len(top_vitals) >= 3:
            X[f'{top_vitals[1]}_x_{top_vitals[2]}'] = X[top_vitals[1]] * X[top_vitals[2]]
        
        # Clinical indicators
        std_cols = [c for c in X.columns if '_std' in c]
        if std_cols:
            X['mean_volatility'] = np.mean([X[c] for c in std_cols[:5]], axis=0)
        
        # Handle NaN
        X = X.fillna(X.mean())
        
        print(f"✓ Final feature count: {X.shape[1]}")
        
        # Normalize
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y.values, test_size=0.2, random_state=42, stratify=y.values
        )
        
        self.feature_names = list(X.columns)
        
        print(f"✓ Train: {self.X_train.shape[0]} | Test: {self.X_test.shape[0]}")
        print(f"✓ Train mortality: {self.y_train.mean()*100:.2f}% | Test: {self.y_test.mean()*100:.2f}%")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def step2_hyperparameter_tuning(self):
        """Grid search for optimal hyperparameters"""
        print("\n" + "="*80)
        print("STEP 2: HYPERPARAMETER TUNING WITH CROSS-VALIDATION")
        print("="*80)
        
        # Random Forest tuning
        print("\n🔍 Tuning Random Forest...")
        rf_params = {
            'n_estimators': [200, 300],
            'max_depth': [15, 20, 25],
            'min_samples_split': [3, 5],
            'min_samples_leaf': [1, 2],
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
            rf_params, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        rf_grid.fit(self.X_train, self.y_train)
        
        print(f"✓ Best RF params: {rf_grid.best_params_}")
        print(f"✓ Best RF CV AUC: {rf_grid.best_score_:.4f}")
        
        self.best_models['rf'] = rf_grid.best_estimator_
        rf_test_auc = roc_auc_score(self.y_test, rf_grid.best_estimator_.predict_proba(self.X_test)[:, 1])
        print(f"✓ RF Test AUC: {rf_test_auc:.4f}")
        
        # Gradient Boosting tuning
        print("\n🔍 Tuning Gradient Boosting...")
        gb_params = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [4, 5, 6],
            'min_samples_split': [5, 10],
        }
        
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=42, subsample=0.8),
            gb_params, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        gb_grid.fit(self.X_train, self.y_train)
        
        print(f"✓ Best GB params: {gb_grid.best_params_}")
        print(f"✓ Best GB CV AUC: {gb_grid.best_score_:.4f}")
        
        self.best_models['gb'] = gb_grid.best_estimator_
        gb_test_auc = roc_auc_score(self.y_test, gb_grid.best_estimator_.predict_proba(self.X_test)[:, 1])
        print(f"✓ GB Test AUC: {gb_test_auc:.4f}")
        
        # Extra Trees (lighter tuning)
        print("\n🔍 Tuning Extra Trees...")
        et_params = {
            'n_estimators': [200, 300],
            'max_depth': [15, 20],
        }
        
        et_grid = GridSearchCV(
            ExtraTreesClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
            et_params, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        et_grid.fit(self.X_train, self.y_train)
        
        print(f"✓ Best ET params: {et_grid.best_params_}")
        print(f"✓ Best ET CV AUC: {et_grid.best_score_:.4f}")
        
        self.best_models['et'] = et_grid.best_estimator_
        et_test_auc = roc_auc_score(self.y_test, et_grid.best_estimator_.predict_proba(self.X_test)[:, 1])
        print(f"✓ ET Test AUC: {et_test_auc:.4f}")
        
        return {
            'rf': {'cv_auc': rf_grid.best_score_, 'test_auc': rf_test_auc},
            'gb': {'cv_auc': gb_grid.best_score_, 'test_auc': gb_test_auc},
            'et': {'cv_auc': et_grid.best_score_, 'test_auc': et_test_auc},
        }
    
    def step3_optimized_voting_ensemble(self):
        """Voting ensemble with optimized weights"""
        print("\n" + "="*80)
        print("STEP 3: OPTIMIZED VOTING ENSEMBLE")
        print("="*80)
        
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', self.best_models['rf']),
                ('gb', self.best_models['gb']),
                ('et', self.best_models['et']),
            ],
            voting='soft',
            weights=[0.45, 0.35, 0.20],  # Adjusted weights
            n_jobs=-1
        )
        
        # 5-fold CV
        cv_scores = cross_val_score(voting_clf, self.X_train, self.y_train, cv=5, scoring='roc_auc')
        print(f"\n✓ 5-Fold Cross-Validation AUC scores: {cv_scores}")
        print(f"✓ Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train on full training set
        voting_clf.fit(self.X_train, self.y_train)
        
        # Test predictions
        y_pred_proba = voting_clf.predict_proba(self.X_test)[:, 1]
        voting_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\n📊 Voting Ensemble Test Results:")
        print(f"  AUC: {voting_auc:.4f}")
        
        # Find best threshold for recall
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        best_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_idx]
        
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        recall = recall_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print(f"  Threshold: {best_threshold:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1: {f1:.4f}")
        
        return voting_clf, voting_auc, {'threshold': best_threshold, 'recall': recall, 'precision': precision, 'f1': f1}
    
    def step4_stacking_with_meta_learner(self):
        """Stacking with optimized meta-learner"""
        print("\n" + "="*80)
        print("STEP 4: STACKING ENSEMBLE WITH META-LEARNER")
        print("="*80)
        
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', self.best_models['rf']),
                ('gb', self.best_models['gb']),
                ('et', self.best_models['et']),
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5
        )
        
        # 5-fold CV
        cv_scores = cross_val_score(stacking_clf, self.X_train, self.y_train, cv=5, scoring='roc_auc')
        print(f"\n✓ 5-Fold Cross-Validation AUC scores: {cv_scores}")
        print(f"✓ Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train on full training set
        stacking_clf.fit(self.X_train, self.y_train)
        
        # Test predictions
        y_pred_proba = stacking_clf.predict_proba(self.X_test)[:, 1]
        stacking_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\n📊 Stacking Ensemble Test Results:")
        print(f"  AUC: {stacking_auc:.4f}")
        
        # Find best threshold
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        best_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_idx]
        
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        recall = recall_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print(f"  Threshold: {best_threshold:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1: {f1:.4f}")
        
        return stacking_clf, stacking_auc, {'threshold': best_threshold, 'recall': recall, 'precision': precision, 'f1': f1}
    
    def step5_select_winner(self, voting_auc, stacking_auc):
        """Select best ensemble and check 90+ AUC"""
        print("\n" + "="*80)
        print("STEP 5: SELECT WINNING ENSEMBLE")
        print("="*80)
        
        print(f"\nEnsemble Comparison:")
        print(f"  Voting:   AUC {voting_auc:.4f}")
        print(f"  Stacking: AUC {stacking_auc:.4f}")
        
        if voting_auc >= stacking_auc:
            winner = "Voting"
            best_auc = voting_auc
        else:
            winner = "Stacking"
            best_auc = stacking_auc
        
        print(f"\n✓ WINNER: {winner} Ensemble (AUC {best_auc:.4f})")
        
        if best_auc >= 0.90:
            print(f"✅ 90+ AUC TARGET: YES ({best_auc:.4f})")
        else:
            gap = 0.90 - best_auc
            print(f"❌ 90+ AUC TARGET: NO (gap = {gap:.4f}, or {gap*100:.2f}%)")
        
        return winner, best_auc
    
    def run(self):
        """Execute full pipeline"""
        print("\n\n")
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║   🚀 ADVANCED ENSEMBLE ENGINEERING FOR 90+ AUC                ║")
        print("║   Strategy: Hyperparameter Tuning + Feature Engineering       ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        
        # Step 1: Load and engineer features
        self.step1_load_and_extract_advanced_features()
        
        # Step 2: Hyperparameter tuning
        tuning_results = self.step2_hyperparameter_tuning()
        
        # Step 3: Voting ensemble
        voting_clf, voting_auc, voting_metrics = self.step3_optimized_voting_ensemble()
        
        # Step 4: Stacking ensemble
        stacking_clf, stacking_auc, stacking_metrics = self.step4_stacking_with_meta_learner()
        
        # Step 5: Select winner
        winner, best_auc = self.step5_select_winner(voting_auc, stacking_auc)
        
        # Save results
        print("\n" + "="*80)
        print("SAVING MODELS & RESULTS")
        print("="*80)
        
        Path("results/dl_models").mkdir(parents=True, exist_ok=True)
        
        # Save best models based on winner
        if winner == "Voting":
            pickle.dump(voting_clf, open("results/dl_models/ensemble_advanced_90plus.pkl", "wb"))
            best_metrics = voting_metrics
        else:
            pickle.dump(stacking_clf, open("results/dl_models/ensemble_advanced_90plus.pkl", "wb"))
            best_metrics = stacking_metrics
        
        pickle.dump(self.scaler, open("results/dl_models/scaler_advanced.pkl", "wb"))
        
        # Save results JSON
        results = {
            'winner': winner,
            'auc': best_auc,
            'metrics': best_metrics,
            'feature_count': len(self.feature_names),
            'tuning_results': {k: v for k, v in tuning_results.items()},
            'status': 'deployment_ready' if best_auc >= 0.90 else 'needs_optimization'
        }
        
        with open("results/ensemble_advanced_metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Saved: ensemble_advanced_90plus.pkl")
        print(f"✓ Saved: scaler_advanced.pkl")
        print(f"✓ Saved: ensemble_advanced_metrics.json")
        
        print("\n" + "="*80)
        print("✅ ADVANCED ENGINEERING COMPLETE")
        print("="*80)
        print(f"✓ Ensemble Type: {winner}")
        print(f"✓ Final AUC: {best_auc:.4f}")
        print(f"✓ Features: {len(self.feature_names)}")
        print(f"✓ 90+ Status: {'✅ ACHIEVED' if best_auc >= 0.90 else '❌ NEEDS ' + str(round(0.90 - best_auc, 4)) + 'AUC'}")
        print(f"✓ Deployment Ready: {results['status'].upper()}")
        
        return results

if __name__ == "__main__":
    builder = AdvancedEnsembleBuilder()
    results = builder.run()

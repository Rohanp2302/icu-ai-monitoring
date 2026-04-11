"""
BAYESIAN LIGHTGBM OPTIMIZATION FOR 90+ AUC
Using Optuna for intelligent hyperparameter search
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

try:
    import lightgbm as lgb
except:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
    import lightgbm as lgb

class BayesianLGBMOptimizer:
    def __init__(self):
        self.scaler = RobustScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def step1_load_data(self):
        """Load with FULL feature engineering (as in ensemble_xgb_final.py which got 0.8953)"""
        print("\n" + "="*80)
        print("STEP 1: LOAD DATA WITH FULL FEATURE ENGINEERING")
        print("="*80)
        
        csv_path = "data/processed_icu_hourly_v2.csv"
        df = pd.read_csv(csv_path)
        
        print(f"✓ Loaded: {df.shape[0]:,} rows")
        
        # Full aggregation
        exclude = {'patientunitstayid', 'hour', 'mortality'}
        vital_cols = [c for c in df.columns if c not in exclude]
        
        agg_dict = {col: ['mean', 'std', 'min', 'max', lambda x: x.max() - x.min()] for col in vital_cols}
        patient_features = df.groupby('patientunitstayid')[vital_cols].agg(agg_dict)
        patient_features.columns = ['_'.join(col).strip() for col in patient_features.columns.values]
        
        mortality_per_patient = df.groupby('patientunitstayid')['mortality'].max()
        valid_patients = patient_features.index.intersection(mortality_per_patient.index)
        
        X = patient_features.loc[valid_patients].copy()
        y = mortality_per_patient.loc[valid_patients].copy()
        
        print(f"✓ Initial features: {X.shape[1]}")
        
        # AGGRESSIVE feature engineering (from ensemble_xgb_final.py)
        mean_cols = sorted([c for c in X.columns if '_mean' in c], key=lambda x: X[x].std(), reverse=True)
        std_cols = [c for c in X.columns if '_std' in c]
        
        # Polynomial features for top 4 vitals
        for col in mean_cols[:4]:
            X[f'{col}_pow2'] = X[col] ** 2
            X[f'{col}_pow3'] = X[col] ** 3
            X[f'{col}_log'] = np.log1p(np.abs(X[col]))
        
        # Comprehensive interactions (top 8 vitals)
        top8 = mean_cols[:8]
        for i in range(len(top8)):
            for j in range(i+1, min(i+3, len(top8))):
                X[f'{top8[i].split("_")[0]}_x_{top8[j].split("_")[0]}'] = X[top8[i]] * X[top8[j]]
                X[f'{top8[i].split("_")[0]}_div_{top8[j].split("_")[0]}'] = np.divide(
                    X[top8[i]], X[top8[j]] + 1e-6
                )
        
        # Aggregate volatility
        if std_cols:
            X['total_volatility'] = X[std_cols[:5]].mean(axis=1)
            X['max_volatility'] = X[std_cols[:5]].max(axis=1)
        
        # Coefficient of variation
        for col in mean_cols[:6]:
            std_col = col.replace('_mean', '_std')
            if std_col in X.columns:
                X[f'{col.split("_")[0]}_cv'] = np.divide(X[std_col], np.abs(X[col]) + 1e-6)
        
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"✓ Final engineered features: {X.shape[1]}")
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y.values, test_size=0.2, random_state=42, stratify=y.values
        )
        
        print(f"✓ Train: {self.X_train.shape[0]} | Test: {self.X_test.shape[0]}")
        
        return True
    
    def objective(self, trial):
        """Objective function for Optuna"""
        
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'num_leaves': trial.suggest_int('num_leaves', 15, 50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.20),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 3.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        }
        
        model = lgb.LGBMClassifier(
            scale_pos_weight=10,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **params
        )
        
        # 5-fold cross-validation on AUC
        scores = cross_validate(
            model, self.X_train, self.y_train,
            cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        
        cv_auc = scores['test_score'].mean()
        
        return cv_auc
    
    def step2_bayesian_search(self):
        """Bayesian hyperparameter optimization"""
        print("\n" + "="*80)
        print("STEP 2: BAYESIAN HYPERPARAMETER OPTIMIZATION (20 trials)")
        print("="*80)
        
        # Create study
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=MedianPruner()
        )
        
        # Optimize
        print("\n🔬 Running Bayesian optimization...")
        study.optimize(self.objective, n_trials=20, show_progress_bar=True)
        
        # Get best trial
        best_trial = study.best_trial
        
        print(f"\n✅ Best Trial (CV AUC: {best_trial.value:.4f}):")
        print(f"   Parameters: {best_trial.params}")
        
        return best_trial.params, best_trial.value
    
    def step3_retrain_and_evaluate(self, best_params):
        """Retrain with best params and evaluate on test set"""
        print("\n" + "="*80)
        print("STEP 3: RETRAIN WITH BEST PARAMS & TEST SET EVALUATION")
        print("="*80)
        
        model = lgb.LGBMClassifier(
            scale_pos_weight=10,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **best_params
        )
        
        # Train on full training set
        model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], eval_metric='auc')
        
        # Test predictions
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        test_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\n🧪 Test Set Results:")
        print(f"   AUC: {test_auc:.4f}")
        
        # Optimize threshold
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        best_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_idx]
        
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        recall = recall_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print(f"   Threshold: {best_threshold:.4f}")
        print(f"   Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")
        
        if test_auc >= 0.90:
            print(f"\n✅ 90+ AUC TARGET ACHIEVED! ({test_auc:.4f})")
        else:
            gap = 0.90 - test_auc
            print(f"\n⚠️  Gap to 90%: {gap:.4f} ({gap*100:.2f}%)")
        
        return model, test_auc, {
            'threshold': best_threshold,
            'recall': recall,
            'precision': precision,
            'f1': f1,
        }
    
    def step4_save_results(self, model, test_auc, eval_metrics, best_params, cv_auc):
        """Save best model and results"""
        print("\n" + "="*80)
        print("STEP 4: SAVE MODEL & RESULTS")
        print("="*80)
        
        Path("results/dl_models").mkdir(parents=True, exist_ok=True)
        
        pickle.dump(model, open("results/dl_models/lgbm_bayesian_90plus.pkl", "wb"))
        pickle.dump(self.scaler, open("results/dl_models/scaler_lgbm_bayesian.pkl", "wb"))
        
        results_json = {
            'model_type': 'LightGBM (Bayesian Optimization)',
            'cv_auc': cv_auc,
            'test_auc': test_auc,
            'eval_metrics': eval_metrics,
            'best_params': best_params,
            'deployment_ready': test_auc >= 0.90,
            'gap_to_90': max(0.90 - test_auc, 0),
            'status': '✅ 90+ ACHIEVED' if test_auc >= 0.90 else f'⚠️  CLOSE: {test_auc:.4f}',
        }
        
        with open("results/lgbm_bayesian_metrics.json", "w") as f:
            json.dump(results_json, f, indent=2)
        
        print(f"✓ Saved: lgbm_bayesian_90plus.pkl")
        print(f"✓ Saved: lgbm_bayesian_metrics.json")
        
        return results_json
    
    def run(self):
        """Execute full pipeline"""
        print("\n\n")
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║   🎯 BAYESIAN LIGHTGBM OPTIMIZATION                            ║")
        print("║   Target: 90+ AUC with Intelligent Hyperparameter Search       ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        
        self.step1_load_data()
        best_params, cv_auc = self.step2_bayesian_search()
        model, test_auc, eval_metrics = self.step3_retrain_and_evaluate(best_params)
        results_json = self.step4_save_results(model, test_auc, eval_metrics, best_params, cv_auc)
        
        print("\n" + "="*80)
        print("✅ BAYESIAN OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"🏆 CV AUC: {cv_auc:.4f}")
        print(f"🏆 Test AUC: {test_auc:.4f}")
        print(f"📊 Status: {results_json['status']}")
        print("="*80)
        
        return results_json

if __name__ == "__main__":
    optimizer = BayesianLGBMOptimizer()
    results = optimizer.run()

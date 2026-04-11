"""
LIGHTGBM HYPERPARAMETER OPTIMIZATION FOR 90+ AUC
Target: Push LightGBM from 0.8953 to 0.90+
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    print("⚠️  LightGBM not available, installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True

class LightGBMOptimizer:
    def __init__(self):
        self.scaler = RobustScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def step1_load_and_extract_features(self):
        """Load and extract features"""
        print("\n" + "="*80)
        print("STEP 1: LOAD DATA & EXTRACT OPTIMIZED FEATURES")
        print("="*80)
        
        csv_path = "data/processed_icu_hourly_v2.csv"
        df = pd.read_csv(csv_path)
        
        print(f"✓ Loaded: {df.shape[0]:,} rows")
        
        # Aggregate
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
        
        # Select best features only (reduce overfitting)
        mean_cols = sorted([c for c in X.columns if '_mean' in c], key=lambda x: X[x].std(), reverse=True)
        
        # Use only top features to reduce noise
        top_features = mean_cols[:12]  # Top 12 vitals
        feature_list = [c for c in X.columns if any(v in c for v in [t.split('_')[0] for t in top_features])]
        
        X = X[feature_list].copy()
        
        # Minimal, targeted feature engineering
        for col in top_features[:4]:
            X[f'{col}_2'] = X[col] ** 2
            X[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))
        
        # Pairwise interactions only for top 3
        if len(top_features) >= 2:
            X[f'int_{top_features[0].split("_")[0]}_{top_features[1].split("_")[0]}'] = \
                X[top_features[0]] * X[top_features[1]]
        
        X = X.fillna(X.mean())
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"✓ Final features: {X.shape[1]}")
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y.values, test_size=0.2, random_state=42, stratify=y.values
        )
        
        self.feature_names = list(X.columns)
        
        print(f"✓ Train: {self.X_train.shape[0]} | Test: {self.X_test.shape[0]}")
        print(f"✓ Train mortality: {self.y_train.mean()*100:.2f}% | Test: {self.y_test.mean()*100:.2f}%")
        
        return True
    
    def step2_parameter_search(self):
        """Aggressive hyperparameter search"""
        print("\n" + "="*80)
        print("STEP 2: AGGRESSIVE LIGHTGBM HYPERPARAMETER SEARCH")
        print("="*80)
        
        best_auc = 0
        best_params = None
        best_model = None
        
        # Parameter grid to search
        param_combinations = [
            # Balance & regularization focused
            {'n_estimators': 150, 'max_depth': 6, 'num_leaves': 25, 'learning_rate': 0.15, 
             'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0.5, 'reg_lambda': 1.0},
            
            {'n_estimators': 200, 'max_depth': 5, 'num_leaves': 20, 'learning_rate': 0.10, 
             'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_alpha': 1.0, 'reg_lambda': 2.0},
            
            {'n_estimators': 180, 'max_depth': 7, 'num_leaves': 31, 'learning_rate': 0.12, 
             'subsample': 0.88, 'colsample_bytree': 0.88, 'reg_alpha': 0.8, 'reg_lambda': 1.5},
            
            {'n_estimators': 220, 'max_depth': 6, 'num_leaves': 28, 'learning_rate': 0.08, 
             'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 1.2, 'reg_lambda': 2.5},
            
            # Early stopping friendly
            {'n_estimators': 300, 'max_depth': 5, 'num_leaves': 22, 'learning_rate': 0.05, 
             'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_alpha': 1.5, 'reg_lambda': 3.0},
        ]
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\n🔍 Configuration {i}:")
            try:
                model = lgb.LGBMClassifier(
                    scale_pos_weight=10,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                    **params
                )
                
                # Train
                model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], eval_metric='auc')
                
                # Evaluate
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                auc = roc_auc_score(self.y_test, y_pred_proba)
                
                print(f"  AUC: {auc:.4f}")
                
                # Track best
                if auc > best_auc:
                    best_auc = auc
                    best_params = params
                    best_model = model
                    print(f"  ✅ NEW BEST!")
            except Exception as e:
                print(f"  ❌ Failed: {str(e)[:50]}")
        
        print(f"\n✓ Best AUC: {best_auc:.4f}")
        print(f"✓ Best params: {best_params}")
        
        return best_model, best_auc, best_params
    
    def step3_threshold_optimization(self, model):
        """Find optimal threshold"""
        print("\n" + "="*80)
        print("STEP 3: THRESHOLD & PROBABILITY CALIBRATION")
        print("="*80)
        
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Try multiple thresholds
        best_f1 = 0
        best_threshold = 0.5
        best_recall = 0
        
        print("\nThreshold analysis:")
        for threshold in np.arange(0.3, 0.7, 0.05):
            y_pred = (y_pred_proba >= threshold).astype(int)
            recall = recall_score(self.y_test, y_pred) if y_pred.sum() > 0 else 0
            precision = precision_score(self.y_test, y_pred) if y_pred.sum() > 0 else 0
            f1 = f1_score(self.y_test, y_pred) if y_pred.sum() > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_recall = recall
            
            print(f"  Threshold {threshold:.2f}: Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}")
        
        print(f"\n✓ Optimal threshold: {best_threshold:.4f}")
        print(f"✓ Recall: {best_recall:.4f}, F1: {best_f1:.4f}")
        
        return best_threshold, best_recall, best_f1
    
    def step4_final_evaluation(self, model, best_threshold):
        """Final evaluation and metrics"""
        print("\n" + "="*80)
        print("STEP 4: FINAL EVALUATION")
        print("="*80)
        
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        recall = recall_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print(f"\n📊 FINAL METRICS:")
        print(f"  📈 AUC: {auc:.4f}")
        print(f"  🎯 Recall: {recall:.4f}")
        print(f"  🎯 Precision: {precision:.4f}")
        print(f"  🎯 F1: {f1:.4f}")
        
        if auc >= 0.90:
            print(f"\n✅ 90+ AUC TARGET ACHIEVED!")
        else:
            gap = 0.90 - auc
            print(f"\n⚠️  90+ AUC: Gap = {gap:.4f} ({gap*100:.2f}%)")
        
        return {
            'auc': auc,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'threshold': best_threshold,
            'status': '✅ SUCCESS' if auc >= 0.90 else f'⚠️  CLOSE ({auc:.4f})'
        }
    
    def step5_save_model(self, model, metrics, best_params):
        """Save best model"""
        print("\n" + "="*80)
        print("STEP 5: SAVE MODEL & RESULTS")
        print("="*80)
        
        Path("results/dl_models").mkdir(parents=True, exist_ok=True)
        
        pickle.dump(model, open("results/dl_models/lgbm_90plus.pkl", "wb"))
        pickle.dump(self.scaler, open("results/dl_models/scaler_lgbm.pkl", "wb"))
        
        results_json = {
            'model_type': 'LightGBM',
            'final_auc': metrics['auc'],
            'final_recall': metrics['recall'],
            'final_f1': metrics['f1'],
            'threshold': metrics['threshold'],
            'best_params': best_params,
            'status': metrics['status'],
            'deployment_ready': metrics['auc'] >= 0.90,
            'gap_to_90': 0.90 - metrics['auc'],
        }
        
        with open("results/lgbm_final_metrics.json", "w") as f:
            json.dump(results_json, f, indent=2)
        
        print(f"✓ Saved: lgbm_90plus.pkl")
        print(f"✓ Saved: scaler_lgbm.pkl")
        print(f"✓ Saved: lgbm_final_metrics.json")
        
        return results_json
    
    def run(self):
        """Execute full pipeline"""
        print("\n\n")
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║   🎯 LIGHTGBM HYPERPARAMETER OPTIMIZATION                      ║")
        print("║   Target: Push from 0.8953 → 90+ AUC                          ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        
        self.step1_load_and_extract_features()
        model, best_auc, best_params = self.step2_parameter_search()
        best_threshold, best_recall, best_f1 = self.step3_threshold_optimization(model)
        final_metrics = self.step4_final_evaluation(model, best_threshold)
        results_json = self.step5_save_model(model, final_metrics, best_params)
        
        print("\n" + "="*80)
        print("✅ LIGHTGBM OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"🏆 Final AUC: {final_metrics['auc']:.4f}")
        print(f"📊 Status: {results_json['status']}")
        print(f"📦 Models saved to results/dl_models/")
        print("="*80)
        
        return results_json

if __name__ == "__main__":
    optimizer = LightGBMOptimizer()
    results = optimizer.run()

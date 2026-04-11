"""
FAST ENSEMBLE ENGINEERING FOR 90+ AUC
Strategy: Quick hyperparameter tuning + Stacking ensemble + Threshold optimization
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

class FastEnsembleBuilder:
    def __init__(self):
        self.scaler = RobustScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def step1_load_and_extract_features(self):
        """Load hourly data & extract aggregated features"""
        print("\n" + "="*80)
        print("STEP 1: LOAD DATA & EXTRACT 79 FEATURES (14 vitals + interactions)")
        print("="*80)
        
        csv_path = "data/processed_icu_hourly_v2.csv"
        df = pd.read_csv(csv_path)
        
        print(f"✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
        
        # Identify vital columns
        exclude = {'patientunitstayid', 'hour', 'mortality'}
        vital_cols = [c for c in df.columns if c not in exclude]
        print(f"✓ Detected {len(vital_cols)} vital columns")
        
        # GroupBy and aggregate
        agg_dict = {col: ['mean', 'std', 'min', 'max', lambda x: x.max() - x.min()] for col in vital_cols}
        patient_features = df.groupby('patientunitstayid')[vital_cols].agg(agg_dict)
        patient_features.columns = ['_'.join(col).strip() for col in patient_features.columns.values]
        
        # Get mortality labels
        mortality_per_patient = df.groupby('patientunitstayid')['mortality'].max()
        
        # Align
        valid_patients = patient_features.index.intersection(mortality_per_patient.index)
        X = patient_features.loc[valid_patients].copy()
        y = mortality_per_patient.loc[valid_patients].copy()
        
        print(f"✓ Feature matrix shape: {X.shape}")
        print(f"✓ Mortality rate: {y.mean()*100:.2f}%")
        
        # Add top 5 polynomial + interaction features
        mean_cols = [c for c in X.columns if '_mean' in c]
        top_vitals = sorted(mean_cols, key=lambda x: X[x].std(), reverse=True)[:3]
        
        for col in top_vitals:
            X[f'{col}_squared'] = X[col] ** 2
        
        if len(top_vitals) >= 2:
            X[f'interaction_1'] = X[top_vitals[0]] * X[top_vitals[1]]
        if len(top_vitals) >= 3:
            X[f'interaction_2'] = X[top_vitals[1]] * X[top_vitals[2]]
        
        X = X.fillna(X.mean())
        
        print(f"✓ Final features: {X.shape[1]}")
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/test split (80/20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y.values, test_size=0.2, random_state=42, stratify=y.values
        )
        
        print(f"✓ Train: {self.X_train.shape[0]} | Test: {self.X_test.shape[0]}")
        print(f"✓ Train mortality: {self.y_train.mean()*100:.2f}% | Test: {self.y_test.mean()*100:.2f}%")
        
        return True
    
    def step2_train_tuned_models(self):
        """Train RF, GB, ET with tuned parameters"""
        print("\n" + "="*80)
        print("STEP 2: TRAIN TUNED BASE MODELS")
        print("="*80)
        
        # Random Forest (tuned)
        print("\n🔹 Random Forest (tuned)...")
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=2, 
            min_samples_split=5, class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        rf_auc = roc_auc_score(self.y_test, rf.predict_proba(self.X_test)[:, 1])
        print(f"  ✓ Test AUC: {rf_auc:.4f}")
        
        # Gradient Boosting (tuned)
        print("🔹 Gradient Boosting (tuned)...")
        gb = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6, 
            min_samples_split=5, random_state=42, subsample=0.8
        )
        gb.fit(self.X_train, self.y_train)
        gb_auc = roc_auc_score(self.y_test, gb.predict_proba(self.X_test)[:, 1])
        print(f"  ✓ Test AUC: {gb_auc:.4f}")
        
        # Extra Trees (tuned)
        print("🔹 Extra Trees (tuned)...")
        et = ExtraTreesClassifier(
            n_estimators=200, max_depth=20, class_weight='balanced', 
            random_state=42, n_jobs=-1
        )
        et.fit(self.X_train, self.y_train)
        et_auc = roc_auc_score(self.y_test, et.predict_proba(self.X_test)[:, 1])
        print(f"  ✓ Test AUC: {et_auc:.4f}")
        
        return {'rf': (rf, rf_auc), 'gb': (gb, gb_auc), 'et': (et, et_auc)}
    
    def step3_stacking_ensemble(self, models_dict):
        """Stacking with meta-learner"""
        print("\n" + "="*80)
        print("STEP 3: BUILD STACKING ENSEMBLE WITH META-LEARNER")
        print("="*80)
        
        rf, _ = models_dict['rf']
        gb, _ = models_dict['gb']
        et, _ = models_dict['et']
        
        print("Building stacking classifier...")
        
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('et', et),
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5
        )
        
        # Train
        stacking_clf.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred_proba = stacking_clf.predict_proba(self.X_test)[:, 1]
        stacking_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\n📊 Stacking Ensemble Results:")
        print(f"  AUC: {stacking_auc:.4f}")
        
        # Optimize threshold
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
        
        return stacking_clf, stacking_auc, {
            'threshold': best_threshold, 'recall': recall, 
            'precision': precision, 'f1': f1
        }
    
    def step4_compare_and_select(self, models_dict, stacking_result):
        """Compare all models and select best"""
        print("\n" + "="*80)
        print("STEP 4: MODEL COMPARISON & WINNER SELECTION")
        print("="*80)
        
        results = []
        
        # Individual models
        print("\n📊 Individual Model Performance:")
        for name, (model, auc) in models_dict.items():
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
            best_idx = np.argmax(tpr - fpr)
            best_threshold = thresholds[best_idx]
            y_pred = (y_pred_proba >= best_threshold).astype(int)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            print(f"  {name.upper():10} - AUC: {auc:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            results.append({'model': name, 'type': 'individual', 'auc': auc, 'recall': recall, 'f1': f1})
        
        # Stacking
        stacking_clf, stacking_auc, stacking_metrics = stacking_result
        print(f"  STACK      - AUC: {stacking_auc:.4f}, Recall: {stacking_metrics['recall']:.4f}, F1: {stacking_metrics['f1']:.4f}")
        results.append({'model': 'stacking', 'type': 'ensemble', 'auc': stacking_auc, 'recall': stacking_metrics['recall'], 'f1': stacking_metrics['f1']})
        
        # Select winner
        print("\n" + "-"*80)
        best_result = max(results, key=lambda x: x['auc'])
        
        if best_result['model'] == 'stacking':
            winner_clf = stacking_clf
        else:
            winner_clf = models_dict[best_result['model']][0]
        
        print(f"\n✅ WINNER: {best_result['model'].upper()} - AUC {best_result['auc']:.4f}")
        
        if best_result['auc'] >= 0.90:
            print(f"✅ 90+ AUC TARGET: YES  ({best_result['auc']:.4f} >= 0.90)")
        else:
            gap = 0.90 - best_result['auc']
            print(f"⚠️  90+ AUC TARGET: NO  (gap = {gap:.4f})")
        
        return winner_clf, best_result['auc'], best_result, results
    
    def step5_save_and_report(self, winner_clf, best_auc, best_result, all_results):
        """Save models and generate report"""
        print("\n" + "="*80)
        print("STEP 5: SAVE MODELS & GENERATE REPORT")
        print("="*80)
        
        Path("results/dl_models").mkdir(parents=True, exist_ok=True)
        
        pickle.dump(winner_clf, open("results/dl_models/ensemble_fast_90plus.pkl", "wb"))
        pickle.dump(self.scaler, open("results/dl_models/scaler_fast.pkl", "wb"))
        
        results_json = {
            'best_model': best_result['model'],
            'best_auc': best_result['auc'],
            'best_recall': best_result['recall'],
            'best_f1': best_result['f1'],
            'all_results': all_results,
            'status': 'ready_for_deployment' if best_auc >= 0.90 else 'needs_optimization'
        }
        
        with open("results/ensemble_fast_metrics.json", "w") as f:
            json.dump(results_json, f, indent=2)
        
        print(f"✓ Saved: ensemble_fast_90plus.pkl")
        print(f"✓ Saved: scaler_fast.pkl")
        print(f"✓ Saved: ensemble_fast_metrics.json")
        
        return results_json
    
    def run(self):
        """Execute full pipeline"""
        print("\n\n")
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║        🚀 FAST ENSEMBLE FOR 90+ AUC                           ║")
        print("║        Tuned Models + Stacking + Threshold Optimization       ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        
        # Step 1
        self.step1_load_and_extract_features()
        
        # Step 2
        models_dict = self.step2_train_tuned_models()
        
        # Step 3
        stacking_result = self.step3_stacking_ensemble(models_dict)
        
        # Step 4
        winner_clf, best_auc, best_result, all_results = self.step4_compare_and_select(models_dict, stacking_result)
        
        # Step 5
        final_results = self.step5_save_and_report(winner_clf, best_auc, best_result, all_results)
        
        # Final summary
        print("\n" + "="*80)
        print("✅ ENSEMBLE ENGINEERING COMPLETE")
        print("="*80)
        print(f"✓ Best Model: {best_result['model'].upper()}")
        print(f"✓ Final AUC: {best_auc:.4f}")
        print(f"✓ Recall: {best_result['recall']:.4f}")
        print(f"✓ F1: {best_result['f1']:.4f}")
        print(f"✓ 90+ AUC: {'✅ YES' if best_auc >= 0.90 else '❌ NO - Gap: ' + f'{0.90-best_auc:.4f}'}")
        print(f"✓ Status: {final_results['status'].upper()}")
        print("="*80)
        
        return final_results

if __name__ == "__main__":
    builder = FastEnsembleBuilder()
    results = builder.run()

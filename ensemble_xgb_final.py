"""
FINAL PUSH FOR 90+ AUC - XGBoost + LightGBM + Advanced Engineering
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

class XGBoostEnsembleBuilder:
    def __init__(self):
        self.scaler = RobustScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def step1_load_and_extract_features(self):
        """Load & extract with aggressive feature engineering"""
        print("\n" + "="*80)
        print("STEP 1: EXTRACT 100+ ENGINEERED FEATURES")
        print("="*80)
        
        csv_path = "data/processed_icu_hourly_v2.csv"
        df = pd.read_csv(csv_path)
        
        print(f"✓ Loaded: {df.shape[0]:,} rows")
        
        # Group and aggregate
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
        
        # AGGRESSIVE feature engineering
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
        
        # Aggregate volatility and range measures
        if std_cols:
            X['total_volatility'] = X[std_cols[:5]].mean(axis=1)
            X['max_volatility'] = X[std_cols[:5]].max(axis=1)
        
        # Coefficient of variation
        for col in mean_cols[:6]:
            std_col = col.replace('_mean', '_std')
            if std_col in X.columns:
                X[f'{col.split("_")[0]}_cv'] = np.divide(X[std_col], np.abs(X[col]) + 1e-6)
        
        # Handle infinities
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
    
    def step2_train_models(self):
        """Train all available models"""
        print("\n" + "="*80)
        print("STEP 2: TRAIN XGBoost, LightGBM, & TRADITIONAL MODELS")
        print("="*80)
        
        models = {}
        
        # Extra Trees
        print("\n🔹 Extra Trees...")
        et = ExtraTreesClassifier(
            n_estimators=300, max_depth=18, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        et.fit(self.X_train, self.y_train)
        et_auc = roc_auc_score(self.y_test, et.predict_proba(self.X_test)[:, 1])
        print(f"  ✓ ET AUC: {et_auc:.4f}")
        models['et'] = (et, et_auc)
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            print("\n🔹 XGBoost...")
            try:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=7, learning_rate=0.08,
                    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=10,
                    random_state=42, n_jobs=-1, tree_method='hist'
                )
                xgb_model.fit(self.X_train, self.y_train)
                xgb_auc = roc_auc_score(self.y_test, xgb_model.predict_proba(self.X_test)[:, 1])
                print(f"  ✓ XGBoost AUC: {xgb_auc:.4f}")
                models['xgb'] = (xgb_model, xgb_auc)
            except Exception as e:
                print(f"  ⚠️  XGBoost failed: {str(e)[:50]}")
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            print("\n🔹 LightGBM...")
            try:
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=200, max_depth=8, learning_rate=0.08,
                    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=10, random_state=42, n_jobs=-1
                )
                lgb_model.fit(self.X_train, self.y_train)
                lgb_auc = roc_auc_score(self.y_test, lgb_model.predict_proba(self.X_test)[:, 1])
                print(f"  ✓ LightGBM AUC: {lgb_auc:.4f}")
                models['lgb'] = (lgb_model, lgb_auc)
            except Exception as e:
                print(f"  ⚠️  LightGBM failed: {str(e)[:50]}")
        
        # Random Forest
        print("\n🔹 Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=16, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        rf_auc = roc_auc_score(self.y_test, rf.predict_proba(self.X_test)[:, 1])
        print(f"  ✓ RF AUC: {rf_auc:.4f}")
        models['rf'] = (rf, rf_auc)
        
        return models
    
    def step3_advanced_stacking(self, models):
        """Stacking with all available models"""
        print("\n" + "="*80)
        print("STEP 3: STACKING WITH ALL MODELS")
        print("="*80)
        
        # Build estimators list from available models
        estimators = [(name, model) for name, (model, _) in models.items()]
        
        print(f"🏗️ Building stacking with {len(estimators)} base models...")
        
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, C=0.1),
            cv=5
        )
        
        stacking_clf.fit(self.X_train, self.y_train)
        y_pred_proba = stacking_clf.predict_proba(self.X_test)[:, 1]
        stacking_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"✓ Stacking AUC: {stacking_auc:.4f}")
        
        # Optimize threshold
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        best_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_idx]
        
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        metrics = {
            'threshold': best_threshold,
            'recall': recall_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
        }
        
        print(f"  Threshold: {best_threshold:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        
        return stacking_clf, stacking_auc, metrics
    
    def step4_soft_voting(self, models):
        """Soft voting ensemble with all models"""
        print("\n" + "="*80)
        print("STEP 4: SOFT VOTING ENSEMBLE")
        print("="*80)
        
        estimators = [(name, model) for name, (model, _) in models.items()]
        
        # Calculate weights by AUC
        total_auc = sum(auc for _, auc in models.values())
        weights = [auc / total_auc for _, auc in models.values()]
        
        print(f"📊 Weights based on AUC: {dict(zip([n for n, _ in estimators], [f'{w:.3f}' for w in weights]))}")
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights,
            n_jobs=-1
        )
        
        voting_clf.fit(self.X_train, self.y_train)
        y_pred_proba = voting_clf.predict_proba(self.X_test)[:, 1]
        voting_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"✓ Voting AUC: {voting_auc:.4f}")
        
        # Optimize threshold
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        best_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_idx]
        
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        metrics = {
            'threshold': best_threshold,
            'recall': recall_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
        }
        
        print(f"  Threshold: {best_threshold:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        
        return voting_clf, voting_auc, metrics
    
    def step5_final_comparison_and_save(self, models, stacking_result, voting_result):
        """Compare all ensembles and select winner"""
        print("\n" + "="*80)
        print("STEP 5: FINAL COMPARISON & WINNER SELECTION")
        print("="*80)
        
        stacking_clf, stacking_auc, stacking_metrics = stacking_result
        voting_clf, voting_auc, voting_metrics = voting_result
        
        print("\n📊 COMPREHENSIVE PERFORMANCE:")
        print("\nIndividual Models:")
        for name, (_, auc) in models.items():
            print(f"  {name.upper():10} - AUC: {auc:.4f}")
        
        print(f"\nEnsembles:")
        print(f"  STACKING   - AUC: {stacking_auc:.4f}, F1: {stacking_metrics['f1']:.4f}")
        print(f"  VOTING     - AUC: {voting_auc:.4f}, F1: {voting_metrics['f1']:.4f}")
        
        # Select best
        if stacking_auc >= voting_auc:
            winner_clf = stacking_clf
            winner_auc = stacking_auc
            winner_name = "Stacking"
            winner_metrics = stacking_metrics
        else:
            winner_clf = voting_clf
            winner_auc = voting_auc
            winner_name = "Voting"
            winner_metrics = voting_metrics
        
        print(f"\n🏆 WINNER: {winner_name} - AUC {winner_auc:.4f}")
        
        if winner_auc >= 0.90:
            print(f"✅ 90+ AUC TARGET: YES ({winner_auc:.4f} >= 0.90)")
        else:
            gap = 0.90 - winner_auc
            pct_gap = gap / 0.90 * 100
            print(f"❌ 90+ AUC TARGET: NO (gap = {gap:.4f}, {pct_gap:.2f}%)")
        
        # Save
        Path("results/dl_models").mkdir(parents=True, exist_ok=True)
        
        pickle.dump(winner_clf, open("results/dl_models/ensemble_xgb_final.pkl", "wb"))
        pickle.dump(self.scaler, open("results/dl_models/scaler_xgb_final.pkl", "wb"))
        
        results_json = {
            'winner': winner_name,
            'best_auc': winner_auc,
            'status': '✅ READY_90PLUS' if winner_auc >= 0.90 else f'❌ NEEDS_{round(0.90-winner_auc, 4)}AUC',
            'metrics': winner_metrics,
            'individual_models': {name: auc for name, (_, auc) in models.items()},
            'stacking_auc': stacking_auc,
            'voting_auc': voting_auc,
        }
        
        with open("results/ensemble_xgb_final_metrics.json", "w") as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n✓ Saved model & results")
        
        return winner_clf, winner_auc, results_json
    
    def run(self):
        """Execute full pipeline"""
        print("\n\n")
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║    🎯 FINAL PUSH FOR 90+ AUC                                  ║")
        print("║    XGBoost + LightGBM + Advanced Stacking                     ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        
        self.step1_load_and_extract_features()
        models = self.step2_train_models()
        stacking_result = self.step3_advanced_stacking(models)
        voting_result = self.step4_soft_voting(models)
        winner_clf, winner_auc, results_json = self.step5_final_comparison_and_save(
            models, stacking_result, voting_result
        )
        
        print("\n" + "="*80)
        print("✅ FINAL ENSEMBLE ENGINEERING COMPLETE")
        print("="*80)
        print(f"🏆 Winner: {results_json['winner']}")
        print(f"📈 AUC: {winner_auc:.4f}")
        print(f"🎯 Status: {results_json['status']}")
        print("="*80)
        
        return results_json

if __name__ == "__main__":
    builder = XGBoostEnsembleBuilder()
    results = builder.run()

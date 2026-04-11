"""
ENSEMBLE STACKING MODEL FOR IMPROVED PREDICTIONS

Combines multiple base models into a meta-learner:
1. HistGradientBoosting (proven best)
2. RandomForest (traditional baseline)
3. XGBoost (gradient boosting alternative)
4. LogisticRegression meta-learner (combines predictions)

Literature shows: +2-4% AUC improvement over single best model
"""

import os
import json
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import pickle

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')

def main():
    """Execute ensemble stacking pipeline"""
    
    print("="*80)
    print("ENSEMBLE STACKING: COMBINE MULTIPLE MODELS")
    print("="*80)
    
    # Step 1: Load feature matrix
    print("\n[STEP 1] Loading feature matrix...")
    enhanced_df = pd.read_csv('results/trajectory_features/combined_features_with_trajectory.csv')
    
    feature_cols = [c for c in enhanced_df.columns 
                   if c not in ['patientunitstayid', 'mortality']]
    
    X = enhanced_df[feature_cols].values
    y = enhanced_df['mortality'].values
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"✓ Loaded {X.shape[0]} patients with {X.shape[1]} features")
    
    # Step 2: Split data
    print("\n[STEP 2] Creating train/test/validation split...")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    print(f"✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Val: {X_val.shape[0]}")
    
    # Step 3: Scale features
    print("\n[STEP 3] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    print("✓ Features scaled")
    
    # Step 4: Train base models
    print("\n[STEP 4] Training base models...")
    
    base_models = {}
    base_predictions_train = {}
    
    # Model 1: HistGradientBoosting (best from hyperparameter optimization)
    print("  Training HistGradientBoosting...")
    hgb = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        max_depth=7,
        min_samples_leaf=5,
        l2_regularization=0.1,
        random_state=42
    )
    hgb.fit(X_train_scaled, y_train)
    base_models['hgb'] = hgb
    base_predictions_train['hgb_proba'] = hgb.predict_proba(X_train_scaled)[:, 1]
    print(f"    Train AUC: {roc_auc_score(y_train, base_predictions_train['hgb_proba']):.4f}")
    
    # Model 2: RandomForest (optimized)
    print("  Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='log2',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    base_models['rf'] = rf
    base_predictions_train['rf_proba'] = rf.predict_proba(X_train_scaled)[:, 1]
    print(f"    Train AUC: {roc_auc_score(y_train, base_predictions_train['rf_proba']):.4f}")
    
    # Model 3: XGBoost (gradient boosting alternative)
    if XGBOOST_AVAILABLE:
        print("  Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        xgb.fit(X_train_scaled, y_train)
        base_models['xgb'] = xgb
        base_predictions_train['xgb_proba'] = xgb.predict_proba(X_train_scaled)[:, 1]
        print(f"    Train AUC: {roc_auc_score(y_train, base_predictions_train['xgb_proba']):.4f}")
    else:
        print("  XGBoost not available, skipping")
    
    # Step 5: Create meta-features for meta-learner
    print("\n[STEP 5] Creating meta-features from base model predictions...")
    
    X_train_meta = np.column_stack([
        base_predictions_train['hgb_proba'],
        base_predictions_train['rf_proba']
    ])
    
    if XGBOOST_AVAILABLE:
        X_train_meta = np.column_stack([
            X_train_meta,
            base_predictions_train['xgb_proba']
        ])
    
    print(f"✓ Meta-features shape: {X_train_meta.shape}")
    
    # Step 6: Train meta-learner
    print("\n[STEP 6] Training meta-learner (LogisticRegression)...")
    
    meta_learner = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0
    )
    meta_learner.fit(X_train_meta, y_train)
    
    # Evaluate meta-learner on training
    y_pred_meta_train = meta_learner.predict_proba(X_train_meta)[:, 1]
    auc_meta_train = roc_auc_score(y_train, y_pred_meta_train)
    print(f"  Meta-learner train AUC: {auc_meta_train:.4f}")
    
    # Step 7: Evaluate ensemble on test set
    print("\n[STEP 7] Evaluating ensemble on TEST set...")
    
    # Get base model predictions on test
    hgb_test = hgb.predict_proba(X_test_scaled)[:, 1]
    rf_test = rf.predict_proba(X_test_scaled)[:, 1]
    
    test_results = {
        'HistGradientBoosting': {'proba': hgb_test},
        'RandomForest': {'proba': rf_test}
    }
    
    if XGBOOST_AVAILABLE:
        xgb_test = xgb.predict_proba(X_test_scaled)[:, 1]
        test_results['XGBoost'] = {'proba': xgb_test}
    
    # Create meta-features for test
    X_test_meta = np.column_stack([hgb_test, rf_test])
    if XGBOOST_AVAILABLE:
        X_test_meta = np.column_stack([X_test_meta, xgb_test])
    
    # Ensemble prediction
    ensemble_test = meta_learner.predict_proba(X_test_meta)[:, 1]
    test_results['Ensemble'] = {'proba': ensemble_test}
    
    # Evaluate all models
    print("\nTEST SET PERFORMANCE:")
    for model_name, result in test_results.items():
        auc = roc_auc_score(y_test, result['proba'])
        result['auc'] = auc
        
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(y_test, result['proba'])
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        result['optimal_threshold'] = optimal_threshold
        
        # Evaluate at threshold
        y_pred = (result['proba'] >= optimal_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        result['sensitivity'] = sensitivity
        result['specificity'] = specificity
        result['tp'] = int(tp)
        result['fn'] = int(fn)
        
        print(f"\n  {model_name}:")
        print(f"    AUC: {auc:.4f}")
        print(f"    Threshold: {optimal_threshold:.4f}")
        print(f"    Sensitivity: {sensitivity:.4f} ({tp}/{tp+fn})")
        print(f"    Specificity: {specificity:.4f}")
    
    # Step 8: Validation evaluation
    print("\n[STEP 8] Evaluating on VALIDATION set...")
    
    hgb_val = hgb.predict_proba(X_val_scaled)[:, 1]
    rf_val = rf.predict_proba(X_val_scaled)[:, 1]
    
    X_val_meta = np.column_stack([hgb_val, rf_val])
    if XGBOOST_AVAILABLE:
        xgb_val = xgb.predict_proba(X_val_scaled)[:, 1]
        X_val_meta = np.column_stack([X_val_meta, xgb_val])
    
    ensemble_val = meta_learner.predict_proba(X_val_meta)[:, 1]
    auc_ensemble_val = roc_auc_score(y_val, ensemble_val)
    
    print(f"\nVALIDATION SET:")
    print(f"  Ensemble AUC: {auc_ensemble_val:.4f}")
    
    # Step 9: Summary comparison
    print("\n" + "="*80)
    print("ENSEMBLE STACKING SUMMARY")
    print("="*80)
    
    baseline_auc = 0.8712  # Our best single model (HistGB)
    ensemble_auc = test_results['Ensemble']['auc']
    improvement = (ensemble_auc - baseline_auc) * 100
    
    print(f"\nPerformance Comparison:")
    print(f"  Baseline (best single model): {baseline_auc:.4f}")
    print(f"  Ensemble (stacked): {ensemble_auc:.4f}")
    print(f"  Improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print(f"\n✅ ENSEMBLE WINS! Consider using ensemble for deployment")
    else:
        print(f"\n⚠️ BASELINE WINS! Single model may be better (simpler, faster)")
    
    # Step 10: Save models and results
    print("\n[STEP 9] Saving ensemble models and results...")
    
    output_dir = Path('results/ensemble')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    with open(output_dir / 'base_models.pkl', 'wb') as f:
        pickle.dump(base_models, f)
    
    with open(output_dir / 'meta_learner.pkl', 'wb') as f:
        pickle.dump(meta_learner, f)
    
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save results
    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'ensemble_strategy': 'Stacking (3 base models + LogisticRegression meta-learner)',
        'base_models': list(base_models.keys()),
        'meta_learner': 'LogisticRegression',
        'features': len(feature_cols),
        'test_results': {
            k: {
                'auc': float(v['auc']),
                'sensitivity': float(v['sensitivity']),
                'specificity': float(v['specificity']),
                'optimal_threshold': float(v['optimal_threshold'])
            } for k, v in test_results.items()
        },
        'validation_auc_ensemble': float(auc_ensemble_val),
        'improvement_vs_baseline': f"{improvement:+.2f}%",
        'recommendation': 'Use ensemble if improvement significant, otherwise stick with best single model'
    }
    
    with open(output_dir / 'ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved to: {output_dir}")
    print(f"  - base_models.pkl")
    print(f"  - meta_learner.pkl")
    print(f"  - scaler.pkl")
    print(f"  - ensemble_results.json")
    
    return test_results, base_models, meta_learner, scaler


if __name__ == '__main__':
    results, models, meta, scaler = main()
    print("\n✨ Ensemble stacking COMPLETE!")

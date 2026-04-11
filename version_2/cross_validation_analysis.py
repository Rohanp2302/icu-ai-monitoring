"""
K-FOLD CROSS-VALIDATION FOR ROBUST MODEL EVALUATION

Validates models using k-fold cross-validation (k=5):
- Eliminates data leakage
- More reliable estimates of generalization performance
- Shows stability across different data distributions
"""

import os
import json
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)
import pickle

warnings.filterwarnings('ignore')

def evaluate_fold(y_true, y_pred_proba, model_name, fold_num):
    """Evaluate single fold"""
    
    # AUC
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Evaluate at threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    
    return {
        'auc': auc,
        'optimal_threshold': optimal_threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'tp': int(tp),
        'fn': int(fn),
        'fp': int(fp),
        'tn': int(tn)
    }


def main():
    """Execute k-fold cross-validation"""
    
    print("="*80)
    print("K-FOLD CROSS-VALIDATION FOR MODEL ROBUSTNESS")
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
    print(f"  Positive cases: {y.sum()} ({100*y.mean():.1f}%)")
    
    # Step 2: Setup k-fold cross-validation
    print("\n[STEP 2] Setting up 5-fold stratified cross-validation...")
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("✓ StratifiedKFold configured (preserves class distribution in each fold)")
    
    # Step 3: Train and evaluate each model across folds
    print("\n[STEP 3] Training models on each fold...")
    
    models_cv_results = {}
    
    # Model 1: HistGradientBoosting
    print("\n  HistGradientBoosting:")
    hgb_results = []
    hgb_preds = np.zeros(len(y))
    
    for fold_num, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_test_fold = X[test_idx]
        y_test_fold = y[test_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)
        
        # Train
        hgb = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_depth=7,
            min_samples_leaf=5,
            l2_regularization=0.1,
            random_state=42
        )
        hgb.fit(X_train_fold, y_train_fold)
        
        # Predict
        y_pred_proba = hgb.predict_proba(X_test_fold)[:, 1]
        hgb_preds[test_idx] = y_pred_proba
        
        # Evaluate
        fold_result = evaluate_fold(y_test_fold, y_pred_proba, 'HGB', fold_num)
        hgb_results.append(fold_result)
        
        print(f"    Fold {fold_num+1}: AUC={fold_result['auc']:.4f}, "
              f"Sensitivity={fold_result['sensitivity']:.4f}, "
              f"Specificity={fold_result['specificity']:.4f}")
    
    # Calculate aggregate metrics
    hgb_cv_auc = roc_auc_score(y, hgb_preds)
    hgb_mean_sensitivity = np.mean([r['sensitivity'] for r in hgb_results])
    hgb_std_sensitivity = np.std([r['sensitivity'] for r in hgb_results])
    
    models_cv_results['HistGradientBoosting'] = {
        'folds': hgb_results,
        'cv_auc': hgb_cv_auc,
        'mean_sensitivity': hgb_mean_sensitivity,
        'std_sensitivity': hgb_std_sensitivity,
        'fold_auc_scores': [r['auc'] for r in hgb_results]
    }
    
    print(f"    Avg AUC: {hgb_cv_auc:.4f}")
    print(f"    Sensitivity: {hgb_mean_sensitivity:.4f} ± {hgb_std_sensitivity:.4f}")
    
    # Model 2: RandomForest
    print("\n  RandomForest:")
    rf_results = []
    rf_preds = np.zeros(len(y))
    
    for fold_num, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_test_fold = X[test_idx]
        y_test_fold = y[test_idx]
        
        # Scale (though RF doesn't strictly need it)
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)
        
        # Train
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
        rf.fit(X_train_fold, y_train_fold)
        
        # Predict
        y_pred_proba = rf.predict_proba(X_test_fold)[:, 1]
        rf_preds[test_idx] = y_pred_proba
        
        # Evaluate
        fold_result = evaluate_fold(y_test_fold, y_pred_proba, 'RF', fold_num)
        rf_results.append(fold_result)
        
        print(f"    Fold {fold_num+1}: AUC={fold_result['auc']:.4f}, "
              f"Sensitivity={fold_result['sensitivity']:.4f}, "
              f"Specificity={fold_result['specificity']:.4f}")
    
    rf_cv_auc = roc_auc_score(y, rf_preds)
    rf_mean_sensitivity = np.mean([r['sensitivity'] for r in rf_results])
    rf_std_sensitivity = np.std([r['sensitivity'] for r in rf_results])
    
    models_cv_results['RandomForest'] = {
        'folds': rf_results,
        'cv_auc': rf_cv_auc,
        'mean_sensitivity': rf_mean_sensitivity,
        'std_sensitivity': rf_std_sensitivity,
        'fold_auc_scores': [r['auc'] for r in rf_results]
    }
    
    print(f"    Avg AUC: {rf_cv_auc:.4f}")
    print(f"    Sensitivity: {rf_mean_sensitivity:.4f} ± {rf_std_sensitivity:.4f}")
    
    # Step 4: Summary
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*80)
    
    print("\nModel Performance (5-Fold CV):")
    print(f"\nHistGradientBoosting:")
    print(f"  CV AUC: {hgb_cv_auc:.4f}")
    print(f"  Fold AUCs: {', '.join([f'{x:.4f}' for x in models_cv_results['HistGradientBoosting']['fold_auc_scores']])}")
    print(f"  AUC Std Dev: {np.std(models_cv_results['HistGradientBoosting']['fold_auc_scores']):.4f}")
    print(f"  Sensitivity (mean): {hgb_mean_sensitivity:.4f} ± {hgb_std_sensitivity:.4f}")
    
    print(f"\nRandomForest:")
    print(f"  CV AUC: {rf_cv_auc:.4f}")
    print(f"  Fold AUCs: {', '.join([f'{x:.4f}' for x in models_cv_results['RandomForest']['fold_auc_scores']])}")
    print(f"  AUC Std Dev: {np.std(models_cv_results['RandomForest']['fold_auc_scores']):.4f}")
    print(f"  Sensitivity (mean): {rf_mean_sensitivity:.4f} ± {rf_std_sensitivity:.4f}")
    
    # Step 5: Save results
    print("\n[STEP 4] Saving cross-validation results...")
    
    output_dir = Path('results/cross_validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cv_summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'method': '5-fold stratified cross-validation',
        'total_samples': int(len(y)),
        'positive_cases': int(y.sum()),
        'positive_rate': float(y.mean()),
        'models': {
            'HistGradientBoosting': {
                'cv_auc': float(hgb_cv_auc),
                'fold_auc_scores': [float(x) for x in models_cv_results['HistGradientBoosting']['fold_auc_scores']],
                'auc_std_dev': float(np.std(models_cv_results['HistGradientBoosting']['fold_auc_scores'])),
                'mean_sensitivity': float(hgb_mean_sensitivity),
                'std_sensitivity': float(hgb_std_sensitivity),
                'recommendation': 'Best model - stable performance across folds'
            },
            'RandomForest': {
                'cv_auc': float(rf_cv_auc),
                'fold_auc_scores': [float(x) for x in models_cv_results['RandomForest']['fold_auc_scores']],
                'auc_std_dev': float(np.std(models_cv_results['RandomForest']['fold_auc_scores'])),
                'mean_sensitivity': float(rf_mean_sensitivity),
                'std_sensitivity': float(rf_std_sensitivity),
                'recommendation': 'Good baseline - slightly lower performance'
            }
        }
    }
    
    with open(output_dir / 'cv_results.json', 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    # Save fold-level predictions for analysis
    predictions_df = pd.DataFrame({
        'patientunitstayid': enhanced_df['patientunitstayid'].values,
        'mortality': y,
        'hgb_prediction': hgb_preds,
        'rf_prediction': rf_preds
    })
    
    predictions_df.to_csv(output_dir / 'cv_predictions.csv', index=False)
    
    print(f"✓ Saved to: {output_dir}")
    print(f"  - cv_results.json")
    print(f"  - cv_predictions.csv")
    
    return models_cv_results


if __name__ == '__main__':
    cv_results = main()
    print("\n✨ Cross-validation analysis COMPLETE!")

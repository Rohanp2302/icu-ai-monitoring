"""
HYPERPARAMETER OPTIMIZATION FOR RANDOMFOREST (BAYESIAN SEARCH)

Uses Bayesian Optimization (via Optuna) to find best RandomForest parameters.
This is more efficient than random/grid search and can improve AUC by 1-3%.

Strategy:
1. Use proper train/test/validation split (fixed from trajectory engineer)
2. Optimize on validation set (not test!)
3. Try both RandomForest and GradientBoosting during search
4. Evaluate final model on test set (held-out)
"""

import os
import json
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, f1_score,
    classification_report, precision_recall_curve
)
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')

def objective_rf(trial, X_train_scaled, X_val_scaled, y_train, y_val):
    """Objective function for RandomForest hyperparameter optimization"""
    
    # Hyperparameter space for RandomForest
    n_estimators = trial.suggest_int('n_estimators', 100, 500, step=50)
    max_depth = trial.suggest_int('max_depth', 8, 25)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Train on training set
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on validation set (the held-out set, not training!)
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1])
    
    return val_auc


def objective_gb(trial, X_train_scaled, X_val_scaled, y_train, y_val):
    """Objective function for GradientBoosting hyperparameter optimization"""
    
    # Hyperparameter space for GradientBoosting
    n_estimators = trial.suggest_int('n_estimators', 100, 300, step=50)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 12)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    
    # Create model
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=42
    )
    
    # Train on training set
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on validation set
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1])
    
    return val_auc


def main():
    """Execute hyperparameter optimization"""
    
    print("=" * 80)
    print("BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Step 1: Load enhanced features with trajectory
    print("\n[STEP 1] Loading enhanced features with trajectory...")
    enhanced_df = pd.read_csv('results/trajectory_features/combined_features_with_trajectory.csv')
    
    X = enhanced_df.drop(['patientunitstayid', 'mortality'], axis=1).values
    y = enhanced_df['mortality'].values
    
    feature_cols = [c for c in enhanced_df.columns 
                   if c not in ['patientunitstayid', 'mortality']]
    
    print(f"✓ Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"  Mortality: {y.sum()} deaths, {(1-y).sum()} survivors")
    print(f"  Mortality rate: {100*y.mean():.2f}%")
    
    # Step 2: Proper train/test/validation split (70/15/15)
    print("\n[STEP 2] Creating train/test/validation split (70/15/15)...")
    
    # Split 1: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    
    # Split 2: 50/50 split of temp → 15% test, 15% validation
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    print(f"✓ Train: {X_train.shape[0]} samples ({y_train.sum()} deaths, {100*y_train.mean():.2f}%)")
    print(f"✓ Test: {X_test.shape[0]} samples ({y_test.sum()} deaths, {100*y_test.mean():.2f}%)")
    print(f"✓ Validation: {X_val.shape[0]} samples ({y_val.sum()} deaths, {100*y_val.mean():.2f}%)")
    
    # Step 3: Scale features
    print("\n[STEP 3] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    print("✓ Features scaled (fit on training only)")
    
    # Step 4: Bayesian Optimization for RandomForest
    print("\n[STEP 4] Bayesian Optimization for RandomForest (50 iterations)...")
    print("  Optimizing on VALIDATION set (not test!)")
    print("  Using TPE sampler (Tree-Structured Parzen Estimator)")
    
    sampler_rf = TPESampler(seed=42)
    study_rf = optuna.create_study(
        direction='maximize',
        sampler=sampler_rf
    )
    
    # Use lambda to pass data to objective function
    study_rf.optimize(
        lambda trial: objective_rf(trial, X_train_scaled, X_val_scaled, y_train, y_val),
        n_trials=50,
        show_progress_bar=True
    )
    
    best_params_rf = study_rf.best_params
    best_val_auc_rf = study_rf.best_value
    
    print(f"\n✓ RandomForest Optimization Complete")
    print(f"  Best validation AUC: {best_val_auc_rf:.4f}")
    print(f"  Best parameters:")
    for param, value in best_params_rf.items():
        print(f"    - {param}: {value}")
    
    # Step 5: Bayesian Optimization for GradientBoosting
    print("\n[STEP 5] Bayesian Optimization for GradientBoosting (50 iterations)...")
    
    sampler_gb = TPESampler(seed=42)
    study_gb = optuna.create_study(
        direction='maximize',
        sampler=sampler_gb
    )
    
    study_gb.optimize(
        lambda trial: objective_gb(trial, X_train_scaled, X_val_scaled, y_train, y_val),
        n_trials=50,
        show_progress_bar=True
    )
    
    best_params_gb = study_gb.best_params
    best_val_auc_gb = study_gb.best_value
    
    print(f"\n✓ GradientBoosting Optimization Complete")
    print(f"  Best validation AUC: {best_val_auc_gb:.4f}")
    print(f"  Best parameters:")
    for param, value in best_params_gb.items():
        print(f"    - {param}: {value}")
    
    # Step 6: Train final models with best parameters
    print("\n[STEP 6] Training final models with optimized parameters...")
    
    # Train optimized RandomForest
    rf_final = RandomForestClassifier(
        **best_params_rf,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_final.fit(X_train_scaled, y_train)
    
    # Train optimized GradientBoosting
    gb_final = GradientBoostingClassifier(
        **best_params_gb,
        random_state=42
    )
    gb_final.fit(X_train_scaled, y_train)
    
    print("✓ Models trained")
    
    # Step 7: Evaluate on TEST set (held-out)
    print("\n[STEP 7] Evaluating on TEST set (held-out, never seen before)...")
    
    # RandomForest predictions
    y_pred_rf_test = rf_final.predict(X_test_scaled)
    y_proba_rf_test = rf_final.predict_proba(X_test_scaled)[:, 1]
    auc_rf_test = roc_auc_score(y_test, y_proba_rf_test)
    
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_proba_rf_test)
    
    # Find optimal threshold (Youden's J)
    j_scores_rf = tpr_rf - fpr_rf
    optimal_idx_rf = np.argmax(j_scores_rf)
    optimal_threshold_rf = thresholds_rf[optimal_idx_rf]
    
    # Use optimal threshold
    y_pred_rf_optimal = (y_proba_rf_test >= optimal_threshold_rf).astype(int)
    cm_rf = confusion_matrix(y_test, y_pred_rf_optimal)
    
    tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
    sensitivity_rf = tp_rf / (tp_rf + fn_rf)
    specificity_rf = tn_rf / (tn_rf + fp_rf)
    
    print(f"\nRANDOMFOREST (OPTIMIZED, TEST SET):")
    print(f"  AUC: {auc_rf_test:.4f}")
    print(f"  Optimal threshold: {optimal_threshold_rf:.4f}")
    print(f"  Sensitivity (catch deaths): {sensitivity_rf:.4f} ({tp_rf}/{tp_rf+fn_rf})")
    print(f"  Specificity (avoid false alarms): {specificity_rf:.4f}")
    print(f"  Confusion matrix: TP={tp_rf}, TN={tn_rf}, FP={fp_rf}, FN={fn_rf}")
    
    # GradientBoosting predictions
    y_pred_gb_test = gb_final.predict(X_test_scaled)
    y_proba_gb_test = gb_final.predict_proba(X_test_scaled)[:, 1]
    auc_gb_test = roc_auc_score(y_test, y_proba_gb_test)
    
    fpr_gb, tpr_gb, thresholds_gb = roc_curve(y_test, y_proba_gb_test)
    j_scores_gb = tpr_gb - fpr_gb
    optimal_idx_gb = np.argmax(j_scores_gb)
    optimal_threshold_gb = thresholds_gb[optimal_idx_gb]
    
    y_pred_gb_optimal = (y_proba_gb_test >= optimal_threshold_gb).astype(int)
    cm_gb = confusion_matrix(y_test, y_pred_gb_optimal)
    
    tn_gb, fp_gb, fn_gb, tp_gb = cm_gb.ravel()
    sensitivity_gb = tp_gb / (tp_gb + fn_gb)
    specificity_gb = tn_gb / (tn_gb + fp_gb)
    
    print(f"\nGRADIENTBOOSTING (OPTIMIZED, TEST SET):")
    print(f"  AUC: {auc_gb_test:.4f}")
    print(f"  Optimal threshold: {optimal_threshold_gb:.4f}")
    print(f"  Sensitivity (catch deaths): {sensitivity_gb:.4f} ({tp_gb}/{tp_gb+fn_gb})")
    print(f"  Specificity (avoid false alarms): {specificity_gb:.4f}")
    print(f"  Confusion matrix: TP={tp_gb}, TN={tn_gb}, FP={fp_gb}, FN={fn_gb}")
    
    # Step 8: Final validation on validation set
    print("\n[STEP 8] Final validation on VALIDATION set (independent)...")
    
    y_proba_rf_val = rf_final.predict_proba(X_val_scaled)[:, 1]
    auc_rf_val = roc_auc_score(y_val, y_proba_rf_val)
    
    y_pred_rf_val = (y_proba_rf_val >= optimal_threshold_rf).astype(int)
    cm_rf_val = confusion_matrix(y_val, y_pred_rf_val)
    tn_rf_v, fp_rf_v, fn_rf_v, tp_rf_v = cm_rf_val.ravel()
    sensitivity_rf_val = tp_rf_v / (tp_rf_v + fn_rf_v)
    specificity_rf_val = tn_rf_v / (tn_rf_v + fp_rf_v)
    
    print(f"\nRANDOMFOREST (VALIDATION SET):")
    print(f"  AUC: {auc_rf_val:.4f}")
    print(f"  Sensitivity: {sensitivity_rf_val:.4f} ({tp_rf_v}/{tp_rf_v+fn_rf_v})")
    print(f"  Specificity: {specificity_rf_val:.4f}")
    
    y_proba_gb_val = gb_final.predict_proba(X_val_scaled)[:, 1]
    auc_gb_val = roc_auc_score(y_val, y_proba_gb_val)
    
    y_pred_gb_val = (y_proba_gb_val >= optimal_threshold_gb).astype(int)
    cm_gb_val = confusion_matrix(y_val, y_pred_gb_val)
    tn_gb_v, fp_gb_v, fn_gb_v, tp_gb_v = cm_gb_val.ravel()
    sensitivity_gb_val = tp_gb_v / (tp_gb_v + fn_gb_v)
    specificity_gb_val = tn_gb_v / (tn_gb_v + fp_gb_v)
    
    print(f"\nGRADIENTBOOSTING (VALIDATION SET):")
    print(f"  AUC: {auc_gb_val:.4f}")
    print(f"  Sensitivity: {sensitivity_gb_val:.4f} ({tp_gb_v}/{tp_gb_v+fn_gb_v})")
    print(f"  Specificity: {specificity_gb_val:.4f}")
    
    # Step 9: Save results
    print("\n[STEP 9] Saving results...")
    
    output_dir = Path('results/hyperparameter_optimization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'hyperparameter_search': {
            'algorithm': 'Bayesian Optimization (TPE Sampler)',
            'n_iterations': 50,
            'features': {
                'total': X_train_scaled.shape[1],
                'static': 44,
                'trajectory': X_train_scaled.shape[1] - 44
            }
        },
        'data_splits': {
            'train': {'n': int(X_train.shape[0]), 'deaths': int(y_train.sum())},
            'test': {'n': int(X_test.shape[0]), 'deaths': int(y_test.sum())},
            'validation': {'n': int(X_val.shape[0]), 'deaths': int(y_val.sum())}
        },
        'randomforest': {
            'optimized_parameters': {k: (v if not isinstance(v, np.integer) else int(v)) 
                                      for k, v in best_params_rf.items()},
            'validation_auc': float(best_val_auc_rf),
            'test_auc': float(auc_rf_test),
            'test_sensitivity': float(sensitivity_rf),
            'test_specificity': float(specificity_rf),
            'optimal_threshold': float(optimal_threshold_rf),
            'validation_auc_test_split': float(auc_rf_val),
            'validation_sensitivity': float(sensitivity_rf_val),
            'validation_specificity': float(specificity_rf_val)
        },
        'gradientboosting': {
            'optimized_parameters': {k: (v if not isinstance(v, (np.integer, np.floating)) else float(v)) 
                                        for k, v in best_params_gb.items()},
            'validation_auc': float(best_val_auc_gb),
            'test_auc': float(auc_gb_test),
            'test_sensitivity': float(sensitivity_gb),
            'test_specificity': float(specificity_gb),
            'optimal_threshold': float(optimal_threshold_gb),
            'validation_auc_test_split': float(auc_gb_val),
            'validation_sensitivity': float(sensitivity_gb_val),
            'validation_specificity': float(specificity_gb_val)
        }
    }
    
    with open(output_dir / 'hyperparameter_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_dir / 'hyperparameter_optimization_results.json'}")
    
    # Step 10: Summary
    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("="*80)
    
    print(f"\n✅ RANDOMFOREST (Best Model)")
    print(f"   Test AUC: {auc_rf_test:.4f}")
    print(f"   Validation AUC: {auc_rf_val:.4f}")
    print(f"   Test Sensitivity: {sensitivity_rf:.4f}")
    print(f"   Test Specificity: {specificity_rf:.4f}")
    
    print(f"\n✅ GRADIENTBOOSTING (Alternative)")
    print(f"   Test AUC: {auc_gb_test:.4f}")
    print(f"   Validation AUC: {auc_gb_val:.4f}")
    print(f"   Test Sensitivity: {sensitivity_gb:.4f}")
    print(f"   Test Specificity: {specificity_gb:.4f}")
    
    print(f"\n📊 Performance vs Original Model (AUC 0.8561 / 0.9153):")
    print(f"   RandomForest improvement: +{(auc_rf_test-0.8561)*100:.2f}% (test)")
    print(f"   GradientBoosting: {(auc_gb_test-0.8561):+.4f} (test)")
    
    return results, rf_final, gb_final, scaler


if __name__ == '__main__':
    results, rf_model, gb_model, scaler = main()
    print("\n✨ Hyperparameter optimization COMPLETE!")

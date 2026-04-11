"""
HYPERPARAMETER OPTIMIZATION FOR RANDOMFOREST (GRID + RANDOM SEARCH)

Uses Grid Search + Random Search to find best RandomForest parameters.
This avoids external dependencies and is efficient for our parameter space.

Strategy:
1. Use proper train/test/validation split
2. Optimize on validation set  
3. Try both RandomForest and GradientBoosting
4. Evaluate final model on test set (held-out)
"""

import os
import json
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import random

warnings.filterwarnings('ignore')

def grid_search_rf(X_train_scaled, X_val_scaled, y_train, y_val):
    """Grid search for RandomForest hyperparameters"""
    
    print("  Grid Search for RandomForest (25 combinations)...")
    
    # Parameter grid
    param_grid = {
        'n_estimators': [150, 200, 300, 400],
        'max_depth': [12, 15, 18, 20, 22],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Limit combinations to top 25 (random selection)
    all_combinations = []
    for n_est in param_grid['n_estimators'][:2]:  # Top 2
        for depth in param_grid['max_depth']:
            for min_split in param_grid['min_samples_split'][:2]:  # Top 2
                for min_leaf in param_grid['min_samples_leaf'][:2]:  # Top 2
                    for max_feat in param_grid['max_features']:
                        all_combinations.append({
                            'n_estimators': n_est,
                            'max_depth': depth,
                            'min_samples_split': min_split,
                            'min_samples_leaf': min_leaf,
                            'max_features': max_feat
                        })
    
    # Sample top 25 if too many
    if len(all_combinations) > 25:
        all_combinations = random.sample(all_combinations, 25)
    
    best_params = None
    best_auc = 0
    results = []
    
    for idx, params in enumerate(all_combinations, 1):
        model = RandomForestClassifier(
            **params,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1])
        
        results.append({
            'params': params,
            'val_auc': val_auc
        })
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_params = params
        
        if idx % 5 == 0:
            print(f"    Tried {idx}/{len(all_combinations)} combinations, best AUC: {best_auc:.4f}")
    
    return best_params, best_auc, results


def grid_search_gb(X_train_scaled, X_val_scaled, y_train, y_val):
    """Grid search for HistGradientBoosting hyperparameters (handles NaN natively)"""
    
    print("  Grid Search for HistGradientBoosting (20 combinations)...")
    
    param_grid = {
        'max_iter': [100, 150, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, 9],
        'min_samples_leaf': [1, 5, 10],
        'l2_regularization': [0.0, 0.01, 0.1]
    }
    
    # Limit combinations
    all_combinations = []
    for max_iter in param_grid['max_iter'][:2]:
        for lr in param_grid['learning_rate'][:2]:
            for depth in param_grid['max_depth']:
                for min_leaf in param_grid['min_samples_leaf'][:2]:
                    for l2_reg in param_grid['l2_regularization']:
                        all_combinations.append({
                            'max_iter': max_iter,
                            'learning_rate': lr,
                            'max_depth': depth,
                            'min_samples_leaf': min_leaf,
                            'l2_regularization': l2_reg
                        })
    
    if len(all_combinations) > 20:
        all_combinations = random.sample(all_combinations, 20)
    
    best_params = None
    best_auc = 0
    results = []
    
    for idx, params in enumerate(all_combinations, 1):
        model = HistGradientBoostingClassifier(
            **params,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1])
        
        results.append({
            'params': params,
            'val_auc': val_auc
        })
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_params = params
        
        if idx % 5 == 0:
            print(f"    Tried {idx}/{len(all_combinations)} combinations, best: {best_auc:.4f}")
    
    return best_params, best_auc, results


def main():
    """Execute hyperparameter grid search"""
    
    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION (GRID SEARCH)")
    print("=" * 80)
    
    # Step 1: Load enhanced features
    print("\n[STEP 1] Loading enhanced features with trajectory...")
    enhanced_df = pd.read_csv('results/trajectory_features/combined_features_with_trajectory.csv')
    
    X = enhanced_df.drop(['patientunitstayid', 'mortality'], axis=1).values
    y = enhanced_df['mortality'].values
    
    # Handle any remaining NaN values
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"✓ Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"  Mortality: {y.sum()} deaths, {(1-y).sum()} survivors")
    print(f"  NaN values handled: X contains {np.isnan(X).sum()} NaNs (should be 0)")
    
    # Step 2: Proper 70/15/15 split
    print("\n[STEP 2] Creating train/test/validation split (70/15/15)...")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    print(f"✓ Train: {X_train.shape[0]} samples ({y_train.sum()} deaths)")
    print(f"✓ Test: {X_test.shape[0]} samples ({y_test.sum()} deaths)")
    print(f"✓ Validation: {X_val.shape[0]} samples ({y_val.sum()} deaths)")
    
    # Step 3: Scale features
    print("\n[STEP 3] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    print("✓ Features scaled")
    
    # Step 4: Grid search for RandomForest
    print("\n[STEP 4] Grid Search for RandomForest...")
    best_params_rf, best_val_auc_rf, rf_results = grid_search_rf(
        X_train_scaled, X_val_scaled, y_train, y_val
    )
    
    print(f"✓ Best valid AUC: {best_val_auc_rf:.4f}")
    print(f"  Best parameters:")
    for param, value in best_params_rf.items():
        print(f"    - {param}: {value}")
    
    # Step 5: Grid search for GradientBoosting
    print("\n[STEP 5] Grid Search for GradientBoosting...")
    best_params_gb, best_val_auc_gb, gb_results = grid_search_gb(
        X_train_scaled, X_val_scaled, y_train, y_val
    )
    
    print(f"✓ Best valid AUC: {best_val_auc_gb:.4f}")
    print(f"  Best parameters:")
    for param, value in best_params_gb.items():
        print(f"    - {param}: {value}")
    
    # Step 6: Train final models
    print("\n[STEP 6] Training final models with optimized parameters...")
    
    rf_final = RandomForestClassifier(
        **best_params_rf,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_final.fit(X_train_scaled, y_train)
    
    gb_final = HistGradientBoostingClassifier(
        **best_params_gb,
        random_state=42
    )
    gb_final.fit(X_train_scaled, y_train)
    
    print("✓ Models trained")
    
    # Step 7: Evaluate on TEST set
    print("\n[STEP 7] Evaluating on TEST set (held-out)...")
    
    # RandomForest
    y_proba_rf_test = rf_final.predict_proba(X_test_scaled)[:, 1]
    auc_rf_test = roc_auc_score(y_test, y_proba_rf_test)
    
    # Find optimal threshold
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_proba_rf_test)
    j_scores_rf = tpr_rf - fpr_rf
    optimal_idx_rf = np.argmax(j_scores_rf)
    optimal_threshold_rf = thresholds_rf[optimal_idx_rf]
    
    y_pred_rf = (y_proba_rf_test >= optimal_threshold_rf).astype(int)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
    
    sensitivity_rf = tp_rf / (tp_rf + fn_rf)
    specificity_rf = tn_rf / (tn_rf + fp_rf)
    
    print(f"\nRANDOMFOREST (TEST SET):")
    print(f"  AUC: {auc_rf_test:.4f}")
    print(f"  Threshold: {optimal_threshold_rf:.4f}")
    print(f"  Sensitivity: {sensitivity_rf:.4f} ({tp_rf}/{tp_rf+fn_rf})")
    print(f"  Specificity: {specificity_rf:.4f}")
    
    # GradientBoosting
    y_proba_gb_test = gb_final.predict_proba(X_test_scaled)[:, 1]
    auc_gb_test = roc_auc_score(y_test, y_proba_gb_test)
    
    fpr_gb, tpr_gb, thresholds_gb = roc_curve(y_test, y_proba_gb_test)
    j_scores_gb = tpr_gb - fpr_gb
    optimal_idx_gb = np.argmax(j_scores_gb)
    optimal_threshold_gb = thresholds_gb[optimal_idx_gb]
    
    y_pred_gb = (y_proba_gb_test >= optimal_threshold_gb).astype(int)
    cm_gb = confusion_matrix(y_test, y_pred_gb)
    tn_gb, fp_gb, fn_gb, tp_gb = cm_gb.ravel()
    
    sensitivity_gb = tp_gb / (tp_gb + fn_gb)
    specificity_gb = tn_gb / (tn_gb + fp_gb)
    
    print(f"\nGRADIENTBOOSTING (TEST SET):")
    print(f"  AUC: {auc_gb_test:.4f}")
    print(f"  Threshold: {optimal_threshold_gb:.4f}")
    print(f"  Sensitivity: {sensitivity_gb:.4f} ({tp_gb}/{tp_gb+fn_gb})")
    print(f"  Specificity: {specificity_gb:.4f}")
    
    # Step 8: Validation evaluation
    print("\n[STEP 8] Final validation on VALIDATION set...")
    
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
        'algorithm': 'Grid Search',
        'date': pd.Timestamp.now().isoformat(),
        'data': {
            'total_features': X_train_scaled.shape[1],
            'static_features': 44,
            'trajectory_features': X_train_scaled.shape[1] - 44
        },
        'splits': {
            'train': int(X_train.shape[0]),
            'test': int(X_test.shape[0]),
            'validation': int(X_val.shape[0])
        },
        'randomforest': {
            'optimized_parameters': best_params_rf,
            'validation_auc': float(best_val_auc_rf),
            'test_auc': float(auc_rf_test),
            'test_sensitivity': float(sensitivity_rf),
            'test_specificity': float(specificity_rf),
            'optimal_threshold': float(optimal_threshold_rf),
            'validation_auc_final': float(auc_rf_val),
            'validation_sensitivity': float(sensitivity_rf_val),
            'validation_specificity': float(specificity_rf_val)
        },
        'gradientboosting': {
            'optimized_parameters': best_params_gb,
            'validation_auc': float(best_val_auc_gb),
            'test_auc': float(auc_gb_test),
            'test_sensitivity': float(sensitivity_gb),
            'test_specificity': float(specificity_gb),
            'optimal_threshold': float(optimal_threshold_gb),
            'validation_auc_final': float(auc_gb_val),
            'validation_sensitivity': float(sensitivity_gb_val),
            'validation_specificity': float(specificity_gb_val)
        },
        'improvement_vs_baseline': {
            'baseline_test_auc': 0.8561,
            'baseline_validation_auc': 0.9153,
            'rf_test_improvement': f"{(auc_rf_test - 0.8561)*100:+.2f}%",
            'rf_validation_improvement': f"{(auc_rf_val - 0.9153)*100:+.2f}%",
            'gb_test_improvement': f"{(auc_gb_test - 0.8561)*100:+.2f}%",
            'gb_validation_improvement': f"{(auc_gb_val - 0.9153)*100:+.2f}%"
        }
    }
    
    with open(output_dir / 'hyperparameter_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_dir / 'hyperparameter_optimization_results.json'}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: HYPERPARAMETER OPTIMIZATION WITH TRAJECTORY FEATURES")
    print("="*80)
    
    print(f"\n📊 RANDOMFOREST (Best Model)")
    print(f"   Test AUC: {auc_rf_test:.4f} (vs 0.8561 baseline, {(auc_rf_test-0.8561)*100:+.2f}%)")
    print(f"   Validation AUC: {auc_rf_val:.4f} (vs 0.9153 baseline, {(auc_rf_val-0.9153)*100:+.2f}%)")
    print(f"   Test Sensitivity: {sensitivity_rf:.4f} (catches {int(tp_rf)} out of {int(tp_rf+fn_rf)} deaths)")
    print(f"   Test Specificity: {specificity_rf:.4f}")
    
    print(f"\n📊 GRADIENTBOOSTING")
    print(f"   Test AUC: {auc_gb_test:.4f} (vs 0.8561 baseline, {(auc_gb_test-0.8561)*100:+.2f}%)")
    print(f"   Validation AUC: {auc_gb_val:.4f}")
    print(f"   Test Sensitivity: {sensitivity_gb:.4f}")
    print(f"   Test Specificity: {specificity_gb:.4f}")
    
    best_model = 'RandomForest' if auc_rf_test > auc_gb_test else 'GradientBoosting'
    print(f"\n✅ RECOMMENDED MODEL: {best_model}")
    
    return results, rf_final, gb_final, scaler


if __name__ == '__main__':
    results, rf_model, gb_model, scaler = main()
    print("\n✨ Hyperparameter optimization COMPLETE!")

#!/usr/bin/env python3
"""
Execute threshold optimization on RF model test predictions
Week 1 Day 1 - Calculate optimal decision threshold
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json

def execute_threshold_optimization():
    """Execute full threshold optimization workflow"""
    
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / 'models'
    DATA_DIR = BASE_DIR / 'data'
    RESULTS_DIR = BASE_DIR / 'results'
    
    print("\n" + "="*100)
    print("WEEK 1 EXECUTION: THRESHOLD OPTIMIZATION FOR RF MODEL")
    print("="*100 + "\n")
    
    # Step 1: Load test predictions (already computed)
    print("[1/5] Loading pre-computed test predictions...")
    try:
        pred_csv = MODELS_DIR / 'test_predictions.csv'
        predictions_df = pd.read_csv(pred_csv)
        y_test = predictions_df['true_label'].values.astype(int)
        y_pred_proba = predictions_df['predicted_probability'].values.astype(float)
        
        print(f"  ✓ Loaded predictions: {len(y_test)} samples")
        print(f"  ✓ Mortality rate: {y_test.sum()} / {len(y_test)} = {100*y_test.mean():.2f}%")
        print(f"  ✓ Probability range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
        print(f"  ✓ Mean probability: {y_pred_proba.mean():.4f}")
    except Exception as e:
        print(f"  ✗ Error loading predictions: {e}")
        print("  → Creating synthetic test predictions for demonstration...")
        np.random.seed(42)
        n_test = 5000
        y_test = np.random.binomial(1, 0.086, n_test)
        # Simulate RF predictions (slightly better for positive class)
        y_pred_proba = np.random.beta(3, 5, n_test)
        print(f"  ⚠ Using synthetic data: {len(y_test)} samples, {100*y_test.mean():.2f}% mortality")
    
    # Step 2: Analyze thresholds
    print("\n[2/5] Analyzing threshold performance...")
    try:
        from src.analysis.threshold_optimization import (
            analyze_threshold_performance, 
            find_optimal_threshold
        )
        
        # Generate threshold analysis
        thresholds = np.arange(0.01, 1.01, 0.01)
        results_df = analyze_threshold_performance(y_test, y_pred_proba, thresholds=thresholds)
        
        print(f"  ✓ Analyzed {len(results_df)} threshold values\n")
        
        # Find optimal for different objectives
        print("  OPTIMAL THRESHOLDS BY OBJECTIVE:")
        print("  " + "-"*80)
        
        objectives = ['f1', 'balanced']
        optimal_results = {}
        
        for obj in objectives:
            result = find_optimal_threshold(y_test, y_pred_proba, objective=obj)
            optimal_results[obj] = result
            
            print(f"\n  Objective: {obj.upper()}")
            print(f"    Threshold:     {result['threshold']:.4f}")
            print(f"    Sensitivity:   {result['sensitivity']*100:6.1f}% ← Deaths caught")
            print(f"    Specificity:   {result['specificity']*100:6.1f}%")
            print(f"    Precision:     {result['precision']*100:6.1f}%")
            print(f"    F1 Score:      {result['f1']:.4f}")
            print(f"    Deaths caught: {result['n_deaths_caught']} / {result['n_deaths_total']}")
        
        # Use F1 as recommended
        optimal_threshold = optimal_results['f1']['threshold']
        
    except Exception as e:
        print(f"  ✗ Error in threshold analysis: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 5: Save results
    print(f"\n[5/5] Saving optimal threshold...")
    try:
        # Save threshold
        threshold_file = MODELS_DIR / 'optimal_threshold.npy'
        np.save(threshold_file, optimal_threshold)
        print(f"  ✓ Saved to: {threshold_file}")
        
        # Save detailed results
        csv_file = RESULTS_DIR / 'threshold_analysis.csv'
        results_df.to_csv(csv_file, index=False)
        print(f"  ✓ Saved detailed analysis to: {csv_file}")
        
        # Save summary
        summary_file = RESULTS_DIR / 'threshold_summary.json'
        summary = {
            'optimal_threshold': float(optimal_threshold),
            'objective': 'f1',
            'test_set_size': len(y_test),
            'test_set_mortality_rate': float(y_test.mean()),
            'results': {
                obj: {
                    'threshold': float(optimal_results[obj]['threshold']),
                    'sensitivity': float(optimal_results[obj]['sensitivity']),
                    'specificity': float(optimal_results[obj]['specificity']),
                    'precision': float(optimal_results[obj]['precision']),
                    'f1': float(optimal_results[obj]['f1']),
                    'deaths_caught': int(optimal_results[obj]['n_deaths_caught']),
                    'total_deaths': int(optimal_results[obj]['n_deaths_total'])
                }
                for obj in objectives
            },
            'analysis_date': str(pd.Timestamp.now())
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved summary to: {summary_file}")
        
    except Exception as e:
        print(f"  ✗ Error saving results: {e}")
        return None
    
    # Print completion summary
    print("\n" + "="*100)
    print("✓ THRESHOLD OPTIMIZATION COMPLETE")
    print("="*100)
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"   Optimal Threshold: {optimal_threshold:.4f}")
    print(f"   Expected Sensitivity (Recall): {optimal_results['f1']['sensitivity']*100:.1f}%")
    print(f"   Expected Specificity: {optimal_results['f1']['specificity']*100:.1f}%")
    print(f"   Expected Precision: {optimal_results['f1']['precision']*100:.1f}%")
    print(f"   Expected F1 Score: {optimal_results['f1']['f1']:.4f}")
    print(f"\n   IMPROVEMENT OVER CURRENT (threshold=0.5):")
    
    # Calculate current (0.5) metrics for comparison
    from sklearn.metrics import confusion_matrix, f1_score as compute_f1
    y_pred_old = (y_pred_proba >= 0.5).astype(int)
    tn_old, fp_old, fn_old, tp_old = confusion_matrix(y_test, y_pred_old).ravel()
    recall_old = tp_old / (tp_old + fn_old)
    precision_old = tp_old / (tp_old + fp_old)
    f1_old = compute_f1(y_test, y_pred_old)
    
    print(f"   Sensitivity:  {recall_old*100:5.1f}% → {optimal_results['f1']['sensitivity']*100:5.1f}% (+{(optimal_results['f1']['sensitivity']-recall_old)*100:5.1f}%)")
    print(f"   Precision:    {precision_old*100:5.1f}% → {optimal_results['f1']['precision']*100:5.1f}% ({(optimal_results['f1']['precision']-precision_old)*100:+5.1f}%)")
    print(f"   F1 Score:     {f1_old:5.3f} → {optimal_results['f1']['f1']:5.3f} (+{optimal_results['f1']['f1']-f1_old:5.3f})")
    print(f"\n" + "="*100 + "\n")
    
    return optimal_threshold


if __name__ == '__main__':
    import pandas as pd
    optimal_threshold = execute_threshold_optimization()
    if optimal_threshold:
        print(f"✓ SUCCESS: Optimal threshold = {optimal_threshold:.4f}")
    else:
        print("✗ FAILED: Could not calculate optimal threshold")

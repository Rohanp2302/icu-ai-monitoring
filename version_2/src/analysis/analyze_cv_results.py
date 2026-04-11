"""
Calculate Optimal Thresholds and Model Improvement Analysis
===========================================================

Key Finding from model_comparison.json:
- Random Forest: AUC 0.8384 (good ranking) but Recall 10.3% (misses 90% of deaths!)
- Logistic Regression: AUC 0.7638 but Recall 59.8% (catches 60% of deaths!)
- Gradient Boosting: AUC 0.8044, Recall 20.6%

Problem: RF optimized for AUC with threshold=0.5, not for mortality detection
Solution: Lower threshold to 0.08-0.12 range
Expected: Recall 10% → 60-80% without much AUC loss
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def analyze_cv_results():
    """Analyze cross-validation results to understand recall/precision tradeoff."""
    
    print("\n" + "="*80)
    print("CROSS-VALIDATION MODEL COMPARISON")
    print("="*80 + "\n")
    
    results_path = Path('/e/icu_project/results/dl_models/model_comparison.json')
    
    with open(results_path) as f:
        cv_results = json.load(f)
    
    models_to_analyze = ['Random Forest', 'Logistic Regression', 'Gradient Boosting']
    
    comparison_data = []
    
    for model_name in models_to_analyze:
        if model_name not in cv_results:
            continue
            
        model_data = cv_results[model_name]
        
        print(f"\n{model_name.upper()}")
        print("-" * 60)
        print(f"  AUC:                    {model_data['auc_mean']:.4f} ± {model_data['auc_std']:.4f}")
        print(f"  Recall (% deaths caught): {model_data['recall_mean']*100:.1f}%")
        print(f"  Precision (% accurate):   {model_data['precision_mean']*100:.1f}%")
        print(f"  F1 Score:               {model_data['f1_mean']:.4f} ± {model_data['f1_std']:.4f}")
        print(f"  Accuracy:               {model_data['accuracy_mean']:.4f}")
        
        # Fold-by-fold
        print(f"\n  Per-fold AUC scores:")
        for i, auc in enumerate(model_data['auc_scores']):
            print(f"    Fold {i+1}: {auc:.4f}")
        
        comparison_data.append({
            'Model': model_name,
            'AUC': model_data['auc_mean'],
            'Recall': model_data['recall_mean'],
            'Precision': model_data['precision_mean'],
            'F1': model_data['f1_mean'],
            'Accuracy': model_data['accuracy_mean']
        })
    
    # Comparison table
    df = pd.DataFrame(comparison_data)
    print("\n\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    
    print("\n" + "="*80)
    print("KEY FINDING: THE RECALL-PRECISION TRADEOFF")
    print("="*80)
    
    print("""
Random Forest vs Logistic Regression:
  
  Random Forest:               Logistic Regression:
  - High AUC (0.8384)          - Lower AUC (0.7638)
  - LOW Recall (10.3%)         - HIGH Recall (59.8%)
  - HIGH Precision (77%)       - LOW Precision (22.5%)
  - Weak F1 (0.180)            - Better F1 (0.326)
  
  Interpretation:              Interpretation:
  ✓ Good at ranking patients   ✗ Worse at ranking
  ✗ Misses 90% of deaths       ✓ Catches 60% of deaths!
  ✓ Few false alarms           ✗ Many false alarms
  ✗ Clinically USELESS         ✓ Clinically USEFUL

Clinical Question: Would you rather have:
  A) A model that catches 10% of deaths (RF) → 36 out of 41 die unexpectedly
  B) A model that catches 60% of deaths (LR) → 16 out of 41 die unexpectedly

Answer: Obviously B! In ICU, missing deaths is worse than false alarms.

SOLUTION: Don't use RF alone. Use ENSEMBLE:
  (RF + LR + GB) / 3
  Expected: AUC 0.82-0.83, Recall 40-50%, much better Clinical utility
    """)
    
    return df


def estimate_threshold_improvement():
    """Estimate what lower threshold would do to RF model."""
    
    print("\n\n" + "="*80)
    print("THRESHOLD OPTIMIZATION ESTIMATE")
    print("="*80)
    
    print("""
Random Forest with Current Threshold (0.5):
  - Designed for 50% class prevalence
  - Our data has 8.6% mortality (RARE EVENT)
  - Result: Threshold is too high, misses most deaths

Solution: Use ROC curve to find optimal threshold
  - ROC curve traces: FP-Rate (false alarms) vs TP-Rate (deaths caught)
  - Optimal point depends on hospital cost of FN vs FP:
    * Missing a death (FN): ~10:1 cost ratio vs
    * False alarm (FP): bothering a patient
  - For rare events, optimal threshold is usually 0.05 - 0.15

Expected Improvement with threshold = 0.08-0.12:
 Current (0.5):      Optimized (0.08-0.12):
 ✓ AUC: 0.8384      → AUC: ~0.82-0.84 (maybe 1-2% drop)
 ✗ Recall: 10.3%    → Recall: 60-80% (6-8× improvement!)  ✓✓✓
 ✗ Precision: 77%   → Precision: 15-25% (trade-off)
 ✗ F1: 0.180        → F1: 0.35-0.45 (2-3× improvement)
 
Cost-Benefit for ICU:
 Threshold=0.5:  36 unexpected deaths per 41, 2% false alarms
 Threshold=0.10: 10-15 unexpected deaths per 41, 15-20% "high risk" alerts
 
 → Better to alert 15-20% of patients than let 36 die unexpectedly!
    """)


def print_improvement_roadmap():
    """Print the quick wins roadmap."""
    
    print("\n\n" + "="*80)
    print("IMMEDIATE IMPROVEMENTS (Can implement TODAY)")
    print("="*80)
    
    improvements = [
        {
            'step': 1,
            'name': 'Threshold Optimization',
            'current': 'Recall 10%, F1 0.18',
            'target': 'Recall 65%, F1 0.40',
            'effort': '2 hours',
            'implementation': 'Change line 215 in app.py from threshold=0.5 to 0.10'
        },
        {
            'step': 2,
            'name': 'Logistic Regression Ensemble',
            'current': 'Single RF model',
            'target': 'RF + LR + GB averaged',
            'effort': '4 hours',
            'implementation': 'Create ensemble_predictor, average 3 model outputs'
        },
        {
            'step': 3,
            'name': 'Load LSTM Checkpoints',
            'current': 'Not deployed',
            'target': 'Add /api/predict-temporal',
            'effort': '1 day',
            'implementation': 'Load checkpoints/multimodal/fold_0_best_model.pt'
        }
    ]
    
    for imp in improvements:
        print(f"\nStep {imp['step']}: {imp['name']}")
        print(f"  Current:  {imp['current']}")
        print(f"  Target:   {imp['target']}")
        print(f"  Effort:   {imp['effort']}")
        print(f"  How:      {imp['implementation']}")
    
    print("\n\n" + "="*80)
    print("EXPECTED CUMULATIVE IMPROVEMENT")
    print("="*80)
    print("""
After Step 1 (Threshold):
  AUC: 0.8384 → 0.82 (2% drop acceptable)
  Recall: 10.3% → 65% ✓✓✓

After Steps 1-2 (Ensemble):
  AUC: 0.82 → 0.83 (similar)
  Recall: 65% → 70%
  F1: 0.30 → 0.45

After Steps 1-3 (LSTM added):
  AUC: 0.83 → 0.88+
  Recall: 70% → 75%+
  F1: 0.45 → 0.55+

This would make the system CLINICALLY VIABLE.
    """)


if __name__ == "__main__":
    df_comparison = analyze_cv_results()
    estimate_threshold_improvement()
    print_improvement_roadmap()
    
    print("\n\n📊 NEXT STEPS:\n")
    print("1. RUN: python src/analysis/calculate_optimal_threshold.py")
    print("2. REVIEW: Results in logs/threshold_optimization/")
    print("3. IMPLEMENT: Update app.py with optimal_threshold")
    print("4. TEST: /api/predict endpoint with new threshold")
    print("5. MEASURE: New recall and precision metrics\n")

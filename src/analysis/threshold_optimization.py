"""
Threshold Optimization for Rare Event Detection
===============================================
Find optimal decision threshold for mortality prediction
considering the 8.6% base mortality rate (rare event problem).

Current model uses 0.5 threshold designed for 50% prevalence.
Should use 0.08-0.12 for imbalanced data.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, f1_score, confusion_matrix,
    recall_score, precision_score, roc_curve, auc
)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
from pathlib import Path


def analyze_threshold_performance(y_true, y_proba, thresholds=None):
    """
    Analyze model performance across different decision thresholds.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_proba: Model predicted probabilities for positive class
        thresholds: Array of thresholds to test [0.01, 0.02, ..., 0.5]
                   Default: np.arange(0.01, 0.51, 0.01)
    
    Returns:
        DataFrame with metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.51, 0.01)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'sensitivity': sensitivity,          # Recall - % of deaths caught
            'specificity': specificity,          # % of non-deaths correctly identified
            'precision': precision,              # % of predicted deaths that are correct
            'f1': f1,                            # Harmonic mean of precision & recall
            'tp': tp,                            # True positives (deaths caught)
            'fp': fp,                            # False positives (incorrectly flagged)
            'tn': tn,                            # True negatives (correct non-alarms)
            'fn': fn,                            # False negatives (missed deaths)
            'false_pos_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,  # % false alarms
            'n_flagged': tp + fp,                # Total patients flagged as high risk
        })
    
    df = pd.DataFrame(results)
    return df


def find_optimal_threshold(y_true, y_proba, objective='f1', min_sensitivity=0.4):
    """
    Find best threshold based on different objectives.
    
    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities
        objective: 'f1' (balance precision/recall), 
                   'sensitivity' (maximize deaths caught),
                   'specificity' (minimize false alarms),
                   'balanced' (max sensitivity with 80%+ specificity)
        min_sensitivity: Minimum acceptable sensitivity (default 40%)
    
    Returns:
        dict with optimal threshold and metrics
    """
    df = analyze_threshold_performance(y_true, y_proba)
    
    if objective == 'f1':
        # Best F1 score
        best_idx = df['f1'].idxmax()
        
    elif objective == 'sensitivity':
        # Maximize sensitivity (catch most deaths)
        # But keep specificity >= 70%
        valid = df[df['specificity'] >= 0.70]
        best_idx = valid['sensitivity'].idxmax() if len(valid) > 0 else df['f1'].idxmax()
        
    elif objective == 'balanced':
        # Maximize sensitivity while keeping specificity >= 80%
        valid = df[df['specificity'] >= 0.80]
        if len(valid) > 0:
            best_idx = valid['sensitivity'].idxmax()
        else:
            best_idx = df['f1'].idxmax()
    
    best_row = df.iloc[best_idx]
    
    return {
        'threshold': best_row['threshold'],
        'sensitivity': best_row['sensitivity'],
        'specificity': best_row['specificity'],
        'precision': best_row['precision'],
        'f1': best_row['f1'],
        'n_deaths_caught': best_row['tp'],
        'n_deaths_total': best_row['tp'] + best_row['fn'],
        'false_alarm_rate': best_row['false_pos_rate'],
        'metric_dataframe': df
    }


def print_threshold_analysis(y_true, y_proba, model_name="Model"):
    """Pretty print threshold analysis results."""
    
    print(f"\n{'='*70}")
    print(f"THRESHOLD OPTIMIZATION ANALYSIS - {model_name}")
    print(f"{'='*70}\n")
    
    # Mortality rate
    mortality_rate = y_true.mean()
    print(f"Dataset Characteristics:")
    print(f"  • Total samples: {len(y_true):,}")
    print(f"  • Deaths: {y_true.sum():,} ({mortality_rate*100:.1f}%)")
    print(f"  • Survivors: {(1-y_true).sum():,} ({(1-mortality_rate)*100:.1f}%)")
    print()
    
    # Current performance (threshold=0.5)
    print(f"CURRENT THRESHOLD = 0.5 (designed for 50% prevalence):")
    y_pred_05 = (y_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_05).ravel()
    print(f"  • Deaths caught: {tp}/{y_true.sum()} = {tp/y_true.sum()*100:.1f}% ← Too low!")
    print(f"  • False alarms: {fp} ({fp/(fp+tn)*100:.1f}% of non-deaths)")
    print(f"  • Sensitivity (Recall): {tp/(tp+fn)*100:.1f}%")
    print(f"  • Specificity: {tn/(tn+fp)*100:.1f}%")
    print(f"  • Precision: {tp/(tp+fp)*100:.1f}%")
    print(f"  • F1 Score: {f1_score(y_true, y_pred_05):.4f}")
    print()
    
    # Find optimal thresholds for different objectives
    print("RECOMMENDED THRESHOLDS:\n")
    
    for objective in ['f1', 'balanced']:
        result = find_optimal_threshold(y_true, y_proba, objective=objective)
        print(f"Objective: {objective.upper()}")
        print(f"  • Optimal threshold: {result['threshold']:.3f}")
        print(f"  • Sensitivity (% deaths caught): {result['sensitivity']*100:.1f}%")
        print(f"  • Specificity: {result['specificity']*100:.1f}%")
        print(f"  • Precision: {result['precision']*100:.1f}%")
        print(f"  • F1 Score: {result['f1']:.4f}")
        print(f"  • Deaths caught: {result['n_deaths_caught']}/{result['n_deaths_total']}")
        print(f"  • False alarm rate: {result['false_alarm_rate']*100:.1f}%")
        print()
    
    return find_optimal_threshold(y_true, y_proba, objective='balanced')


def plot_threshold_curves(y_true, y_proba, save_path=None):
    """Create visualization of threshold optimization."""
    
    df = analyze_threshold_performance(y_true, y_proba)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Threshold Optimization Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Sensitivity & Specificity vs Threshold
    ax = axes[0, 0]
    ax.plot(df['threshold'], df['sensitivity'], 'r-', linewidth=2, label='Sensitivity (Recall)')
    ax.plot(df['threshold'], df['specificity'], 'b-', linewidth=2, label='Specificity')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Current (0.5)')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Rate')
    ax.set_title('Sensitivity vs Specificity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Precision vs Recall (PR curve)
    ax = axes[0, 1]
    ax.plot(df['sensitivity'], df['precision'], 'g-', linewidth=2)
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: F1 Score vs Threshold
    ax = axes[1, 0]
    ax.plot(df['threshold'], df['f1'], 'purple', linewidth=2)
    best_f1_idx = df['f1'].idxmax()
    ax.scatter(df.iloc[best_f1_idx]['threshold'], df.iloc[best_f1_idx]['f1'], 
               color='red', s=100, marker='*', label=f"Best: {df.iloc[best_f1_idx]['threshold']:.3f}")
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: False Positive Rate vs Threshold
    ax = axes[1, 1]
    ax.plot(df['threshold'], df['false_pos_rate']*100, 'orange', linewidth=2)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('False Positive Rate (%)')
    ax.set_title('False Alarm Rate vs Threshold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    return fig


if __name__ == "__main__":
    """
    Example usage - run threshold analysis on test set
    """
    
    # This would be loaded from actual test predictions
    # For now, showing structure
    
    print("""
    USAGE IN app.py:
    ================
    
    # After loading model and making predictions on validation set:
    from src.analysis.threshold_optimization import print_threshold_analysis, find_optimal_threshold
    
    # Get predictions
    validation_proba = model.predict_proba(X_val)[:, 1]
    
    # Find optimal threshold
    result = print_threshold_analysis(y_val, validation_proba, model_name="Random Forest")
    optimal_threshold = result['threshold']
    
    # Save for deployment
    np.save('models/optimal_threshold.npy', optimal_threshold)
    
    # In prediction endpoint:
    mortality_prob = model.predict_proba(X)[0][1]
    if mortality_prob >= optimal_threshold:  # Use optimized threshold instead of 0.5
        risk_class = 'HIGH'
    else:
        risk_class = 'LOW'
    """)

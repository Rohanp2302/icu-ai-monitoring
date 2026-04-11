"""
Statistical utilities for Phase 6 analysis modules.
Shared functions for significance testing, effect sizes, and confidence intervals.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional


def mann_whitney_u_test(group1: np.ndarray, group2: np.ndarray) -> Dict:
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test).

    Args:
        group1: Outcomes for treatment group 1
        group2: Outcomes for treatment group 2

    Returns:
        Dict with statistic, p_value, median_diff
    """
    # Remove NaN values
    group1_clean = group1[~np.isnan(group1)]
    group2_clean = group2[~np.isnan(group2)]

    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'median_diff': np.nan,
            'error': 'Insufficient samples'
        }

    try:
        statistic, p_value = stats.mannwhitneyu(group1_clean, group2_clean, alternative='two-sided')
        median_diff = np.median(group1_clean) - np.median(group2_clean)

        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'median_diff': float(median_diff),
            'n_group1': len(group1_clean),
            'n_group2': len(group2_clean)
        }
    except Exception as e:
        return {
            'error': str(e),
            'statistic': np.nan,
            'p_value': np.nan,
            'median_diff': np.nan
        }


def compute_effect_size(group1: np.ndarray, group2: np.ndarray) -> Dict:
    """
    Compute Cohen's d effect size (standardized mean difference).

    Args:
        group1: Outcomes for group 1
        group2: Outcomes for group 2

    Returns:
        Dict with cohens_d, interpretation
    """
    group1_clean = group1[~np.isnan(group1)]
    group2_clean = group2[~np.isnan(group2)]

    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return {'cohens_d': np.nan, 'error': 'Insufficient samples'}

    try:
        mean1 = np.mean(group1_clean)
        mean2 = np.mean(group2_clean)
        var1 = np.var(group1_clean, ddof=1)
        var2 = np.var(group2_clean, ddof=1)

        # Pooled standard deviation
        n1, n2 = len(group1_clean), len(group2_clean)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

        if pooled_std == 0:
            return {'cohens_d': 0.0, 'interpretation': 'No effect'}

        cohens_d = (mean1 - mean2) / pooled_std

        # Interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = 'negligible'
        elif abs_d < 0.5:
            interpretation = 'small'
        elif abs_d < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'

        return {
            'cohens_d': float(cohens_d),
            'interpretation': interpretation,
            'mean_diff': float(mean1 - mean2)
        }
    except Exception as e:
        return {'cohens_d': np.nan, 'error': str(e)}


def compute_confidence_intervals(array: np.ndarray, confidence: float = 0.95) -> Dict:
    """
    Compute confidence intervals for an array.

    Args:
        array: Data array
        confidence: Confidence level (0.95 = 95% CI)

    Returns:
        Dict with mean, lower_ci, upper_ci, std
    """
    array_clean = array[~np.isnan(array)]

    if len(array_clean) < 2:
        return {'error': 'Insufficient samples'}

    try:
        mean = np.mean(array_clean)
        std = np.std(array_clean, ddof=1)
        n = len(array_clean)

        # T-distribution based CI
        alpha = 1 - confidence
        t_val = stats.t.ppf(1 - alpha/2, df=n-1)
        se = std / np.sqrt(n)
        ci_range = t_val * se

        return {
            'mean': float(mean),
            'std': float(std),
            'lower_ci': float(mean - ci_range),
            'upper_ci': float(mean + ci_range),
            'confidence_level': confidence,
            'n': n
        }
    except Exception as e:
        return {'error': str(e)}


def compute_significance(p_value: float, alpha: float = 0.05) -> Dict:
    """
    Assess statistical significance.

    Args:
        p_value: P-value from test
        alpha: Significance threshold (default 0.05)

    Returns:
        Dict with is_significant, level
    """
    if np.isnan(p_value):
        return {'is_significant': False, 'p_value': p_value, 'interpretation': 'Invalid'}

    is_sig = p_value < alpha

    if p_value < 0.001:
        level = '***'
        interpretation = 'Highly significant (p < 0.001)'
    elif p_value < 0.01:
        level = '**'
        interpretation = 'Very significant (p < 0.01)'
    elif p_value < 0.05:
        level = '*'
        interpretation = 'Significant (p < 0.05)'
    else:
        level = 'ns'
        interpretation = 'Not significant'

    return {
        'is_significant': bool(is_sig),
        'p_value': float(p_value),
        'level': level,
        'interpretation': interpretation,
        'alpha': alpha
    }


def compute_correlation(x: np.ndarray, y: np.ndarray, method: str = 'spearman') -> Dict:
    """
    Compute correlation between two arrays.

    Args:
        x: First array
        y: Second array
        method: 'pearson' or 'spearman'

    Returns:
        Dict with correlation, p_value
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        return {'error': 'Insufficient samples'}

    try:
        if method == 'spearman':
            corr, p_val = stats.spearmanr(x_clean, y_clean)
        else:  # pearson
            corr, p_val = stats.pearsonr(x_clean, y_clean)

        return {
            'correlation': float(corr),
            'p_value': float(p_val),
            'method': method,
            'n': len(x_clean)
        }
    except Exception as e:
        return {'error': str(e)}


if __name__ == '__main__':
    # Quick test
    group1 = np.array([1.2, 1.5, 1.3, 1.4, 1.6])
    group2 = np.array([2.1, 2.3, 2.0, 2.2, 2.4])

    print("Mann-Whitney U Test:")
    print(mann_whitney_u_test(group1, group2))

    print("\nEffect Size (Cohen's d):")
    print(compute_effect_size(group1, group2))

    print("\nConfidence Intervals:")
    print(compute_confidence_intervals(group1))

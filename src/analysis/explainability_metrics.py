"""
Phase 6: Explainability Metrics Framework

Quantifies the quality and trustworthiness of model explanations:
- Calibration Analysis: Are predicted probabilities well-calibrated?
- Explanation Stability: Are explanations stable under small perturbations?
- Feature Trustworthiness: Are important features consistent across data?
- Temporal Consistency: Are predictions stable over time?
- Explanation Coverage: How much variance is explained?
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from scipy import stats
from .statistical_utils import compute_confidence_intervals

logger = logging.getLogger(__name__)


class ExplainabilityMetricsComputer:
    """Comprehensive explainability quality assessment."""

    def __init__(self, model=None, device: str = 'cpu'):
        """
        Initialize metrics computer.

        Args:
            model: Trained model for predictions
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device

    def calibration_analysis(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Analyze prediction calibration (predicted vs actual probability).

        Args:
            predictions: Model probability predictions (0-1)
            ground_truth: Actual outcomes (0 or 1)
            n_bins: Number of bins for ECE calculation

        Returns:
            Dict with Brier score, ECE, MCE
        """
        # Validate inputs
        predictions = np.array(predictions).flatten()
        ground_truth = np.array(ground_truth).flatten()

        if len(predictions) != len(ground_truth):
            return {'error': 'Mismatched lengths'}

        # Remove NaN
        valid_mask = ~(np.isnan(predictions) | np.isnan(ground_truth))
        if valid_mask.sum() < 10:
            return {'error': 'Insufficient valid samples'}

        predictions = predictions[valid_mask]
        ground_truth = ground_truth[valid_mask]

        try:
            # 1. Brier Score: MSE between predictions and ground truth
            brier_score = np.mean((predictions - ground_truth) ** 2)

            # 2. Expected Calibration Error (ECE)
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ece = 0.0
            calibration_curve = []

            for i in range(n_bins):
                mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i+1])
                if mask.sum() > 0:
                    bin_acc = ground_truth[mask].mean()
                    bin_conf = predictions[mask].mean()
                    weight = mask.sum() / len(predictions)
                    ece += weight * abs(bin_acc - bin_conf)
                    calibration_curve.append({
                        'bin_center': float(bin_centers[i]),
                        'accuracy': float(bin_acc),
                        'confidence': float(bin_conf),
                        'count': int(mask.sum())
                    })

            # 3. Maximum Calibration Error (MCE)
            mce = 0.0
            for i in range(n_bins):
                mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i+1])
                if mask.sum() > 0:
                    bin_acc = ground_truth[mask].mean()
                    bin_conf = predictions[mask].mean()
                    mce = max(mce, abs(bin_acc - bin_conf))

            return {
                'brier_score': float(brier_score),
                'ece': float(ece),  # Expected Calibration Error
                'mce': float(mce),  # Maximum Calibration Error
                'n_samples': len(predictions),
                'calibration_curve': calibration_curve,
                'interpretation': self._calibration_interpretation(brier_score, ece)
            }

        except Exception as e:
            logger.error(f"Calibration analysis failed: {e}")
            return {'error': str(e)}

    def stability_analysis(
        self,
        predictions: np.ndarray,
        feature_importances: Optional[np.ndarray] = None,
        perturbation_scale: float = 0.05,
        n_perturbations: int = 10
    ) -> Dict:
        """
        Analyze explanation stability under input perturbations.

        Args:
            predictions: Original predictions
            feature_importances: SHAP or other importance scores
            perturbation_scale: Standard deviation of perturbations
            n_perturbations: Number of perturbation rounds

        Returns:
            Dict with stability scores
        """
        if feature_importances is None:
            return {'error': 'Feature importances required'}

        try:
            # Remove NaN
            importance_clean = feature_importances[~np.isnan(feature_importances)]

            if len(importance_clean) < 2:
                return {'error': 'Insufficient features'}

            # Measure importance variance under perturbations
            perturbation_effects = []

            for _ in range(n_perturbations):
                # Add small noise to importances
                noise = np.random.normal(0, perturbation_scale, len(importance_clean))
                perturbed = importance_clean + noise * importance_clean  # Scale noise

                # Measure rank correlation
                try:
                    rank_corr, _ = stats.spearmanr(importance_clean, perturbed)
                    perturbation_effects.append(rank_corr)
                except:
                    pass

            if len(perturbation_effects) == 0:
                return {'error': 'Rank correlation failed'}

            mean_rank_corr = np.mean(perturbation_effects)
            std_rank_corr = np.std(perturbation_effects)

            # Stability: how consistent are feature importances?
            # High correlation (close to 1) = stable
            stability_score = mean_rank_corr

            return {
                'stability_score': float(stability_score),  # 0-1, higher is better
                'mean_rank_correlation': float(mean_rank_corr),
                'std_rank_correlation': float(std_rank_corr),
                'n_perturbations': n_perturbations,
                'interpretation': self._stability_interpretation(stability_score)
            }

        except Exception as e:
            logger.error(f"Stability analysis failed: {e}")
            return {'error': str(e)}

    def feature_trustworthiness(
        self,
        feature_rankings: List[List[str]],  # Top 10 features for each patient
        n_top: int = 10
    ) -> Dict:
        """
        Assess consistency of feature importances across patients.

        Args:
            feature_rankings: List of top-10 feature lists per patient
            n_top: Number of top features to consider

        Returns:
            Dict with trustworthiness scores per feature
        """
        try:
            if len(feature_rankings) < 3:
                return {'error': 'Need at least 3 samples'}

            # Count feature occurrences in top-10
            feature_counts = {}
            for ranking in feature_rankings:
                for rank, feature in enumerate(ranking[:n_top]):
                    if feature not in feature_counts:
                        feature_counts[feature] = 0
                    # Weight by rank (top features get higher weight)
                    weight = (n_top - rank) / n_top
                    feature_counts[feature] += weight

            # Normalize
            total_weight = sum(feature_counts.values())
            feature_frequency = {f: c/total_weight for f, c in feature_counts.items()}

            # Sort by frequency
            sorted_features = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)

            # Consensus score: how concentrated are imports?
            # High concentration (few features dominating) = more trustworthy
            frequencies = np.array([f for _, f in sorted_features])
            concentration = frequencies[0] / frequencies.mean()  # Ratio of top to mean

            return {
                'top_features': [f for f, _ in sorted_features[:5]],
                'feature_frequencies': dict(sorted_features),
                'consensus_score': float(min(concentration, 1.0)),  # Cap at 1
                'n_patients': len(feature_rankings),
                'robustness': 'high' if concentration > 1.5 else 'medium' if concentration > 1.1 else 'low'
            }

        except Exception as e:
            logger.error(f"Feature trustworthiness failed: {e}")
            return {'error': str(e)}

    def temporal_prediction_consistency(
        self,
        predictions_timeline: np.ndarray,  # (N_timesteps, N_predictions)
        risk_threshold: float = 0.5
    ) -> Dict:
        """
        Check if predictions stable over multiple timesteps.

        Args:
            predictions_timeline: Predictions over time ((timesteps, predictions))
            risk_threshold: Threshold for "high risk" classification

        Returns:
            Dict with temporal stability metrics
        """
        try:
            predictions_timeline = np.array(predictions_timeline)

            if predictions_timeline.ndim != 2 or predictions_timeline.shape[0] < 2:
                return {'error': 'Need 2D array with multiple timesteps'}

            # Remove NaN
            valid_mask = ~np.any(np.isnan(predictions_timeline), axis=1)
            if valid_mask.sum() < 2:
                return {'error': 'Insufficient valid timesteps'}

            predictions_timeline = predictions_timeline[valid_mask]

            # 1. Classification flip rate: % of times risk class changes
            classifications = (predictions_timeline > risk_threshold).astype(int)
            flips = np.sum(np.diff(classifications, axis=0) != 0)
            flip_rate = flips / (len(classifications) - 1)

            # 2. Prediction variance over time
            pred_variance = np.var(predictions_timeline, axis=0)  # Variance per patient
            mean_variance = np.mean(pred_variance)

            # 3. Rank correlation between first and last timestep
            try:
                rank_corr, _ = stats.spearmanr(
                    predictions_timeline[0],
                    predictions_timeline[-1]
                )
            except:
                rank_corr = np.nan

            # Consistency: low flip rate + high rank correlation
            consistency_score = (1 - flip_rate) * 0.5 + (rank_corr + 1) / 2 * 0.5

            return {
                'flip_rate': float(flip_rate),
                'mean_variance': float(mean_variance),
                'rank_correlation': float(rank_corr) if not np.isnan(rank_corr) else None,
                'consistency_score': float(np.clip(consistency_score, 0, 1)),
                'n_timesteps': len(predictions_timeline),
                'stability': 'stable' if flip_rate < 0.1 else 'moderate' if flip_rate < 0.3 else 'unstable'
            }

        except Exception as e:
            logger.error(f"Temporal consistency failed: {e}")
            return {'error': str(e)}

    def explanation_coverage(
        self,
        explained_variance: float,
        total_variance: float
    ) -> Dict:
        """
        How much prediction variance is explained by features?

        Args:
            explained_variance: Variance from SHAP/important features
            total_variance: Total prediction variance

        Returns:
            Dict with coverage score
        """
        try:
            if total_variance <= 0:
                return {'coverage': 0.0, 'warning': 'Zero total variance'}

            coverage = explained_variance / total_variance
            coverage = np.clip(coverage, 0, 1)

            return {
                'coverage': float(coverage),
                'explained_variance': float(explained_variance),
                'total_variance': float(total_variance),
                'interpretation': 'good' if coverage > 0.7 else 'moderate' if coverage > 0.4 else 'low'
            }

        except Exception as e:
            logger.error(f"Coverage calculation failed: {e}")
            return {'error': str(e)}

    def compute_all_metrics(
        self,
        predictions: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        feature_importances: Optional[np.ndarray] = None,
        feature_rankings: Optional[List[List[str]]] = None,
        predictions_timeline: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute all explainability metrics.

        Args:
            predictions: Model predictions
            ground_truth: (Optional) Actual outcomes for calibration
            feature_importances: (Optional) Feature importance scores
            feature_rankings: (Optional) Top features per patient
            predictions_timeline: (Optional) Predictions over time

        Returns:
            Comprehensive metrics dict
        """
        results = {
            'timestamp': str(np.datetime64('now')),
            'metrics': {},
            'warnings': []
        }

        # 1. Calibration
        if ground_truth is not None:
            results['metrics']['calibration'] = self.calibration_analysis(predictions, ground_truth)
        else:
            results['warnings'].append('Calibration skipped (no ground truth)')

        # 2. Stability
        if feature_importances is not None:
            results['metrics']['stability'] = self.stability_analysis(predictions, feature_importances)
        else:
            results['warnings'].append('Stability skipped (no feature importances)')

        # 3. Trustworthiness
        if feature_rankings is not None:
            results['metrics']['trustworthiness'] = self.feature_trustworthiness(feature_rankings)
        else:
            results['warnings'].append('Trustworthiness skipped (no feature rankings)')

        # 4. Temporal Consistency
        if predictions_timeline is not None:
            results['metrics']['temporal_consistency'] = self.temporal_prediction_consistency(predictions_timeline)
        else:
            results['warnings'].append('Temporal consistency skipped (no timeline data)')

        return results

    @staticmethod
    def _calibration_interpretation(brier_score: float, ece: float) -> str:
        """Interpret calibration metrics."""
        if brier_score < 0.1 and ece < 0.05:
            return 'Excellent calibration'
        elif brier_score < 0.2 and ece < 0.1:
            return 'Good calibration'
        elif brier_score < 0.3 and ece < 0.15:
            return 'Moderate calibration'
        else:
            return 'Poor calibration - predictions not well-calibrated'

    @staticmethod
    def _stability_interpretation(stability_score: float) -> str:
        """Interpret stability score."""
        if stability_score > 0.8:
            return 'Very stable - explanations robust to perturbations'
        elif stability_score > 0.6:
            return 'Stable - generally robust'
        elif stability_score > 0.4:
            return 'Moderate stability - some sensitivity to noise'
        else:
            return 'Unstable - explanations sensitive to input variations'


if __name__ == '__main__':
    # Quick test
    np.random.seed(42)

    computer = ExplainabilityMetricsComputer()

    # Test data
    predictions = np.random.uniform(0, 1, 100)
    ground_truth = (np.random.uniform(0, 1, 100) > 0.5).astype(int)

    print("Calibration Analysis:")
    print(computer.calibration_analysis(predictions, ground_truth))

    print("\n\nAll Metrics:")
    result = computer.compute_all_metrics(
        predictions=predictions,
        ground_truth=ground_truth,
        feature_importances=np.random.rand(42)
    )
    print(result)

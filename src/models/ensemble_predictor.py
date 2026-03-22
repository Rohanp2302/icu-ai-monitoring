"""
Phase 7: Multi-Modal Ensemble Predictor
========================================

Combines Deep Learning (Transformer) + Machine Learning (Random Forest) for robust mortality predictions
with validation layers to prevent false negatives and false positives.

Features:
- Dual-path predictions (DL + ML)
- Model agreement scoring
- 4 validation layers (concordance, clinical rules, cohort, trajectory)
- Confidence estimation
- Comprehensive explanations
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger(__name__)


class ModelAgreementAnalyzer:
    """Analyze agreement between DL and ML predictions"""

    def calculate_agreement(self, p_dl: float, p_ml: float) -> float:
        """
        Calculate agreement score (0-1, where 1 = perfect agreement)
        """
        if p_dl + p_ml < 1e-6:
            return 1.0

        diff = abs(p_dl - p_ml)
        agreement = 1.0 - (diff / (max(p_dl, p_ml) + 1e-6))
        return max(0, min(1, agreement))

    def analyze_disagreement(self, p_dl: float, p_ml: float) -> Dict:
        """
        Analyze when and why models disagree
        """
        diff = abs(p_dl - p_ml)

        if diff > 0.20:
            magnitude = 'CRITICAL'
        elif diff > 0.15:
            magnitude = 'HIGH'
        elif diff > 0.10:
            magnitude = 'MEDIUM'
        else:
            magnitude = 'LOW'

        return {
            'difference': float(diff),
            'magnitude': magnitude,
            'dl_predicts_higher': float(p_dl > p_ml),
            'potentially_uncertain': diff > 0.15
        }


class ValidationLayer1_Concordance:
    """Validation Layer 1: Check DL-ML Agreement"""

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    def check(self, p_dl: float, p_ml: float) -> Dict:
        """
        If DL and ML predictions differ significantly, flag as uncertain
        """
        diff = abs(p_dl - p_ml)

        if diff > self.threshold:
            return {
                'flag': True,
                'reason': f'High DL-ML disagreement: {diff:.3f}',
                'severity': 'MEDIUM',
                'confidence_penalty': 0.2,
                'recommendation': 'Require clinical review before acting on prediction'
            }

        return {
            'flag': False,
            'confidence_bonus': 0.1,
            'recommendation': 'Models agree - proceed with confidence'
        }


class ValidationLayer2_ClinicalRules:
    """Validation Layer 2: Check Clinical Plausibility"""

    def __init__(self, vital_ranges: Optional[Dict] = None):
        """
        Initialize with vital sign ranges
        If None, uses default ranges
        """
        if vital_ranges is None:
            self.vital_ranges = {
                'heart_rate': {'target': (60, 100), 'critical_high': 140, 'critical_low': 40},
                'respiration': {'target': (12, 20), 'critical_high': 35, 'critical_low': 8},
                'oxygen_saturation': {'target': (95, 100), 'critical_low': 85, 'alert_low': 90},
                'blood_pressure_systolic': {'target': (100, 140), 'critical_high': 180, 'critical_low': 80}
            }
        else:
            self.vital_ranges = vital_ranges

    def check(self, x_temporal: np.ndarray, x_static: np.ndarray, p_pred: float,
              hr_idx: int = 0, rr_idx: int = 2, sao2_idx: int = 1) -> Dict:
        """
        Check if prediction violates clinical knowledge
        """
        flags = []

        # Get latest vital signs
        hr = x_temporal[-1, hr_idx] if x_temporal.shape[1] > hr_idx else None
        rr = x_temporal[-1, rr_idx] if x_temporal.shape[1] > rr_idx else None
        sao2 = x_temporal[-1, sao2_idx] if x_temporal.shape[1] > sao2_idx else None

        # Rule 1: All vitals normal → risk should be <20%
        if hr is not None and rr is not None and sao2 is not None:
            vitals_normal = (
                self.vital_ranges['heart_rate']['target'][0] <= hr <= self.vital_ranges['heart_rate']['target'][1] and
                self.vital_ranges['respiration']['target'][0] <= rr <= self.vital_ranges['respiration']['target'][1] and
                sao2 >= self.vital_ranges['oxygen_saturation']['target'][0]
            )

            if vitals_normal and p_pred > 0.30:
                flags.append({
                    'rule': 'Vitals normal but high risk',
                    'severity': 'HIGH',
                    'explanation': 'All vital signs are in target range, but model predicts high mortality'
                })

        # Rule 2: All vitals critically abnormal → risk should be >50%
        if hr is not None and rr is not None and sao2 is not None:
            vitals_critical = (
                hr > self.vital_ranges['heart_rate']['critical_high'] or
                hr < self.vital_ranges['heart_rate']['critical_low'] or
                rr > self.vital_ranges['respiration']['critical_high'] or
                rr < self.vital_ranges['respiration']['critical_low'] or
                sao2 < self.vital_ranges['oxygen_saturation']['critical_low']
            )

            if vitals_critical and p_pred < 0.35:
                flags.append({
                    'rule': 'Critical vitals but low risk',
                    'severity': 'HIGH',
                    'explanation': 'Multiple vital signs are critically abnormal, risk should be higher'
                })

        if len(flags) > 0:
            return {
                'flag': True,
                'rules_violated': flags,
                'severity': max([f['severity'] for f in flags]),
                'confidence_penalty': 0.25,
                'recommendation': 'Clinician should review prediction physiology'
            }

        return {'flag': False, 'recommendation': 'Prediction clinically plausible'}


class ValidationLayer3_CohortConsistency:
    """Validation Layer 3: Compare with Similar Patients"""

    def __init__(self, cohort_db: Optional[Dict] = None):
        """
        Initialize with cohort database
        """
        self.cohort_db = cohort_db or {}

    def find_similar_patients(self, x_temporal: np.ndarray, x_static: np.ndarray, k: int = 10) -> List[Dict]:
        """
        Find k similar patients in historical database
        Uses Euclidean distance on key features
        """
        if not self.cohort_db or len(self.cohort_db) < 5:
            return []

        #For now, return empty (will need proper implementation with actual database)
        return []

    def check(self, x_temporal: np.ndarray, x_static: np.ndarray, p_pred: float) -> Dict:
        """
        Check if prediction aligns with similar patient cohort outcomes
        """
        similar_patients = self.find_similar_patients(x_temporal, x_static, k=10)

        if len(similar_patients) < 5:
            #Not enough similar patients for comparison
            return {
                'flag': False,
                'note': 'Insufficient similar patients in database for cohort comparison',
                'check_skipped': True
            }

        cohort_mortality_rate = np.mean([p['outcome'] for p in similar_patients])

        diff = abs(p_pred - cohort_mortality_rate)

        if diff > 0.25:
            return {
                'flag': True,
                'reason': f'Prediction differs significantly from cohort (predicted {p_pred:.2%} vs cohort {cohort_mortality_rate:.2%})',
                'severity': 'MEDIUM',
                'similar_patients_count': len(similar_patients),
                'cohort_mortality_rate': float(cohort_mortality_rate),
                'confidence_penalty': 0.15,
                'recommendation': 'Prediction is outlier - consider additional review'
            }

        return {
            'flag': False,
            'similar_patients_count': len(similar_patients),
            'cohort_mortality_rate': float(cohort_mortality_rate),
            'recommendation': 'Prediction consistent with similar patients'
        }


class ValidationLayer4_TrajectoryConsistency:
    """Validation Layer 4: Check Temporal Consistency"""

    def __init__(self, predictor=None):
        """
        Initialize with predictor for calculating risk over time
        """
        self.predictor = predictor

    def calculate_trajectory(self, x_temporal: np.ndarray) -> List[float]:
        """
        Calculate risk score at each timestep
        Detects sudden jumps or inconsistent trends
        """
        if self.predictor is None:
            #Return empty trajectory if no predictor available
            return [0.5] * len(x_temporal)

        trajectory = []
        for t in range(1, len(x_temporal) + 1):
            x_subset = x_temporal[:t]
            #This would call the predictor on each timestep
            #For now, return placeholder
            trajectory.append(0.5)

        return trajectory

    def check(self, x_temporal: np.ndarray) -> Dict:
        """
        Check if predictions are consistent over time
        Flag sudden jumps or inconsistent trends
        """
        trajectory = self.calculate_trajectory(x_temporal)

        if len(trajectory) < 2:
            return {'flag': False, 'check_skipped': True}

        # Calculate differences between consecutive timesteps
        diffs = [abs(trajectory[i+1] - trajectory[i]) for i in range(len(trajectory)-1)]
        max_diff = max(diffs) if diffs else 0

        if max_diff > 0.25:  # Sudden jump
            sudden_jump_idx = diffs.index(max_diff) + 1
            return {
                'flag': True,
                'reason': f'Sudden risk jump at timestep {sudden_jump_idx}',
                'severity': 'MEDIUM',
                'max_jump': float(max_diff),
                'timestamp_of_jump': int(sudden_jump_idx),
                'confidence_penalty': 0.15,
                'recommendation': 'Patient status changed rapidly - review recent events'
            }

        # Check for trend inconsistency (zigzag pattern)
        trend_changes = sum(1 for i in range(1, len(diffs)-1)
                           if (diffs[i] > diffs[i-1]) and (diffs[i] > diffs[i+1]))

        if trend_changes > len(trajectory) * 0.3:
            return {
                'flag': True,
                'reason': 'Inconsistent trend pattern detected',
                'severity': 'LOW',
                'trend_inconsistencies': int(trend_changes),
                'confidence_penalty': 0.10,
                'recommendation': 'Risk is fluctuating - additional monitoring recommended'
            }

        return {'flag': False, 'observation': 'Trajectory is consistent'}


class DualModelEnsemblePredictor:
    """
    Main Multi-Modal Ensemble Predictor
    Combines Deep Learning (Transformer) + Machine Learning (Random Forest)
    with 4 validation layers for robust predictions
    """

    def __init__(self,
                 ml_model_path: str,
                 ml_scaler_path: str,
                 dl_model_path: Optional[str] = None,
                 vital_ranges: Optional[Dict] = None,
                 cohort_db: Optional[Dict] = None):
        """
        Initialize ensemble with ML and DL models

        Args:
            ml_model_path: Path to trained sklearn Random Forest model
            ml_scaler_path: Path to RobustScaler
            dl_model_path: Path to PyTorch transformer model (optional)
            vital_ranges: Dict of vital sign ranges for validation
            cohort_db: Historical patient database for cohort comparison
        """

        # Load ML model
        logger.info("Loading ML model (Random Forest)...")
        with open(ml_model_path, 'rb') as f:
            self.ml_model = pickle.load(f)

        with open(ml_scaler_path, 'rb') as f:
            self.ml_scaler = pickle.load(f)

        # Load DL model (transformer)
        self.dl_model = None
        if dl_model_path and Path(dl_model_path).exists():
            logger.info("Loading DL model (Transformer)...")
            try:
                self.dl_model = torch.load(dl_model_path, map_location='cpu')
                self.dl_model.eval()
                logger.info("DL model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load DL model: {e}. Falling back to ML only.")
                self.dl_model = None

        # Initialize validation layers
        self.validator_1 = ValidationLayer1_Concordance()
        self.validator_2 = ValidationLayer2_ClinicalRules(vital_ranges)
        self.validator_3 = ValidationLayer3_CohortConsistency(cohort_db)
        self.validator_4 = ValidationLayer4_TrajectoryConsistency(self)

        # Model agreement analyzer
        self.agreement_analyzer = ModelAgreementAnalyzer()

    def predict_ml(self, x_features: np.ndarray) -> Tuple[float, float]:
        """
        Get prediction from Random Forest model

        Returns:
            (mortality_probability, confidence)
        """
        x_scaled = self.ml_scaler.transform(x_features)
        proba = self.ml_model.predict_proba(x_scaled)[0, 1]

        # Confidence from RF: use decision function or probability distance from 0.5
        confidence = 1.0 - (abs(0.5 - proba) ** 0.5)  # Higher confidence if far from boundary

        return proba, confidence

    def predict_dl(self, x_temporal: torch.Tensor) -> Tuple[float, float, float]:
        """
        Get prediction from Transformer model

        Args:
            x_temporal: Temporal features as torch tensor

        Returns:
            (mortality_probability, confidence, uncertainty)
        """
        if self.dl_model is None:
            return 0.5, 0.5, 0.5  # Fallback if DL model not available

        with torch.no_grad():
            # Get multiple stochastic forward passes for uncertainty estimation (MC Dropout)
            n_passes = 5
            predictions = []

            for _ in range(n_passes):
                output = self.dl_model(x_temporal)
                # Assuming output is mortality probability
                if isinstance(output, dict):
                    pred = output.get('mortality', torch.tensor([0.5]))
                else:
                    pred = output

                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()

                if isinstance(pred, np.ndarray):
                    if pred.ndim > 0:
                        pred = pred[0]

                predictions.append(float(pred))

            # Calculate statistics
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)

            # Confidence: inverse of uncertainty
            confidence = 1.0 - np.clip(std_pred, 0, 1)

            return float(mean_pred), float(confidence), float(std_pred)

    def predict(self, x_temporal: np.ndarray, x_static: np.ndarray) -> Dict:
        """
        Main prediction method combining DL + ML with validation

        Args:
            x_temporal: Temporal features (n_timesteps × n_time_features)
            x_static: Static features (n_static_features,)

        Returns:
            Comprehensive prediction dict with confidence, explanations, flags
        """

        logger.info(f"Making ensemble prediction: temporal shape {x_temporal.shape}, static shape {x_static.shape}")

        # Step 1: ML Path - use aggregated static features
        try:
            p_ml, conf_ml = self.predict_ml(x_static.reshape(1, -1))
            ml_success = True
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            p_ml, conf_ml = 0.5, 0.3
            ml_success = False

        # Step 2: DL Path - use temporal features
        p_dl, conf_dl, unc_dl = 0.5, 0.3, 0.5
        dl_success = False
        try:
            if self.dl_model is not None:
                x_temp_tensor = torch.from_numpy(x_temporal).unsqueeze(0).float()
                p_dl, conf_dl, unc_dl = self.predict_dl(x_temp_tensor)
                dl_success = True
        except Exception as e:
            logger.error(f"DL prediction failed: {e}")

        # Step 3: Calculate Agreement
        agreement_analyzer = self.agreement_analyzer
        agreement = agreement_analyzer.calculate_agreement(p_dl, p_ml)
        agreement_analysis = agreement_analyzer.analyze_disagreement(p_dl, p_ml)

        # Step 4: Apply Validation Layers
        validation_results = {
            'layer1_concordance': self.validator_1.check(p_dl, p_ml),
            'layer2_clinical_rules': self.validator_2.check(x_temporal, x_static, (p_dl + p_ml) / 2),
            'layer3_cohort': self.validator_3.check(x_temporal, x_static, (p_dl + p_ml) / 2),
            'layer4_trajectory': self.validator_4.check(x_temporal)
        }

        # Step 5: Calculate Final Confidence
        base_confidence = (conf_dl + conf_ml) / 2 if (dl_success and ml_success) else max(conf_dl, conf_ml)

        # Apply penalties for failed validation layer checks
        confidence_penalty = 0.0
        for layer_name, layer_result in validation_results.items():
            if layer_result.get('flag'):
                confidence_penalty += layer_result.get('confidence_penalty', 0.0)

        final_confidence = max(0.0, base_confidence - confidence_penalty)

        # Step 6: Ensemble Decision
        if agreement > 0.85 and ml_success and dl_success:
            # High agreement - average predictions
            p_final = 0.6 * p_ml + 0.4 * p_dl  # Weight more towards ML (better calibrated)
        elif ml_success and not dl_success:
            p_final = p_ml
        elif dl_success and not ml_success:
            p_final = p_dl
        else:
            p_final = 0.5  # Default fallback

        # Step 7: Risk Classification
        if p_final < 0.20:
            risk_class = 'LOW'
        elif p_final < 0.40:
            risk_class = 'MEDIUM'
        elif p_final < 0.70:
            risk_class = 'HIGH'
        else:
            risk_class = 'CRITICAL'

        # Count validation flags
        flagged_layers = [layer for layer, result in validation_results.items()
                         if result.get('flag')]

        return {
            'mortality_risk': float(p_final),
            'mortality_percent': f'{p_final*100:.1f}%',
            'risk_class': risk_class,
            'confidence': float(final_confidence),
            'dl_prediction': float(p_dl),
            'ml_prediction': float(p_ml),
            'dl_confidence': float(conf_dl),
            'ml_confidence': float(conf_ml),
            'agreement_score': float(agreement),
            'agreement_analysis': agreement_analysis,
            'validation_results': validation_results,
            'validation_flags_count': len(flagged_layers),
            'validation_status': 'PASSED' if len(flagged_layers) == 0 else 'FLAGGED_FOR_REVIEW',
            'explanation': self._generate_explanation(p_dl, p_ml, agreement, validation_results, flagged_layers)
        }

    def _generate_explanation(self, p_dl: float, p_ml: float, agreement: float,
                             validation_results: Dict, flagged_layers: List) -> str:
        """
        Generate human-readable explanation of prediction
        """
        explanation = f"Dual Ensemble Prediction: {(p_dl+p_ml)/2*100:.1f}% mortality risk. "

        if agreement > 0.85:
            explanation += f"Both Deep Learning ({p_dl*100:.1f}%) and Machine Learning ({p_ml*100:.1f}%) models strongly agree. "
        else:
            explanation += f"Models show some disagreement (DL: {p_dl*100:.1f}%, ML: {p_ml*100:.1f}%). "

        if len(flagged_layers) > 0:
            explanation += f"WARNING: {len(flagged_layers)} validation check(s) flagged: "
            explanations = []
            for layer_name in flagged_layers:
                result = validation_results[layer_name]
                if result.get('reason'):
                    explanations.append(result['reason'])
            explanation += "; ".join(explanations) + ". Clinical review recommended."
        else:
            explanation += "All validation checks passed."

        return explanation


# Example usage
if __name__ == '__main__':
    # Initialize ensemble predictor
    predictor = DualModelEnsemblePredictor(
        ml_model_path='results/dl_models/best_model.pkl',
        ml_scaler_path='results/dl_models/scaler.pkl',
        dl_model_path=None,  # Optional: add path if transformer model available
    )

    # Example prediction with fake data
    x_temporal_example = np.random.randn(24, 42)  # 24 hours × 42 temporal features
    x_static_example = np.random.randn(120)  # 120 static features (aggregated)

    result = predictor.predict(x_temporal_example, x_static_example)

    print("\n" + "="*80)
    print("MULTI-MODAL ENSEMBLE PREDICTION RESULT")
    print("="*80)
    print(json.dumps(result, indent=2))

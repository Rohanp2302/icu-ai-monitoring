"""
Phase 5: SHAP Feature Importance Analysis

Explain model predictions using SHAP (SHapley Additive exPlanations):
- Global feature importance (which features drive predictions in general)
- Local explanation (why specific patient got specific prediction)
- Per-task breakdown (different features matter for different tasks)
"""

import numpy as np
import torch
import shap
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


# Feature name mapping for 42 engineered features
FEATURE_NAMES = [
    # Original 3 features
    'HR_raw', 'RR_raw', 'SaO2_raw',

    # 1st derivative (3 features)
    'HR_deriv1', 'RR_deriv1', 'SaO2_deriv1',

    # 2nd derivative (3 features)
    'HR_deriv2', 'RR_deriv2', 'SaO2_deriv2',

    # Smoothed values (3 features)
    'HR_smooth', 'RR_smooth', 'SaO2_smooth',

    # Cumulative statistics: mean, std, min, max, p25, p75, range for each feature (7*3=21)
    'HR_cumul_mean', 'HR_cumul_std', 'HR_cumul_min', 'HR_cumul_max', 'HR_cumul_p25', 'HR_cumul_p75', 'HR_cumul_range',
    'RR_cumul_mean', 'RR_cumul_std', 'RR_cumul_min', 'RR_cumul_max', 'RR_cumul_p25', 'RR_cumul_p75', 'RR_cumul_range',
    'SaO2_cumul_mean', 'SaO2_cumul_std', 'SaO2_cumul_min', 'SaO2_cumul_max', 'SaO2_cumul_p25', 'SaO2_cumul_p75', 'SaO2_cumul_range',

    # Therapeutic deviation (3 features)
    'HR_therapeutic_dev', 'RR_therapeutic_dev', 'SaO2_therapeutic_dev',

    # Volatility - rolling coefficient of variation (3 features)
    'HR_volatility', 'RR_volatility', 'SaO2_volatility',
]

assert len(FEATURE_NAMES) == 42, f"Expected 42 feature names, got {len(FEATURE_NAMES)}"


class SHAPExplainer:
    """SHAP-based feature importance for ICU model predictions"""

    def __init__(self, model, device: str = "cpu"):
        """
        Args:
            model: MultiTaskICUModel instance
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device
        self.model.eval()

    def _model_wrapper(self, x_temporal_np: np.ndarray) -> np.ndarray:
        """
        Wrapper for SHAP to call model.
        SHAP needs a function that takes numpy array and returns predictions.

        Args:
            x_temporal_np: (N, 24, 42) numpy array

        Returns:
            (N, output_dim) predictions for the target task
        """
        x_temporal = torch.tensor(x_temporal_np, dtype=torch.float32).to(self.device)
        # Static features: we'll use zeros for now (can be extended)
        batch_size = x_temporal.shape[0]
        x_static = torch.zeros(batch_size, 20).to(self.device)

        with torch.no_grad():
            outputs = self.model(x_temporal, x_static)

        # Return mortality predictions for now
        return outputs['mortality'].cpu().numpy()

    def explain_global_mortality(
        self,
        background_data: np.ndarray,
        n_samples: int = 100
    ) -> Dict:
        """
        Global SHAP explanation for mortality prediction.

        Args:
            background_data: (N_bg, 24, 42) background samples for SHAP baseline
            n_samples: Number of samples for SHAP calculation

        Returns:
            Dict with global feature importance
        """
        # Flatten temporal dimension for SHAP
        # SHAP works on flattened features: (N, 24*42) = (N, 1008)
        N_bg = background_data.shape[0]
        background_flat = background_data.reshape(N_bg, -1)

        # Create explainer
        explainer = shap.KernelExplainer(
            self._model_wrapper_flat,
            background_flat[:min(50, N_bg)]  # Use max 50 background samples
        )

        # Compute SHAP values
        shap_values = explainer.shap_values(background_flat, nsamples=n_samples)

        # Aggregate importance across samples
        global_importance = np.abs(shap_values).mean(axis=0)  # (1008,)

        # Reshape back to features
        feature_importance = global_importance.reshape(24, 42)  # (24, 42)

        # Average across timesteps
        feature_importance_avg = feature_importance.mean(axis=0)  # (42,)

        # Sort by importance
        top_indices = np.argsort(feature_importance_avg)[::-1]

        result = {
            'global_importance': feature_importance_avg.tolist(),
            'top_10_features': [
                {
                    'rank': i + 1,
                    'feature_name': FEATURE_NAMES[idx],
                    'feature_idx': int(idx),
                    'importance': float(feature_importance_avg[idx])
                }
                for i, idx in enumerate(top_indices[:10])
            ],
            'temporal_patterns': {
                'early_predictors': feature_importance[0, :].tolist(),  # Hour 0
                'late_predictors': feature_importance[-1, :].tolist(),   # Hour 23
                'avg_importance': feature_importance.mean(axis=0).tolist()
            }
        }

        return result

    def _model_wrapper_flat(self, x_flat: np.ndarray) -> np.ndarray:
        """Wrapper for SHAP with flattened features"""
        batch_size = x_flat.shape[0]
        x_temporal_np = x_flat.reshape(batch_size, 24, 42)
        return self._model_wrapper(x_temporal_np)

    def explain_patient(
        self,
        x_temporal: np.ndarray,
        x_static: np.ndarray,
        background_data: np.ndarray,
        n_samples: int = 50
    ) -> Dict:
        """
        Local SHAP explanation for a specific patient.

        Args:
            x_temporal: (1, 24, 42) patient temporal features
            x_static: (1, 20) patient static features
            background_data: (N_bg, 24, 42) background data
            n_samples: SHAP samples

        Returns:
            Dict with patient-specific explanations
        """
        # Flatten for SHAP
        x_flat = x_temporal.reshape(1, -1)
        N_bg = background_data.shape[0]
        background_flat = background_data.reshape(N_bg, -1)

        # Create explainer
        explainer = shap.KernelExplainer(
            self._model_wrapper_flat,
            background_flat[:min(50, N_bg)]
        )

        # Compute SHAP for this patient
        shap_values = explainer.shap_values(x_flat, nsamples=n_samples)[0]  # (1008,)

        # Reshape to features
        shap_values_features = shap_values.reshape(24, 42)  # (24, 42)

        # Average across timesteps
        shap_avg = np.abs(shap_values_features).mean(axis=0)  # (42,)

        # Top contributing features
        top_indices = np.argsort(shap_avg)[::-1]

        # Get model prediction
        x_temporal_t = torch.tensor(x_temporal, dtype=torch.float32).to(self.device)
        x_static_t = torch.tensor(x_static, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(x_temporal_t, x_static_t)

        mortality_prob = outputs['mortality'].item()

        result = {
            'patient_mortality_prob': float(mortality_prob),
            'top_contributing_features': [
                {
                    'rank': i + 1,
                    'feature_name': FEATURE_NAMES[idx],
                    'feature_idx': int(idx),
                    'shap_value': float(shap_avg[idx]),
                    'direction': 'increases' if shap_values[idx] > 0 else 'decreases'
                }
                for i, idx in enumerate(top_indices[:10])
            ],
            'temporal_importance': {
                f'hour_{h}': shap_avg_h.tolist()
                for h, shap_avg_h in enumerate(np.abs(shap_values_features))
            }
        }

        return result

    def get_risk_factors(
        self,
        x_temporal: np.ndarray,
        x_static: np.ndarray,
        top_k: int = 5
    ) -> List[str]:
        """
        Extract human-readable risk factors for a patient.

        Args:
            x_temporal: (1, 24, 42) patient temporal features
            x_static: (1, 20) patient static features
            top_k: Number of risk factors to return

        Returns:
            List of human-readable risk factor descriptions
        """
        # Get raw values
        hr_current = x_temporal[0, -1, 0]  # Latest heart rate
        rr_current = x_temporal[0, -1, 1]  # Latest respiration
        sao2_current = x_temporal[0, -1, 2]  # Latest SaO2

        # Get volatility
        hr_volatility = x_temporal[0, -1, 39]  # HR volatility index
        rr_volatility = x_temporal[0, -1, 40]  # RR volatility index

        # Get therapeutic deviation
        hr_dev = x_temporal[0, -1, 36]  # HR therapeutic deviation
        rr_dev = x_temporal[0, -1, 37]  # RR therapeutic deviation

        risk_factors = []

        # Clinical rules
        if hr_current > 110:
            risk_factors.append(f"Tachycardia (HR={hr_current:.0f} bpm)")
        if rr_current > 22:
            risk_factors.append(f"Tachypnea (RR={rr_current:.0f} breaths/min)")
        if sao2_current < 92:
            risk_factors.append(f"Hypoxemia (SaO2={sao2_current:.1f}%)")
        if hr_volatility > np.nanpercentile(x_temporal[:, :, 39], 75):
            risk_factors.append("High heart rate variability")
        if rr_volatility > np.nanpercentile(x_temporal[:, :, 40], 75):
            risk_factors.append("High respiratory rate variability")

        return risk_factors[:top_k]


class AttentionExplainer:
    """Extract and explain Transformer attention patterns"""

    def __init__(self, model, device: str = "cpu"):
        """
        Args:
            model: MultiTaskICUModel instance
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device
        self.model.eval()

    def get_attention_weights(
        self,
        x_temporal: np.ndarray,
        x_static: np.ndarray
    ) -> Dict:
        """
        Extract attention weights from Transformer encoder.

        Args:
            x_temporal: (batch_size, 24, 42) temporal features
            x_static: (batch_size, 20) static features

        Returns:
            Dict with attention patterns
        """
        x_temporal_t = torch.tensor(x_temporal, dtype=torch.float32).to(self.device)
        x_static_t = torch.tensor(x_static, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # Encode temporal
            temporal_encoded = self.model.temporal_encoder(x_temporal_t)  # (B, T, d_model)

            # Get attention from transformer
            # Note: PyTorch TransformerEncoder doesn't expose attention directly,
            # so we need to modify the model or use a workaround
            # For now, we'll use temporal pooling attention

            # Temporal pooling attention
            attn_weights = torch.softmax(
                self.model.temporal_pooling.attention(temporal_encoded), dim=1
            ).squeeze(-1)  # (B, T)

        return {
            'attention_weights': attn_weights.cpu().numpy().tolist(),
            'temporal_importance': {
                f'hour_{h}': float(attn_weights[0, h].cpu())
                for h in range(24)
            },
            'most_important_hours': sorted(
                [(h, float(attn_weights[0, h].cpu())) for h in range(24)],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 5: SHAP EXPLAINER - TEST")
    print("=" * 80)

    from src.models.multitask_model import create_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    model, _ = create_model(device=device)
    print(f"Model created successfully")

    # Create test data
    batch_size = 10
    background_size = 50
    x_bg = np.random.randn(background_size, 24, 42).astype(np.float32)
    x_test = np.random.randn(batch_size, 24, 42).astype(np.float32)
    x_static_test = np.random.randn(batch_size, 20).astype(np.float32)

    # Create explainer
    explainer = SHAPExplainer(model, device=device)

    # Get global importance
    print("\nComputing global feature importance (this may take a moment)...")
    global_importance = explainer.explain_global_mortality(x_bg, n_samples=30)
    print("Top 10 features:")
    for feat in global_importance['top_10_features']:
        print(f"  {feat['rank']:2d}. {feat['feature_name']:25} (importance: {feat['importance']:.4f})")

    # Get patient explanation
    print("\nExplaining individual patient...")
    patient_explanation = explainer.explain_patient(x_test[0:1], x_static_test[0:1], x_bg)
    print(f"Patient mortality probability: {patient_explanation['patient_mortality_prob']:.4f}")
    print("Top contributing features:")
    for feat in patient_explanation['top_contributing_features']:
        print(f"  {feat['rank']:2d}. {feat['feature_name']:25} (SHAP: {feat['shap_value']:.4f})")

    # Get attention
    print("\nExtracting attention patterns...")
    attention_explainer = AttentionExplainer(model, device=device)
    attention = attention_explainer.get_attention_weights(x_test[0:1], x_static_test[0:1])
    print("Most important hours:")
    for h, weight in attention['most_important_hours']:
        print(f"  Hour {h:2d}: {weight:.4f}")

    print("\n" + "=" * 80)
    print("[SUCCESS] SHAP explainer tested successfully")
    print("=" * 80)

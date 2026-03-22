"""
Phase 5: Clinical Interpreter

Unified interface for generating clinical explanations.
Combines SHAP, attention, and rule-based explanations.
"""

import numpy as np
import torch
from typing import Dict, List, Optional
import json
from datetime import datetime

from .shap_explainer import SHAPExplainer, AttentionExplainer, FEATURE_NAMES
from .rule_extractor import RuleExtractor


class ClinicalInterpreter:
    """High-level interface for clinical explanations"""

    def __init__(self, model, device: str = "cpu"):
        """
        Args:
            model: MultiTaskICUModel instance
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device
        self.shap_explainer = SHAPExplainer(model, device)
        self.attention_explainer = AttentionExplainer(model, device)
        self.rule_extractor = RuleExtractor()

    def explain_prediction(
        self,
        patient_id: str,
        x_temporal: np.ndarray,
        x_static: np.ndarray,
        background_data: Optional[np.ndarray] = None,
        include_shap: bool = True,
        include_attention: bool = True,
        include_rules: bool = True
    ) -> Dict:
        """
        Generate comprehensive explanation for a patient's prediction.

        Args:
            patient_id: Unique patient identifier
            x_temporal: (1, 24, 42) temporal features
            x_static: (1, 20) static features
            background_data: (N_bg, 24, 42) background data for SHAP
            include_shap: Whether to compute SHAP explanations
            include_attention: Whether to extract attention patterns
            include_rules: Whether to extract clinical rules

        Returns:
            Comprehensive explanation dict
        """
        # Get model predictions
        x_temporal_t = torch.tensor(x_temporal, dtype=torch.float32).to(self.device)
        x_static_t = torch.tensor(x_static, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(x_temporal_t, x_static_t)

        mortality_pred = outputs['mortality'].item()
        risk_probs = outputs['risk'][0].cpu().numpy()
        risk_class = np.argmax(risk_probs)
        outcomes_pred = outputs['outcomes'][0].cpu().numpy()

        explanation = {
            'patient_id': patient_id,
            'timestamp': datetime.now().isoformat(),
            'predictions': {
                'mortality': float(mortality_pred),
                'mortality_percent': f"{mortality_pred*100:.1f}%",
                'risk_class': int(risk_class),
                'risk_class_name': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'][risk_class],
                'risk_probabilities': {
                    'low': float(risk_probs[0]),
                    'medium': float(risk_probs[1]),
                    'high': float(risk_probs[2]),
                    'critical': float(risk_probs[3])
                },
                'clinical_outcomes': {
                    'sepsis': float(outcomes_pred[0]),
                    'aki': float(outcomes_pred[1]),
                    'ards': float(outcomes_pred[2]),
                    'shock': float(outcomes_pred[3]),
                    'mods': float(outcomes_pred[4]),
                    'arf': float(outcomes_pred[5])
                }
            }
        }

        # SHAP explanations
        if include_shap and background_data is not None:
            try:
                shap_result = self.shap_explainer.explain_patient(
                    x_temporal, x_static, background_data, n_samples=30
                )
                explanation['shap'] = shap_result
            except Exception as e:
                explanation['shap_error'] = str(e)

        # Attention patterns
        if include_attention:
            try:
                attention_result = self.attention_explainer.get_attention_weights(
                    x_temporal, x_static
                )
                explanation['attention'] = attention_result
            except Exception as e:
                explanation['attention_error'] = str(e)

        # Clinical rules
        if include_rules:
            # Get organ status
            organ_status = self.rule_extractor.get_organ_status(
                x_temporal[0], outcomes_pred
            )

            # Extract rules
            vital_rules = self.rule_extractor.extract_vital_rules(
                x_temporal[0], mortality_pred, risk_class
            )
            trajectory_rules = self.rule_extractor.extract_trajectory_rules(
                x_temporal[0], mortality_pred
            )

            # Summary
            summary = self.rule_extractor.generate_summary(
                x_temporal[0], mortality_pred, risk_class, outcomes_pred, organ_status
            )

            explanation['clinical'] = {
                'organ_status': organ_status,
                'vital_rules': [
                    {
                        'condition': r.condition,
                        'consequence': r.consequence,
                        'confidence': float(r.confidence),
                        'specificity': float(r.specificity),
                        'frequency': float(r.frequency)
                    }
                    for r in vital_rules
                ],
                'trajectory_rules': [
                    {
                        'condition': r.condition,
                        'consequence': r.consequence,
                        'confidence': float(r.confidence),
                        'specificity': float(r.specificity),
                        'frequency': float(r.frequency)
                    }
                    for r in trajectory_rules
                ],
                'summary': summary
            }

        return explanation

    def get_risk_dashboard_data(
        self,
        patient_id: str,
        x_temporal: np.ndarray,
        x_static: np.ndarray
    ) -> Dict:
        """
        Get simplified data for dashboard display.

        Args:
            patient_id: Patient ID
            x_temporal: (1, 24, 42) temporal features
            x_static: (1, 20) static features

        Returns:
            Dashboard-friendly dict
        """
        # Get predictions
        x_temporal_t = torch.tensor(x_temporal, dtype=torch.float32).to(self.device)
        x_static_t = torch.tensor(x_static, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(x_temporal_t, x_static_t)

        mortality_pred = outputs['mortality'].item()
        risk_probs = outputs['risk'][0].cpu().numpy()
        risk_class = np.argmax(risk_probs)
        outcomes_pred = outputs['outcomes'][0].cpu().numpy()

        # Get organ status
        organ_status = self.rule_extractor.get_organ_status(
            x_temporal[0], outcomes_pred
        )

        # Extract top risk factors
        vital_rules = self.rule_extractor.extract_vital_rules(
            x_temporal[0], mortality_pred, risk_class
        )

        dashboard_data = {
            'patient_id': patient_id,
            'mortality_risk': f"{mortality_pred*100:.1f}%",
            'risk_level': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'][risk_class],
            'risk_score': float(risk_probs[risk_class]),
            'organ_status': organ_status,
            'top_risk_factors': [r.condition for r in vital_rules[:3]],
            'critical_findings': [
                f"{r.condition}: {r.consequence}"
                for r in vital_rules if r.confidence > 0.7
            ],
            'clinical_outcomes': {
                'sepsis_risk': f"{outcomes_pred[0]*100:.0f}%",
                'aki_risk': f"{outcomes_pred[1]*100:.0f}%",
                'ards_risk': f"{outcomes_pred[2]*100:.0f}%",
                'shock_risk': f"{outcomes_pred[3]*100:.0f}%"
            }
        }

        return dashboard_data

    def export_explanation_json(
        self,
        explanation: Dict,
        filepath: str
    ) -> None:
        """Export explanation as JSON"""
        with open(filepath, 'w') as f:
            json.dump(explanation, f, indent=2, default=str)

    def export_explanation_html(
        self,
        explanation: Dict,
        filepath: str
    ) -> None:
        """Export explanation as HTML report"""
        html_content = self._generate_html_report(explanation)
        with open(filepath, 'w') as f:
            f.write(html_content)

    def _generate_html_report(self, explanation: Dict) -> str:
        """Generate HTML report from explanation"""
        patient_id = explanation.get('patient_id', 'Unknown')
        pred = explanation.get('predictions', {})
        clinical = explanation.get('clinical', {})

        mortality = pred.get('mortality_percent', 'N/A')
        risk_level = pred.get('risk_class_name', 'Unknown')

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ICU Patient Explanation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #1392ec; color: white; padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #1392ec; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .risk-high {{ color: #ef4444; font-weight: bold; }}
                .risk-medium {{ color: #f59e0b; font-weight: bold; }}
                .risk-low {{ color: #10b981; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f0f0f0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ICU Patient Explanation Report</h1>
                <p>Patient ID: {patient_id}</p>
            </div>

            <div class="section">
                <h2>Risk Summary</h2>
                <div class="metric">
                    <strong>Mortality Risk:</strong>
                    <span class="risk-high">{mortality}</span>
                </div>
                <div class="metric">
                    <strong>Risk Level:</strong>
                    <span class="risk-high">{risk_level}</span>
                </div>
            </div>

            <div class="section">
                <h2>Organ Status</h2>
                <table>
                    <tr>
                        <th>Organ System</th>
                        <th>Status</th>
                    </tr>
        """

        for organ, status in clinical.get('organ_status', {}).items():
            html += f"""
                    <tr>
                        <td>{organ.capitalize()}</td>
                        <td>{status}</td>
                    </tr>
            """

        html += """
                </table>
            </div>

            <div class="section">
                <h2>Clinical Summary</h2>
                <pre>""" + clinical.get('summary', 'N/A') + """</pre>
            </div>

            <div class="section">
                <h2>Risk Factors</h2>
                <ul>
        """

        for rule in clinical.get('vital_rules', [])[:5]:
            html += f"""
                    <li>{rule['condition']}</li>
            """

        html += """
                </ul>
            </div>

        </body>
        </html>
        """

        return html


if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 5: CLINICAL INTERPRETER - TEST")
    print("=" * 80)

    from src.models.multitask_model import create_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    model, _ = create_model(device=device)
    interpreter = ClinicalInterpreter(model, device=device)

    # Test data
    x_temporal = np.random.randn(1, 24, 42).astype(np.float32)
    x_temporal[0, :, 0] = np.random.normal(95, 15, 24)  # HR
    x_temporal[0, :, 1] = np.random.normal(18, 4, 24)   # RR
    x_temporal[0, :, 2] = np.random.normal(96, 2, 24)   # SaO2

    x_static = np.random.randn(1, 20).astype(np.float32)

    # Without SHAP (faster)
    print("\nGenerating clinical explanation (without SHAP for speed)...")
    explanation = interpreter.explain_prediction(
        patient_id="P123456",
        x_temporal=x_temporal,
        x_static=x_static,
        include_shap=False,
        include_attention=True,
        include_rules=True
    )

    print(f"Mortality Risk: {explanation['predictions']['mortality_percent']}")
    print(f"Risk Level: {explanation['predictions']['risk_class_name']}")

    if 'clinical' in explanation:
        print("\nOrgan Status:")
        for organ, status in explanation['clinical']['organ_status'].items():
            print(f"  {organ}: {status}")

    # Dashboard data
    print("\nGenerating dashboard data...")
    dashboard = interpreter.get_risk_dashboard_data(
        patient_id="P123456",
        x_temporal=x_temporal,
        x_static=x_static
    )
    print(f"Dashboard data keys: {list(dashboard.keys())}")

    print("\n" + "=" * 80)
    print("[SUCCESS] Clinical interpreter tested successfully")
    print("=" * 80)

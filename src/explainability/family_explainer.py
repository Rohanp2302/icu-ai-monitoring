"""Family-Friendly Explanation Engine - Phase 9"""
from typing import Dict, List
import json

class FamilyExplainerEngine:
    """Generate family-friendly explanations for ICU predictions"""

    def __init__(self):
        self.risk_explanations = {
            'LOW': {
                'title': 'GOOD NEWS - Low Risk',
                'message': 'Your loved one is responding well',
                'details': 'Recovery is very likely'
            },
            'MEDIUM': {
                'title': 'MODERATE RISK',
                'message': 'Needs close attention',
                'details': 'Recovery is possible with treatment'
            },
            'HIGH': {
                'title': 'SERIOUS - Intensive Care',
                'message': 'Seriously ill - needs advanced care',
                'details': 'Critical condition'
            },
            'CRITICAL': {
                'title': 'CRITICAL - Emergency Care',
                'message': 'In critical condition',
                'details': 'Maximum support needed'
            }
        }

    def explain_risk_level(self, risk_class: str, mortality_percent: float) -> Dict:
        """Generate family explanation of risk"""
        if risk_class not in self.risk_explanations:
            risk_class = 'MEDIUM'
        exp = self.risk_explanations[risk_class].copy()
        exp['risk_class'] = risk_class
        exp['mortality_percent'] = f'{mortality_percent:.1f}%'
        return exp

    def explain_prediction_for_family(self, result: Dict) -> Dict:
        """Create comprehensive family explanation"""
        return {
            'main_message': self.explain_risk_level(
                result.get('risk_class', 'MEDIUM'),
                result.get('mortality_risk', 0.5)
            ),
            'hope': 'People recover even from serious conditions'
        }

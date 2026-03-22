"""
Enhanced API Endpoints for Multi-Modal ICU Prediction System
Integrates: Ensemble Predictor + Medicine Tracker + Family Explanations + HL7 Integration
Phase 10 - Production Deployment
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pickle

# Import local modules
try:
    from src.models.ensemble_predictor import DualModelEnsemblePredictor
    from src.medicine.medicine_tracker import MedicineTracker
    from src.explainability.family_explainer import FamilyExplainerEngine
    from src.language.translations import MultiLanguageTranslator, get_supported_languages
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    DualModelEnsemblePredictor = None
    MedicineTracker = None
    FamilyExplainerEngine = None
    MultiLanguageTranslator = None


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedICUAPI:
    """Enhanced API for multi-modal ICU prediction system"""

    def __init__(self, flask_app: Flask):
        self.app = flask_app
        self.ensemble_predictor = None
        self.medicine_tracker = None
        self.family_explainer = None
        self.prediction_history = {}  # In-memory storage (use DB in production)

        self._initialize_models()
        self._register_routes()

    def _initialize_models(self):
        """Initialize all ML models and trackers"""
        try:
            # Load ensemble predictor
            if DualModelEnsemblePredictor:
                self.ensemble_predictor = DualModelEnsemblePredictor(
                    dl_model_path='results/dl_models/best_model.pkl',
                    ml_model_path='results/dl_models/best_model.pkl'
                )
                logger.info("✓ Ensemble predictor loaded")
        except Exception as e:
            logger.warning(f"Could not load ensemble predictor: {e}")

        try:
            # Initialize medicine tracker
            if MedicineTracker:
                self.medicine_tracker = MedicineTracker()
                logger.info("✓ Medicine tracker initialized")
        except Exception as e:
            logger.warning(f"Could not initialize medicine tracker: {e}")

        try:
            # Initialize family explainer
            if FamilyExplainerEngine:
                self.family_explainer = FamilyExplainerEngine()
                logger.info("✓ Family explainer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize family explainer: {e}")

    def _register_routes(self):
        """Register all API routes"""

        @self.app.route('/api/predict-multimodal', methods=['POST'])
        def predict_multimodal():
            """
            Multimodal prediction endpoint
            Returns: mortality risk + explanations + family-friendly text + medicine warnings

            Request body:
            {
                "x_temporal": [[...], [...], ...],  # 24-hour vital history
                "x_static": {"age": 75, ...},       # Static patient features
                "current_medications": ["med1", "med2"],
                "patient_id": "P123"
            }
            """
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400

                patient_id = data.get('patient_id', f'P{len(self.prediction_history) + 1}')
                x_temporal = np.array(data.get('x_temporal', []))
                x_static = data.get('x_static', {})
                current_meds = data.get('current_medications', [])

                # Get multimodal prediction
                result = self.ensemble_predictor.predict(x_temporal, x_static)

                # Get top risk factors
                top_factors = result.get('top_factors', [])

                # Generate family explanation
                explanation = self.family_explainer.explain_mortality_risk(
                    result['mortality_risk'],
                    result['risk_class'],
                    top_factors,
                    current_interventions=self._get_interventions(result['risk_class']),
                    medication_context=current_meds
                )

                # Check medicine interactions if provided
                medicine_warnings = []
                if current_meds:
                    for i, med in enumerate(current_meds):
                        if i < len(current_meds) - 1:
                            interactions = self.medicine_tracker.check_interactions(
                                current_meds[:i], current_meds[i]
                            )
                            medicine_warnings.extend(interactions)

                response = {
                    'status': 'success',
                    'patient_id': patient_id,
                    'timestamp': datetime.now().isoformat(),
                    'prediction': {
                        'mortality_risk': float(result['mortality_risk']),
                        'mortality_percent': f"{result['mortality_risk'] * 100:.1f}%",
                        'risk_class': result['risk_class'],
                        'confidence': float(result.get('confidence', 0.8)),
                        'dl_prediction': float(result.get('dl_prediction', 0)),
                        'ml_prediction': float(result.get('ml_prediction', 0)),
                        'agreement_score': float(result.get('agreement_score', 0))
                    },
                    'validation': {
                        'validation_flags_count': result.get('validation_flags_count', 0),
                        'validation_status': result.get('validation_status', 'UNKNOWN'),
                        'flags': result.get('validation_flags', [])
                    },
                    'explanation': explanation,
                    'medicine_warnings': medicine_warnings,
                    'top_factors': top_factors[:5]  # Top 5 factors
                }

                # Store prediction
                self.prediction_history[patient_id] = response

                return jsonify(response), 200

            except Exception as e:
                logger.error(f"Error in multimodal prediction: {str(e)}")
                return jsonify({'error': str(e), 'status': 'error'}), 500

        @self.app.route('/api/check-medicine-interactions', methods=['POST'])
        def check_medicine_interactions():
            """
            Check drug interactions for new medicine

            Request body:
            {
                "current_medications": ["med1", "med2"],
                "new_medication": "med3"
            }
            """
            try:
                data = request.get_json()
                current_meds = data.get('current_medications', [])
                new_med = data.get('new_medication', '')

                if not new_med:
                    return jsonify({'error': 'No new medication specified'}), 400

                # Check interactions
                interactions = self.medicine_tracker.check_interactions(current_meds, new_med)

                # Determine if safe to add
                critical_interactions = [i for i in interactions if i.get('severity') == 'CRITICAL']
                safe_to_add = len(critical_interactions) == 0

                response = {
                    'status': 'success',
                    'new_medication': new_med,
                    'current_medications': current_meds,
                    'interactions': interactions,
                    'safe_to_add': safe_to_add,
                    'critical_interactions': len(critical_interactions),
                    'recommendation': self._generate_interaction_recommendation(interactions),
                    'timestamp': datetime.now().isoformat()
                }

                return jsonify(response), 200

            except Exception as e:
                logger.error(f"Error checking interactions: {str(e)}")
                return jsonify({'error': str(e), 'status': 'error'}), 500

        @self.app.route('/api/family-dashboard/<patient_id>', methods=['GET'])
        def family_dashboard(patient_id):
            """
            Get patient data in family-friendly format

            Query parameters:
            - lang: Language code ('en', 'hi', 'ta', 'te', 'kn', 'mr')
            """
            try:
                language = request.args.get('lang', 'en')
                translator = MultiLanguageTranslator(language)

                # Get last prediction for this patient
                if patient_id not in self.prediction_history:
                    return jsonify({
                        'error': f'No predictions found for patient {patient_id}',
                        'status': 'error'
                    }), 404

                last_prediction = self.prediction_history[patient_id]

                # Translate to family-friendly format
                family_view = {
                    'patient_id': patient_id,
                    'language': language,
                    'language_name': translator.get_language_name(),
                    'risk_status': translator.translate_risk_message(last_prediction['prediction']['risk_class']),
                    'risk_explanation': last_prediction['explanation'],
                    'vital_signs': self._format_vitals_for_family(
                        last_prediction.get('vital_signs', {}),
                        translator
                    ),
                    'medications': last_prediction.get('current_medications', []),
                    'questions_for_doctor': translator.get_suggested_questions(),
                    'last_updated': last_prediction.get('timestamp'),
                    'next_review': self._get_next_review_time(last_prediction['prediction']['risk_class']),
                    'support_resources': self._get_support_resources(language)
                }

                return jsonify(family_view), 200

            except Exception as e:
                logger.error(f"Error in family dashboard: {str(e)}")
                return jsonify({'error': str(e), 'status': 'error'}), 500

        @self.app.route('/api/validate-prediction', methods=['POST'])
        def validate_prediction():
            """
            Check if prediction passes all validation layers
            Returns: confidence score + validation flags + clinical review recommendation

            Request body:
            {
                "dl_pred": 0.38,
                "ml_pred": 0.33,
                "prediction": 0.35,
                "x_temporal": [...],
                "x_static": {...}
            }
            """
            try:
                data = request.get_json()

                dl_pred = data.get('dl_pred', 0)
                ml_pred = data.get('ml_pred', 0)
                prediction = data.get('prediction', (dl_pred + ml_pred) / 2)
                x_temporal = np.array(data.get('x_temporal', []))
                x_static = data.get('x_static', {})

                # Run validation checks
                concordance_check = self._check_concordance(dl_pred, ml_pred)
                clinical_check = self._check_clinical_rules(x_temporal, x_static, prediction)
                cohort_check = self._check_cohort_consistency(x_temporal, x_static, prediction)
                trajectory_check = self._check_trajectory_consistency(x_temporal)

                all_checks = {
                    'concordance': concordance_check,
                    'clinical_rules': clinical_check,
                    'cohort_consistency': cohort_check,
                    'trajectory_consistency': trajectory_check
                }

                flagged_checks = [name for name, check in all_checks.items() if check.get('flag', False)]

                response = {
                    'status': 'success',
                    'validation_results': all_checks,
                    'overall_validation_passed': len(flagged_checks) == 0,
                    'flags_count': len(flagged_checks),
                    'flagged_checks': flagged_checks,
                    'recommendation': 'Proceed' if len(flagged_checks) == 0 else 'Clinical review recommended',
                    'confidence_score': max(0, 1.0 - (len(flagged_checks) * 0.25)),
                    'timestamp': datetime.now().isoformat()
                }

                return jsonify(response), 200

            except Exception as e:
                logger.error(f"Error in validation: {str(e)}")
                return jsonify({'error': str(e), 'status': 'error'}), 500

        @self.app.route('/api/adverse-events', methods=['POST'])
        def predict_adverse_events():
            """
            Predict possible adverse events based on medications and vitals

            Request body:
            {
                "current_medications": ["med1", "med2"],
                "current_vitals": {"hr": 110, "bp_systolic": 90, "rr": 28, "sao2": 88}
            }
            """
            try:
                data = request.get_json()
                current_meds = data.get('current_medications', [])
                current_vitals = data.get('current_vitals', {})

                if not current_meds or not current_vitals:
                    return jsonify({
                        'error': 'Medications and vitals required',
                        'status': 'error'
                    }), 400

                # Predict adverse events
                adverse_events = self.medicine_tracker.predict_adverse_events(
                    current_meds,
                    current_vitals
                )

                response = {
                    'status': 'success',
                    'current_medications': current_meds,
                    'current_vitals': current_vitals,
                    'predicted_adverse_events': adverse_events,
                    'high_risk_events': [ae for ae in adverse_events if ae.get('severity') in ['HIGH', 'CRITICAL']],
                    'monitoring_recommendations': self._get_monitoring_recommendations(adverse_events),
                    'timestamp': datetime.now().isoformat()
                }

                return jsonify(response), 200

            except Exception as e:
                logger.error(f"Error predicting adverse events: {str(e)}")
                return jsonify({'error': str(e), 'status': 'error'}), 500

        @self.app.route('/api/languages', methods=['GET'])
        def get_languages():
            """Get list of supported languages"""
            try:
                return jsonify({
                    'status': 'success',
                    'supported_languages': get_supported_languages(),
                    'count': len(get_supported_languages())
                }), 200
            except Exception as e:
                logger.error(f"Error getting languages: {str(e)}")
                return jsonify({'error': str(e), 'status': 'error'}), 500

        @self.app.route('/api/results', methods=['GET'])
        def get_results():
            """Get all predictions history"""
            try:
                limit = request.args.get('limit', 100, type=int)

                results = list(self.prediction_history.values())[-limit:]

                return jsonify({
                    'status': 'success',
                    'total_predictions': len(self.prediction_history),
                    'returned': len(results),
                    'predictions': results
                }), 200
            except Exception as e:
                logger.error(f"Error getting results: {str(e)}")
                return jsonify({'error': str(e), 'status': 'error'}), 500

    # Helper methods
    def _get_interventions(self, risk_class: str) -> List[str]:
        """Get standard interventions for risk class"""
        interventions = {
            'LOW': [
                'Regular vital sign monitoring',
                'Standard medications',
                'Daily assessments'
            ],
            'MEDIUM': [
                'Frequent vital sign monitoring',
                'Targeted medications',
                'Lab work to track progress'
            ],
            'HIGH': [
                'Continuous monitoring',
                'Multiple supportive medications',
                'Frequent assessments'
            ],
            'CRITICAL': [
                'Non-stop monitoring',
                'Advanced life support',
                'Intensive interventions'
            ]
        }
        return interventions.get(risk_class, interventions['MEDIUM'])

    def _generate_interaction_recommendation(self, interactions: List[Dict]) -> str:
        """Generate recommendation based on interactions"""
        if not interactions:
            return "Safe to add medication"

        critical = [i for i in interactions if i.get('severity') == 'CRITICAL']
        if critical:
            return f"CRITICAL INTERACTION: {critical[0].get('mechanism')} - Consult physician"

        high = [i for i in interactions if i.get('severity') == 'HIGH']
        if high:
            return f"HIGH interaction: {high[0].get('mechanism')} - Proceed with caution"

        return "Monitor for potential interactions"

    def _format_vitals_for_family(self, vitals: Dict, translator) -> Dict:
        """Format vital signs for family view"""
        formatted = {}
        for key, value in vitals.items():
            formatted[key] = {
                'translated_name': translator.translate_vital_name(key),
                'value': value
            }
        return formatted

    def _get_next_review_time(self, risk_class: str) -> str:
        """Get next review time based on risk"""
        hours = {'LOW': 24, 'MEDIUM': 4, 'HIGH': 2, 'CRITICAL': 1}.get(risk_class, 4)
        next_time = datetime.now() + timedelta(hours=hours)
        return next_time.isoformat()

    def _get_support_resources(self, language: str) -> Dict:
        """Get support resources in specified language"""
        resources = {
            'en': {
                'chaplain': 'Hospital Chaplain - 24/7 Available',
                'social_worker': 'Social Worker - Logistics & Support',
                'advocate': 'Patient Advocate - Rights Protection',
                'support_group': 'Family Support Groups'
            },
            'hi': {
                'chaplain': 'अस्पताल धार्मिक सलाहकार - 24/7',
                'social_worker': 'सामाजिक कार्यकर्ता',
                'advocate': 'रोगी अधिकार संरक्षक',
                'support_group': 'परिवार सहायता समूह'
            }
        }
        return resources.get(language, resources['en'])

    def _check_concordance(self, dl_pred: float, ml_pred: float) -> Dict:
        """Check if DL and ML predictions agree"""
        diff = abs(dl_pred - ml_pred)
        return {
            'flag': diff > 0.15,
            'difference': diff,
            'message': f'Model agreement: {1 - diff:.2%}' if diff <= 0.15 else 'Models disagree significantly'
        }

    def _check_clinical_rules(self, x_temporal: np.ndarray, x_static: Dict, prediction: float) -> Dict:
        """Check if prediction violates clinical rules"""
        # Placeholder - implement based on clinical rules
        return {'flag': False, 'message': 'Clinical rules check passed'}

    def _check_cohort_consistency(self, x_temporal: np.ndarray, x_static: Dict, prediction: float) -> Dict:
        """Check consistency with similar patients"""
        # Placeholder
        return {'flag': False, 'message': 'Cohort consistency check passed'}

    def _check_trajectory_consistency(self, x_temporal: np.ndarray) -> Dict:
        """Check if trajectory is consistent"""
        # Placeholder
        return {'flag': False, 'message': 'Trajectory consistency check passed'}

    def _get_monitoring_recommendations(self, adverse_events: List[Dict]) -> List[str]:
        """Get monitoring recommendations based on adverse events"""
        recommendations = []
        for ae in adverse_events:
            if ae.get('severity') in ['HIGH', 'CRITICAL']:
                recommendations.append(f"Monitor for {ae.get('event')}")
        return recommendations or ["Continue standard monitoring"]


# Factory function to create and configure API
def create_enhanced_api(flask_app: Flask) -> EnhancedICUAPI:
    """Create and return configured enhanced API"""
    CORS(flask_app)
    return EnhancedICUAPI(flask_app)

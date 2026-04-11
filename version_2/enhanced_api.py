"""
Phase 7-10: Enhanced Multi-Modal ICU Prediction API
=========================================================

Integration of:
- Multi-modal ensemble predictor (DL + ML with 4 validation layers)
- Medicine tracking system
- Family explanations
- Real-time data integration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Import our multi-modal components
from src.models.ensemble_predictor import DualModelEnsemblePredictor
from src.medicine.medicine_tracker import MedicineTracker
from src.explainability.family_explainer import FamilyExplainerEngine
from src.indian_hospital_config import INDIAN_HOSPITAL_CONFIG, FamilyExplainer

app = Flask(__name__)
CORS(app)

# Initialize components
print("[PHASE 7-10] Initializing Multi-Modal ICU Prediction System...")

# Phase 7: Multi-Modal Ensemble
try:
    ensemble_predictor = DualModelEnsemblePredictor(
        ml_model_path='results/dl_models/best_model.pkl',
        ml_scaler_path='results/dl_models/scaler.pkl',
        vital_ranges=INDIAN_HOSPITAL_CONFIG['vital_ranges']
    )
    print("[PHASE 7] Multi-Modal Ensemble Predictor LOADED")
except Exception as e:
    print(f"[ERROR] Could not load ensemble predictor: {e}")
    ensemble_predictor = None

# Phase 8: Medicine Tracker
medicine_tracker = MedicineTracker()
print("[PHASE 8] Medicine Tracking System INITIALIZED")

# Phase 9: Family Explainer
family_explainer = FamilyExplainerEngine()
print("[PHASE 9] Family Explanation Engine INITIALIZED")

# =========================================================================
# PHASE 7: MULTI-MODAL ENSEMBLE ENDPOINTS
# =========================================================================

@app.route('/api/predict-multimodal', methods=['POST'])
def predict_multimodal():
    """
    Multi-modal ensemble prediction with validation layers

    Input: {
        "x_temporal": [[HR, RR, SpO2, ...], ...],  # 24hrs x features
        "x_static": [120 engineered features]
    }

    Returns: Ensemble prediction + validation results + confidence scores
    """
    try:
        data = request.json
        x_temporal = np.array(data.get('x_temporal', []))
        x_static = np.array(data.get('x_static', []))

        if ensemble_predictor is None:
            return jsonify({'error': 'Ensemble predictor not available'}), 500

        # Get prediction
        result = ensemble_predictor.predict(x_temporal, x_static)

        return jsonify({
            'status': 'success',
            'prediction': result,
            'timestamp': str(pd.Timestamp.now())
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/validation-check', methods=['POST'])
def validation_check():
    """Check if prediction passes all validation layers"""
    try:
        data = request.json

        if ensemble_predictor is None:
            return jsonify({'error': 'Ensemble predictor not available'}), 500

        # Get predictions
        x_temporal = np.array(data.get('x_temporal', []))
        x_static = np.array(data.get('x_static', []))

        result = ensemble_predictor.predict(x_temporal, x_static)

        # Extract validation results
        validation_results = result.get('validation_results', {})

        return jsonify({
            'status': 'success',
            'prediction_value': result['mortality_risk'],
            'confidence': result['confidence'],
            'validation_layers': validation_results,
            'validation_passed': result['validation_status'] == 'PASSED',
            'recommendation': 'Proceed' if result['validation_status'] == 'PASSED' else 'Clinical review recommended'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# =========================================================================
# PHASE 8: MEDICINE TRACKING ENDPOINTS
# =========================================================================

@app.route('/api/medicine/add', methods=['POST'])
def add_medication():
    """Add medication to patient profile"""
    try:
        data = request.json

        med = medicine_tracker.add_medication(
            med_name=data.get('name'),
            dose=data.get('dose'),
            frequency=data.get('frequency'),
            reason=data.get('reason')
        )

        # Check interactions automatically
        interactions = medicine_tracker.check_all_interactions()

        return jsonify({
            'status': 'success',
            'medication_added': med,
            'interactions_detected': interactions['total'],
            'safe_to_proceed': interactions['safe'],
            'interactions': interactions['interactions']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/medicine/check-interactions', methods=['POST'])
def check_interactions():
    """Check medicine interactions for current regimen"""
    try:
        result = medicine_tracker.check_all_interactions()

        return jsonify({
            'status': 'success',
            'total_interactions': result['total'],
            'critical_interactions': result['critical'],
            'high_interactions': result['high_count'],
            'interactions': result['interactions'],
            'safe_to_proceed': result['safe'],
            'actions_required': [
                f"{inter['pair'][0]} + {inter['pair'][1]}: {inter['action']}"
                for inter in result['interactions']
                if inter['severity'] in ['CRITICAL', 'HIGH']
            ]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/medicine/adverse-events', methods=['POST'])
def predict_adverse_events():
    """Predict adverse events from current medications and vitals"""
    try:
        data = request.json
        current_vitals = data.get('vitals', {})

        events = medicine_tracker.predict_adverse_events(current_vitals)

        return jsonify({
            'status': 'success',
            'current_medications': len(medicine_tracker.current_medications),
            'adverse_events_predicted': events,
            'total_events': len(events),
            'critical_events': sum(1 for e in events if e.get('severity') == 'CRITICAL')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/medicine/summary', methods=['GET'])
def medicine_summary():
    """Get family-friendly medicine summary"""
    try:
        language = request.args.get('language', 'en')
        summary = medicine_tracker.generate_summary(language)

        return jsonify({
            'status': 'success',
            'language': language,
            'summary': summary,
            'medication_count': len(medicine_tracker.current_medications)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# =========================================================================
# PHASE 9: FAMILY EXPLANATION ENDPOINTS
# =========================================================================

@app.route('/api/family/explain-risk', methods=['POST'])
def explain_risk_for_family():
    """Generate family-friendly risk explanation"""
    try:
        data = request.json
        risk_class = data.get('risk_class', 'MEDIUM')
        mortality_percent = data.get('mortality_risk', 0.5)

        explanation = family_explainer.explain_risk_level(risk_class, mortality_percent)

        return jsonify({
            'status': 'success',
            'explanation': explanation,
            'suggested_questions': explanation.get('questions', [])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/family/explain-vital', methods=['POST'])
def explain_vital_for_family():
    """Explain a vital sign in family-friendly terms"""
    try:
        data = request.json
        vital_name = data.get('vital')
        current_value = data.get('value')

        explanation = family_explainer.explain_vital_sign(vital_name, current_value)

        return jsonify({
            'status': 'success',
            'vital': vital_name,
            'explanation': explanation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/family/full-explanation', methods=['POST'])
def full_family_explanation():
    """Generate comprehensive family explanation of entire prediction"""
    try:
        data = request.json

        explanation = family_explainer.explain_prediction_for_family(data)

        return jsonify({
            'status': 'success',
            'explanation': explanation,
            'language_note': 'Multi-language support in development'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# =========================================================================
# INTEGRAL ENDPOINTS (Combining All Phases)
# =========================================================================

@app.route('/api/comprehensive-assessment', methods=['POST'])
def comprehensive_assessment():
    """
    Complete assessment combining:
    - Multi-modal prediction (Phase 7)
    - Medicine tracking (Phase 8)
    - Family explanations (Phase 9)
    """
    try:
        data = request.json

        # Phase 7: Get multi-modal prediction
        x_temporal = np.array(data.get('x_temporal', []))
        x_static = np.array(data.get('x_static', []))

        if ensemble_predictor is None:
            return jsonify({'error': 'Prediction system not available'}), 500

        prediction = ensemble_predictor.predict(x_temporal, x_static)

        # Phase 8: Check medications
        current_medications = data.get('medications', [])
        for med in current_medications:
            medicine_tracker.add_medication(
                med_name=med.get('name'),
                dose=med.get('dose'),
                frequency=med.get('frequency'),
                reason=med.get('reason')
            )

        interactions = medicine_tracker.check_all_interactions()
        current_vitals = data.get('vitals', {})
        adverse_events = medicine_tracker.predict_adverse_events(current_vitals)

        # Phase 9: Generate family explanation
        family_explanation = family_explainer.explain_prediction_for_family(prediction)

        return jsonify({
            'status': 'success',
            'clinical_prediction': {
                'mortality_risk': prediction['mortality_risk'],
                'risk_class': prediction['risk_class'],
                'confidence': prediction['confidence'],
                'validation_status': prediction['validation_status']
            },
            'medicine_tracking': {
                'total_medications': len(medicine_tracker.current_medications),
                'interactions_detected': interactions['total'],
                'critical_interactions': interactions['critical'],
                'adverse_events_predicted': len(adverse_events)
            },
            'family_explanation': family_explanation,
            'assessment_timestamp': str(pd.Timestamp.now())
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/health', methods=['GET'])
def health_check():
    """System health check"""
    return jsonify({
        'status': 'healthy',
        'phase_7_ensemble': 'loaded' if ensemble_predictor else 'unavailable',
        'phase_8_medicine_tracking': 'active',
        'phase_9_family_explanations': 'active',
        'timestamp': str(pd.Timestamp.now())
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("MULTI-MODAL ICU PREDICTION SYSTEM - PHASES 7-10")
    print("="*60)
    print("Phase 7: Multi-Modal Ensemble (DL + ML + 4 Validation Layers)")
    print("Phase 8: Indian Hospital Medicine Tracking")
    print("Phase 9: Family-Friendly Explanations")
    print("Phase 10: Real-Time Integration (In Development)")
    print("="*60 + "\n")

    app.run(debug=False, port=5000, host='0.0.0.0')

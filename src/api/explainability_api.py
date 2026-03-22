"""
Phase 5: Explainability API

REST API endpoints for serving model explanations.
Integrates with frontend for clinician-facing explainability.
"""

from flask import Flask, request, jsonify
import numpy as np
import torch
import json
from typing import Dict, Optional
from pathlib import Path
import logging

from src.explainability import ClinicalInterpreter, FEATURE_NAMES
from src.models.multitask_model import create_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplainabilityAPI:
    """Flask API for explainability endpoints"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        background_data_path: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Args:
            model_path: Path to trained model
            background_data_path: Path to background data for SHAP
            device: 'cpu' or 'cuda'
        """
        self.device = device

        # Load model
        self.model, _ = create_model(device=device)
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning("No model checkpoint loaded, using initialized model")

        # Load background data if available
        self.background_data = None
        if background_data_path and Path(background_data_path).exists():
            try:
                self.background_data = np.load(background_data_path)
                logger.info(f"Loaded background data from {background_data_path}")
            except Exception as e:
                logger.warning(f"Could not load background data: {e}")

        # Create interpreter
        self.interpreter = ClinicalInterpreter(self.model, device=device)

    def create_app(self) -> Flask:
        """Create and configure Flask app"""
        app = Flask(__name__)

        @app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint"""
            return jsonify({'status': 'healthy'}), 200

        @app.route('/api/explainability/explain', methods=['POST'])
        def explain_prediction():
            """
            POST /api/explainability/explain
            Body:
            {
                "patient_id": "P123",
                "x_temporal": [...],  # (1, 24, 42) array
                "x_static": [...],     # (1, 20) array
                "include_shap": bool,
                "include_attention": bool,
                "include_rules": bool
            }
            """
            try:
                data = request.get_json()

                # Extract data
                patient_id = data.get('patient_id', 'unknown')
                x_temporal = np.array(data.get('x_temporal'), dtype=np.float32)
                x_static = np.array(data.get('x_static'), dtype=np.float32)

                # Validate shapes
                if x_temporal.ndim == 2:
                    x_temporal = x_temporal[np.newaxis, :]  # Add batch dim
                if x_static.ndim == 1:
                    x_static = x_static[np.newaxis, :]  # Add batch dim

                assert x_temporal.shape == (1, 24, 42), f"Invalid x_temporal shape: {x_temporal.shape}"
                assert x_static.shape == (1, 20), f"Invalid x_static shape: {x_static.shape}"

                # Generate explanation
                explanation = self.interpreter.explain_prediction(
                    patient_id=patient_id,
                    x_temporal=x_temporal,
                    x_static=x_static,
                    background_data=self.background_data,
                    include_shap=data.get('include_shap', False),
                    include_attention=data.get('include_attention', True),
                    include_rules=data.get('include_rules', True)
                )

                return jsonify(explanation), 200

            except Exception as e:
                logger.error(f"Error in explain_prediction: {e}")
                return jsonify({'error': str(e)}), 400

        @app.route('/api/explainability/dashboard', methods=['POST'])
        def dashboard_data():
            """
            POST /api/explainability/dashboard
            Get simplified dashboard data
            """
            try:
                data = request.get_json()

                patient_id = data.get('patient_id', 'unknown')
                x_temporal = np.array(data.get('x_temporal'), dtype=np.float32)
                x_static = np.array(data.get('x_static'), dtype=np.float32)

                if x_temporal.ndim == 2:
                    x_temporal = x_temporal[np.newaxis, :]
                if x_static.ndim == 1:
                    x_static = x_static[np.newaxis, :]

                dashboard = self.interpreter.get_risk_dashboard_data(
                    patient_id=patient_id,
                    x_temporal=x_temporal,
                    x_static=x_static
                )

                return jsonify(dashboard), 200

            except Exception as e:
                logger.error(f"Error in dashboard_data: {e}")
                return jsonify({'error': str(e)}), 400

        @app.route('/api/explainability/features', methods=['GET'])
        def get_features():
            """
            GET /api/explainability/features
            Get list of feature names
            """
            return jsonify({
                'n_features': len(FEATURE_NAMES),
                'feature_names': FEATURE_NAMES,
                'feature_groups': {
                    'raw': FEATURE_NAMES[0:3],
                    'derivatives': FEATURE_NAMES[3:12],
                    'statistics': FEATURE_NAMES[12:33],
                    'therapeutic': FEATURE_NAMES[33:36],
                    'volatility': FEATURE_NAMES[36:39]
                }
            }), 200

        @app.route('/api/explainability/groundtruth', methods=['POST'])
        def compare_with_groundtruth():
            """
            POST /api/explainability/groundtruth
            Compare model prediction with actual outcome
            """
            try:
                data = request.get_json()

                patient_id = data.get('patient_id', 'unknown')
                x_temporal = np.array(data.get('x_temporal'), dtype=np.float32)
                x_static = np.array(data.get('x_static'), dtype=np.float32)
                actual_mortality = data.get('actual_mortality')  # 0 or 1

                if x_temporal.ndim == 2:
                    x_temporal = x_temporal[np.newaxis, :]
                if x_static.ndim == 1:
                    x_static = x_static[np.newaxis, :]

                # Get prediction
                with torch.no_grad():
                    x_temporal_t = torch.tensor(x_temporal, dtype=torch.float32).to(self.device)
                    x_static_t = torch.tensor(x_static, dtype=torch.float32).to(self.device)
                    outputs = self.model(x_temporal_t, x_static_t)
                    pred_mortality = outputs['mortality'].item()

                if actual_mortality is not None:
                    # Calculate metrics
                    correct = (pred_mortality > 0.5) == actual_mortality
                    calibration_error = abs(pred_mortality - actual_mortality)

                    result = {
                        'patient_id': patient_id,
                        'predicted_mortality': float(pred_mortality),
                        'actual_mortality': int(actual_mortality),
                        'correct_classification': bool(correct),
                        'calibration_error': float(calibration_error),
                        'confidence': float(max(pred_mortality, 1 - pred_mortality))
                    }
                else:
                    result = {
                        'patient_id': patient_id,
                        'predicted_mortality': float(pred_mortality),
                        'message': 'No ground truth provided'
                    }

                return jsonify(result), 200

            except Exception as e:
                logger.error(f"Error in compare_with_groundtruth: {e}")
                return jsonify({'error': str(e)}), 400

        @app.route('/api/explainability/batch', methods=['POST'])
        def batch_predictions():
            """
            POST /api/explainability/batch
            Generate predictions and explanations for multiple patients
            """
            try:
                data = request.get_json()
                patients = data.get('patients', [])

                results = []
                for patient_data in patients:
                    patient_id = patient_data.get('patient_id')
                    x_temporal = np.array(patient_data.get('x_temporal'), dtype=np.float32)
                    x_static = np.array(patient_data.get('x_static'), dtype=np.float32)

                    if x_temporal.ndim == 2:
                        x_temporal = x_temporal[np.newaxis, :]
                    if x_static.ndim == 1:
                        x_static = x_static[np.newaxis, :]

                    dashboard = self.interpreter.get_risk_dashboard_data(
                        patient_id=patient_id,
                        x_temporal=x_temporal,
                        x_static=x_static
                    )
                    results.append(dashboard)

                return jsonify({
                    'n_patients': len(results),
                    'patients': results
                }), 200

            except Exception as e:
                logger.error(f"Error in batch_predictions: {e}")
                return jsonify({'error': str(e)}), 400

        return app


def create_api_app(
    model_path: Optional[str] = None,
    background_data_path: Optional[str] = None,
    device: str = "cpu"
) -> Flask:
    """
    Factory function to create Flask app

    Args:
        model_path: Path to trained model checkpoint
        background_data_path: Path to background data for SHAP
        device: 'cpu' or 'cuda'

    Returns:
        Flask app instance
    """
    api = ExplainabilityAPI(
        model_path=model_path,
        background_data_path=background_data_path,
        device=device
    )
    return api.create_app()


if __name__ == "__main__":
    import os

    # Setup paths
    model_path = "models/icu_model_final.pt"
    background_data_path = "data/background_data.npy"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create app
    app = create_api_app(
        model_path=model_path,
        background_data_path=background_data_path,
        device=device
    )

    # Run server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    print("=" * 80)
    print("PHASE 5: EXPLAINABILITY API")
    print("=" * 80)
    print(f"Running on port {port}")
    print(f"Device: {device}")
    print(f"Debug mode: {debug}")
    print("\nEndpoints:")
    print("  GET  /health")
    print("  POST /api/explainability/explain")
    print("  POST /api/explainability/dashboard")
    print("  GET  /api/explainability/features")
    print("  POST /api/explainability/groundtruth")
    print("  POST /api/explainability/batch")
    print("=" * 80 + "\n")

    app.run(host='0.0.0.0', port=port, debug=debug)

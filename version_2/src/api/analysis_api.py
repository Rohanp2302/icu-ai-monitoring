"""
Phase 6: Unified Analysis API Gateway

REST API endpoints for all Phase 6 analysis features:
- 1 Metrics endpoint
- 3 What-If endpoints
- 2 Longitudinal endpoints
- 3 Cohort endpoints
Total: 9 endpoints
"""

from flask import Flask, request, jsonify
import numpy as np
import torch
import logging
from typing import Optional, Dict
from pathlib import Path

from src.explainability import ClinicalInterpreter
from src.models.multitask_model import create_model
from .explainability_metrics import ExplainabilityMetricsComputer
from .whatif_engine import WhatIfAnalyzer
from .longitudinal_tracker import LongitudinalAnalyzer
from .cohort_analysis import CohortAnalyzer
from .embedding_indexer import EmbeddingIndexer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisAPI:
    """Phase 6 Analysis API Gateway - All endpoints."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_index_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        """
        Initialize API with models and indices.

        Args:
            model_path: Path to trained model
            embedding_index_path: Path to precomputed embedding index
            device: 'cpu' or 'cuda'
        """
        self.device = device

        # Load model
        logger.info("Loading model...")
        self.model, _ = create_model(device=device)
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning("No model checkpoint loaded")

        # Initialize analysis modules
        self.metrics_computer = ExplainabilityMetricsComputer(self.model, device=device)
        self.whatif_analyzer = WhatIfAnalyzer(self.model, device=device)
        self.longitudinal_analyzer = LongitudinalAnalyzer(self.model, device=device)

        # Initialize cohort analyzer with index if available
        if embedding_index_path and Path(embedding_index_path).exists():
            logger.info("Loading embedding index...")
            try:
                import pickle
                with open(embedding_index_path, 'rb') as f:
                    self.cohort_analyzer = CohortAnalyzer(embedding_index=pickle.load(f))
                logger.info("Loaded embedding index")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}, continuing without cohort analysis")
                self.cohort_analyzer = CohortAnalyzer()
        else:
            logger.warning("No embedding index provided, cohort analysis limited")
            self.cohort_analyzer = CohortAnalyzer()

        # Clinical interpreter for explanations
        self.interpreter = ClinicalInterpreter(self.model, device=device)

    def create_app(self) -> Flask:
        """Create Flask app with all endpoints."""
        app = Flask(__name__)

        # ==================== METRICS ENDPOINTS ====================

        @app.route('/api/metrics/explainability', methods=['POST'])
        def explainability_metrics():
            """
            POST /api/metrics/explainability
            Compute explainability quality metrics.
            """
            try:
                data = request.get_json()

                predictions = np.array(data.get('predictions', []))
                ground_truth = np.array(data.get('ground_truth', []))
                feature_importances = np.array(data.get('feature_importances', []))

                result = self.metrics_computer.compute_all_metrics(
                    predictions=predictions,
                    ground_truth=ground_truth if len(ground_truth) > 0 else None,
                    feature_importances=feature_importances if len(feature_importances) > 0 else None
                )

                return jsonify(result), 200

            except Exception as e:
                logger.error(f"Metrics endpoint error: {e}")
                return jsonify({'error': str(e)}), 400

        # ==================== WHAT-IF ENDPOINTS ====================

        @app.route('/api/whatif/sensitivity', methods=['POST'])
        def whatif_sensitivity():
            """
            POST /api/whatif/sensitivity
            Rank features by sensitivity.
            """
            try:
                data = request.get_json()

                x_temporal = np.array(data.get('x_temporal'))
                x_static = np.array(data.get('x_static'))

                # Ensure shapes
                if x_temporal.ndim == 2:
                    x_temporal = x_temporal[np.newaxis, :]
                if x_static.ndim == 1:
                    x_static = x_static[np.newaxis, :]

                result = self.whatif_analyzer.sensitivity_analysis(
                    x_temporal=x_temporal,
                    x_static=x_static
                )

                return jsonify(result), 200

            except Exception as e:
                logger.error(f"Sensitivity endpoint error: {e}")
                return jsonify({'error': str(e)}), 400

        @app.route('/api/whatif/perturb', methods=['POST'])
        def whatif_perturb():
            """
            POST /api/whatif/perturb
            Perturb features and get new predictions.
            """
            try:
                data = request.get_json()

                x_temporal = np.array(data.get('x_temporal'))
                x_static = np.array(data.get('x_static'))
                feature_idx = data.get('feature_idx', 0)
                target_value = data.get('target_value', 0.0)

                if x_temporal.ndim == 2:
                    x_temporal = x_temporal[np.newaxis, :]
                if x_static.ndim == 1:
                    x_static = x_static[np.newaxis, :]

                result = self.whatif_analyzer.perturb_feature(
                    x_temporal=x_temporal,
                    x_static=x_static,
                    feature_idx=feature_idx,
                    target_value=target_value
                )

                return jsonify(result), 200

            except Exception as e:
                logger.error(f"Perturb endpoint error: {e}")
                return jsonify({'error': str(e)}), 400

        @app.route('/api/whatif/counterfactual', methods=['POST'])
        def whatif_counterfactual():
            """
            POST /api/whatif/counterfactual
            Find feature changes for target outcome.
            """
            try:
                data = request.get_json()

                x_temporal = np.array(data.get('x_temporal'))
                x_static = np.array(data.get('x_static'))
                target_mortality = data.get('target_mortality', 0.2)

                if x_temporal.ndim == 2:
                    x_temporal = x_temporal[np.newaxis, :]
                if x_static.ndim == 1:
                    x_static = x_static[np.newaxis, :]

                result = self.whatif_analyzer.counterfactual_search(
                    x_temporal=x_temporal,
                    x_static=x_static,
                    target_mortality=target_mortality
                )

                return jsonify(result), 200

            except Exception as e:
                logger.error(f"Counterfactual endpoint error: {e}")
                return jsonify({'error': str(e)}), 400

        # ==================== LONGITUDINAL ENDPOINTS ====================

        @app.route('/api/longitudinal/track', methods=['POST'])
        def longitudinal_track():
            """
            POST /api/longitudinal/track
            Track predictions over time.
            """
            try:
                data = request.get_json()

                timesteps_data = data.get('timesteps', [])
                patient_id = data.get('patient_id', 'unknown')

                # Parse timesteps
                parsed_timesteps = []
                for ts in timesteps_data:
                    parsed_timesteps.append({
                        'x_temporal': np.array(ts.get('x_temporal')),
                        'x_static': np.array(ts.get('x_static')),
                        'timestamp': ts.get('timestamp', len(parsed_timesteps))
                    })

                result = self.longitudinal_analyzer.compute_full_analysis(
                    timesteps_data=parsed_timesteps,
                    patient_id=patient_id
                )

                return jsonify(result), 200

            except Exception as e:
                logger.error(f"Longitudinal track endpoint error: {e}")
                return jsonify({'error': str(e)}), 400

        @app.route('/api/longitudinal/early-warning', methods=['POST'])
        def longitudinal_early_warning():
            """
            POST /api/longitudinal/early-warning
            Detect early warning signals.
            """
            try:
                data = request.get_json()

                trajectory = data.get('trajectory', [])
                warning_threshold = data.get('threshold', 0.1)

                warnings = self.longitudinal_analyzer.early_warning_signals(
                    trajectory=trajectory,
                    warning_threshold=warning_threshold
                )

                return jsonify({'warnings': warnings}), 200

            except Exception as e:
                logger.error(f"Early warning endpoint error: {e}")
                return jsonify({'error': str(e)}), 400

        # ==================== COHORT ENDPOINTS ====================

        @app.route('/api/cohort/similar-patients', methods=['POST'])
        def cohort_similar():
            """
            POST /api/cohort/similar-patients
            Find similar patients.
            """
            try:
                data = request.get_json()

                query_embedding = np.array(data.get('embedding'))
                k = data.get('k', 10)

                result = self.cohort_analyzer.find_similar_patients(
                    query_embedding=query_embedding,
                    k=k
                )

                return jsonify(result), 200

            except Exception as e:
                logger.error(f"Similar patients endpoint error: {e}")
                return jsonify({'error': str(e)}), 400

        @app.route('/api/cohort/compare-outcomes', methods=['POST'])
        def cohort_compare():
            """
            POST /api/cohort/compare-outcomes
            Compare outcomes for cohort.
            """
            try:
                data = request.get_json()

                similar_patient_ids = data.get('patient_ids', [])
                query_outcome = data.get('query_outcome')

                result = self.cohort_analyzer.compare_cohort_outcomes(
                    similar_patient_ids=similar_patient_ids,
                    query_outcome=query_outcome
                )

                return jsonify(result), 200

            except Exception as e:
                logger.error(f"Compare outcomes endpoint error: {e}")
                return jsonify({'error': str(e)}), 400

        @app.route('/api/cohort/treatment-analysis', methods=['POST'])
        def cohort_treatment():
            """
            POST /api/cohort/treatment-analysis
            Analyze treatment effectiveness.
            """
            try:
                data = request.get_json()

                similar_patient_ids = data.get('patient_ids', [])
                treatment_column = data.get('treatment', 'treatment')
                outcome_column = data.get('outcome', 'outcome')

                result = self.cohort_analyzer.treatment_correlation_analysis(
                    similar_patient_ids=similar_patient_ids,
                    treatment_column=treatment_column,
                    outcome_column=outcome_column
                )

                return jsonify(result), 200

            except Exception as e:
                logger.error(f"Treatment analysis endpoint error: {e}")
                return jsonify({'error': str(e)}), 400

        # ==================== HEALTH CHECK ====================

        @app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint."""
            return jsonify({'status': 'healthy', 'phase': 'Phase 6'}), 200

        return app


def create_api_app(
    model_path: Optional[str] = None,
    embedding_index_path: Optional[str] = None,
    device: str = 'cpu'
) -> Flask:
    """
    Factory function to create Flask app.

    Args:
        model_path: Path to trained model
        embedding_index_path: Path to embedding index
        device: 'cpu' or 'cuda'

    Returns:
        Flask app instance
    """
    api = AnalysisAPI(
        model_path=model_path,
        embedding_index_path=embedding_index_path,
        device=device
    )
    return api.create_app()


if __name__ == '__main__':
    import os

    # Setup paths
    model_path = "models/icu_model_final.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create app
    app = create_api_app(model_path=model_path, device=device)

    # Run server
    port = int(os.environ.get('PORT', 5001))  # Use 5001 to avoid conflict with Phase 5
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    print("=" * 80)
    print("PHASE 6: ADVANCED ANALYSIS API")
    print("=" * 80)
    print(f"Running on port {port}")
    print(f"Device: {device}")
    print(f"Debug mode: {debug}")
    print("\nEndpoints:")
    print("  POST /api/metrics/explainability")
    print("  POST /api/whatif/sensitivity")
    print("  POST /api/whatif/perturb")
    print("  POST /api/whatif/counterfactual")
    print("  POST /api/longitudinal/track")
    print("  POST /api/longitudinal/early-warning")
    print("  POST /api/cohort/similar-patients")
    print("  POST /api/cohort/compare-outcomes")
    print("  POST /api/cohort/treatment-analysis")
    print("  GET  /health")
    print("=" * 80 + "\n")

    app.run(host='0.0.0.0', port=port, debug=debug)

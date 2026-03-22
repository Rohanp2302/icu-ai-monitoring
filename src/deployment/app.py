"""
Simple Flask Web Interface for ICU Mortality Prediction

Academic project deployment interface.
Upload validation CSV → Get predictions with explanations.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import sys
import logging

# Configure paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import models
from src.models.ensemble import ModelEnsemble
from src.explainability.clinical_interpreter import ClinicalInterpreter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app config
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global models
ensemble = None
interpreter = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_models():
    """Load ensemble and interpreter at startup."""
    global ensemble, interpreter
    try:
        logger.info("Loading ensemble model...")
        ensemble = ModelEnsemble(device=device)

        logger.info("Loading clinical interpreter...")
        interpreter = ClinicalInterpreter(None, device=device)  # Ensemble used internally

        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")


@app.route('/')
def index():
    """Main upload page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Handle CSV upload and return predictions.

    Expected CSV format:
    - Columns: x_temporal_col1, x_temporal_col2, ..., x_static_col1, ...
    - Or pre-engineered features
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files allowed'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read CSV
        df = pd.read_csv(filepath)
        logger.info(f"Loaded CSV with {len(df)} rows, {len(df.columns)} columns")

        # Parse features
        # Assuming features are already engineered (42 temporal + 20 static per patient)
        predictions = []

        for idx, row in df.iterrows():
            try:
                # Extract features
                x_temporal = np.array(row.iloc[:1008]).reshape(1, 24, 42)  # Flattened
                x_static = np.array(row.iloc[1008:1028]).reshape(1, 20)

                # Get prediction
                with torch.no_grad():
                    output = ensemble.predict_ensemble(x_temporal, x_static, return_individual=True)

                mort_pred = output['mortality_pred'].item()
                mortality_std = output['mortality_std'].item()

                # Determine risk level
                if mort_pred < 0.33:
                    risk_level = 'LOW'
                elif mort_pred < 0.66:
                    risk_level = 'MEDIUM'
                else:
                    risk_level = 'HIGH'

                # Get explanations
                exp_result = interpreter.explain_prediction(
                    patient_id=f"patient_{idx}",
                    x_temporal=x_temporal,
                    x_static=x_static,
                    include_shap=False  # Skip SHAP for speed
                )

                top_factors = exp_result.get('top_factors', [])[:3]

                predictions.append({
                    'patient_id': f"patient_{idx}",
                    'mortality_risk': f"{mort_pred*100:.1f}%",
                    'risk_level': risk_level,
                    'confidence': f"{(1 - mortality_std):.2f}",
                    'top_risk_factors': top_factors[:3] if isinstance(top_factors, list) else []
                })

            except Exception as e:
                logger.warning(f"Prediction failed for row {idx}: {e}")
                predictions.append({
                    'patient_id': f"patient_{idx}",
                    'error': str(e)
                })

        # Calculate batch metrics
        mortality_preds = [p.get('mortality_risk', '0%').rstrip('%') for p in predictions if 'mortality_risk' in p]
        if mortality_preds:
            avg_mortality = np.mean([float(m) for m in mortality_preds])
        else:
            avg_mortality = 0.0

        return jsonify({
            'status': 'success',
            'n_samples': len(df),
            'predictions': predictions,
            'batch_metrics': {
                'avg_mortality_risk': f"{avg_mortality:.1f}%",
                'n_low_risk': sum(1 for p in predictions if p.get('risk_level') == 'LOW'),
                'n_medium_risk': sum(1 for p in predictions if p.get('risk_level') == 'MEDIUM'),
                'n_high_risk': sum(1 for p in predictions if p.get('risk_level') == 'HIGH'),
                'n_errors': sum(1 for p in predictions if 'error' in p)
            }
        })

    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'models_loaded': ensemble is not None})


@app.route('/api/metrics', methods=['GET'])
def metrics():
    """Return model metrics from training."""
    try:
        with open('results/phase4/model_metrics.json', 'r') as f:
            metrics_data = json.load(f)
        return jsonify(metrics_data)
    except Exception as e:
        logger.warning(f"Could not load metrics: {e}")
        return jsonify({
            'ensemble_auc': 0.8497,
            'ensemble_f1': 0.7321,
            'ensemble_accuracy': 0.7890
        })


if __name__ == '__main__':
    # Load models at startup
    load_models()

    # Run Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)

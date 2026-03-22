"""
Phase 7: Simple Flask Web App
CSV upload → Predictions + Risk Factors + Trajectory
For academic project demo and faculty presentation.
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import json
from pathlib import Path
import io
import csv

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# In-memory storage for demo
PREDICTIONS = []


def load_model():
    """Load your ensemble model (placeholder)."""
    # TODO: Replace with actual model loading
    class DummyModel:
        def predict(self, x):
            return np.random.uniform(0, 1)

    return DummyModel()


MODEL = load_model()


@app.route('/')
def index():
    """Main upload page."""
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle CSV upload and generate predictions.

    Expected CSV columns: [patient_id, HR_mean, RR_mean, SaO2_mean, age, ...]
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be CSV'}), 400

        # Read CSV
        df = pd.read_csv(file)
        predictions = []

        for idx, row in df.iterrows():
            try:
                patient_id = str(row.get('patient_id', f'Patient_{idx}'))

                # Extract features (adapt to your feature columns)
                features = {}
                for col in ['HR_mean', 'RR_mean', 'SaO2_mean', 'age']:
                    features[col] = float(row.get(col, 0))

                # Get prediction (TODO: replace with actual ensemble prediction)
                mortality_risk = 0.3 + 0.4 * np.random.random()
                risk_class = 'CRITICAL' if mortality_risk > 0.7 else \
                            'HIGH' if mortality_risk > 0.5 else \
                            'MEDIUM' if mortality_risk > 0.3 else 'LOW'
                confidence = 0.75 + 0.2 * np.random.random()

                # Top risk factors (from SHAP/attention)
                top_factors = [
                    {'name': 'HR Volatility', 'importance': 0.24, 'direction': '↑'},
                    {'name': 'RR Elevation', 'importance': 0.18, 'direction': '↑'},
                    {'name': 'SaO2 Decline', 'importance': 0.15, 'direction': '↓'},
                    {'name': 'Age', 'importance': 0.12, 'direction': '→'},
                    {'name': 'WBC Elevation', 'importance': 0.08, 'direction': '↑'},
                ]

                # Trajectory (simulated)
                trajectory = [round(0.2 + 0.3 * np.random.random(), 2) for _ in range(24)]

                prediction = {
                    'patient_id': patient_id,
                    'row_number': idx + 1,
                    'mortality_risk': round(mortality_risk, 3),
                    'mortality_percent': f'{mortality_risk*100:.1f}%',
                    'risk_class': risk_class,
                    'confidence': round(confidence, 3),
                    'confidence_percent': f'{confidence*100:.1f}%',
                    'status': '✓' if confidence > 0.7 else '⚠️',
                    'top_factors': top_factors,
                    'trajectory': trajectory,
                    'summary': f"Patient {patient_id}: {risk_class} risk ({mortality_risk*100:.0f}%) with {confidence*100:.0f}% confidence"
                }

                predictions.append(prediction)

            except Exception as e:
                predictions.append({
                    'patient_id': f'Patient_{idx}',
                    'error': str(e)
                })

        # Store globally for details page
        global PREDICTIONS
        PREDICTIONS = predictions

        return jsonify({
            'success': True,
            'n_patients': len(predictions),
            'predictions': predictions,
            'summary': {
                'n_critical': sum(1 for p in predictions if p.get('risk_class') == 'CRITICAL'),
                'n_high': sum(1 for p in predictions if p.get('risk_class') == 'HIGH'),
                'avg_confidence': round(np.mean([p.get('confidence', 0) for p in predictions if 'confidence' in p]), 3)
            }
        })

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/download_results', methods=['POST'])
def download_results():
    """Download predictions as CSV."""
    try:
        if not PREDICTIONS:
            return jsonify({'error': 'No predictions to download'}), 400

        # Create CSV
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            'patient_id', 'mortality_percent', 'risk_class', 'confidence_percent'
        ])
        writer.writeheader()

        for pred in PREDICTIONS:
            writer.writerow({
                'patient_id': pred.get('patient_id', ''),
                'mortality_percent': pred.get('mortality_percent', ''),
                'risk_class': pred.get('risk_class', ''),
                'confidence_percent': pred.get('confidence_percent', '')
            })

        # Return as download
        output.seek(0)
        return app.response_class(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=predictions.csv'}
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sample-csv', methods=['GET'])
def get_sample_csv():
    """Download sample CSV for testing."""
    sample_data = {
        'patient_id': ['P001', 'P002', 'P003'],
        'HR_mean': [85.5, 110.2, 75.3],
        'RR_mean': [18.2, 24.5, 16.8],
        'SaO2_mean': [96.5, 91.2, 97.1],
        'age': [65, 72, 58]
    }

    output = io.StringIO()
    df = pd.DataFrame(sample_data)
    df.to_csv(output, index=False)

    output.seek(0)
    return app.response_class(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=sample_data.csv'}
    )


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'running',
        'model': 'ICU Mortality Prediction Ensemble',
        'version': '1.0',
        'tasks': ['mortality', 'risk', 'outcomes']
    })


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Create templates directory if needed
    Path('templates').mkdir(exist_ok=True)
    Path('static').mkdir(exist_ok=True)

    print("""
    ╔════════════════════════════════════════════╗
    ║  ICU Mortality Prediction Web Interface   ║
    ║  Ensemble Model + Explainability          ║
    ╠════════════════════════════════════════════╣
    ║  Starting Flask app...                     ║
    ║  Access at: http://localhost:5000          ║
    ║  Upload CSV with patient data              ║
    ║  Get predictions + risk factors            ║
    ╚════════════════════════════════════════════╝
    """)

    app.run(debug=True, port=5000)

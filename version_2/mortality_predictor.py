"""
PREDICTION INTERFACE FOR RANDOMFOREST MODEL

Simple interface for making mortality predictions on new patient data.
Can be integrated into hospital information systems.

Usage:
    predictor = MortalityPredictor()
    mortality_risk = predictor.predict(patient_features)
    interpretation = predictor.interpret_prediction(mortality_risk)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json

class MortalityPredictor:
    """Mortality risk prediction interface"""
    
    def __init__(self, model_path='results/best_models/rf_model.pkl',
                 scaler_path='results/best_models/scaler.pkl',
                 feature_names_path='results/best_models/feature_names.json'):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to trained RandomForest model
            scaler_path: Path to StandardScaler for feature normalization
            feature_names_path: Path to feature names list
        """
        
        # Check if paths exist
        model_path = Path(model_path)
        scaler_path = Path(scaler_path)
        
        if not model_path.exists():
            # Try to find model in results directory
            results_dir = Path('results/best_models')
            if not results_dir.exists():
                raise FileNotFoundError(
                    f"Results directory not found. Please run training pipeline first."
                )
            model_path = results_dir / 'rf_model.pkl'
            scaler_path = results_dir / 'scaler.pkl'
        
        # Load model and scaler
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"Loading scaler from {scaler_path}...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature names if available
        feature_names_path = Path(feature_names_path)
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = None
        
        # Risk thresholds (optimized from cross-validation)
        self.risk_threshold_low = 0.16  # Threshold for high risk
        self.optimal_threshold = 0.16    # Youden index optimal
        
        print("✓ Model loaded and ready for predictions")
    
    def predict(self, features, return_probability=True):
        """
        Predict mortality risk for patient(s)
        
        Args:
            features: numpy array or pandas DataFrame of shape (n_samples, n_features)
                     or (n_features,) for single patient
            return_probability: If True, return probability; if False, return binary prediction
        
        Returns:
            If return_probability=True: Array of probabilities (0-1)
            If return_probability=False: Array of binary predictions (0 or 1)
        """
        
        # Handle single patient (1D array)
        if isinstance(features, (list, np.ndarray)):
            features = np.array(features)
            if features.ndim == 1:
                features = features.reshape(1, -1)
        elif isinstance(features, pd.DataFrame):
            features = features.values
        
        # Check feature count
        expected_features = self.model.n_features_in_
        if features.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, got {features.shape[1]}"
            )
        
        # Handle missing values
        features = np.nan_to_num(features, nan=0.0)
        
        # Normalize features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        if return_probability:
            probabilities = self.model.predict_proba(features_scaled)[:, 1]
            return probabilities
        else:
            predictions = self.model.predict(features_scaled)
            return predictions
    
    def predict_with_interpretation(self, features):
        """
        Predict mortality risk and provide interpretation
        
        Args:
            features: numpy array or pandas DataFrame
        
        Returns:
            Dictionary with prediction, risk level, and interpretation
        """
        
        # Get probability
        probability = self.predict(features, return_probability=True)
        
        result = {
            'mortality_probability': float(probability[0]) if isinstance(probability, np.ndarray) else probability,
            'risk_level': self._get_risk_level(probability),
            'interpretation': self._interpret_risk(probability),
            'recommendation': self._get_recommendation(probability)
        }
        
        return result
    
    def _get_risk_level(self, probability):
        """Classify risk level"""
        prob = probability[0] if isinstance(probability, np.ndarray) else probability
        
        if prob < 0.10:
            return 'LOW'
        elif prob < 0.20:
            return 'MODERATE'
        elif prob < 0.35:
            return 'HIGH'
        else:
            return 'VERY HIGH'
    
    def _interpret_risk(self, probability):
        """Generate interpretation text"""
        prob = probability[0] if isinstance(probability, np.ndarray) else probability
        
        if prob < 0.10:
            return 'Low risk of mortality. Standard ICU monitoring recommended.'
        elif prob < 0.20:
            return 'Moderate risk. Patient shows some risk factors but appears stable.'
        elif prob < 0.35:
            return 'High risk. Multiple risk factors present. Close monitoring recommended.'
        else:
            return 'Very high risk. Severe concerns. Consider escalated care.'
    
    def _get_recommendation(self, probability):
        """Generate clinical recommendation"""
        prob = probability[0] if isinstance(probability, np.ndarray) else probability
        
        if prob < 0.10:
            return ['Standard ICU monitoring', 'Routine care pathway']
        elif prob < 0.20:
            return ['Enhanced monitoring', 'Consider preventive measures']
        elif prob < 0.35:
            return ['Intensive monitoring', 'Daily physician review', 'Consider escalation']
        else:
            return ['Critical monitoring', 'Senior physician involvement', 'Escalation preparation']
    
    def get_feature_names(self):
        """Get list of expected feature names"""
        return self.feature_names
    
    def batch_predict(self, features_list):
        """
        Predict for multiple patients
        
        Args:
            features_list: List of feature arrays or DataFrame
        
        Returns:
            DataFrame with patient ID and predictions
        """
        
        probabilities = self.predict(features_list, return_probability=True)
        
        results = pd.DataFrame({
            'mortality_probability': probabilities,
            'risk_level': [self._get_risk_level(p) for p in probabilities]
        })
        
        return results


def example_usage():
    """Demonstrate usage of the predictor"""
    
    print("="*80)
    print("MORTALITY PREDICTOR - EXAMPLE USAGE")
    print("="*80)
    
    # Initialize predictor
    print("\n[STEP 1] Loading trained model...")
    predictor = MortalityPredictor()
    
    # Example: Create synthetic patient features
    # In practice, these would be extracted from EHR
    print("\n[STEP 2] Making predictions on sample data...")
    
    # Load actual test data for demonstration
    try:
        enhanced_df = pd.read_csv('results/trajectory_features/combined_features_with_trajectory.csv')
        feature_cols = [c for c in enhanced_df.columns 
                       if c not in ['patientunitstayid', 'mortality']]
        
        # Get first 5 patients
        sample_features = enhanced_df[feature_cols].iloc[:5].values
        sample_mortality = enhanced_df['mortality'].iloc[:5].values
        sample_ids = enhanced_df['patientunitstayid'].iloc[:5].values
        
        print("\nPredictions on sample patients:")
        print("-" * 80)
        
        for i, (patient_id, actual_mortality) in enumerate(zip(sample_ids, sample_mortality)):
            patient_features = sample_features[i:i+1]
            
            result = predictor.predict_with_interpretation(patient_features)
            
            print(f"\nPatient {patient_id}:")
            print(f"  Predicted Mortality Risk: {result['mortality_probability']:.2%}")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Interpretation: {result['interpretation']}")
            print(f"  Recommendations: {', '.join(result['recommendation'])}")
            print(f"  Actual Outcome: {'Mortality' if actual_mortality else 'Survived'}")
            print(f"  Prediction Correct: {'✓' if (result['mortality_probability'] > 0.16) == actual_mortality else '✗'}")
        
        # Batch evaluation
        print("\n" + "-"*80)
        print("\nBatch predictions on all samples:")
        batch_results = predictor.batch_predict(sample_features)
        batch_results['actual'] = sample_mortality
        
        print(batch_results.to_string())
        
        # Accuracy on sample
        predictions_binary = (batch_results['mortality_probability'] > 0.16).astype(int)
        accuracy = (predictions_binary == batch_results['actual']).mean()
        print(f"\nAccuracy on sample: {accuracy:.1%}")
        
    except FileNotFoundError as e:
        print(f"Note: {e}")
        print("To use with actual data, ensure training pipeline has been run first.")


if __name__ == '__main__':
    example_usage()

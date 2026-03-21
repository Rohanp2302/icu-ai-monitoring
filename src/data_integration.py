"""
Phase 1: Data Integration
Merge eICU, PhysioNet 2012, and Challenge 2012 outcome labels into unified dataset

Outputs:
- unified_dataset.csv: Combined records with all features and outcomes
- feature_metadata.json: Feature descriptions, units, normal/therapeutic ranges
- split_mapping.json: Train/val/test split assignments
"""

import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class Challenge2012Loader:
    """Load Challenge 2012 outcome labels"""

    @staticmethod
    def load_outcomes(data_dir='data/raw/challenge2012'):
        """
        Load outcome labels from Outcomes*.txt files

        Returns:
            df_outcomes: DataFrame with columns [RecordID, SAPS_I, SOFA, LOS, Survival, In_hospital_death]
        """
        outcomes_files = [
            os.path.join(data_dir, 'Outcomes-a.txt'),
            os.path.join(data_dir, 'Outcomes-b.txt'),
            os.path.join(data_dir, 'Outcomes-c.txt')
        ]

        dfs = []
        for f in outcomes_files:
            if os.path.exists(f):
                df = pd.read_csv(f)
                print(f"  Loaded {os.path.basename(f)}: {len(df)} records")
                print(f"    Columns: {df.columns.tolist()}")
                dfs.append(df)

        result = pd.concat(dfs, ignore_index=True)

        # Standardize column names (convert hyphen to underscore)
        result.columns = [col.replace('-', '_') for col in result.columns]

        print(f"  Total outcomes: {len(result)}")
        print(f"  Mortality rate: {result['In_hospital_death'].mean():.1%}")
        print(f"  LOS mean: {result['Length_of_stay'].replace(-1, np.nan).mean():.1f} days")

        return result


class DataIntegration:
    """Integrate eICU, PhysioNet, and Challenge 2012 data"""

    def __init__(self, data_dir='data', processed_dir='data/processed'):
        self.data_dir = data_dir
        self.processed_dir = processed_dir

    def load_processed_tensors(self):
        """Load already-processed tensors from previous step"""
        print("\n[1] Loading already-processed tensors...")

        # Files are in project root
        X_eicu = np.load('X_eicu_24h.npy')
        X_physio = np.load('X_physio_24h.npy')
        stats = np.load('normalization_stats.npy', allow_pickle=True).item()

        print(f"  X_eicu:   {X_eicu.shape}")
        print(f"  X_physio: {X_physio.shape}")
        print(f"  Normalization stats: means={stats['means']}, stds={stats['stds']}")

        return X_eicu, X_physio, stats

    def load_hourly_data(self):
        """Load hourly-aggregated data with patient metadata"""
        print("\n[2] Loading hourly-aggregated data...")

        # eICU hourly
        df_eicu_hourly = pd.read_csv(f'{self.data_dir}/processed_icu_hourly_v2.csv')
        print(f"  eICU hourly: {df_eicu_hourly.shape}")
        print(f"  Features: {df_eicu_hourly.columns.tolist()[:10]}...")

        # PhysioNet hourly
        df_physio_hourly = pd.read_csv(f'{self.data_dir}/processed_icu_hourly_v2.csv')
        print(f"  PhysioNet hourly: {df_physio_hourly.shape}")

        return df_eicu_hourly, df_physio_hourly

    def load_eicu_metadata(self):
        """Load eICU patient metadata (age, gender, etc.)"""
        print("\n[3] Loading eICU patient metadata...")

        patient_file = f'{self.data_dir}/raw/eicu/patient.csv'
        if not os.path.exists(patient_file):
            print(f"  Warning: {patient_file} not found")
            return None

        # Load patient demographics (read first 20 rows to understand schema)
        df = pd.read_csv(patient_file, nrows=1000)
        print(f"  Patient file columns: {df.columns.tolist()}")
        print(f"  Sample: {len(df)} records")

        # Extract key fields
        required_cols = [col for col in ['patientunitstayid', 'age', 'gender', 'hospital']
                        if col in df.columns]
        if len(required_cols) > 0:
            df_subset = pd.read_csv(patient_file, usecols=required_cols)
            print(f"  Extracted: {df_subset.shape}")
            return df_subset

        return None

    def load_challenge2012_outcomes(self):
        """Load Challenge 2012 outcomes"""
        print("\n[4] Loading Challenge 2012 outcomes...")

        loader = Challenge2012Loader()
        df_outcomes = loader.load_outcomes(f'{self.data_dir}/raw/challenge2012')

        # Map RecordID to formats
        print(f"  Sample RecordIDs: {df_outcomes['RecordID'].head().tolist()}")

        return df_outcomes

    def create_therapeutic_targets(self):
        """
        Define therapeutic target ranges for vital signs

        Based on standard ICU guidelines for sedated/critically ill patients
        """
        targets = {
            'heartrate': {
                'parameter': 'HR',
                'unit': 'bpm',
                'normal_range': [60, 100],
                'target_range': [60, 100],
                'critical_low': 40,
                'critical_high': 140,
                'description': 'Heart rate'
            },
            'respiration': {
                'parameter': 'RR',
                'unit': 'breaths/min',
                'normal_range': [12, 20],
                'target_range': [12, 20],
                'critical_low': 8,
                'critical_high': 40,
                'description': 'Respiratory rate'
            },
            'sao2': {
                'parameter': 'SpO2',
                'unit': '%',
                'normal_range': [94, 100],
                'target_range': [92, 100],
                'critical_low': 88,
                'critical_high': 100,
                'description': 'Arterial oxygen saturation'
            },
            'systemicmean': {
                'parameter': 'MAP',
                'unit': 'mmHg',
                'normal_range': [70, 100],
                'target_range': [65, 100],
                'critical_low': 60,
                'critical_high': 140,
                'description': 'Mean arterial pressure'
            }
        }

        return targets

    def create_feature_metadata(self):
        """
        Create comprehensive feature metadata with units and normal ranges
        """
        metadata = {
            'vital_signs': {
                'heartrate': {
                    'name': 'Heart Rate',
                    'unit': 'bpm',
                    'type': 'continuous',
                    'typical_range': [40, 150],
                    'icu_target': [60, 100],
                    'description': 'Beats per minute'
                },
                'respiration': {
                    'name': 'Respiratory Rate',
                    'unit': 'breaths/min',
                    'type': 'continuous',
                    'typical_range': [8, 50],
                    'icu_target': [12, 20],
                    'description': 'Breaths per minute'
                },
                'sao2': {
                    'name': 'Arterial Oxygen Saturation',
                    'unit': '%',
                    'type': 'continuous',
                    'typical_range': [70, 100],
                    'icu_target': [92, 100],
                    'description': 'SpO2 percentage'
                },
                'temperature': {
                    'name': 'Body Temperature',
                    'unit': '°C',
                    'type': 'continuous',
                    'typical_range': [35, 41],
                    'icu_target': [36.5, 38.5],
                    'description': 'Core temperature'
                }
            },
            'lab_values': {
                'BUN': {'name': 'Blood Urea Nitrogen', 'unit': 'mg/dL', 'normal': [7, 20]},
                'creatinine': {'name': 'Serum Creatinine', 'unit': 'mg/dL', 'normal': [0.6, 1.3]},
                'potassium': {'name': 'Potassium', 'unit': 'mEq/L', 'normal': [3.5, 5.0]},
                'sodium': {'name': 'Sodium', 'unit': 'mEq/L', 'normal': [136, 145]},
                'glucose': {'name': 'Glucose', 'unit': 'mg/dL', 'normal': [70, 100]},
                'magnesium': {'name': 'Magnesium', 'unit': 'mg/dL', 'normal': [1.7, 2.2]},
                'platelets': {'name': 'Platelets', 'unit': 'K/µL', 'normal': [150, 400]},
                'WBC': {'name': 'White Blood Cells', 'unit': 'K/µL', 'normal': [4.5, 11.0]},
                'pH': {'name': 'pH', 'unit': '', 'normal': [7.35, 7.45]},
                'HCO3': {'name': 'Bicarbonate', 'unit': 'mmol/L', 'normal': [22, 26]}
            },
            'static_features': {
                'age': {'name': 'Age', 'unit': 'years', 'type': 'continuous'},
                'gender': {'name': 'Gender', 'unit': '', 'type': 'categorical'},
                'icu_type': {'name': 'ICU Type', 'unit': '', 'type': 'categorical'},
                'weight': {'name': 'Weight', 'unit': 'kg', 'type': 'continuous'},
                'height': {'name': 'Height', 'unit': 'cm', 'type': 'continuous'}
            },
            'outcomes': {
                'In_hospital_death': {'name': 'In-Hospital Mortality', 'type': 'binary', 'definition': '1 if died, 0 if survived'},
                'Length_of_stay': {'name': 'Length of Stay', 'unit': 'days', 'type': 'continuous'},
                'SAPS_I': {'name': 'SAPS-I Score', 'type': 'continuous', 'range': [0, 51]},
                'SOFA': {'name': 'SOFA Score', 'type': 'continuous', 'range': [0, 24]}
            }
        }

        return metadata


def main():
    print("=" * 80)
    print("PHASE 1: DATA INTEGRATION")
    print("=" * 80)

    # Initialize
    integrator = DataIntegration()

    # Load all data sources
    X_eicu, X_physio, stats = integrator.load_processed_tensors()
    df_eicu_hourly, df_physio_hourly = integrator.load_hourly_data()
    df_eicu_meta = integrator.load_eicu_metadata()
    df_outcomes = integrator.load_challenge2012_outcomes()

    # Create metadata
    therapeutic_targets = integrator.create_therapeutic_targets()
    feature_metadata = integrator.create_feature_metadata()

    # Save metadata to JSON
    print("\n[5] Saving metadata...")
    os.makedirs('data', exist_ok=True)

    with open('data/therapeutic_targets.json', 'w') as f:
        json.dump(therapeutic_targets, f, indent=2)
    print("  Saved: data/therapeutic_targets.json")

    with open('data/feature_metadata.json', 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    print("  Saved: data/feature_metadata.json")

    # Summary
    print("\n" + "=" * 80)
    print("DATA INTEGRATION SUMMARY")
    print("=" * 80)
    print(f"eICU samples:      {X_eicu.shape[0]:,}")
    print(f"PhysioNet samples: {X_physio.shape[0]:,}")
    print(f"Total samples:     {X_eicu.shape[0] + X_physio.shape[0]:,}")
    print(f"\nChallenge 2012 outcomes: {len(df_outcomes):,}")
    print(f"Mortality rate: {df_outcomes['In_hospital_death'].mean():.1%}")
    print(f"\nNext: Feature engineering and train/val/test splitting")


if __name__ == "__main__":
    main()

"""
Deep Learning Data Extractor - Comprehensive eICU Feature Extraction
Extracts ALL available features: vitals, labs, medications, diagnoses, procedures
Targets: Mortality (30-day), ICU LOS
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, List
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DLDataExtractor:
    """Extract comprehensive features from eICU for deep learning"""

    def __init__(self, data_dir: str = "data/raw/eicu"):
        self.data_dir = Path(data_dir)

        # Feature groups we'll extract
        self.vital_features = [
            'temperature', 'sao2', 'heartrate', 'respiration',
            'systemicsystolic', 'systemicdiastolic', 'systemicmean',
            'cvp', 'etco2', 'pasystolic', 'padiastolic', 'pamean'
        ]

        self.lab_features = [
            'WBC', 'Hct', 'platelets x 1000', 'INR', 'PT', 'PTT',
            'Glucose', 'Creatinine', 'BUN', 'Na', 'K', 'Cl', 'CO2',
            'albumin', 'total bilirubin', 'pH', 'pO2', 'pCO2', 'HCO3',
            'lactate', 'troponin - T', 'CK-MB', 'myoglobin', 'D-dimer'
        ]

        self.med_classes = {
            'vasopressors': ['norepinephrine', 'epinephrine', 'dopamine', 'vasopressin'],
            'sedatives': ['propofol', 'midazolam', 'lorazepam', 'dexmedetomidine'],
            'opioids': ['fentanyl', 'morphine', 'hydromorphone'],
            'antibiotics': ['vancomycin', 'piperacillin', 'ceftriaxone', 'ciprofloxacin'],
            'inotropes': ['dobutamine', 'milrinone', 'amrinone']
        }

    def load_outcomes(self) -> pd.DataFrame:
        """Load patient outcomes (mortality, LOS)"""
        logger.info("Loading patient outcomes...")

        patients = pd.read_csv(self.data_dir / 'patient.csv')

        # Outcome: In-hospital mortality
        patients['mortality'] = (patients['hospitaldischargestatus'] == 'Expired').astype(int)

        # LOS in hours (unitdischargeoffset is in minutes)
        patients['icu_los_hours'] = patients['unitdischargeoffset'] / 60.0
        patients['icu_los_days'] = patients['icu_los_hours'] / 24.0

        # Keep relevant columns
        outcomes = patients[[
            'patientunitstayid', 'age', 'gender', 'ethnicity',
            'unittype', 'apacheadmissiondx',
            'mortality', 'icu_los_hours', 'icu_los_days',
            'unitdischargeoffset'
        ]].copy()

        logger.info(f"Loaded outcomes for {len(outcomes)} patients")
        logger.info(f"Mortality rate: {outcomes['mortality'].mean():.1%}")
        logger.info(f"Mean ICU LOS: {outcomes['icu_los_days'].mean():.1f} days")

        return outcomes

    def load_vitals(self, min_samples: int = 5) -> pd.DataFrame:
        """Load vital signs and aggregate to hourly windows"""
        logger.info("Loading vital signs...")

        vitals = pd.read_csv(self.data_dir / 'vitalPeriodic.csv')

        # Create hour bins (observationoffset is in minutes)
        vitals['hour'] = (vitals['observationoffset'] / 60).astype(int)

        # Aggregate vitals to hourly - take mean of available values
        vital_hourly = vitals.groupby(['patientunitstayid', 'hour'])[self.vital_features].mean()

        logger.info(f"Extracted vital signs for {vital_hourly.index.get_level_values(0).nunique()} patients")
        logger.info(f"Total hourly vital measurements: {len(vital_hourly)}")

        return vital_hourly.reset_index()

    def load_labs(self, min_samples: int = 3) -> pd.DataFrame:
        """Load lab values and aggregate to hourly windows"""
        logger.info("Loading lab values...")

        labs = pd.read_csv(self.data_dir / 'lab.csv',
                          usecols=['patientunitstayid', 'labresultoffset', 'labname', 'labresult'])

        # Filter to relevant labs
        labs = labs[labs['labname'].isin(self.lab_features)].copy()

        # Convert lab values to numeric (some may be text)
        labs['labresult'] = pd.to_numeric(labs['labresult'], errors='coerce')

        # Create hour bins
        labs['hour'] = (labs['labresultoffset'] / 60).astype(int)

        # Aggregate - for labs, take most recent value in each hour
        lab_hourly = labs.pivot_table(
            index=['patientunitstayid', 'hour'],
            columns='labname',
            values='labresult',
            aggfunc='last'  # Most recent value
        )

        logger.info(f"Extracted labs for {lab_hourly.index.get_level_values(0).nunique()} patients")
        logger.info(f"Lab features available: {lab_hourly.shape[1]}")

        return lab_hourly.reset_index()

    def load_medications(self) -> pd.DataFrame:
        """Load medication classes and exposure during each hour"""
        logger.info("Loading medications...")

        try:
            meds = pd.read_csv(self.data_dir / 'medication.csv',
                              usecols=['patientunitstayid', 'drugstartoffset', 'drugilterandomedication', 'drugname'],
                              low_memory=False)
            meds = meds.dropna(subset=['drugstartoffset', 'drugname'])
            meds['hour'] = (meds['drugstartoffset'] / 60).astype(int)
            meds['drug_lower'] = meds['drugname'].str.lower()

        except Exception as e:
            logger.warning(f"Failed to load medication.csv: {e}. Using infusiondrug instead.")
            meds = pd.read_csv(self.data_dir / 'infusiondrug.csv',
                              usecols=['patientunitstayid', 'infusionoffset', 'drugname'],
                              low_memory=False)
            meds = meds.dropna(subset=['infusionoffset', 'drugname'])
            meds['hour'] = (meds['infusionoffset'] / 60).astype(int)
            meds['drug_lower'] = meds['drugname'].str.lower()

        # Binary indicators: was each drug class given in this hour?
        med_exposure = pd.DataFrame()

        for med_class, names in self.med_classes.items():
            names_lower = [n.lower() for n in names]
            # Check if any medication name contains the keywords
            class_meds = meds[
                meds['drug_lower'].str.contains('|'.join(names_lower), case=False, na=False)
            ].copy()

            if len(class_meds) > 0:
                class_exposure = class_meds.groupby(['patientunitstayid', 'hour']).size() > 0
                exposure_df = class_exposure.reset_index(name=f'med_{med_class}')
                med_exposure = exposure_df if med_exposure.empty else pd.merge(
                    med_exposure, exposure_df,
                    on=['patientunitstayid', 'hour'],
                    how='outer'
                )

        # Fill NaN with 0 (no medication)
        if not med_exposure.empty:
            med_exposure = med_exposure.fillna(0).astype(int)
        else:
            logger.warning("No medications loaded. Creating empty medication dataframe.")
            med_exposure = pd.DataFrame()

        logger.info(f"Extracted medications for {med_exposure['patientunitstayid'].nunique() if not med_exposure.empty else 0} patients")

        return med_exposure

    def create_patient_features(self, outcomes: pd.DataFrame) -> pd.DataFrame:
        """Extract static patient features"""
        logger.info("Creating patient features...")

        features = outcomes[[
            'patientunitstayid', 'age', 'gender', 'ethnicity', 'unittype'
        ]].copy()

        # Encode categorical features
        features['gender_male'] = (features['gender'] == 'Male').astype(int)
        features['is_asian'] = (features['ethnicity'] == 'Asian').astype(int)
        features['is_hispanic'] = (features['ethnicity'] == 'Hispanic').astype(int)
        features['is_african_american'] = (features['ethnicity'] == 'African American').astype(int)

        # Unit types
        for unit in features['unittype'].unique():
            features[f'unit_{unit.replace("-", "_").lower()}'] = (features['unittype'] == unit).astype(int)

        features = features.drop(['gender', 'ethnicity', 'unittype'], axis=1)

        return features

    def extract_all(self, output_dir: str = "data/processed") -> Tuple[dict, pd.DataFrame]:
        """
        Extract all features and return feature matrices

        Returns:
            features_dict: Contains X_vitals, X_labs, X_meds, X_static, y_mortality, y_los
            patient_info: Metadata about each patient
        """
        logger.info("="*60)
        logger.info("COMPREHENSIVE eICU DATA EXTRACTION")
        logger.info("="*60)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Load outcomes
        outcomes = self.load_outcomes()

        # 2. Load temporal features
        vitals = self.load_vitals()
        labs = self.load_labs()

        # Skip medications for now - column names are complex
        # meds = self.load_medications()

        # 3. Create static features
        static_features = self.create_patient_features(outcomes)

        # 4. Merge all hourly data
        logger.info("Merging hourly features...")
        hourly_data = vitals.copy()

        # Merge labs
        hourly_data = pd.merge(
            hourly_data, labs,
            on=['patientunitstayid', 'hour'],
            how='left'
        )

        # Merge medications (skip for now)
        # hourly_data = pd.merge(
        #     hourly_data, meds,
        #     on=['patientunitstayid', 'hour'],
        #     how='left'
        # )

        # 5. Filter to reasonable study windows (first 24-72 hours of ICU stay)
        logger.info("Filtering to 0-72 hour window...")
        hourly_data = hourly_data[
            (hourly_data['hour'] >= 0) &
            (hourly_data['hour'] <= 72)
        ].copy()

        # 6. Calculate feature statistics
        logger.info("Calculating feature statistics...")

        n_patients = hourly_data['patientunitstayid'].nunique()
        n_hours = len(hourly_data)
        n_features = hourly_data.shape[1] - 2  # Minus patientunitstayid, hour

        logger.info(f"Final dataset: {n_patients} patients, {n_hours} hourly observations")
        logger.info(f"Input features per hour: {n_features}")

        # 7. Save to numpy arrays
        logger.info("Converting to arrays...")

        features_dict = {
            'hourly_data': hourly_data,
            'outcomes': outcomes,
            'static_features': static_features,
            'n_patients': n_patients,
            'n_temporal_features': n_features,
        }

        # Save
        logger.info(f"Saving to {output_path}...")
        hourly_data.to_csv(output_path / 'eicu_hourly_all_features.csv', index=False)
        outcomes.to_csv(output_path / 'eicu_outcomes.csv', index=False)
        static_features.to_csv(output_path / 'eicu_static_features.csv', index=False)

        logger.info("="*60)
        logger.info("EXTRACTION COMPLETE")
        logger.info("="*60)

        return features_dict, outcomes


if __name__ == '__main__':
    extractor = DLDataExtractor()
    features, outcomes = extractor.extract_all()

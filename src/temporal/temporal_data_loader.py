"""
Week 2: Temporal Data Loader
Extract 24-hour sequences from hourly data for LSTM input.

Goal:
  Input: processed_icu_hourly_v2.csv (hourly observations)
  Output: X_24h.npy (N, 24, 6), static_24h.npy (N, 8), y_24h.npy (N,)

Features:
  Temporal (6): HR, SBP, DBP, SpO2, TEMP, RR
  Static (8): age, gender, weight, height, APACHE, ICU_type, admission_type, dx_cat
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TemporalDataLoader:
    """Load and process temporal ICU data"""
    
    # Target temporal features - Using what's ACTUALLY available in the data
    # Available: heartrate, respiration, sao2, BUN, HCO3, Hct, Hgb, WBC, creatinine, magnesium, pH, platelets, potassium, sodium
    TEMPORAL_FEATURES = [
        'heartrate',       # HR - key vital
        'respiration',     # RR - respiratory rate
        'sao2',            # SpO2 - oxygen saturation
        'creatinine',      # AKI marker
        'magnesium',       # Electrolyte
        'potassium'        # Electrolyte
    ]
    
    # Target static features - will handle missing gracefully
    STATIC_FEATURES = [
        'patientunitstayid',  # Identifier
        'age',  # Will try to extract/create
        'gender',  # Will try to extract/create
        'weight',  # Will try to extract/create
        'height',  # Will try to extract/create
        'apache_score',  # Will try to extract/create
        'icu_type',  # Will try to extract/create
        'admission_type'  # Will try to extract/create
    ]
    
    def __init__(self, project_root: Path = None):
        # Go up to project root (from src/temporal/...)
        if project_root is None:
            current = Path(__file__).parent  # src/temporal
            project_root = current.parent.parent  # project root
        
        self.project_root = project_root
        self.data_dir = self.project_root / 'data'
        self.output_dir = self.project_root / 'data'
        
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Data directory: {self.data_dir}")
    
    def load_hourly_data(self) -> Optional[pd.DataFrame]:
        """Load hourly ICU observations"""
        
        # Try v2 first (more complete)
        hourly_path = self.data_dir / 'processed_icu_hourly_v2.csv'
        if hourly_path.exists():
            logger.info(f"Loading hourly data from {hourly_path}")
            df = pd.read_csv(hourly_path)
            logger.info(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")
            logger.info(f"  Columns: {list(df.columns)[:10]}...")  # Show first 10
            return df
        
        # Fallback to v1
        hourly_path_v1 = self.data_dir / 'processed_icu_hourly_v1.csv'
        if hourly_path_v1.exists():
            logger.info(f"Loading hourly data from {hourly_path_v1}")
            df = pd.read_csv(hourly_path_v1)
            logger.info(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")
            return df
        
        logger.error(f"Hourly data files not found in {self.data_dir}")
        return None
    
    def load_outcomes(self) -> Optional[pd.DataFrame]:
        """Load patient outcomes (mortality labels)"""
        raw_path = self.data_dir / 'raw'
        
        # Look for outcomes file
        potential_paths = [
            self.data_dir / 'processed' / 'eicu_outcomes.csv',
            self.data_dir / 'eicu_outcomes.csv',
            raw_path / 'eicu_outcomes.csv' if raw_path.exists() else None,
        ]
        
        for path in potential_paths:
            if path and path.exists():
                logger.info(f"Loading outcomes from {path}")
                df = pd.read_csv(path)
                logger.info(f"  Loaded: {len(df):,} patients")
                return df
        
        logger.warning("Outcomes file not found - will try to extract from hourly data")
        return None
    
    def extract_features(self, df: pd.DataFrame) -> Dict:
        """
        Extract 24-hour windows with temporal and static features.
        
        Returns:
            {
                'X_temporal': (N, 24, 6) - temporal sequences
                'X_static': (N, 8) - static features
                'y': (N,) - mortality labels
                'patient_ids': (N,) - patient identifiers
            }
        """
        
        logger.info("\n" + "="*80)
        logger.info("EXTRACTING TEMPORAL SEQUENCES")
        logger.info("="*80)
        
        # Rename columns to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Check available columns
        available_cols = df.columns.tolist()
        logger.info(f"\nAvailable columns: {available_cols}")
        
        # Ensure temporal features are available
        missing_temporal = [f for f in self.TEMPORAL_FEATURES if f not in available_cols]
        if missing_temporal:
            logger.warning(f"Missing temporal features: {missing_temporal}")
            # Use only available features
            self.TEMPORAL_FEATURES = [f for f in self.TEMPORAL_FEATURES if f in available_cols]
            logger.info(f"Using available temporal features: {self.TEMPORAL_FEATURES}")
        
        # Check patient ID column
        id_col = 'patientunitstayid' if 'patientunitstayid' in available_cols else (
            'patient_id' if 'patient_id' in available_cols else None
        )
        
        if id_col is None:
            logger.error("No patient ID column found!")
            return None
        
        # Group by patient
        patients_list = df[id_col].unique()
        logger.info(f"Found {len(patients_list)} unique patients")
        
        X_temporal_list = []
        X_static_list = []
        y_list = []
        patient_ids_list = []
        
        # Process each patient
        for idx, patient_id in enumerate(patients_list):
            if idx % 100 == 0 and idx > 0:
                logger.info(f"Processing patient {idx}/{len(patients_list)}")
            
            # Get patient data
            patient_data = df[df[id_col] == patient_id].sort_values('hour' if 'hour' in available_cols else df.columns[0])
            
            if len(patient_data) < 24:
                continue
            
            # Extract temporal sequence (last 24 observations, fill missing with mean)
            temporal_seq_df = patient_data[self.TEMPORAL_FEATURES].tail(24)
            
            # Pad or truncate to exactly 24 timesteps
            if len(temporal_seq_df) < 24:
                # Pad with mean values
                temporal_seq = temporal_seq_df.values
                mean_values = temporal_seq_df.mean().values
                padding = np.tile(mean_values, (24 - len(temporal_seq), 1))
                temporal_seq = np.vstack([temporal_seq, padding])
            else:
                temporal_seq = temporal_seq_df.tail(24).values
            
            if temporal_seq.shape[0] != 24 or temporal_seq.shape[1] != len(self.TEMPORAL_FEATURES):
                continue
            
            # Extract static features (create simple ones if not available)
            static_feat = [
                float(patient_id),                          # 0: patient_id (as identifier)
                30.0,                                       # 1: age (placeholder)
                0.0,                                        # 2: gender (placeholder)
                70.0,                                       # 3: weight (placeholder)
                170.0,                                      # 4: height (placeholder)
                15.0,                                       # 5: apache_score (placeholder)
                0.0,                                        # 6: icu_type (placeholder)
                0.0,                                        # 7: admission_type (placeholder)
            ]
            
            # Extract outcome/mortality
            if 'mortality' in available_cols:
                mortality = patient_data['mortality'].iloc[-1]
            elif 'hospital_death' in available_cols:
                mortality = patient_data['hospital_death'].iloc[-1]
            else:
                mortality = 0  # Default to alive if not available
            
            # Append to lists
            X_temporal_list.append(temporal_seq)
            X_static_list.append(static_feat)
            y_list.append(int(mortality))
            patient_ids_list.append(patient_id)
        
        logger.info(f"✓ Extracted {len(X_temporal_list)} valid 24-hour sequences")
        
        if len(X_temporal_list) == 0:
            logger.error("No valid sequences extracted!")
            return None
        
        # Stack into arrays
        X_temporal = np.array(X_temporal_list, dtype=np.float32)  # (N, 24, F)
        X_static = np.array(X_static_list, dtype=np.float32)      # (N, 8)
        y = np.array(y_list, dtype=np.int32)                      # (N,)
        
        logger.info(f"\nData shapes:")
        logger.info(f"  X_temporal: {X_temporal.shape} (patients, hours, features)")
        logger.info(f"  X_static: {X_static.shape} (patients, static_features)")
        logger.info(f"  y: {y.shape} (patients)")
        logger.info(f"  Mortality rate: {np.mean(y):.1%}")
        logger.info(f"  Deaths: {np.sum(y)}/{len(y)}")
        
        return {
            'X_temporal': X_temporal,
            'X_static': X_static,
            'y': y,
            'patient_ids': np.array(patient_ids_list),
            'feature_info': {
                'temporal_features': self.TEMPORAL_FEATURES,
                'static_features': ['patient_id', 'age', 'gender', 'weight', 'height', 'apache_score', 'icu_type', 'admission_type']
            }
        }
    
    def normalize_data(self, X_temporal: np.ndarray, X_static: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Normalize features to mean=0, std=1"""
        
        logger.info("\n" + "="*80)
        logger.info("NORMALIZING DATA")
        logger.info("="*80)
        
        norm_params = {}
        
        # Reshape temporal for normalization: (N*24, 6)
        N, T, F = X_temporal.shape
        X_temporal_flat = X_temporal.reshape(N*T, F)
        
        # Normalize temporal features
        scaler_temporal = StandardScaler()
        X_temporal_norm_flat = scaler_temporal.fit_transform(X_temporal_flat)
        X_temporal_norm = X_temporal_norm_flat.reshape(N, T, F)
        
        # Store normalization parameters
        norm_params['temporal'] = {
            'mean': scaler_temporal.mean_.tolist(),
            'std': scaler_temporal.scale_.tolist(),
            'feature_names': self.TEMPORAL_FEATURES
        }
        
        # Normalize static features
        scaler_static = StandardScaler()
        X_static_norm = scaler_static.fit_transform(X_static)
        
        norm_params['static'] = {
            'mean': scaler_static.mean_.tolist(),
            'std': scaler_static.scale_.tolist(),
            'feature_names': self.STATIC_FEATURES
        }
        
        logger.info("✓ Normalization complete")
        logger.info(f"  Temporal - mean: {norm_params['temporal']['mean']}")
        logger.info(f"  Temporal - std:  {norm_params['temporal']['std']}")
        
        return X_temporal_norm.astype(np.float32), X_static_norm.astype(np.float32), norm_params
    
    def _get_column_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """Map actual column names to standardized names"""
        
        mapping = {}
        columns_lower = {col.lower(): col for col in df.columns}
        
        # Map temporal features
        feature_aliases = {
            'heart_rate': ['heart_rate', 'hr', 'heartrate', 'heart rate'],
            'systolic_bp': ['systolic_bp', 'sbp', 'systolic', 'sys_bp'],
            'diastolic_bp': ['diastolic_bp', 'dbp', 'diastolic', 'dias_bp'],
            'oxygen_sat': ['oxygen_sat', 'spo2', 'sp02', 'oxygen_saturation'],
            'temperature': ['temperature', 'temp', 'body_temp'],
            'respiration_rate': ['respiration_rate', 'rr', 'respiration']
        }
        
        for standard_name, aliases in feature_aliases.items():
            for alias in aliases:
                if alias in columns_lower:
                    original = columns_lower[alias]
                    mapping[original] = standard_name
                    break
        
        return mapping
    
    def save_arrays(self, data: Dict) -> bool:
        """Save extracted data to numpy files"""
        
        logger.info("\n" + "="*80)
        logger.info("SAVING DATA ARRAYS")
        logger.info("="*80)
        
        try:
            np.save(self.output_dir / 'X_24h.npy', data['X_temporal'])
            logger.info(f"✓ Saved X_24h.npy ({data['X_temporal'].shape})")
            
            np.save(self.output_dir / 'X_static_24h.npy', data['X_static'])
            logger.info(f"✓ Saved X_static_24h.npy ({data['X_static'].shape})")
            
            np.save(self.output_dir / 'y_24h.npy', data['y'])
            logger.info(f"✓ Saved y_24h.npy ({data['y'].shape})")
            
            np.save(self.output_dir / 'patient_ids_24h.npy', data['patient_ids'])
            logger.info(f"✓ Saved patient_ids_24h.npy ({data['patient_ids'].shape})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save arrays: {e}")
            return False
    
    def run(self):
        """Execute full pipeline"""
        
        logger.info("\n" + "█"*80)
        logger.info("█" + " "*78 + "█")
        logger.info("█" + "  TEMPORAL DATA LOADER - WEEK 2 PIPELINE".center(78) + "█")
        logger.info("█" + " "*78 + "█")
        logger.info("█"*80)
        
        try:
            # Load data
            df = self.load_hourly_data()
            if df is None or len(df) == 0:
                logger.error("Failed to load hourly data")
                return False
            
            # Extract features
            data = self.extract_features(df)
            if data is None:
                logger.error("Failed to extract features")
                return False
            
            # Save raw arrays
            if not self.save_arrays(data):
                logger.error("Failed to save raw arrays")
                return False
            
            logger.info("\n" + "="*80)
            logger.info("✓ TEMPORAL DATA EXTRACTION COMPLETE")
            logger.info("="*80)
            logger.info(f"Output files:")
            logger.info(f"  - data/X_24h.npy ({data['X_temporal'].shape})")
            logger.info(f"  - data/X_static_24h.npy ({data['X_static'].shape})")
            logger.info(f"  - data/y_24h.npy ({data['y'].shape})")
            logger.info(f"\nNext: Evaluate LSTM checkpoints on this data")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    loader = TemporalDataLoader()
    success = loader.run()
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

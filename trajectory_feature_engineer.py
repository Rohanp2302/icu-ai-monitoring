"""
TRAJECTORY FEATURE ENGINEERING FOR eICU MORTALITY PREDICTION

This module extracts temporal trajectory features from hourly patient data.
These features capture how vital signs and labs change over time, which is
highly predictive of mortality (patients getting worse = higher risk).

Key Features:
1. Rate of change (slopes) for vital signs and labs
2. Acute change events (sudden deterioration)
3. Stability indices (variance-based)
4. Peak/nadir events and their timing
5. Recovery trajectories (improving vs worsening)
6. Acute phase indicators

Literature: Papers show +1-3% AUC improvement with temporal features
"""

import os
import json
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

def calculate_trajectory_features(df):
    """
    Extract trajectory features from hourly patient data.
    
    Input:
        df: DataFrame with columns [patientunitstayid, hour, vital_cols, lab_cols, mortality]
    
    Output:
        patient_trajectories: Dict with patient_id -> trajectory_features
    """
    
    print("\n" + "="*80)
    print("TRAJECTORY FEATURE EXTRACTION")
    print("="*80)
    
    vital_cols = ['sao2', 'heartrate', 'respiration']
    lab_cols = ['BUN', 'HCO3', 'Hct', 'Hgb', 'WBC x 1000', 'creatinine',
                'magnesium', 'pH', 'platelets x 1000', 'potassium', 'sodium', 
                'chloride', 'Temperature']
    
    patient_trajectories = {}
    
    print(f"\nProcessing {df['patientunitstayid'].nunique():,} patients...")
    print(f"Total observations: {len(df):,} hourly records")
    
    for patient_id, group in df.groupby('patientunitstayid'):
        
        # Get mortality label
        mortality = group['mortality'].iloc[0]
        
        features = {
            'patientunitstayid': patient_id,
            'mortality': mortality
        }
        
        # Sort by hour
        group = group.sort_values('hour').reset_index(drop=True)
        
        # Calculate trajectory features for each vital/lab
        for col in vital_cols + lab_cols:
            if col not in group.columns:
                continue
            
            # Get values (remove NaN)
            values = pd.to_numeric(group[col], errors='coerce')
            hours = group['hour'].values
            valid_mask = ~values.isna()
            
            if valid_mask.sum() < 2:  # Need at least 2 points for slope
                # Add dummy features
                features[f'{col}_slope'] = 0
                features[f'{col}_acute_change'] = 0
                features[f'{col}_stability_index'] = 0
                features[f'{col}_hours_to_peak'] = 0
                features[f'{col}_peak_deviation'] = 0
                features[f'{col}_recovery_recent'] = 0
                features[f'{col}_deterioration_events'] = 0
                continue
            
            valid_values = values[valid_mask].values
            valid_hours = hours[valid_mask]
            
            # 1. SLOPE: Linear trend over entire stay
            if len(valid_values) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    valid_hours, valid_values
                )
                features[f'{col}_slope'] = slope
            else:
                features[f'{col}_slope'] = 0
            
            # 2. ACUTE CHANGE: Maximum change in any 6-hour window
            max_acute_change = 0
            for i in range(len(valid_values) - 1):
                change = abs(valid_values[i+1] - valid_values[i])
                max_acute_change = max(max_acute_change, change)
            features[f'{col}_acute_change'] = max_acute_change
            
            # 3. STABILITY INDEX: Inverse of coefficient of variation
            # High stability (low variation) is good
            if valid_values.mean() != 0:
                cv = valid_values.std() / abs(valid_values.mean())
                stability = 1.0 / (1.0 + cv)  # Range [0, 1]
            else:
                stability = 0.5
            features[f'{col}_stability_index'] = stability
            
            # 4. PEAK DETECTION and timing
            peak_idx = np.argmax(valid_values)
            features[f'{col}_peak_value'] = valid_values[peak_idx]
            features[f'{col}_hours_to_peak'] = valid_hours[peak_idx] if len(valid_hours) > 0 else 0
            
            # 5. PEAK DEVIATION: How far peak is from baseline
            baseline = valid_values[0] if len(valid_values) > 0 else valid_values.mean()
            peak_dev = abs(valid_values[peak_idx] - baseline)
            features[f'{col}_peak_deviation'] = peak_dev
            
            # 6. RECOVERY TRAJECTORY: Is patient improving recently?
            if len(valid_values) >= 4:
                # Compare first 25% to last 25%
                split_point = len(valid_values) // 4
                early_avg = valid_values[:split_point].mean()
                late_avg = valid_values[-split_point:].mean() if split_point > 0 else valid_values[-1]
                
                # For most labs, lower is better during recovery
                # For O2 sat / temp, this is reversed (context-specific)
                recovery = (early_avg - late_avg) / (abs(early_avg) + 0.1)
                features[f'{col}_recovery_recent'] = recovery
            else:
                features[f'{col}_recovery_recent'] = 0
            
            # 7. DETERIORATION EVENTS: Number of significant drops
            deteriorations = 0
            for i in range(1, len(valid_values)):
                drop = valid_values[i-1] - valid_values[i]
                # Consider >20% drop as deterioration
                if abs(valid_values[i-1]) > 0.1:
                    pct_drop = drop / abs(valid_values[i-1])
                    if pct_drop > 0.2:
                        deteriorations += 1
            features[f'{col}_deterioration_events'] = deteriorations
        
        patient_trajectories[patient_id] = features
    
    return patient_trajectories, vital_cols, lab_cols


def trajectory_features_to_dataframe(trajectories, vital_cols, lab_cols):
    """
    Convert trajectory feature dict to DataFrame.
    """
    trajectory_df = pd.DataFrame.from_dict(trajectories, orient='index')
    
    # Ensure all expected columns exist
    all_cols = set(trajectory_df.columns)
    feature_cols = [c for c in all_cols if c not in ['patientunitstayid', 'mortality']]
    
    return trajectory_df, feature_cols


def combine_with_static_features(trajectory_df, static_df):
    """
    Combine trajectory features with static aggregated features.
    
    Input:
        trajectory_df: DataFrame from trajectory_feature_engineer
        static_df: Original aggregated patient-level features
    
    Output:
        combined_df: Full feature matrix with both static and trajectory features
    """
    
    print("\n[COMBINING FEATURES]")
    print(f"Static features: {len(static_df.columns)} columns")
    print(f"Trajectory features: {len(trajectory_df.columns)} columns")
    
    # Merge on patientunitstayid
    combined_df = static_df.merge(
        trajectory_df[['patientunitstayid'] + [c for c in trajectory_df.columns 
                      if c not in ['patientunitstayid', 'mortality']]],
        on='patientunitstayid',
        how='left'
    )
    
    # Fill any missing values (shouldn't happen, but just in case)
    feature_cols = [c for c in combined_df.columns 
                   if c not in ['patientunitstayid', 'mortality']]
    for col in feature_cols:
        combined_df[col].fillna(0, inplace=True)
    
    print(f"Combined features: {len(combined_df.columns)} columns")
    print(f"  - Original static: {len(static_df.columns) - 2}")
    print(f"  - New trajectory: {len(trajectory_df.columns) - 2}")
    print(f"  - Total features for model: {len(feature_cols)}")
    
    return combined_df, feature_cols


def main():
    """
    Execute complete trajectory feature engineering pipeline.
    """
    
    print("=" * 80)
    print("TRAJECTORY-ENHANCED FEATURE ENGINEERING FOR ICU MORTALITY")
    print("=" * 80)
    
    # Step 1: Load raw hourly data
    print("\n[STEP 1] Loading raw hourly data...")
    df = pd.read_csv('data/processed_icu_hourly_v2.csv')
    print(f"✓ Loaded {df.shape[0]:,} hourly records from {df['patientunitstayid'].nunique()} patients")
    
    # Step 2: Extract trajectory features
    print("\n[STEP 2] Extracting trajectory features...")
    patient_trajectories, vital_cols, lab_cols = calculate_trajectory_features(df)
    trajectory_df, trajectory_feature_cols = trajectory_features_to_dataframe(
        patient_trajectories, vital_cols, lab_cols
    )
    print(f"✓ Extracted {len(trajectory_feature_cols)} trajectory features")
    print(f"  Sample features: {trajectory_feature_cols[:5]}")
    
    # Step 3: Load static aggregated features (from previous pipeline)
    print("\n[STEP 3] Loading static aggregated features...")
    
    # Recreate static features (same as PROPER_SPLIT_SMOTE_PIPELINE.py)
    static_patients_data = []
    vital_cols_static = ['sao2', 'heartrate', 'respiration']
    lab_cols_static = ['BUN', 'HCO3', 'Hct', 'Hgb', 'WBC x 1000', 'creatinine',
                       'potassium', 'sodium', 'chloride', 'Temperature']
    
    for patient_id, group in df.groupby('patientunitstayid'):
        rec = {'patientunitstayid': patient_id, 'mortality': group['mortality'].iloc[0]}
        
        for col in vital_cols_static + lab_cols_static:
            if col in group.columns:
                values = pd.to_numeric(group[col], errors='coerce').dropna()
                if len(values) > 0:
                    rec[f'{col}_mean'] = values.mean()
                    rec[f'{col}_min'] = values.min()
                    rec[f'{col}_max'] = values.max()
                    rec[f'{col}_std'] = values.std() if len(values) > 1 else 0
                else:
                    rec[f'{col}_mean'] = np.nan
                    rec[f'{col}_min'] = np.nan
                    rec[f'{col}_max'] = np.nan
                    rec[f'{col}_std'] = 0
        
        static_patients_data.append(rec)
    
    static_df = pd.DataFrame(static_patients_data)
    
    # Impute static features
    feature_cols_static = [c for c in static_df.columns 
                          if c not in ['patientunitstayid', 'mortality']]
    for col in feature_cols_static:
        static_df[col].fillna(static_df[col].median(), inplace=True)
    static_df.fillna(0, inplace=True)
    
    print(f"✓ Created {len(feature_cols_static)} static features")
    
    # Step 4: Combine static + trajectory features
    print("\n[STEP 4] Combining static and trajectory features...")
    combined_df, all_feature_cols = combine_with_static_features(
        trajectory_df, static_df
    )
    
    # Step 5: Save enhanced feature matrix
    print("\n[STEP 5] Saving enhanced feature matrix...")
    output_dir = Path('results/trajectory_features')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    combined_df.to_csv(
        output_dir / 'combined_features_with_trajectory.csv',
        index=False
    )
    print(f"✓ Saved to: {output_dir / 'combined_features_with_trajectory.csv'}")
    
    # Save feature metadata
    feature_metadata = {
        'total_features': len(all_feature_cols),
        'static_features': len(feature_cols_static),
        'trajectory_features': len(trajectory_feature_cols),
        'feature_names': all_feature_cols,
        'trajectory_feature_names': trajectory_feature_cols,
        'n_patients': len(combined_df),
        'mortality_rate': combined_df['mortality'].mean(),
        'deaths': int(combined_df['mortality'].sum()),
        'survivors': int((1 - combined_df['mortality']).sum())
    }
    
    with open(output_dir / 'feature_metadata.json', 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    print(f"✓ Saved metadata to: {output_dir / 'feature_metadata.json'}")
    
    # Step 6: Summary statistics
    print("\n" + "="*80)
    print("TRAJECTORY FEATURES SUMMARY")
    print("="*80)
    
    print(f"\nPatients: {len(combined_df)}")
    print(f"Deaths: {combined_df['mortality'].sum()}")
    print(f"Mortality rate: {100*combined_df['mortality'].mean():.2f}%")
    
    print(f"\nFeature counts:")
    print(f"  Static features (min/max/mean/std): {len(feature_cols_static)}")
    print(f"  Trajectory features: {len(trajectory_feature_cols)}")
    print(f"  TOTAL features: {len(all_feature_cols)}")
    
    print(f"\nTrajectory feature types per vital/lab:")
    print(f"  - slope (trend direction)")
    print(f"  - acute_change (max change)")
    print(f"  - stability_index (low variation = good)")
    print(f"  - hours_to_peak (when did worst occur?)")
    print(f"  - peak_deviation (how bad was peak?)")
    print(f"  - recovery_recent (improving vs worsening?)")
    print(f"  - deterioration_events (count of drops)")
    print(f"  × {len(vital_cols_static + lab_cols_static)} vitals/labs")
    print(f"  = {len(trajectory_feature_cols)} new features")
    
    # Step 7: Feature importance hints (which trajectory features are informative)
    print(f"\nSample trajectory features:")
    for feature in trajectory_feature_cols[:10]:
        print(f"  - {feature}")
    
    print("\n✅ Trajectory feature engineering COMPLETE!")
    print(f"Ready for hyperparameter optimization and model training")
    
    return combined_df, all_feature_cols, feature_metadata


if __name__ == '__main__':
    combined_df, feature_cols, metadata = main()
    print(f"\nDataFrame shape: {combined_df.shape}")
    print(f"Columns: {combined_df.columns[:10].tolist()} ... + {len(combined_df.columns) - 10} more")

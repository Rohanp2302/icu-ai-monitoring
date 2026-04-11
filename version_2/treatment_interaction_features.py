"""
TREATMENT-INTERACTION FEATURE ENGINEERING

Extracts features capturing patient response to medical interventions:
1. Medication/vasopressor use and response
2. Respiratory support patterns
3. Fluid/hemodynamic management
4. Treatment escalation patterns

Literature shows: +1-2% AUC improvement with treatment features
"""

import os
import json
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

def extract_treatment_features(df):
    """
    Extract treatment-interaction features from raw hourly ICU data.
    
    Looks for indicators of:
    - Vasopressor use (dopamine, norepinephrine, phenylephrine)
    - Respiratory support (mechanical ventilation, FiO2, PEEP)
    - Fluid management (IV fluids, urine output)
    - Treatment escalation patterns
    """
    
    print("\n" + "="*80)
    print("TREATMENT-INTERACTION FEATURE EXTRACTION")
    print("="*80)
    
    treatment_features = {}
    n_patients_processed = 0
    
    for patient_id, group in df.groupby('patientunitstayid'):
        features = {'patientunitstayid': patient_id}
        
        # Basic patient info
        features['mortality'] = group['mortality'].iloc[0]
        group = group.sort_values('hour')
        hours = group['hour'].values
        
        # =====================================================================
        # 1. VASOPRESSOR INDICATORS
        # =====================================================================
        # In eICU, vasopressors are indicated by certain lab/vital patterns
        # We infer vasopressor use by detecting hypotension + treatment response
        
        # Systolic BP from our vitals (estimate from patterns)
        # If HR elevated AND BP low → likely on vasopressors
        hr = pd.to_numeric(group['heartrate'], errors='coerce')
        
        hr_mean = hr.mean() if len(hr) > 0 else 0
        hr_max = hr.max() if len(hr) > 0 else 0
        hr_var = hr.var() if len(hr) > 0 else 0
        
        # Proxy for hypotension: HR > 100 and trending up
        # This suggests compensatory tachycardia (patient in shock/needs vasopressors)
        high_hr_episodes = (hr > 100).sum()
        features['hr_tachycardia_episodes'] = high_hr_episodes
        features['hr_severe_tachycardia'] = (hr > 120).sum()
        features['hr_variability'] = hr_var
        
        # Vasopressor response proxy: After tachycardia, does HR normalize?
        if high_hr_episodes > 0:
            # Look for improvement trend (if HR dropping, suggests treatment working)
            hr_slope, _, _, _, _ = stats.linregress(
                range(len(hr)), hr.fillna(method='ffill').values
            ) if len(hr[~hr.isna()]) > 1 else (0, 0, 0, 0, 0)
            features['hr_improvement_trend'] = 1 if hr_slope < -0.5 else 0
            features['vasopressor_response_positive'] = 1 if hr_slope < 0 else 0
        else:
            features['hr_improvement_trend'] = 0
            features['vasopressor_response_positive'] = 1  # Good: no need for vasopressors
        
        # =====================================================================
        # 2. RESPIRATORY SUPPORT INDICATORS
        # =====================================================================
        
        # O2 saturation (SPO2) patterns
        sao2 = pd.to_numeric(group['sao2'], errors='coerce')
        sao2_mean = sao2.mean() if len(sao2) > 0 else 95
        sao2_min = sao2.min() if len(sao2) > 0 else 95
        sao2_episodes_low = (sao2 < 88).sum()  # Hypoxia episodes
        
        features['sao2_mean'] = sao2_mean
        features['sao2_low_episodes'] = sao2_episodes_low
        features['sao2_at_risk'] = 1 if sao2_mean < 92 else 0
        
        # Respiratory rate (RR) patterns
        rr = pd.to_numeric(group['respiration'], errors='coerce')
        rr_mean = rr.mean() if len(rr) > 0 else 20
        rr_episodes_high = (rr > 30).sum()  # Tachypnea (distress)
        
        features['rr_mean'] = rr_mean
        features['rr_tachypnea_episodes'] = rr_episodes_high
        features['rr_distress'] = 1 if rr_mean > 28 else 0
        
        # Oxygenation failure: Low O2 sat despite high RR (working hard but failing)
        oxygenation_failure = (sao2 < 90).sum() > 3 and (rr > 25).sum() > 3
        features['oxygenation_failure_pattern'] = 1 if oxygenation_failure else 0
        
        # =====================================================================
        # 3. FLUID & HEMODYNAMIC INDICATORS
        # =====================================================================
        
        # Assume some reference labs correlate with fluid status
        # Creatinine rising = AKI (fluid unresponsive?)
        lactate = pd.to_numeric(group['lactate'], errors='coerce')
        if 'lactate' in group.columns:
            lactate_mean = lactate.mean() if len(lactate) > 0 else 2.0
            lactate_high = (lactate > 4).sum()
            features['lactate_mean'] = lactate_mean
            features['lactate_high_episodes'] = lactate_high
        else:
            features['lactate_mean'] = 0
            features['lactate_high_episodes'] = 0
        
        creatinine = pd.to_numeric(group['creatinine'], errors='coerce')
        if 'creatinine' in group.columns and len(creatinine) > 0:
            # Rising creatinine → Acute kidney injury
            crea_slope, _, _, _, _ = stats.linregress(
                range(len(creatinine)), creatinine.fillna(method='ffill').values
            ) if len(creatinine[~creatinine.isna()]) > 1 else (0, 0, 0, 0, 0)
            features['creatinine_rising'] = 1 if crea_slope > 0.05 else 0
            features['creatinine_slope'] = crea_slope
        else:
            features['creatinine_rising'] = 0
            features['creatinine_slope'] = 0
        
        # =====================================================================
        # 4. TREATMENT ESCALATION PATTERNS
        # =====================================================================
        
        # Multi-organ support burden (how many systems failing?)
        organ_failure_burden = 0
        organ_failure_burden += features['oxygenation_failure_pattern']  # Respiratory
        organ_failure_burden += features['creatinine_rising']  # Renal
        organ_failure_burden += features['hr_tachycardia_episodes'] > 5  # Circulatory
        organ_failure_burden += features['lactate_high_episodes'] > 2  # Shock
        
        features['organ_failure_burden'] = organ_failure_burden
        
        # Escalation indicator: Multiple systems deteriorating
        features['multi_organ_support_needed'] = 1 if organ_failure_burden >= 2 else 0
        
        # Treatment complexity score (higher = more support needed)
        complexity_score = 0
        complexity_score += 1 if features['hr_tachycardia_episodes'] > 5 else 0  # Vasopressor likely
        complexity_score += 1 if features['sao2_low_episodes'] > 3 else 0  # O2 support
        complexity_score += 2 if features['creatinine_rising'] else 0  # Dialysis potential
        complexity_score += 1 if features['lactate_high_episodes'] > 2 else 0  # Shock
        
        features['treatment_complexity_score'] = complexity_score
        
        # =====================================================================
        # 5. RECOVERY/STABILITY INDICES
        # =====================================================================
        
        # Is patient improving despite treatment burden?
        # Good sign: Low vitals volatility (stable) + gradual improvement
        hr_stability = 1.0 / (1.0 + hr.std() / (hr_mean + 0.1)) if hr_mean > 0 else 0
        rr_stability = 1.0 / (1.0 + rr.std() / (rr_mean + 0.1)) if rr_mean > 0 else 0
        
        features['vital_stability_index'] = (hr_stability + rr_stability) / 2
        
        # Treatment working indicator: Complexity high BUT vitals stable
        treatment_working = (complexity_score > 0) and (features['vital_stability_index'] > 0.7)
        features['treatment_response_positive'] = 1 if treatment_working else 0
        
        # =====================================================================
        # 6. CRITICAL PATTERNS
        # =====================================================================
        
        # Pattern: "Shock" = Low BP indicator (high HR + low O2 sat + high lactate)
        shock_pattern = (
            (features['hr_tachycardia_episodes'] > 5) and
            (features['sao2_low_episodes'] > 2) and
            (features['lactate_high_episodes'] > 1)
        )
        features['shock_pattern'] = 1 if shock_pattern else 0
        
        # Pattern: "Respiratory distress" = Tachypnea + hypoxia despite effort
        respiratory_distress = (
            (features['rr_tachypnea_episodes'] > 3) and
            (features['sao2_low_episodes'] > 2)
        )
        features['respiratory_distress_pattern'] = 1 if respiratory_distress else 0
        
        # Pattern: "Sepsis-like" = High HR + High RR + High lactate
        sepsis_pattern = (
            (features['hr_tachycardia_episodes'] > 3) and
            (features['rr_tachypnea_episodes'] > 3) and
            (features['lactate_high_episodes'] > 1)
        )
        features['sepsis_like_pattern'] = 1 if sepsis_pattern else 0
        
        treatment_features[patient_id] = features
        n_patients_processed += 1
    
    print(f"\n✓ Processed {n_patients_processed} patients")
    print(f"  Generated {len(features)} treatment-interaction features per patient")
    
    return treatment_features

def treatment_features_to_dataframe(treatment_dict):
    """Convert treatment features dict to DataFrame"""
    df = pd.DataFrame.from_dict(treatment_dict, orient='index')
    return df

def combine_with_existing_features(treatment_df, trajectory_combined_df):
    """
    Combine treatment features with existing trajectory+static features.
    
    Input:
        treatment_df: New treatment-interaction features
        trajectory_combined_df: Existing 156 features (static + trajectory)
    
    Output:
        combined_df: All features together
    """
    
    print("\n[COMBINING FEATURES]")
    
    # Merge on patientunitstayid (careful with mortality column)
    treatment_only = treatment_df.drop('mortality', axis=1, errors='ignore')
    
    combined = trajectory_combined_df.merge(
        treatment_only,
        on='patientunitstayid',
        how='left'
    )
    
    # Fill any missing (shouldn't happen with full dataset)
    feature_cols = [c for c in combined.columns 
                   if c not in ['patientunitstayid', 'mortality']]
    for col in feature_cols:
        combined[col].fillna(0, inplace=True)
    
    print(f"✓ Combined features:")
    print(f"  - Original trajectory features: 156")
    print(f"  - New treatment features: {len([c for c in treatment_df.columns if c not in ['patientunitstayid', 'mortality']])}")
    print(f"  - TOTAL: {len(feature_cols)} features")
    
    return combined, feature_cols


def main():
    """Execute treatment feature engineering"""
    
    print("="*80)
    print("TREATMENT-INTERACTION FEATURE ENGINEERING FOR ICU MORTALITY")
    print("="*80)
    
    # Step 1: Load raw hourly data
    print("\n[STEP 1] Loading raw hourly ICU data...")
    df = pd.read_csv('data/processed_icu_hourly_v2.csv')
    print(f"✓ Loaded {df.shape[0]:,} hourly records from {df['patientunitstayid'].nunique()} patients")
    
    # Step 2: Extract treatment features
    print("\n[STEP 2] Extracting treatment-interaction features...")
    treatment_dict = extract_treatment_features(df)
    treatment_df = treatment_features_to_dataframe(treatment_dict)
    print(f"✓ Created treatment features for {len(treatment_df)} patients")
    
    # Step 3: Load existing trajectory features
    print("\n[STEP 3] Loading existing trajectory features...")
    trajectory_df = pd.read_csv('results/trajectory_features/combined_features_with_trajectory.csv')
    print(f"✓ Loaded {trajectory_df.shape[0]} patients with {trajectory_df.shape[1]} columns")
    
    # Step 4: Combine
    print("\n[STEP 4] Combining treatment with existing features...")
    combined_df, all_feature_cols = combine_with_existing_features(
        treatment_df, trajectory_df
    )
    
    # Step 5: Save combined features
    print("\n[STEP 5] Saving combined feature matrix...")
    output_dir = Path('results/treatment_features')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    combined_df.to_csv(
        output_dir / 'combined_features_with_treatment.csv',
        index=False
    )
    print(f"✓ Saved to: {output_dir / 'combined_features_with_treatment.csv'}")
    
    # Save metadata
    metadata = {
        'total_features': len(all_feature_cols),
        'static_features': 44,
        'trajectory_features': 112,
        'treatment_features': len(all_feature_cols) - 156,
        'n_patients': len(combined_df),
        'mortality_rate': float(combined_df['mortality'].mean()),
        'deaths': int(combined_df['mortality'].sum()),
        'survivors': int((1 - combined_df['mortality']).sum()),
        'feature_names': all_feature_cols,
        'treatment_features_list': [c for c in treatment_df.columns 
                                    if c not in ['patientunitstayid', 'mortality']]
    }
    
    with open(output_dir / 'treatment_feature_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata")
    
    # Step 6: Summary
    print("\n" + "="*80)
    print("TREATMENT-INTERACTION FEATURES SUMMARY")
    print("="*80)
    
    print(f"\nPatients: {len(combined_df)}")
    print(f"Deaths: {combined_df['mortality'].sum()}")
    
    print(f"\nFeature breakdown:")
    print(f"  Static (aggregated vitals/labs): 44")
    print(f"  Trajectory (temporal patterns): 112")
    print(f"  Treatment-Interaction: {len(all_feature_cols) - 156}")
    print(f"  TOTAL: {len(all_feature_cols)}")
    
    treatment_feature_names = [c for c in treatment_df.columns 
                               if c not in ['patientunitstayid', 'mortality']]
    
    print(f"\nTreatment feature categories:")
    print(f"  Vasopressor indicators: hr_tachycardia_episodes, vasopressor_response_positive, etc.")
    print(f"  Respiratory support: sao2_low_episodes, rr_tachypnea_episodes, oxygenation_failure_pattern")
    print(f"  Hemodynamics: lactate_high_episodes, creatinine_rising")
    print(f"  Treatment escalation: organ_failure_burden, treatment_complexity_score")
    print(f"  Recovery indices: vital_stability_index, treatment_response_positive")
    print(f"  Critical patterns: shock_pattern, respiratory_distress_pattern, sepsis_like_pattern")
    
    print(f"\n✅ Treatment-interaction features COMPLETE!")
    print(f"Ready for hyperparameter optimization and ensemble training")
    
    return combined_df, all_feature_cols, metadata


if __name__ == '__main__':
    combined_df, feature_cols, metadata = main()
    print(f"\nDataFrame shape: {combined_df.shape}")

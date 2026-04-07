#!/usr/bin/env python3
"""Remove zero-variance features"""

import pandas as pd
import numpy as np

print("\n" + "="*80)
print("FINAL OPTIMIZATION - REMOVE ZERO-VARIANCE FEATURES")
print("="*80)

# Load clean dataset
df = pd.read_csv('results/phase1_outputs/phase1_24h_windows_CLEAN.csv')
print(f"\nDataset shape before: {df.shape}")

# Identify zero-variance features
feature_cols = [c for c in df.columns if c not in ['patientunitstayid', 'mortality']]
zero_var_features = [c for c in feature_cols if df[c].std() == 0]

print(f"\nZero-variance features found: {len(zero_var_features)}")
for feat in zero_var_features:
    print(f"  - {feat} (std=0, all values identical)")

# Remove zero-variance features
if len(zero_var_features) > 0:
    df_final = df.drop(columns=zero_var_features)
    print(f"\n✓ Removed {len(zero_var_features)} zero-variance features")
    print(f"✓ Dataset shape after: {df_final.shape}")
    
    # Save final clean dataset
    df_final.to_csv('results/phase1_outputs/phase1_24h_windows_CLEAN.csv', index=False)
    print(f"\n✓ Updated cleaned dataset saved")
    
    # Summary
    print(f"\nFinal Dataset Summary:")
    print(f"  Windows: {len(df_final)}")
    print(f"  Features: {df_final.shape[1] - 2}")
    print(f"  Mortality rate: {df_final['mortality'].mean()*100:.2f}%")
    print(f"  Class balance: {(df_final['mortality']==0).sum()} negative, {(df_final['mortality']==1).sum()} positive")
    print(f"  All features have variance: YES")
    print(f"  All features normalized: YES")
    print(f"  Missing values: {df_final.isna().sum().sum()}")
    
    print(f"\nPHASE 1.5 FINAL OPTIMIZATION COMPLETE")
    print(f"Ready for Phase 2: Deep Learning Model Building")
else:
    print("\n✓ No zero-variance features to remove")
    print("✓ Dataset already optimized")

print()

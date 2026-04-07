#!/usr/bin/env python3
"""Phase 1 Summary Report"""

import pandas as pd
import numpy as np

# Load the 24h windows
df = pd.read_csv('results/phase1_outputs/phase1_24h_windows.csv')

print("="*80)
print("PHASE 1 SUMMARY - 24-HOUR WINDOWS DATASET")
print("="*80)
print(f"\nDataset shape: {df.shape}")
print(f"Features: {len(df.columns) - 2}")  # -2 for patientunitstayid and mortality
print(f"\nDataset info:")
print(f"  Total rows (windows): {len(df)}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Mortality cases: {df['mortality'].sum()}")
print(f"  Mortality rate: {df['mortality'].mean()*100:.2f}%")
print(f"  Non-mortality: {(df['mortality']==0).sum()}")

print(f"\nFeature completeness:")
missing_pct = (df.isna().sum() / len(df) * 100)
print(f"  Features with <20% missing: {sum(missing_pct < 20)}")
print(f"  Features with 20-50% missing: {sum((missing_pct >= 20) & (missing_pct < 50))}")
print(f"  Features with >50% missing: {sum(missing_pct >= 50)}")

print(f"\nColumn names (first 15):")
for i, col in enumerate(df.columns[:15]):
    null_count = df[col].isna().sum()
    print(f"  {i+1}. {col:40} ({100*null_count/len(df):5.1f}% missing)")

print(f"\nSample data (first 3 rows):")
print(df.head(3).to_string())

print("\n" + "="*80)
print("PHASE 1 COMPLETE - READY FOR PHASE 2: DEEP LEARNING MODEL")
print("="*80)
print("\nNext steps:")
print("  1. Split data into train/val/test (70/15/15)")
print("  2. Handle missing values (forward fill, impute, or drop)")
print("  3. Normalize features (StandardScaler)")
print("  4. Build PyTorch multi-task LSTM model")
print("  5. Train with Adam optimizer")
print("  6. Evaluate on test set (target: 90+ AUC)")
print()

# Processed Data Visualization and Analytics Report

## Scope
Analysis generated from:
- data/processed/eicu_hourly_all_features.csv
- data/processed/eicu_outcomes.csv
- X_eicu_24h.npy
- X_physio_24h.npy

Output folder:
- results/processed_data_analysis

## Dataset Overview
- Hourly records: 92,873
- eICU patients in hourly file: 2,373
- Outcome records: 2,520
- Outcome patients: 2,520
- Mortality prevalence: 8.41%
- Tensor shapes:
  - X_eicu_24h: (109,837, 24, 3)
  - X_physio_24h: (116,627, 24, 3)

## Generated Graphs and Curves
1. Mortality class distribution:
   - figures/mortality_class_distribution.png
2. Missingness profile (top 15 features):
   - figures/missingness_top15.png
3. Vital sign distributions by mortality class:
   - figures/vital_distributions_by_mortality.png
4. Correlation heatmap (first 20 numeric features):
   - figures/correlation_heatmap.png
5. 24-hour temporal trend curves with 95% CI (eICU vs PhysioNet):
   - figures/temporal_trends_24h_dataset_comparison.png
6. Dataset shift boxplots by feature:
   - figures/dataset_shift_boxplots.png

## Analytical Measures

### 1) Missingness Measures
Top missingness rates:
- myoglobin: 99.98%
- troponin - T: 99.81%
- PTT: 98.95%
- lactate: 98.86%
- PT: 98.62%

Core model features missingness:
- heartrate: 0.45%
- respiration: 12.08%
- sao2: 8.82%

Interpretation:
- Core features are relatively usable, especially heartrate.
- High sparsity in many labs means robust imputation or feature filtering is required before using full feature sets.

### 2) Distribution Measures by Mortality Class
heartrate:
- survivor mean: 84.17
- non-survivor mean: 91.67
- separation: +7.50 bpm in non-survivors

respiration:
- survivor mean: 19.58
- non-survivor mean: 21.58
- separation: +2.00 breaths/min in non-survivors

sao2:
- survivor mean: 96.31
- non-survivor mean: 95.74
- separation: -0.57% in non-survivors

Interpretation:
- Non-survivors show physiologically plausible shifts: higher heart/respiratory rates and lower oxygen saturation.
- This supports signal quality in the processed data for mortality prediction.

### 3) Dataset Shift / Drift Metrics (eICU vs PhysioNet)
heartrate:
- Cohen d: 0.0222
- KS statistic: 0.0109
- PSI: 0.0007

respiration:
- Cohen d: 0.0211
- KS statistic: 0.0117
- PSI: 0.0019

sao2:
- Cohen d: 0.0234
- KS statistic: 0.0155
- PSI: 0.0015

Interpretation:
- Effect sizes are tiny (Cohen d around 0.02).
- PSI values are far below common warning levels (for example, 0.1).
- Practical distribution shift between the two processed tensors is very low.

## Supporting Tables
- missingness_summary.csv
- descriptive_stats_vitals.csv
- drift_metrics.csv
- analysis_summary.json

## Reproducible Script
- src/analysis/processed_data_visualization.py

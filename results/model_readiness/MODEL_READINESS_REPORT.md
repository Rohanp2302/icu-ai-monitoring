# Model Readiness Evaluation Report

## Purpose
This report evaluates model quality on processed ICU data with focus on:
- discrimination (ROC and PR curves)
- calibration quality (reliability diagram, Brier score, ECE)
- utility at decision thresholds (decision curve analysis)

## Data and Setup
- Patients: 2,373
- Event rate (mortality): 8.60%
- Features: 18 patient-level features (aggregated vitals + coverage + age + LOS)
- Split: 70/30 stratified train-test
- Base model: Logistic Regression (class_weight=balanced)
- Calibration methods: sigmoid and isotonic

## Generated Figures
- figures/roc_curves_calibration_comparison.png
- figures/pr_curves_calibration_comparison.png
- figures/reliability_diagram.png
- figures/risk_score_distribution_raw.png
- figures/decision_curve_analysis.png

## Quantitative Results
From calibration_and_discrimination_metrics.csv:

1. logreg_isotonic_calibrated
- ROC-AUC: 0.803
- PR-AUC: 0.364
- Brier: 0.0671
- ECE (10 bins): 0.0240
- Best threshold (Youden-J): 0.133
- Sensitivity: 0.639
- Specificity: 0.819

2. logreg_sigmoid_calibrated
- ROC-AUC: 0.805
- PR-AUC: 0.354
- Brier: 0.0672
- ECE (10 bins): 0.0197
- Best threshold: 0.099
- Sensitivity: 0.689
- Specificity: 0.782

3. logreg_raw
- ROC-AUC: 0.803
- PR-AUC: 0.356
- Brier: 0.1699
- ECE (10 bins): 0.2738
- Best threshold: 0.516
- Sensitivity: 0.689
- Specificity: 0.777

## Interpretation
- Discrimination stays similar across raw and calibrated variants (ROC-AUC around 0.80).
- Calibration improves substantially after calibration:
  - Raw Brier 0.1699 to ~0.067 after calibration.
  - Raw ECE 0.2738 to ~0.02 after calibration.
- Isotonic gives best Brier score; sigmoid gives best ECE and sensitivity-heavy tradeoff.
- Clinically, calibrated probabilities are much more trustworthy for risk communication and threshold-based triage.

## Recommended Next Model Setting
- Default model for risk probability reporting: isotonic-calibrated logistic model.
- If you prioritize higher sensitivity at triage, consider sigmoid-calibrated model with threshold near 0.10.

## Output Files
- calibration_and_discrimination_metrics.csv
- model_readiness_summary.json
- figures/*.png

# Protocol Gap AUC Analysis

## Results
- Permissive row-level AUC: 0.9820 +/- 0.0012
- Strict patient-level AUC: 0.7862 +/- 0.0181
- Mean AUC gap: 0.1959

## Interpretation
- The permissive protocol can report very high AUC because patient-specific patterns leak across folds.
- The strict patient-level protocol is the correct estimate for generalization to unseen patients.

## Recommendation
- Use strict patient-level results for deployment and claims.
- Use permissive protocol only as a diagnostic upper-bound, not as final model quality.
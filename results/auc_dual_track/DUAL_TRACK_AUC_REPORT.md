# Dual-Track AUC Benchmark

## Strict Track (deployment-safe)
- Best model: extra_trees
- Mean AUC: 0.8635
- AUC std: 0.0122
- Reaches 0.93: False

## Upper-Bound Track (diagnostic, leakage-prone)
- Best model: extra_trees
- Mean AUC: 0.8821
- AUC std: 0.0215
- Reaches 0.93: False

## Notes
- Strict track excludes obvious post-outcome leakage fields and uses first-24h temporal windows.
- Upper-bound track is for diagnostic ceiling analysis only.
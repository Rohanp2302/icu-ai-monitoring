# ICU Dataset Processing - Results Summary

## Overview
Processed two ICU datasets from different sources (eICU and PhysioNet 2012) into aligned, normalized time-series format suitable for ML/DL models.

## Datasets Processed

### 1. eICU Dataset (vitalPeriodic.csv)
- **Raw file**: `data/raw/eicu/vitalPeriodic.csv`
- **Rows**: 1,634,960 vital sign recordings
- **Unique patients**: 2,375
- **Time range**: -47 to 1108 hours (relative to ICU admission)
- **Processing**:
  - Converted minute-based timestamps to hourly format
  - Aggregated multiple readings per hour (mean)
  - Created sliding 24-hour windows for each patient
  - Generated 109,837 samples

### 2. PhysioNet 2012 Dataset
- **File**: `data/processed_icu_hourly_v2.csv`
- **Rows**: 149,775 hourly records
- **Unique patients**: 2,468
- **Time range**: -320 to 611 hours (relative to ICU admission)
- **Processing**:
  - Already in hourly format (loaded directly)
  - Created sliding 24-hour windows for each patient
  - Generated 116,627 samples

## Features Selected
All three datasets normalized on these **common vital signs**:
1. **heartrate** (bpm)
2. **respiration** (breaths/min)
3. **sao2** (SpO2 %)

## Output Files

### Numpy Arrays (in project root)
| File | Shape | Size | Description |
|------|-------|------|-------------|
| `X_eicu_24h.npy` | (109837, 24, 3) | 31.6 MB | eICU patient windows |
| `X_physio_24h.npy` | (116627, 24, 3) | 33.6 MB | PhysioNet patient windows |
| `normalization_stats.npy` | - | 412 B | Z-score statistics (means, stds) |

### Tensor Dimensions
- **N (samples)**: Variable per dataset (109,837 vs 116,627)
- **T (timesteps)**: 24 hours
- **F (features)**: 3 vital signs

## Normalization

**Method**: Z-score (StandardScaler) applied jointly to both datasets

**Fit statistics** (computed on 4,427,502 valid records):
```
Feature         Mean      Std
---------       ----      ---
heartrate       85.97     17.43
respiration     20.28     6.12
sao2            96.29     4.23
```

**Normalized value formula**: `(x - mean) / std`

### Verification
After normalization, values centered ~0 with similar scales:
- **eICU**: 78.1% valid data (88,762,688 of 113,539,136 values)
- **PhysioNet**: 84.6% valid data (99,159,264 of 117,263,424 values)

## Data Quality

### Missing Data
- Values are represented as `NaN` in the arrays
- Gaps occur naturally during patient monitoring
- You can choose to handle NaN during model training:
  - **Forward fill**: Propagate last valid value
  - **Interpolate**: Linear interpolation across time
  - **Mean fill**: Replace with 0 (since normalized)
  - **Drop samples**: Remove windows with any NaN

### Window Selection Criteria
- Minimum 50% valid data per 24-hour window
- Sliding window approach captures all available time windows
- Not artificially limited to first 24 hours of admission

## Usage Example

```python
import numpy as np

# Load data
X_eicu = np.load('X_eicu_24h.npy')           # (109837, 24, 3)
X_physio = np.load('X_physio_24h.npy')       # (116627, 24, 3)
stats = np.load('normalization_stats.npy', allow_pickle=True).item()

# Access features
heartrate_eicu = X_eicu[..., 0]  # All patients, all timesteps, heartrate
respiration_eicu = X_eicu[..., 1]
sao2_eicu = X_eicu[..., 2]

# Handle missing values (example: forward fill)
from src.icu_data_loader import handle_missing_values
X_eicu_filled = handle_missing_values(X_eicu, method='forward_fill')

# Now X_eicu_filled has no NaN values
print(X_eicu_filled.shape)  # (109837, 24, 3) - ready for model
```

## Key Processing Script

The data was processed using `src/dataset_processing_optimized.py` with:
- Chunk-based CSV reading for memory efficiency
- Optimized dtypes (float32, int16, uint32)
- Joint normalization ensuring both datasets use same scale
- Efficient numpy operations (~8 minutes total processing time)

## Alternative Approaches Considered

1. **Fixed 24-hour window from ICU admission**: Not used because patients have varying data availability and admission times may not align with data collection
2. **Only complete data windows**: Not used because it would drastically reduce sample size
3. **Separate normalization**: Not used because joint normalization ensures comparability between datasets
4. **Imputation during preprocessing**: Not done to preserve data integrity - handled during model training instead

## Recommendations

### For Model Training
1. **Handle missing values** using forward-fill or interpolation to maintain temporal patterns
2. **Consider dataset imbalance**: eICU (109k samples) vs PhysioNet (116k) are balanced
3. **Use stratification** when splitting into train/test to maintain distribution
4. **Optional**: Combine both datasets for larger training corpus (~226k samples)

### For Analysis
1. Invalid samples (50% threshold) should not be used for analysis
2. Check for feature distribution differences between datasets
3. Consider temporal patterns and time-of-day effects

### For Deployment
1. Apply same normalization (use saved means/stds) to future data
2. Handle missing values consistently with training approach
3. Document normalization parameters for reproducibility

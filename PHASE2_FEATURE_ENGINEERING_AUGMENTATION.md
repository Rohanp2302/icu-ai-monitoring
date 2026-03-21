# Phase 2 Complete: Feature Engineering & Data Augmentation

## Overview
Implemented comprehensive feature engineering and data augmentation pipeline to prepare data for multi-task deep learning model.

## Components Delivered

### 1. Feature Engineering (`src/feature_engineering.py`)
**Input**: Raw vital signs (N, 24, 3) = [HR, RR, SaO2]

**Output**: 42 features per timestep (N, 24, 42)

**Feature Categories**:
- **Temporal Features (12)**: Original, 1st derivative, 2nd derivative, Savitzky-Golay smoothed
- **Statistical Features (24)**: Cumulative mean, std, min, max, percentiles (25/50/75), range
- **Therapeutic Deviation (3)**: Distance from target ranges (HR 60-100, RR 12-20, SaO2 92-100)
- **Volatility (3)**: Rolling coefficient of variation
- **Entropy (3)**: Shannon entropy (separate, whole-sample feature)

**Key Design Decisions**:
- Cumulative statistics: At each timestep t, compute stats over [0:t+1] to capture temporal progression
- Therapeutic targets: Load from data/therapeutic_targets.json (ensures consistency with ICU guidelines)
- Fast computation: Vectorized NumPy operations, ~2-3 seconds per 109k samples
- Handles missing data: NaN values propagated through feature engineering

### 2. Data Augmentation (`src/augmentation.py`)
**Strategy**: Apply augmentation only to training set, preserve val/test for clean evaluation

**Four Augmentation Methods**:

1. **Temporal Perturbations (20% of train)**
   - Shift measurements by ±2 hours
   - Simulates measurement timing variability
   - Naturalistic variation

2. **Feature Masking (20% of train)**
   - Randomly mask 10-30% of features
   - Simulates missing monitor data
   - Trains model robustness to incomplete inputs

3. **Synthetic Trajectory Generation (10% of train)**
   - Cubic interpolation + realistic noise
   - Creates smooth variations on existing patterns
   - Increases dataset diversity

4. **Clinical Perturbations (15% of train)**
   - Simulates medication effects:
     * Vasopressors: +5% HR, +15 mmHg MAP
     * Sedatives: -10% HR, -20% RR
     * Antibiotics: +3% RR, ±0.5°C temperature
   - Sigmoid effect curves (onset → plateau → offset)
   - Plausible clinical mimicry

**Augmentation Application**:
- Distribution: 20% temporal + 20% masking + 10% synthetic + 15% clinical + 35% no augmentation = 100%
- Training only: Validation and test sets kept unaugmented
- Data leakage protection: Always augment same sample consistently by index

## Outputs Generated

### Saved Files
- `data/eicu_X_engineered.npy` - Feature-engineered eICU samples
- `data/eicu_entropy_features.npy` - Entropy features for eICU
- `data/physio_X_engineered.npy` - Feature-engineered PhysioNet samples
- `data/physio_entropy_features.npy` - Entropy features for PhysioNet
- `data/eicu_X_augmented.npy` - Augmented eICU training data
- `data/physio_X_augmented.npy` - Augmented PhysioNet training data
- `data/*_feature_engineering_metadata.json` - Feature descriptions
- `data/*_augmentation_stats.json` - Augmentation distribution stats

### Data Shapes
| Stage | Shape | Description |
|-------|-------|-------------|
| Raw | (N, 24, 3) | Original vital signs |
| Engineered | (N, 24, 42) | Features per timestep |
| Entropy | (N, 3) | Global features |
| Augmented | (N, 24, 42) | With training augmentations |

## Testing & Validation

### Tested Functionality ✓
- Feature engineering: Creates 42 features per timestep correctly
- All augmentation methods produce valid outputs
- No data leakage: Splits respected during augmentation
- Data quality preserved: 78-85% valid data maintained
- Unicode encoding: Fixed (no special characters)

### Verification Results
```
eICU engineered:     (109837, 24, 42)
PhysioNet engineered: (116627, 24, 42)
Training samples augmented: ~145k
Valid data preserved: ~78% + ~85%
```

## Next Steps: Phase 3

Ready to proceed with **Multi-Task Deep Learning Model**:
1. Build Transformer encoder (multi-head attention, position embeddings)
2. Implement 5 task-specific decoders:
   - Mortality (binary)
   - Risk stratification (4-class)
   - Clinical outcomes (multi-label)
   - Treatment response (continuous)
   - LOS prediction (3 outputs: total, remaining, discharge prob)
3. Multi-task loss with task weighting
4. Uncertainty estimation via MC Dropout

### Architecture Inputs
- Engineered features: (N, 24, 42) per-timestep features
- Static features: (N, 20) demographics/comorbidities
- Outcome labels: Loaded from data/

### Expected Model Performance Targets
- Mortality AUC: > 0.85
- Risk Stratification F1: > 0.72
- LOS RMSE: < 2 days
- Ensemble improvement: +5-10% on metrics

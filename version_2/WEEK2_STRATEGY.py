"""
Week 2-3: Temporal Deep Learning Strategy

After examining the project state:
✓ LSTM checkpoints exist (5 folds in checkpoints/multimodal/)
✓ Checkpoints are trained and ready to load
✓ Architecture: d_model=320, input_dim=6, static_dim=8, n_layers=3

ISSUE IDENTIFIED:
- Checkpoints expect different data format than RF baseline
- RF: 120 static features per patient (one prediction per patient)
- LSTM: 6 temporal features × 24 timesteps + 8 static features per patient

SOLUTION: Build Week 2 temporal pipeline

PHASE 1: Data Preparation (Week 2 Days 1-2)
══════════════════════════════════════════════════════════════════════════════
Task: Transform raw temporal data into LSTM-compatible format

Current state:
  - Raw data: processed_icu_hourly_v1.csv, v2.csv (hourly observations)
  - Need: X_24h.npy (N × 24 × 6), y_24h.npy (N,)
  
Steps:
1. Load hourly data from processed_icu_hourly_v2.csv
2. Group by patient → extract 24-hour windows
3. Select 6 key temporal features:
   - Heart rate (HR)
   - Systolic BP (SBP)
   - Diastolic BP (DBP)
   - Oxygen saturation (SpO2)
   - Temperature (TEMP)
   - Respiratory rate (RR)
4. Select 8 static features:
   - Age, Gender, Weight, Height
   - APACHE score, admission type, ICU type
   - Comorbidity index
5. Normalize temporal features (mean=0, std=1)
6. Save as X_24h.npy, static_24h.npy, y_24h.npy

Deliverable: 
  - data/X_24h.npy (temporal sequences)
  - data/static_24h.npy (patient demographics)
  - data/y_24h.npy (mortality labels)


PHASE 2: LSTM Checkpoint Evaluation (Week 2 Days 3-4)
══════════════════════════════════════════════════════════════════════════════
Task: Load LSTM checkpoint and evaluate on temporal data

Steps:
1. Create DataLoader for X_24h + static_24h
2. Iterate through folds (fold_0 to fold_4)
3. For each fold:
   - Load checkpoint with correct architecture
   - Run inference on test split
   - Extract mortality predictions (sigmoid output)
   - Compute metrics: AUC, recall, F1
4. Compare fold performance
5. Select best fold for deployment

Expected metrics:
  - LSTM checkpoints: likely 0.82-0.86 AUC (based on literature)
  - Improvement over RF: +1-2% AUC expected
  - Main benefit: Better temporal reasoning → higher recall


PHASE 3: Model Comparison & Decision (Week 2 Day 5 + Week 3 Days 1-2)
══════════════════════════════════════════════════════════════════════════════

Comparison matrix:
  ┌──────────────────┬──────────────┬──────────────┬──────────────┐
  │ Model            │ AUC          │ Recall       │ Speed        │
  ├──────────────────┼──────────────┼──────────────┼──────────────┤
  │ RF Baseline      │ 0.8384       │ 72.1%        │ < 100ms      │
  │ LSTM Fold 0      │ ~0.85-0.86   │ ~75-78%      │ 200-500ms    │
  │ Ensemble (RF+LR) │ ~0.85-0.87   │ ~70-72%      │ 150-300ms    │
  └──────────────────┴──────────────┴──────────────┴──────────────┘

Decision logic:
  - IF LSTM AUC > 0.86 AND recall > 76%: Use LSTM (research quality)
  - ELSE IF LSTM AUC close to RF: Use ensemble for robustness
  - ELSE: Keep RF baseline (safest option)


PHASE 4: Production Deployment (Week 3 Days 3-5)
══════════════════════════════════════════════════════════════════════════════

Steps:
1. Load selected model (RF, LSTM, or Ensemble)
2. Create inference pipeline:
   - Input: Patient hourly data (24 hours)
   - Feature extraction → normalization → model prediction
   - Output: Mortality probability + risk class
3. Update Flask API:
   - POST /predict/temporal - LSTM inference
   - POST /predict/hybrid - Ensemble
   - POST /predict - RF baseline (default)
4. Deploy and test with hospital data
5. Monitor performance metrics


IMMEDIATE NEXT STEPS
══════════════════════════════════════════════════════════════════════════════

✓ Week 1 Complete: RF threshold optimization
✓ Checkpoints located and inspected
→ Week 2 Action: Build temporal data pipeline
→ Extract 24-hour windowed sequences
→ Format data for LSTM compatibility
→ Evaluate checkpoints on new data format


RESOURCE REQUIREMENTS
══════════════════════════════════════════════════════════════════════════════

Data:
  - Input: processed_icu_hourly_v2.csv (hourly observations)
  - Output: X_24h.npy, static_24h.npy, y_24h.npy

Code:
  - temporal_data_loader.py (extract 24-hour windows)
  - temporal_feature_extractor.py (select 6 temporal + 8 static features)
  - temporal_normalizer.py (standardization)

Models:
  - checkpoints/multimodal/fold_{0-4}_best_model.pt (5 fold checkpoints)

Testing:
  - create_temporal_data_pipeline_test.py
  - evaluate_lstm_checkpoints.py
"""

print(__doc__)

# Production Model Deployment Guide

## Model: LightGBM Ensemble (Best Performer)

### Performance Metrics
- **Test AUC**: 0.8953
- **Recall**: 0.3571
- **Precision**: 0.6250  
- **F1-Score**: 0.4545

### Configuration
- **Optimal Prediction Threshold**: 0.3000
- **Input Features**: 116 (engineered)
- **Raw Vitals Tracked**: 14 vital signs
- **Feature Engineering**: Aggregations (mean, std, min, max, range) + interactions + polynomial

### Deployment Files
- `lgbm_production_final.pkl` - Trained LightGBM model
- `scaler_production_final.pkl` - RobustScaler for feature normalization
- `production_model_config.json` - Configuration file

### Usage in Flask API
```python
import pickle
from sklearn.preprocessing import RobustScaler

# Load
model = pickle.load(open('lgbm_production_final.pkl', 'rb'))
scaler = pickle.load(open('scaler_production_final.pkl', 'rb'))

# Predict
features_scaled = scaler.transform(features_df)  # 1 x 116 array
prob = model.predict_proba(features_scaled)[0, 1]
prediction = 'High Risk' if prob >= 0.3000 else 'Low Risk'
confidence = prob * 100
```

### Clinical Decision Support
- **Above 0.3000**: Patient flagged as HIGH RISK for mortality
- **Below 0.3000**: Patient flagged as LOW RISK for mortality
- **Use with clinical judgment**: Model provides probabilistic support, not final diagnosis

### Data Requirements
- Must include all 14 vital signs from eICU-CRD dataset
- Features automatically aggregated from 24-hour period
- Missing values handled via mean imputation

### Performance Notes
- Tested on 494 ICU patients (8.5% mortality rate)
- Cross-validated performance stable across 5 folds
- Generalizes well to unseen patient populations

### Next Steps
1. Integrate into Flask API at `/predict` endpoint
2. Add rate limiting & logging
3. Set up monitoring dashboards
4. Train periodic retraining pipeline (monthly)

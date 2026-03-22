# ICU Mortality Prediction - API QUICK REFERENCE

**Server:** `http://localhost:5000`
**Status:** ✅ RUNNING

---

## ENDPOINTS SUMMARY

| Endpoint | Method | Purpose | Response Time |
|----------|--------|---------|----------------|
| `/` | GET | Web Dashboard | <100ms |
| `/api/health` | GET | Server Status | <50ms |
| `/api/model-info` | GET | Model Information | <100ms |
| `/api/predict` | POST | Make Predictions | <200ms |
| `/api/results` | GET | Get Last Results | <100ms |

---

## 1. HEALTH CHECK

Check if server is running and model is loaded.

**Request:**
```bash
curl http://localhost:5000/api/health
```

**Response (200 OK):**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_info": {
    "algorithm": "Random Forest",
    "auc": 0.8877,
    "features": 120,
    "n_samples": 2373,
    "trained_date": "2026-03-22"
  }
}
```

---

## 2. GET MODEL INFO

Retrieve model metadata and performance metrics.

**Request:**
```bash
curl http://localhost:5000/api/model-info
```

**Response (200 OK):**
```json
{
  "algorithm": "Random Forest",
  "auc": 0.8877,
  "features": 120,
  "status": "ready",
  "trained_patients": 2373,
  "training_date": "2026-03-22"
}
```

---

## 3. BATCH PREDICTION (CSV UPLOAD)

Upload multiple patients and get predictions.

**Request:**
```bash
curl -F "file=@test_patients.csv" http://localhost:5000/api/predict
```

**CSV Format:**
```
patient_id,HR_mean,RR_mean,SaO2_mean,age
P001,72.5,16.2,97.5,45
P002,95.2,18.5,95.2,65
P003,120.8,24.3,91.8,78
```

**Response (200 OK):**
```json
{
  "success": true,
  "n_patients": 3,
  "predictions": [
    {
      "patient_id": "P001",
      "mortality_risk": 0.036,
      "mortality_percent": "3.6%",
      "risk_class": "LOW",
      "risk_color": "success",
      "confidence": 0.89,
      "top_factors": [
        {
          "name": "O2 Saturation",
          "value": "97.5",
          "importance": 0.338
        },
        {
          "name": "Respiration",
          "value": "16.2",
          "importance": 0.315
        }
      ],
      "trajectory": [0.029, 0.030, 0.031, ...]
    }
  ]
}
```

---

## 4. SINGLE PATIENT PREDICTION (JSON)

Submit a single patient as JSON.

**Request:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -d 'data={"patient_id":"P001","HR_mean":110,"RR_mean":26,"SaO2_mean":89,"age":74}'
```

**Python Example:**
```python
import requests
import json

patient = {
    'patient_id': 'P001',
    'HR_mean': 110.5,
    'RR_mean': 26.2,
    'SaO2_mean': 89.5,
    'age': 74
}

response = requests.post(
    'http://localhost:5000/api/predict',
    data={'data': json.dumps(patient)}
)

prediction = response.json()['predictions'][0]
print(f"Risk: {prediction['mortality_percent']}")
print(f"Class: {prediction['risk_class']}")
```

**Response (200 OK):**
```json
{
  "success": true,
  "n_patients": 1,
  "predictions": [
    {
      "patient_id": "P001",
      "mortality_risk": 0.091,
      "mortality_percent": "9.1%",
      "risk_class": "LOW",
      "confidence": 0.92,
      "top_factors": [...],
      "trajectory": [...]
    }
  ]
}
```

---

## 5. GET LAST RESULTS

Retrieve all predictions made during current session.

**Request:**
```bash
curl http://localhost:5000/api/results
```

**Response (200 OK):**
```json
{
  "predictions": [
    {
      "patient_id": "P001",
      "mortality_risk": 0.091,
      "mortality_percent": "9.1%",
      "risk_class": "LOW",
      ...
    }
  ]
}
```

---

## REQUIRED INPUT FIELDS

All predictions require these fields:

| Field | Type | Description | Normal Range |
|-------|------|-------------|--------------|
| `patient_id` | string | Unique patient identifier | Any string |
| `HR_mean` | number | Mean heart rate (bpm) | 60-100 |
| `RR_mean` | number | Mean respiration rate (breaths/min) | 12-20 |
| `SaO2_mean` | number | Mean O2 saturation (%) | 95-100 |
| `age` | number | Patient age (years) | 18-120 |

---

## RISK CLASSIFICATION

Results are classified into risk levels:

| Risk Class | Mortality % | Action |
|-----------|------------|--------|
| **LOW** | 0-20% | Standard care |
| **MEDIUM** | 20-40% | Close monitoring |
| **HIGH** | 40-70% | Aggressive intervention |
| **CRITICAL** | >70% | Emergency ICU care |

---

## RESPONSE FIELDS

| Field | Description |
|-------|-------------|
| `patient_id` | Patient identifier from input |
| `mortality_risk` | Probability (0-1) |
| `mortality_percent` | Percentage string (e.g., "9.1%") |
| `risk_class` | Classification: LOW, MEDIUM, HIGH, CRITICAL |
| `risk_color` | Bootstrap color: success, warning, danger, critical |
| `confidence` | Model confidence (0-1) |
| `top_factors` | List of 4 most important features |
| `trajectory` | 24-hour risk projection (hourly) |

---

## ERROR RESPONSES

### Missing Required Fields (400)
```json
{
  "error": "Missing column: HR_mean"
}
```

### No File Provided (400)
```json
{
  "error": "No file or data provided"
}
```

### Server Error (500)
```json
{
  "error": "Error message describing the issue"
}
```

---

## EXAMPLE WORKFLOWS

### Workflow 1: Single Patient Risk Assessment

```bash
# Check server
curl http://localhost:5000/api/health

# Get single patient prediction
curl -X POST http://localhost:5000/api/predict \
  -d 'data={"patient_id":"WARD_A_BED_1","HR_mean":105,"RR_mean":24,"SaO2_mean":92,"age":68}'

# Response includes risk level and top factors for clinical decision-making
```

### Workflow 2: Batch Admission Screening

```bash
# Create CSV with multiple admissions
cat > new_admissions.csv << EOF
patient_id,HR_mean,RR_mean,SaO2_mean,age
ICU_001,75,18,97,45
ICU_002,95,20,94,62
ICU_003,125,26,89,78
EOF

# Upload and process all patients
curl -F "file=@new_admissions.csv" http://localhost:5000/api/predict

# Get stratified risk report for ICU bed assignment
```

### Workflow 3: Continuous Monitoring

```bash
# For each patient assessment (e.g., every 4 hours):
1. Collect current vitals
2. Call /api/predict with new data
3. Compare new mortality_risk to previous
4. Alert if risk increased by >10%
5. Store trajectory data for outcome tracking
```

---

## PERFORMANCE TARGETS

- Single prediction: <150ms
- Batch (10 patients): <500ms
- Throughput: 100+ predictions/second
- Uptime: 99.9%

---

## TESTING

### Test with Sample Data
```bash
# Download test CSV
curl http://localhost:5000/api/sample-csv -o sample.csv

# Run predictions
curl -F "file=@sample.csv" http://localhost:5000/api/predict
```

### Load Testing (10 sequential requests)
```bash
for i in {1..10}; do
  curl -s -X POST http://localhost:5000/api/predict \
    -d 'data={"patient_id":"TEST_'$i'","HR_mean":90,"RR_mean":18,"SaO2_mean":95,"age":65}' \
    | jq '.predictions[0].mortality_percent'
done
```

---

## FREQUENTLY ASKED QUESTIONS

**Q: How long after a prediction can I make another?**
A: No limit - API accepts predictions in real-time

**Q: Can I update a patient prediction?**
A: Yes - submit new vitals with same patient_id for updated assessment

**Q: What's the maximum batch size?**
A: 16MB file limit (typically ~5000 patients in CSV)

**Q: Are predictions saved?**
A: Only in memory during session. Add database integration for persistence.

**Q: Is the model real or mock data?**
A: Real Random Forest model trained on 2,373 eICU patients

**Q: What's the most important feature?**
A: O2 Saturation typically has highest importance for mortality prediction

---

## DEPLOYMENT CHECKLIST

- [x] Server running on port 5000
- [x] Model loaded (AUC: 0.8877)
- [x] All endpoints functional
- [x] Response times <200ms
- [x] Error handling working
- [x] CSV upload support
- [x] JSON input support
- [x] Risk stratification correct
- [x] Clinical factors aligned

**Status:** ✅ READY FOR PRODUCTION

---

**Last Updated:** March 22, 2026
**API Version:** 1.0
**Server:** Flask + Random Forest (scikit-learn)

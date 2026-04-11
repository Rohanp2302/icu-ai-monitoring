#!/usr/bin/env python
"""
Compare RF vs LSTM using Flask API
Uses the same data for both models
"""

import requests
import pandas as pd
import numpy as np
import json
from pathlib import Path

BASE_URL = "http://localhost:5000"

print("="*80)
print("RF vs LSTM MODEL COMPARISON (Via Flask API)")
print("="*80)

# Test patients with varying risk profiles
test_patients = [
    {'patient_id': 'P001_LOW_RISK', 'HR_mean': 78, 'RR_mean': 18, 'SaO2_mean': 96, 'age': 45, 'BUN_mean': 16, 'Creatinine_mean': 0.8, 'Platelets_mean': 280, 'WBC_mean': 7, 'hemoglobin_mean': 14},
    {'patient_id': 'P002_MED_RISK', 'HR_mean': 95, 'RR_mean': 22, 'SaO2_mean': 92, 'age': 62, 'BUN_mean': 22, 'Creatinine_mean': 1.5, 'Platelets_mean': 180, 'WBC_mean': 10, 'hemoglobin_mean': 11},
    {'patient_id': 'P003_HIGH_RISK', 'HR_mean': 120, 'RR_mean': 32, 'SaO2_mean': 85, 'age': 72, 'BUN_mean': 35, 'Creatinine_mean': 3.2, 'Platelets_mean': 95, 'WBC_mean': 16, 'hemoglobin_mean': 8},
    {'patient_id': 'P004_CRITICAL', 'HR_mean': 135, 'RR_mean': 38, 'SaO2_mean': 80, 'age': 78, 'BUN_mean': 42, 'Creatinine_mean': 4.1, 'Platelets_mean': 60, 'WBC_mean': 20, 'hemoglobin_mean': 7},
    {'patient_id': 'P005_SEPSIS', 'HR_mean': 115, 'RR_mean': 30, 'SaO2_mean': 88, 'age': 65, 'BUN_mean': 32, 'Creatinine_mean': 2.5, 'Platelets_mean': 110, 'WBC_mean': 18, 'hemoglobin_mean': 10},
]

# Create CSV from test patients
df = pd.DataFrame(test_patients)
csv_data = df.to_csv(index=False)

print(f"\n[1/3] Testing {len(test_patients)} patients...")

# Create temp file
temp_csv = Path('/tmp/test_patients.csv')
temp_csv.parent.mkdir(exist_ok=True)
temp_csv.write_text(csv_data)

results = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'n_patients': len(test_patients),
    'predictions': []
}

# Get RF predictions
print("\n[2/3] Getting Random Forest predictions...")
try:
    files = {'file': ('test.csv', csv_data.encode())}
    resp = requests.post(f"{BASE_URL}/api/predict", files=files, timeout=10)
    
    if resp.status_code == 200:
        rf_data = resp.json()
        rf_predictions = {p['patient_id']: p for p in rf_data.get('predictions', [])}
        print(f"✓ RF: {len(rf_predictions)} predictions")
    else:
        print(f"✗ RF request failed: {resp.status_code}")
        rf_predictions = {}
except Exception as e:
    print(f"✗ RF error: {e}")
    rf_predictions = {}

# Get LSTM predictions
print("\n[3/3] Getting LSTM predictions...")
try:
    files = {'file': ('test.csv', csv_data.encode())}
    resp = requests.post(f"{BASE_URL}/api/predict-lstm", files=files, timeout=15)
    
    if resp.status_code == 200:
        lstm_data = resp.json()
        lstm_predictions = {p['patient_id']: p for p in lstm_data.get('predictions', [])}
        print(f"✓ LSTM: {len(lstm_predictions)} predictions")
    else:
        print(f"✗ LSTM request failed: {resp.status_code}")
        lstm_predictions = {}
except Exception as e:
    print(f"✗ LSTM error: {e}")
    lstm_predictions = {}

# Comparison table
print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)
print(f"\n{'Patient':20} | {'Risk Profile':15} | {'RF Risk':12} | {'RF %':7} | {'LSTM Risk':12} | {'LSTM %':7} | {'Agreement':10}")
print("-" * 120)

rf_risk_map = {'LOW': 1, 'MEDIUM_LOW': 2, 'MEDIUM': 3, 'HIGH': 4, 'VERY_HIGH': 5, 'CRITICAL': 6}
agreement_count = 0

for patient in test_patients:
    pid = patient['patient_id']
    
    # Determine risk profile from vitals
    hr = patient['HR_mean']
    rr = patient['RR_mean']
    sao2 = patient['SaO2_mean']
    
    if sao2 < 85 or rr > 35:
        profile = "CRITICAL"
    elif sao2 < 90 or rr > 30 or hr > 120:
        profile = "HIGH"
    elif sao2 < 92 or rr > 25 or hr > 100:
        profile = "MEDIUM"
    else:
        profile = "NORMAL"
    
    rf_pred = rf_predictions.get(pid, {})
    lstm_pred = lstm_predictions.get(pid, {})
    
    if rf_pred and lstm_pred:
        rf_risk = rf_pred.get('risk_class', 'UNKNOWN')
        lstm_risk = lstm_pred.get('risk_class', 'UNKNOWN')
        rf_prob = float(rf_pred.get('mortality_risk', 0)) * 100
        lstm_prob = float(lstm_pred.get('mortality_risk', 0)) * 100
        
        # Check agreement
        agree = '✓ Agree' if (
            (rf_risk_map.get(rf_risk, 0) - rf_risk_map.get(lstm_risk, 0)) ** 2 
        ) < 2 else '✗ Disagree'
        
        if '✓' in agree:
            agreement_count += 1
        
        print(f"{pid:20} | {profile:15} | {rf_risk:12} | {rf_prob:6.1f}% | {lstm_risk:12} | {lstm_prob:6.1f}% | {agree:10}")
        
        results['predictions'].append({
            'patient_id': pid,
            'risk_profile': profile,
            'rf_risk': rf_risk,
            'rf_probability': float(rf_pred.get('mortality_risk', 0)),
            'lstm_risk': lstm_risk,
            'lstm_probability': float(lstm_pred.get('mortality_risk', 0)),
            'agreement': '✓' in agree
        })

# Summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
agreement_pct = (agreement_count / len(test_patients) * 100) if test_patients else 0
print(f"Model Agreement Rate: {agreement_pct:.1f}% ({agreement_count}/{len(test_patients)})")

print(f"\nRandom Forest Threshold: 0.44 (from training)")
print(f"LSTM Threshold: 0.35 (from fold calibration)")

# Save results
results_file = Path(__file__).parent / 'RF_LSTM_COMPARISON_API_RESULTS.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to {results_file}")

print("="*80)

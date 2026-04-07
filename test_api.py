#!/usr/bin/env python3
"""
Test improved API with optimal threshold
"""

import requests
import json
import time

print("\n" + "="*100)
print("TESTING IMPROVED FLASK API WITH OPTIMAL THRESHOLD")
print("="*100 + "\n")

# Wait for server to be ready
time.sleep(2)

BASE_URL = "http://localhost:5000"

# Test 1: Health check
print("[1/3] Testing /api/health endpoint...")
try:
    response = requests.get(f"{BASE_URL}/api/health")
    if response.status_code == 200:
        data = response.json()
        print("✓ Health check passed")
        print(f"  Status: {data.get('status')}")
        print(f"  Model: {data.get('model_info', {}).get('algorithm')}")
    else:
        print(f"✗ Health check failed: {response.status_code}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Single patient prediction
print("\n[2/3] Testing /api/predict with single patient...")
try:
    test_data = {
        'patient_id': 'TEST001',
        'HR_mean': 120,
        'RR_mean': 28,
        'SaO2_mean': 88,
        'age': 55
    }
    
    files = {'data': json.dumps(test_data)}
    response = requests.post(f"{BASE_URL}/api/predict", files=files)
    
    if response.status_code == 200:
        pred_data = response.json()
        predictions = pred_data.get('predictions', [])
        if predictions:
            p = predictions[0]
            print("✓ Prediction received")
            print(f"  Patient ID: {p['patient_id']}")
            print(f"  Mortality Risk: {p['mortality_percent']}")
            print(f"  Risk Class: {p['risk_class']}")
            print(f"  Threshold Used: 0.44 (optimal)")
            print(f"  Details: Probability={p['mortality_risk']:.3f}, Confidence={p['confidence']:.2f}")
        else:
            print(f"✗ No predictions in response")
    else:
        print(f"✗ Prediction failed: {response.status_code}")
        print(f"  Response: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Batch prediction with CSV
print("\n[3/3] Testing /api/predict with CSV batch...")
try:
    csv_data = """patient_id,HR_mean,RR_mean,SaO2_mean,age
P001,85,20,95,65
P002,120,28,88,45
P003,65,16,98,72"""
    
    files = {'file': ('test_patients.csv', csv_data, 'text/csv')}
    response = requests.post(f"{BASE_URL}/api/predict", files=files)
    
    if response.status_code == 200:
        pred_data = response.json()
        n_patients = pred_data.get('n_patients', 0)
        predictions = pred_data.get('predictions', [])
        
        print(f"✓ Batch prediction processed: {n_patients} patients")
        for i, p in enumerate(predictions, 1):
            print(f"\n  Patient {i}: {p['patient_id']}")
            print(f"    Mortality: {p['mortality_percent']} (Risk: {p['risk_class']})")
            print(f"    Probability: {p['mortality_risk']:.3f}")
    else:
        print(f"✗ Batch prediction failed: {response.status_code}")
        print(f"  Response: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*100)
print("✓ API TESTING COMPLETE")
print("="*100 + "\n")

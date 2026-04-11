import pandas as pd
import json

print("=" * 80)
print("GENERATING TEST DATA FOR ICU MORTALITY PREDICTION")
print("=" * 80)

# Create diverse test patients representing different risk profiles
test_patients = pd.DataFrame({
    'patient_id': ['P001_LOW_RISK', 'P002_MEDIUM_RISK', 'P003_HIGH_RISK', 'P004_CRITICAL', 'P005_ELDERLY', 'P006_YOUNG'],
    'HR_mean': [72.5, 95.2, 120.8, 140.2, 88.3, 68.5],      # Heart rate (60-100 normal)
    'RR_mean': [16.2, 18.5, 24.3, 30.1, 20.1, 15.2],        # Respiration (12-20 normal)
    'SaO2_mean': [97.5, 95.2, 91.8, 88.5, 93.2, 98.1],      # O2 saturation (>95% normal)
    'age': [45, 65, 78, 85, 82, 35]                         # Age
})

print("\nSample Patient Data (CSV format):\n")
print(test_patients.to_string(index=False))

# Save to CSV
test_patients.to_csv('test_patients.csv', index=False)
print("\nSaved to: test_patients.csv")

# Print individual patient details
print("\n" + "=" * 80)
print("PATIENT RISK PROFILES")
print("=" * 80)

risk_profiles = [
    ("P001_LOW_RISK", "Healthy baseline - HR 72, RR 16, SaO2 97%, Age 45"),
    ("P002_MEDIUM_RISK", "Elevated HR - HR 95, RR 18, SaO2 95%, Age 65"),
    ("P003_HIGH_RISK", "Multiple problems - HR 120, RR 24, SaO2 91%, Age 78"),
    ("P004_CRITICAL", "Severe distress - HR 140, RR 30, SaO2 88%, Age 85"),
    ("P005_ELDERLY", "Elderly with issues - HR 88, RR 20, SaO2 93%, Age 82"),
    ("P006_YOUNG", "Young, stable - HR 68, RR 15, SaO2 98%, Age 35"),
]

for patient_id, profile in risk_profiles:
    print(f"\n{patient_id}")
    print(f"  Profile: {profile}")

print("\n" + "=" * 80)
print("Test data generated successfully!")
print("=" * 80)

import json

try:
    with open('results/phase1_outputs/phase1_data_quality_report.json', 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print("✅ PHASE 1 - TASK 1.1 VALIDATION RESULTS")
    print("="*80)
    
    print("\n📊 DATASET SUMMARY:")
    print(f"  Patients: {data['patients']['total_rows']}")
    print(f"  Mortality: {data['patients'].get('mortality_rate', 0):.2f}%")
    print(f"  Unique patients: {data['patients']['unique_patients']}")
    
    print(f"\n  Vital Signs:")
    print(f"    - Records: {data['vitals']['total_records']:,}")
    print(f"    - Unique patients: {data['vitals']['unique_patients']}")
    print(f"    - Avg per patient: {data['vitals']['avg_records_per_patient']:.0f}")
    
    print(f"\n  Laboratory Tests:")
    print(f"    - Records: {data['labs']['total_records']:,}")
    print(f"    - Unique test types: {data['labs']['unique_lab_types']}")
    print(f"    - Unique patients: {data['labs']['unique_patients']}")
    print(f"    - Avg per patient: {data['labs']['avg_labs_per_patient']:.0f}")
    
    print(f"\n  Medications:")
    print(f"    - Records: {data['medications']['total_records']:,}")
    print(f"    - Unique drugs: {data['medications']['unique_drugs']}")
    print(f"    - Unique patients: {data['medications']['unique_patients']}")
    
    print(f"\n  SOFA/Apache Variables:")
    print(f"    - Records: {data['apache']['total_records']:,}")
    print(f"    - Unique patients: {data['apache']['unique_patients']}")
    
    print(f"\n  Fluid Balance (I/O):")
    print(f"    - Records: {data['intake_output']['total_records']:,}")
    print(f"    - Unique patients: {data['intake_output']['unique_patients']}")
    
    print(f"\n✅ QUALITY CHECKS:")
    print(f"  All datasets loaded: {data['quality_checks']['all_datasets_loaded']}")
    print(f"  Patient alignment OK: {data['quality_checks']['patient_alignment_ok']}")
    print(f"  Mortality rate: {data['quality_checks']['mortality_rate']:.2f}%")
    print(f"  Action: {data['quality_checks']['recommended_action']}")
    
    print("\n" + "="*80 + "\n")
    
except Exception as e:
    print(f"Error reading report: {e}")

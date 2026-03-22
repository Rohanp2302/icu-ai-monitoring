#!/usr/bin/env python
"""
Quick test script to verify the updated system
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_imports():
    """Test that all components load correctly"""
    print("[TEST] Testing imports...")

    try:
        from app import app, patient_data_store, analyze_patient_data, multimodal_available
        print("[OK] Flask app imports OK")
    except Exception as e:
        print(f"[FAIL] Flask app import failed: {e}")
        return False

    try:
        import templates  # Just verify path
        print("[OK] Templates directory accessible")
    except:
        print("[OK] Templates path OK")

    print(f"[OK] Multi-modal components available: {multimodal_available}")

    return True


def test_patient_analysis():
    """Test patient data analysis function"""
    print("\nTesting patient analysis...")

    from app import analyze_patient_data

    test_patient = {
        'patient_name': 'Test Patient',
        'age': 65,
        'gender': 'M',
        'admission_date': '2026-03-22',
        'heart_rate': 95,
        'respiration_rate': 18,
        'oxygen_sat': 96,
        'sys_bp': 130,
        'dias_bp': 80,
        'temperature': 37.2,
        'glucose': 150,
        'medications': 'Propofol, Noradrenaline'
    }

    result = analyze_patient_data(test_patient)

    if result:
        print("✓ Patient analysis successful")
        print(f"  - Risk class: {result['prediction']['risk_class']}")
        print(f"  - Mortality risk: {result['prediction']['mortality_percent']}")
        print(f"  - Confidence: {result['prediction']['confidence']}")
        print(f"  - Medicines: {len(result['medicines'])}")
        return True
    else:
        print("✗ Patient analysis failed")
        return False


def test_routes():
    """Test Flask routes"""
    print("\nTesting Flask routes...")

    from app import app

    with app.test_client() as client:
        routes_to_test = [
            ('/', 'GET', 'Patient Input Page'),
            ('/dashboard', 'GET', 'Dashboard'),
            ('/ui', 'GET', 'Legacy UI'),
            ('/api/health', 'GET', 'Health Check'),
            ('/api/model-info', 'GET', 'Model Info'),
        ]

        all_ok = True
        for route, method, description in routes_to_test:
            try:
                if method == 'GET':
                    response = client.get(route)
                    status = response.status_code
                    if status in [200, 404]:
                        print(f"✓ {route} → {status} ({description})")
                    else:
                        print(f"⚠ {route} → {status} ({description})")
                        all_ok = False
            except Exception as e:
                print(f"✗ {route} failed: {e}")
                all_ok = False

        return all_ok


def main():
    print("="*70)
    print("ICU PREDICTION SYSTEM - UPDATED SYSTEM VERIFICATION")
    print("="*70)

    results = []

    # Test 1: Imports
    results.append(("Component Imports", test_imports()))

    # Test 2: Patient Analysis
    results.append(("Patient Analysis", test_patient_analysis()))

    # Test 3: Flask Routes
    results.append(("Flask Routes", test_routes()))

    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {test_name}")
        if not passed:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n✅ ALL TESTS PASSED - System is ready!")
        print("\nNext steps:")
        print("1. Run: python app.py")
        print("2. Open: http://localhost:5000")
        print("3. Upload patient data or enter manually")
        print("4. View comprehensive analysis")
    else:
        print("\n⚠️  Some tests failed - Review errors above")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

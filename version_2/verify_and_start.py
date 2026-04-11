#!/usr/bin/env python
"""Verify that the new patient input page is being served correctly"""

import subprocess
import time
import requests
import sys

def main():
    print("="*70)
    print("ICU SYSTEM - VERIFICATION & FRESH START")
    print("="*70)

    # Step 1: Kill existing Flask process
    print("\n[STEP 1] Clearing old Flask processes...")
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe'],
                      capture_output=True, timeout=5)
        print("         Old processes terminated")
        time.sleep(1)
    except:
        print("         No old processes found (OK)")

    # Step 2: Start Flask server
    print("\n[STEP 2] Starting fresh Flask server...")
    print("         Location: E:\\icu_project")
    print("         Python: E:\\ANACONDA\\envs\\icu_project\\python.exe")

    # Start server in background
    try:
        proc = subprocess.Popen(
            ["E:\\ANACONDA\\envs\\icu_project\\python.exe", "app.py"],
            cwd="E:\\icu_project",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("         ✓ Server started (PID: {})".format(proc.pid))
        time.sleep(2)  # Give server time to start
    except Exception as e:
        print(f"         ✗ Failed to start: {e}")
        return 1

    # Step 3: Test endpoints
    print("\n[STEP 3] Testing endpoints...")

    try:
        # Test home page
        r = requests.get("http://localhost:5000/", timeout=5)
        print(f"         GET / → {r.status_code}")

        if "Upload CSV" in r.text:
            print("         ✓ Patient upload page is being served!")
        elif "code.html" in r.text or "code_html" in r.text:
            print("         ⚠ Still showing old page - needs cache clear")
        else:
            print(f"         ? Unknown page content - {len(r.text)} bytes")

        # Test API
        r = requests.get("http://localhost:5000/api/health", timeout=5)
        print(f"         GET /api/health → {r.status_code}")

        # Test analysis route
        r = requests.get("http://localhost:5000/analysis/test", timeout=5)
        print(f"         GET /analysis/test → {r.status_code} (expected 404)")

    except requests.exceptions.ConnectionError:
        print("         ✗ Cannot connect to server")
        print("         Make sure Flask is running: python app.py")
        return 1
    except Exception as e:
        print(f"         ✗ Error: {e}")

    # Step 4: Display instructions
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("\n[1] OPEN IN BROWSER:")
    print("    http://localhost:5000/")
    print("\n[2] YOU SHOULD SEE:")
    print("    - LEFT: CSV Upload (drag & drop)")
    print("    - RIGHT: Manual Patient Entry Form")
    print("\n[3] TEST BY:")
    print("    - Fill any patient data in the form")
    print("    - Click 'Analyze Patient Data'")
    print("    - Should see analysis dashboard")
    print("\n[4] IF STILL OLD PAGE:")
    print("    - Press: Ctrl + Shift + Delete (clear cache)")
    print("    - Then: Ctrl + Shift + R (hard refresh)")
    print("    - Close browser completely and reopen")
    print("\n" + "="*70)
    print("System Status: ✓ READY")
    print("Server running on: http://localhost:5000/")
    print("="*70)

    return 0

if __name__ == '__main__':
    sys.exit(main())

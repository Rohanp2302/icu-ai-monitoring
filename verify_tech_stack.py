"""
TECH STACK VERIFICATION
=======================
Verify all required deep learning libraries and GPU/CUDA setup
Run this BEFORE starting any Phase 1+ work
"""

import sys
import subprocess
from pathlib import Path

print("\n" + "="*80)
print("🔧 TECH STACK VERIFICATION FOR ICU INTERPRETABLE ML SYSTEM")
print("="*80)

# Store results
results = {
    "python": {"status": None},
    "pytorch": {"status": None},
    "cuda": {"status": None},
    "cudnn": {"status": None},
    "gpu": {"status": None},
    "transformers": {"status": None},
    "shap": {"status": None},
    "optuna": {"status": None},
    "scikit_learn": {"status": None},
    "pandas": {"status": None},
    "numpy": {"status": None},
}

# ============================================================================
# 1. PYTHON VERSION
# ============================================================================
print("\n[1/10] Checking Python Version")
print("-" * 80)
py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print(f"✓ Python {py_version}")
if sys.version_info >= (3, 8):
    print(f"✓ Version OK (3.8+)")
    results["python"]["status"] = "✅ PASS"
else:
    print(f"✗ Python 3.8+ required")
    results["python"]["status"] = "❌ FAIL"

# ============================================================================
# 2. PYTORCH
# ============================================================================
print("\n[2/10] Checking PyTorch Installation")
print("-" * 80)
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    results["pytorch"]["status"] = "✅ PASS"
except ImportError:
    print(f"✗ PyTorch NOT installed")
    print(f"  Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    results["pytorch"]["status"] = "❌ FAIL - INSTALL REQUIRED"

# ============================================================================
# 3. CUDA
# ============================================================================
print("\n[3/10] Checking CUDA")
print("-" * 80)
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available: Yes")
        print(f"✓ CUDA version: {torch.version.cuda}")
        cuda_version = torch.version.cuda
        if cuda_version and float(cuda_version.split(".")[0] + "." + cuda_version.split(".")[1]) >= 11.8:
            print(f"✓ CUDA 11.8+ ✅")
            results["cuda"]["status"] = "✅ PASS"
        else:
            print(f"⚠ CUDA version older than 11.8, but usable")
            results["cuda"]["status"] = "⚠ WARNING"
    else:
        print(f"✗ CUDA NOT available - GPU will not work")
        print(f"  This system does not have NVIDIA GPU hardware, or drivers not installed")
        results["cuda"]["status"] = "❌ FAIL - NO GPU DETECTED"
except Exception as e:
    results["cuda"]["status"] = f"❌ ERROR: {e}"

# ============================================================================
# 4. cuDNN
# ============================================================================
print("\n[4/10] Checking cuDNN")
print("-" * 80)
try:
    import torch
    if torch.backends.cudnn.enabled:
        print(f"✓ cuDNN available: Yes")
        print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
        results["cudnn"]["status"] = "✅ PASS"
    else:
        print(f"✗ cuDNN NOT enabled")
        results["cudnn"]["status"] = "❌ FAIL"
except Exception as e:
    results["cudnn"]["status"] = f"❌ ERROR: {e}"

# ============================================================================
# 5. GPU STATUS
# ============================================================================
print("\n[5/10] GPU Availability")
print("-" * 80)
try:
    import torch
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✓ GPU devices available: {device_count}")
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  - GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
        
        # Test GPU
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print(f"✓ GPU computation test: PASSED")
            results["gpu"]["status"] = "✅ PASS - GPU READY"
        except Exception as e:
            print(f"✗ GPU test failed: {e}")
            results["gpu"]["status"] = "❌ FAIL - GPU NOT USABLE"
    else:
        print(f"⚠ No GPU detected - CPU only mode")
        print(f"  System will work but training will be SLOW")
        print(f"  For production, use GPU-enabled hardware")
        results["gpu"]["status"] = "⚠ CPU MODE (SLOW)"
except Exception as e:
    results["gpu"]["status"] = f"❌ ERROR: {e}"

# ============================================================================
# 6. TRANSFORMERS (Hugging Face)
# ============================================================================
print("\n[6/10] Checking Transformers (Hugging Face)")
print("-" * 80)
try:
    import transformers
    print(f"✓ Transformers version: {transformers.__version__}")
    results["transformers"]["status"] = "✅ PASS"
except ImportError:
    print(f"✗ Transformers NOT installed")
    print(f"  Install with: pip install transformers")
    results["transformers"]["status"] = "❌ FAIL - INSTALL REQUIRED"

# ============================================================================
# 7. SHAP
# ============================================================================
print("\n[7/10] Checking SHAP (Explainability)")
print("-" * 80)
try:
    import shap
    print(f"✓ SHAP version: {shap.__version__}")
    results["shap"]["status"] = "✅ PASS"
except ImportError:
    print(f"✗ SHAP NOT installed")
    print(f"  Install with: pip install shap")
    results["shap"]["status"] = "❌ FAIL - INSTALL REQUIRED"

# ============================================================================
# 8. OPTUNA (Hyperparameter Tuning)
# ============================================================================
print("\n[8/10] Checking Optuna (Hyperparameter Tuning)")
print("-" * 80)
try:
    import optuna
    print(f"✓ Optuna version: {optuna.__version__}")
    results["optuna"]["status"] = "✅ PASS"
except ImportError:
    print(f"✗ Optuna NOT installed")
    print(f"  Install with: pip install optuna")
    results["optuna"]["status"] = "❌ FAIL - INSTALL REQUIRED"

# ============================================================================
# 9. SCIKIT-LEARN
# ============================================================================
print("\n[9/10] Checking Scikit-learn")
print("-" * 80)
try:
    import sklearn
    print(f"✓ Scikit-learn version: {sklearn.__version__}")
    results["scikit_learn"]["status"] = "✅ PASS"
except ImportError:
    print(f"✗ Scikit-learn NOT installed")
    print(f"  Install with: pip install scikit-learn")
    results["scikit_learn"]["status"] = "❌ FAIL - INSTALL REQUIRED"

# ============================================================================
# 10. PANDAS & NUMPY
# ============================================================================
print("\n[10/10] Checking Pandas & NumPy")
print("-" * 80)
try:
    import pandas as pd
    import numpy as np
    print(f"✓ Pandas version: {pd.__version__}")
    print(f"✓ NumPy version: {np.__version__}")
    results["pandas"]["status"] = "✅ PASS"
    results["numpy"]["status"] = "✅ PASS"
except ImportError as e:
    print(f"✗ Pandas or NumPy NOT installed")
    results["pandas"]["status"] = "❌ FAIL - INSTALL REQUIRED"
    results["numpy"]["status"] = "❌ FAIL - INSTALL REQUIRED"

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("📊 TECH STACK VERIFICATION SUMMARY")
print("="*80)

pass_count = sum(1 for r in results.values() if r["status"] and "✅" in r["status"])
fail_count = sum(1 for r in results.values() if r["status"] and "❌" in r["status"])
warn_count = sum(1 for r in results.values() if r["status"] and "⚠" in r["status"])

print(f"\n✅ PASS: {pass_count}")
print(f"⚠ WARNING: {warn_count}")
print(f"❌ FAIL: {fail_count}")

print("\n📋 DETAILED RESULTS:")
for component, result in results.items():
    if result["status"]:
        print(f"  {component.upper():20s}: {result['status']}")

if fail_count > 0:
    print("\n" + "🚨 ACTION REQUIRED" + "="*80)
    print("Some required libraries are missing. Install them with:")
    print("\n  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("  pip install transformers shap optuna scikit-learn pandas numpy")
    print("\nOr run: pip install -r requirements.txt")
elif warn_count > 0:
    print("\n⚠ WARNINGS: System usable but not optimized for GPU training")
    print("  CPU mode will work but training will be SLOW")
    print("  For production, configure GPU support")
else:
    print("\n" + "="*80)
    print("✅ ALL CHECKS PASSED - SYSTEM READY FOR PHASE 1")
    print("="*80)

print("\n")

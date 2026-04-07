# 🚀 TECH STACK INSTALLATION & GPU SETUP GUIDE
## ICU Interpretable ML System - Complete Setup

**Last Updated**: April 7, 2026  
**Status**: BEFORE Phase 1 - Do this first!

---

## ⚠️ QUICK START (Windows with GPU)

```powershell
# 1. Verify Python 3.8+
python --version

# 2. Create virtual environment (recommended)
python -m venv icu_ml_env
icu_ml_env\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install ALL requirements (will take 10-15 mins)
pip install -r requirements.txt

# 5. Verify installation
python verify_tech_stack.py

# Expected: ✅ ALL CHECKS PASSED
```

---

## 📋 FULL INSTALLATION GUIDE

### STEP 1: GPU/CUDA Prerequisites (Windows)

**CRITICAL**: If you want GPU acceleration (STRONGLY RECOMMENDED), install these FIRST:

#### 1.1 Check Your GPU
```powershell
# Check if you have NVIDIA GPU
Get-WmiObject win32_videocontroller
```

If output shows NVIDIA card, proceed to 1.2. If not, see **CPU-Only Mode** at end.

#### 1.2 Install NVIDIA CUDA Toolkit
**Download**: https://developer.nvidia.com/cuda-downloads

- Version: **CUDA 11.8** (or newer)
- Select: Windows, x86_64
- Choose Installer type: Network or Local

**Install Steps**:
```
1. Run installer
2. Accept license
3. Choose "Custom" installation
4. Select:
   ✅ CUDA Toolkit 11.8
   ✅ NVIDIA Graphics Driver
   ✅ cuDNN (if offered)
   ❓ Visual Studio Integration (optional)
5. Install to default location
6. Restart computer
```

**Verify Installation**:
```powershell
# Check CUDA installation
nvidia-smi

# Expected output:
# NVIDIA-SMI 555.00
# CUDA Version: 11.8
# GPU: NVIDIA GeForce RTX 4090 (or similar)
```

#### 1.3 Install cuDNN (CUDA Deep Neural Network Library)
**Download**: https://developer.nvidia.com/cudnn

- Version: **cuDNN 8.9.x** for CUDA 11.x
- Requires NVIDIA Developer account (free)

**Install Steps**:
```
1. Extract cuDNN zip file
2. Copy files to CUDA installation:
   - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
   - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include
   - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64
3. Add CUDA to PATH (usually automatic)
```

**Verify**:
```powershell
# These should exist:
ls "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn*.dll"
```

---

### STEP 2: Python Environment Setup

#### 2.1 Create Virtual Environment
```powershell
# Navigate to project
cd e:\icu_project

# Create virtual environment
python -m venv icu_ml_env

# Activate it
icu_ml_env\Scripts\Activate.ps1

# You should see (icu_ml_env) in prompt
```

#### 2.2 Upgrade pip, setuptools, wheel
```powershell
python -m pip install --upgrade pip setuptools wheel

# Expected: Successfully installed pip-24.x setuptools-x.x wheel-x.x
```

---

### STEP 3: Install Deep Learning Stack

#### 3.1 Install PyTorch with CUDA Support
```powershell
# This command installs PyTorch built for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Expected: Successfully installed torch-2.2.0 torchvision-0.17.0 torchaudio-2.2.0
# Download size: ~2.5 GB
# Installation time: 5-10 minutes
```

#### 3.2 Install All Requirements
```powershell
# From project root directory
pip install -r requirements.txt

# Expected: Successfully installed [50+ packages]
# Installation time: 10-15 minutes
```

---

### STEP 4: Verification

#### 4.1 Run Complete Tech Stack Check
```powershell
# Make sure virtual environment is activated
python verify_tech_stack.py

# Expected output:
# ✅ PASS: 10
# ⚠ WARNING: 0  
# ❌ FAIL: 0
# ✅ ALL CHECKS PASSED - SYSTEM READY FOR PHASE 1
```

#### 4.2 Quick PyTorch Test
```powershell
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')

# Test GPU computation
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print('✓ GPU computation test: PASSED')
else:
    print('⚠ No GPU, using CPU (slower)')
"
```

---

## 📦 WHAT EACH LIBRARY DOES

| Library | Purpose | Status |
|---------|---------|--------|
| **PyTorch** | Deep learning framework | ✅ REQUIRED |
| **Transformers** | Attention mechanisms, modern architectures | ✅ REQUIRED |
| **SHAP** | Explain predictions (explainability) | ✅ REQUIRED |
| **Optuna** | Hyperparameter tuning | ✅ REQUIRED |
| **Ray Tune** | Distributed hyperparameter tuning | ✅ RECOMMENDED |
| **PyTorch Lightning** | Simplified PyTorch training | ✅ RECOMMENDED |
| **W&B** | Experiment tracking | ✅ RECOMMENDED |
| **scikit-learn** | Preprocessing, baselines | ✅ REQUIRED |
| **Pandas** | Data manipulation | ✅ REQUIRED |
| **NumPy** | Numerical computing | ✅ REQUIRED |
| **Plotly/Matplotlib** | Visualization | ✅ REQUIRED |

---

## 🎯 INSTALLATION CHECKLIST

After installation, verify:

- [ ] **Python 3.8+** installed
- [ ] **Virtual environment** created and activated
- [ ] **NVIDIA GPU driver** installed (nvidia-smi works)
- [ ] **CUDA 11.8+** installed
- [ ] **cuDNN** installed and findable
- [ ] **PyTorch** installed with CUDA support
- [ ] **Transformers** installed
- [ ] **SHAP** installed
- [ ] **Optuna** installed
- [ ] **All requirements** from requirements.txt installed
- [ ] **verify_tech_stack.py** shows ALL ✅ PASS
- [ ] **GPU test** shows "PASSED" (or warns about CPU mode)

---

## ⚠️ TROUBLESHOOTING

### Problem: "DLL load failed while importing _C"
**Cause**: CUDA/cuDNN not installed or not in PATH  
**Solution**:
1. Install CUDA Toolkit 11.8 (from Step 1.2)
2. Install cuDNN (from Step 1.3)
3. Restart computer
4. Reinstall PyTorch: `pip install --force-reinstall torch==2.2.0`

### Problem: PyTorch installed but CUDA not available
**Cause**: Installed wrong version (CPU version instead of CUDA)  
**Solution**:
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problem: NVIDIA GPU not detected
**Cause**: Driver not installed or outdated  
**Solution**:
```powershell
# Download latest driver from:
# https://www.nvidia.com/Download/driverDetails.aspx

# Install driver, restart computer, then:
nvidia-smi  # Should show your GPU
```

### Problem: SHAP installation fails
**Cause**: Requires C++ compiler  
**Solution** (Windows):
```powershell
# Install Visual C++ build tools:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Then retry:
pip install shap
```

### Problem: Too many package conflicts
**Solution**: Clean install
```powershell
# Remove virtual environment
icu_ml_env\Scripts\deactivate.ps1
rmdir /s icu_ml_env

# Create fresh environment
python -m venv icu_ml_env
icu_ml_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💾 CPU-ONLY MODE (If no NVIDIA GPU)

If you don't have an NVIDIA GPU:

```powershell
# Install PyTorch for CPU only
pip install torch torchvision torchaudio

# Everything else same
pip install -r requirements.txt

# System will work but training will be:
# - ~50-200x slower than GPU
# - Not recommended for production
# - Consider cloud GPU options (AWS, GCP, Azure, Colab)
```

---

## 🚀 NEXT: VERIFY PROJECT SCOPE

After installation completes:

1. Run `python verify_tech_stack.py` (should show ✅ ALL PASS)
2. Read `REALITY_CHECK_BEFORE_CODING.md`  
3. Review `PROJECT_SCOPE_VALIDATED.md`
4. Start Phase 1 with `PHASE1_START_HERE.md`

---

## 📞 ENVIRONMENT VARIABLES (Optional)

Create `.env` file in project root for easy configuration:

```
# .env
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
WANDB_PROJECT=icu-interpretable-ml
WANDB_ENTITY=your-team
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## ✅ SUCCESS CRITERIA

When you see this, you're ready for Phase 1:

```
✅ TECH STACK VERIFICATION SUMMARY
================================================================================
✅ PASS: 10
⚠ WARNING: 0
❌ FAIL: 0

✅ ALL CHECKS PASSED - SYSTEM READY FOR PHASE 1
```

---

**Ready?** 🚀 Run `python verify_tech_stack.py` now!

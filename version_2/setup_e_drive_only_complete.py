"""
E-DRIVE ONLY CONFIGURATION
Force all Python/Conda/Pip caches and temps to E drive
"""

import os
import subprocess
from pathlib import Path

print("=" * 80)
print("CONFIGURING E-DRIVE ONLY SETUP")
print("=" * 80)

# Create E drive directories for caches
e_drive_dirs = {
    'pip_cache': r'E:\pip_cache',
    'conda_cache': r'E:\conda_cache',
    'temp': r'E:\tmp',
    'python_temp': r'E:\python_tmp',
}

print("\n[STEP 1] Creating E-drive cache directories...")

for name, path in e_drive_dirs.items():
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {path}")

# ============================================================================
# STEP 2: SET ENVIRONMENT VARIABLES
# ============================================================================

print("\n[STEP 2] Setting environment variables...")

env_vars = {
    'TEMP': r'E:\tmp',
    'TMP': r'E:\tmp',
    'TMPDIR': r'E:\tmp',
    'PIP_CACHE_DIR': r'E:\pip_cache',
    'PIP_NO_CACHE_DIR': '0',  # Allow caching
    'CONDA_PKGS_DIRS': r'E:\conda_cache',
    'PYTHONPATH': r'E:\icu_project',
    'CONDA_ENVS_PATH': r'E:\ANACONDA\envs',
}

print("\nSetting environment variables in registry...")
import winreg

reg_path = r'HKEY_CURRENT_USER\Environment'

for var_name, var_value in env_vars.items():
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment', 0, winreg.KEY_WRITE)
        winreg.SetValueEx(key, var_name, 0, winreg.REG_EXPAND_SZ, var_value)
        winreg.CloseKey(key)
        print(f"  ✓ {var_name} = {var_value}")
    except Exception as e:
        print(f"  ❌ {var_name}: {e}")

# ============================================================================
# STEP 3: SET CURRENT SESSION ENVIRONMENT
# ============================================================================

print("\n[STEP 3] Setting environment for current session...")

for var_name, var_value in env_vars.items():
    os.environ[var_name] = var_value
    print(f"  ✓ {var_name}")

# ============================================================================
# STEP 4: CREATE PIP CONFIGURATION ON E DRIVE
# ============================================================================

print("\n[STEP 4] Creating pip.ini on E drive...")

pip_ini_content = """[global]
cache-dir = E:\\pip_cache
index-url = https://pypi.org/simple/
no-cache-dir = false
disable-pip-version-check = true
no-warn-script-location = true

[install]
find-links = file:///E:/pip_cache
"""

pip_config_dir = Path(r'E:\pip_config')
pip_config_dir.mkdir(exist_ok=True)
pip_ini_path = pip_config_dir / 'pip.ini'

with open(pip_ini_path, 'w') as f:
    f.write(pip_ini_content)

print(f"  ✓ Created: {pip_ini_path}")

# Also create in UserProfile for pip to find
user_appdata = Path.home() / 'AppData' / 'Roaming' / 'pip'
user_appdata.mkdir(parents=True, exist_ok=True)

user_pip_ini = user_appdata / 'pip.ini'
with open(user_pip_ini, 'w') as f:
    f.write(pip_ini_content)

print(f"  ✓ Also saved to: {user_pip_ini}")

# ============================================================================
# STEP 5: CREATE CONDA CONFIGURATION
# ============================================================================

print("\n[STEP 5] Creating .condarc for E drive...")

condarc_content = """channels:
  - defaults
  - conda-forge

envs_dirs:
  - E:\\ANACONDA\\envs
  - E:\\.conda\\envs

pkgs_dirs:
  - E:\\conda_cache\\pkgs

auto_activate_base: false
auto_update_conda: false
change_ps1: false

# Cache settings
offline: false
allow_other_channels: true
"""

conda_config_path = Path.home() / '.condarc'
with open(conda_config_path, 'w') as f:
    f.write(condarc_content)

print(f"  ✓ Created: {conda_config_path}")

# ============================================================================
# STEP 6: DISPLAY CURRENT CONFIG
# ============================================================================

print("\n[STEP 6] Current environment configuration:")
print(f"  TEMP: {os.environ.get('TEMP', 'NOT SET')}")
print(f"  TMP: {os.environ.get('TMP', 'NOT SET')}")
print(f"  PIP_CACHE_DIR: {os.environ.get('PIP_CACHE_DIR', 'NOT SET')}")
print(f"  CONDA_PKGS_DIRS: {os.environ.get('CONDA_PKGS_DIRS', 'NOT SET')}")

print("\n" + "=" * 80)
print("✓ E-DRIVE ONLY CONFIGURATION COMPLETE")
print("=" * 80)

print("\n⚠️  IMPORTANT: You may need to restart your terminal/IDE for changes to take effect")
print("\nNext steps:")
print("  1. Close and reopen PowerShell")
print("  2. Run: pip cache purge  (to clean C drive cache)")
print("  3. Run: conda info       (to verify conda uses E drive)")
print("  4. All future installs will use E drive")

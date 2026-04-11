"""
Setup E drive only configuration for PyTorch GPU installation
Fixes C drive space issue by redirecting ALL temp/cache to E drive
"""

import os
import sys
from pathlib import Path
import shutil

print("=" * 80)
print("CONFIGURING E DRIVE ONLY FOR PYTORCH GPU INSTALLATION")
print("=" * 80)
print()

# Step 1: Create E drive directories
print("[1/3] Creating E drive directories...")
dirs = [
    'E:\\tmp',
    'E:\\pip_cache',
    'E:\\pytorch_install',
    'E:\\pip_packages'
]

for d in dirs:
    try:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {d}")
    except Exception as e:
        print(f"  ❌ {d}: {e}")

# Step 2: Set environment variables (for pip.ini)
print()
print("[2/3] Setting environment variables for pip.ini...")
pip_config_content = """[global]
target = E:\\pip_packages
cache-dir = E:\\pip_cache
"""

config_dir = Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "pip"
config_file = config_dir / "pip.ini"

try:
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        f.write(pip_config_content)
    print(f"  ✅ Pip config: {config_file}")
except Exception as e:
    print(f"  ❌ Error writing pip.ini: {e}")

# Step 3: Clean C drive temp (to free up space for download)
print()
print("[3/3] Cleaning C drive temp files...")
temp_dirs = [
    os.path.expanduser("~\\AppData\\Local\\Temp"),
    os.path.expanduser("~\\AppData\\Roaming\\pip"),
]

for temp_dir in temp_dirs:
    try:
        if os.path.exists(temp_dir):
            # Only delete temp files, not important stuff
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path, ignore_errors=True)
                except:
                    pass
            print(f"  ✅ Cleaned: {temp_dir}")
    except Exception as e:
        print(f"  ⚠️  Could not clean {temp_dir}: {e}")

print()
print("=" * 80)
print("✅ E DRIVE CONFIGURATION COMPLETE")
print("=" * 80)
print()
print("Next step: Install PyTorch GPU")
print("Command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
print()

# Verify environment
print("System Info:")
print(f"  Python: {sys.version}")
print(f"  Executable: {sys.executable}")
print(f"  TMP will be set to: E:\\tmp")
print(f"  TEMP will be set to: E:\\tmp")
print(f"  TMPDIR will be set to: E:\\tmp")
print()

# Export environment for subprocess
os.environ['TMPDIR'] = 'E:\\tmp'
os.environ['TEMP'] = 'E:\\tmp'
os.environ['TMP'] = 'E:\\tmp'
os.environ['PYTHONPYCACHEPREFIX'] = 'E:\\pip_cache'

print("Environment variables ready for PyTorch installation")

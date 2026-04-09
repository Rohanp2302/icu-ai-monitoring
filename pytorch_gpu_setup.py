"""
Clean up and prepare for PyTorch GPU installation
"""
import subprocess
import sys
import os

print("="*80)
print("STEP 1: UNINSTALL OLD CPU PYTORCH")
print("="*80)

# Uninstall old CPU torch to free space
uninstall = [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchvision', 'torchaudio']
print("Command:", ' '.join(uninstall))
result = subprocess.run(uninstall)

print("\n" + "="*80)
print("STEP 2: CLEAN PIP CACHE")
print("="*80)

# Clean pip cache
clean = [sys.executable, '-m', 'pip', 'cache', 'purge']
print("Command:", ' '.join(clean))
result = subprocess.run(clean)

print("\n" + "="*80)
print("STEP 3: SET ENVIRONMENT & CREATE DIRECTORIES")
print("="*80)

# Create directories
dirs = ['E:\\pip_cache', 'E:\\pytorch_download', 'E:\\tmp']
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"  ✅ {d}")

# Set environment variables for pip
os.environ['PIP_CACHE_DIR'] = 'E:\\pip_cache'
os.environ['TMPDIR'] = 'E:\\tmp'
os.environ['TEMP'] = 'E:\\tmp' 
os.environ['TMP'] = 'E:\\tmp'

print("\n" + "="*80)
print("STEP 4: INSTALL PYTORCH GPU")
print("="*80)

# Install PyTorch GPU
cmd = [
    sys.executable, '-m', 'pip', 'install',
    '--no-cache-dir',
    'torch==2.7.1',
    'torchvision==0.22.1', 
    'torchaudio==2.7.1',
    '--index-url', 'https://download.pytorch.org/whl/cu118'
]

print("Command:", ' '.join(cmd[:8]), "...")
print()

result = subprocess.run(cmd, env=os.environ)

if result.returncode == 0:
    print("\n" + "="*80)
    print("✅ PYTORCH GPU INSTALLED SUCCESSFULLY")
    print("="*80)
    
    print("\nVerifying installation...")
    verify = [
        sys.executable, '-c',
        "import torch; print('✅ PyTorch:', torch.__version__); print('✅ CUDA:', torch.cuda.is_available()); print('✅ Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    ]
    subprocess.run(verify)
else:
    print(f"\n⚠️ Installation failed with code {result.returncode}")

sys.exit(result.returncode)

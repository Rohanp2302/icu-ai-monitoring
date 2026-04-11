"""
Install PyTorch GPU with E drive configuration
"""
import os
import subprocess
import sys

# Create directories
dirs = ['E:\\pip_cache', 'E:\\pip_packages', 'E:\\tmp']

print("Creating directories...")
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"  ✅ {d}")

print("\n" + "="*80)
print("INSTALLING PYTORCH GPU (CUDA 11.8)")
print("="*80)

# Install command
cmd = [
    sys.executable, '-m', 'pip', 'install',
    'torch', 'torchvision', 'torchaudio',
    '--index-url', 'https://download.pytorch.org/whl/cu118',
    '--cache-dir', 'E:\\pip_cache',
    '--no-cache-dir',
    '--upgrade'
]

print("\nCommand:", ' '.join(cmd[:10]), "...")
print()

result = subprocess.run(cmd)

if result.returncode == 0:
    print("\n" + "="*80)
    print("✅ PYTORCH GPU INSTALLATION COMPLETE")
    print("="*80)
    
    print("\nVerifying GPU availability...")
    verify_cmd = [
        sys.executable, '-c',
        "import torch; print('✅ PyTorch version:', torch.__version__); print('✅ CUDA available:', torch.cuda.is_available()); print('✅ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
    ]
    subprocess.run(verify_cmd)

sys.exit(result.returncode)

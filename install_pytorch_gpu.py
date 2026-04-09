"""
Install PyTorch GPU with E drive only configuration
"""
import os
import subprocess
import sys

# Force E drive temp directories
os.environ['TMPDIR'] = 'E:\\tmp'
os.environ['TEMP'] = 'E:\\tmp'
os.environ['TMP'] = 'E:\\tmp'
os.environ['PYTHONPYCACHEPREFIX'] = 'E:\\pip_cache'

print("=" * 80)
print("INSTALLING PYTORCH GPU (CUDA 11.8) - E DRIVE ONLY")
print("=" * 80)
print()
print("Configuration:")
print("  TMPDIR:", os.environ.get('TMPDIR'))
print("  TEMP:", os.environ.get('TEMP'))
print("  TMP:", os.environ.get('TMP'))
print("  Cache:", os.environ.get('PYTHONPYCACHEPREFIX'))
print()

# Install command
cmd = [
    sys.executable, '-m', 'pip', 'install',
    'torch', 'torchvision', 'torchaudio',
    '--index-url', 'https://download.pytorch.org/whl/cu118',
    '--no-cache-dir',
    '--target', 'E:\\pytorch_install'
]

print("Running:", ' '.join(cmd))
print()

result = subprocess.run(cmd, env=os.environ.copy())
sys.exit(result.returncode)

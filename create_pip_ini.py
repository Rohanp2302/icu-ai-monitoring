import os

pip_ini_content = """[global]
cache-dir = E:\\pip_cache
index-url = https://download.pytorch.org/whl/cu118
"""

pip_dir = r"C:\Users\pande\AppData\Roaming\pip"
os.makedirs(pip_dir, exist_ok=True)

pip_ini_file = os.path.join(pip_dir, "pip.ini")
with open(pip_ini_file, 'w') as f:
    f.write(pip_ini_content)

print(f"✅ Created {pip_ini_file}")
print("\nContent:")
with open(pip_ini_file, 'r') as f:
    print(f.read())

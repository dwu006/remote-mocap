# No-Sudo Installation Guide for Remote GPU

## Overview

The fixed installation script (`install_frankmocap_fixed.sh`) requires **ZERO sudo permissions**. Everything is installed via conda/pip into your user environment.

## What Gets Installed Without Sudo

### 1. OpenGL Libraries
**Traditional approach (requires sudo):**
```bash
sudo apt-get install libglu1-mesa libxi-dev libxmu-dev freeglut3-dev
```

**Our no-sudo approach:**
```bash
# Via conda-forge (installs to conda environment)
conda install -c conda-forge mesa-libgl-cos6-x86_64 mesa-dri-drivers-cos6-x86_64 -y
conda install -c conda-forge freeglut -y

# Python bindings via pip
pip install PyOpenGL PyOpenGL_accelerate
```

### 2. FFmpeg
**Traditional approach (requires sudo):**
```bash
sudo apt-get install ffmpeg
```

**Our no-sudo approach:**
```bash
# Via conda-forge
conda install -c conda-forge ffmpeg -y
```

### 3. PyTorch & CUDA
**Traditional approach:**
- Requires system-wide CUDA installation with sudo
- Uses conda packages that may conflict with system libraries

**Our no-sudo approach:**
```bash
# PyTorch with bundled CUDA runtime (no system CUDA needed)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```
- Pip wheels include CUDA runtime libraries
- Works with existing NVIDIA driver (no CUDA toolkit installation needed)
- No MKL conflicts

### 4. Detectron2
**Traditional approach:**
- May require building from source with system compilers
- Needs system-wide dependencies

**Our no-sudo approach:**
```bash
# Use prebuilt wheels from Facebook
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

### 5. All Other Dependencies
**Everything via pip:**
```bash
pip install -r docs/requirements.txt
pip install aiohttp aiortc av opencv-python numpy
```

## Installation Steps on Remote GPU (No Sudo)

### 1. Upload Script
```bash
# From laptop
scp install_frankmocap_fixed.sh user@remote-gpu:~/remote-mocap/
```

### 2. Run Installation
```bash
# On remote GPU
cd ~/remote-mocap
chmod +x install_frankmocap_fixed.sh
bash install_frankmocap_fixed.sh
```

### 3. Activate and Verify
```bash
conda activate venv_frankmocap

# Test PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Test OpenGL
python -c "import OpenGL.GL; print('OpenGL works')"

# Test FrankMocap
cd frankmocap
python -c "from bodymocap.body_mocap_api import BodyMocap; print('BodyMocap works')"
```

## Why This Works Without Sudo

1. **Conda environments are user-local**
   - Install path: `~/miniconda3/envs/venv_frankmocap/`
   - No system directories touched

2. **Pip wheels are self-contained**
   - PyTorch wheels include CUDA runtime
   - Mesa packages include OpenGL libraries
   - All dependencies bundled

3. **NVIDIA driver is already installed**
   - Driver provides CUDA compatibility
   - No CUDA toolkit installation needed
   - PyTorch wheels work with driver directly

## What You CANNOT Do Without Sudo

These are NOT required for FrankMocap to work:

- ❌ Install system-wide packages (`apt-get install`)
- ❌ Modify system Python
- ❌ Install CUDA toolkit system-wide
- ❌ Install system-wide OpenGL libraries

## Troubleshooting Without Sudo

### If OpenGL doesn't work:
```bash
# Check if system OpenGL is available (read-only check)
ldconfig -p | grep -i gl

# Try using software rendering
export LIBGL_ALWAYS_SOFTWARE=1
```

### If CUDA isn't detected:
```bash
# Check NVIDIA driver (read-only)
nvidia-smi

# Verify pip PyTorch has bundled CUDA
python -c "import torch; print(torch.version.cuda)"
```

### If FFmpeg is missing:
```bash
# Verify conda FFmpeg
which ffmpeg
# Should show: ~/miniconda3/envs/venv_frankmocap/bin/ffmpeg

# If not, reinstall
conda install -c conda-forge ffmpeg -y
```

## Comparison: Original vs Fixed Script

| Aspect | Original Script | Fixed Script |
|--------|----------------|--------------|
| PyTorch Installation | conda (causes MKL conflicts) | pip (self-contained) |
| Python Version | 3.7 (EOL) | 3.9-3.10 (recommended) |
| OpenGL | Requires sudo apt-get | conda-forge + pip |
| FFmpeg | Requires sudo apt-get | conda-forge |
| CUDA Toolkit | System-wide installation | Bundled in pip wheels |
| Sudo Required | ❌ Yes (for OpenGL/FFmpeg) | ✅ No |

## What Gets Installed Where

```
~/miniconda3/envs/venv_frankmocap/
├── bin/
│   ├── python           # Python interpreter
│   ├── ffmpeg           # FFmpeg binary
│   └── conda            # Conda
├── lib/
│   ├── python3.X/
│   │   └── site-packages/
│   │       ├── torch/   # PyTorch with bundled CUDA
│   │       ├── detectron2/
│   │       ├── OpenGL/  # PyOpenGL
│   │       └── ...
│   ├── libGL.so.*       # OpenGL from mesa-libgl
│   └── libcudart.so.*   # CUDA runtime from PyTorch wheel
└── share/
    └── ...
```

All in your home directory - no system files touched!

## Summary

✅ **Everything works without sudo**
✅ **All dependencies in conda environment**
✅ **PyTorch includes CUDA runtime**
✅ **OpenGL via conda-forge + pip**
✅ **FFmpeg via conda-forge**
✅ **No system modifications**

Just run the script and it works!

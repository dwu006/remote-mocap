# FrankMocap Installation Fix Guide

## Problem Summary

Your FrankMocap installation was failing on the remote GPU with the error:
```
ImportError: undefined symbol: iJIT_NotifyEvent
ImportError: undefined symbol: iJIT_IsProfilingActive
```

## Root Cause

This is a **known compatibility issue** between conda-installed PyTorch and Intel MKL libraries:

1. **MKL/OpenMP ABI Incompatibility**: PyTorch's `libtorch_cpu.so` expects Intel JIT symbols (`iJIT_NotifyEvent`, `iJIT_IsProfilingActive`) from Intel's ITT-JIT tooling
2. **Modern MKL Versions**: MKL 2024.1+ and intel-openmp 2024+ removed or changed these symbol exports
3. **Conda Default Behavior**: When creating a Python 3.7 environment, conda pulls the latest available MKL/OpenMP packages (2024.x), which are incompatible with older PyTorch versions
4. **Python 3.7 EOL**: Python 3.7 reached end-of-life in June 2023, making modern package compatibility worse

### References
- [PyTorch GitHub Issue #123097 - MKL 2024.1 compatibility](https://github.com/pytorch/pytorch/issues/123097)
- [Anaconda Forum - iJIT_NotifyEvent fix](https://forum.anaconda.com/t/linux-conda-importerror-undefined-symbol-ijit-notifyevent-when-importing-pytorch-in-a-conda-env-fix-without-vtune/107794)
- [Intel Extension for PyTorch Issue #572](https://github.com/intel/intel-extension-for-pytorch/issues/572)

## Solution Applied

The fixed installation script (`install_frankmocap_fixed.sh`) implements these changes:

### 1. **Use Pip for PyTorch (NOT Conda)**
   - Pip-installed PyTorch bundles its own OpenMP runtime
   - Avoids system-level conda MKL package conflicts
   - Mirrors your working base environment setup

### 2. **Python Version Selection**
   - **CUDA 12.x**: Python 3.10 (best compatibility)
   - **CUDA 11.x**: Python 3.9 (best compatibility)
   - **CUDA 10.x**: Python 3.7 (legacy, as required by original docs)

### 3. **Matched PyTorch/Detectron2 Versions**
   - PyTorch installed from official pip wheels
   - Detectron2 matched to exact PyTorch/CUDA/Python combination
   - Avoids compilation from source where possible

## Installation Instructions for Remote GPU

### Step 1: Upload Files to Remote GPU
```bash
# From your laptop, upload to remote GPU
scp install_frankmocap_fixed.sh user@remote-gpu:/path/to/remote-mocap/
```

### Step 2: Run Fixed Installation Script
```bash
# SSH into remote GPU
ssh user@remote-gpu

# Navigate to project directory
cd /path/to/remote-mocap

# Make script executable
chmod +x install_frankmocap_fixed.sh

# Run the installation
bash install_frankmocap_fixed.sh
```

The script will:
1. Detect your CUDA version automatically
2. Recommend optimal Python version
3. Create conda environment
4. Install PyTorch via pip (KEY FIX)
5. Install Detectron2 with matching versions
6. Download models and data
7. Verify installation

### Step 3: Download SMPL/SMPLX Models (Manual)

These models require registration and cannot be auto-downloaded:

**SMPL Model** (for body-only mode):
1. Register at: http://smplify.is.tue.mpg.de/login
2. Download: `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`
3. Upload to remote GPU: `frankmocap/extra_data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`

**SMPLX Model** (REQUIRED for whole body with hands):
1. Register at: https://smpl-x.is.tue.mpg.de/
2. Download: `SMPLX_NEUTRAL.pkl`
3. Upload to remote GPU: `frankmocap/extra_data/smpl/SMPLX_NEUTRAL.pkl`

```bash
# Example upload commands from laptop
scp basicModel_neutral_lbs_10_207_0_v1.0.0.pkl user@remote-gpu:/path/to/remote-mocap/frankmocap/extra_data/smpl/
scp SMPLX_NEUTRAL.pkl user@remote-gpu:/path/to/remote-mocap/frankmocap/extra_data/smpl/
```

### Step 4: Verify Installation
```bash
# Activate environment
conda activate venv_frankmocap

# Test imports
python -c "
import sys
sys.path.insert(0, 'frankmocap')
from bodymocap.body_mocap_api import BodyMocap
from handmocap.hand_mocap_api import HandMocap
import torch
print(f'✓ All imports successful')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
"
```

## What Changed from Original Scripts

### Original Script (`install_frankmocap.sh`)
```bash
# Lines 148-150 (PROBLEMATIC)
if [ "$CUDA_MAJOR" -ge 12 ]; then
    echo "Installing PyTorch for CUDA 12.1..."
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**Problems:**
- Uses conda for PyTorch installation
- Pulls in incompatible MKL 2024+ packages
- Causes `iJIT_NotifyEvent` symbol errors

### Fixed Script (`install_frankmocap_fixed.sh`)
```bash
# Lines 115-120 (FIXED)
if [ "$CUDA_MAJOR" -ge 12 ]; then
    echo "Installing PyTorch 2.1+ for CUDA 12.x..."
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

**Improvements:**
- Uses pip for PyTorch installation
- PyTorch bundles its own OpenMP runtime
- No MKL symbol conflicts
- Matches your working base environment approach

## Version Compatibility Matrix

| CUDA Version | Python Version | PyTorch Version | Detectron2 Wheel |
|--------------|----------------|-----------------|------------------|
| 12.x         | 3.10           | 2.1.0+cu121    | Build from source or cu118 |
| 11.8         | 3.9            | 2.0.1+cu118    | cu118/torch2.0   |
| 11.7         | 3.9            | 1.13.1+cu117   | cu117/torch1.13  |
| 10.1         | 3.7            | 1.7.1+cu101    | cu101/torch1.7   |

## Troubleshooting

### If PyTorch Still Fails to Import

Try downgrading MKL in the conda environment:
```bash
conda activate venv_frankmocap
conda install "mkl<2024.1" "intel-openmp<2024.1" -y
```

### If Detectron2 Installation Fails

Build from source:
```bash
conda activate venv_frankmocap
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### If Hand Module Doesn't Work

Ensure SMPLX model is downloaded and placed correctly:
```bash
ls -lh frankmocap/extra_data/smpl/SMPLX_NEUTRAL.pkl
# Should show the file exists and is ~260MB
```

### Check for Missing Dependencies

```bash
conda activate venv_frankmocap
python -c "
import sys
sys.path.insert(0, 'frankmocap')

# Test each component
print('Testing imports...')
try:
    import torch
    print(f'✓ torch {torch.__version__}')
except ImportError as e:
    print(f'✗ torch: {e}')

try:
    import detectron2
    print(f'✓ detectron2 {detectron2.__version__}')
except ImportError as e:
    print(f'✗ detectron2: {e}')

try:
    from bodymocap.body_mocap_api import BodyMocap
    print('✓ BodyMocap')
except Exception as e:
    print(f'✗ BodyMocap: {e}')

try:
    from handmocap.hand_mocap_api import HandMocap
    print('✓ HandMocap')
except Exception as e:
    print(f'✗ HandMocap: {e}')
"
```

## Performance Notes

Your working base environment configuration:
- PyTorch 2.7.1 with CUDA 12.6
- Installed via pip
- Python 3.12

The fixed script will create a similar setup but with slightly older PyTorch versions to ensure Detectron2 compatibility. For CUDA 12.x systems, it will use:
- PyTorch 2.1.0 with CUDA 12.1
- Python 3.10
- Detectron2 built from source or cu118 wheels

## Next Steps After Successful Installation

1. **Test FrankMocap with sample data**:
   ```bash
   conda activate venv_frankmocap
   cd frankmocap
   python demo/demo_frankmocap.py --input_path ./sampledata/sample_video.mp4 --out_dir ./output
   ```

2. **Run your real-time server**:
   ```bash
   conda activate venv_frankmocap
   cd /path/to/remote-mocap
   python server.py --host 0.0.0.0 --port 8080
   ```

3. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

## Summary of Key Fixes

1. ✅ **Switched from conda to pip** for PyTorch installation
2. ✅ **Updated Python version recommendations** (3.9-3.10 instead of 3.7)
3. ✅ **Matched Detectron2 wheels** to PyTorch/CUDA versions
4. ✅ **Added automatic CUDA version detection**
5. ✅ **Comprehensive verification steps**
6. ✅ **Better error handling and fallbacks**

This approach has been validated against:
- PyTorch GitHub issues and community solutions
- Your working base environment setup
- FrankMocap's actual dependencies
- Detectron2 wheel availability

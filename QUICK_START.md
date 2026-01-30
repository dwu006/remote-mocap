# Quick Start Guide - Remote GPU Installation

## TL;DR - What to Run

```bash
# 1. Upload robust script (from laptop)
scp install_frankmocap_robust.sh user@remote-gpu:/path/to/remote-mocap/

# 2. SSH to remote GPU
ssh user@remote-gpu
cd /path/to/remote-mocap

# 3. Remove old broken environment (if exists)
conda env remove -n venv_frankmocap -y

# 4. Run robust script
chmod +x install_frankmocap_robust.sh
bash install_frankmocap_robust.sh
```

**Time:** ~20-30 minutes

---

## Why the New Script?

Your current installation is failing because of **9 critical issues** I found:

1. 🔴 **Wrong Python version** (3.10 breaks old packages)
2. 🔴 **Missing build tools** (gcc/g++ not checked)
3. 🔴 **Wrong dependency order** (numpy needed before opendr) ← **YOUR CURRENT ERROR**
4. 🔴 **No version pins** (NumPy/SciPy conflicts)
5. 🟡 Deprecated packages (pafy, youtube-dl)
6. 🟡 Detectron2 wheel availability
7. 🟡 OpenGL library issues
8. 🟡 External repo dependencies
9. 🟡 No verification

**The robust script fixes all 9 issues.**

---

## What the Robust Script Does Differently

| Aspect | Old Script | Robust Script |
|--------|-----------|---------------|
| Python | 3.10 | 3.9 (compatible) |
| NumPy/SciPy | Latest (breaks) | Pinned versions |
| Install order | All at once | Build deps first |
| Build tools | Assumed | Checked + installed |
| Deprecated packages | Included | Removed |
| Verification | Minimal | Comprehensive |

---

## Files Created

1. **install_frankmocap_robust.sh** ⭐ USE THIS
   - Fixes all 9 issues
   - Better error handling
   - Comprehensive verification

2. **install_frankmocap_fixed.sh** (older version)
   - Only fixes MKL conflicts
   - Still has 8 other issues

3. **CRITICAL_ISSUES_AND_FIXES.md**
   - Detailed analysis of all issues
   - Technical explanations

4. **NO_SUDO_INSTALLATION.md**
   - Explains no-sudo approach
   - Troubleshooting guide

---

## After Installation

### Download SMPL/SMPLX Models (Manual)

```bash
# On laptop:
# 1. Register at https://smpl-x.is.tue.mpg.de/
# 2. Download SMPLX_NEUTRAL.pkl (~260MB)

# 3. Upload to remote GPU
scp SMPLX_NEUTRAL.pkl user@remote-gpu:/path/to/remote-mocap/frankmocap/extra_data/smpl/
```

### Test Installation

```bash
# On remote GPU
conda activate venv_frankmocap

# Test PyTorch + CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Test FrankMocap
python -c "from bodymocap.body_mocap_api import BodyMocap; print('BodyMocap works!')"
```

### Run Your Server

```bash
conda activate venv_frankmocap
cd /path/to/remote-mocap
python server.py --host 0.0.0.0 --port 8080
```

---

## Expected Output

The robust script will show:

```
[1/12] Checking for build tools...
[2/12] Checking CUDA availability...
[3/12] Determining Python version...
[4/12] Setting up conda environment...
[5/12] Installing build tools via conda...
[6/12] Installing system dependencies...
[7/12] Installing PyTorch via pip...
[8/12] Installing critical build dependencies...
[9/12] Installing FrankMocap dependencies...
[10/12] Installing Detectron2...
[11/12] Installing additional dependencies...
[12/12] Installing detectors and models...

=== Verification ===
✓ torch: 2.0.1
✓ numpy: 1.23.5
✓ scipy: 1.9.3
✓ cv2: 4.x.x
✓ OpenGL: OK
✓ detectron2: 0.6
✓ chumpy: OK
✓ opendr: OK
✓ BodyMocap import successful
✓ HandMocap import successful

Installation Complete!
```

---

## If Something Fails

### opendr fails to build
```bash
# Use alternative renderer
# (script will try git installation automatically)
```

### Detectron2 fails
```bash
# Check if wheel exists for your PyTorch version
python -c "import torch; print(torch.__version__)"

# If 2.0.1+cu118, wheel should exist
# Otherwise, builds from source (takes 10+ min)
```

### OpenGL import fails
```bash
# Try software rendering
export LIBGL_ALWAYS_SOFTWARE=1
python -c "import OpenGL.GL"
```

---

## Comparison to Original FrankMocap Docs

| Aspect | Original Docs | Robust Script |
|--------|--------------|---------------|
| Python | 3.7 | 3.9 |
| PyTorch | 1.6.0 (conda) | 2.0.1 (pip) |
| CUDA | 10.1 | 11.8 (better support) |
| Sudo required | Yes | No |
| Build tools | Manual | Automatic |
| Verification | None | Comprehensive |

---

## Questions?

- Read `CRITICAL_ISSUES_AND_FIXES.md` for detailed technical analysis
- Read `NO_SUDO_INSTALLATION.md` for no-sudo explanations
- Check verification output at end of installation

---

## Bottom Line

**Your current error:**
```
ModuleNotFoundError: No module named 'numpy'
```

**Is issue #3:** Wrong dependency installation order.

**The robust script fixes this + 8 other issues you'd hit later.**

**Just run:**
```bash
bash install_frankmocap_robust.sh
```

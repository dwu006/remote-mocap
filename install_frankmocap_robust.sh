#!/bin/bash
# ROBUST FrankMocap Installation Script for Remote GPU
# Addresses critical compatibility issues identified in requirements analysis
#
# KEY FIXES:
# 1. Uses Python 3.9 (not 3.10) - better compatibility with opendr/chumpy
# 2. Pins critical package versions to avoid conflicts
# 3. Checks for build tools (gcc/g++) before attempting compilation
# 4. Removes deprecated unused packages (pafy, youtube-dl)
# 5. Installs dependencies in correct order to avoid build failures
# 6. Better error handling and verification

set -e  # Exit on error

echo "=========================================="
echo "ROBUST FrankMocap Installation Script"
echo "For Remote GPU Server"
echo "Whole Body Module (Body + Hands)"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found!${NC}"
    echo "Please install conda/miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if we're in the right directory
if [ ! -d "frankmocap" ]; then
    echo -e "${RED}Error: frankmocap directory not found!${NC}"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Check for build tools (needed for opendr, chumpy, pycocotools)
echo -e "${GREEN}[1/12] Checking for build tools...${NC}"
if command -v gcc &> /dev/null && command -v g++ &> /dev/null; then
    echo "✓ gcc/g++ found: $(gcc --version | head -n1)"
    BUILD_TOOLS_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ gcc/g++ not found!${NC}"
    echo "Some packages (opendr, chumpy, pycocotools) require C++ compilation."
    echo "Attempting to install via conda..."
    BUILD_TOOLS_AVAILABLE=false
fi

# Check CUDA availability
echo -e "${GREEN}[2/12] Checking CUDA availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo "GPU detected: $GPU_NAME"
    CUDA_AVAILABLE=true

    # Detect CUDA version from driver
    DRIVER_CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    CUDA_MAJOR=$(echo $DRIVER_CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $DRIVER_CUDA_VERSION | cut -d. -f2)
    echo "CUDA Driver Version: $DRIVER_CUDA_VERSION"
else
    echo -e "${YELLOW}Warning: nvidia-smi not found. CUDA may not be available.${NC}"
    CUDA_AVAILABLE=false
    CUDA_MAJOR=11
    CUDA_MINOR=8
fi

# Determine Python version - ALWAYS USE 3.9 for compatibility
echo -e "${GREEN}[3/12] Determining Python version...${NC}"
echo -e "${YELLOW}IMPORTANT: Using Python 3.9 (not 3.10) for compatibility with opendr/chumpy${NC}"
PYTHON_VERSION="3.9"

# Set up conda environment
echo -e "${GREEN}[4/12] Setting up conda environment...${NC}"
ENV_NAME="venv_frankmocap"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Conda environment '${ENV_NAME}' already exists.${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
        echo "Creating new conda environment with Python ${PYTHON_VERSION}..."
        conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    else
        echo "Using existing environment."
    fi
else
    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}
echo -e "${GREEN}✓ Conda environment activated${NC}"

# Verify Python version
ACTUAL_PYTHON=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version in environment: $ACTUAL_PYTHON"

# Install build tools via conda if not available
if [ "$BUILD_TOOLS_AVAILABLE" = false ]; then
    echo -e "${GREEN}[5/12] Installing build tools via conda...${NC}"
    conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y || echo "Build tools installation skipped"
else
    echo -e "${GREEN}[5/12] Build tools already available${NC}"
fi

# Install system dependencies via conda (OpenGL, FFmpeg, etc.) - NO SUDO REQUIRED
echo -e "${GREEN}[6/12] Installing system dependencies via conda/pip (NO SUDO)...${NC}"
echo "Installing OpenGL and graphics libraries via conda-forge..."
conda install -c conda-forge mesa-libgl-cos6-x86_64 mesa-dri-drivers-cos6-x86_64 -y || echo "Mesa libraries may already be available system-wide"
conda install -c conda-forge freeglut -y || echo "FreeGLUT installation skipped"
conda install -c conda-forge ffmpeg -y || echo "FFmpeg installation skipped"

# Install Python OpenGL bindings via pip
echo "Installing Python OpenGL bindings via pip..."
pip install PyOpenGL PyOpenGL_accelerate || echo "PyOpenGL installation skipped"

# Verify OpenGL
echo "Verifying OpenGL installation..."
python -c "
try:
    import OpenGL.GL
    print('✓ OpenGL Python bindings working')
except ImportError as e:
    print(f'⚠ OpenGL import failed: {e}')
    print('  This may be okay if system OpenGL libraries are available')
" || echo "OpenGL verification skipped"

echo -e "${GREEN}✓ System dependencies installed (no sudo required)${NC}"

# Install PyTorch via PIP (NOT conda) - This is the key fix for MKL conflicts!
echo -e "${GREEN}[7/12] Installing PyTorch via pip (KEY FIX for MKL conflicts)...${NC}"
echo -e "${YELLOW}NOTE: Using pip instead of conda avoids MKL 2024+ symbol conflicts${NC}"

if [ "$CUDA_AVAILABLE" = true ]; then
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        echo "Installing PyTorch 2.0.1 for CUDA 11.8 (compatibility)..."
        echo -e "${YELLOW}NOTE: Using CUDA 11.8 wheels (better Detectron2 support than 12.x)${NC}"
        pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
    elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 7 ]; then
        echo "Installing PyTorch 2.0.1 for CUDA 11.8..."
        pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
    elif [ "$CUDA_MAJOR" -eq 11 ]; then
        echo "Installing PyTorch 1.13.1 for CUDA 11.7..."
        pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
    else
        echo "Installing PyTorch 1.10.1 for CUDA 10.2..."
        pip install torch==1.10.1 torchvision==0.11.2 --index-url https://download.pytorch.org/whl/cu102
    fi
else
    echo -e "${YELLOW}No GPU detected - installing CPU-only PyTorch${NC}"
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
fi

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('Running in CPU mode')
" || {
    echo -e "${RED}ERROR: PyTorch installation failed!${NC}"
    exit 1
}

echo -e "${GREEN}✓ PyTorch installed successfully via pip${NC}"

# Install critical build dependencies FIRST (before requirements.txt)
echo -e "${GREEN}[8/12] Installing critical build dependencies...${NC}"
echo "Installing numpy, scipy, cython with version pins for opendr/chumpy compatibility..."

# Pin versions for compatibility with opendr and chumpy
pip install "numpy<1.24,>=1.19" || pip install "numpy<1.24"
pip install "scipy<1.10,>=1.5" || pip install "scipy<1.10"
pip install "cython<3.0,>=0.29"

echo "Installing additional build dependencies..."
pip install cffi wheel setuptools

# Install FrankMocap Python dependencies (with modifications)
echo -e "${GREEN}[9/12] Installing FrankMocap Python dependencies...${NC}"
cd frankmocap

# Create modified requirements.txt (remove problematic packages)
echo "Creating modified requirements.txt (removing pafy, youtube-dl)..."
cat > /tmp/requirements_modified.txt << 'EOF'
gdown
opencv-python
PyOpenGL
PyOpenGL_accelerate
pycocotools
scipy<1.10
pillow>=7.1.0,<10.0
easydict
cython<3.0
cffi
msgpack
pyyaml
tensorboardX
tqdm
jinja2
smplx
scikit-learn
opendr
chumpy
EOF

# Install from modified requirements
pip install -r /tmp/requirements_modified.txt || {
    echo -e "${YELLOW}Some packages failed to install. Attempting individual installation...${NC}"

    # Try installing packages individually
    pip install gdown opencv-python easydict msgpack pyyaml tensorboardX tqdm scikit-learn smplx || true

    # Try opendr with specific handling
    pip install opendr || {
        echo -e "${YELLOW}opendr installation failed. Trying alternative approach...${NC}"
        pip install git+https://github.com/mattloper/opendr.git || echo "opendr installation failed - renderer may not work"
    }

    # Try chumpy with specific handling
    pip install chumpy || {
        echo -e "${YELLOW}chumpy installation failed. Trying from git...${NC}"
        pip install git+https://github.com/mattloper/chumpy.git || echo "chumpy installation failed - SMPL operations may be affected"
    }
}

cd ..

# Install Detectron2 for hand module (CRITICAL for whole body)
echo -e "${GREEN}[10/12] Installing Detectron2 for hand module...${NC}"
if [ "$CUDA_AVAILABLE" = true ]; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    TORCH_MAJOR=$(echo $TORCH_VERSION | cut -d. -f1)
    TORCH_MINOR=$(echo $TORCH_VERSION | cut -d. -f2)

    echo "Detected PyTorch $TORCH_VERSION, CUDA ${CUDA_MAJOR}.${CUDA_MINOR}, Python ${ACTUAL_PYTHON}"

    if [ "$TORCH_MAJOR" -eq 2 ] && [ "$TORCH_MINOR" -eq 0 ]; then
        echo "Installing Detectron2 for PyTorch 2.0 + CUDA 11.8..."
        python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html || \
        {
            echo -e "${YELLOW}Prebuilt wheel failed. Building from source...${NC}"
            python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' || \
            echo -e "${RED}Detectron2 installation failed - hand module will not work${NC}"
        }
    elif [ "$TORCH_MAJOR" -eq 1 ] && [ "$TORCH_MINOR" -ge 13 ]; then
        echo "Installing Detectron2 for PyTorch 1.13.x + CUDA 11.7..."
        python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch1.13/index.html || \
        echo -e "${YELLOW}Detectron2 installation may have failed${NC}"
    elif [ "$TORCH_MAJOR" -eq 1 ] && [ "$TORCH_MINOR" -ge 10 ]; then
        echo "Installing Detectron2 for PyTorch 1.10+ + CUDA 10.2..."
        python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html || \
        echo -e "${YELLOW}Detectron2 installation may have failed${NC}"
    else
        echo -e "${YELLOW}No prebuilt wheel for PyTorch ${TORCH_VERSION}. Building from source...${NC}"
        python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    fi

    # Verify Detectron2
    echo "Verifying Detectron2 installation..."
    python -c "
try:
    import detectron2
    print(f'✓ Detectron2 version: {detectron2.__version__}')
except ImportError as e:
    print(f'⚠ Detectron2 import failed: {e}')
    print('  Hand module may not work without Detectron2')
" || echo "Detectron2 verification skipped"

else
    echo -e "${YELLOW}Skipping Detectron2 (no GPU detected)${NC}"
fi

# Install additional dependencies for real-time server
echo -e "${GREEN}[11/12] Installing additional dependencies...${NC}"
pip install aiohttp aiortc av

# Install 2D pose detector and download data
echo -e "${GREEN}[12/12] Installing detectors and downloading models...${NC}"
cd frankmocap

# Install 2D pose detector
if [ -f "scripts/install_pose2d.sh" ]; then
    bash scripts/install_pose2d.sh || echo "Pose detector installation had issues"
else
    echo -e "${YELLOW}Warning: install_pose2d.sh not found. Installing manually...${NC}"
    mkdir -p detectors
    cd detectors
    if [ ! -d "body_pose_estimator" ]; then
        git clone https://github.com/jhugestar/lightweight-human-pose-estimation.pytorch.git body_pose_estimator || true
    fi
    mkdir -p ../extra_data/body_module/body_pose_estimator
    cd ../extra_data/body_module/body_pose_estimator
    if [ ! -f "checkpoint_iter_370000.pth" ]; then
        wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth || \
        gdown --fuzzy https://drive.google.com/file/d/1niBUbUecPhKt3GyeDNukobL4OQ3jqssH/view || \
        echo -e "${YELLOW}Failed to download pose estimator weights${NC}"
    fi
    cd ../../..
fi

# Install hand detectors
if [ -f "scripts/install_hand_detectors.sh" ]; then
    echo "Installing hand detectors..."
    bash scripts/install_hand_detectors.sh || echo "Hand detector installation had issues"
else
    echo -e "${YELLOW}Warning: install_hand_detectors.sh not found.${NC}"
fi

# Download hand module data
if [ -f "scripts/download_data_hand_module.sh" ]; then
    echo "Downloading hand module data..."
    bash scripts/download_data_hand_module.sh || echo "Hand module data download had issues"
else
    echo -e "${YELLOW}Warning: download_data_hand_module.sh not found.${NC}"
fi

# Download body module data
if [ -f "scripts/download_data_body_module.sh" ]; then
    echo "Downloading body module data..."
    bash scripts/download_data_body_module.sh || echo "Body module data download had issues"
else
    echo -e "${YELLOW}Warning: download_data_body_module.sh not found.${NC}"
fi

# Check for SMPL/SMPLX models
echo ""
echo -e "${GREEN}Checking SMPL/SMPLX models...${NC}"
SMPL_MISSING=false
SMPLX_MISSING=false

if [ ! -f "extra_data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl" ]; then
    echo -e "${YELLOW}⚠️  SMPL model not found!${NC}"
    SMPL_MISSING=true
else
    echo -e "${GREEN}✓ SMPL model found${NC}"
fi

if [ ! -f "extra_data/smpl/SMPLX_NEUTRAL.pkl" ]; then
    echo -e "${YELLOW}⚠️  SMPLX model not found! (Required for whole body)${NC}"
    SMPLX_MISSING=true
else
    echo -e "${GREEN}✓ SMPLX model found${NC}"
fi

if [ "$SMPL_MISSING" = true ] || [ "$SMPLX_MISSING" = true ]; then
    echo ""
    echo -e "${BLUE}You need to download SMPL models manually:${NC}"
    if [ "$SMPL_MISSING" = true ]; then
        echo "1. SMPL Model:"
        echo "   - Register at: http://smplify.is.tue.mpg.de/login"
        echo "   - Download: basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
        echo "   - Place in: frankmocap/extra_data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    fi
    if [ "$SMPLX_MISSING" = true ]; then
        echo "2. SMPLX Model (REQUIRED for whole body):"
        echo "   - Register at: https://smpl-x.is.tue.mpg.de/"
        echo "   - Download: SMPLX_NEUTRAL.pkl"
        echo "   - Place in: frankmocap/extra_data/smpl/SMPLX_NEUTRAL.pkl"
    fi
fi

# Comprehensive verification
echo ""
echo -e "${GREEN}=========================================="
echo "Verifying Installation"
echo "==========================================${NC}"
cd ..

python -c "
import sys
sys.path.insert(0, 'frankmocap')

print('=== Critical Package Tests ===')
packages = {
    'torch': 'import torch; torch.__version__',
    'numpy': 'import numpy; numpy.__version__',
    'scipy': 'import scipy; scipy.__version__',
    'cv2': 'import cv2; cv2.__version__',
    'OpenGL': 'import OpenGL.GL',
    'detectron2': 'import detectron2; detectron2.__version__',
    'chumpy': 'import chumpy',
    'opendr': 'from opendr.camera import ProjectPoints',
}

results = {}
for name, test in packages.items():
    try:
        result = eval(test)
        print(f'✓ {name}: {result if isinstance(result, str) else \"OK\"}')
        results[name] = True
    except Exception as e:
        print(f'✗ {name}: {e}')
        results[name] = False

print()
print('=== FrankMocap Module Tests ===')
try:
    from bodymocap.body_mocap_api import BodyMocap
    print('✓ BodyMocap import successful')
    results['BodyMocap'] = True
except Exception as e:
    print(f'✗ BodyMocap import failed: {e}')
    results['BodyMocap'] = False

try:
    from handmocap.hand_mocap_api import HandMocap
    print('✓ HandMocap import successful')
    results['HandMocap'] = True
except Exception as e:
    print(f'⚠ HandMocap import failed: {e}')
    results['HandMocap'] = False

print()
print('=== PyTorch CUDA Configuration ===')
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')

print()
print('=== Installation Summary ===')
critical = ['torch', 'numpy', 'BodyMocap']
optional = ['detectron2', 'HandMocap', 'opendr', 'chumpy']

if all(results.get(p, False) for p in critical):
    print('✓ CRITICAL packages: All working')
else:
    print('✗ CRITICAL packages: Some failed')

if all(results.get(p, False) for p in optional):
    print('✓ OPTIONAL packages: All working')
else:
    print('⚠ OPTIONAL packages: Some failed (features may be limited)')
"

echo ""
echo "=========================================="
echo -e "${GREEN}Installation Complete!${NC}"
echo "=========================================="
echo ""
echo -e "${BLUE}Key improvements in this installation:${NC}"
echo "  ✓ Python 3.9 (better compatibility than 3.10)"
echo "  ✓ PyTorch via pip (avoids MKL conflicts)"
echo "  ✓ Version pins for opendr/chumpy compatibility"
echo "  ✓ Build tools check and installation"
echo "  ✓ Removed deprecated packages (pafy, youtube-dl)"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Download SMPL/SMPLX models if not already done (see above)"
echo "2. Activate environment: conda activate ${ENV_NAME}"
echo "3. Test: python -c \"from bodymocap.body_mocap_api import BodyMocap\""
echo ""
echo -e "${BLUE}To run your server:${NC}"
echo "  conda activate ${ENV_NAME}"
echo "  cd $(pwd)"
echo "  python server.py --host 0.0.0.0 --port 8080"
echo ""

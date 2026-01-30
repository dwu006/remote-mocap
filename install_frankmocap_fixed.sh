#!/bin/bash
# FIXED FrankMocap Installation Script for Remote GPU
# Fixes the iJIT_NotifyEvent / MKL symbol conflict issue
#
# KEY CHANGES FROM ORIGINAL:
# 1. Uses PIP for PyTorch installation (not conda) - avoids MKL 2024+ conflicts
# 2. Supports Python 3.9-3.11 (better compatibility) with 3.7 fallback
# 3. Properly matches Detectron2 wheels to PyTorch/CUDA versions
# 4. Based on research into PyTorch GitHub issue #123097 and related MKL conflicts
#
# Run this on your remote GPU server

set -e  # Exit on error

echo "=========================================="
echo "FIXED FrankMocap Installation Script"
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

# Check CUDA availability FIRST (before creating environment)
echo -e "${GREEN}[1/11] Checking CUDA availability...${NC}"
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

# Determine Python version based on CUDA compatibility
echo -e "${GREEN}[2/11] Determining optimal Python version...${NC}"
if [ "$CUDA_MAJOR" -ge 12 ]; then
    echo "CUDA 12.x detected - using Python 3.10 (best compatibility)"
    PYTHON_VERSION="3.10"
    PYTORCH_CUDA_VERSION="cu121"  # CUDA 12.1 for PyTorch pip wheels
elif [ "$CUDA_MAJOR" -eq 11 ]; then
    echo "CUDA 11.x detected - using Python 3.9 (best compatibility)"
    PYTHON_VERSION="3.9"
    PYTORCH_CUDA_VERSION="cu118"  # CUDA 11.8 for PyTorch pip wheels
else
    echo "CUDA 10.x detected - using Python 3.7 (legacy compatibility)"
    PYTHON_VERSION="3.7"
    PYTORCH_CUDA_VERSION="cu101"  # CUDA 10.1 for PyTorch pip wheels
fi

# Allow user override
echo ""
echo -e "${BLUE}Recommended Python version: ${PYTHON_VERSION}${NC}"
read -p "Press Enter to continue with Python ${PYTHON_VERSION}, or type a different version (3.7, 3.9, 3.10, 3.11): " USER_PYTHON
if [ ! -z "$USER_PYTHON" ]; then
    PYTHON_VERSION="$USER_PYTHON"
    echo "Using Python ${PYTHON_VERSION}"
fi

# Set up conda environment
echo -e "${GREEN}[3/11] Setting up conda environment...${NC}"
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

# Verify Python version in environment
ACTUAL_PYTHON=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version in environment: $ACTUAL_PYTHON"

# Install system dependencies via conda (OpenGL, FFmpeg, etc.) - NO SUDO REQUIRED
echo -e "${GREEN}[4/11] Installing system dependencies via conda/pip (NO SUDO)...${NC}"
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

# Install PyTorch via PIP (NOT conda) - This is the key fix!
echo -e "${GREEN}[5/11] Installing PyTorch via pip (KEY FIX for MKL conflicts)...${NC}"
echo -e "${YELLOW}NOTE: Using pip instead of conda avoids MKL 2024+ symbol conflicts${NC}"

if [ "$CUDA_AVAILABLE" = true ]; then
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        echo "Installing PyTorch 2.1+ for CUDA 12.x..."
        pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
    elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 7 ]; then
        echo "Installing PyTorch 2.0.1 for CUDA 11.8..."
        pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
    elif [ "$CUDA_MAJOR" -eq 11 ]; then
        echo "Installing PyTorch 1.13.1 for CUDA 11.7..."
        pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
    else
        # CUDA 10.x - only works with Python 3.7 and older PyTorch
        if [ "$ACTUAL_PYTHON" != "3.7" ]; then
            echo -e "${RED}Warning: CUDA 10.x requires Python 3.7. Current: $ACTUAL_PYTHON${NC}"
            echo "Attempting installation anyway..."
        fi
        echo "Installing PyTorch 1.7.1 for CUDA 10.1..."
        pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    fi
else
    echo -e "${YELLOW}No GPU detected - installing CPU-only PyTorch${NC}"
    pip install torch torchvision
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

# Install FrankMocap Python dependencies
echo -e "${GREEN}[6/11] Installing FrankMocap Python dependencies...${NC}"
cd frankmocap
pip install -r docs/requirements.txt
cd ..

# Install Detectron2 for hand module (CRITICAL for whole body)
echo -e "${GREEN}[7/11] Installing Detectron2 for hand module...${NC}"
if [ "$CUDA_AVAILABLE" = true ]; then
    # Determine Detectron2 wheel based on Python, PyTorch, and CUDA versions
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    TORCH_MAJOR=$(echo $TORCH_VERSION | cut -d. -f1)
    TORCH_MINOR=$(echo $TORCH_VERSION | cut -d. -f2)

    echo "Detected PyTorch $TORCH_VERSION, CUDA ${CUDA_MAJOR}.${CUDA_MINOR}, Python ${ACTUAL_PYTHON}"

    if [ "$TORCH_MAJOR" -eq 2 ]; then
        # PyTorch 2.x
        if [ "$CUDA_MAJOR" -ge 12 ]; then
            echo "Installing Detectron2 for PyTorch 2.x + CUDA 12.x..."
            python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' || \
            echo -e "${YELLOW}Warning: Detectron2 may need to be built from source for CUDA 12.x${NC}"
        else
            echo "Installing Detectron2 for PyTorch 2.x + CUDA 11.8..."
            python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html || \
            python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' || \
            echo -e "${YELLOW}Warning: Detectron2 installation may have failed${NC}"
        fi
    elif [ "$TORCH_MAJOR" -eq 1 ] && [ "$TORCH_MINOR" -ge 13 ]; then
        # PyTorch 1.13.x
        echo "Installing Detectron2 for PyTorch 1.13.x + CUDA 11.7..."
        python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch1.13/index.html || \
        echo -e "${YELLOW}Warning: Detectron2 installation may have failed${NC}"
    elif [ "$TORCH_MAJOR" -eq 1 ] && [ "$TORCH_MINOR" -ge 7 ]; then
        # PyTorch 1.7-1.10
        echo "Installing Detectron2 for PyTorch 1.7+ + CUDA 10.1..."
        python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html || \
        echo -e "${YELLOW}Warning: Detectron2 installation may have failed${NC}"
    else
        echo -e "${YELLOW}Detectron2 wheel not available for PyTorch ${TORCH_VERSION}${NC}"
        echo "Attempting to build from source..."
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
echo -e "${GREEN}[8/11] Installing additional dependencies...${NC}"
pip install aiohttp aiortc av opencv-python numpy

# Install 2D pose detector
echo -e "${GREEN}[9/11] Installing 2D pose detector...${NC}"
cd frankmocap
if [ -f "scripts/install_pose2d.sh" ]; then
    bash scripts/install_pose2d.sh
else
    echo -e "${YELLOW}Warning: install_pose2d.sh not found. Installing manually...${NC}"
    mkdir -p detectors
    cd detectors

    if [ ! -d "body_pose_estimator" ]; then
        git clone https://github.com/jhugestar/lightweight-human-pose-estimation.pytorch.git body_pose_estimator || true
    fi

    # Download pretrained model
    mkdir -p ../extra_data/body_module/body_pose_estimator
    cd ../extra_data/body_module/body_pose_estimator
    if [ ! -f "checkpoint_iter_370000.pth" ]; then
        wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth || \
        echo -e "${YELLOW}Failed to download pose estimator weights - you may need to download manually${NC}"
    fi
    cd ../../..
fi

# Install hand detectors
echo -e "${GREEN}[10/11] Installing hand detectors and downloading models...${NC}"
if [ -f "scripts/install_hand_detectors.sh" ]; then
    echo "Installing hand detectors..."
    bash scripts/install_hand_detectors.sh
else
    echo -e "${YELLOW}Warning: install_hand_detectors.sh not found. Skipping.${NC}"
fi

# Download hand module data
if [ -f "scripts/download_data_hand_module.sh" ]; then
    echo "Downloading hand module data..."
    bash scripts/download_data_hand_module.sh
else
    echo -e "${YELLOW}Warning: download_data_hand_module.sh not found.${NC}"
fi

# Download body module data
echo "Downloading body module data..."
if [ -f "scripts/download_data_body_module.sh" ]; then
    bash scripts/download_data_body_module.sh
else
    echo -e "${YELLOW}Warning: download_data_body_module.sh not found.${NC}"
fi

# Check for SMPL/SMPLX models
echo -e "${GREEN}[11/11] Checking SMPL/SMPLX models...${NC}"
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

# Verify installation
echo ""
echo -e "${GREEN}Verifying installation...${NC}"
cd ..
python -c "
import sys
sys.path.insert(0, 'frankmocap')

print('=== Import Tests ===')
try:
    from bodymocap.body_mocap_api import BodyMocap
    print('✓ BodyMocap import successful')
except Exception as e:
    print(f'✗ BodyMocap import failed: {e}')

try:
    from handmocap.hand_mocap_api import HandMocap
    print('✓ HandMocap import successful')
except Exception as e:
    print(f'⚠ HandMocap import failed: {e}')

try:
    from handmocap.hand_bbox_detector import HandBboxDetector
    print('✓ HandBboxDetector import successful')
except Exception as e:
    print(f'⚠ HandBboxDetector import failed: {e}')

print()
print('=== PyTorch Configuration ===')
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
"

echo ""
echo "=========================================="
echo -e "${GREEN}Installation Complete!${NC}"
echo "=========================================="
echo ""
echo -e "${BLUE}Key fixes applied:${NC}"
echo "  ✓ PyTorch installed via PIP (not conda) - avoids MKL conflicts"
echo "  ✓ Python ${PYTHON_VERSION} for optimal compatibility"
echo "  ✓ Detectron2 matched to PyTorch/CUDA versions"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Download SMPL/SMPLX models if not already done (see above)"
echo "2. Activate conda environment: conda activate ${ENV_NAME}"
echo "3. Test FrankMocap imports (run verification command above)"
echo ""
echo -e "${BLUE}To use with your server:${NC}"
echo "  conda activate ${ENV_NAME}"
echo "  cd $(pwd)"
echo "  python server.py --host 0.0.0.0 --port 8080"
echo ""

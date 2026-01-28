#!/bin/bash
# Automated installation script for FrankMocap (Whole Body Module)
# Optimized for real-time WebRTC mocap server
# Run this on your remote GPU server
# Uses conda for environment management

set -e  # Exit on error

echo "=========================================="
echo "FrankMocap Installation Script"
echo "For Real-Time WebRTC Mocap Server"
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

cd frankmocap

# Check Python version
echo -e "${GREEN}[1/10] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $PYTHON_VERSION"
if [ "$(printf '%s\n' "3.7" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.7" ]; then
    echo -e "${YELLOW}Warning: Python 3.7+ recommended. Current: $PYTHON_VERSION${NC}"
fi

# Check CUDA availability
echo -e "${GREEN}[2/10] Checking CUDA availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader | head -n1
    CUDA_AVAILABLE=true
else
    echo -e "${YELLOW}Warning: nvidia-smi not found. CUDA may not be available.${NC}"
    CUDA_AVAILABLE=false
fi

# Detect CUDA version
if [ "$CUDA_AVAILABLE" = true ]; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "CUDA Version: $CUDA_VERSION"
else
    CUDA_VERSION="11.8"  # Default
    echo "Using default CUDA version: $CUDA_VERSION"
fi

# Install system dependencies
echo -e "${GREEN}[3/10] Installing system dependencies...${NC}"
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y \
        libglu1-mesa \
        libxi-dev \
        libxmu-dev \
        libglu1-mesa-dev \
        freeglut3-dev \
        libosmesa6-dev \
        ffmpeg \
        wget \
        git
    echo "System dependencies installed."
else
    echo -e "${YELLOW}Warning: apt-get not found. Please install dependencies manually.${NC}"
fi

# Set up conda environment
echo -e "${GREEN}[4/10] Setting up conda environment...${NC}"
ENV_NAME="venv_frankmocap"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Conda environment '${ENV_NAME}' already exists.${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
        echo "Creating new conda environment..."
        conda create -n ${ENV_NAME} python=3.7 -y
    else
        echo "Using existing environment."
    fi
else
    echo "Creating conda environment '${ENV_NAME}' with Python 3.7..."
    conda create -n ${ENV_NAME} python=3.7 -y
fi

echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}
echo -e "${GREEN}✓ Conda environment activated${NC}"

# Install CUDA toolkit via conda (if needed)
echo -e "${GREEN}[5/10] Installing CUDA toolkit...${NC}"
if [ "$CUDA_AVAILABLE" = true ]; then
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    
    if [ "$CUDA_MAJOR" -ge 11 ]; then
        echo "Installing CUDA 11.x toolkit via conda..."
        conda install -c conda-forge cudatoolkit=11.8 cudnn -y || echo "CUDA toolkit installation skipped (may already be installed)"
    else
        echo "Installing CUDA 10.1 toolkit via conda..."
        conda install cudatoolkit=10.1 cudnn=7.6.0 -y || echo "CUDA toolkit installation skipped (may already be installed)"
    fi
else
    echo -e "${YELLOW}Skipping CUDA toolkit (no GPU detected)${NC}"
fi

# Install PyTorch with CUDA
echo -e "${GREEN}[6/10] Installing PyTorch with CUDA support...${NC}"
# Determine PyTorch installation command based on CUDA version
if [ "$CUDA_AVAILABLE" = true ]; then
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        echo "Installing PyTorch for CUDA 12.1..."
        conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
    elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        echo "Installing PyTorch for CUDA 11.8..."
        conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
    elif [ "$CUDA_MAJOR" -eq 11 ]; then
        echo "Installing PyTorch for CUDA 11.7..."
        conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia -y
    else
        echo "Installing PyTorch for CUDA 10.1 (legacy)..."
        conda install -c pytorch pytorch==1.6.0 torchvision cudatoolkit=10.1 -y
    fi
else
    echo "Installing PyTorch CPU version..."
    conda install pytorch torchvision cpuonly -c pytorch -y
fi

# Verify PyTorch CUDA
echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Install FrankMocap Python dependencies
echo -e "${GREEN}[7/10] Installing FrankMocap Python dependencies...${NC}"
pip install -r docs/requirements.txt

# Install Detectron2 for hand module (required for whole body)
echo -e "${GREEN}[8/10] Installing Detectron2 for hand module...${NC}"
if [ "$CUDA_AVAILABLE" = true ]; then
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    if [ "$CUDA_MAJOR" -ge 11 ]; then
        echo "Installing Detectron2 for CUDA 11.x..."
        python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html || \
        python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch2.0/index.html || \
        echo -e "${YELLOW}Warning: Detectron2 installation may have failed. Try manual installation.${NC}"
    else
        echo "Installing Detectron2 for CUDA 10.1..."
        python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html || \
        echo -e "${YELLOW}Warning: Detectron2 installation may have failed. Try manual installation.${NC}"
    fi
else
    echo -e "${YELLOW}Skipping Detectron2 (no GPU detected)${NC}"
fi

# Install additional dependencies for real-time server
echo "Installing additional dependencies for WebRTC server..."
pip install aiohttp aiortc av opencv-python numpy

# Install 2D pose detector
echo -e "${GREEN}[9/10] Installing 2D pose detector...${NC}"
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
        wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth
    fi
    cd ../../..
fi

# Install hand detectors and download all data
echo -e "${GREEN}[10/10] Installing hand detectors and downloading all models/data...${NC}"

# Install hand detectors
if [ -f "scripts/install_hand_detectors.sh" ]; then
    echo "Installing hand detectors..."
    bash scripts/install_hand_detectors.sh
else
    echo -e "${YELLOW}Warning: install_hand_detectors.sh not found. Skipping hand detectors.${NC}"
fi

# Download hand module data
if [ -f "scripts/download_data_hand_module.sh" ]; then
    echo "Downloading hand module data..."
    bash scripts/download_data_hand_module.sh
else
    echo -e "${YELLOW}Warning: download_data_hand_module.sh not found. Skipping hand module data.${NC}"
fi

# Download pretrained models and data for body module
echo "Downloading body module data..."
if [ -f "scripts/download_data_body_module.sh" ]; then
    bash scripts/download_data_body_module.sh
else
    echo -e "${YELLOW}Warning: download_data_body_module.sh not found. Downloading manually...${NC}"
    mkdir -p extra_data/body_module
    cd extra_data/body_module
    
    # Download SPIN data
    if [ ! -d "data_from_spin" ]; then
        wget http://visiondata.cis.upenn.edu/spin/data.tar.gz && tar -xvf data.tar.gz && rm data.tar.gz
        mv data data_from_spin
    fi
    
    # Download pretrained weights
    mkdir -p pretrained_weights
    cd pretrained_weights
    if [ ! -f "2020_05_31-00_50_43-best-51.749683916568756.pt" ]; then
        wget https://dl.fbaipublicfiles.com/eft/2020_05_31-00_50_43-best-51.749683916568756.pt
    fi
    if [ ! -f "smplx-03-28-46060-w_spin_mlc3d_46582-2089_2020_03_28-21_56_16.pt" ]; then
        wget https://dl.fbaipublicfiles.com/eft/fairmocap_data/body_module/smplx-03-28-46060-w_spin_mlc3d_46582-2089_2020_03_28-21_56_16.pt
    fi
    cd ..
    
    # Download J_regressor
    if [ ! -f "J_regressor_extra_smplx.npy" ]; then
        wget https://dl.fbaipublicfiles.com/eft/fairmocap_data/body_module/J_regressor_extra_smplx.npy
    fi
    
    cd ../..
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

# Verify installation
echo ""
echo -e "${GREEN}Verifying installation...${NC}"
cd ..
python3 -c "
import sys
sys.path.insert(0, 'frankmocap')
try:
    from bodymocap.body_mocap_api import BodyMocap
    print('✓ BodyMocap import successful')
except Exception as e:
    print(f'✗ BodyMocap import failed: {e}')

try:
    from handmocap.hand_mocap_api import HandMocap
    print('✓ HandMocap import successful')
except Exception as e:
    print(f'⚠ HandMocap import failed (may need Detectron2): {e}')

try:
    from handmocap.hand_bbox_detector import HandBboxDetector
    print('✓ HandBboxDetector import successful')
except Exception as e:
    print(f'⚠ HandBboxDetector import failed: {e}')

import torch
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=========================================="
echo -e "${GREEN}Installation Complete!${NC}"
echo "=========================================="
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Download SMPL/SMPLX models if not already done (see above)"
echo "2. Activate conda environment: conda activate ${ENV_NAME}"
echo "3. Test server: python server.py --host 0.0.0.0 --port 8080"
echo ""
echo -e "${BLUE}Real-time optimizations enabled:${NC}"
echo "  - AMP (FP16) support"
echo "  - Reduced HMR iterations (n_iter=1)"
echo "  - Bbox tracking"
echo "  - Whole body support (body + hands)"
echo ""
echo -e "${BLUE}To use the server:${NC}"
echo "  conda activate ${ENV_NAME}"
echo "  cd $(pwd)"
echo "  python server.py --host 0.0.0.0 --port 8080"
echo ""

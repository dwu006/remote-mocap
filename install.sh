#!/bin/bash
# Unified Installation Script for Remote Mocap System
# Combines FrankMocap + WebRTC server setup
# Optimized for GPU servers with cross-platform compatibility

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration (matching FrankMocap official requirements)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRANKMOCAP_DIR="${SCRIPT_DIR}/frankmocap"
CONDA_ENV_NAME="frankmocap"  # User requested name
PYTHON_VERSION="3.7"  # FrankMocap requires Python 3.7
PYTORCH_VERSION="1.6.0"  # FrankMocap requires PyTorch 1.6.0
CUDA_VERSION_OFFICIAL="10.1"  # FrankMocap uses CUDA 10.1

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Remote Mocap Installation Script${NC}"
echo -e "${CYAN}FrankMocap + WebRTC Server${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}[STATUS]${NC} ${message}"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python() {
    print_status $BLUE "Checking Python version..."
    
    if ! command_exists python3; then
        print_error "Python 3 not found! Please install Python 3.7"
        exit 1
    fi
    
    local python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python version: $python_version"
    
    # Check if version matches FrankMocap requirement
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)" 2>/dev/null; then
        if python3 -c "import sys; exit(1 if sys.version_info >= (3, 8) else 0)" 2>/dev/null; then
            print_success "Python version meets FrankMocap requirements (3.7.x)"
        else
            print_warning "FrankMocap requires Python 3.7, you have $python_version. This may cause compatibility issues."
        fi
    else
        print_error "Python 3.7+ required. Current: $python_version"
        exit 1
    fi
}

# Function to check GPU/CUDA availability
check_gpu() {
    print_status $BLUE "Checking GPU availability..."
    
    if command_exists nvidia-smi; then
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        local cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
        print_success "GPU detected: $gpu_name"
        print_success "CUDA version: $cuda_version"
        echo "CUDA_AVAILABLE=true" >> "${SCRIPT_DIR}/install_env.sh"
        echo "CUDA_VERSION=$cuda_version" >> "${SCRIPT_DIR}/install_env.sh"
        return 0
    else
        print_warning "No NVIDIA GPU detected. Will use CPU (slower performance)"
        echo "CUDA_AVAILABLE=false" >> "${SCRIPT_DIR}/install_env.sh"
        return 1
    fi
}

# Function to check conda
check_conda() {
    print_status $BLUE "Checking conda installation..."
    
    if ! command_exists conda; then
        print_error "conda not found! Please install Miniconda or Anaconda"
        echo "Install from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    print_success "conda found: $(conda --version)"
}

# Function to setup conda environment
setup_conda_env() {
    print_status $BLUE "Setting up conda environment..."
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        print_warning "Conda environment '${CONDA_ENV_NAME}' already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status $YELLOW "Removing existing environment..."
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            print_success "Using existing environment"
            return 0
        fi
    fi
    
    print_status $BLUE "Creating conda environment '${CONDA_ENV_NAME}'..."
    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
    
    print_success "Conda environment created"
}

# Function to activate conda environment
activate_conda_env() {
    print_status $BLUE "Activating conda environment..."
    
    # Source conda initialization
    if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        print_error "conda initialization script not found"
        exit 1
    fi
    
    conda activate ${CONDA_ENV_NAME}
    print_success "Conda environment activated"
}

# Function to install PyTorch (FrankMocap official versions)
install_pytorch() {
    print_status $BLUE "Installing PyTorch ${PYTORCH_VERSION} (FrankMocap official)..."
    
    # Install CUDA toolkit first if GPU available
    if [ "$CUDA_AVAILABLE" = "true" ]; then
        print_status $BLUE "Installing CUDA toolkit ${CUDA_VERSION_OFFICIAL}..."
        conda install cudatoolkit=${CUDA_VERSION_OFFICIAL} cudnn=7.6.0 -y
    fi
    
    # Install PyTorch and torchvision with official FrankMocap versions
    if [ "$CUDA_AVAILABLE" = "true" ]; then
        print_status $BLUE "Installing PyTorch ${PYTORCH_VERSION} with CUDA ${CUDA_VERSION_OFFICIAL}..."
        conda install -c pytorch pytorch==${PYTORCH_VERSION} torchvision cudatoolkit=${CUDA_VERSION_OFFICIAL} -y
    else
        print_status $BLUE "Installing PyTorch ${PYTORCH_VERSION} CPU version..."
        conda install -c pytorch pytorch==${PYTORCH_VERSION} torchvision cpuonly -y
    fi
    
    # Verify PyTorch installation
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('Using CPU mode')
"
    
    print_success "PyTorch ${PYTORCH_VERSION} installed successfully"
}

# Function to clone/update FrankMocap
setup_frankmocap() {
    print_status $BLUE "Setting up FrankMocap..."
    
    if [ ! -d "$FRANKMOCAP_DIR" ]; then
        print_status $BLUE "Cloning FrankMocap repository..."
        git clone https://github.com/facebookresearch/frankmocap.git "$FRANKMOCAP_DIR"
        print_success "FrankMocap cloned"
    else
        print_success "FrankMocap directory already exists"
    fi
    
    cd "$FRANKMOCAP_DIR"
    
    # Install FrankMocap dependencies
    print_status $BLUE "Installing FrankMocap dependencies..."
    if [ -f "docs/requirements.txt" ]; then
        pip install -r docs/requirements.txt
        print_success "FrankMocap dependencies installed"
    else
        print_warning "docs/requirements.txt not found, installing manually..."
        pip install smplx trimesh pyrender chumpy
    fi
}

# Function to install Detectron2 (for hand module) - FrankMocap official version
install_detectron2() {
    print_status $BLUE "Installing Detectron2 (for hand module)..."
    
    if [ "$CUDA_AVAILABLE" = "true" ]; then
        print_status $BLUE "Installing Detectron2 for PyTorch ${PYTORCH_VERSION} + CUDA ${CUDA_VERSION_OFFICIAL}..."
        # Use official FrankMocap Detectron2 installation command
        python -m pip install detectron2 -f \
            https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html || \
            print_warning "Detectron2 installation failed. You may need to install manually following Detectron2 docs."
    else
        print_warning "Skipping Detectron2 (no GPU detected). Hand module will not be available."
    fi
}

# Function to install system dependencies (FrankMocap official)
install_system_deps() {
    print_status $BLUE "Installing system dependencies..."
    
    if command_exists apt-get; then
        print_status $BLUE "Installing OpenGL and system dependencies..."
        sudo apt-get update
        sudo apt-get install -y \
            libglu1-mesa \
            libxi-dev \
            libxmu-dev \
            libglu1-mesa-dev \
            freeglut3-dev \
            libosmesa6-dev \
            ffmpeg
        print_success "System dependencies installed"
    else
        print_warning "apt-get not found. Please install system dependencies manually:"
        echo "  - libglu1-mesa, libxi-dev, libxmu-dev, libglu1-mesa-dev"
        echo "  - freeglut3-dev, libosmesa6-dev, ffmpeg"
    fi
}

# Function to install WebRTC dependencies
install_webrtc_deps() {
    print_status $BLUE "Installing WebRTC server dependencies..."
    
    cd "$SCRIPT_DIR"
    
    # Install from requirements.txt
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "WebRTC dependencies installed from requirements.txt"
    else
        print_warning "requirements.txt not found, installing manually..."
        pip install aiohttp aiortc av opencv-python numpy python-dotenv
    fi
}

# Function to install optional PyTorch3D (FrankMocap optional)
install_pytorch3d() {
    print_status $BLUE "Installing PyTorch3D (optional, for alternative rendering)..."
    
    # Try to install PyTorch3D (optional, may fail)
    pip install pytorch3d || print_warning "PyTorch3D installation failed (optional). You can skip this."
}

# Function to download FrankMocap models and data (official FrankMocap script)
download_frankmocap_data() {
    print_status $BLUE "Downloading FrankMocap models and data..."
    
    cd "$FRANKMOCAP_DIR"
    
    # Run the official FrankMocap installation script
    if [ -f "scripts/install_frankmocap.sh" ]; then
        print_status $BLUE "Running official FrankMocap installation script..."
        bash scripts/install_frankmocap.sh
        print_success "FrankMocap official script completed"
    else
        print_warning "Official install script not found, installing manually..."
        
        # Install 2D pose detector
        if [ -f "scripts/install_pose2d.sh" ]; then
            print_status $BLUE "Installing 2D pose detector..."
            bash scripts/install_pose2d.sh
        fi
        
        # Install hand detectors
        if [ -f "scripts/install_hand_detectors.sh" ]; then
            print_status $BLUE "Installing hand detectors..."
            bash scripts/install_hand_detectors.sh
        fi
        
        # Download body module data
        if [ -f "scripts/download_data_body_module.sh" ]; then
            print_status $BLUE "Downloading body module data..."
            bash scripts/download_data_body_module.sh
        fi
        
        # Download hand module data
        if [ -f "scripts/download_data_hand_module.sh" ]; then
            print_status $BLUE "Downloading hand module data..."
            bash scripts/download_data_hand_module.sh
        fi
        
        print_success "FrankMocap data download completed"
    fi
}

# Function to check SMPL models (FrankMocap official paths)
check_smpl_models() {
    print_status $BLUE "Checking SMPL models..."
    
    # FrankMocap expects models in extra_data/smpl/, but we have them in smpl/
    local smpl_dir_extra="${FRANKMOCAP_DIR}/extra_data/smpl"
    local smpl_dir_current="${FRANKMOCAP_DIR}/smpl"
    local smpl_found=true
    
    # Check if models are in current location (smpl/)
    if [ -d "$smpl_dir_current" ]; then
        print_status $BLUE "Found models in: $smpl_dir_current"
        
        if [ ! -f "$smpl_dir_current/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl" ]; then
            print_warning "SMPL model not found: basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
            smpl_found=false
        else
            print_success "SMPL model found"
        fi
        
        if [ ! -f "$smpl_dir_current/SMPLX_NEUTRAL.pkl" ]; then
            print_warning "SMPLX model not found: SMPLX_NEUTRAL.pkl"
            smpl_found=false
        else
            print_success "SMPLX model found"
        fi
        
        # Create symlink or copy to expected location if needed
        if [ ! -d "$smpl_dir_extra" ]; then
            mkdir -p "$smpl_dir_extra"
            print_status $BLUE "Creating expected directory: $smpl_dir_extra"
        fi
        
        # Copy models to expected FrankMocap location
        if [ -f "$smpl_dir_current/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl" ] && [ ! -f "$smpl_dir_extra/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl" ]; then
            cp "$smpl_dir_current/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl" "$smpl_dir_extra/"
            print_success "SMPL model copied to expected location"
        fi
        
        if [ -f "$smpl_dir_current/SMPLX_NEUTRAL.pkl" ] && [ ! -f "$smpl_dir_extra/SMPLX_NEUTRAL.pkl" ]; then
            cp "$smpl_dir_current/SMPLX_NEUTRAL.pkl" "$smpl_dir_extra/"
            print_success "SMPLX model copied to expected location"
        fi
    else
        print_warning "SMPL directory not found: $smpl_dir_current"
        smpl_found=false
    fi
    
    if [ "$smpl_found" = "true" ]; then
        print_success "All SMPL models found and properly placed!"
        return 0
    else
        print_warning "Some SMPL models missing. See manual download instructions below."
        return 1
    fi
}

# Function to verify installation
verify_installation() {
    print_status $BLUE "Verifying installation..."
    
    cd "$SCRIPT_DIR"
    
    python3 -c "
import sys
import os
sys.path.insert(0, 'frankmocap')

print('Testing imports...')

try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'✗ PyTorch import failed: {e}')

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

try:
    import aiohttp
    import aiortc
    print('✓ WebRTC dependencies available')
except Exception as e:
    print(f'✗ WebRTC dependencies failed: {e}')

print('\\nInstallation verification complete!')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation verification passed!"
        return 0
    else
        print_error "Installation verification failed!"
        return 1
    fi
}

# Function to show next steps
show_next_steps() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${GREEN}Installation Complete!${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo ""
    echo "1. Activate conda environment:"
    echo "   conda activate frankmocap"
    echo ""
    echo "2. Start the WebRTC server:"
    echo "   python server.py --host 0.0.0.0 --port 8080"
    echo ""
    echo "3. Open client.html in your browser:"
    echo "   - Update serverUrl in client.html if needed"
    echo "   - Allow camera permissions"
    echo "   - Click 'Start Capture'"
    echo ""
    
    if ! check_smpl_models; then
        echo ""
        echo -e "${YELLOW}⚠️  Manual SMPL model download required:${NC}"
        echo ""
        echo "SMPL Model (for body module):"
        echo "  - Register: http://smplify.is.tue.mpg.de/login"
        echo "  - Download: basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
        echo "  - Place in: frankmocap/smpl/"
        echo ""
        echo "SMPLX Model (for hand/whole body module):"
        echo "  - Register: https://smpl-x.is.tue.mpg.de/"
        echo "  - Download: SMPLX_NEUTRAL.pkl"
        echo "  - Place in: frankmocap/smpl/"
        echo ""
        echo "Note: Script will automatically copy models to expected FrankMocap locations"
        echo ""
    fi
    
    echo -e "${BLUE}Performance tips:${NC}"
    echo "  - Use GPU server for best performance"
    echo "  - Adjust resize_short in config.py for speed/accuracy tradeoff"
    echo "  - Set process_every_n=2 for lower GPU usage"
    echo ""
}

# Main installation function
main() {
    # Create environment file
    > "${SCRIPT_DIR}/install_env.sh"
    
    # Detect OS
    local os=$(detect_os)
    print_success "OS detected: $os"
    
    # Run installation steps (FrankMocap official order)
    check_python
    check_gpu
    check_conda
    setup_conda_env
    activate_conda_env
    install_pytorch
    install_system_deps
    setup_frankmocap
    install_detectron2
    install_pytorch3d
    install_webrtc_deps
    download_frankmocap_data
    check_smpl_models
    verify_installation
    
    # Clean up environment file
    if [ -f "${SCRIPT_DIR}/install_env.sh" ]; then
        rm "${SCRIPT_DIR}/install_env.sh"
    fi
    
    show_next_steps
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --verify-only  Skip installation, only verify existing setup"
        echo "  --smpl-check   Only check SMPL models"
        echo ""
        exit 0
        ;;
    --verify-only)
        activate_conda_env
        verify_installation
        exit $?
        ;;
    --smpl-check)
        check_smpl_models
        exit $?
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use --help for available options"
        exit 1
        ;;
esac
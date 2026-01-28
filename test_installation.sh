#!/bin/bash
# Quick test script to verify FrankMocap installation

echo "Testing FrankMocap installation..."

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate frankmocap

# Test imports
python3 -c "
import sys
sys.path.insert(0, 'frankmocap')

print('Testing FrankMocap imports...')

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
    import aiohttp
    import aiortc
    print('✓ WebRTC dependencies available')
except Exception as e:
    print(f'✗ WebRTC dependencies failed: {e}')

print('\\nTest completed!')
"

echo ""
echo "To run the server:"
echo "conda activate frankmocap"
echo "python server.py --host 0.0.0.0 --port 8080"
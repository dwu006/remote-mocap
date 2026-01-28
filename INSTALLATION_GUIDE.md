# Installation Guide - Where to Install What

## Quick Answer

**FrankMocap installation: ONLY on the GPU server** (where inference happens)
**Your laptop: Just needs a web browser** (no installation needed!)

## Architecture Overview

```
Your Laptop                    Internet              Remote GPU Server
┌─────────────┐                                  ┌──────────────────┐
│  Browser    │ ──── WebRTC Video ────>        │  server.py       │
│  (client)   │                                  │  + FrankMocap    │
│             │ <─── Pose JSON ─────             │  + CUDA          │
│  Webcam     │                                  │  + Models        │
└─────────────┘                                  └──────────────────┘
   NO INSTALL                                        FULL INSTALL
   NEEDED!                                           REQUIRED!
```

## Installation Locations

### ✅ Remote GPU Server (SSH into this)

**You MUST install everything here:**

1. **Python 3.9+** with virtual environment
2. **PyTorch with CUDA** support
3. **FrankMocap** (full installation)
4. **SMPL/SMPLX models** (download from official site)
5. **Pretrained weights** (download via scripts)
6. **All dependencies** (from requirements.txt)
7. **This project** (server.py, etc.)

**Why?** This is where the actual inference happens - the GPU runs FrankMocap models.

### ❌ Your Laptop (Local Computer)

**You DON'T need to install anything!**

- Just needs a **web browser** (Chrome, Firefox, Edge, Safari)
- Open `client.html` in the browser
- That's it!

**Why?** The browser handles everything - webcam capture, WebRTC connection, displaying results. No Python, no FrankMocap, no models needed.

## Step-by-Step Installation

### On Remote GPU Server (via SSH)

#### 1. Clone This Project

```bash
ssh user@your-remote-gpu-server
cd ~
git clone <your-remote-mocap-repo-url>
cd remote-mocap
```

#### 2. Clone FrankMocap

```bash
git clone https://github.com/facebookresearch/frankmocap.git
```

#### 3. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA (adjust for your CUDA version)
# Check your CUDA version first:
nvidia-smi

# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 4. Install FrankMocap Dependencies

```bash
cd frankmocap

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev freeglut3-dev libosmesa6-dev ffmpeg

# Install Python dependencies
pip install -r docs/requirements.txt

# Install body module dependencies
bash scripts/install_pose2d.sh

# Download pretrained models and data
bash scripts/download_data_body_module.sh
```

#### 5. Download SMPL Models

**Important:** You need to register and download from official sites:

1. **SMPL Model** (for body-only):
   - Register at: http://smplify.is.tue.mpg.de/login
   - Download: `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`
   - Place in: `frankmocap/extra_data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`

2. **SMPLX Model** (optional, for hands):
   - Register at: https://smpl-x.is.tue.mpg.de/
   - Download: `SMPLX_NEUTRAL.pkl`
   - Place in: `frankmocap/extra_data/smpl/SMPLX_NEUTRAL.pkl`

#### 6. Install This Project's Dependencies

```bash
cd ~/remote-mocap  # Back to project root
pip install -r requirements.txt
```

#### 7. Verify Installation

```bash
# Test FrankMocap import
python -c "import sys; sys.path.insert(0, 'frankmocap'); from bodymocap.body_mocap_api import BodyMocap; print('✓ FrankMocap OK')"

# Test server
python server.py --host 0.0.0.0 --port 8080
# Should see: "Starting server on 0.0.0.0:8080"
```

### On Your Laptop

**No installation needed!** Just:

1. **Download or clone this repo** (to get `client.html`)
   ```bash
   git clone <your-remote-mocap-repo-url>
   # Or just download client.html
   ```

2. **Open `client.html` in your browser**
   - Double-click the file, or
   - Right-click → Open with → Browser

3. **Update the server URL** in `client.html`:
   ```javascript
   const serverUrl = "http://YOUR_REMOTE_SERVER_IP:8080";
   ```

4. **Click "Start Capture"** - done!

## What Gets Installed Where

### Remote GPU Server Needs:

| Component | Why |
|-----------|-----|
| Python 3.9+ | Run server.py |
| PyTorch + CUDA | Run FrankMocap models on GPU |
| FrankMocap | The actual pose estimation |
| SMPL models | 3D body model |
| Pretrained weights | Trained neural networks |
| OpenCV, NumPy, etc. | Image processing |
| aiortc, aiohttp | WebRTC server |

### Your Laptop Needs:

| Component | Why |
|-----------|-----|
| Web browser | That's it! |
| Webcam | Built-in or USB |

## Common Questions

### Q: Do I need to install FrankMocap on my laptop?

**A: No!** Your laptop just runs a browser. The browser:
- Captures webcam (via WebRTC)
- Sends video to remote server
- Receives pose data back
- Displays results

No Python, no FrankMocap, no models needed on laptop.

### Q: Can I test locally (same machine)?

**A: Yes!** If you want to test on the same machine:
- Install everything on that machine
- Run `server.py` locally
- Open `http://localhost:8080/` in browser
- Works the same way!

### Q: What if I don't have a GPU server yet?

**A: You can:**
1. Use a cloud GPU service (Vast.ai, RunPod, AWS, etc.)
2. Use a local machine with GPU for testing
3. Use CPU (very slow, not real-time) - just for testing

### Q: Do I need Detectron2?

**A: Only if using hand module.** For body-only (which we use), you don't need Detectron2.

### Q: What Python version?

**A: Python 3.9+** (3.10 recommended). FrankMocap originally used 3.7, but 3.9+ works fine.

## Troubleshooting

### "FrankMocap not found" on server

- Make sure `frankmocap/` directory exists in project root
- Check Python path includes frankmocap: `python -c "import sys; sys.path.insert(0, 'frankmocap'); ..."`

### "CUDA not available"

- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU: `nvidia-smi`

### "SMPL model not found"

- Download from official site (requires registration)
- Place in: `frankmocap/extra_data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`

### Browser can't connect to server

- Check server is running: `ps aux | grep server.py`
- Check firewall: `sudo ufw allow 8080/tcp`
- Verify server IP: `curl http://YOUR_SERVER_IP:8080/`

## Summary

**Install on GPU server:**
- ✅ Everything (Python, PyTorch, FrankMocap, models, dependencies)

**Install on laptop:**
- ❌ Nothing! Just use a browser.

The browser does all the client-side work - no installation needed!

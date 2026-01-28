# WebRTC FrankMocap - Real-Time Motion Capture

A real-time motion capture system using WebRTC for low-latency video streaming and FrankMocap for 3D pose estimation. The system streams webcam video from a browser to a remote GPU server, runs FrankMocap inference, and sends 3D joint data back via WebRTC DataChannel.

## Features

- **Low-latency WebRTC streaming** - Direct peer-to-peer video streaming with minimal delay
- **GPU-accelerated inference** - Runs FrankMocap on CUDA-enabled GPUs (with CPU fallback)
- **Real-time 3D pose estimation** - Extracts 3D joint positions from video frames
- **Browser-based client** - No installation needed on client side, works in modern browsers
- **Compact JSON output** - Efficient data transfer with only essential pose data

## Architecture

```
Browser (Client)          WebRTC          Server (GPU)
┌─────────────┐                          ┌─────────────┐
│  Webcam     │ ────── Video ──────>     │  Receive    │
│  Capture    │                          │  Frames     │
│             │                          │             │
│  Display    │ <─── Pose JSON ────      │  FrankMocap │
│  Results    │                          │  Inference  │
└─────────────┘                          └─────────────┘
```

## Prerequisites

### Server Requirements

- **Python 3.9+** (3.10 recommended)
- **NVIDIA GPU** with CUDA support (recommended) or CPU (slower)
- **CUDA toolkit** (if using GPU)
- **PyTorch** with CUDA support (install from [pytorch.org](https://pytorch.org/))

### Client Requirements

- Modern web browser (Chrome, Firefox, Edge, Safari)
- Webcam access permissions

## Installation

**Important:** FrankMocap only needs to be installed on the **remote GPU server** (where inference happens). Your laptop just needs a web browser - no installation needed! See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for details.

### 1. Clone FrankMocap Repository

If you haven't already, clone the FrankMocap repository:

```bash
git clone https://github.com/facebookresearch/frankmocap.git
```

Or if you've already cloned it, make sure it's in the `frankmocap/` directory relative to this project.

### 2. Install FrankMocap Dependencies

Follow the [FrankMocap installation instructions](https://github.com/facebookresearch/frankmocap/blob/main/docs/INSTALL.md):

```bash
cd frankmocap
# Install dependencies as per FrankMocap README
# This typically includes:
# - Installing PyTorch with CUDA
# - Installing SMPL models
# - Downloading pretrained weights
```

### 3. Download FrankMocap Weights

Download the required model weights:

```bash
cd frankmocap
# Run the download script (check FrankMocap README for exact command)
# Typically something like:
# bash scripts/download_weights.sh
```

Make sure the following files exist:
- `frankmocap/extra_data/body_module/pretrained_weights/` (body model weights)
- `frankmocap/extra_data/hand_module/pretrained_weights/` (hand model weights)
- `frankmocap/extra_data/smpl/` (SMPL model files)

### 4. Install Python Dependencies

Install the required Python packages:

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA (adjust for your CUDA version)
# Visit https://pytorch.org/ for the correct command
# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 5. Verify Installation

Test that FrankMocap can be imported:

```bash
python -c "import sys; sys.path.insert(0, 'frankmocap'); from bodymocap.body_mocap_api import BodyMocap; print('FrankMocap import successful')"
```

## Deployment Scenarios

### Scenario 1: Remote GPU Server (Recommended)

**Use Case:** You have a GPU server in the cloud (Vast.ai, RunPod, AWS, etc.) and want to use it from your local browser.

**Setup:**
1. Deploy `server.py` on your remote GPU server
2. Update `client.html` to point to your remote server URL
3. Configure TURN server for NAT traversal
4. Access from your local browser

See [Remote GPU Deployment](#remote-gpu-deployment-wan) section below for detailed steps.

### Scenario 2: Local Testing

**Use Case:** Testing on the same machine (server and browser on same computer).

**Setup:**
1. Run `server.py` locally
2. Open `http://localhost:8080/` in browser
3. No TURN server needed

## Usage

### Starting the Server

Run the WebRTC server:

```bash
python server.py --host 0.0.0.0 --port 8080
```

Options:
- `--host`: Host to bind to (default: `0.0.0.0` for all interfaces)
- `--port`: Port to bind to (default: `8080`)

The server will:
1. Detect available device (CUDA GPU or CPU)
2. Load FrankMocap models
3. Start HTTP server for signaling
4. Serve the client HTML page at `http://localhost:8080/`

### Accessing the Client

**For Remote GPU Server (SSH setup):**
1. SSH into your remote GPU server
2. Start the server: `python server.py --host 0.0.0.0 --port 8080`
3. On your **local laptop**, open `client.html` in a browser
4. Update the `serverUrl` in `client.html` to your remote server's IP:
   ```javascript
   const serverUrl = "http://YOUR_REMOTE_SERVER_IP:8080";
   ```
5. Click "Start Capture" - your laptop's webcam will stream to the remote GPU
6. Pose data will appear in real-time

**For Local Testing:**
1. Open a web browser and navigate to `http://localhost:8080/`
2. Click "Start Capture" to begin webcam streaming
3. Allow camera permissions when prompted
4. Pose data will appear in real-time in the JSON display

### Local Testing

For local testing (same machine):
- Server: `http://localhost:8080/`
- No TURN server needed

### Remote GPU Deployment (WAN)

**This is the primary use case** - running the server on a remote GPU machine (cloud GPU, Vast.ai, RunPod, etc.) and connecting from your local browser.

#### Architecture for Remote GPU

```
Your Local Computer          Internet          Remote GPU Server
┌─────────────┐                              ┌─────────────┐
│  Browser    │ ──── WebRTC Video ────>     │  server.py  │
│  (client)   │                              │  + GPU      │
│             │ <─── Pose JSON ─────         │  + CUDA     │
│  Webcam     │                              │  + Models   │
└─────────────┘                              └─────────────┘
```

#### Step 1: Deploy Server on Remote GPU

**On your remote GPU server** (Vast.ai, RunPod, AWS, GCP, etc.):

1. **SSH into your remote GPU instance**
2. **Clone this repository and FrankMocap:**
   ```bash
   git clone <this-repo-url>
   cd remote-mocap
   git clone https://github.com/facebookresearch/frankmocap.git
   ```

3. **Install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # Adjust CUDA version
   pip install -r requirements.txt
   ```

4. **Set up FrankMocap** (download weights, install dependencies)

5. **Start the server:**
   ```bash
   python server.py --host 0.0.0.0 --port 8080
   ```
   The server will bind to `0.0.0.0` to accept connections from anywhere.

#### Step 2: Configure Client for Remote Server

**On your local computer**, update `client.html` to point to your remote server:

1. **Find your remote server's IP/domain:**
   - If using Vast.ai/RunPod: Check the instance's public IP
   - If using cloud provider: Use the public IP or domain name

2. **Update the fetch URL in `client.html`:**
   ```javascript
   // Change this line in client.html:
   const resp = await fetch("http://YOUR_REMOTE_SERVER_IP:8080/offer", {
   // Or use a domain:
   const resp = await fetch("https://your-server.com:8080/offer", {
   ```

3. **For production, use HTTPS** (required for camera access):
   - Set up nginx reverse proxy with SSL
   - Or use aiohttp with SSL certificates

#### Step 3: Configure TURN Server (Required for NAT Traversal)

For remote deployment, you **must** configure a TURN server because:
- Both client and server are often behind NATs
- WebRTC needs TURN to relay traffic when direct connection fails

**Option A: Use a hosted TURN service**
- Services like Twilio, Metered.ca, or Xirsys offer TURN servers
- Add credentials to `client.html`:
  ```javascript
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    { urls: "turn:your-turn-service.com:3478", 
      username: "your-username", 
      credential: "your-password" }
  ]
  ```

**Option B: Set up your own TURN server (coturn)**
- Install coturn on your remote server or a separate machine
- Configure firewall to allow UDP 3478 and relay ports
- Add TURN credentials to `client.html`

#### Step 4: Firewall Configuration

**On your remote GPU server:**
- Open TCP port 8080 (or your chosen port) for HTTP signaling
- If running TURN server: Open UDP 3478 and relay port range (e.g., 49152-65535)

**Example (Ubuntu/Debian):**
```bash
sudo ufw allow 8080/tcp
sudo ufw allow 3478/udp
sudo ufw allow 49152:65535/udp  # TURN relay ports
```

#### Step 5: Access from Local Browser

1. **Open `client.html` in your browser** (or serve it from the remote server)
2. **Update the server URL** if needed
3. **Click "Start Capture"** - your webcam will stream to the remote GPU
4. **Pose data will be sent back** in real-time

#### Quick Test

**Test connection first:**
```bash
# From your local machine, test if server is reachable:
curl http://YOUR_REMOTE_SERVER_IP:8080/
```

If you get HTML back, the server is accessible!

## Configuration

### Server Configuration

Edit `server.py` or create a `config.py` to adjust:

- `process_every_n`: Process every Nth frame (default: 1, process all frames)
- `resize_short`: Resize input frames to this short edge (default: 512px)
- `use_smplx`: Use SMPLX model (default: True) or SMPL
- `fast_mode`: Use fast detection mode (default: True)

### Performance Tuning

**For Lower Latency:**
- Reduce `resize_short` (e.g., 384px)
- Set `process_every_n=2` (process every other frame)
- Use FP16/AMP (enabled by default if available)
- Use a server close to the client

**For Higher Accuracy:**
- Increase `resize_short` (e.g., 640px)
- Set `process_every_n=1` (process all frames)
- Use SMPLX model

**For Lower GPU Usage:**
- Set `process_every_n=2` or higher
- Reduce `resize_short`
- Use CPU mode (slower but no GPU required)

## Troubleshooting

### "FrankMocap not available" Error

- Verify FrankMocap is installed and in the `frankmocap/` directory
- Check that model weights are downloaded
- Verify Python path includes the frankmocap directory

### "CUDA not available" Warning

- Install PyTorch with CUDA support
- Verify CUDA toolkit is installed: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Connection Issues

- **Local**: Check firewall isn't blocking port 8080
- **Remote**: Configure TURN server for NAT traversal
- Check browser console for WebRTC errors
- Verify STUN/TURN servers are reachable

### High Latency

- Check GPU utilization: `nvidia-smi`
- Reduce input resolution
- Use `process_every_n=2` to skip frames
- Check network latency between client and server

### No Pose Detection

- Ensure person is visible in frame
- Check lighting conditions
- Verify FrankMocap models loaded successfully (check server logs)
- Test with dummy inference first (modify `FrankMocapWrapper.infer()`)

## Development

### Project Structure

```
remote-mocap/
├── server.py              # WebRTC server with FrankMocap integration
├── client.html            # Browser client for webcam capture
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── config.py              # Optional configuration (create if needed)
└── frankmocap/            # FrankMocap repository (cloned separately)
```

### Extending the System

**Add Skeleton Visualization:**
- Modify `client.html` to parse pose JSON and draw skeleton on canvas
- Use 2D joints (`joints2d`) for overlay on video

**Add ROS Integration:**
- Add ROS publisher in `server.py` after inference
- Publish joint angles or positions to ROS topics

**Add Multiple Person Support:**
- Modify `FrankMocapWrapper.infer()` to return poses for all detected people
- Update JSON format to include `person_id` for each pose

**Improve 3D Joint Extraction:**
- Currently uses simplified 2D-to-3D conversion
- Extract true 3D joints from SMPL output in `BodyMocap.regress()`

## License

This project uses FrankMocap, which is licensed under its own terms. Please refer to the [FrankMocap license](https://github.com/facebookresearch/frankmocap/blob/main/LICENSE).

## Acknowledgments

- [FrankMocap](https://github.com/facebookresearch/frankmocap) by Facebook Research
- [aiortc](https://github.com/aiortc/aiortc) for WebRTC support
- [SMPL](https://smpl.is.tue.mpg.de/) for the body model

## Support

For issues related to:
- **This WebRTC wrapper**: Open an issue in this repository
- **FrankMocap**: See [FrankMocap issues](https://github.com/facebookresearch/frankmocap/issues)
- **WebRTC/aiortc**: See [aiortc documentation](https://aiortc.readthedocs.io/)

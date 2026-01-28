# SSH Setup Guide - Remote GPU Server

Quick guide for running the server on a remote GPU via SSH and using your laptop's webcam.

## Architecture

```
Your Laptop (Browser)  ──WebRTC──>  Remote GPU Server (SSH)
     Webcam                          server.py + CUDA
```

## Step-by-Step Setup

### 1. On Remote GPU Server (via SSH)

```bash
# SSH into your remote server
ssh user@your-remote-server-ip

# Navigate to project directory
cd ~/remote-mocap  # or wherever you cloned it

# Activate virtual environment (if using one)
source venv/bin/activate

# Make sure FrankMocap is set up
cd frankmocap
# Download weights, install dependencies, etc.

# Go back to project root
cd ..

# Start the server (binds to 0.0.0.0 to accept remote connections)
python server.py --host 0.0.0.0 --port 8080
```

**Keep server running:**
- Use `screen` or `tmux` to keep session alive after SSH disconnect
- Or use systemd service (see DEPLOYMENT.md)

```bash
# Using screen
screen -S mocap-server
python server.py --host 0.0.0.0 --port 8080
# Press Ctrl+A then D to detach
# Reattach with: screen -r mocap-server
```

### 2. On Your Laptop

1. **Open `client.html`** in your browser (Chrome/Edge recommended)

2. **Update the server URL** in `client.html`:
   ```javascript
   // Find this line around line 150:
   const serverUrl = "";  // Empty = same origin
   
   // Change to:
   const serverUrl = "http://YOUR_REMOTE_SERVER_IP:8080";
   ```

3. **Click "Start Capture"** - your laptop's webcam will stream to the remote GPU

4. **Allow camera permissions** when prompted

5. **Watch pose data** appear in real-time!

## Firewall Configuration

### On Remote Server

Make sure port 8080 is open:

```bash
# Ubuntu/Debian
sudo ufw allow 8080/tcp
sudo ufw status

# Or if using iptables
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
```

### On Your Laptop

Usually no firewall changes needed (outgoing connections are typically allowed).

## Testing Connection

### From Your Laptop

Test if server is reachable:

```bash
# Test HTTP connection
curl http://YOUR_REMOTE_SERVER_IP:8080/

# Should return HTML (the client.html page)
```

### Check Server Logs

On remote server, you should see:
```
INFO: Using GPU: NVIDIA GeForce RTX 3090
INFO: Loading FrankMocap models...
INFO: Bbox detector loaded
INFO: Body mocap model loaded
INFO: Enabled AMP (FP16) for faster inference
INFO: Starting server on 0.0.0.0:8080
```

## Real-Time Performance

The server is now optimized for real-time:

- **SMPL model** (faster than SMPLX)
- **FP16/AMP** enabled
- **Bbox tracking** (skip detection every 5 frames)
- **384px input** (faster inference)
- **torch.compile** (PyTorch 2.0+)

**Expected performance on RTX 3090:**
- 20-25 FPS
- 40-50ms latency per frame

## Troubleshooting

### "Connection refused"

- Check server is running: `ps aux | grep server.py`
- Check firewall: `sudo ufw status`
- Check server is bound to 0.0.0.0, not 127.0.0.1

### "WebRTC connection failed"

- Configure TURN server (required for NAT traversal)
- See DEPLOYMENT.md for TURN setup

### "Camera not working"

- Make sure you're using HTTPS in production (or localhost for testing)
- Check browser permissions for camera access
- Try a different browser (Chrome/Edge work best)

### "Slow performance"

- Check GPU utilization: `nvidia-smi` on remote server
- Reduce `resize_short` to 256px
- Set `process_every_n=2` to skip frames
- See REALTIME_OPTIMIZATION.md for more tips

## Quick Test Script

Create `test_connection.sh` on your laptop:

```bash
#!/bin/bash
SERVER_IP="YOUR_REMOTE_SERVER_IP"
PORT="8080"

echo "Testing connection to $SERVER_IP:$PORT..."
if curl -s "http://$SERVER_IP:$PORT/" > /dev/null; then
    echo "✓ Server is reachable!"
    echo "Open client.html and set serverUrl to: http://$SERVER_IP:$PORT"
else
    echo "✗ Cannot reach server. Check:"
    echo "  1. Server is running"
    echo "  2. Firewall allows port $PORT"
    echo "  3. Server IP is correct"
fi
```

## Next Steps

- See `REALTIME_OPTIMIZATION.md` for performance tuning
- See `DEPLOYMENT.md` for production deployment
- See `README.md` for full documentation

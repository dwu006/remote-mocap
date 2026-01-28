# Remote GPU Deployment Guide

This guide covers deploying the WebRTC FrankMocap server on remote GPU services like Vast.ai, RunPod, AWS, GCP, etc.

## Quick Start: Vast.ai / RunPod

### 1. Create GPU Instance

**Vast.ai:**
1. Go to [vast.ai](https://vast.ai)
2. Search for GPU instances (RTX 3090, A100, etc.)
3. Create instance with:
   - Ubuntu 20.04/22.04
   - Public IP enabled
   - SSH access configured

**RunPod:**
1. Go to [runpod.io](https://www.runpod.io)
2. Create a GPU pod (RTX 3090, A100, etc.)
3. Select template with CUDA/PyTorch pre-installed

### 2. SSH into Instance

```bash
ssh root@YOUR_INSTANCE_IP
# Or use the SSH command provided by the service
```

### 3. Install Dependencies

```bash
# Update system
apt-get update && apt-get install -y python3-pip git

# Clone repositories
cd ~
git clone <your-remote-mocap-repo-url>
cd remote-mocap
git clone https://github.com/facebookresearch/frankmocap.git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA (adjust version for your CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install FrankMocap dependencies (check their README)
cd frankmocap
# Follow FrankMocap installation instructions
# Download weights, etc.
```

### 4. Download FrankMocap Weights

```bash
cd ~/remote-mocap/frankmocap
# Run download script (check FrankMocap README)
bash scripts/download_weights.sh
# Or manually download from their instructions
```

### 5. Configure Firewall

```bash
# Allow HTTP signaling port
ufw allow 8080/tcp

# Allow TURN server ports (if running TURN on same machine)
ufw allow 3478/udp
ufw allow 49152:65535/udp
```

### 6. Start Server

```bash
cd ~/remote-mocap
source venv/bin/activate
python server.py --host 0.0.0.0 --port 8080
```

**Keep server running:**
- Use `screen` or `tmux` to keep session alive
- Or use systemd service (see below)

### 7. Configure Client

On your **local computer**, update `client.html`:

```javascript
// Change this line:
const serverUrl = "http://YOUR_INSTANCE_IP:8080";
```

Or serve `client.html` from the remote server and access via:
```
http://YOUR_INSTANCE_IP:8080/
```

### 8. Set Up TURN Server

**Option A: Use Hosted TURN Service**

Update `client.html` with TURN credentials:
```javascript
iceServers: [
  { urls: "stun:stun.l.google.com:19302" },
  { urls: "turn:your-turn-service.com:3478",
    username: "your-username",
    credential: "your-password" }
]
```

**Option B: Install coturn on Remote Server**

```bash
apt-get install coturn

# Edit /etc/turnserver.conf
# Set:
listening-port=3478
realm=your-domain.com
user=username:password

# Start coturn
systemctl start coturn
systemctl enable coturn
```

## Running as a Service (systemd)

Create `/etc/systemd/system/frankmocap-server.service`:

```ini
[Unit]
Description=WebRTC FrankMocap Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/remote-mocap
Environment="PATH=/root/remote-mocap/venv/bin"
ExecStart=/root/remote-mocap/venv/bin/python server.py --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
systemctl enable frankmocap-server
systemctl start frankmocap-server
systemctl status frankmocap-server
```

## HTTPS Setup (Production)

For production, use HTTPS (required for camera access):

### Option 1: Nginx Reverse Proxy

```bash
apt-get install nginx certbot python3-certbot-nginx

# Configure nginx
cat > /etc/nginx/sites-available/frankmocap <<EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

ln -s /etc/nginx/sites-available/frankmocap /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx

# Get SSL certificate
certbot --nginx -d your-domain.com
```

### Option 2: Direct SSL with aiohttp

Modify `server.py` to use SSL:
```python
import ssl

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('cert.pem', 'key.pem')

web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)
```

## Troubleshooting Remote Deployment

### Connection Issues

**Test server accessibility:**
```bash
# From local machine
curl http://YOUR_SERVER_IP:8080/
```

**Check firewall:**
```bash
# On remote server
ufw status
netstat -tulpn | grep 8080
```

**Check server logs:**
```bash
# On remote server
journalctl -u frankmocap-server -f
# Or if running manually, check terminal output
```

### WebRTC Connection Fails

**Common causes:**
1. **No TURN server** - Required for NAT traversal
2. **Firewall blocking UDP** - WebRTC uses UDP for media
3. **STUN server unreachable** - Check network connectivity

**Debug:**
- Check browser console for WebRTC errors
- Check server logs for connection attempts
- Test with TURN server explicitly configured

### GPU Not Detected

**Check CUDA:**
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Verify PyTorch CUDA:**
```bash
python -c "import torch; print(torch.version.cuda)"
```

### High Latency

**Optimize:**
1. Use server close to your location
2. Reduce input resolution (`resize_short=384`)
3. Process every other frame (`process_every_n=2`)
4. Check GPU utilization: `watch -n 1 nvidia-smi`

## Cost Optimization

**For Vast.ai/RunPod:**
- Use spot instances for lower cost
- Stop instance when not in use
- Monitor GPU utilization to right-size instance

**For AWS/GCP:**
- Use reserved instances for long-term use
- Consider GPU instances with lower specs if latency allows
- Use auto-scaling if serving multiple users

## Security Considerations

1. **Authentication:** Add API key or token authentication to `/offer` endpoint
2. **Rate Limiting:** Limit connections per IP
3. **HTTPS:** Always use HTTPS in production
4. **Firewall:** Only open necessary ports
5. **Updates:** Keep dependencies updated

## Example: Complete Vast.ai Setup

```bash
# 1. SSH into instance
ssh root@YOUR_VAST_IP

# 2. One-time setup
cd ~
git clone <repo>
cd remote-mocap
git clone https://github.com/facebookresearch/frankmocap.git
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# ... install FrankMocap weights ...

# 3. Configure
ufw allow 8080/tcp

# 4. Start (in screen/tmux)
screen -S server
python server.py --host 0.0.0.0 --port 8080
# Press Ctrl+A then D to detach

# 5. On local machine, update client.html:
# const serverUrl = "http://YOUR_VAST_IP:8080";

# 6. Open client.html in browser and test!
```

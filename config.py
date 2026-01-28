# config.py
# Configuration file for WebRTC FrankMocap server

import os

# Server configuration
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8080))

# FrankMocap configuration
FRANKMOCAP_CONFIG = {
    # Use SMPLX model (True) or SMPL (False)
    "use_smplx": os.getenv("USE_SMPLX", "true").lower() == "true",
    
    # Fast mode: body detection first, then hand bbox from body (faster)
    # False: detect both body and hand bboxes separately (slower but more accurate)
    "fast_mode": os.getenv("FAST_MODE", "true").lower() == "true",
    
    # Path to FrankMocap directory (relative to this file)
    "frankmocap_dir": os.getenv("FRANKMOCAP_DIR", "frankmocap"),
}

# Video processing configuration
VIDEO_CONFIG = {
    # Process every Nth frame (1 = process all frames, 2 = every other frame, etc.)
    "process_every_n": int(os.getenv("PROCESS_EVERY_N", "1")),
    
    # Resize input frames to this short edge (in pixels)
    # Smaller = faster but less accurate, Larger = slower but more accurate
    "resize_short": int(os.getenv("RESIZE_SHORT", "512")),
    
    # Target frame rate (not enforced, but used for logging)
    "target_fps": int(os.getenv("TARGET_FPS", "30")),
}

# Device configuration
DEVICE_CONFIG = {
    # Force CPU even if CUDA is available (for testing)
    "force_cpu": os.getenv("FORCE_CPU", "false").lower() == "true",
    
    # Use mixed precision (FP16) if available (faster on modern GPUs)
    "use_amp": os.getenv("USE_AMP", "true").lower() == "true",
}

# WebRTC configuration
WEBRTC_CONFIG = {
    # STUN server (for NAT traversal)
    "stun_servers": [
        {"urls": "stun:stun.l.google.com:19302"}
    ],
    
    # TURN servers (for NAT traversal in restrictive networks)
    # Add your TURN server credentials here:
    "turn_servers": [
        # Example:
        # {
        #     "urls": "turn:your-turn-server.com:3478",
        #     "username": "your-username",
        #     "credential": "your-password"
        # }
    ],
}

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Performance monitoring
PERFORMANCE_CONFIG = {
    # Log inference time for every Nth frame
    "log_inference_every_n": int(os.getenv("LOG_INFERENCE_EVERY_N", "30")),
    
    # Log connection statistics
    "log_connection_stats": os.getenv("LOG_CONNECTION_STATS", "true").lower() == "true",
}

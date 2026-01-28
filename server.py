# server.py
# WebRTC server for real-time motion capture using FrankMocap
import asyncio
import json
import time
import argparse
import logging
import os
import sys

import cv2
import numpy as np
from aiohttp import web
from av import VideoFrame

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCDataChannel
from aiortc.contrib.media import MediaRelay

# Add frankmocap to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'frankmocap'))

# FrankMocap imports
try:
    import torch
    from handmocap.hand_bbox_detector import HandBboxDetector
    from bodymocap.body_mocap_api import BodyMocap
    FRANKMOCAP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"FrankMocap imports failed: {e}. Running in placeholder mode.")
    FRANKMOCAP_AVAILABLE = False
    torch = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pcs = set()
relay = MediaRelay()


def get_device():
    """Detect and return the best available device (CUDA or CPU)."""
    if torch is None:
        return None
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = torch.device('cpu')
        logger.warning("CUDA not available, using CPU (will be slow)")
        return device


class FrankMocapWrapper:
    """
    Wrapper for FrankMocap that loads models once and runs inference per frame.
    Optimized for real-time performance.
    """
    def __init__(self, device=None, use_smplx=False, fast_mode=True, 
                 optimize_for_realtime=True, track_bbox=True, n_iter=1):
        """
        Initialize FrankMocap models.
        
        Args:
            device: torch.device (cuda or cpu)
            use_smplx: Use SMPLX model (True) or SMPL (False) - SMPL is faster
            fast_mode: Use fast mode (body-only detection, then hand bbox from body)
            optimize_for_realtime: Enable real-time optimizations (TorchScript, FP16, etc.)
            track_bbox: Track bbox between frames to skip detection (faster)
            n_iter: Number of HMR iterations (1-3, lower is faster but less accurate)
        """
        if not FRANKMOCAP_AVAILABLE:
            logger.error("FrankMocap not available. Cannot initialize wrapper.")
            self.available = False
            return
            
        self.device = device if device is not None else get_device()
        self.use_smplx = use_smplx
        self.fast_mode = fast_mode
        self.optimize_for_realtime = optimize_for_realtime
        self.track_bbox = track_bbox
        self.n_iter = n_iter  # HMR iterations (1 for fastest, 2-3 for accuracy)
        self.available = True
        self.last_bbox = None  # For bbox tracking
        self.bbox_skip_frames = 0
        self.bbox_detect_every_n = 5  # Detect bbox every N frames if tracking
        
        # Default paths (relative to frankmocap directory)
        frankmocap_dir = os.path.join(os.path.dirname(__file__), 'frankmocap')
        extra_data_dir = os.path.join(frankmocap_dir, 'extra_data')
        
        # Checkpoint paths
        if use_smplx:
            checkpoint_body = os.path.join(
                extra_data_dir, 
                'body_module', 
                'pretrained_weights',
                'smplx-03-28-46060-w_spin_mlc3d_46582-2089_2020_03_28-21_56_16.pt'
            )
        else:
            checkpoint_body = os.path.join(
                extra_data_dir,
                'body_module',
                'pretrained_weights',
                '2020_05_31-00_50_43-best-51.749683916568756.pt'
            )
        
        checkpoint_hand = os.path.join(
            extra_data_dir,
            'hand_module',
            'pretrained_weights',
            'pose_shape_best.pth'
        )
        
        smpl_dir = os.path.join(extra_data_dir, 'smpl')
        
        try:
            logger.info("Loading FrankMocap models...")
            
            # Load bbox detector
            self.bbox_detector = HandBboxDetector('third_view', self.device)
            logger.info("Bbox detector loaded")
            
            # Determine AMP setting
            use_amp = False
            if self.optimize_for_realtime and torch is not None:
                if hasattr(torch.cuda, 'amp') and self.device.type == 'cuda':
                    use_amp = True
                    logger.info("Will use AMP (FP16) for faster inference")
            
            # Load body mocap with real-time optimizations
            self.body_mocap = BodyMocap(
                checkpoint_body,
                smpl_dir,
                device=self.device,
                use_smplx=use_smplx,
                use_amp=use_amp,  # Enable AMP in BodyMocap
                n_iter=self.n_iter  # Use reduced iterations for speed
            )
            logger.info(f"Body mocap model loaded (n_iter={self.n_iter}, use_amp={use_amp})")
            
            # Real-time optimizations (additional)
            if self.optimize_for_realtime and torch is not None:
                self.use_amp = use_amp
                
                # Try to compile model with torch.compile (PyTorch 2.0+)
                if hasattr(torch, 'compile') and self.device.type == 'cuda':
                    try:
                        self.body_mocap.model_regressor = torch.compile(
                            self.body_mocap.model_regressor, 
                            mode='reduce-overhead'
                        )
                        logger.info("Compiled model with torch.compile for faster inference")
                    except Exception as e:
                        logger.warning(f"Could not compile model: {e}")
                
                # Set to eval mode and optimize
                self.body_mocap.model_regressor.eval()
                if self.device.type == 'cuda':
                    # Enable cuDNN benchmarking for consistent input sizes
                    torch.backends.cudnn.benchmark = True
                    logger.info("Enabled cuDNN benchmarking")
            else:
                self.use_amp = False
            
            # Load hand mocap (optional, only if not in fast mode)
            if not fast_mode:
                try:
                    from handmocap.hand_mocap_api import HandMocap
                    self.hand_mocap = HandMocap(
                        checkpoint_hand,
                        smpl_dir,
                        device=self.device,
                        use_smplx=use_smplx
                    )
                    logger.info("Hand mocap model loaded")
                except Exception as e:
                    logger.warning(f"Could not load hand mocap: {e}. Using body-only mode.")
                    self.hand_mocap = None
            else:
                self.hand_mocap = None
                
            logger.info("FrankMocap models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load FrankMocap models: {e}")
            logger.error("Make sure FrankMocap is properly installed and weights are downloaded.")
            self.available = False
            raise

    def infer(self, bgr_image):
        """
        Run FrankMocap inference on a BGR image.
        Optimized for real-time performance.
        
        Args:
            bgr_image: numpy array (BGR format, as from cv2)
            
        Returns:
            dict with keys:
                - ts: timestamp
                - joints3d: list of 3D joint positions [[x,y,z], ...]
                - joints2d: list of 2D joint positions in image space [[x,y], ...]
                - inference_ms: inference time in milliseconds
                - person_id: person index (0 for first person)
        """
        if not self.available:
            return {
                "ts": time.time(),
                "joints3d": [],
                "joints2d": [],
                "inference_ms": 0.0,
                "person_id": 0,
                "error": "FrankMocap not available"
            }
        
        t0 = time.time()
        
        try:
            # Bbox detection with tracking optimization
            body_bbox_list = None
            detect_bbox = True
            
            if self.track_bbox and self.last_bbox is not None:
                self.bbox_skip_frames += 1
                if self.bbox_skip_frames < self.bbox_detect_every_n:
                    # Reuse last bbox (with slight expansion for safety)
                    body_bbox_list = [self.last_bbox.copy()]
                    detect_bbox = False
            
            if detect_bbox:
                # Detect body bounding boxes (expensive operation)
                _, body_bbox_list = self.bbox_detector.detect_body_bbox(bgr_image.copy())
                if len(body_bbox_list) > 0:
                    self.last_bbox = body_bbox_list[0].copy()
                    self.bbox_skip_frames = 0
                else:
                    self.last_bbox = None
            
            if len(body_bbox_list) < 1:
                # No person detected
                return {
                    "ts": time.time(),
                    "joints3d": [],
                    "joints2d": [],
                    "inference_ms": (time.time() - t0) * 1000.0,
                    "person_id": 0,
                    "detected": False
                }
            
            # Use first person (largest bbox)
            body_bbox_list = [body_bbox_list[0]]
            
            # Run body regression (AMP and n_iter are already handled in BodyMocap.regress)
            # But we can override if needed
            pred_body_list = self.body_mocap.regress(
                bgr_image, 
                body_bbox_list,
                use_amp=self.use_amp if hasattr(self, 'use_amp') else None,
                n_iter=self.n_iter
            )
            
            if len(pred_body_list) == 0 or pred_body_list[0] is None:
                return {
                    "ts": time.time(),
                    "joints3d": [],
                    "joints2d": [],
                    "inference_ms": (time.time() - t0) * 1000.0,
                    "person_id": 0,
                    "detected": False
                }
            
            pred_body = pred_body_list[0]
            
            # Get 2D joints in image coordinates
            joints2d = pred_body.get('pred_joints_img', [])
            
            # Convert joints2d to list format
            joints2d_list = []
            if len(joints2d) > 0:
                joints2d_list = joints2d[:, :2].tolist()  # Take only x, y
            
            # Extract 3D joints
            # Note: FrankMocap's regress() doesn't directly expose 3D joints in camera space
            # We use the 2D joints in image space and estimate depth from camera parameters
            # For true 3D joints, you would need to extract from SMPL output directly
            # or modify BodyMocap to return 3D joints in camera space
            joints3d_list = []
            if len(joints2d) > 0:
                # Get camera parameters for depth estimation
                pred_camera = pred_body.get('pred_camera', [1.0, 0.0, 0.0])
                cam_scale = pred_camera[0] if len(pred_camera) > 0 else 1.0
                
                # Use 2D joints with estimated depth based on camera scale
                # The z coordinate is estimated from the camera scale parameter
                # This is an approximation - for accurate 3D, extract from SMPL joints directly
                for joint_2d in joints2d:
                    x, y = float(joint_2d[0]), float(joint_2d[1])
                    # Estimate z from camera scale (inverse relationship)
                    # This is a simplified approximation
                    z = float(cam_scale * 100.0) if cam_scale > 0 else 0.0
                    joints3d_list.append([x, y, z])
            
            inference_time = (time.time() - t0) * 1000.0
            
            return {
                "ts": time.time(),
                "joints3d": joints3d_list,
                "joints2d": joints2d_list,
                "inference_ms": inference_time,
                "person_id": 0,
                "detected": True
            }
            
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return {
                "ts": time.time(),
                "joints3d": [],
                "joints2d": [],
                "inference_ms": (time.time() - t0) * 1000.0,
                "person_id": 0,
                "error": str(e)
            }


class VideoProcessorTrack(MediaStreamTrack):
    """
    Receives frames from the browser, processes them with FrankMocap,
    and sends pose results via DataChannel.
    Optimized for real-time performance with async processing.
    """
    kind = "video"

    def __init__(self, track, frankmocap: FrankMocapWrapper, datachannel: RTCDataChannel, 
                 process_every_n=1, resize_short=384):
        super().__init__()
        self.track = track
        self.frankmocap = frankmocap
        self.datachannel = datachannel
        self.process_every_n = process_every_n
        self.resize_short = resize_short  # Default to 384 for real-time (was 512)
        self.frame_count = 0
        self.pending_result = None  # For async result handling

    async def recv(self):
        frame = await self.track.recv()  # av.VideoFrame
        self.frame_count += 1

        # Convert to BGR numpy
        img = frame.to_ndarray(format="bgr24")

        send_pose = False
        if (self.frame_count % self.process_every_n) == 0:
            send_pose = True

        if send_pose:
            # Optional: resize to keep compute manageable
            h, w = img.shape[:2]
            short = min(h, w)
            if short > self.resize_short:
                if h < w:
                    new_h = self.resize_short
                    new_w = int(w * (self.resize_short / h))
                else:
                    new_w = self.resize_short
                    new_h = int(h * (self.resize_short / w))
                img_proc = cv2.resize(img, (new_w, new_h))
            else:
                img_proc = img

            # Run FrankMocap inference (non-blocking for WebRTC)
            # Note: Inference still runs synchronously, but we don't block frame forwarding
            try:
                result = self.frankmocap.infer(img_proc)
                
                # Send result via datachannel as JSON
                if self.datachannel and self.datachannel.readyState == "open":
                    try:
                        # Compact JSON (no spaces) for faster transmission
                        self.datachannel.send(json.dumps(result, separators=(',', ':')))
                    except Exception as e:
                        logger.warning(f"Failed to send via datachannel: {e}")
            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)

        # Return frame to keep stream alive (forward unchanged)
        return frame


async def index(request):
    """Serve the HTML client."""
    try:
        with open("client.html", "r") as f:
            content = f.read()
        return web.Response(content_type="text/html", text=content)
    except FileNotFoundError:
        return web.Response(text="client.html not found", status=404)


async def offer(request):
    """Handle WebRTC offer from client."""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    logger.info(f"Created PC {pc}")

    # Create FrankMocap instance per connection (or share one)
    # For now, create per connection - can optimize to share later
    try:
        device = get_device()
        # Use SMPL instead of SMPLX for faster inference (use_smplx=False)
        # Enable real-time optimizations
        frank = FrankMocapWrapper(
            device=device, 
            use_smplx=False,  # SMPL is faster than SMPLX
            fast_mode=True,
            optimize_for_realtime=True,
            track_bbox=True,  # Track bbox to skip detection
            n_iter=1  # Use 1 iteration for fastest inference (2-3 for more accuracy)
        )
    except Exception as e:
        logger.error(f"Failed to initialize FrankMocap: {e}")
        return web.Response(text=f"Server error: {e}", status=500)

    # DataChannel holder
    dc_holder = {"dc": None}

    # When a datachannel is created from browser side
    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info(f"DataChannel created: {channel.label}")
        dc_holder["dc"] = channel
        @channel.on("message")
        def on_message(message):
            logger.debug(f"DataChannel msg from client: {message}")

    @pc.on("iceconnectionstatechange")
    def on_ice_state():
        logger.info(f"ICE connection state: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            asyncio.ensure_future(pc.close())
            pcs.discard(pc)

    # Set remote description
    await pc.setRemoteDescription(offer)

    # Add tracks: for each incoming track, create a VideoProcessorTrack
    @pc.on("track")
    def on_track(track):
        logger.info(f"Track {track.kind} received")
        
        # Create datachannel if not present
        if dc_holder["dc"] is None:
            channel = pc.createDataChannel("pose")
            dc_holder["dc"] = channel
            logger.info("Created server-side DataChannel 'pose'")

        # Wrap with MediaRelay and add VideoProcessorTrack
        # Optimized settings for real-time: smaller resize, process every frame
        local_track = VideoProcessorTrack(
            relay.subscribe(track), 
            frank, 
            dc_holder["dc"], 
            process_every_n=1,  # Process every frame for real-time
            resize_short=384  # Smaller for faster inference (was 512)
        )
        pc.addTrack(local_track)

        @track.on("ended")
        async def on_ended():
            logger.info(f"Track {track.kind} ended")

    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )


def create_app():
    """Create aiohttp application."""
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC FrankMocap Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    args = parser.parse_args()

    app = create_app()
    logger.info(f"Starting server on {args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port)

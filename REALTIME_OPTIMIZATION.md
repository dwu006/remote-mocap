# Real-Time Optimization Guide

FrankMocap is not inherently real-time, but with optimizations, you can achieve near real-time performance (15-30 FPS) on modern GPUs.

## Optimizations Implemented

### 1. Model Optimizations

- **SMPL instead of SMPLX**: SMPL is faster (~2x) with minimal accuracy loss
- **torch.compile()**: PyTorch 2.0+ compilation for faster inference
- **FP16/AMP**: Mixed precision for ~2x speedup on modern GPUs
- **cuDNN Benchmarking**: Optimizes for consistent input sizes

### 2. Bbox Tracking

- **Skip bbox detection**: Track bbox between frames, only detect every 5 frames
- **Bbox reuse**: Reuse previous bbox with slight expansion for safety
- **Saves ~20-30ms per frame** (bbox detection is expensive)

### 3. Input Resolution

- **384px short edge** (default) instead of 512px
- Smaller input = faster inference
- Can go down to 256px for even faster (but less accurate)

### 4. Frame Processing

- Process every frame (`process_every_n=1`) for real-time
- Can set to 2 to skip frames if GPU can't keep up

## Performance Targets

### On RTX 3090 / A100:
- **Target**: 20-30 FPS
- **Latency**: 30-50ms per frame
- **Settings**: 384px, SMPL, FP16, bbox tracking

### On RTX 2080 / 3060:
- **Target**: 15-20 FPS  
- **Latency**: 50-70ms per frame
- **Settings**: 384px, SMPL, FP16, bbox tracking

### On CPU:
- **Target**: 2-5 FPS (not real-time)
- **Latency**: 200-500ms per frame
- **Not recommended for real-time**

## Configuration

### Fastest (Lower Accuracy)
```python
frank = FrankMocapWrapper(
    device=device,
    use_smplx=False,  # SMPL is faster
    fast_mode=True,
    optimize_for_realtime=True,
    track_bbox=True
)

# In VideoProcessorTrack:
resize_short=256  # Very small
process_every_n=1
```

### Balanced (Recommended)
```python
frank = FrankMocapWrapper(
    device=device,
    use_smplx=False,
    fast_mode=True,
    optimize_for_realtime=True,
    track_bbox=True
)

# In VideoProcessorTrack:
resize_short=384  # Good balance
process_every_n=1
```

### Highest Quality (Slower)
```python
frank = FrankMocapWrapper(
    device=device,
    use_smplx=True,  # SMPLX is more accurate
    fast_mode=True,
    optimize_for_realtime=True,
    track_bbox=True
)

# In VideoProcessorTrack:
resize_short=512  # Higher quality
process_every_n=1
```

## Additional Optimizations

### 1. Use TensorRT (Advanced)

Convert PyTorch model to TensorRT for even faster inference:

```python
# Requires TensorRT installation
import tensorrt as trt
# Convert model to TensorRT format
# See TensorRT documentation
```

### 2. Batch Processing (Future)

If processing multiple streams, batch frames together:

```python
# Collect N frames, process in batch
# Not implemented yet, but would be faster
```

### 3. Async Processing

Move inference to separate thread/process:

```python
# Use asyncio or threading
# Not implemented yet - current is synchronous
```

### 4. Model Quantization

Use INT8 quantization for faster inference:

```python
# Requires quantization-aware training or post-training quantization
# See PyTorch quantization docs
```

## Monitoring Performance

### Check FPS:
```python
# Server logs inference time
# Check "inference_ms" in pose JSON
```

### GPU Utilization:
```bash
watch -n 1 nvidia-smi
```

### Latency Breakdown:
- Bbox detection: ~20-30ms (if not tracked)
- Body regression: ~30-50ms
- Post-processing: ~5-10ms
- **Total: ~50-90ms per frame** (20-10 FPS)

## Troubleshooting

### Still Too Slow?

1. **Reduce resolution**: 384px → 256px
2. **Skip frames**: `process_every_n=2`
3. **Disable hand tracking**: Already disabled in fast_mode
4. **Use CPU**: Not recommended, but possible
5. **Upgrade GPU**: RTX 3090+ recommended

### Too Fast / GPU Underutilized?

1. **Increase resolution**: 384px → 512px
2. **Use SMPLX**: More accurate, slower
3. **Process every frame**: Already doing this

## Alternative: Faster Pose Estimators

If FrankMocap is still too slow, consider:

1. **MediaPipe Pose**: Very fast, less accurate
2. **OpenPose**: Faster than FrankMocap
3. **YOLO + Lightweight Pose**: Fast detection + pose
4. **PoseNet**: Browser-based, very fast

These would require replacing the FrankMocap integration.

## Expected Performance

With all optimizations enabled on RTX 3090:
- **Input**: 384x384px
- **Model**: SMPL (not SMPLX)
- **FP16**: Enabled
- **Bbox tracking**: Enabled
- **Expected**: 20-25 FPS
- **Latency**: 40-50ms per frame

This is **near real-time** and suitable for teleoperation!

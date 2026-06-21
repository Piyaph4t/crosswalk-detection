# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
A Proof of Concept (PoC) for detecting humans crossing crosswalks on a Raspberry Pi 4 Model B. The core goal is achieving 5-10 FPS using a quantized YOLOv8n model via ONNX Runtime.

## Hardware Context
- **Target:** Raspberry Pi 4 Model B (4GB).
- **OS:** Raspbian (Standard system Python).
- **Vision Inputs:** Supports RTSP (CCTV stream), PiCamera (CSI), and USB webcams via provider pattern.

## Development Workflow

### Environment Setup
The project uses `uv` for fast, reproducible dependency management linked to the system Python binary.

**Initial Setup (on Pi):**
```bash
chmod +x setup_env.sh && ./setup_env.sh
sudo reboot
```

**After reboot:**
```bash
source .venv/bin/activate
```

**IMPORTANT: Always use `uv run` for executing Python commands:**
```bash
# Run main application
uv run python src/main.py

# Run tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_rtsp_provider.py -v
```

**Adding new dependencies:**
```bash
uv add <package>
```

### Configuration
Edit `config.yaml` to select your vision provider:
- `vision.provider`: Set to `"rtsp"`, `"picamera"`, or `"usb_webcam"`
- Configure provider-specific settings (URL, resolution, etc.)

### Common Tasks
- **Run Inference:** `uv run python src/main.py`
- **Run All Tests:** `uv run pytest tests/ -v`
- **Check Test Coverage:** `uv run pytest tests/ --cov=src/providers`

## Architecture

### Vision Provider Pattern
The project uses a provider pattern to decouple hardware inputs from the AI engine.

**Available Providers:**
- **RTSPProvider** - CCTV stream via RTSP (with exponential backoff reconnection)
- **PiCameraProvider** - CSI camera module via V4L2
- **USBWebcamProvider** - USB webcam
- **VideoFileProvider** - Video file for testing

**Provider Interface** (`VisionProvider` ABC):
- `get_frame() -> tuple[bool, np.ndarray]` - Get next frame
- `is_connected() -> bool` - Check connection status
- `reconnect() -> bool` - Attempt reconnection
- `release()` - Clean up resources

### Inference Pipeline
`VisionProvider.get_frame()` → `Pre-processing (letterbox_resize)` → `YOLO Inference` → `Visualization` → Display

**Key Files:**
- `src/providers/` - All provider implementations
- `src/providers/registry.py` - Provider factory
- `src/utils/config_loader.py` - Config validation
- `config.yaml` - Runtime configuration
- `src/main.py` - Main inference loop

## Critical Constraints
- **Performance:** Target is 5-10 FPS. Avoid heavy PyTorch calls; stick to ONNX.
- **Memory:** 4GB limit. Use `opencv-python-headless` to reduce overhead.
- **Quantization:** Models must be INT8 quantized for the target hardware.
- **Python Execution:** Always use `uv run` to ensure correct virtual environment and dependencies.

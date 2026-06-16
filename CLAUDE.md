# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
A Proof of Concept (PoC) for detecting humans crossing crosswalks on a Raspberry Pi 4 Model B. The core goal is achieving 5-10 FPS using a quantized YOLOv8n model via ONNX Runtime.

## Hardware Context
- **Target:** Raspberry Pi 4 Model B (4GB).
- **OS:** Raspbian (Standard system Python).
- **Inputs:** Supports both Pi Cam (v4l2/OpenCV) and HuskyLens (I2C/Serial).

## Development Workflow

### Environment Setup
The project uses `uv` for fast, reproducible dependency management linked to the system Python binary.
- **Initial Setup (on Pi):** Run `chmod +x setup_env.sh && ./setup_env.sh` then `sudo reboot`.
- **Virtual Env Activation:** `source .venv/bin/activate`
- **Adding Dependencies:** `uv add <package>`

### Common Tasks
- **Run Inference PoC:** (Pending implementation) `python src/main.py`
- **Verify Model Loading:** (Pending implementation) `python -m pytest tests/test_inference.py`

## Architecture
The project follows a "Vision Provider" pattern to decouple hardware inputs from the AI engine.

### Core Components
- **`src/providers/`**: Implements `VisionProvider` base class.
    - `pi_cam.py`: Captures frames via OpenCV.
    - `husky_lens.py`: Captures detections via I2C.
- **`src/core/engine.py`**: The AI brain. Uses **ONNX Runtime** with the **XNNPACK** execution provider for optimized ARM CPU inference.
- **`src/core/metrics.py`**: Implements a sliding window average to track and report stable FPS.

### Inference Pipeline
`VisionProvider` $\rightarrow$ `Pre-processing (Resize/Normalize)` $\rightarrow$ `ONNX Engine (XNNPACK)` $\rightarrow$ `Post-processing (NMS)` $\rightarrow$ `Metrics/Visualization`.

## Critical Constraints
- **Performance:** Target is 5-10 FPS. Avoid heavy PyTorch calls; stick to ONNX.
- **Memory:** 4GB limit. Use `opencv-python-headless` to reduce overhead.
- **Quantization:** Models must be INT8 quantized for the target hardware.

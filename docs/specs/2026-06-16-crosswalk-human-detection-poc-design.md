# Design Spec: Human Crosswalk Detection Proof of Concept (PoC)
Date: 2026-06-16
Status: Draft
Target Hardware: Raspberry Pi 4 Model B (4GB)
Target Performance: 5-10 FPS

## 1. Objectives
The primary goal of this project is to implement a Proof of Concept (PoC) that detects humans crossing a crosswalk using AI. The core success metric for this phase is achieving a stable inference rate of 5-10 frames per second (FPS) on the Raspberry Pi 4 CPU.

## 2. Hardware Configuration
- **Compute:** Raspberry Pi 4 Model B (4GB RAM).
- **Vision Inputs (Interchangeable):**
    - **Pi Cam:** Standard CSI camera using Video4Linux2 (v4l2).
    - **HuskyLens AI Cam:** External AI camera communicating via I2C/Serial.

## 3. AI Architecture & Optimization
To meet the performance targets, the system will move away from heavy PyTorch models in favor of a streamlined inference pipeline.

### 3.1 Model Selection
- **Architecture:** YOLOv8n (Nano) - the smallest variant of the YOLOv8 family.
- **Inference Engine:** ONNX Runtime.
- **Optimization:** 
    - **Format:** Exported to `.onnx`.
    - **Quantization:** INT8 Quantization to reduce weight precision from 32-bit float to 8-bit integer, significantly increasing ARM CPU throughput.
    - **Backend:** Explicit use of the **XNNPACK** execution provider for optimized ARM CPU operations.

### 3.2 Performance Pipeline
- **Resolution:** Input frames will be resized to $320 \times 320$ or $640 \times 640$ to balance accuracy and speed.
- **NMS:** Fast Non-Max Suppression to filter overlapping bounding boxes.
- **Metrics:** A sliding window average (last 30 frames) will be used to report a stable FPS.

## 4. Software Architecture
The system is designed around the "Provider" pattern to allow seamless switching between hardware inputs.

### 4.1 Project Structure
```text
crosswalk-detection/
├── pyproject.toml        # UV managed dependencies
├── assets/
│   └── models/           # Quantized YOLOv8n.onnx
├── src/
│   ├── main.py           # Orchestrator
│   ├── core/
│   │   ├── engine.py     # ONNX Runtime logic
│   │   └── metrics.py     # FPS calculation
│   ├── providers/
│   │   ├── base.py      # VisionProvider Abstract Base Class
│   │   ├── pi_cam.py     # OpenCV implementation
│   │   └── husky_lens.py # I2C/Serial implementation
│   └── utils/
│       └── visualization.py # Bounding box drawing
└── tests/
    └── test_inference.py # Model load/sanity tests
```

### 4.2 Data Flow
`VisionProvider` (Pi Cam/HuskyLens) $\rightarrow$ `Pre-processing` (Resize/Normalize) $\rightarrow$ `ONNX Engine` (XNNPACK) $\rightarrow$ `Post-processing` (NMS) $\rightarrow$ `Visualization` $\rightarrow$ `Metrics Log`.

## 5. Development Environment & Setup
To ensure reproducibility on Raspbian OS, a dedicated automation script (`setup_env.sh`) will be used.

### 5.1 System Dependencies
The script will install:
- `libatlas-base-dev` (Critical for NumPy/OpenCV performance on Pi)
- `libjasper-dev`, `libpq5`, `libopenjp2-7`, `libtiff5-dev` (OpenCV image format support)
- `python3-dev`

### 5.2 Python Environment
- **Manager:** `uv` (for ultra-fast installation and locking).
- **Key Libraries:**
    - `onnxruntime` (Inference)
    - `opencv-python-headless` (Image processing)
    - `numpy` (Array manipulation)
    - `smbus2` (HuskyLens I2C communication)
    - `pyyaml` (Configuration management)

### 5.3 Git Workflow
- Initialize git repository.
- Configure `.gitignore` to exclude `.venv` and large binary `.onnx` files.

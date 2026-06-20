# YOLO Model Benchmarking Guide

This guide explains how to benchmark YOLO models (ONNX and NCNN formats) for Raspberry Pi 4 CPU performance.

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run benchmark on all models
python benchmark_models.py

# Custom options
python benchmark_models.py --warmup 20 --runs 200 --output results.json
```

## What Gets Benchmarked

The script automatically discovers and tests:

### ONNX Models
- `yolo26n.onnx`
- `yolov8n.onnx`
- `yolov9t.onnx`
- `yolov9s.onnx`
- `yolov10n.onnx`

### NCNN Models
All `*_ncnn_model/` and `*_ncnn_model_int8/` directories, including:
- `yolo26n_ncnn_model/`
- `yolo26n_ncnn_model_int8/` (INT8 quantized)
- `yolo26s_ncnn_model/`
- `yolov8n_ncnn_model/`
- `yolov9t_ncnn_model/`
- `yolov9s_ncnn_model/`
- `yolov10n_saved_model/`

## Performance Metrics

Each model is evaluated on:

| Metric | Description | Target |
|--------|-------------|--------|
| **FPS** | Frames per second | 5-10 FPS |
| **Avg Latency** | Mean inference time | < 200ms |
| **Min/Max Latency** | Best/worst case | Consistency check |
| **Memory** | Memory delta during inference | < 500MB |
| **Detections** | Number of objects found | Sanity check |

## Benchmark Process

1. **Model Discovery**: Scans `assets/model/` for ONNX and NCNN files
2. **Warmup**: Runs 10 inference passes to stabilize performance
3. **Benchmark**: Runs 100 iterations measuring latency
4. **Analysis**: Calculates FPS, memory usage, and statistics
5. **Report**: Generates summary table and JSON results

## Understanding Results

### Console Output

```
================================================================================
BENCHMARK SUMMARY (Target: 5-10 FPS on Pi4)
================================================================================
Model                     Format   FPS      Latency      Memory     Status
--------------------------------------------------------------------------------
yolo26n_ncnn_model_int8   NCNN     8.45     118.35       145.20     ✅ PASS
yolov8n_ncnn_model        NCNN     7.21     138.72       178.45     ✅ PASS
yolo26n                   ONNX     6.10     163.93       220.15     ✅ PASS
yolov9t                   ONNX     4.85     206.19       198.30     ❌ FAIL
yolov9s                   ONNX     2.34     427.35       310.80     ❌ FAIL
--------------------------------------------------------------------------------

🏆 Best Overall: yolo26n_ncnn_model_int8 (8.45 FPS)
🥇 Best ONNX: yolo26n (6.10 FPS)
🥇 Best NCNN: yolo26n_ncnn_model_int8 (8.45 FPS)

✓ 3/5 models meet the 5 FPS minimum target
```

### JSON Output

Results are saved to `benchmark_results.json`:

```json
{
  "benchmark_config": {
    "num_warmup": 10,
    "num_runs": 100,
    "input_size": 640,
    "hardware": "Raspberry Pi 4 Model B (4GB)",
    "backend": "CPU"
  },
  "results": [
    {
      "model_name": "yolo26n_ncnn_model_int8",
      "model_format": "ncnn",
      "avg_fps": 8.45,
      "avg_latency_ms": 118.35,
      "min_latency_ms": 112.20,
      "max_latency_ms": 145.80,
      "memory_mb": 145.20,
      "num_detections": 1200,
      "warmup_time_ms": 1250.45
    }
  ]
}
```

## Command-Line Options

```bash
python benchmark_models.py [OPTIONS]

Options:
  --model-dir PATH    Directory with YOLO models (default: assets/model)
  --warmup N          Warmup iterations (default: 10)
  --runs N            Benchmark iterations (default: 100)
  --output FILE       JSON output path (default: benchmark_results.json)
```

### Examples

```bash
# Quick test (fewer iterations)
python benchmark_models.py --warmup 5 --runs 50

# Thorough benchmark
python benchmark_models.py --warmup 20 --runs 300

# Test specific directory
python benchmark_models.py --model-dir /path/to/models

# Custom output file
python benchmark_models.py --output pi4_results.json
```

## Interpreting Results

### What to Look For

1. **FPS ≥ 5**: Meets minimum target for real-time crosswalk detection
2. **FPS ≥ 10**: Exceeds target, headroom for additional processing
3. **Latency consistency**: Small gap between min/max indicates stable performance
4. **Memory < 500MB**: Fits comfortably in Pi4's 4GB RAM
5. **INT8 models**: Should show ~2-4x speedup vs FP32

### Expected Performance on Pi4

| Model Type | Expected FPS | Notes |
|------------|-------------|-------|
| NCNN INT8 (nano) | 7-12 FPS | Best choice for Pi4 |
| NCNN FP16 (nano) | 4-7 FPS | Decent fallback |
| ONNX (nano) | 3-6 FPS | Slower than NCNN |
| Larger models (small/medium) | 1-3 FPS | Too slow for real-time |

### Recommendations

- **Production use**: Choose the fastest NCNN INT8 model that meets 5 FPS
- **Accuracy vs Speed**: Test detection quality after benchmark
- **Fallback**: Keep one ONNX model for compatibility
- **Monitor**: Re-benchmark after Ultralytics or ONNX Runtime updates

## Troubleshooting

### Model Loading Fails

```
❌ Failed to load model: [Errno 2] No such file or directory
```

**Fix**: Ensure NCNN models have both `.param` and `.bin` files in the directory.

### Low FPS on Development Machine

The script runs on whatever hardware you execute it on. For accurate Pi4 results:
- Run directly on the Raspberry Pi 4
- Or use QEMU emulation (slower, less accurate)
- Development machine results will be much faster

### Memory Errors

```
MemoryError: Unable to allocate array
```

**Fix**: 
- Close other applications
- Reduce `--runs` parameter
- Test models individually

### No Models Discovered

```
🔍 Discovered Models:
  ONNX: 0 models
  NCNN: 0 models
```

**Fix**: Check `--model-dir` path or verify model files exist in `assets/model/`.

## Next Steps

After benchmarking:

1. **Select best model** based on FPS results
2. **Test detection accuracy** on real crosswalk images
3. **Integrate** the chosen model into `src/core/engine.py`
4. **Validate** on actual Pi4 hardware with camera input
5. **Monitor** real-world performance with metrics logging

## Performance Optimization Tips

If no model meets 5 FPS target:

1. **Use INT8 quantization** (NCNN format preferred)
2. **Reduce input size** from 640x640 to 416x416
3. **Lower confidence threshold** to reduce NMS overhead
4. **Enable XNNPACK** for ONNX Runtime (check `engine.py`)
5. **Overclock Pi4** (with adequate cooling)
6. **Consider Coral TPU** as alternative accelerator

## Related Files

- `benchmark_models.py` - Main benchmark script
- `src/core/engine.py` - Inference engine wrapper
- `CLAUDE.md` - Project constraints and targets
- `pyproject.toml` - Python dependencies

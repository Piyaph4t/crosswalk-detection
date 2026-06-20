#!/usr/bin/env python3
"""
Benchmark YOLO models (ONNX and NCNN) for Raspberry Pi 4 CPU.
Measures FPS, latency, memory usage, and detection accuracy.
"""

import time
import psutil
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass, asdict
from ultralytics import YOLO


@dataclass
class BenchmarkResult:
    """Store benchmark results for a single model."""
    model_name: str
    model_format: str  # 'onnx' or 'ncnn'
    model_path: str
    avg_fps: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    memory_mb: float
    num_detections: int
    warmup_time_ms: float
    total_frames: int
    inference_only_ms: float  # Excludes pre/post-processing


class ModelBenchmark:
    """Benchmark YOLO models on CPU."""

    def __init__(self, num_warmup: int = 10, num_runs: int = 100, source=None, show: bool = False, imgsz: int = 640):
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        self.input_size = imgsz
        self.show = show
        self.source = source
        self.capture = None
        self.use_video_source = source is not None

        # Initialize video source if provided
        if self.use_video_source:
            self._init_video_source()
        else:
            # Create synthetic test image
            self.test_frame = self._create_test_frame()

    def _init_video_source(self):
        """Initialize video capture from webcam or file."""
        # Try to parse as integer (webcam index)
        try:
            source_int = int(self.source)
            self.capture = cv2.VideoCapture(source_int)
            source_type = f"Webcam {source_int}"
        except ValueError:
            # Treat as file path
            self.capture = cv2.VideoCapture(self.source)
            source_type = f"Video file: {self.source}"

        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        print(f"✓ Video source initialized: {source_type}")

        # Get frame dimensions
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        print(f"  Resolution: {width}x{height} @ {fps if fps > 0 else '?'} FPS")

    def _create_test_frame(self) -> np.ndarray:
        """Create a synthetic test image with some visual features."""
        frame = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)

        # Add some rectangles to simulate crosswalk/person
        cv2.rectangle(frame, (100, 200), (300, 500), (255, 128, 64), -1)  # Person-like blob
        cv2.rectangle(frame, (50, 550), (590, 600), (200, 200, 200), -1)  # Crosswalk stripes

        # Add noise for realism
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)

        return frame

    def _get_frame(self) -> np.ndarray:
        """Get next frame from video source or return synthetic frame."""
        if self.use_video_source:
            ret, frame = self.capture.read()
            if not ret:
                # Loop video or return None for webcam failure
                if isinstance(self.source, str) and self.source.isdigit():
                    # Webcam - return None to signal error
                    return None
                else:
                    # Video file - loop to beginning
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.capture.read()
                    if not ret:
                        return None
            return frame
        else:
            return self.test_frame.copy()

    def _letterbox_resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to input_size while maintaining aspect ratio."""
        h, w = image.shape[:2]
        scale = self.input_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        x_offset = (self.input_size - new_w) // 2
        y_offset = (self.input_size - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas

    def find_models(self, model_dir: Path) -> Dict[str, List[Path]]:
        """Discover all ONNX and NCNN models in the directory."""
        models = {
            'onnx': [],
            'ncnn': []
        }

        # Find ONNX models
        for onnx_file in model_dir.glob("*.onnx"):
            models['onnx'].append(onnx_file)

        # Find NCNN models (look for directories with .param and .bin)
        for ncnn_dir in model_dir.glob("*_ncnn_model*"):
            if ncnn_dir.is_dir():
                param_files = list(ncnn_dir.glob("*.param"))
                if param_files:
                    models['ncnn'].append(ncnn_dir)

        return models

    def benchmark_model(self, model_path: Path, model_format: str) -> BenchmarkResult:
        """Run benchmark on a single model."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_path.name} ({model_format.upper()})")
        print(f"{'='*60}")

        # Load model
        print("Loading model...")
        try:
            model = YOLO(str(model_path), task="detect")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None

        # Warmup phase
        print(f"Warming up ({self.num_warmup} runs)...")
        warmup_start = time.perf_counter()
        for i in range(self.num_warmup):
            frame = self._get_frame()
            if frame is None:
                print(f"\n❌ Failed to read frame during warmup")
                return None
            if self.use_video_source:
                frame = self._letterbox_resize(frame)
            _ = model.predict(frame, conf=0.25, imgsz=self.input_size, verbose=False)
            print(f"  Warmup {i+1}/{self.num_warmup}", end='\r')
        warmup_time = (time.perf_counter() - warmup_start) * 1000
        print(f"\n✓ Warmup completed in {warmup_time:.1f}ms")

        # Benchmark phase
        print(f"\nRunning benchmark ({self.num_runs} iterations)...")
        if self.show:
            print("  Press 'q' to abort benchmark")

        latencies = []
        total_detections = 0

        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        for i in range(self.num_runs):
            frame = self._get_frame()
            if frame is None:
                print(f"\n❌ Failed to read frame at iteration {i+1}")
                break

            # Resize frame if using video source
            if self.use_video_source:
                processed_frame = self._letterbox_resize(frame)
            else:
                processed_frame = frame

            start = time.perf_counter()
            results = model.predict(processed_frame, conf=0.25, imgsz=self.input_size, verbose=True)
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

            # Count detections
            num_dets = 0
            for r in results:
                num_dets += len(r.boxes)
                total_detections += len(r.boxes)

            # Show visualization if enabled
            if self.show:
                # Use YOLO's annotated frame directly
                vis_frame = results[0].plot() if results else processed_frame.copy()

                # Add benchmark info overlay
                fps_current = 1000 / latency_ms
                fps_avg = 1000 / np.mean(latencies)
                info_text = [
                    f"Model: {model_path.stem}",
                    f"Frame: {i+1}/{self.num_runs}",
                    f"FPS: {fps_current:.1f} (avg: {fps_avg:.1f})",
                    f"Latency: {latency_ms:.1f}ms",
                    f"Detections: {num_dets}"
                ]

                y_offset = 30
                for text in info_text:
                    cv2.putText(vis_frame, text, (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_offset += 25

                cv2.imshow(f"Benchmark: {model_path.stem}", vis_frame)

                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(f"\n⚠️  Benchmark aborted by user at iteration {i+1}")
                    break

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{self.num_runs} | "
                      f"Avg: {np.mean(latencies):.1f}ms | "
                      f"FPS: {1000/np.mean(latencies):.1f}", end='\r')

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = mem_after - mem_before

        # Close visualization window
        if self.show:
            cv2.destroyAllWindows()

        # Calculate statistics
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        avg_fps = 1000 / avg_latency

        print(f"\n")
        print(f"✓ Benchmark completed")
        print(f"  Avg FPS: {avg_fps:.2f}")
        print(f"  Avg Latency: {avg_latency:.2f}ms")
        print(f"  Min Latency: {min_latency:.2f}ms")
        print(f"  Max Latency: {max_latency:.2f}ms")
        print(f"  Memory Delta: {memory_used:.2f}MB")
        print(f"  Total Detections: {total_detections}")

        return BenchmarkResult(
            model_name=model_path.stem if model_format == 'onnx' else model_path.name,
            model_format=model_format,
            model_path=str(model_path),
            avg_fps=round(avg_fps, 2),
            avg_latency_ms=round(avg_latency, 2),
            min_latency_ms=round(min_latency, 2),
            max_latency_ms=round(max_latency, 2),
            memory_mb=round(memory_used, 2),
            num_detections=total_detections,
            warmup_time_ms=round(warmup_time, 2),
            total_frames=self.num_runs,
            inference_only_ms=round(avg_latency, 2)  # Ultralytics includes pre/post
        )

    def run_all_benchmarks(self, model_dir: Path) -> List[BenchmarkResult]:
        """Run benchmarks on all discovered models."""
        models = self.find_models(model_dir)
        results = []

        print(f"\n🔍 Discovered Models:")
        print(f"  ONNX: {len(models['onnx'])} models")
        print(f"  NCNN: {len(models['ncnn'])} models")

        # Benchmark ONNX models
        for model_path in sorted(models['onnx']):
            result = self.benchmark_model(model_path, 'onnx')
            if result:
                results.append(result)

        # Benchmark NCNN models
        for model_path in sorted(models['ncnn']):
            result = self.benchmark_model(model_path, 'ncnn')
            if result:
                results.append(result)

        return results

    def cleanup(self):
        """Release video capture resources."""
        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()

    def save_results(self, results: List[BenchmarkResult], output_path: Path):
        """Save benchmark results to JSON."""
        results_dict = [asdict(r) for r in results]

        with open(output_path, 'w') as f:
            json.dump({
                'benchmark_config': {
                    'num_warmup': self.num_warmup,
                    'num_runs': self.num_runs,
                    'input_size': self.input_size,
                    'video_source': 'real' if self.use_video_source else 'synthetic',
                    'hardware': 'Raspberry Pi 4 Model B (4GB)',
                    'backend': 'CPU'
                },
                'results': results_dict
            }, f, indent=2)

        print(f"\n💾 Results saved to {output_path}")

    def print_summary(self, results: List[BenchmarkResult]):
        """Print a formatted summary table."""
        if not results:
            print("\n❌ No results to display")
            return

        print(f"\n{'='*80}")
        print(f"BENCHMARK SUMMARY (Target: 5-10 FPS on Pi4)")
        print(f"{'='*80}")

        # Sort by FPS (descending)
        results_sorted = sorted(results, key=lambda x: x.avg_fps, reverse=True)

        # Print header
        print(f"{'Model':<25} {'Format':<8} {'FPS':<8} {'Latency':<12} {'Memory':<10} {'Status'}")
        print(f"{'-'*80}")

        for r in results_sorted:
            status = "✅ PASS" if r.avg_fps >= 5 else "❌ FAIL"
            print(f"{r.model_name:<25} {r.model_format.upper():<8} "
                  f"{r.avg_fps:<8.2f} {r.avg_latency_ms:<12.2f} "
                  f"{r.memory_mb:<10.2f} {status}")

        print(f"{'-'*80}")

        # Print best performers
        best_overall = results_sorted[0]
        best_onnx = next((r for r in results_sorted if r.model_format == 'onnx'), None)
        best_ncnn = next((r for r in results_sorted if r.model_format == 'ncnn'), None)

        print(f"\n🏆 Best Overall: {best_overall.model_name} ({best_overall.avg_fps:.2f} FPS)")
        if best_onnx:
            print(f"🥇 Best ONNX: {best_onnx.model_name} ({best_onnx.avg_fps:.2f} FPS)")
        if best_ncnn:
            print(f"🥇 Best NCNN: {best_ncnn.model_name} ({best_ncnn.avg_fps:.2f} FPS)")

        # Count passing models
        passing = sum(1 for r in results if r.avg_fps >= 5)
        print(f"\n✓ {passing}/{len(results)} models meet the 5 FPS minimum target")


def main():
    """Main benchmark execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark YOLO models for Pi4 CPU')
    parser.add_argument('--model-dir', type=str, default='assets/model',
                        help='Directory containing YOLO models')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations')
    parser.add_argument('--runs', type=int, default=100,
                        help='Number of benchmark iterations')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output JSON file for results')
    parser.add_argument('--source', type=str, default=None,
                        help='Video source: webcam index (0, 1, ...) or video file path. '
                             'If not specified, uses synthetic test frames.')
    parser.add_argument('--show', action='store_true',
                        help='Display real-time detection results during benchmark')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size (width=height). Common: 320, 416, 640 (default)')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return

    print("="*80)
    print("YOLO Model Benchmark for Raspberry Pi 4 (CPU)")
    print("="*80)
    print(f"Model Directory: {model_dir}")
    print(f"Input Size: {args.imgsz}x{args.imgsz}")
    print(f"Warmup Iterations: {args.warmup}")
    print(f"Benchmark Iterations: {args.runs}")
    print(f"Target Performance: 5-10 FPS")

    if args.source:
        source_desc = f"Webcam {args.source}" if args.source.isdigit() else args.source
        print(f"Video Source: {source_desc}")
    else:
        print(f"Video Source: Synthetic test frames")

    print(f"Visualization: {'Enabled' if args.show else 'Disabled'}")

    # Run benchmarks
    try:
        benchmark = ModelBenchmark(
            num_warmup=args.warmup,
            num_runs=args.runs,
            source=args.source,
            show=args.show,
            imgsz=args.imgsz
        )
        results = benchmark.run_all_benchmarks(model_dir)

        # Display summary
        benchmark.print_summary(results)

        # Save results
        output_path = Path(args.output)
        benchmark.save_results(results, output_path)

        print("\n✅ Benchmark complete!")

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup resources
        if 'benchmark' in locals():
            benchmark.cleanup()


if __name__ == "__main__":
    main()

"""
benchmark.py — Evaluate model speed, accuracy, and export performance.

Usage:
  python benchmark.py --model yolo11n.pt
  python benchmark.py --model yolo11n.pt --data coco.yaml --device cuda
  python benchmark.py --model best.pt --export onnx tensorrt
"""

import argparse
import time
import logging
import json
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark YOLO model speed & accuracy")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--data", default=None, help="Dataset YAML for mAP evaluation")
    p.add_argument("--device", default="auto")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--runs", type=int, default=100, help="Inference runs for speed test")
    p.add_argument("--warmup-runs", type=int, default=10)
    p.add_argument("--export", nargs="+", default=[],
                   choices=["onnx", "tflite", "coreml", "engine"],
                   help="Formats to export")
    p.add_argument("--out", default="data/outputs/benchmark.json")
    return p.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def benchmark_speed(model, device: str, imgsz: int, runs: int, warmup: int) -> dict:
    """Measure pure inference latency."""
    import numpy as np
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    logger.info(f"Warming up ({warmup} runs)...")
    for _ in range(warmup):
        model(dummy, device=device, verbose=False)

    logger.info(f"Benchmarking ({runs} runs)...")
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(dummy, device=device, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms": round(float(times.mean()), 2),
        "median_ms": round(float(np.median(times)), 2),
        "std_ms": round(float(times.std()), 2),
        "min_ms": round(float(times.min()), 2),
        "max_ms": round(float(times.max()), 2),
        "fps": round(1000 / float(times.mean()), 1),
        "runs": runs,
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(args.model)
    logger.info(f"Model: {args.model} | Device: {device} | imgsz: {args.imgsz}")

    report = {
        "model": args.model,
        "device": device,
        "imgsz": args.imgsz,
    }

    # Speed benchmark
    speed = benchmark_speed(model, device, args.imgsz, args.runs, args.warmup_runs)
    report["speed"] = speed
    logger.info(
        f"Speed — mean: {speed['mean_ms']} ms | FPS: {speed['fps']} | "
        f"std: {speed['std_ms']} ms"
    )

    # Accuracy (mAP)
    if args.data:
        logger.info(f"Evaluating mAP on {args.data}...")
        val = model.val(data=args.data, device=device, imgsz=args.imgsz)
        report["accuracy"] = {
            "mAP50": round(val.results_dict.get("metrics/mAP50(B)", 0), 4),
            "mAP50_95": round(val.results_dict.get("metrics/mAP50-95(B)", 0), 4),
            "precision": round(val.results_dict.get("metrics/precision(B)", 0), 4),
            "recall": round(val.results_dict.get("metrics/recall(B)", 0), 4),
        }
        logger.info(f"mAP50: {report['accuracy']['mAP50']} | "
                    f"mAP50-95: {report['accuracy']['mAP50_95']}")

    # Model info
    info = model.info(verbose=False)
    report["model_info"] = {
        "parameters": getattr(info, "n", "N/A"),
        "layers": getattr(info, "n_l", "N/A"),
        "gflops": getattr(info, "gflops", "N/A"),
    }

    # Export
    for fmt in args.export:
        logger.info(f"Exporting to {fmt}...")
        try:
            out = model.export(format=fmt, device=device, imgsz=args.imgsz)
            report.setdefault("exports", {})[fmt] = str(out)
            logger.info(f"Exported → {out}")
        except Exception as e:
            logger.warning(f"Export {fmt} failed: {e}")
            report.setdefault("exports", {})[fmt] = f"FAILED: {e}"

    # Save report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved → {out_path}")

    # Print summary table
    print("\n" + "═" * 50)
    print(f"  BENCHMARK REPORT — {Path(args.model).stem}")
    print("═" * 50)
    print(f"  Device      : {device.upper()}")
    print(f"  Mean latency: {speed['mean_ms']} ms")
    print(f"  Throughput  : {speed['fps']} FPS")
    if "accuracy" in report:
        print(f"  mAP50       : {report['accuracy']['mAP50']}")
        print(f"  mAP50-95    : {report['accuracy']['mAP50_95']}")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    main()

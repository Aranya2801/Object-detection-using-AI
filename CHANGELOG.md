# Changelog

All notable changes are documented here.

## [2.0.0] — 2025-05-01

### 🚀 Major Overhaul — Production-Ready Rewrite

#### Added
- **YOLO11 support** (latest 2024 model — best accuracy/speed balance)
- **YOLOv9 & YOLOv10** support alongside YOLOv8
- **Multi-object tracking** via ByteTrack and BoTSORT
- **Instance segmentation** with mask overlay rendering
- **Pose estimation** task support
- **Oriented Bounding Boxes (OBB)** task
- `ObjectDetector` class with full Python API
- `Detection` and `DetectionResult` dataclasses with serialization
- **Analytics logger** — per-frame JSON + CSV exports
- **Zone counting** — ROI polygon counting & tripwires
- **Benchmark script** — latency, FPS, mAP evaluation
- **Batch processing** script for image folders
- **Docker support** — GPU + CPU Dockerfiles
- **GitHub Actions CI/CD** pipeline (Python 3.10/3.11/3.12)
- **Pytest test suite** with unit + integration tests
- Hardware-aware device selection (CUDA / MPS / CPU auto)
- Rolling FPS HUD overlay
- ONNX / TensorRT / CoreML export support
- Full YAML configuration files

#### Changed
- Replaced raw OpenCV+YOLOv3 pipeline with Ultralytics unified API
- Modular architecture: `src/`, `utils/`, `configs/`, `scripts/`

#### Removed
- Legacy `main.py` / `tempCodeRunnerFile.py` (replaced by modular system)

---

## [1.0.0] — 2023

### Initial Release
- YOLOv3 object detection using OpenCV DNN
- Basic image and video support
- CLI with `--image` / `--video` / `--output` flags

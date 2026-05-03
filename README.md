<div align="center">

<img src="https://raw.githubusercontent.com/ultralytics/assets/main/yolo/performance-comparison.png" alt="YOLO Performance" width="800"/>

# 🎯 Object Detection using AI

### Production-grade real-time object detection powered by **YOLO11**, **YOLOv8**, and the Ultralytics ecosystem

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyeiIvPjwvc3ZnPg==)](https://ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Aranya2801/Object-detection-using-AI/ci.yml?style=for-the-badge&label=CI)](https://github.com/Aranya2801/Object-detection-using-AI/actions)

[**Quick Start**](#-quick-start) · [**Features**](#-features) · [**Models**](#-supported-models) · [**Usage**](#-usage) · [**Training**](#-custom-training) · [**Docker**](#-docker)

</div>

---

## 📖 Overview

This project is a **daily-use, production-ready** AI object detection system that brings cutting-edge computer vision to your fingertips. Built on the latest **YOLO11** (2024) and **YOLOv8** architectures via [Ultralytics](https://ultralytics.com), it detects **80+ object classes** in real time from webcams, videos, images, and RTSP streams — with optional **multi-object tracking**, **instance segmentation**, **pose estimation**, and **zone-based counting**.

Whether you're building a **security system**, **retail analytics dashboard**, **smart home automation**, or just exploring AI — this project has you covered.

---

## ✨ Features

| Category | Feature |
|----------|---------|
| 🤖 **Models** | YOLO11, YOLOv8, YOLOv9, YOLOv10, custom `.pt` |
| 🎥 **Sources** | Webcam, video files, images, folders, RTSP/HTTP streams |
| 🏃 **Tracking** | ByteTrack & BoTSORT multi-object tracking |
| 🎭 **Tasks** | Detection · Segmentation · Pose · Classification · OBB |
| 📊 **Analytics** | Per-frame JSON logs, CSV stats, FPS/latency HUD |
| 🗺️ **Zones** | ROI zone counting and virtual tripwires |
| 📦 **Export** | ONNX, TensorRT, CoreML, TFLite |
| 🐳 **Docker** | Full GPU/CPU Docker support |
| 🧪 **Testing** | Pytest unit + integration tests with CI/CD |
| 💾 **Training** | Fine-tune on custom datasets with one command |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Aranya2801/Object-detection-using-AI.git
cd Object-detection-using-AI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run on Webcam (30 seconds to first detection!)

```bash
python detect.py --source 0
```

### 3. Run on a Video File

```bash
python detect.py --source path/to/video.mp4 --track --save
```

### 4. Run on an Image

```bash
python detect.py --source path/to/image.jpg --model yolo11m.pt
```

---

## 🧠 Supported Models

All models auto-download on first use from Ultralytics servers.

### YOLO11 Family (Latest — 2024) ⭐ *Recommended*

| Model | Size | mAP<sup>val<br>50-95 | Speed CPU<br>ONNX (ms) | Params (M) | FLOPs (B) |
|-------|------|----------------------|------------------------|-----------|-----------|
| **YOLO11n** | 640 | 39.5 | 56.1 | 2.6 | 6.5 |
| **YOLO11s** | 640 | 47.0 | 90.0 | 9.4 | 21.5 |
| **YOLO11m** | 640 | 51.5 | 183.2 | 20.1 | 68.0 |
| **YOLO11l** | 640 | 53.4 | 238.6 | 25.3 | 86.9 |
| **YOLO11x** | 640 | 54.7 | 462.8 | 56.9 | 194.9 |

### YOLOv8 Family (2023)

| Model | mAP50-95 | CPU (ms) | GPU (ms) | Params |
|-------|---------|----------|----------|--------|
| YOLOv8n | 37.3 | 80.4 | 0.99 | 3.2M |
| YOLOv8s | 44.9 | 128.4 | 1.20 | 11.2M |
| YOLOv8m | 50.2 | 234.7 | 1.83 | 25.9M |
| YOLOv8l | 52.9 | 375.2 | 2.39 | 43.7M |
| YOLOv8x | 53.9 | 479.1 | 3.53 | 68.2M |

> 💡 **Tip:** Use `yolo11n.pt` for real-time webcam. Use `yolo11l.pt` for maximum accuracy on saved video.

---

## 🎛️ Usage

### Command-Line Interface

```bash
# Webcam with tracking
python detect.py --source 0 --track

# Video with custom model + save output
python detect.py --source input.mp4 --model yolo11m.pt --save --save-json

# Filter only people and cars
python detect.py --source 0 --classes 0 2

# Instance segmentation
python detect.py --source video.mp4 --model yolo11n-seg.pt --task segment

# Pose estimation
python detect.py --source 0 --model yolo11n-pose.pt --task pose

# RTSP stream
python detect.py --source rtsp://192.168.1.100/stream --model yolo11n.pt

# Batch image folder
python scripts/batch_detect.py --input data/samples/ --save-json
```

### Full CLI Reference

```
Arguments:
  --source      Input source (0=webcam, path, URL, RTSP)
  --model       Model: yolo11n/s/m/l/x.pt or custom .pt
  --task        detect | segment | pose | classify | obb
  --conf        Confidence threshold [default: 0.40]
  --iou         NMS IoU threshold [default: 0.45]
  --device      auto | cpu | cuda | cuda:0 | mps
  --track       Enable ByteTrack multi-object tracking
  --tracker     bytetrack.yaml | botsort.yaml
  --classes     Filter class IDs (e.g. 0 2 5)
  --save        Save annotated output to data/outputs/
  --save-json   Export detection data as JSON
  --save-csv    Append per-frame stats to CSV
  --zone        Enable interactive ROI zone counting
  --fps-limit   Cap processing FPS (0 = unlimited)
  --warmup      Warm up GPU before inference
  --no-show     Disable display window (headless/server)
```

### Python API

```python
from src.detector import ObjectDetector
import cv2

# Initialize with latest YOLO11
detector = ObjectDetector(
    model_name="yolo11n.pt",
    conf_threshold=0.45,
    track=True,          # Enable ByteTrack
    device="auto",       # Auto-selects CUDA/MPS/CPU
)
detector.warmup()

# Run on an image
frame = cv2.imread("street.jpg")
result = detector.predict(frame)

print(f"Detected {result.count} objects")
print(f"Inference: {result.inference_time_ms:.1f} ms")
print(f"Classes: {result.class_counts()}")

# Annotate and display
annotated = detector.draw(frame, result)
cv2.imshow("Result", annotated)
cv2.waitKey(0)
```

---

## 📦 Project Structure

```
Object-detection-using-AI/
├── 📄 detect.py              # Main CLI — webcam / video / image inference
├── 📄 train.py               # Custom model training
├── 📄 benchmark.py           # Speed & accuracy evaluation
│
├── 📁 src/
│   ├── detector.py           # Core ObjectDetector engine (YOLO11/v8/v9/v10)
│   └── __init__.py
│
├── 📁 utils/
│   ├── io.py                 # Video/image I/O helpers
│   ├── zone.py               # ROI zone counting & tripwires
│   ├── analytics.py          # Detection logging (JSON/CSV)
│   └── __init__.py
│
├── 📁 configs/
│   ├── default.yaml          # Default inference settings
│   └── train.yaml            # Training hyperparameters
│
├── 📁 scripts/
│   ├── quick_start.py        # One-file webcam demo
│   └── batch_detect.py       # Batch folder processing
│
├── 📁 tests/
│   └── test_detector.py      # Pytest unit & integration tests
│
├── 📁 data/
│   ├── samples/              # Put test images here
│   └── outputs/              # Annotated results saved here
│
├── 📁 .github/workflows/
│   └── ci.yml                # GitHub Actions CI/CD
│
├── 🐳 Dockerfile             # GPU + CPU Docker builds
├── 📄 requirements.txt
└── 📄 setup.py
```

---

## 🏋️ Custom Training

Fine-tune YOLO11 on your own dataset in minutes:

### Step 1: Prepare Your Dataset

Use [Roboflow](https://roboflow.com) to annotate and export in YOLO format. Your `data/custom_dataset.yaml` should look like:

```yaml
path: data/my_dataset
train: images/train
val: images/val
test: images/test

nc: 3                     # number of classes
names: ['cat', 'dog', 'bird']
```

### Step 2: Train

```bash
python train.py \
  --data data/custom_dataset.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --batch 16 \
  --device cuda
```

### Step 3: Use Your Custom Model

```bash
python detect.py --source 0 --model runs/train/custom_model/weights/best.pt
```

---

## 📊 Benchmark Your Model

```bash
# Speed test only
python benchmark.py --model yolo11n.pt --runs 200

# Speed + accuracy (requires COCO dataset)
python benchmark.py --model yolo11n.pt --data coco.yaml --device cuda

# Export to ONNX and benchmark
python benchmark.py --model yolo11n.pt --export onnx
```

---

## 🐳 Docker

### CPU Build
```bash
docker build -t od-ai:cpu .
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  od-ai:cpu \
  python detect.py --source data/samples/ --no-show --save
```

### GPU Build (NVIDIA)
```bash
docker build -t od-ai:gpu .
docker run -it --rm --gpus all \
  -v $(pwd)/data:/app/data \
  od-ai:gpu \
  python detect.py --source 0 --model yolo11m.pt
```

---

## 🗺️ COCO Class Reference (80 Classes)

The default pre-trained models detect all 80 COCO classes:

```
0: person        1: bicycle      2: car           3: motorcycle   4: airplane
5: bus           6: train        7: truck          8: boat         9: traffic light
10: fire hydrant 11: stop sign   12: parking meter 13: bench       14: bird
15: cat          16: dog         17: horse         18: sheep       19: cow
20: elephant     21: bear        22: zebra         23: giraffe     24: backpack
25: umbrella     26: handbag     27: tie           28: suitcase    29: frisbee
30: skis         31: snowboard   32: sports ball   33: kite        34: baseball bat
35: baseball glove 36: skateboard 37: surfboard   38: tennis racket 39: bottle
40: wine glass   41: cup         42: fork          43: knife       44: spoon
45: bowl         46: banana      47: apple         48: sandwich    49: orange
50: broccoli     51: carrot      52: hot dog       53: pizza       54: donut
55: cake         56: chair       57: couch         58: potted plant 59: bed
60: dining table 61: toilet      62: tv            63: laptop      64: mouse
65: remote       66: keyboard    67: cell phone    68: microwave   69: oven
70: toaster      71: sink        72: refrigerator  73: book        74: clock
75: vase         76: scissors    77: teddy bear    78: hair drier  79: toothbrush
```

---

## 📈 Architecture Overview

```
Input Source (Webcam / Video / Image / RTSP)
        │
        ▼
┌───────────────────────────────┐
│      ObjectDetector           │
│  ┌─────────────────────────┐  │
│  │  YOLO11 / YOLOv8 model  │  │
│  │  Backbone: C3k2 + C2PSA │  │
│  │  Neck: SPPF + FPN/PAN   │  │
│  │  Head: Decoupled, AnchorFree │
│  └─────────────────────────┘  │
│  NMS → Detection[]            │
│  SpeedTracker (rolling FPS)   │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Optional Post-Processing     │
│  ├── ByteTrack / BoTSORT      │
│  ├── ZoneCounter (ROI)        │
│  └── AnalyticsLogger          │
└───────────────────────────────┘
        │
        ▼
  Annotated Frame + JSON/CSV
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Run tests: `pytest tests/ -v`
4. Submit a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — YOLO11 / YOLOv8 framework
- [OpenCV](https://opencv.org) — Image processing backbone
- [PyTorch](https://pytorch.org) — Deep learning framework
- [COCO Dataset](https://cocodataset.org) — Pre-training benchmark

---

<div align="center">

**Built with ❤️ by [Aranya2801](https://github.com/Aranya2801)**

*If this project helped you, please ⭐ star the repo!*

</div>

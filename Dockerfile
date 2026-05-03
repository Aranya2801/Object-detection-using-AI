# ─────────────────────────────────────────────────────────────────
# Dockerfile — Object Detection AI
# Supports CPU and CUDA (GPU) inference
# ─────────────────────────────────────────────────────────────────

# Use CUDA base for GPU support, or python:3.11-slim for CPU-only
ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04 AS base

LABEL maintainer="Aranya2801"
LABEL description="Advanced Object Detection using YOLO11/YOLOv8"
LABEL version="2.0.0"

# ── System deps ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    wget curl git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ── Working dir ──────────────────────────────────────────────────
WORKDIR /app

# ── Python deps ──────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url \
        https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project ─────────────────────────────────────────────────
COPY . .

# ── Create output dirs ───────────────────────────────────────────
RUN mkdir -p data/outputs data/samples

# ── Pre-download YOLO11n model ───────────────────────────────────
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# ── Expose port for potential API server ─────────────────────────
EXPOSE 8000

# ── Default command ──────────────────────────────────────────────
CMD ["python", "detect.py", "--source", "0", "--model", "yolo11n.pt", "--no-show", "--save"]

# ─────────────────────────────────────────────────────────────────
# CPU-only build:
#   docker build --build-arg CUDA_VERSION=cpu -t od-ai:cpu .
#   FROM python:3.11-slim AS base   (replace nvidia base above)
#
# GPU build:
#   docker build -t od-ai:gpu .
#   docker run --gpus all od-ai:gpu
# ─────────────────────────────────────────────────────────────────

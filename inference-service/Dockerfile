# Base image ships Python + PyTorch + CUDA/cuDNN preinstalled, avoiding a
# hand-matched torch-wheel-to-CUDA-version installation (a common source of
# GPU-not-detected bugs). Confirmed compatible 2026-08: gpu-ml's driver
# supports CUDA 13.0 (via nvidia-smi), well above this image's CUDA 12.4.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# libgl1/libglib2.0-0/libsm6/libxext6/libxrender1/libxcb1: ultralytics
# depends on opencv-python (the GUI-capable build), which needs these X11
# shared libraries even for headless use -- this minimal PyTorch runtime
# image doesn't include them by default. Without them, importing
# ultralytics fails at container startup with "ImportError: libxcb.so.1:
# cannot open shared object file" -- confirmed via a live crash 2026-08.
#
# git: needed for `pip install git+https://github.com/openai/CLIP.git` in
# requirements.txt (YOLO-World's set_classes() requires real OpenAI CLIP,
# not the unrelated PyPI package literally named "clip" -- confirmed via a
# live "No module named 'clip'" failure, 2026-08). pip needs the git binary
# on the system to clone that URL during install.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3005

# --workers 1 is deliberate, not a default left unconfigured: each worker
# would load its own copy of the YOLO-World model into this device's shared
# 6GB VRAM (already split with immich-machine-learning and ollama -- see
# README's VRAM contention note). A second worker could OOM the GPU.
# --timeout 300: model weights download on first request (lazy-loaded, see
# tasks/object_detect.py) plus inference can exceed Gunicorn's 30s default.
CMD ["gunicorn", "--bind", "0.0.0.0:3005", "--workers", "1", "--timeout", "300", "app:app"]

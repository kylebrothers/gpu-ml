# Base image ships Python + PyTorch + CUDA/cuDNN preinstalled, avoiding a
# hand-matched torch-wheel-to-CUDA-version installation (a common source of
# GPU-not-detected bugs). Confirmed compatible 2026-08: gpu-ml's driver
# supports CUDA 13.0 (via nvidia-smi), well above this image's CUDA 12.4.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# ultralytics depends on opencv-python (the GUI-capable build), which needs
# X11 shared libraries this minimal PyTorch runtime image doesn't include
# (it's built for headless training/inference, no GUI stack). Without
# these, importing ultralytics fails at container startup with
# "ImportError: libxcb.so.1: cannot open shared object file" -- confirmed
# via a live crash 2026-08. libgl1/libglib2.0-0/libsm6/libxext6/
# libxrender1/libxcb1 is the standard set opencv-python's shared-library
# loading needs on a Debian-based image; none of these pull in an actual
# GUI/display stack, they're just the .so files the import machinery checks
# for even in headless use.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
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

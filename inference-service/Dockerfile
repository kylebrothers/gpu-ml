# Base image ships Python + PyTorch + CUDA/cuDNN preinstalled, avoiding a
# hand-matched torch-wheel-to-CUDA-version installation (a common source of
# GPU-not-detected bugs). UNVERIFIED, flag for Kyle to confirm before first
# build: this CUDA version (12.4) must be compatible with the actual NVIDIA
# driver installed on the gpu-ml host -- check with `nvidia-smi` on that
# device (top-right "CUDA Version" is the max supported by the driver, must
# be >= this image's CUDA version) before building. If incompatible, switch
# to a pytorch/pytorch tag with an older cuda## suffix that fits.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

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

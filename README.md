# gpu-ml

Shared GPU-accelerated inference services for home network projects, running on a device with an NVIDIA GPU (currently a GTX 1060, 6GB VRAM). Kept as its own repo, separate from any single client project, because it's expected to serve multiple projects over time.

## Current services

- **immich-machine-learning** — CLIP embedding generation and face detection/recognition backend for Immich. Used by the `photo-search` project's `immich-server`, which points at this device via `IMMICH_MACHINE_LEARNING_URL`.

## Adding a new service

Add a new service block to `docker-compose.yml` with its own port, following the existing pattern. Keep services independent of each other — a bad deploy of one shouldn't take down another project's inference backend.

## Requirements

- NVIDIA GPU with `nvidia-container-toolkit` installed on the host
- Docker + Docker Compose

## Deployment

```
make up      # start all services
make down    # stop all services
make logs    # tail logs
make status  # container status
```

# gpu-ml

Shared GPU-accelerated inference services for home network projects, running on a device with an NVIDIA GPU (currently a GTX 1060, 6GB VRAM). Kept as its own repo, separate from any single client project, because it's expected to serve multiple projects over time.

## Current services

- **immich-machine-learning** — CLIP embedding generation and face detection/recognition backend for Immich. Used by the `photo-search` project's `immich-server`, which points at this device via `IMMICH_MACHINE_LEARNING_URL`.
- **ollama** — local LLM inference, used by `photo-search`'s `search-api` for natural-language query parsing (`query_parser_llm.py`), grounded with real data pulled from Immich at request time. Model: `llama3.2:3b` (~2GB at 4-bit) — chosen deliberately small given this device's 6GB VRAM is shared with `immich-machine-learning`, which can spike during embedding backfills. `keep_alive` is set short (5 minutes) by the caller so the model unloads between requests rather than sitting resident in VRAM indefinitely.

**First-time setup after `make up`:** pull the model into the container (not baked into the image):
```
docker exec -it ollama ollama pull llama3.2:3b
```

**Caveat, not silently resolved:** both services share one 6GB GPU. Under simultaneous heavy load (a large Immich embedding backfill running at the same time as LLM query parsing), you could see contention or OOM — not yet stress-tested. If this becomes a problem, options include a smaller/faster model, tighter `keep_alive`, or scheduling backfills to avoid overlapping with expected search usage.

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

# gpu-ml

Shared GPU-accelerated inference services for home network projects, running on a device with an NVIDIA GPU (currently a GTX 1060, 6GB VRAM). Kept as its own repo, separate from any single client project, because it's expected to serve multiple projects over time.

## Current services

- **immich-machine-learning** — CLIP embedding generation and face detection/recognition backend for Immich. Used by the `photo-search` project's `immich-server`, which points at this device via `IMMICH_MACHINE_LEARNING_URL`.
- **ollama** — local LLM inference, used by `photo-search`'s `search-api` for natural-language query parsing (`query_parser_llm.py`), grounded with real data pulled from Immich at request time. Model: `llama3.2:3b` (~2GB at 4-bit) — chosen deliberately small given this device's 6GB VRAM is shared with `immich-machine-learning`, which can spike during embedding backfills. `keep_alive` is set short (5 minutes) by the caller so the model unloads between requests rather than sitting resident in VRAM indefinitely.
- **inference-service** — generic, reusable GPU inference service, built here (not an off-the-shelf image like the two above). Exposes a task-based HTTP protocol rather than one endpoint per model, specifically so future models don't each need their own service:
  - `GET /health` — status check
  - `GET /v1/tasks` — lists registered tasks and each one's current `model_version`
  - `POST /v1/infer/<task_name>` — multipart form: an `image` file plus a JSON `params` field (task-specific). Returns `{"task", "model_version", "result"}`.

  The service has no knowledge of Immich, Dropbox, or any specific client project — callers send raw image bytes, not an asset ID or URL, and are responsible for fetching/resizing images themselves. This keeps the service genuinely reusable across projects, matching this repo's whole reason for existing separately.

  Tasks are registered in `inference-service/tasks/__init__.py`; adding a new model means adding a new task module there, not building new infrastructure. First task: **object_detect** (YOLO-World, open-vocabulary), used by `photo-search`'s `sidecar/enrichment/object_detect.py`. Model choice, VRAM/worker-count reasoning, and a documented Ultralytics `set_classes()` bug workaround are all in `inference-service/tasks/object_detect.py`'s docstring.

**First-time setup after `make up`:** pull the Ollama model into the container (not baked into the image):
```
docker exec -it ollama ollama pull llama3.2:3b
```

**Caveat, not silently resolved:** all three services share one 6GB GPU. Under simultaneous heavy load (e.g. an Immich embedding backfill running at the same time as LLM query parsing or an object-detection batch), you could see contention or OOM — not yet stress-tested. If this becomes a problem, options include a smaller/faster model, tighter `keep_alive`/timeout settings, or scheduling batch enrichment jobs to avoid overlapping with expected interactive usage.

## Adding a new service

Two paths, depending on what you're adding:
- **A genuinely new kind of service** (not GPU inference, or an off-the-shelf image like the two above): add a new service block to `docker-compose.yml` with its own port, following the existing pattern. Keep services independent of each other — a bad deploy of one shouldn't take down another project's inference backend.
- **A new GPU inference model**: prefer adding it as a new task to `inference-service/tasks/` instead of a new service — reuses the existing protocol, container, and port, and avoids yet another slice of the shared 6GB VRAM budget sitting idle in its own process.

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

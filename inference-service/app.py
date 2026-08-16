"""
Generic GPU inference service. Not model-specific: exposes a reusable
task-based protocol so new models (dense captioning, DELF/DELG landmark
embeddings, etc. -- see photo-search/docs/sidecar-augmentation.md) register
as new tasks without any new service, protocol, or client-integration work.

Protocol:
  GET  /health              -> {"status": "ok"}
  GET  /v1/tasks             -> {"tasks": {"<name>": {"model_version": "..."}}}
  POST /v1/infer/<task_name> -> multipart/form-data:
                                   "image": raw image bytes (file)
                                   "params": JSON string (task-specific)
                               -> {"task": ..., "model_version": ..., "result": {...}}

Callers send raw image bytes, not an asset_id or URL -- this service has no
knowledge of Immich, Dropbox, or any specific client project (deliberate:
see gpu-ml/README.md, "expected to serve multiple projects over time").
The caller is responsible for fetching/resizing the image before sending it.

Single-process, single-worker by design (see Dockerfile CMD) -- multiple
workers would each load their own model copy into the shared 6GB VRAM
budget, which this device cannot afford (see README's VRAM contention note).
"""
import io
import json
import logging

from flask import Flask, request, jsonify

from tasks import TASK_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/v1/tasks")
def list_tasks():
    return jsonify({
        "tasks": {
            name: {"model_version": task.model_version()}
            for name, task in TASK_REGISTRY.items()
        }
    })


@app.route("/v1/infer/<task_name>", methods=["POST"])
def infer(task_name):
    task = TASK_REGISTRY.get(task_name)
    if task is None:
        return jsonify({
            "error": f"unknown task {task_name!r}",
            "available_tasks": list(TASK_REGISTRY.keys()),
        }), 404

    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file in multipart form"}), 400

    image_bytes = request.files["image"].read()

    params = {}
    raw_params = request.form.get("params")
    if raw_params:
        try:
            params = json.loads(raw_params)
        except json.JSONDecodeError as e:
            return jsonify({"error": f"'params' is not valid JSON: {e}"}), 400

    try:
        result = task.infer(image_bytes, params)
    except Exception as e:
        logger.exception(f"inference failed for task {task_name!r}")
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "task": task_name,
        "model_version": task.model_version(),
        "result": result,
    })


if __name__ == "__main__":
    # Dev-only entry point; real deployment uses gunicorn (see Dockerfile).
    app.run(host="0.0.0.0", port=3005)

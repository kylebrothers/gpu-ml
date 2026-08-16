"""
Task registry. Each task module defines a handler class implementing:
  .infer(image_bytes: bytes, params: dict) -> dict   (the JSON-serializable result)
  .model_version() -> str

Adding a new task (e.g. a future dense-captioning or landmark-embedding
model): create tasks/<new_task>.py with a handler class, then register it
below. No changes to app.py or the protocol itself are needed -- this is
the reusability the task-registry pattern is for.
"""
from .object_detect import ObjectDetectTask

TASK_REGISTRY = {
    "object_detect": ObjectDetectTask(),
}

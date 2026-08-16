"""
YOLO-World open-vocabulary object detection. First registered task, proving
the reusable protocol (see app.py / tasks/__init__.py).

Model choice: yolov8s-worldv2.pt -- Ultralytics explicitly recommends the
"worldv2" variant over v1 for reliable custom-class support and export
(confirmed via docs.ultralytics.com/models/yolo-world, 2026-08). "s" (small)
chosen given this device's 6GB VRAM is shared with immich-machine-learning
and ollama -- see gpu-ml/README's VRAM contention note. Revisit to m/l if
accuracy proves insufficient; no schema change needed, just a new
model_version string.

Known Ultralytics bug (github.com/ultralytics/ultralytics issue #20889):
calling set_classes() again on an already-used model instance can raise
"Inference tensors do not track version counter". Avoided here by caching
which vocabulary_version is currently applied and only calling
set_classes() when it actually changes -- the common case (one enrichment
run = one fixed vocabulary) never re-triggers it.

UNVERIFIED, to confirm on first real run (not yet tested against live
hardware from this chat): that model.names correctly reflects the custom
class list (not the original COCO defaults) after set_classes(), with
indices matching detection results' box.cls values, per Ultralytics' own
documented example usage.
"""
import io
import logging

from PIL import Image
from ultralytics import YOLOWorld

logger = logging.getLogger(__name__)

MODEL_WEIGHTS = "yolov8s-worldv2.pt"
DEFAULT_CONFIDENCE = 0.25


class ObjectDetectTask:
    def __init__(self):
        self._model = None
        self._applied_vocabulary_version = None

    def _ensure_model_loaded(self):
        """Lazy-load on first request, not at import time -- keeps container
        startup fast and avoids loading the model at all if this task is
        never actually called."""
        if self._model is None:
            logger.info(f"loading {MODEL_WEIGHTS}...")
            self._model = YOLOWorld(MODEL_WEIGHTS)
            try:
                import torch
                if torch.cuda.is_available():
                    self._model.to("cuda")
                    logger.info("using CUDA")
                else:
                    logger.warning("CUDA not available, falling back to CPU (will be slow)")
            except ImportError:
                logger.warning("torch not importable for device check; using model's default device")

    def model_version(self):
        """
        Combines the fixed model weights with whatever vocabulary was most
        recently applied -- so a vocabulary change (a new
        OBJECT_DETECT_VOCABULARY_VERSION on the sidecar side) is visible
        here too, not just in the caller's own bookkeeping. Before any
        request has set a vocabulary, reports "unset".
        """
        vocab_tag = self._applied_vocabulary_version or "unset"
        return f"{MODEL_WEIGHTS}:{vocab_tag}"

    def infer(self, image_bytes, params):
        """
        params (required):
          vocabulary: list[str] -- open-vocab class names to detect
          vocabulary_version: str -- caller's version tag for that list,
            used to decide whether set_classes() needs to run again
        params (optional):
          confidence: float, default 0.25

        Returns: {class_name: {"count": int, "avg_confidence": float}, ...}
        Classes with zero detections are simply absent from the result --
        this is a valid, correct outcome (see sidecar's enrichment_status
        table for how "zero detections" is distinguished from "not yet
        processed").
        """
        vocabulary = params.get("vocabulary")
        vocabulary_version = params.get("vocabulary_version")
        if not vocabulary or not vocabulary_version:
            raise ValueError("params must include non-empty 'vocabulary' and 'vocabulary_version'")
        confidence = params.get("confidence", DEFAULT_CONFIDENCE)

        self._ensure_model_loaded()

        if vocabulary_version != self._applied_vocabulary_version:
            logger.info(f"applying new vocabulary (version={vocabulary_version}, {len(vocabulary)} classes)")
            self._model.set_classes(vocabulary)
            self._applied_vocabulary_version = vocabulary_version

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = self._model.predict(image, conf=confidence, verbose=False)

        # Aggregate per-class detections across all result frames (a single
        # image produces one Results object, but this loop is harmless/
        # correct either way).
        confidences_by_class = {}
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self._model.names[cls_id]
                confidences_by_class.setdefault(cls_name, []).append(float(box.conf[0]))

        return {
            cls_name: {
                "count": len(confs),
                "avg_confidence": sum(confs) / len(confs),
            }
            for cls_name, confs in confidences_by_class.items()
        }

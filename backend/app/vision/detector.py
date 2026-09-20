"""Detection backends (spec Agent 1 - Vision Intelligence).

Three interchangeable backends, selected by SENTINEL_DETECTOR:

  yolo            Ultralytics YOLO11/YOLOv8 - the real-time workhorse.
  locateanything  nvidia/LocateAnything-3B - an open-vocabulary grounding VLM.
                  It accepts free-text prompts ("a person carrying a black
                  backpack", "an unattended suitcase"), so it covers the long
                  tail YOLO's fixed class list misses.  It is far too slow for
                  every frame, so it runs as a *refinement* pass on keyframes
                  and on operator queries, not in the hot loop.
  ensemble        YOLO every frame + LocateAnything on an interval / on demand.

If neither is installed the MotionDetector fallback keeps the whole pipeline
exercisable, and reports itself honestly as a fallback in the model registry.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import get_settings
from ..core.device import get_device_manager
from .types import Detection

log = logging.getLogger("sentinel.detector")

# Classes Eagles Eye cares about, mapped to the roles the agents expect.
PERSON_CLASSES = {"person"}
BAG_CLASSES = {"backpack", "handbag", "suitcase", "briefcase"}
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle", "train"}
DEFAULT_CLASSES = sorted(PERSON_CLASSES | BAG_CLASSES | VEHICLE_CLASSES | {"bottle", "laptop", "cell phone"})


class BaseDetector:
    name = "base"
    backend = "none"
    ready = False

    def detect(self, frame: np.ndarray, **kwargs: Any) -> List[Detection]:
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:
        return {"name": self.name, "backend": self.backend, "ready": self.ready}


# --------------------------------------------------------------------- YOLO
class YoloDetector(BaseDetector):
    name = "yolo"
    backend = "ultralytics"

    def __init__(self, weights: Optional[str] = None, classes: Optional[Sequence[str]] = None) -> None:
        s = get_settings()
        self.weights = weights or s.yolo_weights
        self.conf = s.yolo_conf
        self.iou = s.yolo_iou
        self.imgsz = s.yolo_imgsz
        self.allowed = set(classes) if classes else None
        self.weights_source = "configured"
        self._model = None
        self._names: Dict[int, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            log.warning("ultralytics unavailable (%s) - YOLO backend disabled", exc)
            return
        try:
            dm = get_device_manager()
            weights_path = self._resolve_weights()
            self._model = YOLO(weights_path)
            self._names = dict(self._model.names)
            # Register the underlying nn.Module so the GPU toggle can move it.
            inner = getattr(self._model, "model", None)
            if inner is not None:
                dm.register_model(f"detector:{self.name}", inner)
            dm.on_change(self._on_device_change)
            self.ready = True
            log.info("YOLO ready: %s on %s", weights_path, dm.active_device)
        except Exception:
            log.exception("failed to load YOLO weights %s", self.weights)

    def _resolve_weights(self) -> str:
        """Prefer a weights file inside storage/models, else let Ultralytics fetch."""
        from pathlib import Path

        s = get_settings()
        # A fine-tune only exists here if training/compare_detectors.py showed
        # it is not worse than stock YOLO11m on either evaluation set. It is
        # used automatically unless the operator pinned a different file.
        promoted = s.storage_dir / "models" / "sentinel_detector.pt"
        if self.weights == "yolo11m.pt" and promoted.is_file():
            self.weights_source = "fine-tuned (promoted after comparison)"
            return str(promoted)
        self.weights_source = "configured"
        candidate = Path(self.weights)
        if candidate.is_file():
            return str(candidate)
        local = s.storage_dir / "models" / self.weights
        if local.is_file():
            return str(local)
        return self.weights

    def _on_device_change(self, target: str) -> None:
        if self._model is None:
            return
        try:
            self._model.to(target)
        except Exception:
            log.exception("YOLO could not move to %s", target)

    def detect(self, frame: np.ndarray, **kwargs: Any) -> List[Detection]:
        if self._model is None:
            return []
        dm = get_device_manager()
        conf = float(kwargs.get("conf", self.conf))
        predict_kwargs = {
            "conf": conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": dm.active_device,
            "verbose": False,
            **self._precision_kwargs(dm.use_half),
        }
        results = self._model.predict(frame, **predict_kwargs)
        out: List[Detection] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for bb, sc, ci in zip(xyxy, confs, clss):
                cname = self._names.get(int(ci), str(ci))
                if self.allowed and cname not in self.allowed:
                    continue
                out.append(
                    Detection(
                        bbox=(float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])),
                        score=float(sc),
                        class_id=int(ci),
                        class_name=cname,
                    )
                )
        return out

    @staticmethod
    def _precision_kwargs(use_half: bool) -> Dict[str, Any]:
        """Half-precision argument, across Ultralytics versions.

        8.4 deprecated `half=` in favour of `quantize=`; older releases do not
        know `quantize` at all. Probing the signature keeps both working and
        keeps the deprecation warning out of the logs.
        """
        if not use_half:
            return {}
        import inspect

        try:
            from ultralytics.engine.model import Model

            params = inspect.signature(Model.predict).parameters
            if "quantize" in params:
                return {"quantize": "half"}
        except Exception:
            pass
        return {"half": True}

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "ready": self.ready,
            "weights": self.weights,
            "weights_source": self.weights_source,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "classes": len(self._names),
        }


# --------------------------------------------------- LocateAnything-3B (VLM)
class LocateAnythingDetector(BaseDetector):
    """Open-vocabulary detection / grounding with nvidia/LocateAnything-3B.

    It takes free-text categories and phrases ("an unattended suitcase", "a
    person carrying a backpack"), so it reaches the long tail YOLO's fixed
    80 classes cannot express. It is a 3B-parameter vision-language model:
    one call costs on the order of a second on a GPU (tens of seconds on a
    CPU), so it runs on keyframes and for operator queries, never per frame.

    Interface and output format follow the model card: prompts use its task
    templates, and the answer is a sequence of
    ``<ref>label</ref><box><x1><y1><x2><y2></box>...`` with coordinates
    normalised to 0-1000; every box belongs to the most recent <ref>.
    The model produces no confidence score, so boxes carry a fixed score and
    ``attributes["score_calibrated"] = False``.

    Licence: NVIDIA License - non-commercial (research / evaluation) use only.
    Keep it disabled on any commercial deployment.
    """

    name = "locateanything"
    backend = "transformers"
    FIXED_SCORE = 0.5

    def __init__(self, model_id: Optional[str] = None, enabled: Optional[bool] = None,
                 classes: Optional[Sequence[str]] = None) -> None:
        import threading

        s = get_settings()
        self.model_id = model_id or s.locateanything_model
        self.mode = s.locateanything_mode
        self._enabled = s.locateanything_enabled if enabled is None else enabled
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self.max_new_tokens = 384
        self.last_latency_ms: float = 0.0
        self.calls = 0
        self.load_error: Optional[str] = None
        # Categories grounded on keyframes when a camera gives no prompts.
        self.default_categories: List[str] = [
            "person", "backpack", "handbag", "suitcase", "unattended bag", "smoke", "fire",
        ]
        if classes:
            self.default_categories = list(classes)
        self._load()

    def _load(self) -> None:
        if not self._enabled:
            log.info("LocateAnything disabled by configuration")
            return
        try:
            import torch  # noqa: F401  - load before transformers/decord (DLL order on Windows)
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
        except Exception as exc:
            self.load_error = f"transformers unavailable: {exc}"
            log.warning("LocateAnything disabled - %s", self.load_error)
            return
        try:
            import torch

            dm = get_device_manager()
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(self.model_id, dtype=torch.bfloat16,
                                              trust_remote_code=True)
            model.eval()
            self._model = dm.register_model(f"detector:{self.name}", model)
            self.ready = True
            log.info("LocateAnything-3B ready on %s", dm.active_device)
        except Exception as exc:
            self.load_error = str(exc)
            log.exception("failed to load %s", self.model_id)

    # ------------------------------------------------------------ inference
    def detect(self, frame: np.ndarray, **kwargs: Any) -> List[Detection]:
        categories = list(kwargs.get("prompts") or self.default_categories)
        return self.locate(frame, categories)

    def locate(self, frame: np.ndarray, categories: Sequence[str]) -> List[Detection]:
        """Every instance of each category (the card's detection template)."""
        if not categories:
            return []
        prompt = ("Locate all the instances that matches the following description: "
                  + "</c>".join(c.strip() for c in categories) + ".")
        return self._run(frame, prompt, fallback_label=None)

    def ground(self, frame: np.ndarray, phrase: str) -> List[Detection]:
        """All instances matching one free-form phrase (grounding template)."""
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self._run(frame, prompt, fallback_label=phrase)

    def _run(self, frame: np.ndarray, prompt: str, fallback_label: Optional[str]) -> List[Detection]:
        if self._model is None or self._processor is None:
            return []
        try:
            import torch
            from PIL import Image
        except Exception:
            return []

        image = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))   # BGR -> RGB
        h, w = frame.shape[:2]
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": prompt}]}]
        started = time.perf_counter()
        try:
            with self._lock:
                param = next(self._model.parameters())
                text = self._processor.py_apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                images, videos = self._processor.process_vision_info(messages)
                inputs = self._processor(text=[text], images=images, videos=videos,
                                         return_tensors="pt").to(param.device)
                with torch.inference_mode():
                    response = self._model.generate(
                        pixel_values=inputs["pixel_values"].to(param.dtype),
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        image_grid_hws=inputs.get("image_grid_hws"),
                        tokenizer=self._tokenizer,
                        # Each box costs ~6 tokens, so this still allows ~50
                        # objects. 1024 only lengthened the worst-case call,
                        # and every keyframe stalls the shared GPU meanwhile.
                        max_new_tokens=self.max_new_tokens,
                        use_cache=True,
                        generation_mode=self.mode,
                        do_sample=False,          # detection must be repeatable
                        verbose=False,
                    )
            answer = response[0] if isinstance(response, tuple) else response
            if isinstance(answer, (list, tuple)):
                answer = answer[0]
        except Exception:
            log.exception("LocateAnything inference failed for %r", prompt)
            return []
        finally:
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0
            self.calls += 1

        out: List[Detection] = []
        for label, box in parse_locateanything(str(answer), w, h, default_label=fallback_label):
            out.append(Detection(
                bbox=box, score=self.FIXED_SCORE, class_id=-1,
                class_name=_prompt_to_class(label),
                attributes={"label": label, "source": "locateanything",
                            "score_calibrated": False},
            ))
        return out

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "ready": self.ready,
            "model_id": self.model_id,
            "mode": self.mode,
            "last_latency_ms": round(self.last_latency_ms, 1),
            "calls": self.calls,
            "load_error": self.load_error,
            "licence": "NVIDIA License - non-commercial research/evaluation use only",
            "role": "open-vocabulary keyframe refinement / forensic grounding",
        }


def _prompt_to_class(prompt: str) -> str:
    """Map a free-text label onto the class names the agents understand."""
    p = prompt.lower().strip()
    words = set(p.replace(",", " ").split())
    if words & {"person", "people", "man", "woman", "men", "women", "boy", "girl", "child",
                "pedestrian"}:
        return "person"
    for c in BAG_CLASSES:
        if c in p:
            return c
    if "bag" in p or "luggage" in p:
        return "suitcase"
    if "smoke" in p or "fire" in p or "flame" in p:
        return "fire_smoke"
    for c in VEHICLE_CLASSES:
        if c in p:
            return c
    return p or "object"


_LA_TOKENS = re.compile(
    r"<ref>(?P<ref>.*?)</ref>|<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>")


def parse_locateanything(text: str, width: int, height: int,
                         default_label: Optional[str] = None):
    """Yield (label, (x1, y1, x2, y2) in pixels) from a LocateAnything answer.

    Boxes belong to the most recent ``<ref>label</ref>``; boxes before any ref
    (single-phrase grounding) take ``default_label``. Coordinates are
    normalised to 0-1000. Malformed or degenerate boxes are dropped rather
    than guessed at.
    """
    label = default_label or "object"
    raw = []
    for m in _LA_TOKENS.finditer(text):
        if m.group("ref") is not None:
            label = m.group("ref").strip() or label
            continue
        x1, y1, x2, y2 = (int(m.group(k)) for k in ("x1", "y1", "x2", "y2"))
        if max(x1, y1, x2, y2) > 1000 or x2 <= x1 or y2 <= y1:
            continue
        raw.append((label, (x1, y1, x2, y2)))
    for label, (x1, y1, x2, y2) in _drop_degenerate_runs(raw):
        yield label, (x1 / 1000.0 * width, y1 / 1000.0 * height,
                      x2 / 1000.0 * width, y2 / 1000.0 * height)


def _drop_degenerate_runs(boxes, min_run: int = 6, tol: int = 3):
    """Remove the runaway pattern a VLM can fall into when decoding loops.

    Observed with LocateAnything under greedy hybrid decoding: after the real
    boxes it emits a "sliding" chain - same y1/y2, each box starting where
    the previous one ended - until max_new_tokens. Real objects do not tile
    a row edge to edge, so a chain of min_run or more such boxes is dropped.
    """
    n = len(boxes)
    keep = [True] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n:
            (la, a), (lb, b) = boxes[j], boxes[j + 1]
            if la == lb and abs(a[1] - b[1]) <= tol and abs(a[3] - b[3]) <= tol                     and abs(b[0] - a[2]) <= tol * 2:
                j += 1
            else:
                break
        if j - i + 1 >= min_run:
            for k in range(i, j + 1):
                keep[k] = False
        i = j + 1
    if not all(keep):
        log.warning("LocateAnything produced a degenerate box chain; dropped %d box(es)",
                    keep.count(False))
    return [b for b, k in zip(boxes, keep) if k]


# ----------------------------------------------------------------- fallback
class MotionDetector(BaseDetector):
    """MOG2 background subtraction, used when no learned detector is present.

    This exists so the full agent pipeline, dashboard and alerting path can be
    exercised end-to-end before model weights are downloaded.  It reports
    ``backend="fallback-motion"`` so the UI never implies a trained model is
    running.
    """

    name = "motion"
    backend = "fallback-motion"

    def __init__(self, min_area_ratio: float = 0.0012, max_area_ratio: float = 0.45) -> None:
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self._bg = None
        try:
            import cv2

            self._cv2 = cv2
            self._bg = cv2.createBackgroundSubtractorMOG2(
                history=400, varThreshold=28, detectShadows=True
            )
            self.ready = True
        except Exception as exc:
            log.warning("OpenCV unavailable (%s) - motion fallback disabled", exc)
            self._cv2 = None

    def detect(self, frame: np.ndarray, **kwargs: Any) -> List[Detection]:
        if self._bg is None or self._cv2 is None:
            return []
        cv2 = self._cv2
        h, w = frame.shape[:2]
        frame_area = float(h * w)

        mask = self._bg.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            ratio = area / frame_area
            if ratio < self.min_area_ratio or ratio > self.max_area_ratio:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bh / max(1.0, bw)
            # upright blobs read as people, wide low blobs as generic objects
            cname = "person" if aspect > 1.15 else "object"
            score = float(min(0.9, 0.45 + ratio * 4))
            out.append(
                Detection(
                    bbox=(float(x), float(y), float(x + bw), float(y + bh)),
                    score=score,
                    class_id=0 if cname == "person" else 99,
                    class_name=cname,
                    attributes={"source": "motion-fallback"},
                )
            )
        return out


# ---------------------------------------------------------------- ensemble
class EnsembleDetector(BaseDetector):
    """YOLO on every frame, fused with LocateAnything-3B on keyframes.

    YOLO is the real-time workhorse. LocateAnything reaches objects YOLO's 80
    classes miss or under-detect, but costs ~a second per call, so:

    * it runs in a background thread on every Nth frame and never blocks the
      camera loop;
    * its boxes become *hints* that persist until the next keyframe result,
      because the tracker needs several consecutive hits to confirm a track -
      a box injected on a single frame would never become one;
    * hints are limited to objects that stay put (bags, luggage, smoke/fire,
      open-vocabulary objects). People and vehicles move a long way in a
      second and YOLO already sees them every frame, so stale boxes for them
      would create ghost tracks rather than recall;
    * a hint is only added where no YOLO box overlaps it (IoU >= 0.5).
    """

    name = "ensemble"
    backend = "yolo+locateanything"
    MOVING_CLASSES = PERSON_CLASSES | VEHICLE_CLASSES

    def __init__(self, refine_every: Optional[int] = None,
                 classes: Optional[Sequence[str]] = None,
                 grounding: Optional["LocateAnythingDetector"] = None) -> None:
        from concurrent.futures import ThreadPoolExecutor

        s = get_settings()
        self.yolo = YoloDetector(classes=classes)
        self.grounding = grounding if grounding is not None else LocateAnythingDetector()
        self.refine_every = max(1, int(refine_every or s.locateanything_every_n_frames))
        self._counter = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="locateanything")
        self._future = None
        self._hints: List[Detection] = []
        self.hints_added = 0
        self.ready = self.yolo.ready or self.grounding.ready
        if self.yolo.ready and not self.grounding.ready:
            log.warning("ensemble running YOLO only: LocateAnything is not loaded (%s)",
                        self.grounding.load_error or "disabled")

    def detect(self, frame: np.ndarray, **kwargs: Any) -> List[Detection]:
        dets = self.yolo.detect(frame, **kwargs) if self.yolo.ready else []
        if not self.grounding.ready:
            return dets
        self._counter += 1

        if self._future is not None and self._future.done():
            try:
                self._hints = [d for d in self._future.result()
                               if d.class_name not in self.MOVING_CLASSES]
            except Exception:
                log.exception("LocateAnything keyframe failed")
                self._hints = []
            self._future = None

        if self._future is None and self._counter % self.refine_every == 0:
            prompts = kwargs.get("prompts")
            self._future = self._executor.submit(
                self.grounding.detect, frame.copy(), prompts=prompts)

        extra = _dedupe_against(self._hints, dets, iou_threshold=0.5)
        self.hints_added += len(extra)
        return dets + [Detection(bbox=d.bbox, score=d.score, class_id=d.class_id,
                                 class_name=d.class_name, attributes=dict(d.attributes))
                       for d in extra]

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend if self.grounding.ready else "ultralytics (LocateAnything not loaded)",
            "ready": self.ready,
            "yolo": self.yolo.info(),
            "locateanything": self.grounding.info(),
            "refine_every": self.refine_every,
            "active_hints": len(self._hints),
            "hints_added": self.hints_added,
        }


def _dedupe_against(new: List[Detection], existing: List[Detection], iou_threshold: float) -> List[Detection]:
    if not existing or not new:
        return new
    from .tracker import iou_matrix

    a = np.array([d.bbox for d in new], dtype=np.float32)
    b = np.array([d.bbox for d in existing], dtype=np.float32)
    ious = iou_matrix(a, b)
    keep = [new[i] for i in range(len(new)) if ious[i].max(initial=0.0) < iou_threshold]
    return keep


# ---------------------------------------------------------------- factory
_cache: Dict[str, BaseDetector] = {}


def build_detector(kind: Optional[str] = None, **kwargs: Any) -> BaseDetector:
    """Instantiate a detector, falling back gracefully.

    `kind` overrides SENTINEL_DETECTOR, so an individual camera can pin its own
    backend. Instances are cached per kind.
    """
    s = get_settings()
    kind = (kind or s.detector).lower()
    if kind in _cache:
        return _cache[kind]

    started = time.time()
    det: BaseDetector
    if kind == "motion":
        # Explicitly requested, e.g. by the synthetic demo camera.
        det = MotionDetector()
    elif kind == "locateanything":
        det = LocateAnythingDetector(**kwargs)
    elif kind == "ensemble":
        det = EnsembleDetector(**kwargs)
    else:
        det = YoloDetector(**kwargs)

    if not det.ready:
        log.warning("%s detector not ready - using motion fallback", kind)
        fallback = MotionDetector()
        if fallback.ready:
            det = fallback

    _cache[kind] = det
    log.info("detector '%s' built in %.2fs (backend=%s)", kind, time.time() - started, det.backend)
    return det


def reset_detector_cache() -> None:
    _cache.clear()

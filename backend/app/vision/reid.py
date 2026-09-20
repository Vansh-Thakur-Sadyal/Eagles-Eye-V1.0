"""Appearance embeddings for cross-camera Re-ID (spec Agents 3, 4, 5).

Two embedding families:

  body  - OSNet via torchreid when installed; otherwise a deterministic
          handcrafted descriptor (striped HSV histograms + edge-density
          profile) that is genuinely discriminative for short-horizon
          cross-camera association and needs no weights.
  face  - InsightFace when installed; otherwise OpenCV's DNN/Haar face
          detector paired with the handcrafted descriptor.

Every embedding is L2-normalised, so cosine similarity is a dot product.
The backend in use is reported upward so the UI can state which is running
instead of implying a trained model when there is none.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import get_settings
from ..core.device import get_device_manager

log = logging.getLogger("sentinel.reid")


def l2_normalise(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


# ------------------------------------------------------------- handcrafted
def handcrafted_embedding(crop: np.ndarray, stripes: int = 6, bins: int = 12) -> Optional[np.ndarray]:
    """Striped HSV colour histogram + edge density.

    Horizontal stripes preserve vertical clothing structure (head / torso /
    legs), which is what makes this usable for re-identification rather than a
    plain global colour histogram.
    """
    try:
        import cv2
    except Exception:
        return None
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 16 or w < 8:
        return None

    crop = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)

    band = 128 // stripes
    feats: List[np.ndarray] = []
    for i in range(stripes):
        y0, y1 = i * band, (i + 1) * band
        seg = hsv[y0:y1]
        hist_h = cv2.calcHist([seg], [0], None, [bins], [0, 180]).flatten()
        hist_s = cv2.calcHist([seg], [1], None, [bins], [0, 256]).flatten()
        hist_v = cv2.calcHist([seg], [2], None, [bins // 2], [0, 256]).flatten()
        edge_density = np.array([edges[y0:y1].mean() / 255.0], dtype=np.float32)
        for hst in (hist_h, hist_s, hist_v):
            total = hst.sum()
            feats.append(hst / total if total > 0 else hst)
        feats.append(edge_density)

    return l2_normalise(np.concatenate(feats).astype(np.float32))


# ------------------------------------------------------------------ bodies
class BodyEmbedder:
    """Person appearance embeddings for cross-camera association."""

    def __init__(self) -> None:
        s = get_settings()
        self.model_name = s.reid_weights
        self.backend = "handcrafted"
        self._model = None
        self._torch = None
        self._transform = None
        self.dim = 0
        self.weights_provenance = "none"
        self._load()

    def _load(self) -> None:
        try:
            import torch
            import torchreid
            from torchvision import transforms
        except Exception as exc:
            log.info("torchreid unavailable (%s) - using handcrafted body embeddings", exc)
            self.dim = self._probe_handcrafted_dim()
            return
        try:
            dm = get_device_manager()
            model = torchreid.models.build_model(
                name=self.model_name, num_classes=1000, pretrained=True
            )

            # `pretrained=True` loads ImageNet weights, NOT re-identification
            # weights. ImageNet features are better than a handcrafted
            # descriptor but they were never trained to tell two people apart,
            # so the provenance is tracked and reported rather than glossed.
            self.weights_provenance = "imagenet"
            weights_path = self._reid_weights_path()
            if weights_path is not None:
                try:
                    loaded = self._load_reid_checkpoint(model, weights_path)
                    self.weights_provenance = f"reid:{weights_path.stem}"
                    log.info("loaded Re-ID weights from %s (%d tensors)", weights_path, loaded)
                except Exception:
                    log.exception("could not load Re-ID weights from %s", weights_path)

            model.eval()
            dm.register_model("reid:body", model)
            self._model = model
            self._torch = torch
            self._transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((256, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            self.backend = f"torchreid:{self.model_name}"
            self.dim = 512
            if self.weights_provenance == "imagenet":
                log.warning(
                    "Re-ID is running on ImageNet weights, which were not trained to "
                    "distinguish people. Cross-camera association will be weaker than it "
                    "looks. Drop a Market-1501 checkpoint in storage/models/reid/ - see "
                    "docs/INSTALL.md."
                )
            log.info("Re-ID %s (%s) ready on %s",
                     self.model_name, self.weights_provenance, dm.active_device)
        except Exception:
            log.exception("failed to load Re-ID model %s", self.model_name)
            self.dim = self._probe_handcrafted_dim()

    @staticmethod
    def _load_reid_checkpoint(model, path) -> int:
        """Load a torchreid training checkpoint into `model`.

        torchreid's own loader calls torch.load with the torch >= 2.6 default
        weights_only=True, which rejects its own full checkpoints. The file is
        one this platform trained (training/train_reid.py), so it is loaded in
        full; only tensors whose names and shapes match are copied, which drops
        the 751-identity training classifier the embedder never uses.
        """
        import torch

        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        own = model.state_dict()
        matched = {}
        for key, value in state.items():
            key = key[7:] if key.startswith("module.") else key
            if key in own and own[key].shape == value.shape:
                matched[key] = value
        if len(matched) < 0.9 * len([k for k in own if not k.startswith("classifier")]):
            raise ValueError(f"checkpoint matches only {len(matched)}/{len(own)} tensors")
        own.update(matched)
        model.load_state_dict(own)
        return len(matched)

    @staticmethod
    def _reid_weights_path():
        """A real Re-ID checkpoint, if the operator has supplied one."""
        folder = get_settings().storage_dir / "models" / "reid"
        if not folder.is_dir():
            return None
        for candidate in sorted(folder.glob("*.pth")) + sorted(folder.glob("*.pth.tar")):
            return candidate
        return None

    @staticmethod
    def _probe_handcrafted_dim() -> int:
        probe = handcrafted_embedding(np.zeros((128, 64, 3), dtype=np.uint8))
        return int(probe.shape[0]) if probe is not None else 0

    def embed(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if self._model is None or self._torch is None:
            return handcrafted_embedding(crop)
        try:
            dm = get_device_manager()
            tensor = self._transform(crop[:, :, ::-1].copy()).unsqueeze(0).to(dm.active_device)
            if dm.use_half:
                tensor = tensor.half()
            with self._torch.inference_mode():
                feat = self._model(tensor)
            vec = feat.float().cpu().numpy().flatten()
            return l2_normalise(vec)
        except Exception:
            log.exception("body embedding failed, falling back")
            return handcrafted_embedding(crop)

    def embed_batch(self, crops: Sequence[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
        """Embed every crop of a frame in ONE forward pass.

        One call per person cost ~6 ms each: 147 ms per frame in a crowd of 25,
        which was the pipeline's bottleneck (detection is ~9 ms). Batching makes
        it one GPU call. `None` entries are preserved so the result lines up
        with the detections it came from.
        """
        out: List[Optional[np.ndarray]] = [None] * len(crops)
        valid = [(i, c) for i, c in enumerate(crops) if c is not None and c.size]
        if not valid:
            return out
        if self._model is None or self._torch is None:
            for i, crop in valid:
                out[i] = handcrafted_embedding(crop)
            return out
        try:
            dm = get_device_manager()
            batch = self._torch.stack(
                [self._transform(c[:, :, ::-1].copy()) for _, c in valid]
            ).to(dm.active_device)
            if dm.use_half:
                batch = batch.half()
            with self._torch.inference_mode():
                feats = self._model(batch)
            for (i, _), vec in zip(valid, feats.float().cpu().numpy()):
                out[i] = l2_normalise(vec.flatten())
            return out
        except Exception:
            log.exception("batched body embedding failed, falling back to one at a time")
            for i, crop in valid:
                out[i] = self.embed(crop)
            return out

    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "dim": self.dim,
            "model": self.model_name,
            "weights": self.weights_provenance,
            "trained_for_reid": self.weights_provenance.startswith("reid:"),
        }


# ------------------------------------------------------------------- faces
class FaceEmbedder:
    """Face detection + embedding, used only for authorised watchlist matching."""

    def __init__(self) -> None:
        s = get_settings()
        self.model_name = s.face_model
        self.backend = "opencv-haar+handcrafted"
        self._app = None
        self._cascade = None
        self.dim = 0
        self.provider = "CPUExecutionProvider"
        self._load()

    def _load(self) -> None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except Exception as exc:
            log.info("insightface unavailable (%s) - using OpenCV face fallback", exc)
            self._load_cascade()
            return
        try:
            dm = get_device_manager()
            available = self._available_providers(dm.gpu_enabled)
            wants_gpu = dm.gpu_enabled and "CUDAExecutionProvider" in available

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if wants_gpu else ["CPUExecutionProvider"]
            )
            app = FaceAnalysis(name=self.model_name, providers=providers)
            app.prepare(ctx_id=0 if wants_gpu else -1, det_size=(640, 640))

            self._app = app
            self.provider = "CUDAExecutionProvider" if wants_gpu else "CPUExecutionProvider"
            self.backend = f"insightface:{self.model_name}"
            self.dim = 512
            if dm.gpu_enabled and not wants_gpu:
                log.warning(
                    "InsightFace is running on CPU: onnxruntime has no usable CUDA provider. "
                    "Install a build matching your CUDA version "
                    "(onnxruntime-gpu for CUDA 12.x), or accept slower face matching."
                )
            log.info("InsightFace %s ready on %s", self.model_name, self.provider)
        except Exception:
            log.exception("failed to initialise InsightFace")
            self._load_cascade()

    @staticmethod
    def _available_providers(want_gpu: bool) -> List[str]:
        """Which ONNX Runtime execution providers can actually be used.

        onnxruntime ships its own CUDA/cuDNN expectations and frequently cannot
        see a CUDA install that torch is using happily. Newer builds expose
        `preload_dlls()`, which points it at the libraries already on the path -
        worth trying before giving up and running faces on CPU.
        """
        try:
            # Import torch first: onnxruntime reuses the CUDA/cuDNN DLLs torch
            # has already loaded, which is what lets a single CUDA install serve
            # both. Without this the CUDA provider can be invisible.
            try:
                import torch  # noqa: F401
            except Exception:
                pass
            import onnxruntime as ort
        except Exception:
            return ["CPUExecutionProvider"]

        providers = list(ort.get_available_providers())
        if want_gpu and "CUDAExecutionProvider" not in providers:
            try:
                ort.preload_dlls(cuda=True, cudnn=True)
                providers = list(ort.get_available_providers())
            except Exception:
                pass
        return providers

    def _load_cascade(self) -> None:
        try:
            import cv2

            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(path)
            if self._cascade.empty():
                self._cascade = None
            probe = handcrafted_embedding(np.zeros((128, 64, 3), dtype=np.uint8))
            self.dim = int(probe.shape[0]) if probe is not None else 0
        except Exception:
            log.exception("OpenCV face cascade unavailable")

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Return [{bbox, score, embedding, landmarks}] for each face found."""
        if self._app is not None:
            try:
                faces = self._app.get(image)
                out = []
                for f in faces:
                    emb = getattr(f, "normed_embedding", None)
                    if emb is None:
                        emb = l2_normalise(np.asarray(f.embedding, dtype=np.float32))
                    out.append(
                        {
                            "bbox": [float(v) for v in f.bbox],
                            "score": float(getattr(f, "det_score", 0.0)),
                            "embedding": np.asarray(emb, dtype=np.float32),
                            "landmarks": (
                                np.asarray(f.kps).tolist() if getattr(f, "kps", None) is not None else None
                            ),
                        }
                    )
                return out
            except Exception:
                log.exception("InsightFace inference failed")
                return []

        if self._cascade is None:
            return []
        try:
            import cv2

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            rects = self._cascade.detectMultiScale(gray, 1.15, 5, minSize=(36, 36))
            out = []
            for (x, y, w, h) in rects:
                crop = image[y : y + h, x : x + w]
                emb = handcrafted_embedding(crop, stripes=4)
                if emb is None:
                    continue
                out.append(
                    {
                        "bbox": [float(x), float(y), float(x + w), float(y + h)],
                        "score": 0.6,
                        "embedding": emb,
                        "landmarks": None,
                    }
                )
            return out
        except Exception:
            return []

    def embed_largest(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[float]], float]:
        faces = self.detect(image)
        if not faces:
            return None, None, 0.0
        best = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        return best["embedding"], best["bbox"], best["score"]

    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "dim": self.dim,
            "model": self.model_name,
            "provider": self.provider,
            "on_gpu": self.provider == "CUDAExecutionProvider",
        }


# -------------------------------------------------------- occlusion agent 5
FACE_COVERING_LABELS = ("mask", "helmet", "hood", "scarf", "sunglasses", "cap")


def assess_face_visibility(
    person_crop: np.ndarray,
    face_boxes: Sequence[Sequence[float]],
    *,
    min_face_area_ratio: float = 0.012,
) -> Dict[str, Any]:
    """Agent 5: report how observable a face is - never that a person is suspect.

    A covered face lowers identity confidence.  It is deliberately NOT a risk
    signal on its own; the threat agent gives it weight 0.
    """
    if person_crop is None or person_crop.size == 0:
        return {"visibility": "unknown", "identity_confidence": 0.0, "labels": [], "reason": "no crop"}

    ph, pw = person_crop.shape[:2]
    person_area = float(ph * pw)
    if not face_boxes:
        return {
            "visibility": "absent",
            "identity_confidence": 0.05,
            "labels": [],
            "reason": "no face observed in this view",
        }

    largest = max(
        face_boxes, key=lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    )
    face_area = max(0.0, largest[2] - largest[0]) * max(0.0, largest[3] - largest[1])
    ratio = face_area / person_area if person_area else 0.0

    if ratio >= min_face_area_ratio * 2.5:
        visibility, confidence = "visible", 0.85
    elif ratio >= min_face_area_ratio:
        visibility, confidence = "partial", 0.45
    else:
        visibility, confidence = "covered", 0.2

    return {
        "visibility": visibility,
        "identity_confidence": round(confidence, 3),
        "face_area_ratio": round(ratio, 5),
        "labels": [],
        "reason": {
            "visible": "face observable",
            "partial": "face partially observable",
            "covered": "face observation below usable size",
        }[visibility],
    }


# ------------------------------------------------------------------ gallery
class EmbeddingGallery:
    """In-memory nearest-neighbour index over appearance embeddings.

    Backs cross-camera Re-ID (Agent 3) and watchlist matching.  Brute-force
    cosine over a normalised matrix: exact, dependency-free, and fast enough
    for the gallery sizes the policy caps at.
    """

    def __init__(self, dim: int = 0, max_size: int = 5000) -> None:
        self.dim = dim
        self.max_size = max_size
        self._keys: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._meta: Dict[str, Dict[str, Any]] = {}

    def add(self, key: str, embedding: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> None:
        emb = l2_normalise(np.asarray(embedding, dtype=np.float32))
        if self.dim and emb.shape[0] != self.dim:
            return
        if not self.dim:
            self.dim = int(emb.shape[0])

        if key in self._meta:
            idx = self._keys.index(key)
            self._matrix[idx] = emb
            self._meta[key].update(meta or {})
            return

        self._keys.append(key)
        self._meta[key] = dict(meta or {})
        self._matrix = emb[None, :] if self._matrix is None else np.vstack([self._matrix, emb])

        if len(self._keys) > self.max_size:
            drop = len(self._keys) - self.max_size
            for k in self._keys[:drop]:
                self._meta.pop(k, None)
            self._keys = self._keys[drop:]
            self._matrix = self._matrix[drop:]

    def search(self, embedding: np.ndarray, top_k: int = 5, threshold: float = 0.0):
        """Return [(key, similarity, meta)] sorted by descending similarity."""
        if self._matrix is None or not self._keys:
            return []
        emb = l2_normalise(np.asarray(embedding, dtype=np.float32))
        if emb.shape[0] != self._matrix.shape[1]:
            return []
        sims = self._matrix @ emb
        order = np.argsort(-sims)[: max(1, top_k)]
        return [
            (self._keys[i], float(sims[i]), self._meta.get(self._keys[i], {}))
            for i in order
            if sims[i] >= threshold
        ]

    def remove(self, key: str) -> bool:
        if key not in self._meta:
            return False
        idx = self._keys.index(key)
        self._keys.pop(idx)
        self._meta.pop(key, None)
        self._matrix = np.delete(self._matrix, idx, axis=0) if self._matrix is not None else None
        return True

    def clear(self) -> None:
        self._keys.clear()
        self._meta.clear()
        self._matrix = None

    def __len__(self) -> int:
        return len(self._keys)


# ------------------------------------------------------------------ globals
_body: Optional[BodyEmbedder] = None
_face: Optional[FaceEmbedder] = None


def get_body_embedder() -> BodyEmbedder:
    global _body
    if _body is None:
        _body = BodyEmbedder()
    return _body


def get_face_embedder() -> FaceEmbedder:
    global _face
    if _face is None:
        _face = FaceEmbedder()
    return _face

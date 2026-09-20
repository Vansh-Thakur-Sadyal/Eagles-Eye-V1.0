"""Appearance-robustness augmentation (spec Agent 4 and dataset category 10).

The spec is explicit about the intent: generate *variation of an authorised
reference* to make matching robust, rather than fabricating hundreds of
identities and storing them as surveillance records.  Every output therefore
carries its source identity and the exact parameters used, so the pipeline is
auditable and reversible.

These same transforms produce dataset category 9 (face occlusion) and 10
(appearance variation) from a handful of real reference photographs.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("sentinel.augment")


@dataclass
class AugmentedSample:
    image: np.ndarray
    augmentation: str
    params: Dict[str, Any] = field(default_factory=dict)
    source_identity: str = ""
    occlusion_label: Optional[str] = None


def _cv2():
    import cv2

    return cv2


# ---------------------------------------------------------- photometric ----
def adjust_lighting(img: np.ndarray, brightness: float = 1.0, contrast: float = 1.0) -> np.ndarray:
    out = img.astype(np.float32)
    mean = out.mean()
    out = (out - mean) * contrast + mean * brightness
    return np.clip(out, 0, 255).astype(np.uint8)


def colour_shift(img: np.ndarray, hue_delta: int = 0, sat_scale: float = 1.0) -> np.ndarray:
    cv2 = _cv2()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + hue_delta) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def motion_blur(img: np.ndarray, kernel: int = 9, angle: float = 0.0) -> np.ndarray:
    cv2 = _cv2()
    kernel = max(3, kernel | 1)
    k = np.zeros((kernel, kernel), dtype=np.float32)
    k[kernel // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((kernel / 2 - 0.5, kernel / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, m, (kernel, kernel))
    total = k.sum()
    if total > 0:
        k /= total
    return cv2.filter2D(img, -1, k)


def degrade_resolution(img: np.ndarray, scale: float = 0.35, jpeg_quality: int = 40) -> np.ndarray:
    """Simulate a distant or low-bitrate camera."""
    cv2 = _cv2()
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    ok, enc = cv2.imencode(".jpg", back, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else back


def add_sensor_noise(img: np.ndarray, sigma: float = 9.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ------------------------------------------------------------- geometric ----
def change_viewpoint(img: np.ndarray, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    """Approximate a different camera angle with a perspective warp."""
    cv2 = _cv2()
    h, w = img.shape[:2]
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    dx = math.sin(yaw) * w * 0.25
    dy = math.sin(pitch) * h * 0.25
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32(
        [
            [0 + max(0.0, dx), 0 + max(0.0, dy)],
            [w + min(0.0, dx), 0 + max(0.0, dy)],
            [w - max(0.0, dx), h - max(0.0, dy)],
            [0 - min(0.0, dx), h - max(0.0, dy)],
        ]
    )
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def random_erase(img: np.ndarray, area_ratio: float = 0.12, position: str = "random") -> np.ndarray:
    """Partial occlusion by another body or a structure."""
    out = img.copy()
    h, w = out.shape[:2]
    eh = int(h * math.sqrt(area_ratio))
    ew = int(w * math.sqrt(area_ratio))
    if position == "bottom":
        y0, x0 = h - eh, (w - ew) // 2
    elif position == "left":
        y0, x0 = (h - eh) // 2, 0
    elif position == "right":
        y0, x0 = (h - eh) // 2, w - ew
    else:
        y0 = np.random.randint(0, max(1, h - eh))
        x0 = np.random.randint(0, max(1, w - ew))
    out[max(0, y0) : y0 + eh, max(0, x0) : x0 + ew] = np.random.randint(0, 90, (1, 1, 3), dtype=np.uint8)
    return out


# -------------------------------------------------------- face coverings ----
def _face_region(img: np.ndarray, face_bbox: Optional[Tuple[float, float, float, float]]):
    h, w = img.shape[:2]
    if face_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
    else:
        # assume a head-and-shoulders framing when no detector box is given
        x1, y1 = int(w * 0.22), int(h * 0.05)
        x2, y2 = int(w * 0.78), int(h * 0.42)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def apply_mask(img: np.ndarray, face_bbox=None, colour=(240, 240, 240)) -> np.ndarray:
    """Draw a surgical-style mask over the lower half of the face region."""
    cv2 = _cv2()
    out = img.copy()
    x1, y1, x2, y2 = _face_region(out, face_bbox)
    fh = y2 - y1
    my1 = y1 + int(fh * 0.48)
    pts = np.array(
        [
            [x1 + int((x2 - x1) * 0.05), my1],
            [x2 - int((x2 - x1) * 0.05), my1],
            [x2 - int((x2 - x1) * 0.12), y2],
            [x1 + int((x2 - x1) * 0.12), y2],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(out, [pts], colour)
    cv2.polylines(out, [pts], True, tuple(int(c * 0.8) for c in colour), 2)
    return out


def apply_sunglasses(img: np.ndarray, face_bbox=None) -> np.ndarray:
    cv2 = _cv2()
    out = img.copy()
    x1, y1, x2, y2 = _face_region(out, face_bbox)
    fw, fh = x2 - x1, y2 - y1
    ey = y1 + int(fh * 0.34)
    eh = max(4, int(fh * 0.16))
    cv2.rectangle(out, (x1 + int(fw * 0.08), ey), (x2 - int(fw * 0.08), ey + eh), (18, 18, 18), -1)
    return out


def apply_hood_or_cap(img: np.ndarray, face_bbox=None, kind: str = "cap") -> np.ndarray:
    cv2 = _cv2()
    out = img.copy()
    x1, y1, x2, y2 = _face_region(out, face_bbox)
    fw, fh = x2 - x1, y2 - y1
    colour = (52, 52, 60) if kind == "cap" else (38, 42, 52)
    cover_h = int(fh * (0.28 if kind == "cap" else 0.45))
    cv2.ellipse(
        out,
        (x1 + fw // 2, y1 + cover_h // 2),
        (int(fw * 0.62), max(4, cover_h)),
        0, 0, 360, colour, -1,
    )
    return out


def apply_helmet(img: np.ndarray, face_bbox=None) -> np.ndarray:
    cv2 = _cv2()
    out = img.copy()
    x1, y1, x2, y2 = _face_region(out, face_bbox)
    fw, fh = x2 - x1, y2 - y1
    cv2.ellipse(out, (x1 + fw // 2, y1 + int(fh * 0.34)), (int(fw * 0.70), int(fh * 0.56)),
                0, 180, 360, (40, 60, 120), -1)
    cv2.rectangle(out, (x1, y1 + int(fh * 0.30)), (x2, y1 + int(fh * 0.52)), (28, 28, 34), -1)
    return out


def apply_facial_hair(img: np.ndarray, face_bbox=None) -> np.ndarray:
    cv2 = _cv2()
    out = img.copy()
    x1, y1, x2, y2 = _face_region(out, face_bbox)
    fw, fh = x2 - x1, y2 - y1
    overlay = out.copy()
    cv2.ellipse(overlay, (x1 + fw // 2, y1 + int(fh * 0.80)), (int(fw * 0.34), int(fh * 0.20)),
                0, 0, 360, (36, 30, 26), -1)
    return cv2.addWeighted(overlay, 0.65, out, 0.35, 0)


def change_hair(img: np.ndarray, face_bbox=None, colour=(30, 25, 22)) -> np.ndarray:
    cv2 = _cv2()
    out = img.copy()
    x1, y1, x2, y2 = _face_region(out, face_bbox)
    fw, fh = x2 - x1, y2 - y1
    overlay = out.copy()
    cv2.ellipse(overlay, (x1 + fw // 2, y1 + int(fh * 0.12)), (int(fw * 0.58), int(fh * 0.30)),
                0, 0, 360, colour, -1)
    return cv2.addWeighted(overlay, 0.8, out, 0.2, 0)


def simulate_aging(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Coarse ageing proxy: reduced local contrast plus a warmer cast."""
    cv2 = _cv2()
    out = adjust_lighting(img, brightness=1.0 - 0.06 * strength, contrast=1.0 - 0.18 * strength)
    out = cv2.bilateralFilter(out, 7, 45, 45)
    return colour_shift(out, hue_delta=int(3 * strength), sat_scale=1.0 - 0.15 * strength)


# ------------------------------------------------------------- the pipeline
AUGMENTATIONS: Dict[str, Tuple[Callable[..., np.ndarray], Dict[str, Any], Optional[str]]] = {
    "mask":            (apply_mask,          {},                                   "mask"),
    "sunglasses":      (apply_sunglasses,    {},                                   "sunglasses"),
    "cap":             (apply_hood_or_cap,   {"kind": "cap"},                      "cap"),
    "hood":            (apply_hood_or_cap,   {"kind": "hood"},                     "hood"),
    "helmet":          (apply_helmet,        {},                                   "helmet"),
    "facial_hair":     (apply_facial_hair,   {},                                   None),
    "hair_change":     (change_hair,         {"colour": (24, 22, 20)},             None),
    "hair_light":      (change_hair,         {"colour": (150, 150, 140)},          None),
    "aging":           (simulate_aging,      {"strength": 0.6},                    None),
    "low_light":       (adjust_lighting,     {"brightness": 0.55, "contrast": 0.8}, None),
    "bright_light":    (adjust_lighting,     {"brightness": 1.35, "contrast": 1.1}, None),
    "warm_cast":       (colour_shift,        {"hue_delta": 8, "sat_scale": 1.15},  None),
    "cool_cast":       (colour_shift,        {"hue_delta": -8, "sat_scale": 0.9},  None),
    "motion_blur":     (motion_blur,         {"kernel": 11, "angle": 20.0},        None),
    "low_resolution":  (degrade_resolution,  {"scale": 0.3, "jpeg_quality": 35},   None),
    "sensor_noise":    (add_sensor_noise,    {"sigma": 12.0},                      None),
    "viewpoint_left":  (change_viewpoint,    {"yaw_deg": -28.0},                   None),
    "viewpoint_right": (change_viewpoint,    {"yaw_deg": 28.0},                    None),
    "viewpoint_high":  (change_viewpoint,    {"pitch_deg": 22.0},                  None),
    "occluded_left":   (random_erase,        {"area_ratio": 0.18, "position": "left"},   "partial"),
    "occluded_right":  (random_erase,        {"area_ratio": 0.18, "position": "right"},  "partial"),
    "occluded_bottom": (random_erase,        {"area_ratio": 0.22, "position": "bottom"}, "partial"),
}

FACE_AUGMENTATIONS = [k for k, (_, _, lbl) in AUGMENTATIONS.items() if lbl]
PHOTOMETRIC_AUGMENTATIONS = [
    "low_light", "bright_light", "warm_cast", "cool_cast",
    "motion_blur", "low_resolution", "sensor_noise",
]
GEOMETRIC_AUGMENTATIONS = ["viewpoint_left", "viewpoint_right", "viewpoint_high"]


def augment(
    image: np.ndarray,
    kinds: Optional[List[str]] = None,
    *,
    face_bbox: Optional[Tuple[float, float, float, float]] = None,
    source_identity: str = "",
    compose: int = 1,
) -> List[AugmentedSample]:
    """Apply each named augmentation, optionally composing several at once."""
    kinds = kinds or list(AUGMENTATIONS.keys())
    samples: List[AugmentedSample] = []

    for kind in kinds:
        entry = AUGMENTATIONS.get(kind)
        if entry is None:
            continue
        fn, params, occ_label = entry
        try:
            kwargs = dict(params)
            if _accepts_face_bbox(fn):
                kwargs["face_bbox"] = face_bbox
            out = fn(image, **kwargs)
            samples.append(
                AugmentedSample(
                    image=out,
                    augmentation=kind,
                    params=dict(params),
                    source_identity=source_identity,
                    occlusion_label=occ_label,
                )
            )
        except Exception:
            log.exception("augmentation %s failed", kind)

    if compose > 1 and len(samples) >= 2:
        rng = np.random.default_rng(1337)
        for _ in range(min(compose, 8)):
            picks = rng.choice(len(kinds), size=2, replace=False)
            chain = [kinds[int(i)] for i in picks]
            img = image
            params: Dict[str, Any] = {}
            labels: List[str] = []
            ok = True
            for kind in chain:
                entry = AUGMENTATIONS.get(kind)
                if entry is None:
                    ok = False
                    break
                fn, p, occ_label = entry
                kwargs = dict(p)
                if _accepts_face_bbox(fn):
                    kwargs["face_bbox"] = face_bbox
                try:
                    img = fn(img, **kwargs)
                except Exception:
                    ok = False
                    break
                params[kind] = p
                if occ_label:
                    labels.append(occ_label)
            if ok:
                samples.append(
                    AugmentedSample(
                        image=img,
                        augmentation="+".join(chain),
                        params=params,
                        source_identity=source_identity,
                        occlusion_label=labels[0] if labels else None,
                    )
                )
    return samples


def _accepts_face_bbox(fn: Callable[..., Any]) -> bool:
    import inspect

    try:
        return "face_bbox" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False

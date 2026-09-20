"""Density-map crowd counting (CSRNet trained on UCF-QNRF).

Detection-based counting needs every person to be a separable box. In a packed
crowd - heads a few pixels wide, heavily occluded - that assumption breaks and
the count collapses. A density network predicts people-per-pixel instead, and
its integral is the count.

Loaded only when training/train_crowd.py promoted a model, i.e. when it beat the
detector-count baseline on the UCF-QNRF test split. Otherwise `available` is
False and the crowd agent keeps counting detections, exactly as before.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..config import get_settings
from ..core.device import get_device_manager

log = logging.getLogger("sentinel.crowd_density")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_csrnet(pretrained: bool = True):
    """CSRNet: VGG16 frontend (to conv4_3) + dilated backend, density at 1/8."""
    import torch.nn as nn
    from torchvision.models import VGG16_Weights, vgg16

    class CSRNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
            self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
            layers, channels = [], 512
            for out in (512, 512, 512, 256, 128, 64):
                layers += [nn.Conv2d(channels, out, 3, padding=2, dilation=2), nn.ReLU(inplace=True)]
                channels = out
            self.backend = nn.Sequential(*layers)
            self.head = nn.Conv2d(64, 1, 1)

        def forward(self, x):
            return self.head(self.backend(self.frontend(x)))

    return CSRNet()


class DensityCounter:
    def __init__(self) -> None:
        self._model = None
        self.meta: Dict[str, Any] = {}
        self.scale = 100.0
        self.max_side = 1024
        self._lock = threading.Lock()
        self._load()

    @property
    def available(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        path = get_settings().storage_dir / "models" / "crowd" / "csrnet_qnrf.pt"
        if not path.exists():
            return
        try:
            import torch

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = build_csrnet(pretrained=False)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.meta = checkpoint.get("meta", {})
            self.scale = float(self.meta.get("density_scale", 100.0))
            # Serve at the resolution it was trained and tested at: on the
            # UCF-QNRF test split, 1024 px gave MAE 126.7 (bias -42) versus
            # 119.1 (bias -7) at the training size of 1536 px.
            self.max_side = int(self.meta.get("max_side", self.max_side))
            self._model = get_device_manager().register_model("crowd:csrnet", model)
            log.info("crowd density model loaded (test MAE %s vs detector %s)",
                     self.meta.get("density_net", {}).get("mae"),
                     self.meta.get("detector_count_baseline", {}).get("mae"))
        except Exception:
            log.exception("failed to load crowd density model")
            self._model = None

    def estimate(self, frame: np.ndarray) -> Optional[Tuple[float, np.ndarray]]:
        """(count, density map at 1/8 resolution) for one BGR frame."""
        if self._model is None or frame is None:
            return None
        import cv2
        import torch

        h, w = frame.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        frame = frame[: h - h % 8, : w - w % 8]
        rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())[None]
        with self._lock, torch.inference_mode():
            param = next(self._model.parameters())
            density = self._model(tensor.to(param.device, param.dtype)).float()[0, 0].cpu().numpy()
        density = np.clip(density, 0, None) / self.scale
        return float(density.sum()), density

    def info(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "model": self.meta.get("model"),
            "test_mae": self.meta.get("density_net", {}).get("mae"),
            "detector_baseline_mae": self.meta.get("detector_count_baseline", {}).get("mae"),
        }


_counter: Optional[DensityCounter] = None


def get_density_counter() -> DensityCounter:
    global _counter
    if _counter is None:
        _counter = DensityCounter()
    return _counter

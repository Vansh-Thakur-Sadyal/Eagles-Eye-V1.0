"""Scene-level anomaly and violence recognition (spec S3 "Events", S30).

Uses the heads trained by training/train_anomaly.py on UCF-Crime (anomaly) and
XD-Violence (violence), on top of a Kinetics-pretrained R(2+1)D-18 video
backbone. Unlike the geometric agents, this one looks at *what is happening*
in the scene - a fight, a robbery, a collision - rather than at trajectories.

Temporal scale matters and is matched deliberately. In training each 16-frame
clip spans one of 32 segments of a video, which for UCF-Crime works out to
roughly 7-8 seconds. Feeding the heads 16 consecutive frames (half a second)
would show them something unlike anything they learned from, so frames are
sampled at ~2 fps into a rolling 16-frame buffer and scored every few seconds.

The agent is silent until a head that passed its training gate exists in
storage/models/anomaly/. What it reports is a scene-level score with the
model's measured accuracy attached - it never names or accuses a person.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from ..config import get_settings
from ..core.device import get_device_manager
from .base import Agent, FrameContext, Finding

log = logging.getLogger("sentinel.agents.video_events")

CLIP_LEN = 16
SIZE = 112
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

TASKS = {
    # task -> (finding behaviour, human label)
    "anomaly": ("anomalous_activity", "anomalous activity"),
    "violence": ("violence_detected", "possible violent activity"),
}


def _preprocess(frame: np.ndarray) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    scale = 128 / min(h, w)
    small = cv2.resize(frame, (max(SIZE, int(w * scale)), max(SIZE, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    y, x = (h - SIZE) // 2, (w - SIZE) // 2
    crop = small[y:y + SIZE, x:x + SIZE, ::-1].astype(np.float32) / 255.0
    return (crop - MEAN) / STD


class VideoEventAgent(Agent):
    name = "video_events"
    spec_id = 15
    description = "Scene-level anomaly and violence recognition (UCF-Crime / XD-Violence heads)"

    SAMPLE_INTERVAL = 0.5     # seconds between buffered frames (~2 fps)
    SCORE_INTERVAL = 4.0      # seconds between scoring passes
    REPORT_COOLDOWN = 60.0

    def __init__(self) -> None:
        super().__init__()
        self._backbone = None
        self._heads: Dict[str, Any] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._buffers: Dict[str, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=CLIP_LEN))
        self._last_sample: Dict[str, float] = defaultdict(lambda: -1e9)
        self._last_score: Dict[str, float] = defaultdict(lambda: -1e9)
        self._last_report: Dict[str, float] = {}
        self._latest: Dict[str, Dict[str, float]] = {}
        self._load()

    # ---------------------------------------------------------------- load
    def _load(self) -> None:
        folder = get_settings().storage_dir / "models" / "anomaly"
        head_files = {t: folder / f"{t}_head.pt" for t in TASKS}
        if not any(p.exists() for p in head_files.values()):
            log.info("no trained anomaly/violence head in %s - agent idle", folder)
            return
        try:
            import torch
            from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

            import torch.nn as nn

            dm = get_device_manager()
            backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
            backbone.fc = nn.Identity()
            backbone.eval()
            self._backbone = dm.register_model("video_events:backbone", backbone)

            for task, path in head_files.items():
                if not path.exists():
                    continue
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                head = nn.Sequential(
                    nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.6),
                    nn.Linear(512, 32), nn.ReLU(), nn.Dropout(0.6),
                    nn.Linear(32, 1), nn.Sigmoid(),
                )
                head.load_state_dict(checkpoint["state_dict"])
                head.eval()
                self._heads[task] = dm.register_model(f"video_events:{task}", head)
                self._meta[task] = checkpoint.get("meta", {})
                log.info("%s head loaded (video AUC %s)", task,
                         self._meta[task].get("video_auc"))
        except Exception:
            log.exception("failed to load video event models")
            self._backbone, self._heads = None, {}

    @property
    def ready(self) -> bool:
        return self._backbone is not None and bool(self._heads)

    def reset(self, camera_id: Optional[str] = None) -> None:
        if camera_id:
            self._buffers.pop(camera_id, None)
            self._latest.pop(camera_id, None)

    # ------------------------------------------------------------- process
    def process(self, ctx: FrameContext) -> List[Finding]:
        if not self.ready or ctx.frame is None:
            return []
        cam = ctx.camera.camera_id
        now = ctx.elapsed_seconds

        if now - self._last_sample[cam] >= self.SAMPLE_INTERVAL:
            self._buffers[cam].append(_preprocess(ctx.frame))
            self._last_sample[cam] = now

        buffer = self._buffers[cam]
        if len(buffer) < CLIP_LEN or now - self._last_score[cam] < self.SCORE_INTERVAL:
            return []
        self._last_score[cam] = now

        scores = self._score(np.stack(buffer))
        self._latest[cam] = scores
        cfg = ctx.policy.get("video_events", {})
        findings: List[Finding] = []

        for task, score in scores.items():
            behaviour, label = TASKS[task]
            threshold = float(cfg.get(f"{task}_threshold",
                                      self._meta.get(task, {}).get("threshold", 0.6)))
            if score < threshold:
                continue
            key = f"{cam}:{task}"
            last = self._last_report.get(key)
            if last is not None and now - last < self.REPORT_COOLDOWN:
                continue
            self._last_report[key] = now
            meta = self._meta.get(task, {})
            confidence = float(np.clip(score, 0.0, 0.99))
            findings.append(Finding(
                behavior=behaviour,
                confidence=round(confidence, 3),
                severity="high" if confidence >= 0.8 else "medium",
                zone_id=ctx.camera.zone_id,
                explanation=(
                    f"Scene-level {label} score {score:.2f} over the last ~8 seconds at "
                    f"{ctx.camera.name} (threshold {threshold:.2f}). This model scores the "
                    f"scene as a whole; it does not identify or accuse any person. Measured "
                    f"accuracy: video-level AUC {meta.get('video_auc', 'n/a')} "
                    f"({meta.get('evaluation', 'evaluation details unavailable')})."
                ),
                evidence={
                    "score": round(float(score), 4),
                    "threshold": threshold,
                    "window_seconds": CLIP_LEN * self.SAMPLE_INTERVAL,
                    "model": meta.get("backbone", "r2plus1d_18"),
                    "trained_on": "UCF-Crime" if task == "anomaly" else "XD-Violence",
                    "video_auc": meta.get("video_auc"),
                },
                dedupe_key=f"video_event:{key}",
            ))
        return findings

    def _score(self, clip: np.ndarray) -> Dict[str, float]:
        import torch

        dm = get_device_manager()
        tensor = torch.from_numpy(clip).permute(3, 0, 1, 2)[None].to(dm.active_device)  # 1,C,T,H,W
        with torch.inference_mode():
            # Match whatever precision/device the GPU toggle last put each
            # module in, rather than converting the modules themselves.
            tensor = tensor.to(next(self._backbone.parameters()).dtype)
            feature = self._backbone(tensor)
            scores = {}
            for task, head in self._heads.items():
                dtype = next(head.parameters()).dtype
                scores[task] = float(head(feature.to(dtype)).float().item())
            return scores

    # ------------------------------------------------------------- readers
    def latest(self, camera_id: str) -> Optional[Dict[str, float]]:
        return self._latest.get(camera_id)

    def status(self) -> Dict[str, Any]:
        base = super().status()
        base.update({
            "ready": self.ready,
            "heads": {t: {"video_auc": m.get("video_auc"), "evaluation": m.get("evaluation")}
                      for t, m in self._meta.items()},
            "backend": "r2plus1d_18 + MIL heads" if self.ready else "idle - no trained head",
        })
        return base

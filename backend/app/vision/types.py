"""Shared value objects passed between vision stages and agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    """One object detected in one frame. xyxy in pixel coordinates."""

    bbox: Tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str
    mask: Optional[Any] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def xywh(self) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox
        return x1, y1, x2 - x1, y2 - y1

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def foot_point(self) -> Tuple[float, float]:
        """Bottom-centre - the best single-point ground proxy for a person."""
        x1, _y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, y2

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class TrackState:
    """Live state of one tracked object inside the pipeline."""

    track_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    score: float
    first_frame: int
    last_frame: int
    first_seen: datetime
    last_seen: datetime
    history: List[Dict[str, Any]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)
    speed: float = 0.0
    embedding: Optional[np.ndarray] = None
    embedding_history: List[np.ndarray] = field(default_factory=list)
    global_id: Optional[str] = None
    db_id: Optional[str] = None
    zone_id: Optional[str] = None
    zones_visited: List[str] = field(default_factory=list)
    face_visibility: str = "unknown"
    occlusion_labels: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    flags: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def foot_point(self) -> Tuple[float, float]:
        x1, _y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, y2

    @property
    def age_frames(self) -> int:
        return self.last_frame - self.first_frame + 1

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.last_seen - self.first_seen).total_seconds())

    def heading_deg(self) -> Optional[float]:
        vx, vy = self.velocity
        if abs(vx) < 1e-3 and abs(vy) < 1e-3:
            return None
        return float((np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0)


@dataclass
class FrameResult:
    """Everything one pipeline tick produced for one camera."""

    camera_id: str
    frame_index: int
    timestamp: datetime
    width: int
    height: int
    detections: List[Detection] = field(default_factory=list)
    tracks: List[TrackState] = field(default_factory=list)
    inference_ms: float = 0.0
    device: str = "cpu"
    detector_backend: str = ""

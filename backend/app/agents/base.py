"""Agent contract and the per-frame context they share.

Design follows spec S4: no single monolithic model.  Each agent owns one
narrow competence, reads the shared spatio-temporal world model, and emits
`Finding` objects.  The orchestrator fans the frame out to the agents and
funnels findings into the threat assessor and incident commander.

Two rules hold everywhere in this package, taken directly from the spec:
  * agents report *confidence*, never certainty about identity;
  * agents recommend, they never take an irreversible action themselves.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ..vision.types import Detection, TrackState

log = logging.getLogger("sentinel.agents")


@dataclass
class Finding:
    """One observation from one agent."""

    behavior: str
    confidence: float
    severity: str = "info"                 # info | low | medium | high | critical
    track_id: Optional[str] = None
    secondary_track_id: Optional[str] = None
    object_id: Optional[str] = None
    zone_id: Optional[str] = None
    explanation: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    agent: str = ""
    dedupe_key: str = ""

    def key(self) -> str:
        return self.dedupe_key or f"{self.agent}:{self.behavior}:{self.track_id}:{self.zone_id}"


@dataclass
class CameraContext:
    """Static-ish camera facts an agent needs; refreshed when the record changes."""

    camera_id: str
    name: str
    width: int
    height: int
    fps: float
    zone_id: Optional[str] = None
    site_id: Optional[str] = None
    zones: List[Dict[str, Any]] = field(default_factory=list)
    expected_flow_deg: Optional[float] = None
    homography: Optional[List[List[float]]] = None
    line_crossings: List[Dict[str, Any]] = field(default_factory=list)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    floor: int = 0
    location: str = ""


@dataclass
class FrameContext:
    """Everything the agents see for one tick of one camera."""

    camera: CameraContext
    frame_index: int
    timestamp: datetime
    frame: Optional[np.ndarray]
    detections: List[Detection]
    tracks: List[TrackState]
    policy: Dict[str, Any]
    elapsed_seconds: float = 0.0
    device: str = "cpu"
    extras: Dict[str, Any] = field(default_factory=dict)

    def tracks_of(self, class_name: str) -> List[TrackState]:
        return [t for t in self.tracks if t.class_name == class_name]

    def persons(self) -> List[TrackState]:
        return self.tracks_of("person")

    def crop(self, track: TrackState) -> Optional[np.ndarray]:
        if self.frame is None:
            return None
        x1, y1, x2, y2 = [int(max(0, v)) for v in track.bbox]
        y2 = min(y2, self.frame.shape[0])
        x2 = min(x2, self.frame.shape[1])
        if x2 <= x1 or y2 <= y1:
            return None
        return self.frame[y1:y2, x1:x2]


class Agent:
    """Base class. Subclasses implement `process` and keep their own state."""

    name: str = "agent"
    description: str = ""
    # Which of the spec's numbered agents this implements, for the UI.
    spec_id: int = 0

    def __init__(self) -> None:
        self.enabled = True
        self.last_run_ms: float = 0.0
        self.total_runs: int = 0
        self.total_findings: int = 0
        self.last_error: Optional[str] = None

    def process(self, ctx: FrameContext) -> List[Finding]:
        raise NotImplementedError

    def run(self, ctx: FrameContext) -> List[Finding]:
        """Timed, error-isolated wrapper. One failing agent never stops a frame."""
        if not self.enabled:
            return []
        started = time.perf_counter()
        try:
            findings = self.process(ctx) or []
            for f in findings:
                f.agent = f.agent or self.name
            self.total_findings += len(findings)
            self.last_error = None
            return findings
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.exception("agent %s failed on camera %s", self.name, ctx.camera.camera_id)
            return []
        finally:
            self.last_run_ms = (time.perf_counter() - started) * 1000.0
            self.total_runs += 1

    def reset(self, camera_id: Optional[str] = None) -> None:
        """Drop per-camera state, e.g. when a stream restarts."""

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "spec_id": self.spec_id,
            "description": self.description,
            "enabled": self.enabled,
            "last_run_ms": round(self.last_run_ms, 2),
            "total_runs": self.total_runs,
            "total_findings": self.total_findings,
            "last_error": self.last_error,
        }


def severity_from_confidence(confidence: float, base: str = "medium") -> str:
    """Map a confidence to a severity band, capped by the behaviour's base."""
    order = ["info", "low", "medium", "high", "critical"]
    if confidence >= 0.85:
        level = "high"
    elif confidence >= 0.65:
        level = "medium"
    elif confidence >= 0.45:
        level = "low"
    else:
        level = "info"
    return order[min(order.index(level), order.index(base))]

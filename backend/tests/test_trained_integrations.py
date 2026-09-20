"""Integration of the trained models into the live agents.

Uses stand-in models so the wiring is verified independently of how the real
training runs turn out: thresholds, cooldowns, density take-over, honest
explanations, and that nothing activates without a promoted model.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.agents.base import CameraContext, FrameContext
from app.agents.crowd import CrowdAgent
from app.agents.video_events import VideoEventAgent
from app.config import load_policy
from app.vision.types import TrackState

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
BANNED = ("criminal", "stalker", "terrorist", "suspect", "perpetrator")


def _ctx(frame, elapsed, tracks=(), camera="CAM_T"):
    return FrameContext(
        camera=CameraContext(camera_id=camera, name="Test Cam", width=640, height=360, fps=12),
        frame_index=int(elapsed * 12), timestamp=T0 + timedelta(seconds=elapsed),
        frame=frame, detections=[], tracks=list(tracks), policy=load_policy(),
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------- video events
torch = pytest.importorskip("torch")


class _ConstantHead(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.anchor = torch.nn.Parameter(torch.zeros(1))    # gives the module a dtype

    def forward(self, x):
        return torch.full((x.shape[0], 1), self.value, dtype=x.dtype, device=x.device)


class _FakeBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros((x.shape[0], 512), dtype=x.dtype, device=x.device)


def _armed_agent(violence: float, anomaly: float) -> VideoEventAgent:
    agent = VideoEventAgent()
    agent._backbone = _FakeBackbone()
    agent._heads = {"violence": _ConstantHead(violence), "anomaly": _ConstantHead(anomaly)}
    agent._meta = {
        "violence": {"threshold": 0.6, "video_auc": 0.9, "evaluation": "test"},
        "anomaly": {"threshold": 0.6, "video_auc": 0.8, "evaluation": "test"},
    }
    return agent


def _feed(agent: VideoEventAgent, seconds: float, start: float = 0.0):
    frame = np.full((360, 640, 3), 120, dtype=np.uint8)
    findings = []
    t = start
    while t < start + seconds:
        findings += agent.run(_ctx(frame, t))
        t += 0.25
    return findings


def test_video_agent_is_idle_without_a_promoted_model():
    agent = VideoEventAgent()
    agent._backbone, agent._heads = None, {}
    assert agent.ready is False
    assert _feed(agent, 20) == []


def test_video_agent_needs_a_full_clip_before_scoring():
    """16 frames at ~2 fps = 8 s of context before any score is produced."""
    agent = _armed_agent(violence=0.95, anomaly=0.1)
    assert _feed(agent, 6.0) == [], "scored before a full 16-frame clip existed"


def test_violence_above_threshold_fires_and_explains_itself():
    agent = _armed_agent(violence=0.95, anomaly=0.1)
    findings = _feed(agent, 12.0)
    violence = [f for f in findings if f.behavior == "violence_detected"]
    assert len(violence) == 1
    f = violence[0]
    assert f.severity == "high"
    assert "does not identify or accuse" in f.explanation
    assert "AUC" in f.explanation
    assert not any(word in f.explanation.lower() for word in BANNED)
    assert not [x for x in findings if x.behavior == "anomalous_activity"], \
        "anomaly fired although its score was below threshold"


def test_video_agent_respects_its_cooldown():
    agent = _armed_agent(violence=0.95, anomaly=0.95)
    findings = _feed(agent, 40.0)
    assert sum(f.behavior == "violence_detected" for f in findings) == 1
    assert sum(f.behavior == "anomalous_activity" for f in findings) == 1


def test_policy_threshold_overrides_the_calibrated_one():
    agent = _armed_agent(violence=0.7, anomaly=0.1)
    frame = np.full((360, 640, 3), 120, dtype=np.uint8)
    policy = load_policy()
    policy["video_events"] = {"violence_threshold": 0.9}
    findings, t = [], 0.0
    while t < 12:
        ctx = _ctx(frame, t)
        ctx.policy = policy
        findings += agent.run(ctx)
        t += 0.25
    assert not findings, "an operator-raised threshold was ignored"


def test_video_findings_carry_threat_weight():
    policy = load_policy()
    assert policy["threat_weights"]["violence_detected"] > 0
    assert policy["threat_weights"]["anomalous_activity"] > 0


# ------------------------------------------------------------ crowd density
class _FakeCounter:
    def __init__(self, count: float) -> None:
        self.count = count
        self.available = True

    def estimate(self, frame):
        density = np.zeros((45, 80), dtype=np.float32)
        density[10:30, 10:60] = self.count / (20 * 50)
        return self.count, density


def _person(tid: int, x: float, y: float) -> TrackState:
    return TrackState(track_id=tid, class_name="person", bbox=(x - 10, y - 40, x + 10, y),
                      score=0.9, first_frame=0, last_frame=10, first_seen=T0, last_seen=T0)


def test_density_estimate_takes_over_in_a_dense_crowd():
    agent = CrowdAgent()
    agent.density = _FakeCounter(120.0)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    tracks = [_person(i, 50 + i * 40, 200) for i in range(6)]   # detector sees 6
    agent.run(_ctx(frame, 0.0, tracks))
    metrics = agent.latest("CAM_T")
    assert metrics["count_method"] == "density"
    assert metrics["count"] == 120
    assert metrics["count_detected"] == 6


def test_detection_count_kept_when_density_agrees():
    agent = CrowdAgent()
    agent.density = _FakeCounter(7.0)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    tracks = [_person(i, 50 + i * 40, 200) for i in range(6)]
    agent.run(_ctx(frame, 0.0, tracks))
    metrics = agent.latest("CAM_T")
    assert metrics["count_method"] == "detection"
    assert metrics["count"] == 6
    assert metrics["count_density"] == 7


def test_crowd_agent_unchanged_without_a_density_model():
    agent = CrowdAgent()
    agent.density.available and pytest.skip("a real density model is installed")
    tracks = [_person(i, 50 + i * 40, 200) for i in range(4)]
    agent.run(_ctx(np.zeros((360, 640, 3), dtype=np.uint8), 0.0, tracks))
    metrics = agent.latest("CAM_T")
    assert metrics["count"] == 4
    assert metrics["count_method"] == "detection"
    assert metrics["count_density"] is None


def test_pooling_preserves_the_count():
    density = np.random.default_rng(0).random((45, 80)).astype(np.float32)
    pooled = CrowdAgent._pool(density, rows=5, cols=10)
    assert pooled.shape == (5, 10)
    assert abs(pooled.sum() - density.sum()) < 1e-3


# ---------------------------------------------------------------- detector
def test_detector_reports_where_its_weights_came_from():
    from app.vision.detector import YoloDetector

    detector = YoloDetector.__new__(YoloDetector)
    detector.weights = "yolo11m.pt"
    detector.weights_source = "configured"
    path = detector._resolve_weights()
    assert detector.weights_source in ("configured", "fine-tuned (promoted after comparison)")
    if detector.weights_source.startswith("fine-tuned"):
        assert path.endswith("sentinel_detector.pt")


# ------------------------------------------------------------------- re-id
def test_promoted_reid_checkpoint_actually_loads():
    """A promoted checkpoint must be used, not silently replaced by ImageNet weights."""
    pytest.importorskip("torchreid")
    from app.vision.reid import BodyEmbedder

    path = BodyEmbedder._reid_weights_path()
    if path is None:
        pytest.skip("no Re-ID checkpoint promoted")
    import torchreid

    model = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=False)
    loaded = BodyEmbedder._load_reid_checkpoint(model, path)
    assert loaded > 100, "checkpoint matched too few tensors"

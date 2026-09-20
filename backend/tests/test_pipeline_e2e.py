"""End-to-end pipeline validation on the synthetic scene.

Drives the real capture -> detect -> track -> agents -> incident path with no
hardware and no downloaded weights, and asserts the agents actually fire on the
behaviours the scene was built to contain.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from app.agents.base import CameraContext, FrameContext
from app.agents.orchestrator import Orchestrator
from app.config import load_policy
from app.pipeline.capture import SourceSpec, build_source
from app.vision.detector import MotionDetector
from app.vision.tracker import ByteTracker


def _run_scene(frames: int = 340, policy_overrides=None):
    """Play the synthetic scene through the full stack; return findings + results."""
    spec = SourceSpec(
        source_type="synthetic", uri="", width=960, height=540, fps=12,
        options={"crowd": 9, "seed": 7},
    )
    source = build_source(spec)
    assert source.open(), source.last_error

    detector = MotionDetector()
    assert detector.ready, "OpenCV motion fallback unavailable"

    tracker = ByteTracker(high_thresh=0.45, low_thresh=0.1, min_hits=3, max_age=40)
    orch = Orchestrator()

    policy = load_policy()
    # Compress the time-based thresholds so a short scene exercises them.
    policy["behavior"]["loitering_seconds"] = 6
    policy["behavior"]["counter_flow_min_frames"] = 8
    policy["object"]["unattended_seconds"] = 8
    policy["object"]["warning_seconds"] = 4
    policy["following"]["min_duration_seconds"] = 6
    policy["crowd"]["flow_window_seconds"] = 8
    if policy_overrides:
        for section, values in policy_overrides.items():
            policy.setdefault(section, {}).update(values)
    orch._policy = policy

    captured = []
    orch.add_sink(captured.append)

    t0 = datetime(2026, 1, 1, 18, 40, 0, tzinfo=timezone.utc)
    behaviors = set()
    results = []

    for i in range(frames):
        ok, frame = source.read()
        assert ok and frame is not None
        now = t0 + timedelta(seconds=i / 12.0)

        detections = detector.detect(frame)
        tracks = tracker.update(detections, timestamp=now, fps=12.0)

        ctx = FrameContext(
            camera=CameraContext(
                camera_id="CAM_TEST", name="Synthetic Concourse",
                width=frame.shape[1], height=frame.shape[0], fps=12.0,
                zone_id="ZONE_A", site_id="SITE_TEST", location="Test Concourse",
                zones=[
                    {
                        "id": "ZONE_RESTRICTED",
                        "name": "Restricted Apron",
                        "zone_type": "restricted",
                        "polygon": [[0.72, 0.55], [0.97, 0.55], [0.97, 0.92], [0.72, 0.92]],
                        "risk_weight": 1.3,
                    }
                ],
            ),
            frame_index=i,
            timestamp=now,
            frame=frame,
            detections=detections,
            tracks=tracks,
            policy=policy,
            elapsed_seconds=i / 12.0,
        )
        res = orch.process_frame(ctx)
        results.append(res)

    for evt in orch._open.values():
        for oi in evt:
            behaviors.update(oi.behaviors)

    source.release()
    return {
        "orchestrator": orch,
        "results": results,
        "incidents": captured,
        "behaviors": behaviors,
    }


@pytest.fixture(scope="module")
def scene():
    return _run_scene()


def test_synthetic_source_produces_frames():
    spec = SourceSpec(source_type="synthetic", uri="", width=640, height=360, fps=12)
    src = build_source(spec)
    assert src.open()
    ok, frame = src.read()
    assert ok and frame is not None
    assert frame.shape == (360, 640, 3)
    src.release()


def test_detector_finds_people_in_the_scene(scene):
    detections_seen = sum(r["person_count"] for r in scene["results"])
    assert detections_seen > 200, f"only {detections_seen} person-frames tracked"


def test_tracks_are_created_and_persist(scene):
    peak = max(r["track_count"] for r in scene["results"])
    assert peak >= 4, f"peak concurrent tracks was only {peak}"


def test_agents_actually_fire(scene):
    """The scene contains loitering, counter-flow and an abandoned bag."""
    behaviors = scene["behaviors"]
    assert behaviors, "no behaviours detected at all across the whole scene"
    expected_any = {"loitering", "counter_flow", "unattended_object", "object_separation",
                    "following", "restricted_entry", "crowd_surge", "abnormal_trajectory"}
    assert behaviors & expected_any, f"none of the scripted behaviours fired: {behaviors}"


def test_incidents_are_created_with_explanations(scene):
    incidents = scene["incidents"]
    assert incidents, "no incidents were raised"
    for inc in incidents:
        assert inc["summary"], "incident has no summary"
        assert inc["explanation"], "incident has no explanation"
        assert 0 <= inc["risk_score"] <= 100
        assert inc["severity"] in ("info", "low", "medium", "high", "critical")
        assert inc["recommended_actions"], "incident carries no recommended actions"
        assert inc["timeline"], "incident has an empty timeline"


def test_every_high_consequence_action_requires_approval(scene):
    """Structural guarantee: the AI recommends, a human decides."""
    for inc in scene["incidents"]:
        for action in inc["recommended_actions"]:
            if action["consequence"] in ("physical response", "physical access control",
                                         "privacy escalation", "public communication"):
                assert action["requires_human_approval"] is True, action
            assert action["status"] == "proposed"


def test_threat_score_is_always_explained(scene):
    for inc in scene["incidents"]:
        assert len(inc["explanation"]) > 30
        assert isinstance(inc["risk_factors"], list)
        for factor in inc["risk_factors"]:
            assert "weight" in factor and "confidence" in factor and "contribution" in factor


def test_observability_signals_carry_no_risk_weight(scene):
    """Spec S8: a covered face must never inflate a score."""
    for inc in scene["incidents"]:
        for factor in inc["risk_factors"]:
            if factor["behavior"] in ("face_unavailable", "appearance_change",
                                      "cross_camera_association"):
                assert factor["weight"] == 0, factor
                assert factor["counted"] is False


def test_no_incident_claims_identity_or_contents(scene):
    """Spec S12/S6: no contents inference, no identity assertion."""
    banned = ["explosive", "bomb", "stalker", "criminal", "terrorist",
              "definitely the same person", "this is the same person"]
    for inc in scene["incidents"]:
        blob = (inc["summary"] + " " + inc["explanation"]).lower()
        for word in banned:
            assert word not in blob, f"incident text contained banned phrase '{word}'"


def test_crowd_telemetry_is_measured(scene):
    crowd_samples = [r["crowd"] for r in scene["results"] if r.get("crowd")]
    assert crowd_samples, "crowd agent produced no telemetry"
    last = crowd_samples[-1]
    for key in ("count", "density", "density_band", "risk", "compression"):
        assert key in last
    assert last["density_band"] in ("low", "medium", "high", "critical")


def test_incidents_are_correlated_not_duplicated(scene):
    """One ongoing situation must not produce a new incident per frame."""
    ids = {inc["id"] for inc in scene["incidents"]}
    assert len(ids) < 25, f"correlation failed: {len(ids)} distinct incidents raised"


def test_orchestrator_reports_agent_health(scene):
    status = scene["orchestrator"].status()
    names = {a["name"] for a in status["agents"]}
    assert len(status["agents"]) == len(scene["orchestrator"].agents)
    assert {"reid", "crowd", "watchlist", "video_events"} <= names, names
    for agent in status["agents"]:
        assert agent["last_error"] is None, f"{agent['name']} errored: {agent['last_error']}"
        assert agent["total_runs"] > 0

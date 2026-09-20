"""Regression tests for bugs found by the agent evaluation harness.

Each test here corresponds to a defect that was actually shipped and then
fixed. They are written to fail loudly if the same mistake returns, because
every one of them was silent in normal operation.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.agents.base import CameraContext, FrameContext
from app.agents.objects import ObjectAgent
from app.agents.relationship import RelationshipAgent
from app.config import load_policy
from app.vision.types import TrackState

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
FPS = 12.0


def _policy(**overrides):
    p = load_policy()
    for section, values in overrides.items():
        p.setdefault(section, {}).update(values)
    return p


def _track(tid, cls, path, *, start=0):
    """Build a track whose history carries the full path."""
    st = TrackState(
        track_id=tid, class_name=cls,
        bbox=(path[-1][0] - 20, path[-1][1] - 100, path[-1][0] + 20, path[-1][1]),
        score=0.9, first_frame=start, last_frame=start + len(path) - 1,
        first_seen=T0, last_seen=T0 + timedelta(seconds=len(path) / FPS),
    )
    for i, (x, y) in enumerate(path):
        st.history.append({"t": (T0 + timedelta(seconds=i / FPS)).isoformat(), "f": start + i,
                           "x": float(x), "y": float(y), "w": 40.0, "h": 100.0,
                           "conf": 0.9, "vx": 0.0, "vy": 0.0})
    if len(path) >= 2:
        st.velocity = ((path[-1][0] - path[-2][0]) * FPS, (path[-1][1] - path[-2][1]) * FPS)
        st.speed = float(math.hypot(*st.velocity))
    return st


def _ctx(tracks, policy, elapsed, frame_index=1):
    return FrameContext(
        camera=CameraContext(camera_id="CAM_R", name="Regression", width=960, height=540,
                             fps=FPS, zone_id="Z", site_id="S", location="Test"),
        frame_index=frame_index, timestamp=T0 + timedelta(seconds=elapsed),
        frame=None, detections=[], tracks=tracks, policy=policy, elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# BUG: a stranger walking past a bag reset the abandonment clock, so in a
# crowded concourse - the primary deployment - a bag was never flagged.
# ---------------------------------------------------------------------------
def test_passer_by_does_not_reset_the_abandonment_clock():
    agent = ObjectAgent()
    policy = _policy(object={"tracked_classes": ["suitcase"], "unattended_seconds": 6,
                             "warning_seconds": 3, "separation_distance_px": 150})
    bag_at = (500.0, 400.0)
    findings = []

    # Phase 1: the owner settles beside the bag long enough to be established.
    for step in range(20):
        owner = _track(1, "person", [(505.0, 402.0)] * 30)
        bag = _track(2, "suitcase", [bag_at] * 30)
        findings += agent.run(_ctx([owner, bag], policy, step * 0.25))

    # Phase 2: owner leaves for good, but a stream of strangers keeps walking
    # right past the bag, well within the separation distance.
    for step in range(20, 140):
        elapsed = step * 0.25
        owner = _track(1, "person", [(505.0 + (step - 20) * 40, 402.0)] * 30)
        stranger = _track(100 + (step // 8), "person", [(bag_at[0] + 40, bag_at[1] + 20)] * 30)
        bag = _track(2, "suitcase", [bag_at] * 30)
        findings += agent.run(_ctx([owner, stranger, bag], policy, elapsed))

    kinds = {f.behavior for f in findings}
    assert "unattended_object" in kinds, (
        "constant passers-by suppressed the unattended-object escalation entirely"
    )
    hit = next(f for f in findings if f.behavior == "unattended_object")
    assert hit.evidence["owner_track_id"] == 1, "the owner was reassigned to a passer-by"
    assert hit.evidence["bystander_passes"] > 0, "bystander proximity was not recorded"


# ---------------------------------------------------------------------------
# BUG: ownership was granted to whoever happened to be nearest on the object's
# first frame, so the "last associated subject" trajectory named a stranger.
# ---------------------------------------------------------------------------
def test_ownership_requires_sustained_proximity_not_one_frame():
    agent = ObjectAgent()
    policy = _policy(object={"tracked_classes": ["suitcase"], "unattended_seconds": 6,
                             "warning_seconds": 3, "separation_distance_px": 150})
    bag_at = (400.0, 300.0)

    # A stranger is momentarily closest when the bag first appears...
    for step in range(3):
        stranger = _track(99, "person", [(410.0, 305.0)] * 20)
        owner = _track(7, "person", [(520.0, 300.0)] * 20)
        agent.run(_ctx([stranger, owner, _track(2, "suitcase", [bag_at] * 20)], policy, step * 0.2))

    # ...but the real owner is the one who stays beside it.
    for step in range(3, 40):
        owner = _track(7, "person", [(408.0, 303.0)] * 20)
        agent.run(_ctx([owner, _track(2, "suitcase", [bag_at] * 20)], policy, step * 0.2))

    snapshot = agent.snapshot("CAM_R")
    assert snapshot, "the object was not tracked at all"
    assert snapshot[0]["owner_track_id"] == 7, (
        f"ownership went to the brief passer-by ({snapshot[0]['owner_track_id']}) "
        "instead of the subject who stayed with the object"
    )


# ---------------------------------------------------------------------------
# BUG: the follower test could not tell "beside" from "behind", so every pair
# of companions walking together scored as a following pattern.
# ---------------------------------------------------------------------------
def _leader_route(n):
    """An L-shaped route: east, then south."""
    pts = [(100.0 + i * 4, 200.0) for i in range(n // 2)]
    last_x = pts[-1][0]
    pts += [(last_x, 200.0 + i * 4) for i in range(n - len(pts))]
    return pts


def test_companions_walking_abreast_are_not_reported_as_following():
    agent = RelationshipAgent()
    policy = _policy(following={"min_duration_seconds": 3, "confidence_floor": 0.60,
                                "max_pair_distance_px": 300})
    findings = []
    for step in range(120):
        n = step + 30
        leader = _leader_route(n)
        # Perpendicular offset at every step: genuinely side by side.
        companion = []
        for i, p in enumerate(leader):
            nxt = leader[min(i + 1, len(leader) - 1)]
            dx, dy = nxt[0] - p[0], nxt[1] - p[1]
            norm = math.hypot(dx, dy) or 1.0
            companion.append((p[0] - dy / norm * 60, p[1] + dx / norm * 60))
        findings += agent.run(
            _ctx([_track(1, "person", leader), _track(2, "person", companion)], policy, step * 0.4)
        )
    assert not [f for f in findings if f.behavior == "following"], (
        "two people walking abreast were reported as a following pattern"
    )


def test_genuine_follower_is_still_detected():
    """The companion fix must not blind the agent to real trailing."""
    agent = RelationshipAgent()
    policy = _policy(following={"min_duration_seconds": 3, "confidence_floor": 0.60,
                                "max_pair_distance_px": 300})
    findings = []
    for step in range(120):
        n = step + 30
        leader = _leader_route(n)
        lag = 12
        follower = [leader[max(0, i - lag)] for i in range(len(leader))]
        follower = [(x - 8, y - 8) for x, y in follower]
        findings += agent.run(
            _ctx([_track(1, "person", leader), _track(2, "person", follower)], policy, step * 0.4)
        )
    hits = [f for f in findings if f.behavior == "following"]
    assert hits, "a subject trailing by a fixed lag along the same route was not detected"
    best = max(hits, key=lambda f: f.confidence)
    assert best.evidence["leader_consistency"] > 0.75
    assert best.evidence["mean_cross_track_share"] < 0.6


def test_trailing_offset_separates_beside_from_behind():
    """Unit check on the geometry that the whole discrimination rests on."""
    leader = _track(1, "person", [(100.0 + i * 5, 300.0) for i in range(20)])

    behind = _track(2, "person", [(100.0 + i * 5 - 80, 300.0) for i in range(20)])
    beside = _track(3, "person", [(100.0 + i * 5, 360.0) for i in range(20)])

    along_behind, cross_behind = RelationshipAgent._trailing_offset(leader, behind)
    along_beside, cross_beside = RelationshipAgent._trailing_offset(leader, beside)

    assert along_behind > 0.9, f"a directly-trailing subject scored {along_behind}"
    assert cross_behind < 0.2
    assert abs(along_beside) < 0.2, f"a subject walking abreast scored {along_beside} along-track"
    assert cross_beside > 0.9


# ---------------------------------------------------------------------------
# BUG: Track.global_id is a foreign key, but the Re-ID agent mints subject ids
# in memory. Every associated track failed to persist on an FK violation.
# ---------------------------------------------------------------------------
def test_track_with_a_global_id_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DATABASE_URL", f"sqlite:///{(tmp_path / 'fk.db').as_posix()}")
    monkeypatch.setenv("SENTINEL_STORAGE_DIR", str(tmp_path / "storage"))

    from app.config import get_settings

    get_settings.cache_clear()

    import importlib

    import app.db as db_module

    importlib.reload(db_module)
    import app.core.persistence as persistence_module

    importlib.reload(persistence_module)

    db_module.init_db()
    from sqlalchemy import select

    from app.models import Camera, GlobalSubject, Track, new_id, utcnow

    with db_module.session_scope() as db:
        db.add(Camera(id="CAM_FK", name="FK", source_type="synthetic", source_uri=""))

    service = persistence_module.PersistenceService()
    service.on_frame_result({
        "camera_id": "CAM_FK",
        # Track rows are written on a cadence of (measured_fps * 2) frames, so
        # the index must land on that boundary for the write to happen at all.
        "frame_index": 24,
        "measured_fps": 12,
        "timestamp": utcnow().isoformat(),
        "tracks": [{
            "track_id": 1, "global_id": "SUBJ_TESTFK", "class_name": "person",
            "bbox": [0, 0, 10, 10], "score": 0.9, "speed": 1.0, "heading_deg": 0.0,
            "zone_id": None, "age_frames": 5, "face_visibility": "unknown",
            "duration_seconds": 1.0,
        }],
        "crowd": None,
    })

    assert service.errors == 0, "persisting an associated track raised an error"
    with db_module.session_scope() as db:
        track = db.scalar(select(Track).where(Track.camera_id == "CAM_FK"))
        assert track is not None, "the track was not written at all"
        assert track.global_id == "SUBJ_TESTFK"
        subject = db.get(GlobalSubject, "SUBJ_TESTFK")
        assert subject is not None, "no GlobalSubject row was created for the association"
        assert "CAM_FK" in subject.camera_ids

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# BUG: measured_fps reported processing capacity rather than the delivered
# frame rate, inflating every speed and manufacturing false "running" events.
# ---------------------------------------------------------------------------
def test_measured_fps_tracks_wall_clock_not_processing_speed():
    import time

    from app.pipeline.capture import SourceSpec
    from app.pipeline.runner import CameraRuntime, CameraWorker

    runtime = CameraRuntime(
        camera_id="CAM_FPS", name="fps", source_type="synthetic", source_uri="",
        width=320, height=240, fps=10,
    )
    worker = CameraWorker(runtime)
    worker.start()
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            if worker.frame_index > 60:
                break
            time.sleep(0.25)
        assert worker.frame_index > 30, "the worker produced almost no frames"
        # A target of 10 fps must not report 80: that would mean the number is
        # processing throughput, and every downstream speed would be wrong.
        assert worker.measured_fps <= runtime.fps * 1.5, (
            f"measured_fps={worker.measured_fps} far exceeds the {runtime.fps} fps target, "
            "which means it is reporting processing capacity again"
        )
        assert worker.measured_fps > runtime.fps * 0.4
    finally:
        worker.stop()
        worker.join(timeout=5)

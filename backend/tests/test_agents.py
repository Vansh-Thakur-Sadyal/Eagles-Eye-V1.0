"""Targeted agent tests with hand-built scenarios.

These drive each agent with trajectories whose ground truth is known, so a
pass means the agent detected the thing it claims to detect - and, just as
importantly, that it does NOT fire on the benign control case.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.agents.base import CameraContext, FrameContext
from app.agents.behavior import BehaviorAgent
from app.agents.commander import IncidentCommander
from app.agents.crowd import CrowdAgent
from app.agents.objects import ObjectAgent
from app.agents.relationship import RelationshipAgent
from app.agents.threat import ThreatAssessor
from app.agents.watchlist import EnrolledSubject, WatchlistAgent
from app.config import load_policy
from app.vision.types import TrackState

T0 = datetime(2026, 1, 1, 18, 40, 0, tzinfo=timezone.utc)
FPS = 12.0


def _policy(**overrides):
    p = load_policy()
    for section, values in overrides.items():
        p.setdefault(section, {}).update(values)
    return p


def _track(tid, cls, path, *, start_frame=0, class_name=None):
    """Build a TrackState from an explicit list of (x, y) centre points."""
    cls = class_name or cls
    x, y = path[0]
    st = TrackState(
        track_id=tid, class_name=cls,
        bbox=(x - 20, y - 100, x + 20, y),
        score=0.9, first_frame=start_frame, last_frame=start_frame + len(path) - 1,
        first_seen=T0, last_seen=T0 + timedelta(seconds=len(path) / FPS),
    )
    for i, (px, py) in enumerate(path):
        st.history.append(
            {"t": (T0 + timedelta(seconds=i / FPS)).isoformat(), "f": start_frame + i,
             "x": float(px), "y": float(py), "w": 40.0, "h": 100.0, "conf": 0.9,
             "vx": 0.0, "vy": 0.0}
        )
    if len(path) >= 2:
        dx = (path[-1][0] - path[-2][0]) * FPS
        dy = (path[-1][1] - path[-2][1]) * FPS
        st.velocity = (dx, dy)
        st.speed = float(math.hypot(dx, dy))
    st.bbox = (path[-1][0] - 20, path[-1][1] - 100, path[-1][0] + 20, path[-1][1])
    return st


def _ctx(tracks, *, policy, elapsed, zones=None, frame_index=1, width=960, height=540):
    return FrameContext(
        camera=CameraContext(
            camera_id="CAM_T", name="Test Camera", width=width, height=height, fps=FPS,
            zone_id="ZONE_A", site_id="SITE_T", location="Test Hall", zones=zones or [],
        ),
        frame_index=frame_index, timestamp=T0 + timedelta(seconds=elapsed),
        frame=None, detections=[], tracks=tracks, policy=policy, elapsed_seconds=elapsed,
    )


# ------------------------------------------------------------------ behaviour
def test_loitering_fires_on_confined_dwell():
    agent = BehaviorAgent()
    policy = _policy(behavior={"loitering_seconds": 10, "loitering_radius_px": 60,
                               "min_track_age_frames": 4})
    findings = []
    for step in range(200):
        elapsed = step * 0.5
        # jitters inside a 20 px box - waiting, not travelling
        path = [(400 + 8 * math.sin(i / 5), 300 + 6 * math.cos(i / 5)) for i in range(step + 6)]
        findings += agent.run(_ctx([_track(1, "person", path[-40:] or path)],
                                   policy=policy, elapsed=elapsed))
    assert any(f.behavior == "loitering" for f in findings)
    loiter = next(f for f in findings if f.behavior == "loitering")
    assert loiter.evidence["dwell_seconds"] >= 10
    assert "remained within" in loiter.explanation


def test_loitering_does_not_fire_on_someone_walking_through():
    agent = BehaviorAgent()
    policy = _policy(behavior={"loitering_seconds": 10, "loitering_radius_px": 60,
                               "min_track_age_frames": 4})
    findings = []
    for step in range(200):
        path = [(100 + i * 6, 300) for i in range(step + 6)]
        findings += agent.run(_ctx([_track(1, "person", path[-40:] or path)],
                                   policy=policy, elapsed=step * 0.5))
    assert not any(f.behavior == "loitering" for f in findings), \
        "a person walking straight through was flagged as loitering"


def test_counter_flow_needs_a_crowd_to_oppose():
    """A lone subject cannot be 'counter-flow' - there is no flow."""
    agent = BehaviorAgent()
    policy = _policy(behavior={"counter_flow_min_frames": 3, "min_track_age_frames": 2})
    solo = _track(1, "person", [(800 - i * 10, 300) for i in range(30)])
    findings = []
    for step in range(20):
        findings += agent.run(_ctx([solo], policy=policy, elapsed=step))
    assert not any(f.behavior == "counter_flow" for f in findings)


def test_counter_flow_fires_against_a_real_crowd():
    agent = BehaviorAgent()
    policy = _policy(behavior={"counter_flow_min_frames": 3, "min_track_age_frames": 2})
    findings = []
    for step in range(30):
        crowd = [
            _track(10 + k, "person", [(100 + i * 9, 250 + k * 40) for i in range(25)])
            for k in range(5)
        ]
        wrong_way = _track(1, "person", [(800 - i * 9, 300) for i in range(25)])
        findings += agent.run(_ctx(crowd + [wrong_way], policy=policy, elapsed=step))
    hits = [f for f in findings if f.behavior == "counter_flow"]
    assert hits, "counter-flow subject was not detected against a moving crowd"
    assert hits[0].track_id == "1"
    assert hits[0].evidence["deviation_deg"] > 120


def test_restricted_zone_entry_fires_once_per_entry():
    agent = BehaviorAgent()
    policy = _policy(behavior={"min_track_age_frames": 2})
    zones = [{"id": "Z_RES", "name": "Apron", "zone_type": "restricted",
              "polygon": [[0.6, 0.5], [0.95, 0.5], [0.95, 0.95], [0.6, 0.95]]}]
    findings = []
    for step in range(12):
        # deep inside the restricted polygon
        path = [(700 + i, 400 + i) for i in range(20)]
        findings += agent.run(_ctx([_track(1, "person", path)], policy=policy,
                                   elapsed=step, zones=zones))
    hits = [f for f in findings if f.behavior == "restricted_entry"]
    assert len(hits) == 1, f"expected one entry event, got {len(hits)}"
    assert hits[0].severity == "high"
    assert hits[0].zone_id == "Z_RES"


# --------------------------------------------------------------- relationship
def test_following_detected_for_a_trailing_mirrored_subject():
    agent = RelationshipAgent()
    policy = _policy(following={"min_duration_seconds": 3, "confidence_floor": 0.4,
                                "max_pair_distance_px": 300})
    findings = []
    for step in range(120):
        n = step + 25
        # leader walks an L-shape; follower repeats it 12 samples later
        def leader_at(i):
            return (100 + i * 4, 200) if i < 40 else (260, 200 + (i - 40) * 4)

        def follower_at(i):
            j = max(0, i - 12)
            lx, ly = leader_at(j)
            return (lx - 60, ly + 25)

        lead = _track(1, "person", [leader_at(i) for i in range(n)])
        foll = _track(2, "person", [follower_at(i) for i in range(n)])
        findings += agent.run(_ctx([lead, foll], policy=policy, elapsed=step * 0.4))

    hits = [f for f in findings if f.behavior == "following"]
    assert hits, "a clearly trailing, route-mirroring subject was not flagged"
    best = max(hits, key=lambda f: f.confidence)
    assert best.secondary_track_id == "1"
    assert "human review" in best.explanation.lower()
    assert "stalker" not in best.explanation.lower()
    assert set(best.evidence["components"]) == {
        "route_overlap", "distance_stability", "trailing_persistence",
        "stop_synchronisation", "mirrored_turns", "lag_correlation",
    }


def test_two_friends_walking_side_by_side_are_not_flagged():
    """The control case the spec cares about: co-travel is not following."""
    agent = RelationshipAgent()
    policy = _policy(following={"min_duration_seconds": 3, "confidence_floor": 0.55,
                                "max_pair_distance_px": 300})
    findings = []
    for step in range(120):
        n = step + 25
        a = _track(1, "person", [(100 + i * 4, 200) for i in range(n)])
        b = _track(2, "person", [(100 + i * 4, 260) for i in range(n)])   # abreast, no lag
        findings += agent.run(_ctx([a, b], policy=policy, elapsed=step * 0.4))
    hits = [f for f in findings if f.behavior == "following"]
    assert not hits, f"side-by-side companions were misreported as following: {hits}"


# --------------------------------------------------------------------- object
def test_unattended_object_escalates_and_names_the_last_owner():
    agent = ObjectAgent()
    policy = _policy(object={"tracked_classes": ["suitcase"], "unattended_seconds": 5,
                             "warning_seconds": 2, "separation_distance_px": 150})
    findings = []

    # phase 1: owner stands beside the bag
    for step in range(10):
        owner = _track(1, "person", [(500, 400)] * 20)
        bag = _track(2, "suitcase", [(520, 410)] * 20)
        findings += agent.run(_ctx([owner, bag], policy=policy, elapsed=step * 0.5))

    # phase 2: owner walks away, bag stays put
    for step in range(10, 60):
        walk = 520 + (step - 10) * 20
        owner = _track(1, "person", [(walk, 400)] * 20)
        bag = _track(2, "suitcase", [(520, 410)] * 20)
        findings += agent.run(_ctx([owner, bag], policy=policy, elapsed=step * 0.5))

    kinds = {f.behavior for f in findings}
    assert "unattended_object" in kinds, f"escalation never reached UNATTENDED: {kinds}"
    hit = next(f for f in findings if f.behavior == "unattended_object")
    assert hit.evidence["owner_track_id"] == 1, "last associated subject was not recorded"
    assert hit.evidence["contents_inference"] == "not performed - out of scope by design"
    assert hit.severity == "high"
    assert "contents are not inferred" in hit.explanation.lower()


def test_attended_bag_never_escalates():
    agent = ObjectAgent()
    policy = _policy(object={"tracked_classes": ["suitcase"], "unattended_seconds": 5,
                             "warning_seconds": 2, "separation_distance_px": 150})
    findings = []
    for step in range(60):
        owner = _track(1, "person", [(500, 400)] * 20)
        bag = _track(2, "suitcase", [(520, 410)] * 20)
        findings += agent.run(_ctx([owner, bag], policy=policy, elapsed=step * 0.5))
    assert not [f for f in findings if f.behavior in ("unattended_object", "object_separation")]


# ---------------------------------------------------------------------- crowd
def test_crowd_metrics_and_surge():
    agent = CrowdAgent()
    policy = _policy(crowd={"flow_window_seconds": 20, "surge_percent_threshold": 40})
    for step in range(12):
        few = [_track(i, "person", [(100 + i * 50 + s * 6, 300) for s in range(12)])
               for i in range(3)]
        agent.run(_ctx(few, policy=policy, elapsed=step * 1.1))
    findings = []
    for step in range(12, 30):
        many = [_track(i, "person", [(80 + i * 35 + s * 6, 280 + (i % 3) * 40) for s in range(12)])
                for i in range(18)]
        findings += agent.run(_ctx(many, policy=policy, elapsed=step * 1.1))

    metrics = agent.latest("CAM_T")
    assert metrics["count"] == 18
    assert metrics["density"] > 0
    assert metrics["density_band"] in ("low", "medium", "high", "critical")
    assert any(f.behavior == "crowd_surge" for f in findings), "occupancy tripling raised no surge"


# --------------------------------------------------------------------- threat
def test_threat_score_is_explainable_and_weighted():
    from app.agents.base import Finding

    assessor = ThreatAssessor()
    policy = load_policy()
    findings = [
        Finding("unattended_object", 0.9, "high", started_at=T0, ended_at=T0),
        Finding("restricted_zone_entry", 0.8, "high", started_at=T0, ended_at=T0),
        Finding("face_unavailable", 1.0, "info", started_at=T0, ended_at=T0),
    ]
    result = assessor.assess(findings, policy, now=T0)
    assert result.score > 0
    assert result.dominant_factor == "unattended_object"
    face = next(f for f in result.factors if f["behavior"] == "face_unavailable")
    assert face["weight"] == 0 and face["contribution"] == 0
    assert "because of" in result.explanation


def test_weak_signals_cannot_manufacture_a_critical_incident():
    from app.agents.base import Finding

    assessor = ThreatAssessor()
    policy = load_policy()
    weak = [
        Finding(b, 0.3, "low", started_at=T0, ended_at=T0)
        for b in ("loitering", "counter_flow", "running", "sudden_movement",
                  "abnormal_trajectory", "crowd_anomaly")
    ]
    result = assessor.assess(weak, policy, now=T0)
    assert result.band != "critical", f"six weak signals produced {result.band}"


def test_stale_signals_decay():
    from app.agents.base import Finding

    assessor = ThreatAssessor()
    policy = load_policy()
    fresh = assessor.assess(
        [Finding("unattended_object", 0.9, "high", started_at=T0, ended_at=T0)], policy, now=T0
    )
    stale = assessor.assess(
        [Finding("unattended_object", 0.9, "high", started_at=T0, ended_at=T0)],
        policy, now=T0 + timedelta(minutes=12),
    )
    assert stale.score < fresh.score


# ------------------------------------------------------------------ watchlist
def _fake_embedding(seed: int, dim: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_watchlist_expiry_and_scope_are_enforced():
    subj = EnrolledSubject(
        subject_id="W1", label="Test Subject", category="missing_person",
        priority="high", status="active",
        expires_at=T0 - timedelta(hours=1),
    )
    assert not subj.active_now(T0), "an expired subject is still matching"

    live = EnrolledSubject(
        subject_id="W2", label="Scoped", category="vip", priority="high",
        status="active", scope_camera_ids=["CAM_X"],
    )
    assert live.active_now(T0)
    assert live.in_scope("CAM_X", None)
    assert not live.in_scope("CAM_Y", None), "camera scope was not enforced"

    pending = EnrolledSubject(
        subject_id="W3", label="Unapproved", category="person_of_interest",
        priority="high", status="pending",
    )
    assert not pending.active_now(T0), "an unapproved subject is matching"


def test_watchlist_body_only_match_is_capped_and_flagged():
    from app.agents.watchlist import BODY_CONFIDENCE_CEILING

    agent = WatchlistAgent()
    emb = _fake_embedding(1)
    subj = EnrolledSubject(
        subject_id="W1", label="Subject", category="person_of_interest",
        priority="medium", status="active", body_embeddings=[emb],
    )
    best = agent._best_subject([subj], None, emb, 0.55, 0.60)
    assert best is not None
    _subject, modality, _sim, confidence = best
    assert modality == "body"
    assert confidence <= BODY_CONFIDENCE_CEILING, \
        "a clothing-only match was presented as strongly as a face match"


def test_watchlist_ignores_a_different_person():
    agent = WatchlistAgent()
    subj = EnrolledSubject(
        subject_id="W1", label="Subject", category="person_of_interest",
        priority="medium", status="active",
        face_embeddings=[_fake_embedding(1)], body_embeddings=[_fake_embedding(1)],
    )
    other = _fake_embedding(999)
    assert agent._best_subject([subj], other, other, 0.55, 0.60) is None


# ------------------------------------------------------------------ commander
def test_commander_never_proposes_autonomous_high_impact_action():
    from app.agents.base import Finding

    commander = IncidentCommander()
    assessor = ThreatAssessor()
    policy = load_policy()
    findings = [Finding("unattended_object", 0.95, "high", started_at=T0, ended_at=T0)]
    assessment = assessor.assess(findings, policy, now=T0)
    out = commander.compose(
        findings=findings, assessment=assessment,
        camera={"id": "CAM_T", "name": "Gate 4", "location": "Terminal A"},
        zone={"id": "Z1", "name": "Gate 4", "zone_type": "public"}, now=T0,
    )
    assert out["title"] == "Potentially unattended object"
    dispatch = [a for a in out["recommended_actions"] if a["key"] == "dispatch_team"]
    if dispatch:
        assert dispatch[0]["requires_human_approval"] is True
    assert all(a["status"] == "proposed" for a in out["recommended_actions"])
    assert out["timeline"], "commander produced no timeline"


def test_watchlist_runs_on_live_tracks_that_carry_an_embedding():
    """Regression: the live tracker attaches a numpy embedding to every person.

    `track.embedding or ...` raised on that array, so once anyone was enrolled
    the agent failed on every live track and person-of-interest matching never
    fired. Drive process() the way the pipeline does, not _best_subject().
    """
    agent = WatchlistAgent()
    emb = _fake_embedding(7, dim=512)
    agent.load_subjects([EnrolledSubject(
        subject_id="W7", label="Subject 7", category="person_of_interest",
        priority="high", status="active", body_embeddings=[emb],
    )])
    track = _track(1, "person", [(400 + i, 300) for i in range(12)])
    track.embedding = emb.copy()
    ctx = _ctx([track], policy=_policy(), elapsed=5.0, frame_index=12)
    ctx.frame = np.full((540, 960, 3), 110, dtype=np.uint8)

    findings = []
    for _ in range(WatchlistAgent.CHECK_EVERY_N_FRAMES):      # it samples every Nth frame
        findings += agent.run(ctx)
        assert agent.last_error is None, agent.last_error
    matches = [f for f in findings if f.behavior == "watchlist_match"]
    assert matches, "an enrolled subject standing in view was not matched"
    assert matches[0].evidence["subject_id"] == "W7"

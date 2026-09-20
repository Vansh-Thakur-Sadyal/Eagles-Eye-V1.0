"""Tracker + geometry validation against synthetic ground truth."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from app.vision.geometry import (
    angular_difference,
    crossing_direction,
    direction_changes,
    dominant_heading,
    point_in_polygon,
    trajectory_similarity,
)
from app.vision.tracker import ByteTracker, iou_matrix
from app.vision.types import Detection


def _box(cx, cy, w=40, h=100):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=np.float32)
    ious = iou_matrix(a, b)
    assert abs(ious[0, 0] - 1.0) < 1e-5
    assert abs(ious[0, 1] - (25 / 175)) < 1e-4
    assert ious[0, 2] == 0.0


def test_tracker_keeps_identity_for_two_crossing_people():
    """Two subjects walking in opposite directions must not swap IDs."""
    tracker = ByteTracker(min_hits=2, max_age=15)
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    ids_a, ids_b = [], []

    for f in range(60):
        ax, ay = 100 + f * 8, 300
        bx, by = 580 - f * 8, 300
        dets = [
            Detection(_box(ax, ay), 0.9, 0, "person"),
            Detection(_box(bx, by), 0.9, 0, "person"),
        ]
        tracks = tracker.update(dets, timestamp=t0 + timedelta(seconds=f / 12), fps=12)
        if f > 5:
            # left-moving vs right-moving, identified by velocity sign
            for tr in tracks:
                if tr.velocity[0] > 0:
                    ids_a.append(tr.track_id)
                elif tr.velocity[0] < 0:
                    ids_b.append(tr.track_id)

    assert len(set(ids_a)) == 1, f"rightward subject fragmented into {set(ids_a)}"
    assert len(set(ids_b)) == 1, f"leftward subject fragmented into {set(ids_b)}"
    assert set(ids_a) != set(ids_b)


def test_tracker_survives_short_occlusion():
    """A subject that disappears for 8 frames keeps the same track id."""
    tracker = ByteTracker(min_hits=2, max_age=30)
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    seen = []
    for f in range(50):
        occluded = 20 <= f < 28
        dets = [] if occluded else [Detection(_box(100 + f * 6, 300), 0.9, 0, "person")]
        tracks = tracker.update(dets, timestamp=t0 + timedelta(seconds=f / 12), fps=12)
        for tr in tracks:
            seen.append(tr.track_id)
    assert len(set(seen)) == 1, f"occlusion split the track: {set(seen)}"


def test_tracker_recovers_low_confidence_detections():
    """ByteTrack's second pass must keep a track alive through a weak stretch."""
    tracker = ByteTracker(min_hits=2, max_age=30, high_thresh=0.5, low_thresh=0.1)
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    ids = set()
    for f in range(40):
        score = 0.2 if 15 <= f < 25 else 0.9        # weak but non-zero mid-run
        tracker.update(
            [Detection(_box(120 + f * 5, 260), score, 0, "person")],
            timestamp=t0 + timedelta(seconds=f / 12),
            fps=12,
        )
        for tr in tracker.all_tracks():
            ids.add(tr.track_id)
    assert len(ids) == 1, f"low-confidence stretch spawned extra tracks: {ids}"


def test_tracker_does_not_associate_across_classes():
    tracker = ByteTracker(min_hits=1, max_age=10)
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    tracker.update([Detection(_box(200, 300), 0.9, 0, "person")], timestamp=t0, fps=12)
    tracker.update(
        [Detection(_box(202, 302), 0.9, 28, "suitcase")],
        timestamp=t0 + timedelta(seconds=1 / 12), fps=12,
    )
    classes = {t.class_name for t in tracker.all_tracks()}
    assert classes == {"person", "suitcase"}, classes


def test_point_in_polygon():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_polygon((5, 5), square)
    assert not point_in_polygon((15, 5), square)
    assert not point_in_polygon((5, -1), square)


def test_line_crossing_direction_is_signed():
    line = [[100, 0], [100, 400]]
    assert crossing_direction((90, 200), (110, 200), line) is not None
    assert crossing_direction((110, 200), (90, 200), line) is not None
    assert crossing_direction((90, 200), (95, 200), line) is None
    forward = crossing_direction((90, 200), (110, 200), line)
    backward = crossing_direction((110, 200), (90, 200), line)
    assert forward != backward


def test_trajectory_similarity_detects_a_shadowed_route():
    """A follower on a parallel offset route scores high; a crosser scores low."""
    leader = [(x, 100 + 40 * math.sin(x / 60)) for x in range(0, 600, 10)]
    follower = [(x, 160 + 40 * math.sin(x / 60)) for x in range(0, 600, 10)]
    crosser = [(300, y) for y in range(0, 600, 10)]

    same = trajectory_similarity(leader, follower)
    different = trajectory_similarity(leader, crosser)
    assert same > 0.85, same
    assert different < same


def test_direction_changes_found_on_a_zigzag():
    zig = [(x, 0) for x in range(0, 200, 10)] + [(200, y) for y in range(0, 200, 10)]
    assert len(direction_changes(zig, min_angle_deg=45)) >= 1


def test_angular_difference_wraps_around_zero():
    assert angular_difference(350, 10) == 20
    assert angular_difference(10, 350) == 20
    assert angular_difference(0, 180) == 180


def test_dominant_heading_uses_circular_mean():
    # a plain arithmetic mean of these would give 180, which is backwards
    assert dominant_heading([350, 10, 0]) < 20 or dominant_heading([350, 10, 0]) > 340

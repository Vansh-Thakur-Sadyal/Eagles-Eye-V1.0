"""Agent 6 - Behavioural Intelligence (spec S9).

Detects loitering, counter-flow, running, sudden movement, restricted-zone
entry, line crossings and abnormal trajectories from track geometry alone -
no identity is required or used.

Context matters and the spec says so explicitly: waiting at an airport is
normal.  Loitering is therefore scored on *confined dwell* (long presence
inside a small radius) rather than on presence alone, and counter-flow is
measured against the flow the rest of the crowd is actually taking right now,
not against a fixed compass bearing.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from ..vision.geometry import (
    angular_difference,
    crossing_direction,
    direction_changes,
    dominant_heading,
    normalised_polygon_to_pixels,
    path_length,
    point_in_polygon,
)
from ..vision.types import TrackState
from .base import Agent, FrameContext, Finding, severity_from_confidence


@dataclass
class _TrackMemory:
    anchor: Tuple[float, float]
    anchor_since: float                       # monotonic-ish seconds (frame time)
    zones: Dict[str, float] = field(default_factory=dict)   # zone_id -> entered_at
    reported: Dict[str, float] = field(default_factory=dict)
    speeds: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    headings: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    counter_flow_frames: int = 0
    last_point: Optional[Tuple[float, float]] = None


class BehaviorAgent(Agent):
    name = "behavior"
    spec_id = 6
    description = "Loitering, counter-flow, running, sudden movement, restricted entry"

    # Re-report the same ongoing behaviour at most this often (seconds).
    REPORT_COOLDOWN = 45.0

    def __init__(self) -> None:
        super().__init__()
        self._memory: Dict[str, Dict[int, _TrackMemory]] = defaultdict(dict)
        self._flow_history: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=240)
        )

    def reset(self, camera_id: Optional[str] = None) -> None:
        if camera_id:
            self._memory.pop(camera_id, None)
            self._flow_history.pop(camera_id, None)
        else:
            self._memory.clear()
            self._flow_history.clear()

    # ------------------------------------------------------------------ main
    def process(self, ctx: FrameContext) -> List[Finding]:
        policy = ctx.policy.get("behavior", {})
        cam = ctx.camera
        now = ctx.elapsed_seconds
        persons = ctx.persons()
        mem = self._memory[cam.camera_id]
        findings: List[Finding] = []

        # Current crowd flow, used as the counter-flow reference.
        live_headings = [t.heading_deg() for t in persons if t.speed > 12]
        crowd_heading = dominant_heading([h for h in live_headings if h is not None])
        if crowd_heading is not None:
            self._flow_history[cam.camera_id].append((now, crowd_heading))
        reference_heading = (
            cam.expected_flow_deg if cam.expected_flow_deg is not None else crowd_heading
        )

        min_age = int(policy.get("min_track_age_frames", 8))
        seen_ids = set()

        for track in persons:
            seen_ids.add(track.track_id)
            if track.age_frames < min_age:
                continue

            point = track.foot_point
            record = mem.get(track.track_id)
            if record is None:
                record = _TrackMemory(anchor=point, anchor_since=now)
                mem[track.track_id] = record

            record.speeds.append(track.speed)
            heading = track.heading_deg()
            if heading is not None:
                record.headings.append(heading)

            findings.extend(self._check_loitering(ctx, track, record, point, now, policy))
            findings.extend(self._check_running(ctx, track, record, now, policy))
            findings.extend(self._check_sudden_movement(ctx, track, record, now, policy))
            findings.extend(
                self._check_counter_flow(ctx, track, record, now, policy, reference_heading, len(persons))
            )
            findings.extend(self._check_zones(ctx, track, record, point, now))
            findings.extend(self._check_lines(ctx, track, record, point))
            findings.extend(self._check_abnormal_trajectory(ctx, track, record, now))

            record.last_point = point

        # forget tracks that have left the scene
        for gone in [tid for tid in mem if tid not in seen_ids]:
            mem.pop(gone, None)

        return findings

    # ------------------------------------------------------------ behaviours
    def _check_loitering(self, ctx, track, record, point, now, policy) -> List[Finding]:
        radius = float(policy.get("loitering_radius_px", 90))
        threshold = float(policy.get("loitering_seconds", 120))

        drift = math.dist(point, record.anchor)
        if drift > radius:
            # subject genuinely moved on - restart the dwell clock
            record.anchor = point
            record.anchor_since = now
            return []

        dwell = now - record.anchor_since
        if dwell < threshold or not self._cooldown_ok(record, "loitering", now):
            return []

        # Confidence grows with dwell but is tempered when the subject is
        # drifting near the edge of the radius (browsing, not loitering).
        over = (dwell - threshold) / max(1.0, threshold)
        compactness = 1.0 - min(1.0, drift / max(1.0, radius))
        confidence = float(np.clip(0.5 + 0.3 * min(2.0, over) + 0.2 * compactness, 0.0, 0.95))
        record.reported["loitering"] = now

        return [
            Finding(
                behavior="loitering",
                confidence=round(confidence, 3),
                severity=severity_from_confidence(confidence, "medium"),
                track_id=str(track.track_id),
                zone_id=track.zone_id or ctx.camera.zone_id,
                duration_seconds=round(dwell, 1),
                explanation=(
                    f"Subject has remained within {int(drift)}px of the same position for "
                    f"{int(dwell)}s (threshold {int(threshold)}s)."
                ),
                evidence={
                    "dwell_seconds": round(dwell, 1),
                    "drift_px": round(drift, 1),
                    "radius_px": radius,
                    "threshold_seconds": threshold,
                    "anchor": [round(v, 1) for v in record.anchor],
                },
            )
        ]

    def _check_running(self, ctx, track, record, now, policy) -> List[Finding]:
        limit = float(policy.get("running_speed_px_s", 210))
        if len(record.speeds) < 5:
            return []
        recent = float(np.mean(list(record.speeds)[-5:]))
        if recent < limit or not self._cooldown_ok(record, "running", now):
            return []
        record.reported["running"] = now
        confidence = float(np.clip(0.5 + (recent - limit) / max(1.0, limit), 0.0, 0.95))
        return [
            Finding(
                behavior="running",
                confidence=round(confidence, 3),
                severity=severity_from_confidence(confidence, "medium"),
                track_id=str(track.track_id),
                zone_id=track.zone_id or ctx.camera.zone_id,
                explanation=f"Sustained speed {int(recent)} px/s exceeds the running threshold "
                            f"of {int(limit)} px/s.",
                evidence={"speed_px_s": round(recent, 1), "threshold_px_s": limit},
            )
        ]

    def _check_sudden_movement(self, ctx, track, record, now, policy) -> List[Finding]:
        accel_limit = float(policy.get("sudden_movement_accel", 160))
        if len(record.speeds) < 8:
            return []
        speeds = list(record.speeds)
        baseline = float(np.mean(speeds[-8:-3]))
        current = float(np.mean(speeds[-3:]))
        delta = current - baseline
        if delta < accel_limit or not self._cooldown_ok(record, "sudden_movement", now):
            return []
        record.reported["sudden_movement"] = now
        confidence = float(np.clip(0.45 + delta / (accel_limit * 3), 0.0, 0.9))
        return [
            Finding(
                behavior="sudden_movement",
                confidence=round(confidence, 3),
                severity=severity_from_confidence(confidence, "medium"),
                track_id=str(track.track_id),
                zone_id=track.zone_id or ctx.camera.zone_id,
                explanation=f"Speed increased from {int(baseline)} to {int(current)} px/s "
                            f"within roughly half a second.",
                evidence={
                    "baseline_px_s": round(baseline, 1),
                    "current_px_s": round(current, 1),
                    "delta_px_s": round(delta, 1),
                },
            )
        ]

    def _check_counter_flow(self, ctx, track, record, now, policy, reference, crowd_size) -> List[Finding]:
        """Counter-flow only means something when there IS a flow to oppose."""
        if reference is None or crowd_size < 3:
            record.counter_flow_frames = 0
            return []
        heading = track.heading_deg()
        if heading is None or track.speed < 15:
            return []

        angle_limit = float(policy.get("counter_flow_angle_deg", 120))
        min_frames = int(policy.get("counter_flow_min_frames", 15))
        deviation = angular_difference(heading, reference)

        if deviation < angle_limit:
            record.counter_flow_frames = max(0, record.counter_flow_frames - 1)
            return []

        record.counter_flow_frames += 1
        if record.counter_flow_frames < min_frames or not self._cooldown_ok(record, "counter_flow", now):
            return []

        record.reported["counter_flow"] = now
        confidence = float(
            np.clip(0.45 + (deviation - angle_limit) / 180.0 + record.counter_flow_frames / 200.0, 0, 0.92)
        )
        return [
            Finding(
                behavior="counter_flow",
                confidence=round(confidence, 3),
                severity=severity_from_confidence(confidence, "medium"),
                track_id=str(track.track_id),
                zone_id=track.zone_id or ctx.camera.zone_id,
                explanation=(
                    f"Subject heading {int(heading)}deg opposes the prevailing flow of "
                    f"{int(reference)}deg by {int(deviation)}deg, sustained for "
                    f"{record.counter_flow_frames} frames."
                ),
                evidence={
                    "subject_heading_deg": round(heading, 1),
                    "reference_heading_deg": round(reference, 1),
                    "deviation_deg": round(deviation, 1),
                    "sustained_frames": record.counter_flow_frames,
                    "crowd_size": crowd_size,
                },
            )
        ]

    def _check_zones(self, ctx, track, record, point, now) -> List[Finding]:
        findings: List[Finding] = []
        cam = ctx.camera
        inside_now = set()

        for zone in cam.zones:
            poly = zone.get("polygon") or []
            if len(poly) < 3:
                continue
            pixels = normalised_polygon_to_pixels(poly, cam.width, cam.height)
            if not point_in_polygon(point, pixels):
                continue
            zid = zone.get("id")
            inside_now.add(zid)
            if zid not in track.zones_visited:
                track.zones_visited.append(zid)
            track.zone_id = zid

            if zid in record.zones:
                continue                                  # already counted the entry
            record.zones[zid] = now

            if zone.get("zone_type") in ("restricted", "secure"):
                key = f"restricted:{zid}"
                if not self._cooldown_ok(record, key, now):
                    continue
                record.reported[key] = now
                findings.append(
                    Finding(
                        behavior="restricted_entry",
                        confidence=0.88,
                        severity="high",
                        track_id=str(track.track_id),
                        zone_id=zid,
                        explanation=(
                            f"Subject entered {zone.get('name', zid)}, classified "
                            f"'{zone.get('zone_type')}'. Authorisation not verified by video alone."
                        ),
                        evidence={
                            "zone_id": zid,
                            "zone_name": zone.get("name"),
                            "zone_type": zone.get("zone_type"),
                            "entry_point": [round(v, 1) for v in point],
                            "authorized_roles": zone.get("authorized_roles", []),
                        },
                    )
                )

        for zid in [z for z in record.zones if z not in inside_now]:
            record.zones.pop(zid, None)
        return findings

    def _check_lines(self, ctx, track, record, point) -> List[Finding]:
        if record.last_point is None or not ctx.camera.line_crossings:
            return []
        findings: List[Finding] = []
        cam = ctx.camera
        for rule in cam.line_crossings:
            line = rule.get("line") or []
            if len(line) < 2:
                continue
            pixels = normalised_polygon_to_pixels(line, cam.width, cam.height)
            direction = crossing_direction(record.last_point, point, pixels)
            if direction is None:
                continue
            forbidden = rule.get("forbidden_direction")
            if forbidden and direction != forbidden:
                continue
            findings.append(
                Finding(
                    behavior=rule.get("behavior", "line_crossing"),
                    confidence=0.9,
                    severity=rule.get("severity", "medium"),
                    track_id=str(track.track_id),
                    zone_id=rule.get("zone_id") or cam.zone_id,
                    explanation=f"Subject crossed '{rule.get('name', 'tripwire')}' "
                                f"in direction {direction}.",
                    evidence={"rule": rule.get("name"), "direction": direction},
                )
            )
        return findings

    def _check_abnormal_trajectory(self, ctx, track, record, now) -> List[Finding]:
        """Erratic path: many direction reversals over a short ground distance."""
        if len(track.history) < 40 or not self._cooldown_ok(record, "abnormal_trajectory", now):
            return []
        points = [(h["x"], h["y"]) for h in track.history[-90:]]
        turns = direction_changes(points, min_angle_deg=55, window=3)
        distance = path_length(points)
        displacement = math.dist(points[0], points[-1])
        if distance < 120:
            return []
        straightness = displacement / max(1.0, distance)
        if len(turns) < 4 or straightness > 0.35:
            return []
        record.reported["abnormal_trajectory"] = now
        confidence = float(np.clip(0.4 + len(turns) / 20.0 + (0.35 - straightness), 0, 0.85))
        return [
            Finding(
                behavior="abnormal_trajectory",
                confidence=round(confidence, 3),
                severity=severity_from_confidence(confidence, "medium"),
                track_id=str(track.track_id),
                zone_id=track.zone_id or ctx.camera.zone_id,
                explanation=(
                    f"Path shows {len(turns)} sharp direction changes with a straightness "
                    f"ratio of {straightness:.2f} - movement is not goal-directed."
                ),
                evidence={
                    "direction_changes": len(turns),
                    "straightness": round(straightness, 3),
                    "path_length_px": round(distance, 1),
                },
            )
        ]

    # ----------------------------------------------------------------- utils
    def _cooldown_ok(self, record: _TrackMemory, key: str, now: float) -> bool:
        last = record.reported.get(key)
        return last is None or (now - last) >= self.REPORT_COOLDOWN

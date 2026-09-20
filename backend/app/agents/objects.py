"""Agent 9 - Belongings / Object Intelligence (spec S12, S13).

Associates bags and luggage with the person who brought them, watches for
separation, and escalates to UNATTENDED once the object has been stationary
and ownerless past the configured threshold.

The spec's limitation is enforced in code, not just in prose: this agent
reports "potentially unattended object" and the last associated subject's
trajectory.  It never infers contents, and there is no code path that can
label an object as containing anything.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..vision.types import TrackState
from .base import Agent, FrameContext, Finding

# Contents inference is out of scope by design (spec S12).
FORBIDDEN_INFERENCE = ("explosive", "weapon_contents", "bomb")


@dataclass
class _ObjectState:
    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    position: Tuple[float, float]
    anchor: Tuple[float, float]
    stationary_since: Optional[float] = None
    owner_track_id: Optional[int] = None
    owner_confidence: float = 0.0
    owner_last_seen: Optional[float] = None
    owner_last_position: Optional[Tuple[float, float]] = None
    owner_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    separation_at: Optional[float] = None
    status: str = "attended"
    reported_status: Optional[str] = None
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    owner_established: bool = False
    owner_candidates: Dict[int, int] = field(default_factory=dict)
    bystander_passes: int = 0
    nearest_bystander_px: Optional[float] = None
    _bystander_present: bool = False


class ObjectAgent(Agent):
    name = "object"
    spec_id = 9
    description = "Object-person association and unattended-belonging detection"

    # Consecutive in-range frames before a person is treated as the owner.
    OWNERSHIP_FRAMES = 10

    def __init__(self) -> None:
        super().__init__()
        self._objects: Dict[str, Dict[int, _ObjectState]] = defaultdict(dict)

    def reset(self, camera_id: Optional[str] = None) -> None:
        if camera_id:
            self._objects.pop(camera_id, None)
        else:
            self._objects.clear()

    def process(self, ctx: FrameContext) -> List[Finding]:
        cfg = ctx.policy.get("object", {})
        tracked_classes = set(cfg.get("tracked_classes", ["backpack", "handbag", "suitcase"]))
        separation_distance = float(cfg.get("separation_distance_px", 180))
        stationary_tolerance = float(cfg.get("stationary_tolerance_px", 22))
        unattended_seconds = float(cfg.get("unattended_seconds", 60))
        warning_seconds = float(cfg.get("warning_seconds", 30))

        now = ctx.elapsed_seconds
        persons = ctx.persons()
        objects = [t for t in ctx.tracks if t.class_name in tracked_classes]
        store = self._objects[ctx.camera.camera_id]
        findings: List[Finding] = []
        seen = set()

        for obj in objects:
            seen.add(obj.track_id)
            pos = obj.foot_point
            state = store.get(obj.track_id)
            if state is None:
                state = _ObjectState(
                    track_id=obj.track_id,
                    class_name=obj.class_name,
                    first_seen=now,
                    last_seen=now,
                    position=pos,
                    anchor=pos,
                )
                store[obj.track_id] = state

            state.last_seen = now
            state.position = pos
            state.bbox = obj.bbox

            # --- stationarity -------------------------------------------------
            if math.dist(pos, state.anchor) > stationary_tolerance:
                state.anchor = pos
                state.stationary_since = None
            elif state.stationary_since is None:
                state.stationary_since = now

            # --- ownership ----------------------------------------------------
            # Attendance means *the established owner* is present. A stranger
            # walking past a bag does not attend it, and must not reset the
            # abandonment clock - otherwise in a busy concourse, which is the
            # primary deployment, an unattended bag would never be flagged.
            owner_track = self._track_by_id(persons, state.owner_track_id)
            owner_distance = (
                math.dist(pos, owner_track.foot_point) if owner_track is not None else None
            )
            owner_in_range = owner_distance is not None and owner_distance <= separation_distance

            nearest, nearest_distance, nearest_confidence = self._nearest_person(pos, persons)
            bystander_in_range = (
                nearest is not None
                and nearest_distance <= separation_distance
                and nearest.track_id != state.owner_track_id
            )

            # Count distinct passes rather than frames, so the evidence reads
            # "3 people passed nearby", not "412 frames of proximity".
            if bystander_in_range and not state._bystander_present:
                state.bystander_passes += 1
            state._bystander_present = bool(bystander_in_range)
            if bystander_in_range:
                state.nearest_bystander_px = (
                    nearest_distance if state.nearest_bystander_px is None
                    else min(state.nearest_bystander_px, nearest_distance)
                )

            if not state.owner_established:
                # Association phase. Ownership needs *sustained* proximity: the
                # person who carried the object in stays closest for a while,
                # whereas a passer-by is nearest for only a moment. Taking the
                # first frame's nearest person would routinely credit a stranger,
                # and the whole forensic value of this agent is being able to say
                # where the *actual* associated subject went.
                if nearest is not None and nearest_distance <= separation_distance:
                    state.owner_candidates[nearest.track_id] = (
                        state.owner_candidates.get(nearest.track_id, 0) + 1
                    )
                    best_id = max(state.owner_candidates, key=state.owner_candidates.get)
                    if state.owner_candidates[best_id] >= self.OWNERSHIP_FRAMES:
                        state.owner_track_id = best_id
                        state.owner_confidence = nearest_confidence
                        state.owner_established = True
                    state.owner_last_seen = now
                    state.owner_last_position = nearest.foot_point
                    state.owner_trajectory = list(nearest.history[-150:])
                    state.separation_at = None
                    new_status = "attended"
                else:
                    new_status = "unowned"
            elif owner_in_range:
                state.owner_confidence = max(state.owner_confidence, nearest_confidence)
                state.owner_last_seen = now
                state.owner_last_position = owner_track.foot_point
                state.owner_trajectory = list(owner_track.history[-150:])
                state.separation_at = None
                new_status = "attended"
            else:
                if state.separation_at is None:
                    state.separation_at = now
                new_status = "separated"

            # --- escalation ---------------------------------------------------
            # `is not None` matters: these are elapsed-second timestamps and a
            # legitimate value of 0.0 is falsy.
            stationary_for = (now - state.stationary_since) if state.stationary_since is not None else 0.0
            separated_for = (now - state.separation_at) if state.separation_at is not None else 0.0
            ownerless_for = (
                min(stationary_for, separated_for) if state.separation_at is not None else 0.0
            )

            # Escalate only when the object is BOTH stationary and ownerless:
            # a bag moving away with someone is not an abandonment.
            if state.separation_at is not None and state.stationary_since is not None:
                if ownerless_for >= unattended_seconds:
                    new_status = "unattended"
                elif ownerless_for >= warning_seconds:
                    new_status = "warning"

            state.status = new_status

            if new_status in ("warning", "unattended") and state.reported_status != new_status:
                state.reported_status = new_status
                findings.append(self._build_finding(ctx, state, stationary_for, ownerless_for, now))

            if new_status == "attended" and owner_in_range and                     state.reported_status in ("warning", "unattended"):
                state.reported_status = None
                findings.append(
                    Finding(
                        behavior="object_reclaimed",
                        confidence=0.8,
                        severity="info",
                        object_id=str(state.track_id),
                        track_id=str(state.owner_track_id) if state.owner_track_id else None,
                        zone_id=ctx.camera.zone_id,
                        explanation=f"{state.class_name} at camera {ctx.camera.name} was "
                                    f"reclaimed by an associated subject.",
                        evidence={"object_class": state.class_name},
                        dedupe_key=f"reclaim:{ctx.camera.camera_id}:{state.track_id}",
                    )
                )

        for gone in [oid for oid in store if oid not in seen]:
            st = store[gone]
            if (now - st.last_seen) > 120:
                store.pop(gone, None)

        return findings

    # ------------------------------------------------------------------ bits
    @staticmethod
    def _track_by_id(persons: List[TrackState], track_id: Optional[int]) -> Optional[TrackState]:
        if track_id is None:
            return None
        return next((p for p in persons if p.track_id == track_id), None)

    @staticmethod
    def _nearest_person(
        position: Tuple[float, float], persons: List[TrackState]
    ) -> Tuple[Optional[TrackState], float, float]:
        if not persons:
            return None, float("inf"), 0.0
        best = min(persons, key=lambda p: math.dist(position, p.foot_point))
        distance = math.dist(position, best.foot_point)
        # Confidence decays with distance; 0 px -> 1.0, 300 px -> ~0.1
        confidence = float(np.clip(math.exp(-distance / 130.0), 0.0, 1.0))
        return best, distance, confidence

    def _build_finding(
        self, ctx: FrameContext, state: _ObjectState, stationary_for: float,
        ownerless_for: float, now: float
    ) -> Finding:
        unattended = state.status == "unattended"
        last_seen_ago = (now - state.owner_last_seen) if state.owner_last_seen is not None else None

        explanation = (
            f"A {state.class_name} has been stationary for {int(stationary_for)}s with no "
            f"associated subject within range for {int(ownerless_for)}s."
        )
        if state.owner_track_id is not None:
            explanation += (
                f" Last associated subject was track {state.owner_track_id}, "
                f"observed {int(last_seen_ago or 0)}s ago."
            )
        if state.bystander_passes:
            explanation += (
                f" {state.bystander_passes} other subject(s) passed within range during this "
                "period; none of them is the associated subject."
            )
        explanation += " Contents are not inferred and cannot be determined from video."

        return Finding(
            behavior="unattended_object" if unattended else "object_separation",
            confidence=0.9 if unattended else 0.7,
            severity="high" if unattended else "medium",
            object_id=str(state.track_id),
            track_id=str(state.owner_track_id) if state.owner_track_id else None,
            zone_id=ctx.camera.zone_id,
            duration_seconds=round(ownerless_for, 1),
            explanation=explanation,
            evidence={
                "object_class": state.class_name,
                "stationary_seconds": round(stationary_for, 1),
                "ownerless_seconds": round(ownerless_for, 1),
                "owner_track_id": state.owner_track_id,
                "owner_confidence": round(state.owner_confidence, 3),
                "owner_established_by": "sustained proximity",
                "owner_last_seen_seconds_ago": round(last_seen_ago, 1) if last_seen_ago else None,
                "owner_last_position": (
                    [round(v, 1) for v in state.owner_last_position]
                    if state.owner_last_position else None
                ),
                "owner_trajectory_points": len(state.owner_trajectory),
                "bystander_passes": state.bystander_passes,
                "nearest_bystander_px": (
                    round(state.nearest_bystander_px, 1)
                    if state.nearest_bystander_px is not None else None
                ),
                "bbox": [round(v, 1) for v in state.bbox],
                "contents_inference": "not performed - out of scope by design",
            },
            dedupe_key=f"unattended:{ctx.camera.camera_id}:{state.track_id}",
        )

    # -------------------------------------------------------------- readers
    def snapshot(self, camera_id: str) -> List[Dict[str, Any]]:
        """Feed the Object Intelligence screen."""
        return [
            {
                "object_track_id": st.track_id,
                "class_name": st.class_name,
                "status": st.status,
                "owner_track_id": st.owner_track_id,
                "owner_confidence": round(st.owner_confidence, 3),
                "stationary": st.stationary_since is not None,
                "bystander_passes": st.bystander_passes,
                "bbox": [round(v, 1) for v in st.bbox],
            }
            for st in self._objects.get(camera_id, {}).values()
        ]

    def owner_trajectory(self, camera_id: str, object_track_id: int) -> List[Dict[str, Any]]:
        """The last associated subject's path - spec S13's core forensic answer."""
        st = self._objects.get(camera_id, {}).get(object_track_id)
        return list(st.owner_trajectory) if st else []

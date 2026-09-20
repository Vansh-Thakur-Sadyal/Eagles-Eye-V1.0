"""Agent 7 - Following / Relationship Intelligence (spec S10).

Scores *persistent following patterns* between pairs of tracks.  The spec is
emphatic on the framing and this module honours it: the output is
"possible persistent following pattern - priority for human review", never an
accusation, and never the word "stalker".

Six independent signals are combined, so that ordinary co-travel does not
trigger.  Two friends walking together sit at a near-constant small distance
with no lag; a follower holds a *trailing* offset, mirrors direction changes
after a delay, and stops when the leader stops.  The lag correlation and the
mirrored-turn count are what separate the two, and a pair must clear a minimum
duration before anything is emitted at all.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..vision.geometry import direction_changes, trajectory_similarity
from ..vision.types import TrackState
from .base import Agent, FrameContext, Finding


@dataclass
class _PairState:
    first_seen: float
    last_seen: float
    samples: int = 0
    distances: List[float] = field(default_factory=list)
    behind_frames: int = 0
    trailing_samples: List[float] = field(default_factory=list)
    cross_samples: List[float] = field(default_factory=list)
    leader_votes: Dict[int, int] = field(default_factory=dict)
    stop_matches: int = 0
    stop_opportunities: int = 0
    mirrored_turns: int = 0
    leader_turns: int = 0
    reported_at: Optional[float] = None
    peak_confidence: float = 0.0


class RelationshipAgent(Agent):
    name = "relationship"
    spec_id = 7
    description = "Persistent following patterns between subject pairs"

    REPORT_COOLDOWN = 90.0
    MAX_PAIRS = 400

    def __init__(self) -> None:
        super().__init__()
        self._pairs: Dict[str, Dict[Tuple[int, int], _PairState]] = defaultdict(dict)

    def reset(self, camera_id: Optional[str] = None) -> None:
        if camera_id:
            self._pairs.pop(camera_id, None)
        else:
            self._pairs.clear()

    def process(self, ctx: FrameContext) -> List[Finding]:
        cfg = ctx.policy.get("following", {})
        max_distance = float(cfg.get("max_pair_distance_px", 260))
        min_distance = float(cfg.get("min_distance_px", 30))
        min_duration = float(cfg.get("min_duration_seconds", 25))
        sim_threshold = float(cfg.get("trajectory_similarity", 0.72))
        min_turn_matches = int(cfg.get("direction_change_matches", 2))
        floor = float(cfg.get("confidence_floor", 0.55))

        persons = [t for t in ctx.persons() if len(t.history) >= 10]
        if len(persons) < 2:
            return []

        now = ctx.elapsed_seconds
        pairs = self._pairs[ctx.camera.camera_id]
        findings: List[Finding] = []
        active_keys = set()

        for i, a in enumerate(persons):
            for b in persons[i + 1 :]:
                distance = math.dist(a.foot_point, b.foot_point)
                if distance > max_distance or distance < min_distance:
                    continue

                key = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
                active_keys.add(key)
                state = pairs.get(key)
                if state is None:
                    if len(pairs) >= self.MAX_PAIRS:
                        continue
                    state = _PairState(first_seen=now, last_seen=now)
                    pairs[key] = state

                state.last_seen = now
                state.samples += 1
                state.distances.append(distance)
                if len(state.distances) > 300:
                    state.distances.pop(0)

                leader, follower = self._orient(a, b)
                if leader is None or follower is None:
                    continue

                geometry = self._trailing_offset(leader, follower)
                if geometry is not None:
                    along, cross = geometry
                    if along > 0.5:
                        state.behind_frames += 1
                    state.trailing_samples.append(along)
                    state.cross_samples.append(cross)
                    if len(state.trailing_samples) > 300:
                        state.trailing_samples.pop(0)
                        state.cross_samples.pop(0)
                    # Who leads should be stable for a follower and should flip
                    # roughly 50/50 for two people walking abreast.
                    state.leader_votes[leader.track_id] = (
                        state.leader_votes.get(leader.track_id, 0) + 1
                    )

                # Does the follower stop when the leader stops?
                if leader.speed < 12:
                    state.stop_opportunities += 1
                    if follower.speed < 20:
                        state.stop_matches += 1

                duration = now - state.first_seen
                if duration < min_duration:
                    continue
                if state.reported_at is not None and (now - state.reported_at) < self.REPORT_COOLDOWN:
                    continue

                assessment = self._score(
                    leader, follower, state,
                    sim_threshold=sim_threshold,
                    min_turn_matches=min_turn_matches,
                    max_distance=max_distance,
                )
                confidence = assessment["confidence"]
                state.peak_confidence = max(state.peak_confidence, confidence)
                if confidence < floor:
                    continue

                state.reported_at = now
                findings.append(
                    Finding(
                        behavior="following",
                        confidence=round(confidence, 3),
                        severity="high" if confidence >= 0.78 else "medium",
                        track_id=str(follower.track_id),
                        secondary_track_id=str(leader.track_id),
                        zone_id=follower.zone_id or ctx.camera.zone_id,
                        duration_seconds=round(duration, 1),
                        explanation=(
                            f"Possible persistent following pattern: track {follower.track_id} has "
                            f"trailed track {leader.track_id} for {int(duration)}s at a mean "
                            f"separation of {int(assessment['mean_distance'])}px, mirroring "
                            f"{assessment['mirrored_turns']} direction change(s). "
                            "Flagged for human review - this is a correlation, not an identification."
                        ),
                        evidence=assessment,
                        dedupe_key=f"following:{ctx.camera.camera_id}:{key[0]}:{key[1]}",
                    )
                )

        for stale in [k for k, s in pairs.items() if k not in active_keys and (now - s.last_seen) > 30]:
            pairs.pop(stale, None)

        return findings

    # ------------------------------------------------------------- scoring
    @staticmethod
    def _orient(a: TrackState, b: TrackState) -> Tuple[Optional[TrackState], Optional[TrackState]]:
        """Whoever is ahead along the shared direction of travel is the leader."""
        direction = RelationshipAgent._smoothed_heading(a)
        if direction is None:
            direction = RelationshipAgent._smoothed_heading(b)
        if direction is None:
            return None, None
        pa = np.array(a.foot_point)
        pb = np.array(b.foot_point)
        return (a, b) if float(np.dot(pa - pb, direction)) > 0 else (b, a)

    @staticmethod
    def _smoothed_heading(track: TrackState, window: int = 6) -> Optional[np.ndarray]:
        """Unit direction of travel over several frames.

        The instantaneous velocity of a real detection is far too noisy for this
        decomposition: detector jitter of a few pixels swamps a walking pace of
        ~3 px/frame, and the resulting heading swings wildly. Averaging the
        displacement over a short window recovers a stable direction.
        """
        history = track.history
        if len(history) < window + 1:
            vx, vy = track.velocity
            norm = math.hypot(vx, vy)
            return np.array([vx / norm, vy / norm]) if norm > 1e-6 else None
        recent, past = history[-1], history[-1 - window]
        dx, dy = recent["x"] - past["x"], recent["y"] - past["y"]
        norm = math.hypot(dx, dy)
        return np.array([dx / norm, dy / norm]) if norm > 1e-6 else None

    @staticmethod
    def _trailing_offset(
        leader: TrackState, follower: TrackState
    ) -> Optional[Tuple[float, float]]:
        """How much of the pair's separation is *behind* rather than *beside*.

        Returns (along, cross). `along` is +1 directly behind, 0 abreast, -1
        ahead; `cross` is how far to the side they sit, 0..1.

        Note that `along` alone is not enough: the leader/follower assignment
        already puts whoever is further along the route in front, so `along` is
        biased positive by construction. `cross` is the honest discriminator -
        a companion walking beside you is dominated by it.

        A binary "is behind" test cannot separate a companion from a follower,
        because someone walking beside you is neither clearly behind nor ahead
        and noise decides the answer. The geometry that actually distinguishes
        them is this decomposition: companions sit near 0, a follower sits high.
        """
        forward = RelationshipAgent._smoothed_heading(leader)
        if forward is None:
            return None
        offset = np.array(follower.foot_point) - np.array(leader.foot_point)
        norm = float(np.linalg.norm(offset))
        if norm < 1e-6:
            return None
        unit = offset / norm
        # along: how much of the separation is behind the leader (+) or ahead (-)
        # cross: how much is *beside* them. Companions are dominated by cross.
        along = float(-np.dot(unit, forward))
        # 2-D cross product by hand: np.cross on 2-vectors is deprecated in
        # NumPy 2.0 and removed later.
        cross = float(abs(unit[0] * forward[1] - unit[1] * forward[0]))
        return along, cross

    def _score(
        self,
        leader: TrackState,
        follower: TrackState,
        state: _PairState,
        *,
        sim_threshold: float,
        min_turn_matches: int,
        max_distance: float,
    ) -> Dict[str, Any]:
        lead_pts = [(h["x"], h["y"]) for h in leader.history[-120:]]
        foll_pts = [(h["x"], h["y"]) for h in follower.history[-120:]]

        # 1. route overlap
        similarity = trajectory_similarity(lead_pts, foll_pts)

        # 2. distance stability - a follower holds range, a passer-by does not
        distances = np.array(state.distances[-120:], dtype=np.float64)
        mean_distance = float(distances.mean()) if len(distances) else max_distance
        distance_cv = float(distances.std() / mean_distance) if mean_distance > 0 and len(distances) > 3 else 1.0
        stability = float(np.clip(1.0 - distance_cv, 0.0, 1.0))

        # 3. trailing persistence - is the separation *behind* or *beside*?
        behind_ratio = state.behind_frames / max(1, state.samples)
        mean_trailing = (
            float(np.mean(state.trailing_samples)) if state.trailing_samples else 0.0
        )
        mean_cross = float(np.mean(state.cross_samples)) if state.cross_samples else 1.0
        # Penalise side-by-side geometry directly.
        trailing_score = float(np.clip(mean_trailing * (1.0 - mean_cross), 0.0, 1.0))

        # 3b. leader consistency - a follower trails the same person throughout;
        # two companions swap who is nominally in front as they walk.
        total_votes = sum(state.leader_votes.values())
        leader_consistency = (
            max(state.leader_votes.values()) / total_votes if total_votes else 0.0
        )

        # 4. stop synchronisation
        stop_sync = state.stop_matches / state.stop_opportunities if state.stop_opportunities >= 3 else 0.0

        # 5. mirrored direction changes with a lag
        mirrored, leader_turns = self._mirrored_turns(lead_pts, foll_pts)
        state.mirrored_turns = mirrored
        state.leader_turns = leader_turns
        turn_score = min(1.0, mirrored / max(1, min_turn_matches)) if leader_turns else 0.0

        # 6. lag correlation - the follower's path echoes the leader's earlier path
        lag_corr = self._lag_correlation(lead_pts, foll_pts)

        weights = {
            "route_overlap": 0.24,
            "distance_stability": 0.16,
            "trailing_persistence": 0.18,
            "stop_synchronisation": 0.14,
            "mirrored_turns": 0.16,
            "lag_correlation": 0.12,
        }
        components = {
            "route_overlap": similarity if similarity >= sim_threshold else similarity * 0.5,
            "distance_stability": stability,
            "trailing_persistence": trailing_score,
            "stop_synchronisation": stop_sync,
            "mirrored_turns": turn_score,
            "lag_correlation": lag_corr,
        }
        confidence = float(sum(components[k] * w for k, w in weights.items()))

        # An unstable leader means the pair is walking together, not one
        # trailing the other, whatever the other signals say.
        if leader_consistency < 0.75:
            confidence *= 0.35

        # Guard against the "two friends" false positive. Co-travel shows a
        # trailing offset too (whoever is a step ahead is "in front"), so the
        # distinguishing evidence is temporal: does the follower occupy where
        # the leader was, and do they repeat the leader's turns? With neither,
        # this is two people going the same way, which is not a signal.
        # Walking beside someone is co-travel, not following. Require either a
        # genuine trailing offset or repeated mirrored turns before this scores.
        temporal_evidence = max(lag_corr, turn_score)
        if trailing_score < 0.35 and turn_score < 0.5:
            confidence *= 0.30
        elif temporal_evidence < 0.25:
            confidence *= 0.45

        return {
            "confidence": float(np.clip(confidence, 0.0, 0.97)),
            "components": {k: round(v, 3) for k, v in components.items()},
            "weights": weights,
            "mean_distance": round(mean_distance, 1),
            "distance_stability": round(stability, 3),
            "behind_ratio": round(behind_ratio, 3),
            "mean_trailing_share": round(mean_trailing, 3),
            "mean_cross_track_share": round(mean_cross, 3),
            "leader_consistency": round(leader_consistency, 3),
            "stop_synchronisation": round(stop_sync, 3),
            "mirrored_turns": mirrored,
            "leader_turns": leader_turns,
            "duration_seconds": round(state.last_seen - state.first_seen, 1),
            "samples": state.samples,
            "note": "Correlation-based association for human review, not an identification.",
        }

    @staticmethod
    def _mirrored_turns(lead_pts, foll_pts, lag_tolerance: int = 18) -> Tuple[int, int]:
        """Count leader turns the follower repeats a short time later."""
        lead_turns = direction_changes(lead_pts, min_angle_deg=40, window=3)
        foll_turns = direction_changes(foll_pts, min_angle_deg=40, window=3)
        if not lead_turns or not foll_turns:
            return 0, len(lead_turns)
        matched = 0
        used = set()
        for lt in lead_turns:
            for j, ft in enumerate(foll_turns):
                if j in used:
                    continue
                if 0 <= ft - lt <= lag_tolerance:
                    matched += 1
                    used.add(j)
                    break
        return matched, len(lead_turns)

    @staticmethod
    def _lag_correlation(lead_pts, foll_pts, max_lag: int = 30) -> float:
        """How much better the follower matches the leader's PAST position than
        the leader's current position.

        Shape correlation alone is useless here: two people walking the same
        corridor in parallel correlate near-perfectly at every lag. What
        separates a follower is that they occupy where the leader *was*. So this
        compares the best lagged distance against the zero-lag distance and
        returns the relative improvement - parallel travel gains nothing and
        scores 0, while genuine trailing scores high.
        """
        if len(lead_pts) < 20 or len(foll_pts) < 20:
            return 0.0
        n = min(len(lead_pts), len(foll_pts))
        lead = np.asarray(lead_pts[-n:], dtype=np.float64)
        foll = np.asarray(foll_pts[-n:], dtype=np.float64)

        zero_lag = float(np.linalg.norm(lead - foll, axis=1).mean())
        if zero_lag < 1e-6:
            return 0.0

        best_lagged = zero_lag
        for lag in range(3, min(max_lag, n // 2)):
            a = lead[: n - lag]
            b = foll[lag:]
            if len(a) < 10:
                break
            distance = float(np.linalg.norm(a - b, axis=1).mean())
            best_lagged = min(best_lagged, distance)

        # 0 when the lag buys nothing (parallel), -> 1 when it collapses the gap.
        return float(np.clip(1.0 - best_lagged / zero_lag, 0.0, 1.0))

    def pair_report(self, camera_id: str) -> List[Dict[str, Any]]:
        """Expose live pair statistics for the Relationship Intelligence screen."""
        out = []
        for (a, b), st in self._pairs.get(camera_id, {}).items():
            out.append(
                {
                    "track_a": a,
                    "track_b": b,
                    "duration_seconds": round(st.last_seen - st.first_seen, 1),
                    "samples": st.samples,
                    "mean_distance_px": round(float(np.mean(st.distances)), 1) if st.distances else None,
                    "behind_frames": st.behind_frames,
                    "mean_trailing_share": (
                        round(float(np.mean(st.trailing_samples)), 3)
                        if st.trailing_samples else None
                    ),
                    "mirrored_turns": st.mirrored_turns,
                    "peak_confidence": round(st.peak_confidence, 3),
                    "reported": st.reported_at is not None,
                }
            )
        return sorted(out, key=lambda r: -r["peak_confidence"])

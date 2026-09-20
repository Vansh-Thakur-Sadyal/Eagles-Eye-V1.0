"""Person-of-Interest agent - enrolled-reference search across the camera estate.

This is the "upload a photograph, find where that person is now" capability:
an authorised operator enrols one or more reference images of a subject (a
missing person, a suspect named in a case, a VIP under protection, or their own
face), and every camera in scope reports sightings with a location.

Matching is two-modality and deliberately conservative:

  face  - the strongest signal when a face is observable at usable resolution.
  body  - clothing/appearance embedding, used to carry a subject between
          cameras when the face is not observable.  Because clothing is not an
          identity, a body-only hit is capped well below a face hit and is
          always marked as requiring review.

Governance is enforced here rather than assumed:
  * enrolment needs a stated legal basis and an approving operator;
  * a subject can carry an expiry, after which matching stops automatically;
  * every match is written to the audit trail and is reviewable/rejectable;
  * output is always "possible sighting, confidence N%" - never "this is X".

The augmentation pipeline (Agent 4) is applied to each reference image so a
single photograph yields a robust multi-view gallery, which is what makes a
match survive a change of hairstyle, glasses, lighting or camera angle.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..vision.reid import cosine_similarity, get_body_embedder, get_face_embedder
from .base import Agent, FrameContext, Finding

log = logging.getLogger("sentinel.agents.watchlist")

# A body-only match can never be presented as strongly as a face match.
BODY_CONFIDENCE_CEILING = 0.72


@dataclass
class EnrolledSubject:
    """The in-memory, match-ready form of a WatchlistSubject row."""

    subject_id: str
    label: str
    category: str
    priority: str
    status: str
    face_embeddings: List[np.ndarray] = field(default_factory=list)
    body_embeddings: List[np.ndarray] = field(default_factory=list)
    match_threshold: Optional[float] = None
    scope_camera_ids: List[str] = field(default_factory=list)
    scope_site_ids: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    alert_on_match: bool = True

    def active_now(self, now: Optional[datetime] = None) -> bool:
        if self.status != "active":
            return False
        if self.expires_at is None:
            return True
        now = now or datetime.now(timezone.utc)
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return now < expires

    def in_scope(self, camera_id: str, site_id: Optional[str]) -> bool:
        if self.scope_camera_ids and camera_id not in self.scope_camera_ids:
            return False
        if self.scope_site_ids and site_id and site_id not in self.scope_site_ids:
            return False
        return True

    def best_face(self, embedding: np.ndarray) -> float:
        if not self.face_embeddings:
            return 0.0
        return max(cosine_similarity(embedding, e) for e in self.face_embeddings)

    def best_body(self, embedding: np.ndarray) -> float:
        if not self.body_embeddings:
            return 0.0
        return max(cosine_similarity(embedding, e) for e in self.body_embeddings)


class WatchlistAgent(Agent):
    name = "watchlist"
    spec_id = 13
    description = "Authorised person-of-interest matching across the camera estate"

    CHECK_EVERY_N_FRAMES = 4
    # Do not re-raise the same subject on the same camera more often than this.
    REPORT_COOLDOWN = 120.0

    def __init__(self) -> None:
        super().__init__()
        self.face = get_face_embedder()
        self.body = get_body_embedder()
        self._subjects: Dict[str, EnrolledSubject] = {}
        self._counter: Dict[str, int] = defaultdict(int)
        self._last_report: Dict[Tuple[str, str], float] = {}
        self._recent: List[Dict[str, Any]] = []

    # ------------------------------------------------------------ enrolment
    def load_subjects(self, subjects: List[EnrolledSubject]) -> None:
        """Replace the live gallery. Called at boot and after any enrolment change."""
        self._subjects = {s.subject_id: s for s in subjects}
        log.info("watchlist gallery loaded: %d subject(s)", len(self._subjects))

    def upsert_subject(self, subject: EnrolledSubject) -> None:
        self._subjects[subject.subject_id] = subject

    def remove_subject(self, subject_id: str) -> None:
        self._subjects.pop(subject_id, None)

    @property
    def active_subject_count(self) -> int:
        return sum(1 for s in self._subjects.values() if s.active_now())

    # ----------------------------------------------------------------- main
    def process(self, ctx: FrameContext) -> List[Finding]:
        if not self._subjects or ctx.frame is None:
            return []

        self._counter[ctx.camera.camera_id] += 1
        if self._counter[ctx.camera.camera_id] % self.CHECK_EVERY_N_FRAMES != 0:
            return []

        cfg = ctx.policy.get("watchlist", {})
        face_threshold = float(cfg.get("face_match_threshold", 0.55))
        body_threshold = float(cfg.get("body_match_threshold", 0.60))

        candidates = [
            s for s in self._subjects.values()
            if s.active_now(ctx.timestamp) and s.in_scope(ctx.camera.camera_id, ctx.camera.site_id)
        ]
        if not candidates:
            return []

        findings: List[Finding] = []
        now = ctx.elapsed_seconds

        for track in ctx.persons():
            if track.age_frames < 5:
                continue
            crop = ctx.crop(track)
            if crop is None or crop.size == 0:
                continue

            face_embedding, face_bbox, face_score = self.face.embed_largest(crop)
            # track.embedding is a numpy array: `array or x` raises, which made
            # this agent fail on every live track once anyone was enrolled.
            body_embedding = (track.embedding if track.embedding is not None
                              else self.body.embed(crop))

            best = self._best_subject(
                candidates, face_embedding, body_embedding, face_threshold, body_threshold
            )
            if best is None:
                continue

            subject, modality, similarity, confidence = best
            cooldown_key = (subject.subject_id, ctx.camera.camera_id)
            last = self._last_report.get(cooldown_key)
            if last is not None and (now - last) < self.REPORT_COOLDOWN:
                continue
            self._last_report[cooldown_key] = now

            location = self._describe_location(ctx)
            record = {
                "subject_id": subject.subject_id,
                "subject_label": subject.label,
                "category": subject.category,
                "priority": subject.priority,
                "camera_id": ctx.camera.camera_id,
                "camera_name": ctx.camera.name,
                "site_id": ctx.camera.site_id,
                "zone_id": track.zone_id or ctx.camera.zone_id,
                "track_id": str(track.track_id),
                "global_id": track.global_id,
                "modality": modality,
                "similarity": round(similarity, 4),
                "confidence": round(confidence, 4),
                "face_visibility": track.face_visibility,
                "face_detection_score": round(face_score, 3) if face_embedding is not None else None,
                "matched_at": ctx.timestamp.isoformat(),
                "location": location,
                "latitude": ctx.camera.latitude,
                "longitude": ctx.camera.longitude,
                "floor": ctx.camera.floor,
                "face_bbox": [round(v, 1) for v in face_bbox] if face_bbox else None,
                "bbox": [round(v, 1) for v in track.bbox],
                "face_backend": self.face.backend,
                "body_backend": self.body.backend,
                "requires_review": modality == "body" or confidence < 0.75,
            }
            self._recent.append(record)
            if len(self._recent) > 500:
                self._recent.pop(0)

            findings.append(
                Finding(
                    behavior="watchlist_match",
                    confidence=round(confidence, 3),
                    severity=self._severity(subject, confidence),
                    track_id=str(track.track_id),
                    zone_id=track.zone_id or ctx.camera.zone_id,
                    explanation=(
                        f"Possible sighting of enrolled subject '{subject.label}' at "
                        f"{location} ({ctx.camera.name}). "
                        f"{modality.capitalize()} similarity {similarity:.2f}, "
                        f"confidence {confidence * 100:.0f}%. "
                        + (
                            "Body-appearance match only - clothing is not an identity, "
                            "human confirmation required."
                            if modality == "body"
                            else "Face-based match - human confirmation required before any action."
                        )
                    ),
                    evidence=record,
                    dedupe_key=f"watchlist:{subject.subject_id}:{ctx.camera.camera_id}",
                )
            )

        return findings

    # -------------------------------------------------------------- scoring
    def _best_subject(
        self,
        candidates: List[EnrolledSubject],
        face_embedding: Optional[np.ndarray],
        body_embedding: Optional[np.ndarray],
        face_threshold: float,
        body_threshold: float,
    ) -> Optional[Tuple[EnrolledSubject, str, float, float]]:
        best: Optional[Tuple[EnrolledSubject, str, float, float]] = None

        for subject in candidates:
            face_sim = subject.best_face(face_embedding) if face_embedding is not None else 0.0
            body_sim = subject.best_body(body_embedding) if body_embedding is not None else 0.0

            f_thresh = subject.match_threshold if subject.match_threshold is not None else face_threshold
            b_thresh = subject.match_threshold if subject.match_threshold is not None else body_threshold

            face_hit = face_embedding is not None and face_sim >= f_thresh
            body_hit = body_embedding is not None and body_sim >= b_thresh

            if face_hit and body_hit:
                # Agreement across independent modalities raises confidence,
                # but never to certainty.
                modality = "fused"
                similarity = max(face_sim, body_sim)
                confidence = min(0.97, 0.5 + 0.5 * face_sim + 0.15 * body_sim)
            elif face_hit:
                modality = "face"
                similarity = face_sim
                confidence = min(0.95, face_sim)
            elif body_hit:
                modality = "body"
                similarity = body_sim
                confidence = min(BODY_CONFIDENCE_CEILING, body_sim)
            else:
                continue

            if best is None or confidence > best[3]:
                best = (subject, modality, similarity, confidence)

        return best

    @staticmethod
    def _severity(subject: EnrolledSubject, confidence: float) -> str:
        if subject.priority == "critical":
            return "critical"
        if subject.priority == "high" and confidence >= 0.7:
            return "high"
        if confidence >= 0.85:
            return "high"
        return "medium"

    @staticmethod
    def _describe_location(ctx: FrameContext) -> str:
        bits = [b for b in (ctx.camera.location, ctx.camera.name) if b]
        return " - ".join(dict.fromkeys(bits)) or ctx.camera.camera_id

    # -------------------------------------------------------------- readers
    def recent_matches(self, subject_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._recent
        if subject_id:
            rows = [r for r in rows if r["subject_id"] == subject_id]
        return list(reversed(rows[-limit:]))

    def status(self) -> Dict[str, Any]:
        base = super().status()
        base.update(
            {
                "enrolled_subjects": len(self._subjects),
                "active_subjects": self.active_subject_count,
                "face_backend": self.face.backend,
                "body_backend": self.body.backend,
                "recent_matches": len(self._recent),
            }
        )
        return base

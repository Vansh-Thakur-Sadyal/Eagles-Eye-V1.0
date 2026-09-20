"""Agents 3, 4 and 5 - cross-camera Re-ID, appearance robustness, occlusion.

Three tightly related competences kept in one module because they share the
embedding gallery:

  ReIDAgent         associates tracks across cameras, emitting a *confidence*
                    ("cross-camera association confidence: 91%") rather than an
                    identity claim, exactly as spec S6 requires.
  AppearanceAgent   keeps a multi-view embedding bank per subject so a change of
                    jacket, lighting or angle does not spawn a new identity
                    (spec S7).
  OcclusionAgent    reports face observability and lowers identity confidence
                    accordingly.  Per spec S8 a covering is explicitly NOT a
                    risk signal on its own.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models import new_id
from ..vision.reid import (
    EmbeddingGallery,
    assess_face_visibility,
    cosine_similarity,
    get_body_embedder,
    get_face_embedder,
)
from .base import Agent, FrameContext, Finding

log = logging.getLogger("sentinel.agents.identity")


@dataclass
class _SubjectBank:
    """Multi-view appearance memory for one cross-camera subject."""

    global_id: str
    label: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    cameras: List[str] = field(default_factory=list)
    tracks: List[str] = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    confidence: float = 0.0
    max_views: int = 12

    def add_view(self, embedding: np.ndarray) -> None:
        """Keep views that are mutually diverse, so one pose cannot dominate."""
        if not self.embeddings:
            self.embeddings.append(embedding)
            return
        sims = [cosine_similarity(embedding, e) for e in self.embeddings]
        if max(sims) > 0.97:
            return                                    # near-duplicate view
        self.embeddings.append(embedding)
        if len(self.embeddings) > self.max_views:
            # drop the view most redundant with the rest
            redundancy = [
                sum(cosine_similarity(e, o) for o in self.embeddings if o is not e)
                for e in self.embeddings
            ]
            self.embeddings.pop(int(np.argmax(redundancy)))

    def best_similarity(self, embedding: np.ndarray) -> float:
        if not self.embeddings:
            return 0.0
        return max(cosine_similarity(embedding, e) for e in self.embeddings)

    def centroid(self) -> Optional[np.ndarray]:
        if not self.embeddings:
            return None
        avg = np.mean(np.stack(self.embeddings), axis=0)
        norm = np.linalg.norm(avg)
        return avg / norm if norm else avg


class ReIDAgent(Agent):
    """Agent 3 - cross-camera association."""

    name = "reid"
    spec_id = 3
    description = "Cross-camera re-identification with explicit association confidence"

    def __init__(self) -> None:
        super().__init__()
        self._banks: Dict[str, _SubjectBank] = {}
        self._gallery = EmbeddingGallery()
        self._track_to_global: Dict[Tuple[str, int], str] = {}
        self._counter = 0
        self.embedder = get_body_embedder()

    def reset(self, camera_id: Optional[str] = None) -> None:
        if camera_id is None:
            self._banks.clear()
            self._gallery.clear()
            self._track_to_global.clear()

    # ------------------------------------------------------------------ main
    def process(self, ctx: FrameContext) -> List[Finding]:
        cfg = ctx.policy.get("reid", {})
        threshold = float(cfg.get("match_threshold", 0.62))
        max_gap = float(cfg.get("max_time_gap_seconds", 900))
        findings: List[Finding] = []

        for track in ctx.persons():
            if track.age_frames < 6:
                continue

            key = (ctx.camera.camera_id, track.track_id)
            embedding = track.embedding
            if embedding is None:
                crop = ctx.crop(track)
                if crop is None:
                    continue
                embedding = self.embedder.embed(crop)
                if embedding is None:
                    continue
                track.embedding = embedding

            existing = self._track_to_global.get(key)
            if existing and existing in self._banks:
                bank = self._banks[existing]
                bank.last_seen = ctx.timestamp
                bank.add_view(embedding)
                track.global_id = existing
                continue

            match_id, similarity = self._best_match(
                embedding, ctx.camera.camera_id, ctx.timestamp, max_gap
            )

            if match_id and similarity >= threshold:
                bank = self._banks[match_id]
                is_new_camera = ctx.camera.camera_id not in bank.cameras
                bank.add_view(embedding)
                bank.last_seen = ctx.timestamp
                bank.confidence = similarity
                if is_new_camera:
                    bank.cameras.append(ctx.camera.camera_id)
                bank.tracks.append(f"{ctx.camera.camera_id}:{track.track_id}")
                track.global_id = match_id
                self._track_to_global[key] = match_id

                if is_new_camera:
                    findings.append(
                        Finding(
                            behavior="cross_camera_association",
                            confidence=round(similarity, 3),
                            severity="info",
                            track_id=str(track.track_id),
                            zone_id=ctx.camera.zone_id,
                            explanation=(
                                f"Track {track.track_id} on {ctx.camera.name} is most consistent "
                                f"with subject {bank.label}, previously observed on "
                                f"{len(bank.cameras) - 1} other camera(s). "
                                f"Cross-camera association confidence: {similarity * 100:.0f}%. "
                                "This is an appearance-based association, not an identification."
                            ),
                            evidence={
                                "global_id": match_id,
                                "label": bank.label,
                                "similarity": round(similarity, 4),
                                "threshold": threshold,
                                "cameras": list(bank.cameras),
                                "views_held": len(bank.embeddings),
                                "embedding_backend": self.embedder.backend,
                            },
                            dedupe_key=f"assoc:{match_id}:{ctx.camera.camera_id}",
                        )
                    )
            else:
                gid = self._create_subject(ctx, track, embedding, similarity)
                track.global_id = gid
                self._track_to_global[key] = gid

        return findings

    def _best_match(
        self, embedding: np.ndarray, camera_id: str, now: datetime, max_gap: float
    ) -> Tuple[Optional[str], float]:
        best_id, best_sim = None, 0.0
        for gid, bank in self._banks.items():
            if bank.last_seen and (now - bank.last_seen).total_seconds() > max_gap:
                continue
            sim = bank.best_similarity(embedding)
            if sim > best_sim:
                best_id, best_sim = gid, sim
        return best_id, best_sim

    def _create_subject(self, ctx: FrameContext, track, embedding: np.ndarray, sim: float) -> str:
        self._counter += 1
        gid = new_id("SUBJ")
        label = f"SUBJECT_{self._counter:04d}"
        bank = _SubjectBank(
            global_id=gid,
            label=label,
            cameras=[ctx.camera.camera_id],
            tracks=[f"{ctx.camera.camera_id}:{track.track_id}"],
            first_seen=ctx.timestamp,
            last_seen=ctx.timestamp,
            confidence=1.0,
        )
        bank.add_view(embedding)
        self._banks[gid] = bank
        self._gallery.add(gid, embedding, {"label": label, "camera": ctx.camera.camera_id})
        return gid

    # -------------------------------------------------------------- readers
    def subject_summary(self, limit: int = 100) -> List[Dict[str, Any]]:
        rows = []
        for gid, bank in self._banks.items():
            rows.append(
                {
                    "global_id": gid,
                    "label": bank.label,
                    "cameras": list(dict.fromkeys(bank.cameras)),
                    "camera_count": len(set(bank.cameras)),
                    "track_count": len(bank.tracks),
                    "views_held": len(bank.embeddings),
                    "association_confidence": round(bank.confidence, 3),
                    "first_seen": bank.first_seen.isoformat() if bank.first_seen else None,
                    "last_seen": bank.last_seen.isoformat() if bank.last_seen else None,
                }
            )
        rows.sort(key=lambda r: r["last_seen"] or "", reverse=True)
        return rows[:limit]

    def match_embedding(self, embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Query the live gallery - backs 'find this person' forensic search."""
        out = []
        for gid, bank in self._banks.items():
            sim = bank.best_similarity(embedding)
            if sim <= 0:
                continue
            out.append(
                {
                    "global_id": gid,
                    "label": bank.label,
                    "similarity": round(sim, 4),
                    "cameras": list(dict.fromkeys(bank.cameras)),
                    "last_seen": bank.last_seen.isoformat() if bank.last_seen else None,
                }
            )
        out.sort(key=lambda r: -r["similarity"])
        return out[:top_k]

    def trajectory_for(self, global_id: str) -> List[str]:
        bank = self._banks.get(global_id)
        return list(bank.tracks) if bank else []


class AppearanceAgent(Agent):
    """Agent 4 - appearance robustness.

    Measures how much a subject's appearance has drifted and, when the drift is
    large but the track is continuous, records the new view so the subject is
    not re-identified as somebody new.  It reports variation; it never asserts
    that a change of appearance is suspicious.
    """

    name = "appearance"
    spec_id = 4
    description = "Robust matching across clothing, lighting, pose and quality changes"

    DRIFT_REPORT_THRESHOLD = 0.45

    def __init__(self, reid_agent: Optional[ReIDAgent] = None) -> None:
        super().__init__()
        self._reid = reid_agent
        self._last_embedding: Dict[Tuple[str, int], np.ndarray] = {}
        self._drift_reported: Dict[Tuple[str, int], bool] = defaultdict(bool)

    def bind(self, reid_agent: ReIDAgent) -> None:
        self._reid = reid_agent

    def reset(self, camera_id: Optional[str] = None) -> None:
        self._last_embedding.clear()
        self._drift_reported.clear()

    def process(self, ctx: FrameContext) -> List[Finding]:
        findings: List[Finding] = []
        for track in ctx.persons():
            if track.embedding is None or track.age_frames < 10:
                continue
            key = (ctx.camera.camera_id, track.track_id)
            previous = self._last_embedding.get(key)
            self._last_embedding[key] = track.embedding

            if previous is None:
                continue
            similarity = cosine_similarity(previous, track.embedding)
            drift = 1.0 - similarity
            track.attributes["appearance_drift"] = round(drift, 3)

            if drift < self.DRIFT_REPORT_THRESHOLD or self._drift_reported[key]:
                continue
            self._drift_reported[key] = True

            findings.append(
                Finding(
                    behavior="appearance_change",
                    confidence=round(float(np.clip(drift, 0, 0.95)), 3),
                    severity="info",
                    track_id=str(track.track_id),
                    zone_id=ctx.camera.zone_id,
                    explanation=(
                        f"Appearance of track {track.track_id} shifted materially "
                        f"(embedding drift {drift:.2f}) while tracking remained continuous. "
                        "The subject's existing association has been preserved rather than "
                        "creating a new identity. Appearance change is not itself a risk signal."
                    ),
                    evidence={
                        "drift": round(drift, 4),
                        "similarity_to_previous_view": round(similarity, 4),
                        "track_continuous": True,
                        "global_id": track.global_id,
                    },
                    dedupe_key=f"appearance:{ctx.camera.camera_id}:{track.track_id}",
                )
            )
        return findings


class OcclusionAgent(Agent):
    """Agent 5 - face covering and observability.

    Emits an observability report, never a suspicion.  The threat model gives
    `face_unavailable` a weight of 0 precisely so this cannot inflate a score.
    """

    name = "occlusion"
    spec_id = 5
    description = "Face observability and covering assessment (not a risk signal)"

    CHECK_EVERY_N_FRAMES = 8

    def __init__(self) -> None:
        super().__init__()
        self.face = get_face_embedder()
        self._counter: Dict[str, int] = defaultdict(int)

    def process(self, ctx: FrameContext) -> List[Finding]:
        self._counter[ctx.camera.camera_id] += 1
        if self._counter[ctx.camera.camera_id] % self.CHECK_EVERY_N_FRAMES != 0:
            return []
        if ctx.frame is None:
            return []

        cfg = ctx.policy.get("occlusion", {})
        min_area = float(cfg.get("face_visible_min_area", 0.012))
        findings: List[Finding] = []

        for track in ctx.persons():
            crop = ctx.crop(track)
            if crop is None or crop.size == 0:
                continue
            faces = self.face.detect(crop)
            report = assess_face_visibility(
                crop, [f["bbox"] for f in faces], min_face_area_ratio=min_area
            )
            previous = track.face_visibility
            track.face_visibility = report["visibility"]
            track.attributes["identity_confidence"] = report["identity_confidence"]

            if report["visibility"] in ("covered", "absent") and previous != report["visibility"]:
                findings.append(
                    Finding(
                        behavior="face_unavailable",
                        confidence=round(1.0 - report["identity_confidence"], 3),
                        severity="info",
                        track_id=str(track.track_id),
                        zone_id=ctx.camera.zone_id,
                        explanation=(
                            f"Face visibility: {report['visibility'].upper()}. "
                            f"{report['reason'].capitalize()}. Identity confidence is therefore "
                            f"{report['identity_confidence']:.2f}. "
                            "A covered face is ordinary behaviour and carries no risk weight."
                        ),
                        evidence={
                            **report,
                            "faces_found": len(faces),
                            "backend": self.face.backend,
                            "risk_weight": 0,
                        },
                        dedupe_key=f"face:{ctx.camera.camera_id}:{track.track_id}",
                    )
                )
        return findings

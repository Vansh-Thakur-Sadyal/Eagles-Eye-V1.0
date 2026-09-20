"""Writes live pipeline output into the database.

Subscribes to the orchestrator's incident sink and the pipeline's per-frame
results, and keeps the durable record in step with what the agents are seeing:
incidents and their timelines, behaviour events, tracks, tracked objects,
crowd samples, watchlist matches and alerts.

Runs on the camera worker threads, so it uses its own short-lived sessions and
never holds a request-scoped one.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from ..config import get_settings
from ..db import session_scope
from ..models import (
    Alert,
    BehaviorEvent,
    CrowdSample,
    GlobalSubject,
    Incident,
    IncidentTimelineEntry,
    Track,
    TrackedObject,
    WatchlistMatch,
    new_id,
    utcnow,
)
from .events import bus

log = logging.getLogger("sentinel.persistence")


def _parse(ts: Optional[str]) -> datetime:
    if not ts:
        return utcnow()
    try:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return utcnow()


class PersistenceService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._crowd_last_write: Dict[str, float] = {}
        self.incidents_written = 0
        self.events_written = 0
        self.tracks_written = 0
        self.errors = 0

    # ------------------------------------------------------------- incidents
    def on_incident(self, payload: Dict[str, Any]) -> None:
        """Insert or update one incident and append its new timeline lines."""
        try:
            with session_scope() as db:
                incident = db.get(Incident, payload["id"])
                started = _parse(payload.get("started_at"))
                updated = _parse(payload.get("last_update_at"))

                if incident is None:
                    incident = Incident(
                        id=payload["id"],
                        site_id=payload.get("site_id"),
                        camera_id=payload.get("camera_id"),
                        camera_ids=[payload["camera_id"]] if payload.get("camera_id") else [],
                        zone_id=payload.get("zone_id"),
                        event_type=payload.get("event_type", "observation"),
                        title=payload.get("title", "Incident"),
                        started_at=started,
                        last_update_at=updated,
                        status="open",
                    )
                    db.add(incident)
                    self.incidents_written += 1

                incident.summary = payload.get("summary")
                incident.explanation = payload.get("explanation")
                incident.risk_score = float(payload.get("risk_score", 0.0))
                incident.severity = payload.get("severity", "low")
                incident.risk_factors = payload.get("risk_factors", [])
                incident.confidence = float(payload.get("confidence", 0.0))
                incident.track_ids = payload.get("track_ids", [])
                incident.global_ids = payload.get("global_ids", [])
                incident.object_ids = payload.get("object_ids", [])
                incident.recommended_actions = payload.get("recommended_actions", [])
                incident.location = payload.get("location", {})
                incident.last_update_at = updated
                incident.event_type = payload.get("event_type", incident.event_type)
                incident.title = payload.get("title", incident.title)
                incident.meta = {
                    "behaviors": payload.get("behaviors", []),
                    "narration_source": payload.get("narration_source"),
                    "camera_name": payload.get("camera_name"),
                }

                existing = {
                    (e.at.isoformat(), e.text)
                    for e in db.scalars(
                        select(IncidentTimelineEntry).where(
                            IncidentTimelineEntry.incident_id == incident.id
                        )
                    )
                }
                for entry in payload.get("timeline", []):
                    at = _parse(entry.get("at"))
                    if (at.isoformat(), entry.get("text", "")) in existing:
                        continue
                    db.add(
                        IncidentTimelineEntry(
                            id=new_id("TLE"),
                            incident_id=incident.id,
                            at=at,
                            kind=entry.get("kind", "observation"),
                            actor=entry.get("actor", "system"),
                            text=entry.get("text", ""),
                            camera_id=payload.get("camera_id"),
                            confidence=entry.get("confidence"),
                            payload=entry.get("payload", {}),
                        )
                    )

                if payload.get("is_new"):
                    db.add(
                        Alert(
                            id=new_id("ALR"),
                            incident_id=incident.id,
                            channel="dashboard",
                            severity=incident.severity,
                            title=incident.title,
                            body=incident.summary,
                            payload={
                                "risk_score": incident.risk_score,
                                "camera_id": incident.camera_id,
                                "camera_name": payload.get("camera_name"),
                            },
                        )
                    )
        except Exception:
            self.errors += 1
            log.exception("failed to persist incident %s", payload.get("id"))

        # Index for forensic retrieval. Without this an incident is stored but
        # unsearchable, which is how a "no results" answer can be wrong.
        try:
            from ..rag.forensic import get_forensic

            get_forensic().index_incident(
                {
                    "id": payload["id"],
                    "title": payload.get("title"),
                    "summary": payload.get("summary"),
                    "explanation": payload.get("explanation"),
                    "behaviors": payload.get("behaviors", []),
                    "camera_id": payload.get("camera_id"),
                    "camera_name": payload.get("camera_name"),
                    "zone_id": payload.get("zone_id"),
                    "site_id": payload.get("site_id"),
                    "event_type": payload.get("event_type"),
                    "severity": payload.get("severity"),
                    "risk_score": payload.get("risk_score", 0),
                    "started_at": payload.get("started_at"),
                    "location": payload.get("location", {}),
                    "track_ids": payload.get("track_ids", []),
                }
            )
        except Exception:
            log.exception("failed to index incident %s for search", payload.get("id"))

    # ------------------------------------------------------------- findings
    def on_finding(self, evt) -> None:
        """Persist every agent finding as a BehaviorEvent row."""
        if evt.topic != "finding":
            return
        p = evt.payload
        try:
            with session_scope() as db:
                db.add(
                    BehaviorEvent(
                        id=new_id("BEV"),
                        camera_id=p.get("camera_id"),
                        zone_id=p.get("zone_id"),
                        track_id=p.get("track_id"),
                        secondary_track_id=p.get("secondary_track_id"),
                        behavior=p.get("behavior", "unknown"),
                        severity=p.get("severity", "info"),
                        confidence=float(p.get("confidence", 0.0)),
                        started_at=_parse(p.get("at")),
                        ended_at=_parse(p.get("at")),
                        duration_seconds=float(p.get("evidence", {}).get("duration_seconds", 0.0) or 0.0),
                        evidence=p.get("evidence", {}),
                        explanation=p.get("explanation"),
                        agent=p.get("agent", ""),
                    )
                )
                self.events_written += 1
        except Exception:
            self.errors += 1
            log.exception("failed to persist finding")

    # ---------------------------------------------------------- frame output
    def on_frame_result(self, result: Dict[str, Any], force: bool = False) -> None:
        """Upsert tracks and crowd telemetry on a throttled cadence.

        ``force`` skips the frame-based throttle: edge nodes already send only
        about one result per camera every two seconds.
        """
        camera_id = result.get("camera_id")
        if not camera_id:
            return
        # Track rows are written every ~2 seconds of wall time, not every frame.
        if not force and result.get("frame_index", 0) % max(1, int(result.get("measured_fps") or 12) * 2) != 0:
            return
        try:
            with session_scope() as db:
                now = _parse(result.get("timestamp"))
                for t in result.get("tracks", []):
                    row = db.scalar(
                        select(Track).where(
                            Track.camera_id == camera_id,
                            Track.local_track_id == t["track_id"],
                            Track.active.is_(True),
                        )
                    )
                    if row is None:
                        row = Track(
                            id=new_id("TRK"),
                            camera_id=camera_id,
                            local_track_id=t["track_id"],
                            class_name=t.get("class_name", "person"),
                            first_seen=now,
                            last_seen=now,
                        )
                        db.add(row)
                        self.tracks_written += 1
                    row.last_seen = now
                    # Track.global_id is a foreign key. The Re-ID agent mints
                    # subject ids in memory, so the row must exist before a
                    # track can reference it - otherwise every associated track
                    # silently fails to persist on a FK violation.
                    global_id = t.get("global_id")
                    if global_id:
                        self._ensure_subject(db, global_id, camera_id, now)
                    row.global_id = global_id
                    row.frame_count = t.get("age_frames", 0)
                    row.avg_speed = t.get("speed", 0.0)
                    row.max_speed = max(row.max_speed or 0.0, t.get("speed", 0.0))
                    row.dwell_seconds = t.get("duration_seconds", 0.0)
                    row.face_visibility = t.get("face_visibility", "unknown")
                    if t.get("zone_id") and t["zone_id"] not in (row.zones_visited or []):
                        row.zones_visited = (row.zones_visited or []) + [t["zone_id"]]

                crowd = result.get("crowd")
                if crowd:
                    db.add(
                        CrowdSample(
                            id=new_id("CRW"),
                            camera_id=camera_id,
                            sampled_at=now,
                            count=int(crowd.get("count", 0)),
                            density=float(crowd.get("density", 0.0)),
                            density_band=crowd.get("density_band", "low"),
                            mean_speed=float(crowd.get("mean_speed", 0.0)),
                            flow_direction_deg=crowd.get("flow_direction_deg"),
                            flow_variance=float(crowd.get("flow_variance", 0.0)),
                            counter_flow_ratio=float(crowd.get("counter_flow_ratio", 0.0)),
                            compression=float(crowd.get("compression", 0.0)),
                            delta_percent=float(crowd.get("delta_percent", 0.0)),
                            risk=crowd.get("risk", "low"),
                        )
                    )
        except Exception:
            self.errors += 1
            log.exception("failed to persist frame result for %s", camera_id)

    @staticmethod
    def _ensure_subject(db, global_id: str, camera_id: str, now: datetime) -> None:
        """Create or extend the GlobalSubject a track is associated with."""
        subject = db.get(GlobalSubject, global_id)
        if subject is None:
            subject = GlobalSubject(
                id=global_id,
                label=global_id.replace("SUBJ_", "SUBJECT_"),
                first_seen=now,
                last_seen=now,
                camera_ids=[camera_id],
            )
            db.add(subject)
            db.flush()
            return
        subject.last_seen = now
        if camera_id not in (subject.camera_ids or []):
            subject.camera_ids = (subject.camera_ids or []) + [camera_id]

    # ------------------------------------------------------ watchlist matches
    def on_watchlist_finding(self, evt) -> None:
        if evt.topic != "finding" or evt.payload.get("behavior") != "watchlist_match":
            return
        ev = evt.payload.get("evidence", {})
        try:
            with session_scope() as db:
                db.add(
                    WatchlistMatch(
                        id=new_id("WLM"),
                        subject_id=ev.get("subject_id"),
                        camera_id=ev.get("camera_id"),
                        site_id=ev.get("site_id"),
                        zone_id=ev.get("zone_id"),
                        track_id=ev.get("track_id"),
                        global_id=ev.get("global_id"),
                        matched_at=_parse(ev.get("matched_at")),
                        modality=ev.get("modality", "face"),
                        similarity=float(ev.get("similarity", 0.0)),
                        confidence=float(ev.get("confidence", 0.0)),
                        face_visibility=ev.get("face_visibility", "unknown"),
                        location_name=ev.get("location"),
                        latitude=ev.get("latitude"),
                        longitude=ev.get("longitude"),
                        floor=ev.get("floor"),
                        review_status="unreviewed",
                        meta=ev,
                    )
                )
        except Exception:
            self.errors += 1
            log.exception("failed to persist watchlist match")

    def status(self) -> Dict[str, Any]:
        return {
            "incidents_written": self.incidents_written,
            "events_written": self.events_written,
            "tracks_written": self.tracks_written,
            "errors": self.errors,
        }


persistence = PersistenceService()


def wire_persistence() -> None:
    """Connect the service to the orchestrator, pipeline and event bus."""
    from ..agents.orchestrator import get_orchestrator
    from ..pipeline.runner import get_pipeline

    orchestrator = get_orchestrator()
    orchestrator.add_sink(persistence.on_incident)
    get_pipeline().add_result_sink(persistence.on_frame_result)
    bus.on(persistence.on_finding)
    bus.on(persistence.on_watchlist_finding)
    log.info("persistence wired")

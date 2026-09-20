"""Intelligence endpoints: tracks, subjects, behaviour, crowd, objects, threat."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...agents.orchestrator import get_orchestrator
from ...agents.threat import ThreatAssessor
from ...config import load_policy
from ...core.security import current_user, has_permission, require
from ...db import get_session
from ...models import (
    ApprovalRequest,
    BehaviorEvent,
    Camera,
    CrowdSample,
    GlobalSubject,
    Incident,
    Track,
    TrackedObject,
    User,
    new_id,
    utcnow,
)
from ...pipeline.runner import get_pipeline
from ..deps import Page, get_or_404, pagination, record, to_dict

router = APIRouter(prefix="/api/intel", tags=["intelligence"])


# ------------------------------------------------------------------ tracks
@router.get("/tracks")
def list_tracks(
    camera_id: Optional[str] = None,
    class_name: Optional[str] = None,
    active_only: bool = False,
    since_minutes: Optional[int] = None,
    page: Dict[str, int] = Depends(pagination),
    user: User = Depends(require("track:read")),
    db: Session = Depends(get_session),
):
    stmt = select(Track)
    if camera_id:
        stmt = stmt.where(Track.camera_id == camera_id)
    if class_name:
        stmt = stmt.where(Track.class_name == class_name)
    if active_only:
        stmt = stmt.where(Track.active.is_(True))
    if since_minutes:
        stmt = stmt.where(
            Track.last_seen >= datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
        )

    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(
        stmt.order_by(Track.last_seen.desc()).limit(page["limit"]).offset(page["offset"])
    ).all()

    privacy = get_orchestrator().privacy
    items = privacy.filter_many(
        [to_dict(t, exclude={"trajectory", "world_path"}) for t in rows], role=user.role
    )
    return Page.of(items, total, page["limit"], page["offset"])


@router.get("/tracks/live")
def live_tracks(user: User = Depends(require("track:read"))):
    """Everything currently tracked, straight from the running workers."""
    pipeline = get_pipeline()
    out = []
    for camera_id in pipeline.running_ids():
        worker = pipeline.get(camera_id)
        if worker is None:
            continue
        out.append(
            {
                "camera_id": camera_id,
                "camera_name": worker.runtime.name,
                "tracks": worker.snapshot_tracks(),
                "measured_fps": worker.measured_fps,
            }
        )
    return {"cameras": out, "total_tracks": sum(len(c["tracks"]) for c in out)}


@router.get("/tracks/{track_id}")
def get_track(
    track_id: str,
    request: Request,
    user: User = Depends(require("track:read")),
    db: Session = Depends(get_session),
):
    track = get_or_404(db, Track, track_id, "track")
    record(db, request, user, "track.read", "track", track_id)
    camera = db.get(Camera, track.camera_id)
    return {
        **to_dict(track),
        "camera_name": camera.name if camera else None,
        "camera_location": camera.location if camera else None,
    }


# ---------------------------------------------------------------- subjects
@router.get("/subjects")
def list_subjects(
    limit: int = 100,
    user: User = Depends(require("track:read")),
    db: Session = Depends(get_session),
):
    """Cross-camera association hypotheses - never identity records."""
    live = get_orchestrator().reid.subject_summary(limit=limit)
    stored = db.scalars(
        select(GlobalSubject).order_by(GlobalSubject.last_seen.desc()).limit(limit)
    ).all()
    return {
        "live": live,
        "stored": [to_dict(s) for s in stored],
        "note": "Association confidence expresses appearance similarity across cameras. "
                "It is not an identification.",
        "embedding_backend": get_orchestrator().reid.embedder.info(),
    }


@router.get("/subjects/{global_id}/trajectory")
def subject_trajectory(
    global_id: str,
    request: Request,
    user: User = Depends(require("track:read")),
    db: Session = Depends(get_session),
):
    """Reconstruct a subject's cross-camera path (spec S6, S13)."""
    reid = get_orchestrator().reid
    track_keys = reid.trajectory_for(global_id)
    rows = db.scalars(
        select(Track).where(Track.global_id == global_id).order_by(Track.first_seen)
    ).all()
    cameras = {c.id: c for c in db.scalars(select(Camera)).all()}

    legs = []
    for t in rows:
        camera = cameras.get(t.camera_id)
        legs.append(
            {
                "track_id": t.id,
                "camera_id": t.camera_id,
                "camera_name": camera.name if camera else None,
                "location": camera.location if camera else None,
                "latitude": camera.latitude if camera else None,
                "longitude": camera.longitude if camera else None,
                "floor": camera.floor if camera else None,
                "first_seen": t.first_seen.isoformat(),
                "last_seen": t.last_seen.isoformat(),
                "dwell_seconds": t.dwell_seconds,
                "zones_visited": t.zones_visited,
            }
        )

    record(db, request, user, "subject.trajectory", "global_subject", global_id)
    return {
        "global_id": global_id,
        "legs": legs,
        "camera_sequence": [l["camera_id"] for l in legs],
        "live_track_keys": track_keys,
        "confidence_note": "Each leg is an appearance-based association with its own confidence.",
    }


# ------------------------------------------------------------ behaviour log
@router.get("/behavior")
def behavior_events(
    behavior: Optional[str] = None,
    camera_id: Optional[str] = None,
    severity: Optional[str] = None,
    min_confidence: float = 0.0,
    since_minutes: int = 1440,
    page: Dict[str, int] = Depends(pagination),
    user: User = Depends(require("incident:read")),
    db: Session = Depends(get_session),
):
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    stmt = select(BehaviorEvent).where(BehaviorEvent.started_at >= cutoff)
    if behavior:
        stmt = stmt.where(BehaviorEvent.behavior == behavior)
    if camera_id:
        stmt = stmt.where(BehaviorEvent.camera_id == camera_id)
    if severity:
        stmt = stmt.where(BehaviorEvent.severity == severity)
    if min_confidence:
        stmt = stmt.where(BehaviorEvent.confidence >= min_confidence)

    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(
        stmt.order_by(BehaviorEvent.started_at.desc())
        .limit(page["limit"]).offset(page["offset"])
    ).all()
    return Page.of([to_dict(r) for r in rows], total, page["limit"], page["offset"])


@router.get("/behavior/stats")
def behavior_stats(
    since_hours: int = 24,
    user: User = Depends(require("incident:read")),
    db: Session = Depends(get_session),
):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    rows = db.execute(
        select(BehaviorEvent.behavior, func.count(), func.avg(BehaviorEvent.confidence))
        .where(BehaviorEvent.started_at >= cutoff)
        .group_by(BehaviorEvent.behavior)
        .order_by(func.count().desc())
    ).all()
    return {
        "window_hours": since_hours,
        "items": [
            {"behavior": b, "count": c, "mean_confidence": round(float(a or 0), 3)}
            for b, c, a in rows
        ],
        "total": sum(c for _b, c, _a in rows),
    }


# --------------------------------------------------------------------- crowd
@router.get("/crowd/live")
def crowd_live(user: User = Depends(current_user), db: Session = Depends(get_session)):
    latest = get_orchestrator().crowd.all_latest()
    cameras = {c.id: c for c in db.scalars(select(Camera)).all()}
    items = []
    for camera_id, metrics in latest.items():
        camera = cameras.get(camera_id)
        items.append(
            {
                "camera_id": camera_id,
                "camera_name": camera.name if camera else None,
                "zone_id": camera.zone_id if camera else None,
                **{k: v for k, v in metrics.items() if k != "heatmap"},
            }
        )
    items.sort(key=lambda r: -(r.get("count") or 0))
    total = sum(i.get("count") or 0 for i in items)
    worst = max(items, key=lambda r: ["low", "medium", "high", "critical"].index(
        r.get("risk", "low")) if r.get("risk") in ("low", "medium", "high", "critical") else 0,
        default=None) if items else None
    return {"items": items, "total_people": total, "highest_risk": worst}


@router.get("/crowd/{camera_id}/heatmap")
def crowd_heatmap(camera_id: str, user: User = Depends(current_user)):
    metrics = get_orchestrator().crowd.latest(camera_id)
    if metrics is None:
        raise HTTPException(404, "no crowd telemetry for this camera yet")
    return {
        "camera_id": camera_id,
        "rows": metrics.get("grid_rows"),
        "cols": metrics.get("grid_cols"),
        "heatmap": metrics.get("heatmap"),
        "density_band": metrics.get("density_band"),
        "count": metrics.get("count"),
    }


@router.get("/crowd/history")
def crowd_history(
    camera_id: Optional[str] = None,
    since_minutes: int = 120,
    limit: int = 500,
    user: User = Depends(current_user),
    db: Session = Depends(get_session),
):
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    stmt = select(CrowdSample).where(CrowdSample.sampled_at >= cutoff)
    if camera_id:
        stmt = stmt.where(CrowdSample.camera_id == camera_id)
    rows = db.scalars(stmt.order_by(CrowdSample.sampled_at.desc()).limit(limit)).all()
    return {"items": [to_dict(r, exclude={"heatmap"}) for r in reversed(rows)]}


# ------------------------------------------------------------------ objects
@router.get("/objects")
def list_objects(
    status_filter: Optional[str] = None,
    camera_id: Optional[str] = None,
    page: Dict[str, int] = Depends(pagination),
    user: User = Depends(current_user),
    db: Session = Depends(get_session),
):
    stmt = select(TrackedObject)
    if status_filter:
        stmt = stmt.where(TrackedObject.status == status_filter)
    if camera_id:
        stmt = stmt.where(TrackedObject.camera_id == camera_id)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(
        stmt.order_by(TrackedObject.last_seen.desc())
        .limit(page["limit"]).offset(page["offset"])
    ).all()
    return Page.of([to_dict(r) for r in rows], total, page["limit"], page["offset"])


@router.get("/objects/live")
def live_objects(user: User = Depends(current_user)):
    agent = get_orchestrator().objects
    pipeline = get_pipeline()
    out = []
    for camera_id in pipeline.running_ids():
        worker = pipeline.get(camera_id)
        out.append(
            {
                "camera_id": camera_id,
                "camera_name": worker.runtime.name if worker else None,
                "objects": agent.snapshot(camera_id),
            }
        )
    return {
        "cameras": out,
        "note": "Eagles Eye reports association and stationarity only. "
                "Object contents are never inferred.",
    }


@router.get("/objects/{camera_id}/{object_track_id}/owner-trajectory")
def owner_trajectory(
    camera_id: str,
    object_track_id: int,
    request: Request,
    user: User = Depends(require("track:read")),
    db: Session = Depends(get_session),
):
    """Spec S13: where did the last associated subject go?"""
    points = get_orchestrator().objects.owner_trajectory(camera_id, object_track_id)
    record(db, request, user, "object.owner_trajectory", "tracked_object",
           f"{camera_id}:{object_track_id}")
    if not points:
        raise HTTPException(404, "no associated-subject trajectory is held for this object")
    return {
        "camera_id": camera_id,
        "object_track_id": object_track_id,
        "points": points,
        "point_count": len(points),
        "note": "Trajectory of the subject last associated with this object, by appearance "
                "and proximity. Not an identification.",
    }


# ------------------------------------------------------------------- threat
class ThreatWhatIf(BaseModel):
    behaviors: Dict[str, float]         # behaviour -> confidence
    zone_risk_weight: float = 1.0


@router.post("/threat/simulate")
def simulate_threat(body: ThreatWhatIf, user: User = Depends(require("policy:read"))):
    """Score a hypothetical signal combination against the live weights.

    Lets an operator see exactly what a weight change would do before saving it.
    """
    from ...agents.base import Finding

    now = utcnow()
    findings = [
        Finding(behavior=b, confidence=float(c), severity="medium", started_at=now, ended_at=now)
        for b, c in body.behaviors.items()
    ]
    assessment = ThreatAssessor().assess(
        findings, load_policy(), now=now, zone_risk_weight=body.zone_risk_weight
    )
    return assessment.to_dict()


@router.get("/threat/weights")
def threat_weights(user: User = Depends(require("policy:read"))):
    policy = load_policy()
    return {
        "weights": policy.get("threat_weights", {}),
        "bands": policy.get("threat_bands", {}),
        "zero_weight_behaviors": [
            "face_unavailable", "appearance_change", "cross_camera_association"
        ],
        "note": "Observability signals carry weight 0 by design: a covered face is "
                "ordinary behaviour and must never raise a risk score.",
    }


@router.get("/threat/ranked")
def ranked_threats(
    limit: int = 20,
    user: User = Depends(require("incident:read")),
    db: Session = Depends(get_session),
):
    rows = db.scalars(
        select(Incident)
        .where(Incident.status.in_(("open", "acknowledged", "dispatched", "investigating")))
        .order_by(Incident.risk_score.desc())
        .limit(limit)
    ).all()
    return {
        "items": [
            {
                "id": i.id,
                "title": i.title,
                "risk_score": i.risk_score,
                "severity": i.severity,
                "camera_id": i.camera_id,
                "zone_id": i.zone_id,
                "dominant_factor": (i.risk_factors or [{}])[0].get("behavior") if i.risk_factors else None,
                "explanation": i.explanation,
                "started_at": i.started_at.isoformat(),
            }
            for i in rows
        ]
    }


# --------------------------------------------------- identity escalation gate
class EscalationRequest(BaseModel):
    global_id: str
    justification: str
    case_reference: Optional[str] = None


@router.post("/identity/escalate")
def request_escalation(
    body: EscalationRequest,
    request: Request,
    user: User = Depends(require("identity:escalate")),
    db: Session = Depends(get_session),
):
    """Spec S27: identity is anonymous by default and escalates only via approval."""
    if len(body.justification.strip()) < 15:
        raise HTTPException(400, "a substantive justification is required for identity escalation")

    approval = ApprovalRequest(
        id=new_id("APR"),
        kind="identity_escalation",
        title=f"Identity escalation for subject {body.global_id}",
        detail=f"Case reference: {body.case_reference or 'none supplied'}",
        justification=body.justification,
        requested_by=user.username,
        required_role="commander",
        resource_type="global_subject",
        resource_id=body.global_id,
        payload={"case_reference": body.case_reference},
    )
    db.add(approval)
    db.commit()
    record(db, request, user, "identity.escalation_requested", "global_subject", body.global_id,
           justification=body.justification)
    return {
        "approval_id": approval.id,
        "status": "pending",
        "message": "Identity escalation requires a commander's approval before any "
                   "identity-bearing data is released.",
    }

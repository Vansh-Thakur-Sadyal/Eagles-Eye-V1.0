"""Incident endpoints: listing, detail, timeline, status and action approval."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...agents.commander import ACTION_CATALOGUE
from ...core.events import bus
from ...core.security import current_user, require
from ...db import get_session
from ...portal.scope import visible_camera_ids
from ...models import (
    Alert,
    ApprovalRequest,
    BehaviorEvent,
    Camera,
    Incident,
    IncidentTimelineEntry,
    User,
    new_id,
    utcnow,
)
from ..deps import Page, get_or_404, pagination, record, to_dict

router = APIRouter(prefix="/api/incidents", tags=["incidents"])

OPEN_STATES = ("open", "acknowledged", "dispatched", "investigating")


def _serialise(incident: Incident, *, timeline: bool = False) -> Dict[str, Any]:
    data = to_dict(incident)
    if timeline:
        data["timeline"] = [to_dict(e) for e in incident.timeline]
    return data


@router.get("")
def list_incidents(
    status_filter: Optional[str] = None,
    severity: Optional[str] = None,
    camera_id: Optional[str] = None,
    zone_id: Optional[str] = None,
    event_type: Optional[str] = None,
    since_minutes: Optional[int] = None,
    open_only: bool = False,
    page: Dict[str, int] = Depends(pagination),
    user: User = Depends(require("incident:read")),
    db: Session = Depends(get_session),
):
    stmt = select(Incident)
    if status_filter:
        stmt = stmt.where(Incident.status == status_filter)
    if open_only:
        stmt = stmt.where(Incident.status.in_(OPEN_STATES))
    if severity:
        stmt = stmt.where(Incident.severity == severity)
    if camera_id:
        stmt = stmt.where(Incident.camera_id == camera_id)
    if zone_id:
        stmt = stmt.where(Incident.zone_id == zone_id)
    if event_type:
        stmt = stmt.where(Incident.event_type == event_type)
    # Organisation scoping: an operator only sees incidents from their own
    # cameras. None means administrator - no restriction.
    allowed = visible_camera_ids(db, user)
    if allowed is not None:
        if not allowed:
            return Page.of([], 0, page["limit"], page["offset"])
        stmt = stmt.where(Incident.camera_id.in_(allowed))
    if since_minutes:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
        stmt = stmt.where(Incident.last_update_at >= cutoff)

    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(
        stmt.order_by(Incident.last_update_at.desc())
        .limit(page["limit"]).offset(page["offset"])
    ).all()
    return Page.of([_serialise(i) for i in rows], total, page["limit"], page["offset"])


@router.get("/summary")
def summary(
    since_hours: int = 24,
    user: User = Depends(require("incident:read")),
    db: Session = Depends(get_session),
):
    """KPI block for the command centre - all measured, nothing assumed."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

    by_severity = dict(
        db.execute(
            select(Incident.severity, func.count())
            .where(Incident.status.in_(OPEN_STATES))
            .group_by(Incident.severity)
        ).all()
    )
    by_type = dict(
        db.execute(
            select(Incident.event_type, func.count())
            .where(Incident.started_at >= cutoff)
            .group_by(Incident.event_type)
            .order_by(func.count().desc())
        ).all()
    )
    by_status = dict(
        db.execute(select(Incident.status, func.count()).group_by(Incident.status)).all()
    )

    open_count = db.scalar(
        select(func.count()).select_from(Incident).where(Incident.status.in_(OPEN_STATES))
    ) or 0
    recent = db.scalar(
        select(func.count()).select_from(Incident).where(Incident.started_at >= cutoff)
    ) or 0
    avg_risk = db.scalar(
        select(func.avg(Incident.risk_score)).where(Incident.status.in_(OPEN_STATES))
    )
    resolved = db.scalar(
        select(func.count()).select_from(Incident)
        .where(Incident.status == "resolved", Incident.resolved_at >= cutoff)
    ) or 0

    # Mean time to acknowledge, over incidents acknowledged in the window.
    ack_rows = db.execute(
        select(Incident.started_at, Incident.acknowledged_at)
        .where(Incident.acknowledged_at.is_not(None), Incident.acknowledged_at >= cutoff)
    ).all()
    mtta = None
    if ack_rows:
        deltas = [(a - s).total_seconds() for s, a in ack_rows if a and s]
        if deltas:
            mtta = round(sum(deltas) / len(deltas), 1)

    return {
        "window_hours": since_hours,
        "open": open_count,
        "opened_in_window": recent,
        "resolved_in_window": resolved,
        "by_severity": by_severity,
        "by_event_type": by_type,
        "by_status": by_status,
        "average_open_risk": round(float(avg_risk), 1) if avg_risk is not None else 0.0,
        "mean_time_to_acknowledge_seconds": mtta,
    }


@router.get("/live")
def live_incidents(user: User = Depends(require("incident:read"))):
    """In-memory view straight from the orchestrator, ahead of any DB write."""
    from ...agents.orchestrator import get_orchestrator

    return {"items": get_orchestrator().open_incidents()}


@router.get("/{incident_id}")
def get_incident(
    incident_id: str,
    request: Request,
    user: User = Depends(require("incident:read")),
    db: Session = Depends(get_session),
):
    incident = get_or_404(db, Incident, incident_id, "incident")
    allowed = visible_camera_ids(db, user)
    if allowed is not None and incident.camera_id and incident.camera_id not in allowed:
        raise HTTPException(404, "incident not found")
    record(db, request, user, "incident.read", "incident", incident_id)

    camera = db.get(Camera, incident.camera_id) if incident.camera_id else None
    events = db.scalars(
        select(BehaviorEvent)
        .where(BehaviorEvent.camera_id == incident.camera_id)
        .where(BehaviorEvent.started_at >= incident.started_at)
        .where(BehaviorEvent.started_at <= incident.last_update_at)
        .order_by(BehaviorEvent.started_at)
    ).all()

    return {
        **_serialise(incident, timeline=True),
        "camera": to_dict(camera, exclude={"password"}) if camera else None,
        "behavior_events": [to_dict(e) for e in events],
        "action_catalogue": ACTION_CATALOGUE,
    }


@router.get("/{incident_id}/timeline")
def incident_timeline(
    incident_id: str,
    user: User = Depends(require("incident:read")),
    db: Session = Depends(get_session),
):
    get_or_404(db, Incident, incident_id, "incident")
    rows = db.scalars(
        select(IncidentTimelineEntry)
        .where(IncidentTimelineEntry.incident_id == incident_id)
        .order_by(IncidentTimelineEntry.at)
    ).all()
    return {"incident_id": incident_id, "entries": [to_dict(e) for e in rows]}


class StatusUpdate(BaseModel):
    status: str
    note: Optional[str] = None
    assigned_to: Optional[str] = None


VALID_STATUSES = ("open", "acknowledged", "dispatched", "investigating", "resolved", "false_positive")


@router.post("/{incident_id}/status")
def update_status(
    incident_id: str,
    body: StatusUpdate,
    request: Request,
    user: User = Depends(require("incident:write")),
    db: Session = Depends(get_session),
):
    if body.status not in VALID_STATUSES:
        raise HTTPException(400, f"status must be one of {VALID_STATUSES}")
    incident = get_or_404(db, Incident, incident_id, "incident")
    previous = incident.status

    incident.status = body.status
    if body.assigned_to:
        incident.assigned_to = body.assigned_to
    now = utcnow()
    if body.status == "acknowledged" and incident.acknowledged_at is None:
        incident.acknowledged_at = now
    if body.status in ("resolved", "false_positive"):
        incident.resolved_at = now
        incident.resolution_note = body.note

    db.add(
        IncidentTimelineEntry(
            id=new_id("TLE"),
            incident_id=incident_id,
            at=now,
            kind="operator",
            actor=user.username,
            text=f"Status changed from {previous} to {body.status}."
                 + (f" Note: {body.note}" if body.note else ""),
            payload={"from": previous, "to": body.status},
        )
    )
    db.commit()
    record(db, request, user, "incident.status", "incident", incident_id,
           justification=body.note, detail={"from": previous, "to": body.status})
    bus.publish("incident.status", {"incident_id": incident_id, "status": body.status,
                                    "actor": user.username}, source="api")
    return _serialise(incident, timeline=True)


class ActionRequest(BaseModel):
    action_key: str
    justification: Optional[str] = None


@router.post("/{incident_id}/actions")
def approve_action(
    incident_id: str,
    body: ActionRequest,
    request: Request,
    user: User = Depends(require("incident:write")),
    db: Session = Depends(get_session),
):
    """Approve one recommended action.

    Actions the commander marked as requiring approval are not applied here:
    they create an ApprovalRequest for a commander to decide, which is what
    keeps a high-impact response a human decision.
    """
    incident = get_or_404(db, Incident, incident_id, "incident")
    meta = ACTION_CATALOGUE.get(body.action_key)
    if meta is None:
        raise HTTPException(400, f"unknown action '{body.action_key}'")

    proposed = {a["key"] for a in (incident.recommended_actions or [])}
    if body.action_key not in proposed:
        raise HTTPException(400, f"'{body.action_key}' was not recommended for this incident")

    now = utcnow()
    needs_approval = meta["requires_human_approval"]
    from ...core.security import has_permission

    can_decide = has_permission(user.role, "approval:decide")

    if needs_approval and not can_decide:
        approval = ApprovalRequest(
            id=new_id("APR"),
            kind="dispatch",
            title=f"{meta['label']} for {incident.title}",
            detail=incident.summary,
            justification=body.justification,
            requested_by=user.username,
            required_role="commander",
            resource_type="incident",
            resource_id=incident_id,
            payload={"action_key": body.action_key, "consequence": meta["consequence"]},
        )
        db.add(approval)
        status_text = "pending_approval"
    else:
        status_text = "approved"

    actions = list(incident.recommended_actions or [])
    for action in actions:
        if action["key"] == body.action_key:
            action["status"] = status_text
            action["actor"] = user.username
            action["decided_at"] = now.isoformat()
    incident.recommended_actions = actions

    db.add(
        IncidentTimelineEntry(
            id=new_id("TLE"),
            incident_id=incident_id,
            at=now,
            kind="action",
            actor=user.username,
            text=f"{meta['label']}: {status_text.replace('_', ' ')}."
                 + (f" Justification: {body.justification}" if body.justification else ""),
            payload={"action_key": body.action_key, "consequence": meta["consequence"]},
        )
    )
    db.commit()
    record(db, request, user, "incident.action", "incident", incident_id,
           justification=body.justification,
           detail={"action": body.action_key, "result": status_text})
    bus.publish("incident.action", {"incident_id": incident_id, "action": body.action_key,
                                    "status": status_text, "actor": user.username}, source="api")
    return {"incident_id": incident_id, "action": body.action_key, "status": status_text,
            "requires_human_approval": needs_approval}


class NoteBody(BaseModel):
    text: str


@router.post("/{incident_id}/notes")
def add_note(
    incident_id: str,
    body: NoteBody,
    request: Request,
    user: User = Depends(require("incident:write")),
    db: Session = Depends(get_session),
):
    get_or_404(db, Incident, incident_id, "incident")
    entry = IncidentTimelineEntry(
        id=new_id("TLE"), incident_id=incident_id, at=utcnow(),
        kind="operator", actor=user.username, text=body.text,
    )
    db.add(entry)
    db.commit()
    record(db, request, user, "incident.note", "incident", incident_id)
    return to_dict(entry)


# ------------------------------------------------------------------ alerts
alerts_router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("/{incident_id}/alerts")
def incident_alerts(incident_id: str, user: User = Depends(require("incident:read")),
                    db: Session = Depends(get_session)):
    rows = db.scalars(select(Alert).where(Alert.incident_id == incident_id)).all()
    return {"items": [to_dict(a) for a in rows]}


@alerts_router.get("")
def list_alerts(
    unacknowledged_only: bool = False,
    severity: Optional[str] = None,
    page: Dict[str, int] = Depends(pagination),
    user: User = Depends(require("incident:read")),
    db: Session = Depends(get_session),
):
    stmt = select(Alert)
    if unacknowledged_only:
        stmt = stmt.where(Alert.acknowledged.is_(False))
    if severity:
        stmt = stmt.where(Alert.severity == severity)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(
        stmt.order_by(Alert.created_at.desc()).limit(page["limit"]).offset(page["offset"])
    ).all()
    return Page.of([to_dict(a) for a in rows], total, page["limit"], page["offset"])


@alerts_router.post("/{alert_id}/acknowledge")
def acknowledge_alert(
    alert_id: str,
    request: Request,
    user: User = Depends(require("incident:write")),
    db: Session = Depends(get_session),
):
    alert = get_or_404(db, Alert, alert_id, "alert")
    alert.acknowledged = True
    alert.acknowledged_by = user.username
    db.commit()
    record(db, request, user, "alert.acknowledge", "alert", alert_id)
    return to_dict(alert)


@alerts_router.post("/acknowledge-all")
def acknowledge_all(
    request: Request,
    user: User = Depends(require("incident:write")),
    db: Session = Depends(get_session),
):
    rows = db.scalars(select(Alert).where(Alert.acknowledged.is_(False))).all()
    for alert in rows:
        alert.acknowledged = True
        alert.acknowledged_by = user.username
    db.commit()
    record(db, request, user, "alert.acknowledge_all", "alert", None, detail={"count": len(rows)})
    return {"acknowledged": len(rows)}

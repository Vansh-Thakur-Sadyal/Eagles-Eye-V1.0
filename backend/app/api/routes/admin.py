"""Governance, automation, cases, evidence and reporting."""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...automation.n8n import TRIGGERS, dispatcher
from ...config import get_settings
from ...core.events import bus
from ...core.security import current_user, require
from ...db import get_session
from ...models import (
    ApprovalRequest,
    AuditLog,
    BehaviorEvent,
    Camera,
    Case,
    Evidence,
    Incident,
    Report,
    User,
    WatchlistSubject,
    Workflow,
    WorkflowRun,
    new_id,
    utcnow,
)
from ..deps import Page, get_or_404, pagination, record, to_dict

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ------------------------------------------------------------------- audit
@router.get("/audit")
def audit_log(
    actor: Optional[str] = None,
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    since_hours: int = 168,
    page: Dict[str, int] = Depends(pagination),
    user: User = Depends(require("audit:read")),
    db: Session = Depends(get_session),
):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    stmt = select(AuditLog).where(AuditLog.at >= cutoff)
    if actor:
        stmt = stmt.where(AuditLog.actor == actor)
    if action:
        stmt = stmt.where(AuditLog.action.like(f"{action}%"))
    if resource_type:
        stmt = stmt.where(AuditLog.resource_type == resource_type)
    if resource_id:
        stmt = stmt.where(AuditLog.resource_id == resource_id)

    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(
        stmt.order_by(AuditLog.at.desc()).limit(page["limit"]).offset(page["offset"])
    ).all()
    return Page.of([to_dict(r) for r in rows], total, page["limit"], page["offset"])


@router.get("/audit/summary")
def audit_summary(
    since_hours: int = 24,
    user: User = Depends(require("audit:read")),
    db: Session = Depends(get_session),
):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    by_action = db.execute(
        select(AuditLog.action, func.count()).where(AuditLog.at >= cutoff)
        .group_by(AuditLog.action).order_by(func.count().desc()).limit(25)
    ).all()
    by_actor = db.execute(
        select(AuditLog.actor, func.count()).where(AuditLog.at >= cutoff)
        .group_by(AuditLog.actor).order_by(func.count().desc()).limit(25)
    ).all()
    failures = db.scalar(
        select(func.count()).select_from(AuditLog)
        .where(AuditLog.at >= cutoff, AuditLog.outcome != "success")
    ) or 0
    identity_reads = db.scalar(
        select(func.count()).select_from(AuditLog)
        .where(AuditLog.at >= cutoff,
               AuditLog.action.in_(("watchlist.locate", "watchlist.sightings",
                                    "subject.trajectory", "identity.escalation_requested")))
    ) or 0
    return {
        "window_hours": since_hours,
        "by_action": [{"action": a, "count": c} for a, c in by_action],
        "by_actor": [{"actor": a, "count": c} for a, c in by_actor],
        "failed_actions": failures,
        "identity_sensitive_reads": identity_reads,
    }


# ---------------------------------------------------------------- privacy
@router.get("/privacy")
def privacy_status(
    user: User = Depends(require("system:read")),
    db: Session = Depends(get_session),
):
    from ...agents.orchestrator import get_orchestrator

    s = get_settings()
    agent = get_orchestrator().privacy
    active_watchlist = db.scalar(
        select(func.count()).select_from(WatchlistSubject)
        .where(WatchlistSubject.status == "active")
    ) or 0
    pending = db.scalar(
        select(func.count()).select_from(ApprovalRequest)
        .where(ApprovalRequest.status == "pending")
    ) or 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=s.retention_days)
    expiring = db.scalar(
        select(func.count()).select_from(Incident).where(Incident.started_at < cutoff)
    ) or 0

    return {
        "settings": {
            "bystander_redaction": s.privacy_redaction,
            "anonymous_by_default": s.privacy_anonymous_by_default,
            "retention_days": s.retention_days,
            "audit_all_reads": s.audit_all_reads,
        },
        "runtime": agent.status(),
        "active_watchlist_subjects": active_watchlist,
        "pending_approvals": pending,
        "records_past_retention": expiring,
        "controls": [
            {"control": "Anonymous-by-default tracking", "enforced": s.privacy_anonymous_by_default},
            {"control": "Bystander face redaction in outgoing video", "enforced": s.privacy_redaction},
            {"control": "Role-based access control", "enforced": True},
            {"control": "Immutable audit log", "enforced": True},
            {"control": "Identity escalation requires approval", "enforced": True},
            {"control": "Watchlist enrolment requires a legal basis", "enforced": True},
            {"control": "Watchlist enrolment requires commander approval", "enforced": True},
            {"control": "Retention policy", "enforced": s.retention_days > 0},
            {"control": "Model confidence displayed with every association", "enforced": True},
        ],
    }


@router.post("/privacy/purge-expired")
def purge_expired(
    request: Request,
    dry_run: bool = True,
    user: User = Depends(require("*")),
    db: Session = Depends(get_session),
):
    """Apply the retention policy. Defaults to a dry run so nothing is lost by accident."""
    s = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(days=s.retention_days)

    incidents = db.scalars(select(Incident).where(Incident.started_at < cutoff)).all()
    events = db.scalars(select(BehaviorEvent).where(BehaviorEvent.started_at < cutoff)).all()
    sealed_ids = {
        e.incident_id for e in db.scalars(select(Evidence).where(Evidence.sealed.is_(True))).all()
    }
    deletable = [i for i in incidents if i.id not in sealed_ids]

    summary = {
        "cutoff": cutoff.isoformat(),
        "retention_days": s.retention_days,
        "incidents_past_retention": len(incidents),
        "incidents_held_by_sealed_evidence": len(incidents) - len(deletable),
        "behavior_events_past_retention": len(events),
        "dry_run": dry_run,
    }

    if not dry_run:
        for row in deletable:
            db.delete(row)
        for row in events:
            db.delete(row)
        db.commit()
        summary["deleted_incidents"] = len(deletable)
        summary["deleted_events"] = len(events)

    record(db, request, user, "privacy.purge", "retention", None, detail=summary)
    return summary


# -------------------------------------------------------------- approvals
@router.get("/approvals")
def list_approvals(
    status_filter: str = "pending",
    kind: Optional[str] = None,
    page: Dict[str, int] = Depends(pagination),
    user: User = Depends(current_user),
    db: Session = Depends(get_session),
):
    stmt = select(ApprovalRequest)
    if status_filter != "all":
        stmt = stmt.where(ApprovalRequest.status == status_filter)
    if kind:
        stmt = stmt.where(ApprovalRequest.kind == kind)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(
        stmt.order_by(ApprovalRequest.created_at.desc())
        .limit(page["limit"]).offset(page["offset"])
    ).all()
    return Page.of([to_dict(r) for r in rows], total, page["limit"], page["offset"])


class Decision(BaseModel):
    decision: str            # approve | reject
    note: Optional[str] = None


@router.post("/approvals/{approval_id}/decide")
def decide(
    approval_id: str,
    body: Decision,
    request: Request,
    user: User = Depends(require("approval:decide")),
    db: Session = Depends(get_session),
):
    approval = get_or_404(db, ApprovalRequest, approval_id, "approval request")
    if approval.status != "pending":
        raise HTTPException(409, f"this request was already {approval.status}")
    if body.decision not in ("approve", "reject"):
        raise HTTPException(400, "decision must be 'approve' or 'reject'")
    if approval.requested_by == user.username:
        raise HTTPException(403, "you cannot approve your own request")

    approval.status = "approved" if body.decision == "approve" else "rejected"
    approval.decided_by = user.username
    approval.decided_at = utcnow()
    approval.decision_note = body.note

    # Approving a watchlist enrolment activates it.
    if approval.kind == "watchlist_enrol" and approval.status == "approved":
        subject = db.get(WatchlistSubject, approval.resource_id)
        if subject:
            subject.status = "active"
            subject.approved_by = user.username
            subject.approved_at = utcnow()

    db.commit()
    if approval.kind == "watchlist_enrol":
        from .watchlist import reload_gallery

        reload_gallery(db)

    record(db, request, user, "approval.decide", approval.kind, approval_id,
           justification=body.note, detail={"decision": body.decision})
    bus.publish("approval.decided", {"approval_id": approval_id, "kind": approval.kind,
                                     "decision": body.decision, "actor": user.username},
                source="api")
    return to_dict(approval)


# -------------------------------------------------------------- workflows
class WorkflowIn(BaseModel):
    name: str
    trigger: str
    webhook_url: str
    description: Optional[str] = None
    method: str = "POST"
    headers: Dict[str, Any] = Field(default_factory=dict)
    conditions: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


@router.get("/workflows")
def list_workflows(user: User = Depends(require("workflow:read")),
                   db: Session = Depends(get_session)):
    rows = db.scalars(select(Workflow).order_by(Workflow.name)).all()
    return {
        "items": [to_dict(w) for w in rows],
        "available_triggers": TRIGGERS,
        "dispatcher": dispatcher.status(),
        "condition_help": {
            "min_severity": "info | low | medium | high | critical",
            "min_risk_score": "number 0-100",
            "camera_ids": "list of camera IDs",
            "zone_ids": "list of zone IDs",
            "behaviors": "list of behaviour names; matches if any overlap",
        },
    }


@router.post("/workflows")
def create_workflow(
    body: WorkflowIn,
    request: Request,
    user: User = Depends(require("workflow:write")),
    db: Session = Depends(get_session),
):
    if body.trigger not in TRIGGERS:
        raise HTTPException(400, f"trigger must be one of {TRIGGERS}")
    workflow = Workflow(id=new_id("WKF"), **body.model_dump())
    db.add(workflow)
    db.commit()
    record(db, request, user, "workflow.create", "workflow", workflow.id,
           detail={"name": body.name, "trigger": body.trigger})
    return to_dict(workflow)


@router.patch("/workflows/{workflow_id}")
def update_workflow(
    workflow_id: str,
    body: Dict[str, Any],
    request: Request,
    user: User = Depends(require("workflow:write")),
    db: Session = Depends(get_session),
):
    workflow = get_or_404(db, Workflow, workflow_id, "workflow")
    if "trigger" in body and body["trigger"] not in TRIGGERS:
        raise HTTPException(400, f"trigger must be one of {TRIGGERS}")
    for key, value in body.items():
        if hasattr(workflow, key) and key not in ("id", "created_at", "run_count"):
            setattr(workflow, key, value)
    db.commit()
    record(db, request, user, "workflow.update", "workflow", workflow_id)
    return to_dict(workflow)


@router.delete("/workflows/{workflow_id}")
def delete_workflow(
    workflow_id: str,
    request: Request,
    user: User = Depends(require("workflow:write")),
    db: Session = Depends(get_session),
):
    workflow = get_or_404(db, Workflow, workflow_id, "workflow")
    db.delete(workflow)
    db.commit()
    record(db, request, user, "workflow.delete", "workflow", workflow_id)
    return {"deleted": workflow_id}


@router.post("/workflows/{workflow_id}/test")
def test_workflow(
    workflow_id: str,
    user: User = Depends(require("workflow:write")),
    db: Session = Depends(get_session),
):
    workflow = get_or_404(db, Workflow, workflow_id, "workflow")
    return dispatcher.test(workflow.webhook_url, {"workflow": workflow.name, "test": True})


@router.get("/workflows/{workflow_id}/runs")
def workflow_runs(workflow_id: str, limit: int = 50,
                  user: User = Depends(require("workflow:read")),
                  db: Session = Depends(get_session)):
    rows = db.scalars(
        select(WorkflowRun).where(WorkflowRun.workflow_id == workflow_id)
        .order_by(WorkflowRun.at.desc()).limit(limit)
    ).all()
    return {"items": [to_dict(r) for r in rows]}


# ------------------------------------------------------------------- cases
class CaseIn(BaseModel):
    title: str
    reference: Optional[str] = None
    description: Optional[str] = None
    priority: str = "medium"
    lead: Optional[str] = None
    incident_ids: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


@router.get("/cases")
def list_cases(
    status_filter: Optional[str] = None,
    page: Dict[str, int] = Depends(pagination),
    user: User = Depends(require("case:read")),
    db: Session = Depends(get_session),
):
    stmt = select(Case)
    if status_filter:
        stmt = stmt.where(Case.status == status_filter)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(
        stmt.order_by(Case.updated_at.desc()).limit(page["limit"]).offset(page["offset"])
    ).all()
    return Page.of([to_dict(c) for c in rows], total, page["limit"], page["offset"])


@router.post("/cases")
def create_case(
    body: CaseIn,
    request: Request,
    user: User = Depends(require("case:write")),
    db: Session = Depends(get_session),
):
    case = Case(id=new_id("CASE"), lead=body.lead or user.username,
                **body.model_dump(exclude={"lead"}))
    db.add(case)
    db.commit()
    record(db, request, user, "case.create", "case", case.id, detail={"title": body.title})
    return to_dict(case)


@router.get("/cases/{case_id}")
def get_case(case_id: str, user: User = Depends(require("case:read")),
             db: Session = Depends(get_session)):
    case = get_or_404(db, Case, case_id, "case")
    incidents = db.scalars(
        select(Incident).where(Incident.id.in_(case.incident_ids or []))
    ).all()
    evidence = db.scalars(select(Evidence).where(Evidence.case_id == case_id)).all()
    return {
        **to_dict(case),
        "incidents": [to_dict(i) for i in incidents],
        "evidence": [to_dict(e) for e in evidence],
    }


@router.patch("/cases/{case_id}")
def update_case(
    case_id: str,
    body: Dict[str, Any],
    request: Request,
    user: User = Depends(require("case:write")),
    db: Session = Depends(get_session),
):
    case = get_or_404(db, Case, case_id, "case")
    for key, value in body.items():
        if hasattr(case, key) and key not in ("id", "created_at"):
            setattr(case, key, value)
    if body.get("status") == "closed":
        case.closed_at = utcnow()
    db.commit()
    record(db, request, user, "case.update", "case", case_id)
    return to_dict(case)


# ---------------------------------------------------------------- evidence
class EvidenceIn(BaseModel):
    incident_id: Optional[str] = None
    case_id: Optional[str] = None
    kind: str = "snapshot"
    file_path: Optional[str] = None
    note: Optional[str] = None


@router.get("/evidence")
def list_evidence(
    case_id: Optional[str] = None,
    incident_id: Optional[str] = None,
    page: Dict[str, int] = Depends(pagination),
    user: User = Depends(require("evidence:read")),
    db: Session = Depends(get_session),
):
    stmt = select(Evidence)
    if case_id:
        stmt = stmt.where(Evidence.case_id == case_id)
    if incident_id:
        stmt = stmt.where(Evidence.incident_id == incident_id)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(
        stmt.order_by(Evidence.created_at.desc()).limit(page["limit"]).offset(page["offset"])
    ).all()
    return Page.of([to_dict(e) for e in rows], total, page["limit"], page["offset"])


@router.post("/evidence")
def collect_evidence(
    body: EvidenceIn,
    request: Request,
    user: User = Depends(require("evidence:write")),
    db: Session = Depends(get_session),
):
    """Capture evidence with a hash and an opening chain-of-custody entry."""
    s = get_settings()
    file_path = body.file_path
    sha256 = None
    size = 0

    if body.incident_id and not file_path:
        # Grab the live frame from the incident's camera.
        from ...pipeline.runner import get_pipeline

        incident = get_or_404(db, Incident, body.incident_id, "incident")
        worker = get_pipeline().get(incident.camera_id) if incident.camera_id else None
        jpeg = worker.latest_jpeg() if worker else None
        if jpeg:
            folder = s.storage_dir / "evidence" / body.incident_id
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{new_id('EVD')}.jpg"
            path.write_bytes(jpeg)
            file_path = str(path)

    if file_path and Path(file_path).exists():
        data = Path(file_path).read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        size = len(data)

    evidence = Evidence(
        id=new_id("EVD"),
        case_id=body.case_id,
        incident_id=body.incident_id,
        kind=body.kind,
        file_path=file_path,
        sha256=sha256,
        size_bytes=size,
        collected_by=user.username,
        note=body.note,
        retention_until=utcnow() + timedelta(days=s.retention_days),
        chain_of_custody=[
            {
                "at": utcnow().isoformat(),
                "actor": user.username,
                "role": user.role,
                "action": "collected",
                "sha256": sha256,
            }
        ],
    )
    db.add(evidence)
    db.commit()
    record(db, request, user, "evidence.collect", "evidence", evidence.id,
           detail={"kind": body.kind, "sha256": sha256})
    return to_dict(evidence)


@router.post("/evidence/{evidence_id}/seal")
def seal_evidence(
    evidence_id: str,
    request: Request,
    user: User = Depends(require("evidence:write")),
    db: Session = Depends(get_session),
):
    """Sealing exempts an item from the retention purge."""
    evidence = get_or_404(db, Evidence, evidence_id, "evidence")
    evidence.sealed = True
    evidence.chain_of_custody = (evidence.chain_of_custody or []) + [
        {"at": utcnow().isoformat(), "actor": user.username, "role": user.role, "action": "sealed"}
    ]
    db.commit()
    record(db, request, user, "evidence.seal", "evidence", evidence_id)
    return to_dict(evidence)


@router.get("/evidence/{evidence_id}/verify")
def verify_evidence(evidence_id: str, user: User = Depends(require("evidence:read")),
                    db: Session = Depends(get_session)):
    """Recompute the hash and report whether the file still matches."""
    evidence = get_or_404(db, Evidence, evidence_id, "evidence")
    if not evidence.file_path or not Path(evidence.file_path).exists():
        return {"verified": False, "reason": "the evidence file is missing from storage"}
    current = hashlib.sha256(Path(evidence.file_path).read_bytes()).hexdigest()
    return {
        "verified": current == evidence.sha256,
        "recorded_sha256": evidence.sha256,
        "current_sha256": current,
        "chain_of_custody": evidence.chain_of_custody,
    }


# ----------------------------------------------------------------- reports
class ReportRequest(BaseModel):
    title: Optional[str] = None
    report_type: str = "daily"
    hours: int = 24
    site_id: Optional[str] = None


@router.post("/reports/generate")
def generate_report(
    body: ReportRequest,
    request: Request,
    user: User = Depends(require("report:write")),
    db: Session = Depends(get_session),
):
    """Build a report from real recorded data, narrated by the LLM when available."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=body.hours)

    stmt = select(Incident).where(Incident.started_at >= start)
    if body.site_id:
        stmt = stmt.where(Incident.site_id == body.site_id)
    incidents = db.scalars(stmt.order_by(Incident.risk_score.desc())).all()

    by_severity: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    for i in incidents:
        by_severity[i.severity] = by_severity.get(i.severity, 0) + 1
        by_type[i.event_type] = by_type.get(i.event_type, 0) + 1

    cameras_total = db.scalar(select(func.count()).select_from(Camera)) or 0
    cameras_online = db.scalar(
        select(func.count()).select_from(Camera).where(Camera.status == "online")
    ) or 0
    resolved = sum(1 for i in incidents if i.status in ("resolved", "false_positive"))

    stats = {
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "incidents_total": len(incidents),
        "incidents_resolved": resolved,
        "by_severity": by_severity,
        "by_event_type": by_type,
        "cameras_total": cameras_total,
        "cameras_online": cameras_online,
        "highest_risk": round(max([i.risk_score for i in incidents], default=0.0), 1),
    }

    body_md = _report_markdown(body, stats, incidents)
    report = Report(
        id=new_id("RPT"),
        title=body.title or f"{body.report_type.title()} security report",
        report_type=body.report_type,
        period_start=start,
        period_end=end,
        generated_by=user.username,
        body_markdown=body_md,
        incident_ids=[i.id for i in incidents[:100]],
        stats=stats,
    )
    db.add(report)
    db.commit()
    record(db, request, user, "report.generate", "report", report.id,
           detail={"type": body.report_type, "incidents": len(incidents)})
    dispatcher.emit("daily_report", {"report_id": report.id, "stats": stats})
    return to_dict(report)


def _report_markdown(req: ReportRequest, stats: Dict[str, Any], incidents: List[Incident]) -> str:
    from ...rag.llm import LLMUnavailable, get_llm

    llm = get_llm()
    facts = {
        "statistics": stats,
        "top_incidents": [
            {
                "id": i.id, "title": i.title, "severity": i.severity,
                "risk_score": i.risk_score, "summary": i.summary,
                "camera_id": i.camera_id, "status": i.status,
            }
            for i in incidents[:12]
        ],
    }
    if llm.enabled:
        try:
            import json

            text = llm.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You write factual security operations reports. Use only the supplied "
                            "data. Never assert a person's identity, never infer container contents, "
                            "never accuse anyone. Output markdown with clear sections."
                        ),
                    },
                    {"role": "user", "content": json.dumps(facts, indent=2, default=str)},
                ],
                temperature=0.2,
                max_tokens=1600,
            )
            if text.strip():
                return text.strip()
        except LLMUnavailable:
            pass
        except Exception:
            pass

    lines = [
        f"# {req.report_type.title()} security report",
        "",
        f"**Period:** {stats['period_start']} to {stats['period_end']}",
        "",
        "## Summary",
        f"- Incidents raised: **{stats['incidents_total']}**",
        f"- Incidents resolved or dismissed: **{stats['incidents_resolved']}**",
        f"- Highest assessed risk: **{stats['highest_risk']}%**",
        f"- Cameras online: **{stats['cameras_online']} / {stats['cameras_total']}**",
        "",
        "## By severity",
    ]
    lines += [f"- {k}: {v}" for k, v in sorted(stats["by_severity"].items())] or ["- none"]
    lines += ["", "## By event type"]
    lines += [f"- {k.replace('_', ' ')}: {v}" for k, v in sorted(
        stats["by_event_type"].items(), key=lambda kv: -kv[1])] or ["- none"]
    lines += ["", "## Highest-risk incidents", ""]
    if incidents:
        lines.append("| ID | Title | Severity | Risk | Status |")
        lines.append("|---|---|---|---|---|")
        for i in incidents[:12]:
            lines.append(
                f"| {i.id} | {i.title} | {i.severity} | {i.risk_score:.0f}% | {i.status} |"
            )
    else:
        lines.append("No incidents were recorded in this period.")
    lines += [
        "", "---",
        "_Generated by Eagles Eye from recorded observations. All associations are "
        "confidence-scored and require human confirmation._",
    ]
    return "\n".join(lines)


@router.get("/reports")
def list_reports(
    page: Dict[str, int] = Depends(pagination),
    user: User = Depends(require("report:read")),
    db: Session = Depends(get_session),
):
    total = db.scalar(select(func.count()).select_from(Report)) or 0
    rows = db.scalars(
        select(Report).order_by(Report.created_at.desc())
        .limit(page["limit"]).offset(page["offset"])
    ).all()
    return Page.of([to_dict(r, exclude={"body_markdown"}) for r in rows],
                   total, page["limit"], page["offset"])


@router.get("/reports/{report_id}")
def get_report(report_id: str, user: User = Depends(require("report:read")),
               db: Session = Depends(get_session)):
    return to_dict(get_or_404(db, Report, report_id, "report"))

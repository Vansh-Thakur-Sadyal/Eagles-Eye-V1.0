"""Administrator-only panel and the monitoring feed.

Nobody but the platform administrator can reach any of this - not other
organisations, not their operators. It answers: who registered, who is signed
in right now, what they did, and lets an account be suspended.
"""
from __future__ import annotations

from datetime import timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...core.security import current_user
from ...db import get_session
from ...models import User
from ...portal.models import Organisation, PortalAccount, PortalActivity
from ...portal.service import utcnow

router = APIRouter(prefix="/api/portal/admin", tags=["portal-admin"])

ONLINE_WINDOW_MINUTES = 15


def _aware(dt):
    """SQLite hands back naive datetimes; compare everything in UTC."""
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def admin_only(user: User = Depends(current_user)) -> User:
    """Platform administrator only. An organisation operator gets a flat 404.

    404 rather than 403 on purpose: an operator should not even learn that an
    admin panel exists at this address.
    """
    if user.role != "admin":
        raise HTTPException(404, "Not found")
    return user


def _account_row(a: PortalAccount, org: Optional[Organisation]) -> Dict[str, Any]:
    online_since = utcnow() - timedelta(minutes=ONLINE_WINDOW_MINUTES)
    last_seen = _aware(a.last_seen_at)
    return {
        "id": a.id, "email": a.email, "name": a.full_name, "phone": a.phone,
        "role": a.role, "status": a.status,
        "organisation": {"id": org.id, "name": org.name, "account_type": org.account_type}
        if org else None,
        "verified_at": a.verified_at, "last_login_at": a.last_login_at,
        "last_seen_at": a.last_seen_at, "login_count": a.login_count,
        "online": bool(last_seen and last_seen >= online_since),
        "terms": {"version": a.terms_version, "accepted_at": a.terms_accepted_at,
                  "ip": a.terms_accepted_ip},
        "signup_ip": a.signup_ip, "created_at": a.created_at,
    }


@router.get("/overview")
def overview(user: User = Depends(admin_only), db: Session = Depends(get_session)):
    since = utcnow() - timedelta(minutes=ONLINE_WINDOW_MINUTES)
    day = utcnow() - timedelta(days=1)
    accounts = db.scalars(select(PortalAccount)).all()
    orgs = {o.id: o for o in db.scalars(select(Organisation)).all()}
    online = [a for a in accounts if _aware(a.last_seen_at) and _aware(a.last_seen_at) >= since]
    recent = db.scalars(select(PortalActivity).order_by(PortalActivity.at.desc()).limit(25)).all()
    by_type: Dict[str, int] = {}
    for o in orgs.values():
        by_type[o.account_type] = by_type.get(o.account_type, 0) + 1
    return {
        "accounts": {
            "total": len(accounts),
            "active": sum(1 for a in accounts if a.status == "active"),
            "pending": sum(1 for a in accounts if a.status == "pending"),
            "suspended": sum(1 for a in accounts if a.status == "suspended"),
            "online_now": len(online),
        },
        "organisations": {"total": len(orgs), "by_type": by_type},
        "last_24h": {
            "signups": db.scalar(select(func.count()).select_from(PortalActivity).where(
                PortalActivity.event == "signup_started", PortalActivity.at >= day)) or 0,
            "logins": db.scalar(select(func.count()).select_from(PortalActivity).where(
                PortalActivity.event == "login_ok", PortalActivity.at >= day)) or 0,
            "failed_logins": db.scalar(select(func.count()).select_from(PortalActivity).where(
                PortalActivity.event == "login_failed", PortalActivity.at >= day)) or 0,
            "blocked": db.scalar(select(func.count()).select_from(PortalActivity).where(
                PortalActivity.outcome == "blocked", PortalActivity.at >= day)) or 0,
        },
        "online": [_account_row(a, orgs.get(a.organisation_id)) for a in online],
        "recent_activity": [
            {"at": e.at, "event": e.event, "outcome": e.outcome, "email": e.email,
             "ip": e.ip, "detail": e.detail} for e in recent
        ],
    }


@router.get("/accounts")
def accounts(status_filter: Optional[str] = None, organisation_id: Optional[str] = None,
             user: User = Depends(admin_only), db: Session = Depends(get_session)):
    stmt = select(PortalAccount).order_by(PortalAccount.created_at.desc())
    if status_filter:
        stmt = stmt.where(PortalAccount.status == status_filter)
    if organisation_id:
        stmt = stmt.where(PortalAccount.organisation_id == organisation_id)
    rows = db.scalars(stmt).all()
    orgs = {o.id: o for o in db.scalars(select(Organisation)).all()}
    return {"items": [_account_row(a, orgs.get(a.organisation_id)) for a in rows],
            "total": len(rows)}


@router.get("/organisations")
def organisations(user: User = Depends(admin_only), db: Session = Depends(get_session)):
    orgs = db.scalars(select(Organisation).order_by(Organisation.name)).all()
    counts: Dict[str, int] = {}
    for a in db.scalars(select(PortalAccount)).all():
        if a.organisation_id:
            counts[a.organisation_id] = counts.get(a.organisation_id, 0) + 1
    return {"items": [{"id": o.id, "name": o.name, "account_type": o.account_type,
                       "site_label": o.site_label, "status": o.status,
                       "contact_email": o.contact_email, "accounts": counts.get(o.id, 0),
                       "created_at": o.created_at} for o in orgs]}


@router.get("/activity")
def activity(limit: int = 200, event: Optional[str] = None, email: Optional[str] = None,
             user: User = Depends(admin_only), db: Session = Depends(get_session)):
    stmt = select(PortalActivity).order_by(PortalActivity.at.desc()).limit(min(limit, 1000))
    if event:
        stmt = stmt.where(PortalActivity.event == event)
    if email:
        stmt = stmt.where(PortalActivity.email == email.lower())
    rows = db.scalars(stmt).all()
    return {"items": [{"id": e.id, "at": e.at, "email": e.email, "event": e.event,
                       "outcome": e.outcome, "ip": e.ip, "user_agent": e.user_agent,
                       "organisation_id": e.organisation_id, "detail": e.detail,
                       "note": e.note} for e in rows]}


class StatusIn(BaseModel):
    status: str          # active | suspended


@router.post("/accounts/{account_id}/status")
def set_status(account_id: str, body: StatusIn, request: Request,
               user: User = Depends(admin_only), db: Session = Depends(get_session)):
    if body.status not in ("active", "suspended"):
        raise HTTPException(400, "status must be 'active' or 'suspended'")
    account = db.get(PortalAccount, account_id)
    if account is None:
        raise HTTPException(404, "account not found")
    account.status = body.status
    if account.user_id:
        linked = db.get(User, account.user_id)
        if linked:
            linked.active = body.status == "active"
    db.add(PortalActivity(
        id=f"ACT_{account.id}_{int(utcnow().timestamp())}", at=utcnow(), email=account.email,
        account_id=account.id, organisation_id=account.organisation_id,
        event="admin_status_change", outcome=body.status, ip="",
        detail={"by": user.username}))
    db.commit()
    return {"ok": True, "id": account.id, "status": account.status}

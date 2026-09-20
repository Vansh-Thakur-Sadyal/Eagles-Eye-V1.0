"""Authentication, role-based access control and the audit trail.

Privacy-by-design (spec S27): every read of identity-bearing data is logged,
and identity escalation is gated behind an approval request.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_session
from ..models import AuditLog, User, new_id, utcnow

log = logging.getLogger("sentinel.security")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

# Role -> permissions.  Roles themselves live in the DB; this table defines
# what each role is allowed to do and is the single source of truth for RBAC.
ROLE_PERMISSIONS: Dict[str, List[str]] = {
    "admin": ["*"],
    "commander": [
        "incident:read", "incident:write", "incident:dispatch",
        "camera:read", "camera:write", "track:read", "identity:escalate",
        "approval:decide", "watchlist:read", "watchlist:write", "watchlist:approve",
        "report:read", "report:write", "case:read", "case:write",
        "evidence:read", "evidence:write", "workflow:read", "workflow:write",
        "system:read", "system:gpu", "audit:read", "policy:read", "policy:write",
    ],
    "operator": [
        "incident:read", "incident:write", "camera:read", "track:read",
        "watchlist:read", "report:read", "case:read", "evidence:read",
        "system:read", "system:gpu", "policy:read", "approval:request",
    ],
    "investigator": [
        "incident:read", "camera:read", "track:read", "identity:escalate",
        "watchlist:read", "watchlist:write", "case:read", "case:write",
        "evidence:read", "evidence:write", "report:read", "report:write",
        "system:read", "approval:request",
    ],
    "field": ["incident:read", "camera:read", "ar:read", "system:read"],
    "auditor": ["audit:read", "incident:read", "policy:read", "system:read", "privacy:read"],
    "viewer": ["incident:read", "camera:read", "system:read"],
    # An organisation's own operator, signed up through the public portal.
    # They run THEIR site: attach cameras, work incidents, read their own
    # watchlist. Scoping to their organisation happens in app/portal/scope.py;
    # nothing here grants sight of another tenant or of the admin panel.
    "org_operator": [
        "incident:read", "incident:write", "incident:dispatch",
        "camera:read", "camera:write", "track:read",
        "watchlist:read", "report:read", "report:write", "case:read", "case:write",
        "evidence:read", "evidence:write", "system:read", "policy:read",
        "approval:request", "workflow:read",
    ],
    # A GPU machine that processes cameras for this deployment
    # (tools/gpu_worker.py). Least privilege: it can take camera assignments,
    # report results and mirror the policy and watchlist gallery its agents
    # need - it cannot browse incidents, cases, users or change anything.
    "edge_node": ["edge:node", "policy:read", "watchlist:sync"],
}


def hash_password(password: str) -> str:
    """PBKDF2-SHA256 with a per-password salt (no external dependency)."""
    import os

    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260000)
    return f"pbkdf2_sha256$260000${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        return dk.hex() == hash_hex
    except Exception:
        return False


def create_token(user: User) -> str:
    s = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.id,
        "username": user.username,
        "role": user.role,
        "name": user.full_name,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=s.jwt_ttl_minutes)).timestamp()),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> Dict[str, Any]:
    s = get_settings()
    return jwt.decode(token, s.jwt_secret, algorithms=["HS256"])


def permissions_for(role: str) -> List[str]:
    return ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS["viewer"])


def has_permission(role: str, permission: str) -> bool:
    perms = permissions_for(role)
    if "*" in perms:
        return True
    if permission in perms:
        return True
    # allow "incident:*" style grants
    prefix = permission.split(":", 1)[0]
    return f"{prefix}:*" in perms


async def current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_session),
) -> User:
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    try:
        payload = decode_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired")
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")

    user = db.get(User, payload.get("sub"))
    if user is None or not user.active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Account unavailable")
    request.state.user = user
    return user


def require(permission: str):
    """Dependency factory: `Depends(require("incident:write"))`."""

    async def _guard(user: User = Depends(current_user)) -> User:
        if not has_permission(user.role, permission):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Role '{user.role}' lacks permission '{permission}'",
            )
        return user

    return _guard


def audit(
    db: Session,
    *,
    actor: str,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    actor_role: str = "",
    justification: Optional[str] = None,
    ip_address: Optional[str] = None,
    outcome: str = "success",
    detail: Optional[Dict[str, Any]] = None,
    commit: bool = True,
) -> AuditLog:
    """Append to the immutable audit trail."""
    entry = AuditLog(
        id=new_id("AUD"),
        at=utcnow(),
        actor=actor,
        actor_role=actor_role,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        justification=justification,
        ip_address=ip_address,
        outcome=outcome,
        detail=detail or {},
    )
    db.add(entry)
    if commit:
        db.commit()
    return entry

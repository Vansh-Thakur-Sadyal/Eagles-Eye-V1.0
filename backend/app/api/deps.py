"""Shared FastAPI dependencies and response helpers."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from fastapi import Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from ..core.security import audit, current_user, has_permission, require
from ..db import get_session
from ..models import User


class Page:
    """Standard list envelope so every collection endpoint looks the same."""

    @staticmethod
    def of(items: List[Any], total: int, limit: int, offset: int) -> Dict[str, Any]:
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(items) < total,
        }


def pagination(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, int]:
    return {"limit": limit, "offset": offset}


def client_ip(request: Request) -> Optional[str]:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


def record(
    db: Session,
    request: Request,
    user: User,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    *,
    justification: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
    outcome: str = "success",
) -> None:
    audit(
        db,
        actor=user.username,
        actor_role=user.role,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        justification=justification,
        ip_address=client_ip(request),
        outcome=outcome,
        detail=detail or {},
    )


def to_dict(obj: Any, *, exclude: Iterable[str] = ()) -> Dict[str, Any]:
    """Serialise a SQLAlchemy row without pulling in relationships."""
    from sqlalchemy import inspect as sa_inspect

    exclude = set(exclude)
    data: Dict[str, Any] = {}
    for column in sa_inspect(obj).mapper.column_attrs:
        if column.key in exclude:
            continue
        value = getattr(obj, column.key)
        data[column.key] = value.isoformat() if hasattr(value, "isoformat") else value
    return data


def ensure(condition: bool, message: str, code: int = status.HTTP_400_BAD_REQUEST) -> None:
    if not condition:
        raise HTTPException(code, message)


def get_or_404(db: Session, model: Any, ident: str, label: str) -> Any:
    obj = db.get(model, ident)
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{label} '{ident}' not found")
    return obj


__all__ = [
    "Page",
    "pagination",
    "client_ip",
    "record",
    "to_dict",
    "ensure",
    "get_or_404",
    "current_user",
    "require",
    "has_permission",
    "get_session",
    "Depends",
]

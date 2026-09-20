"""Organisation scoping: one tenant must never see another's data.

Every camera carries the organisation that created it (`meta.organisation_id`).
An operator only ever sees cameras of their own organisation, and only the
incidents, tracks and telemetry belonging to those cameras. The platform
administrator sees everything; a camera with no organisation yet (created
before the portal, or by the administrator) is administrator-only.

Kept as small helpers rather than a query layer so it is obvious at each call
site which set of cameras a request is allowed to touch.
"""
from __future__ import annotations

from typing import List, Optional, Set

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Camera, User
from .models import PortalAccount


def is_admin(user: User) -> bool:
    return user.role == "admin"


def org_of(db: Session, user: User) -> Optional[str]:
    """The organisation this user belongs to, or None (admin / platform user)."""
    account = db.scalar(select(PortalAccount).where(PortalAccount.user_id == user.id))
    if account is None:
        account = db.scalar(select(PortalAccount).where(PortalAccount.email == user.username))
    return account.organisation_id if account else None


def camera_org(camera: Camera) -> Optional[str]:
    return (camera.meta or {}).get("organisation_id")


def visible_camera_ids(db: Session, user: User) -> Optional[Set[str]]:
    """Camera ids this user may see. None means 'everything' (administrator)."""
    if is_admin(user):
        return None
    org = org_of(db, user)
    cameras = db.scalars(select(Camera)).all()
    if not org:
        # A signed-in user with no organisation sees no cameras rather than all
        # of them: failing closed is the only safe direction here.
        return set()
    return {c.id for c in cameras if camera_org(c) == org}


def may_see_camera(db: Session, user: User, camera: Camera) -> bool:
    if is_admin(user):
        return True
    org = org_of(db, user)
    return bool(org) and camera_org(camera) == org


def filter_cameras(db: Session, user: User, cameras: List[Camera]) -> List[Camera]:
    if is_admin(user):
        return cameras
    org = org_of(db, user)
    if not org:
        return []
    return [c for c in cameras if camera_org(c) == org]

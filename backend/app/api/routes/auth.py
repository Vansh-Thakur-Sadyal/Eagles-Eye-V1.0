"""Authentication and session endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...core.security import (
    ROLE_PERMISSIONS,
    create_token,
    current_user,
    hash_password,
    permissions_for,
    require,
    verify_password,
)
from ...db import get_session
from ...models import User, new_id, utcnow
from ..deps import client_ip, record, to_dict

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginBody(BaseModel):
    username: str
    password: str


def _session_payload(user: User) -> Dict[str, Any]:
    return {
        "access_token": create_token(user),
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role,
            "badge": user.badge,
            "post": user.post,
            "permissions": permissions_for(user.role),
        },
    }


def _authenticate(db: Session, request: Request, username: str, password: str) -> Dict[str, Any]:
    user = db.scalar(select(User).where(User.username == username))
    if user is None or not verify_password(password, user.password_hash):
        from ...core.security import audit

        audit(
            db, actor=username, action="auth.login", resource_type="session",
            outcome="failure", ip_address=client_ip(request),
        )
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password")
    if not user.active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "This account is deactivated")

    user.last_login_at = utcnow()
    db.commit()
    record(db, request, user, "auth.login", "session", user.id)
    return _session_payload(user)


@router.post("/login")
def login(body: LoginBody, request: Request, db: Session = Depends(get_session)):
    return _authenticate(db, request, body.username, body.password)


@router.post("/token")
def token(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_session),
):
    """OAuth2 password flow, so the interactive API docs can authenticate."""
    return _authenticate(db, request, form.username, form.password)


@router.get("/me")
def me(user: User = Depends(current_user)):
    return {
        **to_dict(user, exclude={"password_hash"}),
        "permissions": permissions_for(user.role),
    }


@router.post("/logout")
def logout(request: Request, user: User = Depends(current_user), db: Session = Depends(get_session)):
    record(db, request, user, "auth.logout", "session", user.id)
    return {"ok": True}


@router.get("/roles")
def roles(user: User = Depends(current_user)):
    return {
        "roles": [
            {"role": name, "permissions": perms, "wildcard": "*" in perms}
            for name, perms in ROLE_PERMISSIONS.items()
        ]
    }


class UserCreate(BaseModel):
    username: str
    password: str
    full_name: str = ""
    email: Optional[str] = None
    role: str = "viewer"
    badge: Optional[str] = None
    post: Optional[str] = None


@router.get("/users")
def list_users(user: User = Depends(require("audit:read")), db: Session = Depends(get_session)):
    rows = db.scalars(select(User).order_by(User.username)).all()
    return {"items": [to_dict(u, exclude={"password_hash"}) for u in rows]}


@router.post("/users")
def create_user(
    body: UserCreate,
    request: Request,
    user: User = Depends(require("*")),
    db: Session = Depends(get_session),
):
    if body.role not in ROLE_PERMISSIONS:
        raise HTTPException(400, f"unknown role '{body.role}'. Known: {sorted(ROLE_PERMISSIONS)}")
    if db.scalar(select(User).where(User.username == body.username)):
        raise HTTPException(409, f"username '{body.username}' is taken")
    if len(body.password) < 8:
        raise HTTPException(400, "password must be at least 8 characters")

    created = User(
        id=new_id("USR"),
        username=body.username,
        full_name=body.full_name or body.username,
        email=body.email,
        password_hash=hash_password(body.password),
        role=body.role,
        badge=body.badge,
        post=body.post,
    )
    db.add(created)
    db.commit()
    record(db, request, user, "user.create", "user", created.id,
           detail={"username": body.username, "role": body.role})
    return to_dict(created, exclude={"password_hash"})


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


@router.post("/password")
def change_password(
    body: PasswordChange,
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_session),
):
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(403, "current password is incorrect")
    if len(body.new_password) < 8:
        raise HTTPException(400, "new password must be at least 8 characters")
    user.password_hash = hash_password(body.new_password)
    db.commit()
    record(db, request, user, "user.password_change", "user", user.id)
    return {"ok": True}


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    badge: Optional[str] = None
    post: Optional[str] = None
    active: Optional[bool] = None


@router.patch("/users/{user_id}")
def update_user(
    user_id: str,
    body: UserUpdate,
    request: Request,
    user: User = Depends(require("*")),
    db: Session = Depends(get_session),
):
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(404, "user not found")
    if body.role and body.role not in ROLE_PERMISSIONS:
        raise HTTPException(400, f"unknown role '{body.role}'")
    if target.id == user.id and body.active is False:
        raise HTTPException(400, "you cannot deactivate your own account")

    for key, value in body.model_dump(exclude_none=True).items():
        setattr(target, key, value)
    db.commit()
    record(db, request, user, "user.update", "user", target.id,
           detail=body.model_dump(exclude_none=True))
    return to_dict(target, exclude={"password_hash"})

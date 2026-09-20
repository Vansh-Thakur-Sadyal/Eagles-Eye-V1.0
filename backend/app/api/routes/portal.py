"""Public portal: sign-up, verification, sign-in, password reset.

Every call is captcha-gated and rate-limited here, then forwarded to the
Sentinel_Auth workflow in n8n, which owns the passwords and the one-time
codes. On a successful sign-in this mints the ordinary Eagles Eye session token,
so the dashboard, its roles and its audit trail work exactly as before - the
difference is that the account is bound to one organisation and can only see
that organisation's data.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...config import get_settings
from ...core.security import create_token, hash_password
from ...db import get_session
from ...models import User, new_id
from ...portal.models import (ACCOUNT_TYPES, ACCOUNT_TYPE_VALUES, TERMS_VERSION,
                              Organisation, PortalAccount, PortalActivity)
from ...portal.service import (LOGIN_LIMIT, OTP_LIMIT, SIGNUP_LIMIT, check_captcha,
                               client_ip, get_bridge, make_captcha, rate_ok, utcnow)

log = logging.getLogger("sentinel.portal.api")
router = APIRouter(prefix="/api/portal", tags=["portal"])


# ------------------------------------------------------------------ helpers
def record(db: Session, request: Request, event: str, *, email: str = "",
           outcome: str = "ok", account: Optional[PortalAccount] = None,
           detail: Optional[Dict[str, Any]] = None, note: str = "") -> None:
    """Append to the activity trail. Never raises: telemetry must not break sign-in."""
    try:
        db.add(PortalActivity(
            id=new_id("ACT"), at=utcnow(), email=(email or "").lower()[:255],
            account_id=account.id if account else None,
            organisation_id=account.organisation_id if account else None,
            event=event, outcome=outcome, ip=client_ip(request),
            user_agent=request.headers.get("user-agent", "")[:255],
            detail=detail or {}, note=note or None,
        ))
        db.commit()
    except Exception:
        db.rollback()
        log.exception("could not record portal activity %s", event)


def account_for(db: Session, email: str) -> Optional[PortalAccount]:
    return db.scalar(select(PortalAccount).where(PortalAccount.email == email.lower()))


def notify_admin(subject: str, lines: Dict[str, Any]) -> None:
    """Send the administrator a copy through the n8n notify hook."""
    s = get_settings()
    if not (s.portal_admin_email and s.portal_notify_key):
        return
    body = "\n".join(f"{k}: {v}" for k, v in lines.items())
    status, data = get_bridge().call("notify", {
        "key": s.portal_notify_key, "to": s.portal_admin_email,
        "subject": subject, "text": body,
    })
    if status >= 400:
        log.warning("admin notification failed (%s): %s", status, data.get("error"))


def fail(status: int, message: str) -> HTTPException:
    return HTTPException(status, message)


# ------------------------------------------------------------------ schemas
class SignUpIn(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: EmailStr
    phone: str = Field(min_length=10, max_length=20)
    password: str = Field(min_length=8, max_length=200)
    account_type: str
    organisation: str = Field(min_length=2, max_length=120)
    site_label: str = Field(default="", max_length=120)
    terms_accepted: bool = False
    captcha_token: str = ""
    captcha_answer: str = ""


class VerifyIn(BaseModel):
    email: EmailStr
    code: str = Field(min_length=4, max_length=10)


class ResendIn(BaseModel):
    email: EmailStr
    captcha_token: str = ""
    captcha_answer: str = ""


class LoginIn(BaseModel):
    email: EmailStr
    password: str
    captcha_token: str = ""
    captcha_answer: str = ""


class OtpStartIn(BaseModel):
    email: EmailStr
    captcha_token: str = ""
    captcha_answer: str = ""


class OtpLoginIn(BaseModel):
    email: EmailStr
    code: str = Field(min_length=4, max_length=10)


class ForgotIn(BaseModel):
    email: EmailStr
    captcha_token: str = ""
    captcha_answer: str = ""


class ResetIn(BaseModel):
    email: EmailStr
    code: str = Field(min_length=4, max_length=10)
    password: str = Field(min_length=8, max_length=200)


# ------------------------------------------------------------------- public
@router.get("/config")
def portal_config():
    """What the sign-up form needs, so the UI hardcodes nothing."""
    s = get_settings()
    return {
        "enabled": s.portal_enabled,
        "auth_configured": get_bridge().configured,
        "account_types": ACCOUNT_TYPES,
        "terms_version": TERMS_VERSION,
        "allowed_email_domains": ["gmail.com"],
        "password_min_length": 8,
    }


@router.get("/captcha")
def captcha():
    return make_captcha()


@router.post("/signup")
def signup(body: SignUpIn, request: Request, db: Session = Depends(get_session)):
    ip = client_ip(request)
    email = body.email.lower()
    if not rate_ok("signup", ip, SIGNUP_LIMIT):
        record(db, request, "rate_limited", email=email, outcome="blocked",
               detail={"stage": "signup"})
        raise fail(429, "Too many sign-up attempts from this connection. Try later.")
    ok, why = check_captcha(body.captcha_token, body.captcha_answer)
    if not ok:
        record(db, request, "captcha_failed", email=email, outcome="blocked")
        raise fail(400, why)
    if body.account_type not in ACCOUNT_TYPE_VALUES:
        raise fail(400, "Choose the kind of site this account is for.")
    if not body.terms_accepted:
        raise fail(400, "You must accept the Terms and Conditions to register.")

    status, data = get_bridge().call("register", {
        "name": body.name, "email": email, "phone": body.phone,
        "password": body.password, "account_type": body.account_type,
        "organisation": body.organisation, "site_label": body.site_label,
        "terms_accepted": True,
    })
    if not data.get("ok"):
        record(db, request, "signup_started", email=email, outcome="failed",
               detail={"error": data.get("error")})
        raise fail(status if status >= 400 else 400,
                   data.get("error") or "Could not create the account.")

    org = db.scalar(select(Organisation).where(
        Organisation.name == body.organisation,
        Organisation.account_type == body.account_type))
    if org is None:
        org = Organisation(id=new_id("ORG"), name=body.organisation,
                           account_type=body.account_type, site_label=body.site_label,
                           contact_email=email, contact_phone=body.phone)
        db.add(org)
        db.flush()
    account = account_for(db, email)
    if account is None:
        account = PortalAccount(id=new_id("PAC"), email=email, role="org_operator")
        db.add(account)
    account.full_name = body.name
    account.phone = body.phone
    account.organisation_id = org.id
    account.status = "pending"
    account.terms_version = TERMS_VERSION
    account.terms_accepted_at = utcnow()
    account.terms_accepted_ip = ip
    account.signup_ip = ip
    db.commit()

    record(db, request, "signup_started", email=email, account=account,
           detail={"organisation": body.organisation, "account_type": body.account_type})
    notify_admin(f"[Eagles Eye] New sign-up: {body.name} ({body.organisation})", {
        "Name": body.name, "Email": email, "Phone": body.phone,
        "Organisation": body.organisation, "Site type": body.account_type,
        "Site label": body.site_label or "-",
        "Terms": f"accepted {TERMS_VERSION} at {account.terms_accepted_at.isoformat()} from {ip}",
        "Status": "awaiting e-mail/SMS verification",
    })
    return {"ok": True, "message": data.get("message")
            or "We sent a 6-digit code to your e-mail and phone.", "email": email}


@router.post("/verify")
def verify(body: VerifyIn, request: Request, db: Session = Depends(get_session)):
    email = body.email.lower()
    if not rate_ok("verify", client_ip(request), OTP_LIMIT):
        raise fail(429, "Too many attempts. Try again in a few minutes.")
    status, data = get_bridge().call("verify", {"email": email, "code": body.code,
                                                "otp": body.code})
    account = account_for(db, email)
    if not data.get("ok"):
        record(db, request, "signup_verified", email=email, outcome="failed",
               account=account, detail={"error": data.get("error")})
        raise fail(status if status >= 400 else 400,
                   data.get("error") or "That code is not valid.")
    if account is not None:
        account.status = "active"
        account.verified_at = utcnow()
        db.commit()
    record(db, request, "signup_verified", email=email, account=account)
    notify_admin(f"[Eagles Eye] Account verified: {email}",
                 {"Email": email,
                  "Organisation": account.organisation_id if account else "-",
                  "Verified at": utcnow().isoformat()})
    return {"ok": True, "message": "Your account is verified. You can sign in now."}


@router.post("/resend")
def resend(body: ResendIn, request: Request, db: Session = Depends(get_session)):
    if not rate_ok("resend", client_ip(request), OTP_LIMIT):
        raise fail(429, "Too many requests. Try again in a few minutes.")
    ok, why = check_captcha(body.captcha_token, body.captcha_answer)
    if not ok:
        raise fail(400, why)
    status, data = get_bridge().call("resend", {"email": body.email.lower()})
    record(db, request, "otp_sent", email=body.email.lower(),
           outcome="ok" if data.get("ok") else "failed")
    if not data.get("ok"):
        raise fail(status if status >= 400 else 400, data.get("error") or "Could not resend.")
    return {"ok": True, "message": data.get("message") or "A new code is on its way."}


# -------------------------------------------------------------------- login
def _issue_session(db: Session, request: Request, account: PortalAccount,
                   profile: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror the verified portal account into a platform user and sign a token."""
    user = db.get(User, account.user_id) if account.user_id else None
    if user is None:
        user = db.scalar(select(User).where(User.username == account.email))
    if user is None:
        user = User(
            id=new_id("USR"), username=account.email, full_name=account.full_name,
            email=account.email, role=account.role,
            # The password lives in the n8n identity store; this row must never
            # accept a direct password login, so it gets an unusable hash.
            password_hash=hash_password(new_id("LOCKED") + account.email),
            active=True,
        )
        db.add(user)
        db.flush()
        account.user_id = user.id
    user.full_name = account.full_name or user.full_name
    user.role = account.role
    user.active = account.status == "active"
    user.last_login_at = utcnow()
    account.last_login_at = utcnow()
    account.last_seen_at = utcnow()
    account.login_count = (account.login_count or 0) + 1
    db.commit()

    org = db.get(Organisation, account.organisation_id) if account.organisation_id else None
    record(db, request, "login_ok", email=account.email, account=account,
           detail={"organisation": org.name if org else None})
    notify_admin(f"[Eagles Eye] Sign-in: {account.full_name or account.email}", {
        "Email": account.email, "Name": account.full_name,
        "Organisation": org.name if org else "-",
        "Site type": org.account_type if org else "-",
        "At": utcnow().isoformat(), "From": client_ip(request),
        "Sign-in number": account.login_count,
    })
    return {
        "ok": True,
        "access_token": create_token(user),
        "token_type": "bearer",
        "account": {
            "email": account.email, "name": account.full_name, "role": account.role,
            "organisation": {"id": org.id, "name": org.name,
                             "account_type": org.account_type} if org else None,
            "is_admin": account.role == "admin",
        },
    }


@router.post("/login")
def login(body: LoginIn, request: Request, db: Session = Depends(get_session)):
    ip = client_ip(request)
    email = body.email.lower()
    if not rate_ok("login", ip, LOGIN_LIMIT):
        record(db, request, "rate_limited", email=email, outcome="blocked",
               detail={"stage": "login"})
        raise fail(429, "Too many sign-in attempts. Try again shortly.")
    ok, why = check_captcha(body.captcha_token, body.captcha_answer)
    if not ok:
        record(db, request, "captcha_failed", email=email, outcome="blocked")
        raise fail(400, why)

    status, data = get_bridge().call("login", {"email": email, "password": body.password})
    account = account_for(db, email)
    if not data.get("ok"):
        record(db, request, "login_failed", email=email, outcome="failed",
               account=account, detail={"error": data.get("error")})
        raise fail(status if status >= 400 else 401,
                   data.get("error") or "Those details did not match.")
    if account is None:
        raise fail(403, "This account is not registered for Eagles Eye.")
    if account.status == "suspended":
        record(db, request, "login_failed", email=email, outcome="blocked", account=account,
               note="account suspended")
        raise fail(403, "This account has been suspended. Contact the administrator.")
    if account.status != "active":
        raise fail(403, "Verify your e-mail before signing in.")

    # Password alone is not enough: a one-time code follows, as agreed.
    start_status, start = get_bridge().call("otp-start", {"email": email})
    record(db, request, "otp_sent", email=email, account=account,
           outcome="ok" if start.get("ok") else "failed")
    if not start.get("ok"):
        raise fail(start_status if start_status >= 400 else 400,
                   start.get("error") or "Could not send your sign-in code.")
    return {"ok": True, "otp_required": True, "email": email,
            "message": start.get("message") or "Enter the 6-digit code we just sent you."}


@router.post("/otp-start")
def otp_start(body: OtpStartIn, request: Request, db: Session = Depends(get_session)):
    if not rate_ok("otp", client_ip(request), OTP_LIMIT):
        raise fail(429, "Too many code requests. Try again shortly.")
    ok, why = check_captcha(body.captcha_token, body.captcha_answer)
    if not ok:
        raise fail(400, why)
    status, data = get_bridge().call("otp-start", {"email": body.email.lower()})
    record(db, request, "otp_sent", email=body.email.lower(),
           outcome="ok" if data.get("ok") else "failed")
    if not data.get("ok"):
        raise fail(status if status >= 400 else 400,
                   data.get("error") or "Could not send the code.")
    return {"ok": True, "message": data.get("message") or "Code sent."}


@router.post("/otp-login")
def otp_login(body: OtpLoginIn, request: Request, db: Session = Depends(get_session)):
    email = body.email.lower()
    if not rate_ok("otp-login", client_ip(request), LOGIN_LIMIT):
        raise fail(429, "Too many attempts. Try again shortly.")
    status, data = get_bridge().call("otp-login", {"email": email, "code": body.code,
                                                   "otp": body.code})
    account = account_for(db, email)
    if not data.get("ok"):
        record(db, request, "login_failed", email=email, outcome="failed", account=account,
               detail={"error": data.get("error"), "stage": "otp"})
        raise fail(status if status >= 400 else 401,
                   data.get("error") or "That code is not valid.")
    if account is None or account.status != "active":
        raise fail(403, "This account cannot sign in.")
    return _issue_session(db, request, account, data)


@router.post("/forgot")
def forgot(body: ForgotIn, request: Request, db: Session = Depends(get_session)):
    if not rate_ok("forgot", client_ip(request), OTP_LIMIT):
        raise fail(429, "Too many requests. Try again shortly.")
    ok, why = check_captcha(body.captcha_token, body.captcha_answer)
    if not ok:
        raise fail(400, why)
    email = body.email.lower()
    status, data = get_bridge().call("forgot", {"email": email})
    record(db, request, "password_reset", email=email, outcome="requested")
    # Deliberately the same answer either way: whether an address is registered
    # is not something a stranger gets to learn.
    return {"ok": True, "message": "If that address is registered, a reset code is on its way."}


@router.post("/reset")
def reset(body: ResetIn, request: Request, db: Session = Depends(get_session)):
    email = body.email.lower()
    if not rate_ok("reset", client_ip(request), OTP_LIMIT):
        raise fail(429, "Too many attempts. Try again shortly.")
    status, data = get_bridge().call("reset", {"email": email, "code": body.code,
                                               "otp": body.code, "password": body.password})
    account = account_for(db, email)
    if not data.get("ok"):
        record(db, request, "password_reset", email=email, outcome="failed", account=account,
               detail={"error": data.get("error")})
        raise fail(status if status >= 400 else 400,
                   data.get("error") or "Could not reset the password.")
    record(db, request, "password_reset", email=email, outcome="ok", account=account)
    notify_admin(f"[Eagles Eye] Password reset: {email}",
                 {"Email": email, "At": utcnow().isoformat(), "IP": client_ip(request)})
    return {"ok": True, "message": "Password changed. You can sign in now."}

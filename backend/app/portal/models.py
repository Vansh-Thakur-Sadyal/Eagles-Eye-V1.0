"""Organisations, portal accounts and the activity trail.

An account belongs to exactly one organisation - an airport, a railway
station, a campus - and an operator may only ever see that organisation's
cameras and incidents. Only the platform administrator sees across all of
them; nobody but the administrator sees the admin panel at all.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..models.base import Base, IdMixin, TimestampMixin

# The kinds of site an account can be registered for. Kept here and served to
# the sign-up form, so the list is never duplicated in the UI.
ACCOUNT_TYPES: List[Dict[str, str]] = [
    {"value": "airport", "label": "Airport"},
    {"value": "railway_station", "label": "Railway station"},
    {"value": "metro_station", "label": "Metro station"},
    {"value": "bus_terminal", "label": "Bus terminal"},
    {"value": "stadium", "label": "Stadium / arena"},
    {"value": "mall", "label": "Shopping mall"},
    {"value": "campus", "label": "Campus / institution"},
    {"value": "police", "label": "Police / civic control room"},
    {"value": "event", "label": "Event or temporary deployment"},
    {"value": "other", "label": "Other"},
]
ACCOUNT_TYPE_VALUES = {t["value"] for t in ACCOUNT_TYPES}

TERMS_VERSION = "v1.0"


class Organisation(Base, IdMixin, TimestampMixin):
    """One tenant: every camera, incident and user belongs to one of these."""

    __tablename__ = "organisations"

    name: Mapped[str] = mapped_column(String(160), index=True)
    account_type: Mapped[str] = mapped_column(String(32), index=True)
    site_label: Mapped[str] = mapped_column(String(160), default="")
    status: Mapped[str] = mapped_column(String(24), default="active", index=True)
    # active | suspended
    contact_email: Mapped[str] = mapped_column(String(255), default="")
    contact_phone: Mapped[str] = mapped_column(String(32), default="")
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class PortalAccount(Base, IdMixin, TimestampMixin):
    """A person who signed up through the public portal.

    The password never lives here - it stays in the n8n/Mongo identity store.
    This row carries what the platform needs: which organisation they belong
    to, what they agreed to, and whether they may sign in at all.
    """

    __tablename__ = "portal_accounts"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    phone: Mapped[str] = mapped_column(String(32), default="")
    full_name: Mapped[str] = mapped_column(String(160), default="")
    organisation_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("organisations.id"), index=True)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(32), default="org_operator", index=True)
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    # pending (awaiting the code) | active | suspended
    verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    terms_version: Mapped[str] = mapped_column(String(16), default="")
    terms_accepted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    terms_accepted_ip: Mapped[str] = mapped_column(String(64), default="")
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), index=True)
    login_count: Mapped[int] = mapped_column(Integer, default=0)
    signup_ip: Mapped[str] = mapped_column(String(64), default="")
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class PortalActivity(Base, IdMixin):
    """Every portal event, for the administrator's monitoring dashboard.

    Deliberately append-only and readable by the administrator alone: it is a
    record of people's actions, so it is not something one tenant may browse
    about another.
    """

    __tablename__ = "portal_activity"

    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    email: Mapped[str] = mapped_column(String(255), default="", index=True)
    account_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    organisation_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    event: Mapped[str] = mapped_column(String(48), index=True)
    # signup_started | signup_verified | login_ok | login_failed | otp_sent |
    # password_reset | logout | captcha_failed | rate_limited | page_view
    outcome: Mapped[str] = mapped_column(String(24), default="ok", index=True)
    ip: Mapped[str] = mapped_column(String(64), default="")
    user_agent: Mapped[str] = mapped_column(String(255), default="")
    detail: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    note: Mapped[Optional[str]] = mapped_column(Text)

"""Eagles Eye domain model.

Follows the dataset and metadata schemas defined in the master write-up:
cameras/zones metadata (S47), tracking records (S41-D), behaviour annotations
(S42), following pairs (S43), unattended objects (S44) and incidents (S45).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, IdMixin, TimestampMixin


# ---------------------------------------------------------------- topology --
class Site(Base, IdMixin, TimestampMixin):
    """A deployment: airport, railway station, metro, campus, city block."""

    __tablename__ = "sites"

    name: Mapped[str] = mapped_column(String(160), index=True)
    site_type: Mapped[str] = mapped_column(String(48), default="generic")
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    twin_model_url: Mapped[Optional[str]] = mapped_column(String(512))
    floors: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    zones: Mapped[List["Zone"]] = relationship(back_populates="site", cascade="all, delete-orphan")
    cameras: Mapped[List["Camera"]] = relationship(back_populates="site")


class Zone(Base, IdMixin, TimestampMixin):
    """A named area with an access policy and a polygon in twin coordinates."""

    __tablename__ = "zones"

    site_id: Mapped[Optional[str]] = mapped_column(ForeignKey("sites.id"), index=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    zone_type: Mapped[str] = mapped_column(String(48), default="public")
    # public | restricted | secure | transit | platform | gate | perimeter
    floor: Mapped[int] = mapped_column(Integer, default=0)
    polygon: Mapped[List[List[float]]] = mapped_column(JSON, default=list)
    authorized_roles: Mapped[List[str]] = mapped_column(JSON, default=list)
    expected_flow_deg: Mapped[Optional[float]] = mapped_column(Float)
    capacity: Mapped[Optional[int]] = mapped_column(Integer)
    risk_weight: Mapped[float] = mapped_column(Float, default=1.0)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    site: Mapped[Optional[Site]] = relationship(back_populates="zones")


class Camera(Base, IdMixin, TimestampMixin):
    """A video source plus its spatial metadata (S47 camera record)."""

    __tablename__ = "cameras"

    site_id: Mapped[Optional[str]] = mapped_column(ForeignKey("sites.id"), index=True)
    zone_id: Mapped[Optional[str]] = mapped_column(ForeignKey("zones.id"), index=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    location: Mapped[str] = mapped_column(String(160), default="")

    # Source handling: webcam | esp32cam | rtsp | http_mjpeg | file | screen
    source_type: Mapped[str] = mapped_column(String(32), default="rtsp")
    source_uri: Mapped[str] = mapped_column(String(1024), default="")
    username: Mapped[Optional[str]] = mapped_column(String(160))
    password: Mapped[Optional[str]] = mapped_column(String(256))

    # Spatial (digital twin / AR)
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)
    altitude_m: Mapped[Optional[float]] = mapped_column(Float)
    floor: Mapped[int] = mapped_column(Integer, default=0)
    orientation_deg: Mapped[float] = mapped_column(Float, default=0.0)
    tilt_deg: Mapped[float] = mapped_column(Float, default=0.0)
    field_of_view_deg: Mapped[float] = mapped_column(Float, default=82.0)
    range_m: Mapped[float] = mapped_column(Float, default=25.0)
    homography: Mapped[Optional[List[List[float]]]] = mapped_column(JSON)

    # Capture
    fps: Mapped[int] = mapped_column(Integer, default=12)
    width: Mapped[int] = mapped_column(Integer, default=1280)
    height: Mapped[int] = mapped_column(Integer, default=720)
    rotation: Mapped[int] = mapped_column(Integer, default=0)

    # Per-camera analytics switches - all optional, all runtime editable
    enabled_agents: Mapped[List[str]] = mapped_column(JSON, default=list)
    detection_classes: Mapped[List[str]] = mapped_column(JSON, default=list)
    privacy_redaction: Mapped[Optional[bool]] = mapped_column(Boolean)
    line_crossings: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)

    status: Mapped[str] = mapped_column(String(24), default="offline", index=True)
    # offline | connecting | online | degraded | error | disabled
    status_detail: Mapped[Optional[str]] = mapped_column(Text)
    last_frame_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    measured_fps: Mapped[float] = mapped_column(Float, default=0.0)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    site: Mapped[Optional[Site]] = relationship(back_populates="cameras")


# ------------------------------------------------------------- perception --
class Track(Base, IdMixin, TimestampMixin):
    """An anonymous per-camera track. Identity is never stored here."""

    __tablename__ = "tracks"

    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    local_track_id: Mapped[int] = mapped_column(Integer, index=True)
    global_id: Mapped[Optional[str]] = mapped_column(ForeignKey("global_subjects.id"), index=True)
    class_name: Mapped[str] = mapped_column(String(48), default="person", index=True)

    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    frame_count: Mapped[int] = mapped_column(Integer, default=0)

    # trajectory: [{t, x, y, w, h, conf, vx, vy}, ...] in image coordinates
    trajectory: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    world_path: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    zones_visited: Mapped[List[str]] = mapped_column(JSON, default=list)

    avg_speed: Mapped[float] = mapped_column(Float, default=0.0)
    max_speed: Mapped[float] = mapped_column(Float, default=0.0)
    dwell_seconds: Mapped[float] = mapped_column(Float, default=0.0)

    embedding_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    face_visibility: Mapped[str] = mapped_column(String(24), default="unknown")
    # visible | partial | covered | absent | unknown
    occlusion_labels: Mapped[List[str]] = mapped_column(JSON, default=list)
    appearance: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    snapshot_path: Mapped[Optional[str]] = mapped_column(String(512))
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    __table_args__ = (
        Index("ix_track_cam_local", "camera_id", "local_track_id"),
    )


class GlobalSubject(Base, IdMixin, TimestampMixin):
    """A cross-camera association hypothesis - never an identity claim."""

    __tablename__ = "global_subjects"

    label: Mapped[str] = mapped_column(String(64), index=True)          # "SUBJECT_A17"
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    camera_ids: Mapped[List[str]] = mapped_column(JSON, default=list)
    track_ids: Mapped[List[str]] = mapped_column(JSON, default=list)
    association_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    gallery_size: Mapped[int] = mapped_column(Integer, default=0)
    appearance_summary: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    identity_escalated: Mapped[bool] = mapped_column(Boolean, default=False)
    escalation_case_id: Mapped[Optional[str]] = mapped_column(String(48))
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class TrackedObject(Base, IdMixin, TimestampMixin):
    """Bag / luggage / package with owner association (S44 format)."""

    __tablename__ = "tracked_objects"

    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    zone_id: Mapped[Optional[str]] = mapped_column(ForeignKey("zones.id"))
    class_name: Mapped[str] = mapped_column(String(48), index=True)
    local_track_id: Mapped[int] = mapped_column(Integer, default=0)

    owner_track_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)
    owner_global_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)
    owner_confidence: Mapped[float] = mapped_column(Float, default=0.0)

    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    separation_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    stationary_since: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    stationary_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    owner_distance_px: Mapped[float] = mapped_column(Float, default=0.0)

    status: Mapped[str] = mapped_column(String(32), default="attended", index=True)
    # attended | separated | warning | unattended | reclaimed | cleared
    bbox: Mapped[List[float]] = mapped_column(JSON, default=list)
    snapshot_path: Mapped[Optional[str]] = mapped_column(String(512))
    incident_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class BehaviorEvent(Base, IdMixin, TimestampMixin):
    """One behavioural observation (S42 annotation format)."""

    __tablename__ = "behavior_events"

    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    zone_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)
    track_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)
    global_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)
    secondary_track_id: Mapped[Optional[str]] = mapped_column(String(48))

    behavior: Mapped[str] = mapped_column(String(64), index=True)
    # loitering | counter_flow | running | sudden_movement | crowd_reversal
    # | following | restricted_entry | abnormal_trajectory | unattended_object
    # | violence | fire_smoke | watchlist_match
    severity: Mapped[str] = mapped_column(String(16), default="info")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    start_frame: Mapped[int] = mapped_column(Integer, default=0)
    end_frame: Mapped[int] = mapped_column(Integer, default=0)

    evidence: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    explanation: Mapped[Optional[str]] = mapped_column(Text)
    snapshot_path: Mapped[Optional[str]] = mapped_column(String(512))
    incident_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)
    agent: Mapped[str] = mapped_column(String(64), default="")


class CrowdSample(Base, IdMixin, TimestampMixin):
    """Per-camera crowd telemetry sampled on a fixed cadence."""

    __tablename__ = "crowd_samples"

    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    zone_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)
    sampled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    count: Mapped[int] = mapped_column(Integer, default=0)
    density: Mapped[float] = mapped_column(Float, default=0.0)
    density_band: Mapped[str] = mapped_column(String(16), default="low")
    mean_speed: Mapped[float] = mapped_column(Float, default=0.0)
    flow_direction_deg: Mapped[Optional[float]] = mapped_column(Float)
    flow_variance: Mapped[float] = mapped_column(Float, default=0.0)
    counter_flow_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    compression: Mapped[float] = mapped_column(Float, default=0.0)
    delta_percent: Mapped[float] = mapped_column(Float, default=0.0)
    risk: Mapped[str] = mapped_column(String(16), default="low")
    heatmap: Mapped[Optional[List[List[float]]]] = mapped_column(JSON)


# --------------------------------------------------------------- incidents --
class Incident(Base, IdMixin, TimestampMixin):
    """Correlated multi-signal event (S45 incident format)."""

    __tablename__ = "incidents"

    site_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)
    camera_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)
    camera_ids: Mapped[List[str]] = mapped_column(JSON, default=list)
    zone_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)

    event_type: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(255))
    summary: Mapped[Optional[str]] = mapped_column(Text)
    explanation: Mapped[Optional[str]] = mapped_column(Text)

    risk_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    severity: Mapped[str] = mapped_column(String(16), default="low", index=True)
    # low | medium | high | critical
    risk_factors: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    track_ids: Mapped[List[str]] = mapped_column(JSON, default=list)
    global_ids: Mapped[List[str]] = mapped_column(JSON, default=list)
    object_ids: Mapped[List[str]] = mapped_column(JSON, default=list)
    behavior_event_ids: Mapped[List[str]] = mapped_column(JSON, default=list)

    recommended_actions: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="open", index=True)
    # open | acknowledged | dispatched | investigating | resolved | false_positive
    assigned_to: Mapped[Optional[str]] = mapped_column(String(64))
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    resolution_note: Mapped[Optional[str]] = mapped_column(Text)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    last_update_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    snapshot_path: Mapped[Optional[str]] = mapped_column(String(512))
    clip_path: Mapped[Optional[str]] = mapped_column(String(512))
    location: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    timeline: Mapped[List["IncidentTimelineEntry"]] = relationship(
        back_populates="incident", cascade="all, delete-orphan", order_by="IncidentTimelineEntry.at"
    )


class IncidentTimelineEntry(Base, IdMixin):
    """One dated line in the incident narrative (S21)."""

    __tablename__ = "incident_timeline"

    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id"), index=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    kind: Mapped[str] = mapped_column(String(48), default="observation")
    # observation | detection | threshold | assessment | notification | action | operator
    actor: Mapped[str] = mapped_column(String(64), default="system")
    text: Mapped[str] = mapped_column(Text)
    camera_id: Mapped[Optional[str]] = mapped_column(String(48))
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    incident: Mapped[Incident] = relationship(back_populates="timeline")


class Alert(Base, IdMixin, TimestampMixin):
    __tablename__ = "alerts"

    incident_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)
    channel: Mapped[str] = mapped_column(String(32), default="dashboard")
    severity: Mapped[str] = mapped_column(String(16), default="info", index=True)
    title: Mapped[str] = mapped_column(String(255))
    body: Mapped[Optional[str]] = mapped_column(Text)
    target_role: Mapped[Optional[str]] = mapped_column(String(48))
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    acknowledged_by: Mapped[Optional[str]] = mapped_column(String(64))
    delivered: Mapped[bool] = mapped_column(Boolean, default=False)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


# ------------------------------------------------------- person of interest --
class WatchlistSubject(Base, IdMixin, TimestampMixin):
    """An authorised person-of-interest reference (missing person, suspect,
    VIP protection, or the operator's own face for self-location).

    Enrolment requires a stated legal basis and an approving operator; every
    match is written to the audit log.
    """

    __tablename__ = "watchlist_subjects"

    label: Mapped[str] = mapped_column(String(160), index=True)
    category: Mapped[str] = mapped_column(String(48), default="person_of_interest")
    # person_of_interest | missing_person | vip | staff | self_enrolled | test
    priority: Mapped[str] = mapped_column(String(16), default="medium")
    description: Mapped[Optional[str]] = mapped_column(Text)

    legal_basis: Mapped[str] = mapped_column(String(255), default="")
    case_reference: Mapped[Optional[str]] = mapped_column(String(96))
    requested_by: Mapped[Optional[str]] = mapped_column(String(64))
    approved_by: Mapped[Optional[str]] = mapped_column(String(64))
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), index=True)

    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    # pending | active | paused | expired | revoked
    match_threshold: Mapped[Optional[float]] = mapped_column(Float)
    alert_on_match: Mapped[bool] = mapped_column(Boolean, default=True)
    scope_camera_ids: Mapped[List[str]] = mapped_column(JSON, default=list)
    scope_site_ids: Mapped[List[str]] = mapped_column(JSON, default=list)

    face_embedding_count: Mapped[int] = mapped_column(Integer, default=0)
    body_embedding_count: Mapped[int] = mapped_column(Integer, default=0)
    augmentation_count: Mapped[int] = mapped_column(Integer, default=0)
    last_match_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    match_count: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    images: Mapped[List["WatchlistImage"]] = relationship(
        back_populates="subject", cascade="all, delete-orphan"
    )


class WatchlistImage(Base, IdMixin, TimestampMixin):
    __tablename__ = "watchlist_images"

    subject_id: Mapped[str] = mapped_column(ForeignKey("watchlist_subjects.id"), index=True)
    file_path: Mapped[str] = mapped_column(String(512))
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(512))
    kind: Mapped[str] = mapped_column(String(24), default="reference")
    # reference | augmented | detected_crop
    augmentation: Mapped[Optional[str]] = mapped_column(String(64))
    augmentation_params: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    face_embedding_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    body_embedding_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    face_bbox: Mapped[Optional[List[float]]] = mapped_column(JSON)
    usable: Mapped[bool] = mapped_column(Boolean, default=True)
    note: Mapped[Optional[str]] = mapped_column(Text)

    subject: Mapped[WatchlistSubject] = relationship(back_populates="images")


class WatchlistMatch(Base, IdMixin, TimestampMixin):
    """A sighting. Always a confidence, never an assertion of identity."""

    __tablename__ = "watchlist_matches"

    subject_id: Mapped[str] = mapped_column(ForeignKey("watchlist_subjects.id"), index=True)
    camera_id: Mapped[str] = mapped_column(String(48), index=True)
    site_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)
    zone_id: Mapped[Optional[str]] = mapped_column(String(48))
    track_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)
    global_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)

    matched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    modality: Mapped[str] = mapped_column(String(24), default="face")   # face | body | fused
    similarity: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    face_visibility: Mapped[str] = mapped_column(String(24), default="unknown")

    location_name: Mapped[Optional[str]] = mapped_column(String(255))
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)
    floor: Mapped[Optional[int]] = mapped_column(Integer)

    snapshot_path: Mapped[Optional[str]] = mapped_column(String(512))
    crop_path: Mapped[Optional[str]] = mapped_column(String(512))
    review_status: Mapped[str] = mapped_column(String(24), default="unreviewed", index=True)
    # unreviewed | confirmed | rejected | uncertain
    reviewed_by: Mapped[Optional[str]] = mapped_column(String(64))
    incident_id: Mapped[Optional[str]] = mapped_column(String(48))
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


# ---------------------------------------------------- governance and access --
class User(Base, IdMixin, TimestampMixin):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(160), default="")
    email: Mapped[Optional[str]] = mapped_column(String(255))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default="viewer", index=True)
    # admin | commander | operator | investigator | field | auditor | viewer
    badge: Mapped[Optional[str]] = mapped_column(String(48))
    post: Mapped[Optional[str]] = mapped_column(String(160))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    permissions: Mapped[List[str]] = mapped_column(JSON, default=list)


class AuditLog(Base, IdMixin):
    __tablename__ = "audit_log"

    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    actor: Mapped[str] = mapped_column(String(64), index=True)
    actor_role: Mapped[str] = mapped_column(String(32), default="")
    action: Mapped[str] = mapped_column(String(96), index=True)
    resource_type: Mapped[str] = mapped_column(String(48), index=True)
    resource_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    justification: Mapped[Optional[str]] = mapped_column(Text)
    ip_address: Mapped[Optional[str]] = mapped_column(String(64))
    outcome: Mapped[str] = mapped_column(String(24), default="success")
    detail: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class ApprovalRequest(Base, IdMixin, TimestampMixin):
    """Human-in-the-loop gate for identity escalation and high-impact actions."""

    __tablename__ = "approval_requests"

    kind: Mapped[str] = mapped_column(String(64), index=True)
    # identity_escalation | watchlist_enrol | evidence_export | dispatch
    # | retention_override | model_promotion
    title: Mapped[str] = mapped_column(String(255))
    detail: Mapped[Optional[str]] = mapped_column(Text)
    justification: Mapped[Optional[str]] = mapped_column(Text)
    requested_by: Mapped[str] = mapped_column(String(64), index=True)
    required_role: Mapped[str] = mapped_column(String(32), default="commander")
    resource_type: Mapped[Optional[str]] = mapped_column(String(48))
    resource_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    decided_by: Mapped[Optional[str]] = mapped_column(String(64))
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    decision_note: Mapped[Optional[str]] = mapped_column(Text)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


# --------------------------------------------------- investigation artefacts --
class Case(Base, IdMixin, TimestampMixin):
    __tablename__ = "cases"

    title: Mapped[str] = mapped_column(String(255), index=True)
    reference: Mapped[Optional[str]] = mapped_column(String(96), unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default="open", index=True)
    priority: Mapped[str] = mapped_column(String(16), default="medium")
    lead: Mapped[Optional[str]] = mapped_column(String(64))
    incident_ids: Mapped[List[str]] = mapped_column(JSON, default=list)
    subject_ids: Mapped[List[str]] = mapped_column(JSON, default=list)
    tags: Mapped[List[str]] = mapped_column(JSON, default=list)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class Evidence(Base, IdMixin, TimestampMixin):
    __tablename__ = "evidence"

    case_id: Mapped[Optional[str]] = mapped_column(ForeignKey("cases.id"), index=True)
    incident_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)
    kind: Mapped[str] = mapped_column(String(32), default="snapshot")
    # snapshot | clip | report | export | note | trajectory
    file_path: Mapped[Optional[str]] = mapped_column(String(512))
    sha256: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    collected_by: Mapped[str] = mapped_column(String(64), default="system")
    chain_of_custody: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    sealed: Mapped[bool] = mapped_column(Boolean, default=False)
    retention_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    note: Mapped[Optional[str]] = mapped_column(Text)


class Report(Base, IdMixin, TimestampMixin):
    __tablename__ = "reports"

    title: Mapped[str] = mapped_column(String(255))
    report_type: Mapped[str] = mapped_column(String(48), default="incident")
    period_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    generated_by: Mapped[str] = mapped_column(String(64), default="system")
    body_markdown: Mapped[Optional[str]] = mapped_column(Text)
    file_path: Mapped[Optional[str]] = mapped_column(String(512))
    incident_ids: Mapped[List[str]] = mapped_column(JSON, default=list)
    stats: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


# ----------------------------------------------------------- ops and models --
class AgentRun(Base, IdMixin):
    """Per-agent execution telemetry for the orchestrator dashboard."""

    __tablename__ = "agent_runs"

    agent: Mapped[str] = mapped_column(String(64), index=True)
    camera_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    inputs: Mapped[int] = mapped_column(Integer, default=0)
    outputs: Mapped[int] = mapped_column(Integer, default=0)
    device: Mapped[str] = mapped_column(String(24), default="cpu")
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[Optional[str]] = mapped_column(Text)


class ModelRecord(Base, IdMixin, TimestampMixin):
    __tablename__ = "model_registry"

    name: Mapped[str] = mapped_column(String(128), index=True)
    task: Mapped[str] = mapped_column(String(48), index=True)
    # detection | tracking | reid | face | behavior | anomaly | crowd | llm | embedding
    framework: Mapped[str] = mapped_column(String(48), default="pytorch")
    version: Mapped[str] = mapped_column(String(48), default="1.0.0")
    weights_path: Mapped[Optional[str]] = mapped_column(String(512))
    source: Mapped[Optional[str]] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(24), default="registered", index=True)
    # registered | loaded | active | failed | archived
    device: Mapped[str] = mapped_column(String(24), default="cpu")
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    params: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    loaded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    notes: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_name_version"),)


class Workflow(Base, IdMixin, TimestampMixin):
    """An automation binding: Eagles Eye trigger -> n8n webhook."""

    __tablename__ = "workflows"

    name: Mapped[str] = mapped_column(String(160), index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    trigger: Mapped[str] = mapped_column(String(64), index=True)
    # incident_created | incident_severity | unattended_object | camera_offline
    # | watchlist_match | daily_report | crowd_risk | manual
    conditions: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    webhook_url: Mapped[str] = mapped_column(String(1024), default="")
    method: Mapped[str] = mapped_column(String(8), default="POST")
    headers: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)


class WorkflowRun(Base, IdMixin):
    __tablename__ = "workflow_runs"

    workflow_id: Mapped[str] = mapped_column(ForeignKey("workflows.id"), index=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    trigger: Mapped[str] = mapped_column(String(64))
    status_code: Mapped[Optional[int]] = mapped_column(Integer)
    ok: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    request_payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    response_snippet: Mapped[Optional[str]] = mapped_column(Text)
    error: Mapped[Optional[str]] = mapped_column(Text)


class SystemMetric(Base, IdMixin):
    __tablename__ = "system_metrics"

    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    scope: Mapped[str] = mapped_column(String(48), index=True)   # host | gpu | pipeline | camera
    scope_id: Mapped[Optional[str]] = mapped_column(String(48), index=True)
    values: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class ChatMessage(Base, IdMixin):
    """Conversational investigation transcript, kept for audit."""

    __tablename__ = "chat_messages"

    session_id: Mapped[str] = mapped_column(String(48), index=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    role: Mapped[str] = mapped_column(String(16))          # user | assistant | tool
    content: Mapped[str] = mapped_column(Text)
    actor: Mapped[Optional[str]] = mapped_column(String(64))
    citations: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    tool_calls: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)

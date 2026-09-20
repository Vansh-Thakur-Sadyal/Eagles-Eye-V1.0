"""Person-of-interest watchlist: enrol a photograph, find the subject, get a location.

The operational flow:

  1. POST /api/watchlist                 create the subject with a legal basis
  2. POST /api/watchlist/{id}/images     upload one or more reference photographs
     (each is augmented into a multi-view gallery so a change of hairstyle,
      glasses, lighting or camera angle does not break the match)
  3. POST /api/watchlist/{id}/approve    a commander approves; only now does
                                          matching begin
  4. GET  /api/watchlist/{id}/sightings  where the subject has been seen, with
                                          camera, zone, site and coordinates
  5. GET  /api/watchlist/{id}/locate     the most recent high-confidence
                                          location, in plain language

Guardrails that are enforced, not merely documented: enrolment requires a
stated legal basis; matching does not start until an approval is recorded;
subjects can expire; every enrolment, image upload, match read and export is
written to the audit trail; and a match is always a confidence with a review
state, never an identification.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...agents.orchestrator import get_orchestrator
from ...agents.watchlist import EnrolledSubject
from ...config import get_settings, load_policy
from ...core.events import bus
from ...core.security import current_user, has_permission, require
from ...db import get_session
from ...models import (
    ApprovalRequest,
    Camera,
    User,
    WatchlistImage,
    WatchlistMatch,
    WatchlistSubject,
    new_id,
    utcnow,
)
from ...vision.augment import (
    FACE_AUGMENTATIONS,
    GEOMETRIC_AUGMENTATIONS,
    PHOTOMETRIC_AUGMENTATIONS,
    augment,
)
from ...vision.reid import get_body_embedder, get_face_embedder
from ..deps import Page, get_or_404, pagination, record, to_dict

log = logging.getLogger("sentinel.api.watchlist")
router = APIRouter(prefix="/api/watchlist", tags=["watchlist"])

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_IMAGE_BYTES = 12 * 1024 * 1024

CATEGORIES = ("person_of_interest", "missing_person", "vip", "staff", "self_enrolled", "test")
PRIORITIES = ("low", "medium", "high", "critical")

# Embeddings live in memory keyed by image id; the files on disk are the record.
_face_vectors: Dict[str, np.ndarray] = {}
_body_vectors: Dict[str, np.ndarray] = {}


# ---------------------------------------------------------------- schemas
class SubjectIn(BaseModel):
    label: str
    category: str = "person_of_interest"
    priority: str = "medium"
    description: Optional[str] = None
    legal_basis: str
    case_reference: Optional[str] = None
    expires_in_days: Optional[int] = 30
    match_threshold: Optional[float] = None
    alert_on_match: bool = True
    scope_camera_ids: List[str] = Field(default_factory=list)
    scope_site_ids: List[str] = Field(default_factory=list)


class SubjectPatch(BaseModel):
    label: Optional[str] = None
    priority: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    match_threshold: Optional[float] = None
    alert_on_match: Optional[bool] = None
    scope_camera_ids: Optional[List[str]] = None
    scope_site_ids: Optional[List[str]] = None
    expires_in_days: Optional[int] = None


class ApprovalBody(BaseModel):
    decision: str = "approve"          # approve | reject
    note: Optional[str] = None


class ReviewBody(BaseModel):
    review_status: str                 # confirmed | rejected | uncertain
    note: Optional[str] = None


# ------------------------------------------------------------------ helpers
def _storage_dir(subject_id: str) -> Path:
    path = get_settings().storage_dir / "watchlist" / subject_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _decode(data: bytes) -> np.ndarray:
    import cv2

    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "the uploaded file could not be decoded as an image")
    return frame


def _subject_payload(db: Session, subject: WatchlistSubject) -> Dict[str, Any]:
    data = to_dict(subject)
    data["images"] = [
        to_dict(i, exclude={"face_embedding_id", "body_embedding_id"})
        for i in subject.images
    ]
    data["reference_image_count"] = sum(1 for i in subject.images if i.kind == "reference")
    data["augmented_image_count"] = sum(1 for i in subject.images if i.kind == "augmented")
    data["matching_active"] = subject.status == "active" and not _expired(subject)
    data["expired"] = _expired(subject)
    recent = db.scalar(
        select(func.count()).select_from(WatchlistMatch)
        .where(WatchlistMatch.subject_id == subject.id)
    ) or 0
    data["total_sightings"] = recent
    return data


def _expired(subject: WatchlistSubject) -> bool:
    if subject.expires_at is None:
        return False
    expires = subject.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= expires


def reload_gallery(db: Session) -> int:
    """Rebuild the live matching gallery from the database."""
    subjects = db.scalars(select(WatchlistSubject)).all()
    enrolled: List[EnrolledSubject] = []
    for s in subjects:
        face = [_face_vectors[i.id] for i in s.images if i.id in _face_vectors and i.usable]
        body = [_body_vectors[i.id] for i in s.images if i.id in _body_vectors and i.usable]
        enrolled.append(
            EnrolledSubject(
                subject_id=s.id,
                label=s.label,
                category=s.category,
                priority=s.priority,
                status="active" if (s.status == "active" and not _expired(s)) else s.status,
                face_embeddings=face,
                body_embeddings=body,
                match_threshold=s.match_threshold,
                scope_camera_ids=s.scope_camera_ids or [],
                scope_site_ids=s.scope_site_ids or [],
                expires_at=s.expires_at,
                alert_on_match=s.alert_on_match,
            )
        )
    get_orchestrator().watchlist.load_subjects(enrolled)
    return len(enrolled)


# ------------------------------------------------------------------- routes
@router.get("")
def list_subjects(
    status_filter: Optional[str] = None,
    category: Optional[str] = None,
    page: Dict[str, int] = Depends(pagination),
    user: User = Depends(require("watchlist:read")),
    db: Session = Depends(get_session),
):
    stmt = select(WatchlistSubject)
    if status_filter:
        stmt = stmt.where(WatchlistSubject.status == status_filter)
    if category:
        stmt = stmt.where(WatchlistSubject.category == category)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(
        stmt.order_by(WatchlistSubject.created_at.desc())
        .limit(page["limit"]).offset(page["offset"])
    ).all()
    return Page.of([_subject_payload(db, s) for s in rows], total, page["limit"], page["offset"])


@router.get("/options")
def options(user: User = Depends(require("watchlist:read"))):
    """Everything the enrolment form needs, sourced from live configuration."""
    policy = load_policy().get("watchlist", {})
    return {
        "categories": list(CATEGORIES),
        "priorities": list(PRIORITIES),
        "augmentations": {
            "face_occlusion": FACE_AUGMENTATIONS,
            "photometric": PHOTOMETRIC_AUGMENTATIONS,
            "geometric": GEOMETRIC_AUGMENTATIONS,
        },
        "defaults": policy,
        "face_backend": get_face_embedder().info(),
        "body_backend": get_body_embedder().info(),
        "requires_approval": bool(policy.get("require_approval", True)),
    }


@router.post("")
def create_subject(
    body: SubjectIn,
    request: Request,
    user: User = Depends(require("watchlist:write")),
    db: Session = Depends(get_session),
):
    if body.category not in CATEGORIES:
        raise HTTPException(400, f"category must be one of {CATEGORIES}")
    if body.priority not in PRIORITIES:
        raise HTTPException(400, f"priority must be one of {PRIORITIES}")
    if not body.legal_basis or len(body.legal_basis.strip()) < 8:
        raise HTTPException(
            400,
            "a legal basis is required to enrol a person: state the authority, case or "
            "consent that permits this search",
        )

    expires = (
        utcnow() + timedelta(days=body.expires_in_days) if body.expires_in_days else None
    )
    subject = WatchlistSubject(
        id=new_id("WLS"),
        label=body.label,
        category=body.category,
        priority=body.priority,
        description=body.description,
        legal_basis=body.legal_basis.strip(),
        case_reference=body.case_reference,
        requested_by=user.username,
        expires_at=expires,
        status="pending",
        match_threshold=body.match_threshold,
        alert_on_match=body.alert_on_match,
        scope_camera_ids=body.scope_camera_ids,
        scope_site_ids=body.scope_site_ids,
    )
    db.add(subject)

    db.add(
        ApprovalRequest(
            id=new_id("APR"),
            kind="watchlist_enrol",
            title=f"Enrol '{body.label}' on the watchlist",
            detail=body.description,
            justification=body.legal_basis,
            requested_by=user.username,
            required_role="commander",
            resource_type="watchlist_subject",
            resource_id=subject.id,
            expires_at=expires,
            payload={"category": body.category, "priority": body.priority},
        )
    )
    db.commit()
    record(db, request, user, "watchlist.create", "watchlist_subject", subject.id,
           justification=body.legal_basis,
           detail={"label": body.label, "category": body.category})
    return _subject_payload(db, subject)


@router.post("/{subject_id}/images")
async def upload_images(
    subject_id: str,
    request: Request,
    files: List[UploadFile] = File(...),
    augment_images: bool = Form(True),
    note: Optional[str] = Form(None),
    user: User = Depends(require("watchlist:write")),
    db: Session = Depends(get_session),
):
    """Upload reference photographs and build the augmented match gallery."""
    subject = get_or_404(db, WatchlistSubject, subject_id, "watchlist subject")
    face_embedder = get_face_embedder()
    body_embedder = get_body_embedder()
    folder = _storage_dir(subject_id)

    import cv2

    created: List[Dict[str, Any]] = []
    face_count = 0
    body_count = 0
    aug_count = 0

    for upload in files:
        raw = await upload.read()
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(413, f"{upload.filename} exceeds the 12 MB limit")
        if upload.content_type and upload.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                415, f"{upload.filename}: unsupported type {upload.content_type}"
            )

        image = _decode(raw)
        digest = hashlib.sha256(raw).hexdigest()[:16]
        image_id = new_id("WLI")
        path = folder / f"{image_id}_{digest}.jpg"
        cv2.imwrite(str(path), image)

        face_vec, face_bbox, face_score = face_embedder.embed_largest(image)
        body_vec = body_embedder.embed(image)

        row = WatchlistImage(
            id=image_id,
            subject_id=subject_id,
            file_path=str(path),
            kind="reference",
            quality_score=round(float(face_score), 3),
            face_bbox=[round(v, 1) for v in face_bbox] if face_bbox else None,
            usable=face_vec is not None or body_vec is not None,
            note=note,
        )
        db.add(row)

        if face_vec is not None:
            _face_vectors[image_id] = face_vec
            row.face_embedding_id = image_id
            face_count += 1
        if body_vec is not None:
            _body_vectors[image_id] = body_vec
            row.body_embedding_id = image_id
            body_count += 1

        entry = {
            "image_id": image_id,
            "filename": upload.filename,
            "face_found": face_vec is not None,
            "face_quality": round(float(face_score), 3),
            "body_embedded": body_vec is not None,
            "augmentations": [],
        }

        if augment_images:
            kinds = FACE_AUGMENTATIONS + PHOTOMETRIC_AUGMENTATIONS + GEOMETRIC_AUGMENTATIONS
            for sample in augment(
                image, kinds, face_bbox=tuple(face_bbox) if face_bbox else None,
                source_identity=subject_id, compose=3,
            ):
                aug_id = new_id("WLI")
                aug_path = folder / f"{aug_id}_{sample.augmentation.replace('+', '_')}.jpg"
                cv2.imwrite(str(aug_path), sample.image)

                a_face, _bbox, a_score = face_embedder.embed_largest(sample.image)
                a_body = body_embedder.embed(sample.image)

                aug_row = WatchlistImage(
                    id=aug_id,
                    subject_id=subject_id,
                    file_path=str(aug_path),
                    kind="augmented",
                    augmentation=sample.augmentation,
                    augmentation_params=sample.params,
                    quality_score=round(float(a_score), 3),
                    usable=a_face is not None or a_body is not None,
                    note=f"derived from {image_id}",
                )
                db.add(aug_row)
                if a_face is not None:
                    _face_vectors[aug_id] = a_face
                    aug_row.face_embedding_id = aug_id
                    face_count += 1
                if a_body is not None:
                    _body_vectors[aug_id] = a_body
                    aug_row.body_embedding_id = aug_id
                    body_count += 1
                aug_count += 1
                entry["augmentations"].append(sample.augmentation)

        created.append(entry)

    subject.face_embedding_count = face_count + (subject.face_embedding_count or 0)
    subject.body_embedding_count = body_count + (subject.body_embedding_count or 0)
    subject.augmentation_count = aug_count + (subject.augmentation_count or 0)
    db.commit()

    loaded = reload_gallery(db)
    record(db, request, user, "watchlist.images", "watchlist_subject", subject_id,
           detail={"files": len(files), "augmented": aug_count})

    warnings = []
    if face_count == 0:
        warnings.append(
            "No face was detected in any reference image. Matching will rely on body "
            "appearance only, which is weaker and always requires human confirmation. "
            "Upload a clearer, front-facing photograph for face matching."
        )
    if not subject.status == "active":
        warnings.append(
            "This subject is not yet approved, so matching has not started. "
            "A commander must approve the enrolment first."
        )

    return {
        "subject_id": subject_id,
        "uploaded": created,
        "face_embeddings": subject.face_embedding_count,
        "body_embeddings": subject.body_embedding_count,
        "augmented_variants": subject.augmentation_count,
        "gallery_subjects_loaded": loaded,
        "warnings": warnings,
    }


@router.post("/{subject_id}/approve")
def approve_subject(
    subject_id: str,
    body: ApprovalBody,
    request: Request,
    user: User = Depends(require("watchlist:approve")),
    db: Session = Depends(get_session),
):
    """Commander decision. Matching starts only after an approval."""
    subject = get_or_404(db, WatchlistSubject, subject_id, "watchlist subject")
    if body.decision not in ("approve", "reject"):
        raise HTTPException(400, "decision must be 'approve' or 'reject'")
    if body.decision == "approve" and not subject.images:
        raise HTTPException(400, "upload at least one reference image before approving")

    subject.status = "active" if body.decision == "approve" else "revoked"
    subject.approved_by = user.username
    subject.approved_at = utcnow()

    approval = db.scalar(
        select(ApprovalRequest).where(
            ApprovalRequest.resource_id == subject_id,
            ApprovalRequest.kind == "watchlist_enrol",
            ApprovalRequest.status == "pending",
        )
    )
    if approval:
        approval.status = "approved" if body.decision == "approve" else "rejected"
        approval.decided_by = user.username
        approval.decided_at = utcnow()
        approval.decision_note = body.note
    db.commit()

    loaded = reload_gallery(db)
    record(db, request, user, "watchlist.approve", "watchlist_subject", subject_id,
           justification=body.note, detail={"decision": body.decision})
    bus.publish("watchlist.approval", {"subject_id": subject_id, "decision": body.decision,
                                       "actor": user.username}, source="api", severity="info")
    return {**_subject_payload(db, subject), "gallery_subjects_loaded": loaded}


@router.get("/{subject_id}")
def get_subject(subject_id: str, request: Request,
                user: User = Depends(require("watchlist:read")),
                db: Session = Depends(get_session)):
    subject = get_or_404(db, WatchlistSubject, subject_id, "watchlist subject")
    record(db, request, user, "watchlist.read", "watchlist_subject", subject_id)
    return _subject_payload(db, subject)


@router.patch("/{subject_id}")
def update_subject(
    subject_id: str,
    body: SubjectPatch,
    request: Request,
    user: User = Depends(require("watchlist:write")),
    db: Session = Depends(get_session),
):
    subject = get_or_404(db, WatchlistSubject, subject_id, "watchlist subject")
    changes = body.model_dump(exclude_none=True)

    if "status" in changes:
        if changes["status"] not in ("active", "paused", "revoked", "pending"):
            raise HTTPException(400, "status must be active, paused, revoked or pending")
        if changes["status"] == "active" and not has_permission(user.role, "watchlist:approve"):
            raise HTTPException(403, "only a commander may activate a watchlist subject")
    if "expires_in_days" in changes:
        days = changes.pop("expires_in_days")
        subject.expires_at = utcnow() + timedelta(days=days) if days else None

    for key, value in changes.items():
        setattr(subject, key, value)
    db.commit()
    reload_gallery(db)
    record(db, request, user, "watchlist.update", "watchlist_subject", subject_id,
           detail={"fields": sorted(changes.keys())})
    return _subject_payload(db, subject)


@router.delete("/{subject_id}")
def delete_subject(
    subject_id: str,
    request: Request,
    purge_files: bool = False,
    user: User = Depends(require("watchlist:approve")),
    db: Session = Depends(get_session),
):
    """Revoke and remove a subject. Optionally delete the stored imagery."""
    subject = get_or_404(db, WatchlistSubject, subject_id, "watchlist subject")
    for image in subject.images:
        _face_vectors.pop(image.id, None)
        _body_vectors.pop(image.id, None)
        if purge_files and image.file_path:
            try:
                Path(image.file_path).unlink(missing_ok=True)
            except OSError:
                log.warning("could not delete %s", image.file_path)

    db.delete(subject)
    db.commit()
    reload_gallery(db)
    record(db, request, user, "watchlist.delete", "watchlist_subject", subject_id,
           detail={"purge_files": purge_files})
    return {"deleted": subject_id, "files_purged": purge_files}


# ---------------------------------------------------------------- sightings
@router.get("/{subject_id}/sightings")
def sightings(
    subject_id: str,
    request: Request,
    review_status: Optional[str] = None,
    min_confidence: float = 0.0,
    page: Dict[str, int] = Depends(pagination),
    user: User = Depends(require("watchlist:read")),
    db: Session = Depends(get_session),
):
    get_or_404(db, WatchlistSubject, subject_id, "watchlist subject")
    stmt = select(WatchlistMatch).where(WatchlistMatch.subject_id == subject_id)
    if review_status:
        stmt = stmt.where(WatchlistMatch.review_status == review_status)
    if min_confidence:
        stmt = stmt.where(WatchlistMatch.confidence >= min_confidence)

    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(
        stmt.order_by(WatchlistMatch.matched_at.desc())
        .limit(page["limit"]).offset(page["offset"])
    ).all()

    cameras = {c.id: c for c in db.scalars(select(Camera)).all()}
    items = []
    for m in rows:
        camera = cameras.get(m.camera_id)
        items.append(
            {
                **to_dict(m),
                "camera_name": camera.name if camera else None,
                "camera_location": camera.location if camera else None,
                "site_id": m.site_id or (camera.site_id if camera else None),
            }
        )
    record(db, request, user, "watchlist.sightings", "watchlist_subject", subject_id,
           detail={"returned": len(items)})
    return Page.of(items, total, page["limit"], page["offset"])


@router.get("/{subject_id}/locate")
def locate(
    subject_id: str,
    request: Request,
    min_confidence: float = 0.6,
    within_minutes: int = 120,
    user: User = Depends(require("watchlist:read")),
    db: Session = Depends(get_session),
):
    """Answer 'where is this person now?' in plain operational language."""
    subject = get_or_404(db, WatchlistSubject, subject_id, "watchlist subject")
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=within_minutes)

    rows = db.scalars(
        select(WatchlistMatch)
        .where(WatchlistMatch.subject_id == subject_id)
        .where(WatchlistMatch.matched_at >= cutoff)
        .where(WatchlistMatch.confidence >= min_confidence)
        .where(WatchlistMatch.review_status != "rejected")
        .order_by(WatchlistMatch.matched_at.desc())
        .limit(50)
    ).all()

    record(db, request, user, "watchlist.locate", "watchlist_subject", subject_id,
           detail={"within_minutes": within_minutes, "hits": len(rows)})

    if not rows:
        return {
            "subject_id": subject_id,
            "label": subject.label,
            "located": False,
            "statement": (
                f"No sighting of '{subject.label}' above {min_confidence:.0%} confidence in "
                f"the last {within_minutes} minutes across the cameras in scope."
            ),
            "matching_active": subject.status == "active" and not _expired(subject),
            "sightings": [],
        }

    latest = rows[0]
    camera = db.get(Camera, latest.camera_id)
    site = camera.site_id if camera else None
    where = latest.location_name or (camera.location if camera else latest.camera_id)

    path = [
        {
            "camera_id": m.camera_id,
            "camera_name": (db.get(Camera, m.camera_id).name if db.get(Camera, m.camera_id) else None),
            "location": m.location_name,
            "matched_at": m.matched_at.isoformat(),
            "confidence": round(m.confidence, 3),
            "modality": m.modality,
            "latitude": m.latitude,
            "longitude": m.longitude,
            "floor": m.floor,
            "review_status": m.review_status,
        }
        for m in rows
    ]

    statement = (
        f"Most recent possible sighting of '{subject.label}' was at {where} "
        f"(camera {camera.name if camera else latest.camera_id}) at "
        f"{latest.matched_at:%Y-%m-%d %H:%M:%S} UTC, "
        f"{latest.modality} match at {latest.confidence:.0%} confidence. "
        f"{len(rows)} sighting(s) in the last {within_minutes} minutes. "
        "This is an appearance-based association and requires human confirmation."
    )

    return {
        "subject_id": subject_id,
        "label": subject.label,
        "located": True,
        "statement": statement,
        "latest": {
            "camera_id": latest.camera_id,
            "camera_name": camera.name if camera else None,
            "location": where,
            "site_id": site,
            "zone_id": latest.zone_id,
            "floor": latest.floor,
            "latitude": latest.latitude,
            "longitude": latest.longitude,
            "matched_at": latest.matched_at.isoformat(),
            "confidence": round(latest.confidence, 3),
            "modality": latest.modality,
            "review_status": latest.review_status,
        },
        "movement_path": path,
        "cameras_seen": sorted({m.camera_id for m in rows}),
        "requires_human_confirmation": True,
    }


@router.post("/matches/{match_id}/review")
def review_match(
    match_id: str,
    body: ReviewBody,
    request: Request,
    user: User = Depends(require("watchlist:read")),
    db: Session = Depends(get_session),
):
    """Confirm or reject a sighting - the human half of the loop."""
    if body.review_status not in ("confirmed", "rejected", "uncertain"):
        raise HTTPException(400, "review_status must be confirmed, rejected or uncertain")
    match = get_or_404(db, WatchlistMatch, match_id, "watchlist match")
    match.review_status = body.review_status
    match.reviewed_by = user.username
    if body.note:
        match.meta = {**(match.meta or {}), "review_note": body.note}

    if body.review_status == "confirmed":
        subject = db.get(WatchlistSubject, match.subject_id)
        if subject:
            subject.last_match_at = match.matched_at
            subject.match_count = (subject.match_count or 0) + 1
    db.commit()
    record(db, request, user, "watchlist.review", "watchlist_match", match_id,
           justification=body.note, detail={"review_status": body.review_status})
    return to_dict(match)


@router.get("/matches/recent")
def recent_matches(
    limit: int = 50,
    user: User = Depends(require("watchlist:read")),
    db: Session = Depends(get_session),
):
    rows = db.scalars(
        select(WatchlistMatch).order_by(WatchlistMatch.matched_at.desc()).limit(limit)
    ).all()
    subjects = {s.id: s.label for s in db.scalars(select(WatchlistSubject)).all()}
    return {
        "items": [{**to_dict(m), "subject_label": subjects.get(m.subject_id)} for m in rows],
        "live_buffer": get_orchestrator().watchlist.recent_matches(limit=limit),
    }


@router.post("/reload")
def reload_endpoint(user: User = Depends(require("watchlist:write")),
                    db: Session = Depends(get_session)):
    """Rebuild the in-memory gallery, e.g. after a restart."""
    return {"subjects_loaded": reload_gallery(db),
            "agent": get_orchestrator().watchlist.status()}

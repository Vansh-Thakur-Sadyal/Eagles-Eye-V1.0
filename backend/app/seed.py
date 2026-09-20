"""First-run bootstrap.

Creates only what a fresh install genuinely needs: an administrator account, a
site with a couple of zones, one synthetic camera so the platform can be
demonstrated before hardware is attached, and the model-registry rows that
describe what the deployment expects to load.

This is real, editable data, not UI placeholder content - every row can be
renamed or deleted from the dashboard. It runs once; a database that already
has users is left untouched.
"""
from __future__ import annotations

import logging
import os
import secrets
from typing import List

from sqlalchemy import func, select

from .config import get_settings
from .core.security import hash_password
from .db import session_scope
from .models import Camera, ModelRecord, Site, User, Zone, new_id

log = logging.getLogger("sentinel.seed")

DEFAULT_ADMIN_USERNAME = os.environ.get("SENTINEL_ADMIN_USERNAME", "admin")


def seed_if_empty() -> None:
    with session_scope() as db:
        if db.scalar(select(func.count()).select_from(User)):
            return
        log.info("empty database detected - seeding initial data")
        password = _seed_admin(db)
        site = _seed_site(db)
        _seed_zones(db, site)
        _seed_camera(db, site)
        _seed_models(db)

    log.info("=" * 78)
    log.info("  Eagles Eye first-run credentials")
    log.info("    username: %s", DEFAULT_ADMIN_USERNAME)
    log.info("    password: %s", password)
    log.info("  Change this immediately via POST /api/auth/password.")
    log.info("=" * 78)

    path = get_settings().storage_dir / "FIRST_RUN_CREDENTIALS.txt"
    path.write_text(
        f"Eagles Eye first-run administrator\n"
        f"username: {DEFAULT_ADMIN_USERNAME}\n"
        f"password: {password}\n\n"
        f"Change this password immediately, then delete this file.\n",
        encoding="utf-8",
    )


def _seed_admin(db) -> str:
    password = os.environ.get("SENTINEL_ADMIN_PASSWORD") or secrets.token_urlsafe(12)
    db.add(
        User(
            id=new_id("USR"),
            username=DEFAULT_ADMIN_USERNAME,
            full_name="System Administrator",
            password_hash=hash_password(password),
            role="admin",
            post="Security Operations Centre",
        )
    )
    return password


def _seed_site(db) -> Site:
    site = Site(
        id=new_id("SITE"),
        name="Primary Site",
        site_type="generic",
        timezone="UTC",
        floors=[{"level": 0, "name": "Ground"}],
        meta={"seeded": True, "note": "Rename or replace this from Camera Management."},
    )
    db.add(site)
    db.flush()
    return site


def _seed_zones(db, site: Site) -> None:
    zones = [
        Zone(
            id=new_id("ZONE"),
            site_id=site.id,
            name="Public Concourse",
            zone_type="public",
            floor=0,
            # Normalised 0-1 so a zone survives a change of camera resolution.
            polygon=[[0.02, 0.30], [0.68, 0.30], [0.68, 0.98], [0.02, 0.98]],
            risk_weight=1.0,
            meta={"seeded": True},
        ),
        Zone(
            id=new_id("ZONE"),
            site_id=site.id,
            name="Restricted Area",
            zone_type="restricted",
            floor=0,
            polygon=[[0.72, 0.55], [0.97, 0.55], [0.97, 0.92], [0.72, 0.92]],
            authorized_roles=["staff", "security"],
            risk_weight=1.4,
            meta={"seeded": True},
        ),
    ]
    for zone in zones:
        db.add(zone)
    db.flush()


def _seed_camera(db, site: Site) -> None:
    public_zone = db.scalar(
        select(Zone).where(Zone.site_id == site.id, Zone.zone_type == "public")
    )
    db.add(
        Camera(
            id=new_id("CAM"),
            site_id=site.id,
            zone_id=public_zone.id if public_zone else None,
            name="Demo Camera (synthetic)",
            location="Primary Site - Concourse",
            source_type="synthetic",
            source_uri="",
            width=960,
            height=540,
            fps=12,
            latitude=None,
            longitude=None,
            floor=0,
            orientation_deg=90.0,
            field_of_view_deg=82.0,
            range_m=25.0,
            status="offline",
            meta={
                "seeded": True,
                "crowd": 9,
                # The scene renders schematic figures, which a trained detector
                # correctly refuses to call people. Pinning the motion backend
                # keeps the demo working once YOLO is installed.
                "detector": "motion",
                "note": "A generated scene containing loitering, counter-flow, a following "
                        "pair and an abandoned bag, so the pipeline can be demonstrated "
                        "before a real camera is attached. It pins the motion detector "
                        "because its figures are schematic, not photographic. Delete it "
                        "once you attach real hardware.",
            },
        )
    )


def _seed_models(db) -> None:
    s = get_settings()
    records = [
        ModelRecord(
            id=new_id("MDL"), name=s.yolo_weights, task="detection", framework="ultralytics",
            version="11", source="ultralytics", status="registered",
            params={"conf": s.yolo_conf, "iou": s.yolo_iou, "imgsz": s.yolo_imgsz},
            notes="Primary real-time detector. Install ultralytics and the weights to activate.",
        ),
        ModelRecord(
            id=new_id("MDL"), name=s.locateanything_model, task="detection",
            framework="transformers", version="3B", source="huggingface",
            status="registered" if s.locateanything_enabled else "archived",
            params={"enabled": s.locateanything_enabled},
            notes="Open-vocabulary grounding for free-text prompts. Runs on keyframes and "
                  "forensic queries, not every frame.",
        ),
        ModelRecord(
            id=new_id("MDL"), name=s.tracker, task="tracking", framework="native",
            version="1.0", status="active",
            notes="Built-in ByteTrack implementation with optional appearance gating.",
        ),
        ModelRecord(
            id=new_id("MDL"), name=s.reid_weights, task="reid", framework="torchreid",
            version="1.0", status="registered",
            notes="Person re-identification embeddings. Falls back to a handcrafted "
                  "striped-histogram descriptor when torchreid is absent.",
        ),
        ModelRecord(
            id=new_id("MDL"), name=s.face_model, task="face", framework="insightface",
            version="1.0", status="registered",
            notes="Face embeddings for authorised watchlist matching only.",
        ),
        ModelRecord(
            id=new_id("MDL"), name=s.embedding_model, task="embedding",
            framework="sentence-transformers", version="1.0", status="registered",
            notes="Text embeddings for incident retrieval.",
        ),
    ]
    for record in records:
        db.add(record)

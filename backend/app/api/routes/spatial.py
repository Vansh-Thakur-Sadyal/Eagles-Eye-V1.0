"""Sites, zones, digital twin, AR and VR endpoints (spec S17-S20)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...agents.orchestrator import get_orchestrator
from ...agents.spatial import SpatialAgent
from ...core.security import current_user, require
from ...db import get_session
from ...models import Camera, Incident, Site, User, Zone, new_id
from ...pipeline.runner import get_pipeline
from ..deps import Page, get_or_404, pagination, record, to_dict

router = APIRouter(prefix="/api/spatial", tags=["spatial"])

ZONE_TYPES = ("public", "restricted", "secure", "transit", "platform", "gate", "perimeter")
SITE_TYPES = ("airport", "railway_station", "metro", "bus_terminal", "stadium", "campus",
              "smart_city", "mall", "industrial", "corporate", "generic")


# -------------------------------------------------------------------- sites
class SiteIn(BaseModel):
    name: str
    site_type: str = "generic"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timezone: str = "UTC"
    twin_model_url: Optional[str] = None
    floors: List[Dict[str, Any]] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


@router.get("/sites")
def list_sites(user: User = Depends(current_user), db: Session = Depends(get_session)):
    rows = db.scalars(select(Site).order_by(Site.name)).all()
    out = []
    for site in rows:
        cameras = db.scalars(select(Camera).where(Camera.site_id == site.id)).all()
        zones = db.scalars(select(Zone).where(Zone.site_id == site.id)).all()
        out.append({**to_dict(site), "camera_count": len(cameras), "zone_count": len(zones)})
    return {"items": out, "site_types": list(SITE_TYPES)}


@router.post("/sites")
def create_site(
    body: SiteIn,
    request: Request,
    user: User = Depends(require("camera:write")),
    db: Session = Depends(get_session),
):
    site = Site(id=new_id("SITE"), **body.model_dump())
    db.add(site)
    db.commit()
    record(db, request, user, "site.create", "site", site.id, detail={"name": body.name})
    return to_dict(site)


@router.patch("/sites/{site_id}")
def update_site(
    site_id: str,
    body: Dict[str, Any],
    request: Request,
    user: User = Depends(require("camera:write")),
    db: Session = Depends(get_session),
):
    site = get_or_404(db, Site, site_id, "site")
    for key, value in body.items():
        if hasattr(site, key) and key not in ("id", "created_at"):
            setattr(site, key, value)
    db.commit()
    record(db, request, user, "site.update", "site", site_id)
    return to_dict(site)


# -------------------------------------------------------------------- zones
class ZoneIn(BaseModel):
    name: str
    zone_type: str = "public"
    site_id: Optional[str] = None
    floor: int = 0
    polygon: List[List[float]] = Field(default_factory=list)
    authorized_roles: List[str] = Field(default_factory=list)
    expected_flow_deg: Optional[float] = None
    capacity: Optional[int] = None
    risk_weight: float = 1.0
    active: bool = True
    meta: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("zone_type")
    @classmethod
    def _zone_type(cls, v: str) -> str:
        if v not in ZONE_TYPES:
            raise ValueError(f"zone_type must be one of {ZONE_TYPES}")
        return v

    @field_validator("polygon")
    @classmethod
    def _polygon(cls, v: List[List[float]]) -> List[List[float]]:
        if v and len(v) < 3:
            raise ValueError("a polygon needs at least three points")
        for point in v:
            if len(point) != 2:
                raise ValueError("each polygon point must be [x, y]")
            if not all(0.0 <= float(c) <= 1.0 for c in point):
                raise ValueError(
                    "polygon points must be normalised to 0-1 so a zone survives a "
                    "camera resolution change"
                )
        return v


@router.get("/zones")
def list_zones(
    site_id: Optional[str] = None,
    zone_type: Optional[str] = None,
    user: User = Depends(current_user),
    db: Session = Depends(get_session),
):
    stmt = select(Zone).order_by(Zone.name)
    if site_id:
        stmt = stmt.where(Zone.site_id == site_id)
    if zone_type:
        stmt = stmt.where(Zone.zone_type == zone_type)
    return {"items": [to_dict(z) for z in db.scalars(stmt).all()], "zone_types": list(ZONE_TYPES)}


@router.post("/zones")
def create_zone(
    body: ZoneIn,
    request: Request,
    user: User = Depends(require("camera:write")),
    db: Session = Depends(get_session),
):
    zone = Zone(id=new_id("ZONE"), **body.model_dump())
    db.add(zone)
    db.commit()
    record(db, request, user, "zone.create", "zone", zone.id,
           detail={"name": body.name, "zone_type": body.zone_type})
    _restart_affected(db, zone.site_id)
    return to_dict(zone)


@router.patch("/zones/{zone_id}")
def update_zone(
    zone_id: str,
    body: Dict[str, Any],
    request: Request,
    user: User = Depends(require("camera:write")),
    db: Session = Depends(get_session),
):
    zone = get_or_404(db, Zone, zone_id, "zone")
    for key, value in body.items():
        if hasattr(zone, key) and key not in ("id", "created_at"):
            setattr(zone, key, value)
    db.commit()
    record(db, request, user, "zone.update", "zone", zone_id, detail={"fields": sorted(body)})
    _restart_affected(db, zone.site_id)
    return to_dict(zone)


@router.delete("/zones/{zone_id}")
def delete_zone(
    zone_id: str,
    request: Request,
    user: User = Depends(require("camera:write")),
    db: Session = Depends(get_session),
):
    zone = get_or_404(db, Zone, zone_id, "zone")
    site_id = zone.site_id
    db.delete(zone)
    db.commit()
    record(db, request, user, "zone.delete", "zone", zone_id)
    _restart_affected(db, site_id)
    return {"deleted": zone_id}


def _restart_affected(db: Session, site_id: Optional[str]) -> None:
    """Zone edits only take effect in a worker after it reloads its runtime."""
    from .cameras import _runtime

    pipeline = get_pipeline()
    for camera in db.scalars(select(Camera).where(Camera.site_id == site_id)).all():
        if pipeline.is_running(camera.id):
            pipeline.stop(camera.id)
            pipeline.start(_runtime(db, camera))


# ------------------------------------------------------------- digital twin
@router.get("/twin")
def twin(
    site_id: Optional[str] = None,
    user: User = Depends(current_user),
    db: Session = Depends(get_session),
):
    """One payload driving the 3D twin, the map and the VR command centre."""
    cam_stmt = select(Camera)
    zone_stmt = select(Zone).where(Zone.active.is_(True))
    if site_id:
        cam_stmt = cam_stmt.where(Camera.site_id == site_id)
        zone_stmt = zone_stmt.where(Zone.site_id == site_id)

    cameras = db.scalars(cam_stmt).all()
    zones = db.scalars(zone_stmt).all()
    sites = db.scalars(
        select(Site).where(Site.id == site_id) if site_id else select(Site)
    ).all()

    incidents = db.scalars(
        select(Incident)
        .where(Incident.status.in_(("open", "acknowledged", "dispatched", "investigating")))
        .order_by(Incident.risk_score.desc())
        .limit(100)
    ).all()

    spatial = get_orchestrator().spatial
    live_entities = spatial.scene()
    pipeline = get_pipeline()

    return {
        "sites": [to_dict(s) for s in sites],
        "zones": [to_dict(z) for z in zones],
        "cameras": [
            {
                **to_dict(c, exclude={"password"}),
                "running": pipeline.is_running(c.id),
                "coverage": SpatialAgent.camera_coverage(to_dict(c)),
            }
            for c in cameras
        ],
        "incidents": [
            {
                "id": i.id,
                "title": i.title,
                "severity": i.severity,
                "risk_score": i.risk_score,
                "camera_id": i.camera_id,
                "zone_id": i.zone_id,
                "location": i.location,
                "started_at": i.started_at.isoformat(),
                "status": i.status,
            }
            for i in incidents
        ],
        "live_entities": live_entities,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/coverage")
def coverage(site_id: Optional[str] = None, user: User = Depends(current_user),
             db: Session = Depends(get_session)):
    """Camera field-of-view wedges, for the spatial coverage map."""
    stmt = select(Camera)
    if site_id:
        stmt = stmt.where(Camera.site_id == site_id)
    cameras = db.scalars(stmt).all()
    wedges = [SpatialAgent.camera_coverage(to_dict(c)) for c in cameras]
    placed = [w for w in wedges if w]
    return {
        "coverage": placed,
        "cameras_total": len(cameras),
        "cameras_geolocated": len(placed),
        "cameras_without_coordinates": [
            {"id": c.id, "name": c.name} for c in cameras if c.latitude is None
        ],
    }


# ----------------------------------------------------------------------- AR
@router.get("/ar/feed")
def ar_feed(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    radius_m: float = 500.0,
    floor: Optional[int] = None,
    user: User = Depends(current_user),
    db: Session = Depends(get_session),
):
    """What a field device should overlay right now (spec S18)."""
    from ...vision.geometry import haversine_m

    incidents = db.scalars(
        select(Incident)
        .where(Incident.status.in_(("open", "acknowledged", "dispatched", "investigating")))
        .order_by(Incident.risk_score.desc())
        .limit(50)
    ).all()
    cameras = {c.id: to_dict(c, exclude={"password"}) for c in db.scalars(select(Camera)).all()}

    overlays = []
    for incident in incidents:
        camera = cameras.get(incident.camera_id)
        if camera is None:
            continue
        if floor is not None and camera.get("floor") != floor:
            continue
        distance = None
        if latitude is not None and longitude is not None:
            if camera.get("latitude") is None:
                continue
            distance = haversine_m(latitude, longitude, camera["latitude"], camera["longitude"])
            if distance > radius_m:
                continue
        payload = SpatialAgent.ar_payload(
            {
                "id": incident.id,
                "title": incident.title,
                "severity": incident.severity,
                "risk_score": incident.risk_score,
                "summary": incident.summary,
                "zone_id": incident.zone_id,
            },
            camera,
        )
        payload["distance_m"] = round(distance, 1) if distance is not None else None
        overlays.append(payload)

    overlays.sort(key=lambda o: (o["distance_m"] if o["distance_m"] is not None else 1e9))
    restricted = db.scalars(
        select(Zone).where(Zone.zone_type.in_(("restricted", "secure")), Zone.active.is_(True))
    ).all()

    return {
        "overlays": overlays,
        "restricted_zones": [
            {"id": z.id, "name": z.name, "floor": z.floor, "polygon": z.polygon}
            for z in restricted
        ],
        "observer": {"latitude": latitude, "longitude": longitude, "floor": floor,
                     "radius_m": radius_m},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/ar/incident/{incident_id}")
def ar_incident(incident_id: str, user: User = Depends(current_user),
                db: Session = Depends(get_session)):
    incident = get_or_404(db, Incident, incident_id, "incident")
    camera = db.get(Camera, incident.camera_id) if incident.camera_id else None
    if camera is None:
        raise HTTPException(404, "this incident has no camera to anchor an AR overlay to")
    payload = SpatialAgent.ar_payload(
        {
            "id": incident.id, "title": incident.title, "severity": incident.severity,
            "risk_score": incident.risk_score, "summary": incident.summary,
            "zone_id": incident.zone_id,
        },
        to_dict(camera, exclude={"password"}),
    )
    payload["timeline"] = [to_dict(e) for e in incident.timeline]
    return payload


# ----------------------------------------------------------------------- VR
@router.get("/vr/session")
def vr_session(
    site_id: Optional[str] = None,
    replay_incident_id: Optional[str] = None,
    user: User = Depends(current_user),
    db: Session = Depends(get_session),
):
    """Bootstrap payload for the VR command centre (spec S19)."""
    scene = twin(site_id=site_id, user=user, db=db)
    replay = None
    if replay_incident_id:
        incident = get_or_404(db, Incident, replay_incident_id, "incident")
        replay = {
            "incident": to_dict(incident),
            "timeline": [to_dict(e) for e in incident.timeline],
            "camera_ids": incident.camera_ids or [incident.camera_id],
            "track_ids": incident.track_ids,
        }
    return {
        "scene": scene,
        "replay": replay,
        "capabilities": {
            "camera_feeds": True,
            "incident_jump": True,
            "trajectory_replay": True,
            "crowd_density_layer": True,
            "threat_zones": True,
            "ai_query": True,
        },
        "renderer_hint": {
            "webgpu": True,
            "power_preference": "high-performance",
            "note": "Request the high-performance WebGPU adapter so the viewer's discrete "
                    "GPU is used for rendering without any permission prompt.",
        },
    }

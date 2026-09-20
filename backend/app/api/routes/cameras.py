"""Camera management, capture control and live streaming.

Covers the spec's camera-metadata record (S47) and the operator-facing
attachment flow for a USB/external webcam or an ESP32-CAM board, including a
discovery endpoint and a live probe so a device can be validated before it is
saved.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...agents.orchestrator import get_orchestrator
from ...agents.spatial import SpatialAgent
from ...core.events import bus
from ...core.security import current_user, require
from ...db import get_session
from ...models import Camera, Site, User, Zone, new_id, utcnow
from ...pipeline.capture import SOURCE_TYPES, list_local_webcams, probe_esp32cam
from ...pipeline.runner import CameraRuntime, get_pipeline
from ...portal.scope import filter_cameras, may_see_camera, org_of
from ..deps import Page, get_or_404, pagination, record, to_dict

router = APIRouter(prefix="/api/cameras", tags=["cameras"])

SOURCE_HELP = {
    "webcam": "Local USB or built-in camera. source_uri is the device index, e.g. '0'.",
    "esp32cam": "ESP32-CAM board. source_uri is its address, e.g. '192.168.1.42' "
                "or 'http://192.168.1.42'. Stream and snapshot endpoints are auto-detected.",
    "rtsp": "IP camera or NVR, e.g. 'rtsp://192.168.1.10:554/stream1'.",
    "http_mjpeg": "Generic MJPEG-over-HTTP stream URL.",
    "file": "A video file path, for replay and dataset testing.",
    "synthetic": "Built-in generated scene for demos and testing without hardware.",
}


class CameraIn(BaseModel):
    name: str
    source_type: str = "rtsp"
    source_uri: str = ""
    location: str = ""
    site_id: Optional[str] = None
    zone_id: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    floor: int = 0
    orientation_deg: float = 0.0
    tilt_deg: float = 0.0
    field_of_view_deg: float = 82.0
    range_m: float = 25.0
    fps: int = 12
    width: int = 1280
    height: int = 720
    rotation: int = 0
    enabled_agents: List[str] = Field(default_factory=list)
    detection_classes: List[str] = Field(default_factory=list)
    privacy_redaction: Optional[bool] = None
    line_crossings: List[Dict[str, Any]] = Field(default_factory=list)
    homography: Optional[List[List[float]]] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    active: bool = True

    @field_validator("source_type")
    @classmethod
    def _known_source(cls, v: str) -> str:
        if v not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        return v

    @field_validator("rotation")
    @classmethod
    def _rotation(cls, v: int) -> int:
        if v % 90 != 0:
            raise ValueError("rotation must be 0, 90, 180 or 270")
        return v % 360


class CameraPatch(BaseModel):
    name: Optional[str] = None
    source_type: Optional[str] = None
    source_uri: Optional[str] = None
    location: Optional[str] = None
    site_id: Optional[str] = None
    zone_id: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    floor: Optional[int] = None
    orientation_deg: Optional[float] = None
    tilt_deg: Optional[float] = None
    field_of_view_deg: Optional[float] = None
    range_m: Optional[float] = None
    fps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    rotation: Optional[int] = None
    enabled_agents: Optional[List[str]] = None
    detection_classes: Optional[List[str]] = None
    privacy_redaction: Optional[bool] = None
    line_crossings: Optional[List[Dict[str, Any]]] = None
    homography: Optional[List[List[float]]] = None
    meta: Optional[Dict[str, Any]] = None
    active: Optional[bool] = None


def _runtime(db: Session, camera: Camera) -> CameraRuntime:
    """Assemble everything a worker needs, including the zones it must watch."""
    zones = db.scalars(
        select(Zone).where(Zone.active.is_(True)).where(
            (Zone.site_id == camera.site_id) | (Zone.id == camera.zone_id)
        )
    ).all()
    return CameraRuntime(
        camera_id=camera.id,
        name=camera.name,
        source_type=camera.source_type,
        source_uri=camera.source_uri,
        width=camera.width,
        height=camera.height,
        fps=camera.fps,
        rotation=camera.rotation,
        username=camera.username,
        password=camera.password,
        zone_id=camera.zone_id,
        site_id=camera.site_id,
        location=camera.location,
        zones=[
            {
                "id": z.id,
                "name": z.name,
                "zone_type": z.zone_type,
                "polygon": z.polygon,
                "authorized_roles": z.authorized_roles,
                "risk_weight": z.risk_weight,
                "expected_flow_deg": z.expected_flow_deg,
            }
            for z in zones
        ],
        line_crossings=camera.line_crossings or [],
        expected_flow_deg=next(
            (z.expected_flow_deg for z in zones if z.id == camera.zone_id and z.expected_flow_deg),
            None,
        ),
        homography=camera.homography,
        latitude=camera.latitude,
        longitude=camera.longitude,
        floor=camera.floor,
        field_of_view_deg=camera.field_of_view_deg,
        range_m=camera.range_m,
        orientation_deg=camera.orientation_deg,
        enabled_agents=camera.enabled_agents or [],
        detection_classes=camera.detection_classes or [],
        privacy_redaction=camera.privacy_redaction,
        options=dict(camera.meta or {}),
    )


def _serialise(camera: Camera, *, include_secrets: bool = False) -> Dict[str, Any]:
    data = to_dict(camera, exclude=() if include_secrets else {"password"})
    data.setdefault("meta", camera.meta or {})
    pipeline = get_pipeline()
    worker = pipeline.get(camera.id)
    data["running"] = pipeline.is_running(camera.id)
    data["runtime"] = worker.info() if worker else None
    data["source_help"] = SOURCE_HELP.get(camera.source_type)
    data["coverage"] = SpatialAgent.camera_coverage(data)
    return data


# ------------------------------------------------------------------- CRUD
@router.get("")
def list_cameras(
    status_filter: Optional[str] = None,
    site_id: Optional[str] = None,
    page: Dict[str, int] = Depends(pagination),
    user: User = Depends(current_user),
    db: Session = Depends(get_session),
):
    stmt = select(Camera).order_by(Camera.name)
    if status_filter:
        stmt = stmt.where(Camera.status == status_filter)
    if site_id:
        stmt = stmt.where(Camera.site_id == site_id)
    rows = filter_cameras(db, user, list(db.scalars(stmt).all()))
    total = len(rows)
    window = rows[page["offset"] : page["offset"] + page["limit"]]
    return Page.of([_serialise(c) for c in window], total, page["limit"], page["offset"])


@router.get("/source-types")
def source_types(user: User = Depends(current_user)):
    return {
        "types": [
            {"value": k, "label": k.replace("_", " ").title(), "help": SOURCE_HELP.get(k, "")}
            for k in sorted(SOURCE_TYPES)
        ]
    }


@router.get("/discover/webcams")
def discover_webcams(max_index: int = 6, user: User = Depends(require("camera:write"))):
    """Probe local video device indices so the UI can list real hardware."""
    devices = list_local_webcams(max_index=max_index)
    return {
        "devices": devices,
        "count": len(devices),
        "note": "Each entry can be attached directly as a camera with source_type 'webcam' "
                "and source_uri set to its index.",
    }


class Esp32Probe(BaseModel):
    address: str


@router.post("/discover/esp32")
def discover_esp32(body: Esp32Probe, user: User = Depends(require("camera:write"))):
    """Check an ESP32-CAM board and report which of its endpoints respond."""
    result = probe_esp32cam(body.address)
    if not result.get("reachable"):
        result["hint"] = (
            "No endpoint responded. Confirm the board is on the same network, that the "
            "address is right, and that the CameraWebServer sketch is running. The stream "
            "is normally on port 81 and single shots on /capture."
        )
    return result


@router.get("/{camera_id}")
def get_camera(camera_id: str, user: User = Depends(current_user), db: Session = Depends(get_session)):
    camera = get_or_404(db, Camera, camera_id, "camera")
    # Another organisation's camera must look like it does not exist.
    if not may_see_camera(db, user, camera):
        raise HTTPException(404, "camera not found")
    return _serialise(camera)


@router.post("")
def create_camera(
    body: CameraIn,
    request: Request,
    user: User = Depends(require("camera:write")),
    db: Session = Depends(get_session),
):
    if body.site_id and db.get(Site, body.site_id) is None:
        raise HTTPException(400, f"site '{body.site_id}' does not exist")
    if body.zone_id and db.get(Zone, body.zone_id) is None:
        raise HTTPException(400, f"zone '{body.zone_id}' does not exist")

    camera = Camera(id=new_id("CAM"), **body.model_dump())
    # Bind it to the creator's organisation; that is what keeps tenants apart.
    org = org_of(db, user)
    if org:
        camera.meta = {**(camera.meta or {}), "organisation_id": org}
    db.add(camera)
    db.commit()
    record(db, request, user, "camera.create", "camera", camera.id,
           detail={"name": body.name, "source_type": body.source_type})
    bus.publish("camera.created", {"camera_id": camera.id, "name": camera.name}, source="api")
    return _serialise(camera)


@router.patch("/{camera_id}")
def update_camera(
    camera_id: str,
    body: CameraPatch,
    request: Request,
    user: User = Depends(require("camera:write")),
    db: Session = Depends(get_session),
):
    camera = get_or_404(db, Camera, camera_id, "camera")
    changes = body.model_dump(exclude_none=True)
    if "source_type" in changes and changes["source_type"] not in SOURCE_TYPES:
        raise HTTPException(400, f"source_type must be one of {sorted(SOURCE_TYPES)}")

    for key, value in changes.items():
        setattr(camera, key, value)
    db.commit()
    record(db, request, user, "camera.update", "camera", camera.id,
           detail={"fields": sorted(changes.keys())})

    # A running worker picks up configuration changes on restart.
    pipeline = get_pipeline()
    if pipeline.is_running(camera.id):
        pipeline.stop(camera.id)
        pipeline.start(_runtime(db, camera))
    return _serialise(camera)


@router.delete("/{camera_id}")
def delete_camera(
    camera_id: str,
    request: Request,
    user: User = Depends(require("camera:write")),
    db: Session = Depends(get_session),
):
    camera = get_or_404(db, Camera, camera_id, "camera")
    get_pipeline().stop(camera_id)
    db.delete(camera)
    db.commit()
    record(db, request, user, "camera.delete", "camera", camera_id, detail={"name": camera.name})
    return {"deleted": camera_id}


# ---------------------------------------------------------------- control
@router.post("/{camera_id}/start")
def start_camera(
    camera_id: str,
    request: Request,
    user: User = Depends(require("camera:write")),
    db: Session = Depends(get_session),
):
    camera = get_or_404(db, Camera, camera_id, "camera")
    if not camera.active:
        raise HTTPException(400, "camera is marked inactive; set active=true before starting it")
    try:
        info = get_pipeline().start(_runtime(db, camera))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc

    camera.status = info.get("status", "connecting")
    camera.status_detail = info.get("status_detail")
    db.commit()
    record(db, request, user, "camera.start", "camera", camera_id)
    return info


@router.post("/{camera_id}/stop")
def stop_camera(
    camera_id: str,
    request: Request,
    user: User = Depends(require("camera:write")),
    db: Session = Depends(get_session),
):
    camera = get_or_404(db, Camera, camera_id, "camera")
    stopped = get_pipeline().stop(camera_id)
    camera.status = "offline"
    camera.status_detail = "stopped by operator"
    db.commit()
    record(db, request, user, "camera.stop", "camera", camera_id)
    return {"stopped": stopped, "camera_id": camera_id}


@router.post("/{camera_id}/pause")
def pause_camera(camera_id: str, user: User = Depends(require("camera:write"))):
    worker = get_pipeline().get(camera_id)
    if worker is None:
        raise HTTPException(404, "camera is not running")
    worker.pause()
    return worker.info()


@router.post("/{camera_id}/resume")
def resume_camera(camera_id: str, user: User = Depends(require("camera:write"))):
    worker = get_pipeline().get(camera_id)
    if worker is None:
        raise HTTPException(404, "camera is not running")
    worker.resume()
    return worker.info()


@router.post("/start-all")
def start_all(
    request: Request,
    user: User = Depends(require("camera:write")),
    db: Session = Depends(get_session),
):
    pipeline = get_pipeline()
    results = []
    for camera in db.scalars(select(Camera).where(Camera.active.is_(True))).all():
        if pipeline.is_running(camera.id):
            continue
        try:
            results.append(pipeline.start(_runtime(db, camera)))
        except RuntimeError as exc:
            results.append({"camera_id": camera.id, "error": str(exc)})
    record(db, request, user, "camera.start_all", "camera", None, detail={"count": len(results)})
    return {"started": results}


@router.post("/stop-all")
def stop_all(request: Request, user: User = Depends(require("camera:write")),
             db: Session = Depends(get_session)):
    get_pipeline().stop_all()
    for camera in db.scalars(select(Camera)).all():
        camera.status = "offline"
    db.commit()
    record(db, request, user, "camera.stop_all", "camera", None)
    return {"ok": True}


# --------------------------------------------------------------- streaming
@router.get("/{camera_id}/stream")
def stream(camera_id: str, db: Session = Depends(get_session)):
    """MJPEG stream of the annotated, privacy-redacted frame.

    Served without a bearer header because <img src> cannot set one; the frames
    are already redacted by the privacy agent before they reach this point.
    """
    worker = get_pipeline().get(camera_id)
    if worker is None:
        raise HTTPException(404, "camera is not running - start it first")

    def generate():
        import time

        boundary = b"--frame\r\n"
        worker.add_viewer()
        try:
            while True:
                jpeg = worker.latest_jpeg()
                if jpeg is None:
                    time.sleep(0.1)
                    continue
                yield (
                    boundary
                    + b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
                time.sleep(1.0 / max(1, worker.runtime.fps))
        finally:
            # A disconnect (tab closed, navigation away) must release the
            # viewer, or the worker keeps encoding frames for nobody.
            worker.remove_viewer()

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/{camera_id}/snapshot")
def snapshot(camera_id: str, user: User = Depends(current_user)):
    worker = get_pipeline().get(camera_id)
    if worker is None:
        raise HTTPException(404, "camera is not running")
    jpeg = worker.latest_jpeg()
    if jpeg is None:
        raise HTTPException(503, "no frame available yet")
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@router.get("/{camera_id}/tracks")
def camera_tracks(camera_id: str, user: User = Depends(current_user)):
    worker = get_pipeline().get(camera_id)
    if worker is None:
        raise HTTPException(404, "camera is not running")
    return {"camera_id": camera_id, "tracks": worker.snapshot_tracks()}


@router.get("/{camera_id}/crowd")
def camera_crowd(camera_id: str, user: User = Depends(current_user)):
    metrics = get_orchestrator().crowd.latest(camera_id)
    if metrics is None:
        worker = get_pipeline().get(camera_id)
        if getattr(worker, "remote", False):           # runs on an edge node
            metrics = worker.latest_crowd()
    if metrics is None:
        raise HTTPException(404, "no crowd telemetry for this camera yet")
    return metrics


@router.get("/{camera_id}/objects")
def camera_objects(camera_id: str, user: User = Depends(current_user)):
    return {"camera_id": camera_id, "objects": get_orchestrator().objects.snapshot(camera_id)}


@router.get("/{camera_id}/relationships")
def camera_relationships(camera_id: str, user: User = Depends(current_user)):
    return {"camera_id": camera_id, "pairs": get_orchestrator().relationship.pair_report(camera_id)}

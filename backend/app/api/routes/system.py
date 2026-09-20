"""System, compute and configuration endpoints.

Includes the GPU toggle the dashboard drives, the remote-GPU worker registry,
agent enable/disable, policy editing and the model registry.
"""
from __future__ import annotations

import platform
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...agents.orchestrator import get_orchestrator
from ...automation.n8n import dispatcher
from ...config import get_settings, load_policy, save_policy
from ...core.device import get_device_manager
from ...core.events import bus, hub
from ...core.persistence import persistence
from ...core.security import current_user, require
from ...db import get_session
from ...models import Camera, Incident, ModelRecord, SystemMetric, Track, User, new_id, utcnow
from ...pipeline.runner import get_pipeline
from ...rag.forensic import get_forensic
from ...rag.llm import get_llm, reset_llm
from ...rag.vectorstore import get_store
from ..deps import Page, record, to_dict

router = APIRouter(prefix="/api/system", tags=["system"])

_BOOT = time.time()


# ------------------------------------------------------------------- health
@router.get("/health")
def health() -> Dict[str, Any]:
    """Unauthenticated liveness probe."""
    dm = get_device_manager()
    return {
        "status": "ok",
        "service": "sentinel-ai",
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - _BOOT, 1),
        "device": dm.active_device,
        "gpu_enabled": dm.gpu_enabled,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/info")
def info(user: User = Depends(current_user), db: Session = Depends(get_session)) -> Dict[str, Any]:
    s = get_settings()
    dm = get_device_manager()
    pipeline = get_pipeline()

    cameras_total = db.scalar(select(func.count()).select_from(Camera)) or 0
    cameras_online = db.scalar(
        select(func.count()).select_from(Camera).where(Camera.status == "online")
    ) or 0
    incidents_open = db.scalar(
        select(func.count()).select_from(Incident).where(Incident.status == "open")
    ) or 0
    tracks_active = db.scalar(
        select(func.count()).select_from(Track).where(Track.active.is_(True))
    ) or 0

    return {
        "environment": s.env,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
        "compute": dm.status(),
        "pipeline": pipeline.status(),
        "orchestrator": get_orchestrator().status(),
        "persistence": persistence.status(),
        "automation": dispatcher.status(),
        "llm": get_llm().info(),
        "vector_store": get_store().info(),
        "forensic": get_forensic().status(),
        "websocket_clients": hub.count,
        "event_subscribers": bus.subscriber_count,
        "counts": {
            "cameras_total": cameras_total,
            "cameras_online": cameras_online,
            "cameras_running": len(pipeline.running_ids()),
            "incidents_open": incidents_open,
            "tracks_active": tracks_active,
        },
        "privacy": {
            "redaction": s.privacy_redaction,
            "anonymous_by_default": s.privacy_anonymous_by_default,
            "retention_days": s.retention_days,
        },
    }


# ---------------------------------------------------------------------- GPU
class GpuToggle(BaseModel):
    enabled: bool
    gpu_index: Optional[int] = None


@router.get("/gpu")
def gpu_status(user: User = Depends(current_user)) -> Dict[str, Any]:
    """Full compute picture, including any registered remote GPU workers."""
    dm = get_device_manager()
    status = dm.status()
    status["client_acceleration"] = {
        "webgpu_recommended": True,
        "power_preference": "high-performance",
        "note": (
            "The dashboard requests the browser's high-performance WebGPU adapter, "
            "which selects a discrete GPU without any permission prompt. It accelerates "
            "the digital twin, heatmaps and optional in-browser ONNX inference. "
            "Server-side torch models are controlled by the toggle below."
        ),
    }
    return status


@router.post("/gpu")
def set_gpu(
    body: GpuToggle,
    request: Request,
    user: User = Depends(require("system:gpu")),
    db: Session = Depends(get_session),
) -> Dict[str, Any]:
    """Flip server-side inference between GPU and CPU, live.

    Every registered model is relocated immediately; no restart, and the very
    next frame runs on the new device.
    """
    dm = get_device_manager()
    before = dm.active_device
    status = dm.set_gpu(body.enabled, body.gpu_index)

    record(
        db, request, user, "system.gpu.toggle", "compute", None,
        detail={"from": before, "to": status["active_device"], "requested": body.enabled},
    )
    bus.publish(
        "system.gpu",
        {"active_device": status["active_device"], "gpu_enabled": status["gpu_enabled"],
         "actor": user.username},
        source="system",
    )

    if body.enabled and not status["cuda_available"]:
        status["warning"] = (
            "GPU was requested but CUDA is not available to this process. "
            "Inference stays on CPU. Install a CUDA-enabled torch build "
            "(see docs/INSTALL.md) or register a remote GPU worker."
        )
    return status


# ------------------------------------------------------- remote GPU workers
class WorkerRegistration(BaseModel):
    node_id: str
    name: Optional[str] = None
    device: str = "cuda:0"
    gpu_name: str = "unknown"
    total_memory_mb: int = 0
    capabilities: List[str] = Field(default_factory=list)


@router.post("/workers/register")
def register_worker(body: WorkerRegistration, user: User = Depends(require("system:gpu"))):
    """Register a machine offering its GPU to this deployment.

    This is how a dashboard hosted on a CPU-only VPS runs inference on an
    operator's own workstation GPU.
    """
    node = get_device_manager().register_worker(body.model_dump())
    bus.publish("system.worker.registered", {"node_id": node.node_id, "gpu": node.gpu_name},
                source="system")
    return {**node.__dict__, "online": node.online}


@router.post("/workers/{node_id}/heartbeat")
def worker_heartbeat(node_id: str, jobs_completed: Optional[int] = None,
                     user: User = Depends(require("system:gpu"))):
    if not get_device_manager().heartbeat_worker(node_id, jobs_completed):
        raise HTTPException(404, f"worker '{node_id}' is not registered")
    return {"ok": True, "node_id": node_id}


@router.delete("/workers/{node_id}")
def drop_worker(node_id: str, user: User = Depends(require("system:gpu"))):
    return {"removed": get_device_manager().drop_worker(node_id)}


@router.get("/workers")
def list_workers(user: User = Depends(current_user)):
    return {"workers": get_device_manager().status()["workers"]}


# ------------------------------------------------------------------- agents
@router.get("/agents")
def agents(user: User = Depends(current_user)) -> Dict[str, Any]:
    """Roster for the Agent Orchestrator screen, including pipeline stages."""
    orch = get_orchestrator()
    status = orch.status()
    pipeline = get_pipeline().status()

    detector_info = None
    for cam in pipeline.get("cameras", []):
        if cam.get("detector"):
            detector_info = cam["detector"]
            break

    # Agents 1 and 2 live in the capture pipeline rather than the agent loop;
    # they are reported here so the roster matches the specification's numbering.
    stages = [
        {
            "name": "vision",
            "spec_id": 1,
            "description": "Person, vehicle, bag and object detection",
            "enabled": True,
            "location": "pipeline",
            "backend": (detector_info or {}).get("backend"),
            "detail": detector_info,
        },
        {
            "name": "tracking",
            "spec_id": 2,
            "description": "Multi-object tracking (ByteTrack)",
            "enabled": True,
            "location": "pipeline",
            "backend": get_settings().tracker,
        },
    ]
    roster = stages + [{**a, "location": "orchestrator"} for a in status["agents"]]
    roster += [{**r, "location": "reasoner"} for r in status["reasoners"]]
    roster.append({**get_forensic().status(), "enabled": True, "location": "reasoner"})
    roster.sort(key=lambda a: a.get("spec_id", 99))

    return {
        "agents": roster,
        "frames_processed": status["frames_processed"],
        "findings_emitted": status["findings_emitted"],
        "incidents_opened": status["incidents_opened"],
        "open_incidents": status["open_incidents"],
        "device": status["device"],
        "policy_loaded_at": status["policy_loaded_at"],
    }


class AgentToggle(BaseModel):
    enabled: bool


@router.post("/agents/{name}/toggle")
def toggle_agent(
    name: str,
    body: AgentToggle,
    request: Request,
    user: User = Depends(require("policy:write")),
    db: Session = Depends(get_session),
):
    if not get_orchestrator().set_agent_enabled(name, body.enabled):
        raise HTTPException(404, f"no agent named '{name}'")
    record(db, request, user, "agent.toggle", "agent", name, detail={"enabled": body.enabled})
    return {"agent": name, "enabled": body.enabled}


# ------------------------------------------------------------------- policy
@router.get("/policy")
def get_policy(user: User = Depends(require("policy:read"))) -> Dict[str, Any]:
    """Every threshold and weight the platform uses, as stored on disk."""
    return load_policy()


@router.put("/policy")
def update_policy(
    body: Dict[str, Any],
    request: Request,
    user: User = Depends(require("policy:write")),
    db: Session = Depends(get_session),
) -> Dict[str, Any]:
    merged = save_policy(body)
    get_orchestrator().reload_policy()
    record(db, request, user, "policy.update", "policy", None,
           detail={"sections": sorted(body.keys())})
    bus.publish("system.policy", {"sections": sorted(body.keys()), "actor": user.username},
                source="system")
    return merged


# ----------------------------------------------------------- model registry
@router.get("/models")
def list_models(user: User = Depends(current_user), db: Session = Depends(get_session)):
    rows = db.scalars(select(ModelRecord).order_by(ModelRecord.task, ModelRecord.name)).all()
    live = _live_model_state()
    return {"registered": [to_dict(r) for r in rows], "runtime": live}


def _live_model_state() -> List[Dict[str, Any]]:
    """What is actually loaded right now, as opposed to what is registered."""
    from ...vision.reid import get_body_embedder, get_face_embedder

    dm = get_device_manager()
    out: List[Dict[str, Any]] = []

    pipeline = get_pipeline().status()
    for cam in pipeline.get("cameras", []):
        det = cam.get("detector")
        if det:
            out.append({"task": "detection", "camera_id": cam["camera_id"], "device": dm.active_device, **det})
            break

    body = get_body_embedder()
    face = get_face_embedder()
    out.append({"task": "reid", "device": dm.active_device, **body.info()})
    out.append({"task": "face", "device": dm.active_device, **face.info()})
    out.append({"task": "llm", **get_llm().info()})
    out.append({"task": "embedding", **get_store().info()})
    return out


class ModelUpsert(BaseModel):
    name: str
    task: str
    framework: str = "pytorch"
    version: str = "1.0.0"
    weights_path: Optional[str] = None
    source: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None


@router.post("/models")
def register_model(
    body: ModelUpsert,
    request: Request,
    user: User = Depends(require("system:read")),
    db: Session = Depends(get_session),
):
    existing = db.scalar(
        select(ModelRecord).where(ModelRecord.name == body.name, ModelRecord.version == body.version)
    )
    if existing is None:
        existing = ModelRecord(id=new_id("MDL"), **body.model_dump())
        db.add(existing)
    else:
        for k, v in body.model_dump().items():
            setattr(existing, k, v)
    db.commit()
    record(db, request, user, "model.register", "model", existing.id, detail={"name": body.name})
    return to_dict(existing)


# --------------------------------------------------------------------- LLM
@router.get("/llm/health")
def llm_health(user: User = Depends(current_user)):
    return get_llm().health()


@router.post("/llm/reload")
def llm_reload(user: User = Depends(require("policy:write"))):
    reset_llm()
    return get_llm().info()


# ----------------------------------------------------------------- metrics
@router.get("/metrics")
def metrics(
    scope: Optional[str] = None,
    limit: int = 200,
    user: User = Depends(current_user),
    db: Session = Depends(get_session),
):
    stmt = select(SystemMetric).order_by(SystemMetric.at.desc()).limit(limit)
    if scope:
        stmt = stmt.where(SystemMetric.scope == scope)
    return {"items": [to_dict(m) for m in db.scalars(stmt).all()]}


@router.get("/events")
def recent_events(topic: str = "", limit: int = 100, user: User = Depends(current_user)):
    """Replay the in-memory event ring - useful when a dashboard reconnects."""
    return {"items": bus.recent(topic, limit)}


# ------------------------------------------------------------- automation
@router.get("/automation")
def automation_status(user: User = Depends(current_user)):
    return dispatcher.status()


class WebhookTest(BaseModel):
    url: str
    payload: Optional[Dict[str, Any]] = None


@router.post("/automation/test")
def automation_test(body: WebhookTest, user: User = Depends(require("workflow:write"))):
    return dispatcher.test(body.url, body.payload)

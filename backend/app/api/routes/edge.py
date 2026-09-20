"""Edge GPU nodes: registration, camera assignments, and result ingest.

Node-facing endpoints require the ``edge:node`` permission (the dedicated
``edge_node`` role, or an admin). See docs/EDGE_GPU.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ...agents.orchestrator import get_orchestrator
from ...config import get_settings
from ...core.security import current_user, require
from ...edge.hub import digest, get_hub
from ...models import User

router = APIRouter(prefix="/api/edge", tags=["edge"])

MAX_FRAME_BYTES = 3 * 1024 * 1024
MODEL_SUFFIXES = {".pt", ".pth", ".tar", ".pkl", ".json", ".onnx", ".engine"}


class NodeRegistration(BaseModel):
    node_id: str = Field(min_length=3, max_length=64)
    name: str = Field(default="", max_length=160)
    device: str = "cpu"
    gpu_name: str = "none"
    total_memory_mb: int = 0
    max_cameras: int = Field(default=4, ge=1, le=64)
    capabilities: List[str] = Field(default_factory=list)
    software: Dict[str, Any] = Field(default_factory=dict)


class IngestBatch(BaseModel):
    statuses: List[Dict[str, Any]] = Field(default_factory=list)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list)
    incidents: List[Dict[str, Any]] = Field(default_factory=list)


def _known(node_id: str):
    node = get_hub().node(node_id)
    if node is None:
        # 404 tells the node to register again (e.g. after a server restart).
        raise HTTPException(404, f"edge node '{node_id}' is not registered")
    return node


# ----------------------------------------------------------- node-facing
@router.post("/nodes/register")
def register_node(body: NodeRegistration, user: User = Depends(require("edge:node"))):
    node = get_hub().register(body.model_dump())
    return {"registered": node.node_id, "name": node.name,
            "placement": get_settings().processing_placement}


@router.get("/nodes/{node_id}/assignments")
def assignments(node_id: str, user: User = Depends(require("edge:node"))):
    _known(node_id)
    get_hub().touch(node_id)
    return get_hub().assignments(node_id)


@router.post("/nodes/{node_id}/ingest")
def ingest(node_id: str, body: IngestBatch, user: User = Depends(require("edge:node"))):
    _known(node_id)
    return get_hub().ingest(node_id, body.model_dump())


@router.put("/nodes/{node_id}/frames/{camera_id}")
async def put_frame(node_id: str, camera_id: str, request: Request,
                    user: User = Depends(require("edge:node"))):
    _known(node_id)
    data = await request.body()
    if not data:
        raise HTTPException(400, "empty frame")
    if len(data) > MAX_FRAME_BYTES:
        raise HTTPException(413, "frame too large")
    if not data.startswith(b"\xff\xd8"):
        raise HTTPException(415, "frames must be JPEG")
    if not get_hub().ingest_frame(node_id, camera_id, data):
        raise HTTPException(409, f"camera '{camera_id}' is not assigned to node '{node_id}'")
    return {"ok": True}


@router.get("/nodes/{node_id}/policy")
def node_policy(node_id: str, user: User = Depends(require("policy:read"))):
    _known(node_id)
    policy = get_orchestrator().policy
    return {"version": digest(policy), "policy": policy}


@router.get("/nodes/{node_id}/watchlist")
def node_watchlist(node_id: str, user: User = Depends(require("watchlist:sync"))):
    """The live matching gallery (embeddings only - no photographs)."""
    _known(node_id)
    hub = get_hub()
    return {"version": hub.watchlist_version(), "subjects": hub.watchlist_payload()}


@router.delete("/nodes/{node_id}")
def drop_node(node_id: str, user: User = Depends(require("edge:node"))):
    return {"dropped": get_hub().drop(node_id)}


# ----------------------------------------------------------- model sync
def _models_dir() -> Path:
    return (get_settings().storage_dir / "models").resolve()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_hash_cache: Dict[str, Any] = {}


@router.get("/models")
def model_manifest(user: User = Depends(require("edge:node"))):
    """Promoted models a node should run with, so every GPU uses the same ones."""
    root = _models_dir()
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in MODEL_SUFFIXES:
            continue
        stat = p.stat()
        key = f"{p}:{stat.st_size}:{stat.st_mtime_ns}"
        if key not in _hash_cache:
            _hash_cache[key] = _sha256(p)
        files.append({"path": p.relative_to(root).as_posix(), "size": stat.st_size,
                      "sha256": _hash_cache[key]})
    return {"files": files, "version": digest([(f["path"], f["sha256"]) for f in files])}


@router.get("/models/file")
def model_file(path: str, user: User = Depends(require("edge:node"))):
    root = _models_dir()
    target = (root / path).resolve()
    if root not in target.parents or not target.is_file() \
            or target.suffix.lower() not in MODEL_SUFFIXES:
        raise HTTPException(404, "no such model file")
    return FileResponse(str(target), media_type="application/octet-stream")


# ------------------------------------------------------------ dashboard
@router.get("/nodes")
def list_nodes(user: User = Depends(current_user)):
    return get_hub().status()

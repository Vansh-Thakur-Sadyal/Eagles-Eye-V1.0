"""Server side of edge processing.

A dashboard hosted on a CPU-only VPS cannot run detection, tracking, Re-ID and
the behaviour agents at useful speed, and a browser cannot run CUDA. So the
work goes to *edge nodes*: operators' own PCs with their own GPUs, each running
the unchanged Eagles Eye pipeline (tools/gpu_worker.py). Video never has to
leave the site; the server only receives what the agents concluded.

Placement: when a camera is started, the hub decides where it runs (see
Settings.processing_placement). If it lands on a node, the pipeline manager
registers a RemoteCameraWorker in place of a local thread. The proxy has the
same interface as CameraWorker, so every existing route - status, MJPEG
stream, snapshot, tracks, evidence capture - works unchanged, fed by what the
node pushes.

Nodes only ever report on cameras assigned to them; anything else is refused.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..core.events import bus

log = logging.getLogger("sentinel.edge.hub")

PLACEMENTS = ("auto", "server", "edge")


def digest(obj: Any) -> str:
    """Stable short hash of a JSON-able object (used for version checks)."""
    raw = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha1(raw).hexdigest()[:16]


# ---------------------------------------------------------------- nodes
@dataclass
class EdgeNodeState:
    node_id: str
    name: str
    device: str
    gpu_name: str
    total_memory_mb: int
    max_cameras: int = 4
    capabilities: List[str] = field(default_factory=list)
    software: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    frames_received: int = 0
    events_received: int = 0
    incidents_received: int = 0
    last_error: Optional[str] = None

    @property
    def has_gpu(self) -> bool:
        return self.device.startswith("cuda")

    def online(self, timeout: float) -> bool:
        return (time.time() - self.last_seen) < timeout


# ------------------------------------------------------- remote camera proxy
class RemoteCameraWorker:
    """Stands in for a CameraWorker whose camera runs on an edge node.

    Mirrors the parts of CameraWorker's interface the API uses. State is
    whatever the node last pushed; if the node goes quiet the camera is
    reported as degraded, never as silently healthy.
    """

    remote = True

    def __init__(self, runtime, node_id: str, hub: "EdgeHub") -> None:
        self.runtime = runtime
        self.node_id = node_id
        self._hub = hub
        self._lock = threading.Lock()
        self._alive = True
        self.paused = False
        self.started_at = datetime.now(timezone.utc)
        self._info: Dict[str, Any] = {}
        self._jpeg: Optional[bytes] = None
        self._jpeg_at = 0.0
        self._tracks: List[Dict[str, Any]] = []
        self._crowd: Optional[Dict[str, Any]] = None
        self._viewers = 0
        self._last_view_at = 0.0
        self._reported_at = 0.0

    # --- thread-like -------------------------------------------------------
    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: Optional[float] = None) -> None:
        return None

    def stop(self) -> None:
        self._alive = False
        self._hub.unassign(self.runtime.camera_id)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    # --- viewers (drives how often the node sends preview frames) ---------
    def add_viewer(self) -> None:
        with self._lock:
            self._viewers += 1
            self._last_view_at = time.time()

    def remove_viewer(self) -> None:
        with self._lock:
            self._viewers = max(0, self._viewers - 1)
            self._last_view_at = time.time()

    @property
    def viewers(self) -> int:
        # A snapshot poll counts as a viewer for a few seconds, like locally.
        recent = (time.time() - self._last_view_at) < 10.0
        return self._viewers or (1 if recent else 0)

    # --- fed by the node ---------------------------------------------------
    def update_info(self, info: Dict[str, Any]) -> None:
        with self._lock:
            self._info = dict(info)
            self._reported_at = time.time()

    def update_result(self, result: Dict[str, Any]) -> None:
        with self._lock:
            self._tracks = list(result.get("tracks") or [])
            if result.get("crowd"):
                self._crowd = result["crowd"]
            self._reported_at = time.time()

    def update_jpeg(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._jpeg_at = time.time()

    # --- read by the API ---------------------------------------------------
    def latest_jpeg(self) -> Optional[bytes]:
        return self._jpeg

    def latest_frame(self):
        if self._jpeg is None:
            return None
        try:
            import cv2
            import numpy as np

            return cv2.imdecode(np.frombuffer(self._jpeg, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None

    def snapshot_tracks(self) -> List[Dict[str, Any]]:
        return list(self._tracks)

    def latest_crowd(self) -> Optional[Dict[str, Any]]:
        return self._crowd

    @property
    def status(self) -> str:
        return self.info()["status"]

    def info(self) -> Dict[str, Any]:
        node = self._hub.node(self.node_id)
        timeout = get_settings().edge_node_timeout_seconds
        node_online = bool(node and node.online(timeout))
        with self._lock:
            data = dict(self._info)
        if not data:
            data = {
                "camera_id": self.runtime.camera_id,
                "name": self.runtime.name,
                "status": "connecting",
                "status_detail": "waiting for the edge node to open the source",
                "source_type": self.runtime.source_type,
                "measured_fps": 0.0,
                "target_fps": self.runtime.fps,
                "frame_index": 0,
                "track_count": 0,
            }
        if not node_online:
            data["status"] = "degraded"
            data["status_detail"] = (
                f"edge node '{node.name if node else self.node_id}' has not reported for "
                f"over {int(timeout)}s"
            )
        elif self.paused:
            data["status"] = "paused"
        data["viewers"] = self.viewers
        data["processing_node"] = {
            "node_id": self.node_id,
            "name": node.name if node else self.node_id,
            "gpu_name": node.gpu_name if node else "unknown",
            "online": node_online,
        }
        data["preview_age_seconds"] = (
            round(time.time() - self._jpeg_at, 1) if self._jpeg_at else None
        )
        return data


# ---------------------------------------------------------------- the hub
class EdgeHub:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._nodes: Dict[str, EdgeNodeState] = {}
        # camera_id -> RemoteCameraWorker
        self._assigned: Dict[str, RemoteCameraWorker] = {}

    # --- registry ----------------------------------------------------------
    def register(self, payload: Dict[str, Any]) -> EdgeNodeState:
        node_id = str(payload["node_id"])
        with self._lock:
            existing = self._nodes.get(node_id)
            node = EdgeNodeState(
                node_id=node_id,
                name=str(payload.get("name") or node_id),
                device=str(payload.get("device", "cpu")),
                gpu_name=str(payload.get("gpu_name", "none")),
                total_memory_mb=int(payload.get("total_memory_mb", 0) or 0),
                max_cameras=max(1, int(payload.get("max_cameras", 4) or 4)),
                capabilities=list(payload.get("capabilities") or []),
                software=dict(payload.get("software") or {}),
            )
            if existing:
                node.registered_at = existing.registered_at
                node.frames_received = existing.frames_received
                node.events_received = existing.events_received
                node.incidents_received = existing.incidents_received
            self._nodes[node_id] = node
        # Also visible in the existing GPU/worker panel.
        try:
            from ..core.device import get_device_manager

            get_device_manager().register_worker({
                "node_id": node_id, "name": node.name, "device": node.device,
                "gpu_name": node.gpu_name, "total_memory_mb": node.total_memory_mb,
                "capabilities": node.capabilities or ["camera_pipeline"],
            })
        except Exception:
            log.exception("could not mirror node into the device manager")
        bus.publish("edge.node.registered",
                    {"node_id": node_id, "name": node.name, "gpu_name": node.gpu_name,
                     "device": node.device}, source="edge")
        log.info("edge node registered: %s (%s, %s)", node.name, node_id, node.gpu_name)
        return node

    def node(self, node_id: str) -> Optional[EdgeNodeState]:
        return self._nodes.get(node_id)

    def touch(self, node_id: str) -> Optional[EdgeNodeState]:
        node = self._nodes.get(node_id)
        if node is not None:
            node.last_seen = time.time()
            try:
                from ..core.device import get_device_manager

                get_device_manager().heartbeat_worker(node_id)
            except Exception:
                pass
        return node

    def drop(self, node_id: str) -> bool:
        with self._lock:
            removed = self._nodes.pop(node_id, None) is not None
        try:
            from ..core.device import get_device_manager

            get_device_manager().drop_worker(node_id)
        except Exception:
            pass
        return removed

    def online_nodes(self) -> List[EdgeNodeState]:
        timeout = get_settings().edge_node_timeout_seconds
        return [n for n in self._nodes.values() if n.online(timeout)]

    def load(self, node_id: str) -> int:
        return sum(1 for w in self._assigned.values() if w.node_id == node_id and w.is_alive())

    # --- placement ---------------------------------------------------------
    def choose_node(self, runtime) -> Optional[EdgeNodeState]:
        """Where should this camera run? None means on this server.

        Raises RuntimeError when the configuration demands an edge node and
        no suitable one is online - a camera silently falling back to a
        CPU-only VPS is exactly the surprise this feature exists to avoid.
        """
        s = get_settings()
        wanted = str((runtime.options or {}).get("processing_node") or s.processing_placement)
        wanted = wanted.strip() or "auto"
        if wanted == "server":
            return None

        online = self.online_nodes()
        if wanted not in PLACEMENTS:                      # pinned to one node
            node = self._nodes.get(wanted)
            if node is None:
                raise RuntimeError(f"processing node '{wanted}' is not registered")
            if node not in online:
                raise RuntimeError(f"processing node '{node.name}' is offline")
            if self.load(node.node_id) >= node.max_cameras:
                raise RuntimeError(f"processing node '{node.name}' is at its camera limit "
                                   f"({node.max_cameras})")
            return node

        # auto / edge: least-loaded GPU node with spare capacity. A webcam
        # index means "the webcam on whichever machine runs this", so those
        # only move to a node when pinned explicitly.
        if runtime.source_type == "webcam" and wanted == "auto":
            return None
        candidates = [n for n in online if n.has_gpu and self.load(n.node_id) < n.max_cameras]
        if not candidates:
            if wanted == "edge":
                raise RuntimeError("no GPU edge node with spare capacity is online - "
                                   "start tools/gpu_worker.py on a GPU machine")
            return None
        return min(candidates, key=lambda n: (self.load(n.node_id) / n.max_cameras,
                                              -n.total_memory_mb))

    def assign(self, runtime, node: EdgeNodeState) -> RemoteCameraWorker:
        worker = RemoteCameraWorker(runtime, node.node_id, self)
        with self._lock:
            self._assigned[runtime.camera_id] = worker
        bus.publish("edge.camera.assigned",
                    {"camera_id": runtime.camera_id, "name": runtime.name,
                     "node_id": node.node_id, "node_name": node.name}, source="edge")
        log.info("camera %s assigned to edge node %s", runtime.camera_id, node.name)
        return worker

    def unassign(self, camera_id: str) -> None:
        with self._lock:
            self._assigned.pop(camera_id, None)

    def worker_for(self, camera_id: str) -> Optional[RemoteCameraWorker]:
        return self._assigned.get(camera_id)

    # --- what a node polls -------------------------------------------------
    def assignments(self, node_id: str) -> Dict[str, Any]:
        from ..agents.orchestrator import get_orchestrator

        cameras = []
        for worker in list(self._assigned.values()):
            if worker.node_id != node_id or not worker.is_alive():
                continue
            spec = asdict(worker.runtime)
            cameras.append({
                "runtime": spec,
                "config_hash": digest(spec),
                "paused": worker.paused,
                "viewers": worker.viewers,
            })
        orchestrator = get_orchestrator()
        return {
            "node_id": node_id,
            "cameras": cameras,
            "policy_version": digest(orchestrator.policy),
            "watchlist_version": self.watchlist_version(),
            "server_time": datetime.now(timezone.utc).isoformat(),
        }

    # --- watchlist gallery sync -------------------------------------------
    @staticmethod
    def watchlist_payload() -> List[Dict[str, Any]]:
        from ..agents.orchestrator import get_orchestrator

        out = []
        for s in get_orchestrator().watchlist._subjects.values():
            out.append({
                "subject_id": s.subject_id, "label": s.label, "category": s.category,
                "priority": s.priority, "status": s.status,
                "face_embeddings": [e.astype(float).round(6).tolist() for e in s.face_embeddings],
                "body_embeddings": [e.astype(float).round(6).tolist() for e in s.body_embeddings],
                "match_threshold": s.match_threshold,
                "scope_camera_ids": list(s.scope_camera_ids),
                "scope_site_ids": list(s.scope_site_ids),
                "expires_at": s.expires_at.isoformat() if s.expires_at else None,
                "alert_on_match": s.alert_on_match,
            })
        return out

    def watchlist_version(self) -> str:
        from ..agents.orchestrator import get_orchestrator

        subjects = get_orchestrator().watchlist._subjects.values()
        return digest([(s.subject_id, s.status, len(s.face_embeddings), len(s.body_embeddings),
                        s.match_threshold, s.expires_at, list(s.scope_camera_ids),
                        list(s.scope_site_ids)) for s in subjects])

    # --- what a node pushes -----------------------------------------------
    def ingest(self, node_id: str, batch: Dict[str, Any]) -> Dict[str, Any]:
        from ..agents.orchestrator import get_orchestrator
        from ..pipeline.runner import get_pipeline

        node = self.touch(node_id)
        if node is None:
            raise KeyError(node_id)

        def owned(camera_id: Optional[str]) -> Optional[RemoteCameraWorker]:
            w = self._assigned.get(camera_id or "")
            return w if (w is not None and w.node_id == node_id and w.is_alive()) else None

        accepted = rejected = 0
        for info in batch.get("statuses") or []:
            w = owned(info.get("camera_id"))
            if w is None:
                rejected += 1
                continue
            w.update_info(info)
            accepted += 1

        pipeline = get_pipeline()
        for result in batch.get("results") or []:
            w = owned(result.get("camera_id"))
            if w is None:
                rejected += 1
                continue
            w.update_result(result)
            # The node already throttles results to ~one per camera every two
            # seconds, so write each one rather than re-throttling by frame.
            pipeline.emit_external(result)
            accepted += 1

        for evt in batch.get("events") or []:
            payload = evt.get("payload") or {}
            if owned(payload.get("camera_id")) is None:
                rejected += 1
                continue
            bus.publish(str(evt.get("topic", "edge.event")), payload,
                        source=f"edge:{node_id}", severity=str(evt.get("severity", "info")))
            node.events_received += 1
            accepted += 1

        orchestrator = get_orchestrator()
        for incident in batch.get("incidents") or []:
            if owned(incident.get("camera_id")) is None:
                rejected += 1
                continue
            incident.setdefault("meta", {})
            incident["processed_on"] = {"node_id": node_id, "node_name": node.name,
                                        "gpu_name": node.gpu_name}
            orchestrator.ingest_external_incident(incident, source=f"edge:{node_id}")
            node.incidents_received += 1
            accepted += 1

        return {"accepted": accepted, "rejected": rejected}

    def ingest_frame(self, node_id: str, camera_id: str, jpeg: bytes) -> bool:
        node = self.touch(node_id)
        w = self._assigned.get(camera_id)
        if node is None or w is None or w.node_id != node_id or not w.is_alive():
            return False
        w.update_jpeg(jpeg)
        node.frames_received += 1
        return True

    # --- reporting ---------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        timeout = get_settings().edge_node_timeout_seconds
        nodes = []
        for n in self._nodes.values():
            nodes.append({
                **{k: v for k, v in asdict(n).items() if k not in ("software",)},
                "software": n.software,
                "online": n.online(timeout),
                "seconds_since_seen": round(time.time() - n.last_seen, 1),
                "cameras": [cid for cid, w in self._assigned.items()
                            if w.node_id == n.node_id and w.is_alive()],
            })
        return {
            "placement": get_settings().processing_placement,
            "nodes": nodes,
            "online": sum(1 for n in nodes if n["online"]),
            "remote_cameras": len(self._assigned),
        }


_hub: Optional[EdgeHub] = None
_hub_lock = threading.Lock()


def get_hub() -> EdgeHub:
    global _hub
    if _hub is None:
        with _hub_lock:
            if _hub is None:
                _hub = EdgeHub()
    return _hub

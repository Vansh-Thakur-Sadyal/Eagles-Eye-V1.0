"""Edge side: run cameras on this machine's GPU for a remote Eagles Eye server.

The node runs the unchanged pipeline (detection, tracking, Re-ID, face, crowd,
video-event and behaviour agents) on its own GPU and pushes what the agents
conclude to the server:

    poll    GET  /api/edge/nodes/{id}/assignments   every 2 s
    push    POST /api/edge/nodes/{id}/ingest        statuses, results, findings, incidents
    frames  PUT  /api/edge/nodes/{id}/frames/{cam}  JPEG previews (fast while someone watches)
    sync    policy, watchlist gallery and promoted models, when their versions change

Nothing here is started by the server process; tools/gpu_worker.py builds an
EdgeNode after pointing the settings at a node-local storage directory.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import httpx
import numpy as np

log = logging.getLogger("sentinel.edge.node")

FORWARD_TOPICS = ("finding", "camera.status")
RESULT_INTERVAL_S = 2.0          # one result per camera every 2 s (tracks + crowd)
STATUS_INTERVAL_S = 2.0
POLL_INTERVAL_S = 2.0
FLUSH_INTERVAL_S = 0.5
IDLE_FRAME_INTERVAL_S = 5.0      # keep the dashboard snapshot warm
MAX_PREVIEW_FPS = 8.0


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (datetime,)):
        return o.isoformat()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def probe_gpu() -> Dict[str, Any]:
    info: Dict[str, Any] = {"device": "cpu", "gpu_name": "none", "total_memory_mb": 0,
                            "torch": None, "cuda": None}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(device="cuda:0", gpu_name=props.name,
                        total_memory_mb=int(props.total_memory / (1024 * 1024)))
    except Exception:
        pass
    return info


class Outbox:
    """Bounded, thread-safe buffer that survives short server outages."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: Deque[Dict[str, Any]] = deque(maxlen=5000)
        self.incidents: Deque[Dict[str, Any]] = deque(maxlen=2000)
        self.results: Deque[Dict[str, Any]] = deque(maxlen=500)
        self.statuses: Dict[str, Dict[str, Any]] = {}
        self.sent = 0

    def put(self, kind: str, item: Dict[str, Any]) -> None:
        with self._lock:
            getattr(self, kind).append(item)

    def set_status(self, camera_id: str, info: Dict[str, Any]) -> None:
        with self._lock:
            self.statuses[camera_id] = info

    def take(self, max_events: int = 300) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            batch = {
                "events": [self.events.popleft() for _ in range(min(max_events, len(self.events)))],
                "incidents": list(self.incidents),
                "results": list(self.results),
                "statuses": list(self.statuses.values()),
            }
            self.incidents.clear()
            self.results.clear()
            self.statuses.clear()
        return batch

    def restore(self, batch: Dict[str, List[Dict[str, Any]]]) -> None:
        """Put an unsent batch back at the front, oldest first."""
        with self._lock:
            for kind in ("events", "incidents", "results"):
                target: Deque = getattr(self, kind)
                for item in reversed(batch.get(kind, [])):
                    if len(target) < (target.maxlen or 10 ** 9):
                        target.appendleft(item)
            for info in batch.get("statuses", []):
                self.statuses.setdefault(info.get("camera_id", ""), info)

    def pending(self) -> int:
        with self._lock:
            return len(self.events) + len(self.incidents) + len(self.results)


class EdgeNode:
    def __init__(self, api: str, username: str, password: str, *, node_id: str, name: str,
                 max_cameras: int = 4, http_factory: Optional[Callable[[], httpx.Client]] = None
                 ) -> None:
        self.api = api.rstrip("/")
        self.username = username
        self.password = password
        self.node_id = node_id
        self.name = name
        self.max_cameras = max_cameras
        self._http_factory = http_factory or (lambda: httpx.Client(base_url=self.api, timeout=20.0))
        self._http = self._http_factory()
        self._frames_http = self._http_factory()
        self._token: Optional[str] = None
        self._token_lock = threading.Lock()
        self.outbox = Outbox()
        self.running = False
        self.gpu = probe_gpu()

        self._cameras: Dict[str, Dict[str, Any]] = {}     # camera_id -> {hash, fps, viewers...}
        self._viewers_added: Dict[str, int] = {}
        self._last_result_at: Dict[str, float] = {}
        self._last_frame_sent: Dict[str, float] = {}
        self._last_jpeg: Dict[str, bytes] = {}
        self._policy_version: Optional[str] = None
        self._watchlist_version: Optional[str] = None
        self._pipeline = None
        self._orchestrator = None
        self.stats = {"frames_pushed": 0, "batches_sent": 0, "send_failures": 0,
                      "incidents_forwarded": 0, "findings_forwarded": 0}

    # ------------------------------------------------------------- auth / http
    def login(self) -> None:
        r = self._http.post("/api/auth/login",
                            json={"username": self.username, "password": self.password})
        r.raise_for_status()
        with self._token_lock:
            self._token = r.json()["access_token"]

    def _headers(self) -> Dict[str, str]:
        with self._token_lock:
            return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def _request(self, client: httpx.Client, method: str, path: str, **kw) -> httpx.Response:
        extra = kw.pop("headers", {})
        r = client.request(method, path, headers={**self._headers(), **extra}, **kw)
        if r.status_code == 401:                    # token expired
            self.login()
            r = client.request(method, path, headers={**self._headers(), **extra}, **kw)
        return r

    def register(self) -> None:
        import platform

        body = {
            "node_id": self.node_id, "name": self.name, "device": self.gpu["device"],
            "gpu_name": self.gpu["gpu_name"], "total_memory_mb": self.gpu["total_memory_mb"],
            "max_cameras": self.max_cameras, "capabilities": ["camera_pipeline"],
            "software": {"torch": self.gpu["torch"], "cuda": self.gpu["cuda"],
                         "platform": platform.platform(), "python": platform.python_version()},
        }
        r = self._request(self._http, "POST", "/api/edge/nodes/register", json=body)
        r.raise_for_status()
        log.info("registered with %s as %s (%s)", self.api, self.name, self.node_id)

    # ------------------------------------------------------------- models
    def sync_models(self, models_dir: Path) -> Dict[str, int]:
        """Download promoted models the server has and this node lacks."""
        import hashlib

        r = self._request(self._http, "GET", "/api/edge/models")
        r.raise_for_status()
        fetched = kept = 0
        for f in r.json().get("files", []):
            target = models_dir / f["path"]
            if target.is_file() and target.stat().st_size == f["size"]:
                h = hashlib.sha256(target.read_bytes()).hexdigest()
                if h == f["sha256"]:
                    kept += 1
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            h = hashlib.sha256()
            with self._http.stream("GET", "/api/edge/models/file", params={"path": f["path"]},
                                   headers=self._headers(), timeout=600.0) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as out:
                    for chunk in resp.iter_bytes(1 << 20):
                        h.update(chunk)
                        out.write(chunk)
            if h.hexdigest() != f["sha256"]:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"checksum mismatch downloading {f['path']}")
            tmp.replace(target)
            fetched += 1
            log.info("model synced: %s (%.1f MB)", f["path"], f["size"] / 1e6)
        return {"fetched": fetched, "up_to_date": kept}

    # ------------------------------------------------------------- wiring
    def attach(self) -> None:
        """Hook the local orchestrator/pipeline/bus so results flow to the server."""
        from ..agents.orchestrator import get_orchestrator
        from ..core.events import bus
        from ..pipeline.runner import get_pipeline

        self._orchestrator = get_orchestrator()
        self._pipeline = get_pipeline()
        self._orchestrator.add_sink(self._on_incident)
        self._pipeline.add_result_sink(self._on_result)
        bus.on(self._on_event)

    def _mine(self, camera_id: Optional[str]) -> bool:
        return bool(camera_id) and camera_id in self._cameras

    def _on_incident(self, payload: Dict[str, Any]) -> None:
        if self._mine(payload.get("camera_id")):
            self.outbox.put("incidents", json.loads(json.dumps(payload, default=_json_default)))
            self.stats["incidents_forwarded"] += 1

    def _on_result(self, result: Dict[str, Any]) -> None:
        cid = result.get("camera_id")
        if not self._mine(cid):
            return
        now = time.time()
        if now - self._last_result_at.get(cid, 0.0) < RESULT_INTERVAL_S:
            return
        self._last_result_at[cid] = now
        self.outbox.put("results", json.loads(json.dumps(result, default=_json_default)))

    def _on_event(self, evt) -> None:
        if evt.topic not in FORWARD_TOPICS or str(evt.source).startswith("edge:"):
            return
        if not self._mine(evt.payload.get("camera_id")):
            return
        self.outbox.put("events", {
            "topic": evt.topic, "severity": evt.severity, "at": evt.at,
            "payload": json.loads(json.dumps(evt.payload, default=_json_default)),
        })
        if evt.topic == "finding":
            self.stats["findings_forwarded"] += 1

    # ------------------------------------------------------------- control
    def poll_once(self) -> Dict[str, Any]:
        r = self._request(self._http, "GET", f"/api/edge/nodes/{self.node_id}/assignments")
        if r.status_code == 404:                    # server restarted / forgot us
            self.register()
            r = self._request(self._http, "GET", f"/api/edge/nodes/{self.node_id}/assignments")
        r.raise_for_status()
        data = r.json()
        self._reconcile(data.get("cameras", []))
        if data.get("policy_version") != self._policy_version:
            self._sync_policy()
        if data.get("watchlist_version") != self._watchlist_version:
            self._sync_watchlist()
        return data

    def _reconcile(self, cameras: List[Dict[str, Any]]) -> None:
        from ..pipeline.runner import CameraRuntime

        desired = {c["runtime"]["camera_id"]: c for c in cameras}
        for cid in list(self._cameras):
            if cid not in desired:
                log.info("camera %s unassigned - stopping", cid)
                self._pipeline.stop(cid)
                self._forget(cid)

        for cid, spec in desired.items():
            current = self._cameras.get(cid)
            if current is not None and current["hash"] != spec["config_hash"]:
                log.info("camera %s configuration changed - restarting", cid)
                self._pipeline.stop(cid)
                self._forget(cid)
                current = None
            if current is None:
                runtime = CameraRuntime(**spec["runtime"])
                self._cameras[cid] = {"hash": spec["config_hash"], "fps": runtime.fps}
                try:
                    self._pipeline.start(runtime)
                    log.info("camera %s (%s) started on %s", cid, runtime.name,
                             self.gpu["gpu_name"])
                except Exception as exc:
                    log.error("camera %s failed to start: %s", cid, exc)
                    self.outbox.set_status(cid, {"camera_id": cid, "name": runtime.name,
                                                 "status": "error", "status_detail": str(exc)})
                    continue

            worker = self._pipeline.get(cid)
            if worker is None:
                continue
            self._cameras[cid]["viewers"] = int(spec.get("viewers", 0))
            if spec.get("paused") and worker.status != "paused":
                worker.pause()
            elif not spec.get("paused") and worker.status == "paused":
                worker.resume()
            # Mirror dashboard viewers so the worker renders previews at full rate.
            want = 1 if spec.get("viewers", 0) > 0 else 0
            have = self._viewers_added.get(cid, 0)
            if want > have:
                worker.add_viewer()
            elif want < have:
                worker.remove_viewer()
            self._viewers_added[cid] = want

    def _forget(self, cid: str) -> None:
        self._cameras.pop(cid, None)
        self._viewers_added.pop(cid, None)
        self._last_result_at.pop(cid, None)
        self._last_frame_sent.pop(cid, None)
        self._last_jpeg.pop(cid, None)

    def _sync_policy(self) -> None:
        r = self._request(self._http, "GET", f"/api/edge/nodes/{self.node_id}/policy")
        r.raise_for_status()
        data = r.json()
        self._orchestrator.set_policy(data["policy"])
        self._policy_version = data["version"]
        log.info("policy synced (%s)", self._policy_version)

    def _sync_watchlist(self) -> None:
        from ..agents.watchlist import EnrolledSubject

        r = self._request(self._http, "GET", f"/api/edge/nodes/{self.node_id}/watchlist")
        r.raise_for_status()
        data = r.json()
        subjects = []
        for s in data.get("subjects", []):
            expires = s.get("expires_at")
            subjects.append(EnrolledSubject(
                subject_id=s["subject_id"], label=s["label"], category=s["category"],
                priority=s["priority"], status=s["status"],
                face_embeddings=[np.asarray(e, dtype=np.float32) for e in s["face_embeddings"]],
                body_embeddings=[np.asarray(e, dtype=np.float32) for e in s["body_embeddings"]],
                match_threshold=s.get("match_threshold"),
                scope_camera_ids=s.get("scope_camera_ids") or [],
                scope_site_ids=s.get("scope_site_ids") or [],
                expires_at=datetime.fromisoformat(expires) if expires else None,
                alert_on_match=bool(s.get("alert_on_match", True)),
            ))
        self._orchestrator.watchlist.load_subjects(subjects)
        self._watchlist_version = data["version"]
        log.info("watchlist synced: %d subject(s)", len(subjects))

    # ------------------------------------------------------------- pushing
    def collect_statuses(self) -> None:
        for cid in list(self._cameras):
            worker = self._pipeline.get(cid)
            if worker is not None:
                info = worker.info()
                info["processing_gpu"] = self.gpu["gpu_name"]
                self.outbox.set_status(cid, json.loads(json.dumps(info, default=_json_default)))

    def flush(self) -> bool:
        batch = self.outbox.take()
        if not any(batch.values()):
            return True
        try:
            r = self._request(self._http, "POST", f"/api/edge/nodes/{self.node_id}/ingest",
                              content=json.dumps(batch, default=_json_default),
                              headers={"Content-Type": "application/json"})
            if r.status_code == 404:
                self.register()
                raise RuntimeError("node was not registered")
            r.raise_for_status()
            self.stats["batches_sent"] += 1
            return True
        except Exception as exc:
            self.stats["send_failures"] += 1
            self.outbox.restore(batch)
            log.warning("ingest failed (%s); %d item(s) buffered", exc, self.outbox.pending())
            return False

    def push_frames_once(self) -> None:
        now = time.time()
        for cid, meta in list(self._cameras.items()):
            worker = self._pipeline.get(cid)
            if worker is None:
                continue
            watching = meta.get("viewers", 0) > 0
            interval = (1.0 / min(MAX_PREVIEW_FPS, max(1.0, float(meta.get("fps", 12))))
                        if watching else IDLE_FRAME_INTERVAL_S)
            if now - self._last_frame_sent.get(cid, 0.0) < interval:
                continue
            jpeg = worker.latest_jpeg()
            if not jpeg or jpeg is self._last_jpeg.get(cid):
                continue
            try:
                r = self._request(self._frames_http, "PUT",
                                  f"/api/edge/nodes/{self.node_id}/frames/{cid}", content=jpeg,
                                  headers={"Content-Type": "image/jpeg"})
                if r.status_code < 300:
                    self.stats["frames_pushed"] += 1
                    self._last_jpeg[cid] = jpeg
            except Exception as exc:
                log.debug("frame push failed for %s: %s", cid, exc)
            self._last_frame_sent[cid] = now

    # ------------------------------------------------------------- run loop
    def run(self, stop: threading.Event) -> None:
        self.running = True
        threads = [
            threading.Thread(target=self._loop, args=(stop, self.flush, FLUSH_INTERVAL_S),
                             name="edge-flush", daemon=True),
            threading.Thread(target=self._loop, args=(stop, self.push_frames_once, 0.05),
                             name="edge-frames", daemon=True),
            threading.Thread(target=self._loop, args=(stop, self.collect_statuses,
                                                      STATUS_INTERVAL_S),
                             name="edge-status", daemon=True),
        ]
        for t in threads:
            t.start()
        while not stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                log.warning("assignment poll failed: %s", exc)
            stop.wait(POLL_INTERVAL_S)
        self.shutdown()

    @staticmethod
    def _loop(stop: threading.Event, fn: Callable[[], Any], interval: float) -> None:
        while not stop.is_set():
            try:
                fn()
            except Exception:
                log.exception("edge background task failed")
            stop.wait(interval)

    def shutdown(self) -> None:
        self.running = False
        if self._pipeline is not None:
            self._pipeline.stop_all()
        try:
            self.flush()
            self._request(self._http, "DELETE", f"/api/edge/nodes/{self.node_id}")
        except Exception:
            pass
        log.info("edge node stopped: %s", self.stats)

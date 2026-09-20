"""Per-camera pipeline worker and the manager that owns them all.

One thread per camera: grab -> detect -> track -> agents -> persist -> publish.
Each worker keeps the latest annotated frame so the dashboard can pull an MJPEG
stream without the encoder ever blocking inference.

The GPU toggle is honoured live: the worker reads the active device from the
device manager every frame, so flipping it in the UI changes where the very
next inference runs, with no restart.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from ..agents.base import CameraContext, FrameContext
from ..agents.orchestrator import get_orchestrator
from ..config import get_settings
from ..core.device import get_device_manager
from ..core.events import bus
from ..vision.detector import build_detector
from ..vision.geometry import normalised_polygon_to_pixels, point_in_polygon
from ..vision.reid import get_body_embedder
from ..vision.tracker import ByteTracker
from .capture import SourceSpec, build_source

log = logging.getLogger("sentinel.pipeline")


@dataclass
class CameraRuntime:
    """Mutable description of a camera, refreshed from the DB on change."""

    camera_id: str
    name: str
    source_type: str
    source_uri: str
    width: int = 1280
    height: int = 720
    fps: int = 12
    rotation: int = 0
    username: Optional[str] = None
    password: Optional[str] = None
    zone_id: Optional[str] = None
    site_id: Optional[str] = None
    location: str = ""
    zones: List[Dict[str, Any]] = field(default_factory=list)
    line_crossings: List[Dict[str, Any]] = field(default_factory=list)
    expected_flow_deg: Optional[float] = None
    homography: Optional[List[List[float]]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    floor: int = 0
    field_of_view_deg: float = 82.0
    range_m: float = 25.0
    orientation_deg: float = 0.0
    enabled_agents: List[str] = field(default_factory=list)
    detection_classes: List[str] = field(default_factory=list)
    privacy_redaction: Optional[bool] = None
    options: Dict[str, Any] = field(default_factory=dict)


class CameraWorker(threading.Thread):
    """Runs one camera end to end."""

    def __init__(self, runtime: CameraRuntime, on_result=None) -> None:
        super().__init__(name=f"cam-{runtime.camera_id}", daemon=True)
        self.runtime = runtime
        self._on_result = on_result
        # NB: threading.Thread defines an internal _stop(); never shadow it.
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        self.status: str = "starting"
        self.status_detail: Optional[str] = None
        self.measured_fps: float = 0.0
        self.last_frame_at: Optional[datetime] = None
        self.frame_index: int = 0
        self.started_at = datetime.now(timezone.utc)
        self.inference_ms: float = 0.0

        self._source = None
        self._detector = None
        self._tracker: Optional[ByteTracker] = None
        self._embedder = None
        self._latest_jpeg: Optional[bytes] = None
        self._latest_raw: Optional[np.ndarray] = None
        self._jpeg_lock = threading.Lock()
        self._frame_times: Deque[float] = deque(maxlen=60)
        self._last_loop_at: Optional[float] = None
        self.processing_ms: float = 0.0
        self._last_tracks: List[Dict[str, Any]] = []
        self._consecutive_failures = 0
        # Rendering and JPEG-encoding every frame costs real throughput on CPU.
        # Only do it while something is actually consuming the stream.
        self._viewers = 0
        self._last_view_at = 0.0
        self._render_every = 1

    # ---------------------------------------------------------------- setup
    def _build(self) -> bool:
        rt = self.runtime
        spec = SourceSpec(
            source_type=rt.source_type,
            uri=rt.source_uri,
            width=rt.width,
            height=rt.height,
            fps=rt.fps,
            rotation=rt.rotation,
            username=rt.username,
            password=rt.password,
            options=dict(rt.options),
        )
        try:
            self._source = build_source(spec)
        except ValueError as exc:
            self._set_status("error", str(exc))
            return False

        if not self._source.open():
            self._set_status("error", self._source.last_error or "source would not open")
            return False

        s = get_settings()
        # A camera may pin its own detector. The synthetic scene needs this:
        # it renders schematic figures that a real detector correctly does NOT
        # recognise as people, so without an override the demo camera silently
        # produces nothing once YOLO is installed.
        override = rt.options.get("detector")
        if override is None and rt.source_type == "synthetic":
            override = "motion"
        self._detector = build_detector(override, classes=rt.detection_classes or None)
        self._tracker = ByteTracker(
            high_thresh=max(0.25, s.yolo_conf),
            low_thresh=0.1,
            match_thresh=0.8,
            max_age=int(max(15, rt.fps * 2.5)),
            min_hits=3,
            appearance_weight=0.25,
        )
        self._embedder = get_body_embedder()
        self._set_status("online", None)
        return True

    def _set_status(self, status: str, detail: Optional[str]) -> None:
        if self.status != status or self.status_detail != detail:
            self.status = status
            self.status_detail = detail
            bus.publish(
                "camera.status",
                {
                    "camera_id": self.runtime.camera_id,
                    "name": self.runtime.name,
                    "status": status,
                    "detail": detail,
                },
                source="pipeline",
                severity="warning" if status in ("error", "degraded") else "info",
            )

    # ----------------------------------------------------------------- loop
    def run(self) -> None:
        if not self._build():
            return

        orchestrator = get_orchestrator()
        target_interval = 1.0 / max(1, self.runtime.fps)

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.2)
                continue

            loop_start = time.perf_counter()
            ok, frame = self._source.read()

            if not ok or frame is None:
                self._consecutive_failures += 1
                self._set_status(
                    "degraded" if self._consecutive_failures < 5 else "error",
                    self._source.last_error,
                )
                if self._consecutive_failures in (5, 15, 40) or self._consecutive_failures % 60 == 0:
                    log.warning(
                        "camera %s: %d consecutive read failures (%s) - reconnecting",
                        self.runtime.camera_id, self._consecutive_failures,
                        self._source.last_error,
                    )
                    self._source.reconnect()
                time.sleep(min(2.0, 0.15 * self._consecutive_failures))
                continue

            self._consecutive_failures = 0
            if self.status != "online":
                self._set_status("online", None)

            self.frame_index += 1
            now = datetime.now(timezone.utc)
            self.last_frame_at = now
            h, w = frame.shape[:2]
            self.runtime.height, self.runtime.width = h, w

            try:
                result = self._process(frame, now, orchestrator)
                if self._on_result is not None and result is not None:
                    self._on_result(result)
            except Exception:
                log.exception("camera %s: frame processing failed", self.runtime.camera_id)

            elapsed = time.perf_counter() - loop_start
            self.processing_ms = round(elapsed * 1000.0, 2)

            sleep_for = target_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

            # measured_fps must be the real delivered frame rate (wall clock
            # between frames), not processing capacity: every speed, dwell and
            # heading calculation downstream divides by it.
            now_wall = time.perf_counter()
            if self._last_loop_at is not None:
                self._frame_times.append(now_wall - self._last_loop_at)
            self._last_loop_at = now_wall
            if self._frame_times:
                self.measured_fps = round(1.0 / max(1e-6, float(np.mean(self._frame_times))), 2)

        if self._source is not None:
            self._source.release()
        self._set_status("offline", "stopped")

    # -------------------------------------------------------------- one tick
    def _process(self, frame: np.ndarray, now: datetime, orchestrator) -> Optional[Dict[str, Any]]:
        rt = self.runtime
        dm = get_device_manager()
        started = time.perf_counter()

        detections = self._detector.detect(frame)

        # Appearance embeddings feed both the tracker's appearance gate and Re-ID.
        # One batched forward pass, not one per person: embedding people one at
        # a time cost ~6 ms each and dominated the frame budget in a crowd.
        crops: List[Optional[np.ndarray]] = []
        for det in detections:
            if det.class_name != "person":
                crops.append(None)
                continue
            x1, y1, x2, y2 = [int(max(0, v)) for v in det.bbox]
            crop = frame[y1 : min(y2, frame.shape[0]), x1 : min(x2, frame.shape[1])]
            crops.append(crop if crop.size else None)
        embeddings: List[Optional[np.ndarray]] = self._embedder.embed_batch(crops)

        tracks = self._tracker.update(
            detections, timestamp=now, fps=max(1.0, self.measured_fps or rt.fps),
            embeddings=embeddings,
        )
        self.inference_ms = (time.perf_counter() - started) * 1000.0

        # Attach the zone each track currently stands in.
        for track in tracks:
            track.zone_id = self._zone_for(track.foot_point, frame.shape[1], frame.shape[0])

        ctx = FrameContext(
            camera=CameraContext(
                camera_id=rt.camera_id,
                name=rt.name,
                width=frame.shape[1],
                height=frame.shape[0],
                fps=max(1.0, self.measured_fps or rt.fps),
                zone_id=rt.zone_id,
                site_id=rt.site_id,
                zones=rt.zones,
                expected_flow_deg=rt.expected_flow_deg,
                homography=rt.homography,
                line_crossings=rt.line_crossings,
                latitude=rt.latitude,
                longitude=rt.longitude,
                floor=rt.floor,
                location=rt.location,
            ),
            frame_index=self.frame_index,
            timestamp=now,
            frame=frame,
            detections=detections,
            tracks=tracks,
            policy=orchestrator.policy,
            elapsed_seconds=(now - self.started_at).total_seconds(),
            device=dm.active_device,
            extras={"enabled_agents": rt.enabled_agents or None},
        )
        # SpatialAgent reads range_m off the camera context
        ctx.camera.__dict__["range_m"] = rt.range_m
        ctx.camera.__dict__["field_of_view_deg"] = rt.field_of_view_deg
        ctx.camera.__dict__["orientation_deg"] = rt.orientation_deg

        result = orchestrator.process_frame(ctx)
        result["inference_ms"] = round(self.inference_ms, 2)
        result["detector_backend"] = self._detector.backend
        result["measured_fps"] = self.measured_fps

        self._last_tracks = [
            {
                "track_id": t.track_id,
                "global_id": t.global_id,
                "class_name": t.class_name,
                "bbox": [round(v, 1) for v in t.bbox],
                "score": round(t.score, 3),
                "speed": round(t.speed, 1),
                "heading_deg": t.heading_deg(),
                "zone_id": t.zone_id,
                "age_frames": t.age_frames,
                "face_visibility": t.face_visibility,
                "duration_seconds": round(t.duration_seconds, 1),
            }
            for t in tracks
        ]
        result["tracks"] = self._last_tracks

        if self._should_render():
            self._render(frame, ctx, orchestrator)
        return result

    def _zone_for(self, point, width: int, height: int) -> Optional[str]:
        for zone in self.runtime.zones:
            poly = zone.get("polygon") or []
            if len(poly) < 3:
                continue
            if point_in_polygon(point, normalised_polygon_to_pixels(poly, width, height)):
                return zone.get("id")
        return self.runtime.zone_id

    def _should_render(self) -> bool:
        """Render only for live viewers, or occasionally to keep a snapshot warm."""
        if self._viewers > 0 or (time.time() - self._last_view_at) < 10.0:
            return True
        # Nobody watching: keep a recent still for the snapshot endpoint and
        # for evidence capture, but stop paying the cost every frame.
        return self.frame_index % max(1, int(self.runtime.fps) * 2) == 0

    def add_viewer(self) -> None:
        self._viewers += 1
        self._last_view_at = time.time()

    def remove_viewer(self) -> None:
        self._viewers = max(0, self._viewers - 1)
        self._last_view_at = time.time()

    # ------------------------------------------------------------ rendering
    def _render(self, frame: np.ndarray, ctx: FrameContext, orchestrator) -> None:
        """Build the annotated, privacy-redacted JPEG the dashboard consumes."""
        try:
            import cv2
        except Exception:
            return

        redact = (
            self.runtime.privacy_redaction
            if self.runtime.privacy_redaction is not None
            else None
        )
        canvas = orchestrator.privacy.redact_frame(frame, ctx, force=redact)

        for zone in self.runtime.zones:
            poly = zone.get("polygon") or []
            if len(poly) < 3:
                continue
            pts = np.array(
                normalised_polygon_to_pixels(poly, canvas.shape[1], canvas.shape[0]),
                dtype=np.int32,
            )
            colour = (60, 60, 220) if zone.get("zone_type") in ("restricted", "secure") else (150, 120, 60)
            cv2.polylines(canvas, [pts], True, colour, 2)
            if len(pts):
                cv2.putText(canvas, str(zone.get("name", "")), tuple(pts[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

        palette = {"person": (40, 160, 90), "suitcase": (40, 90, 200), "backpack": (40, 90, 200),
                   "handbag": (40, 90, 200)}
        for track in ctx.tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            colour = palette.get(track.class_name, (120, 120, 120))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            label = f"{track.class_name[:3].upper()} #{track.track_id}"
            if track.global_id:
                label += " *"
            cv2.rectangle(canvas, (x1, max(0, y1 - 18)), (x1 + 8 * len(label), y1), colour, -1)
            cv2.putText(canvas, label, (x1 + 2, max(10, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        banner = (
            f"{self.runtime.name} | {self._detector.backend} | "
            f"{get_device_manager().active_device} | {self.measured_fps:.1f} fps | "
            f"{len(ctx.tracks)} tracks"
        )
        cv2.rectangle(canvas, (0, canvas.shape[0] - 24), (canvas.shape[1], canvas.shape[0]),
                      (28, 32, 44), -1)
        cv2.putText(canvas, banner, (8, canvas.shape[0] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 240, 250), 1)

        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if ok:
            with self._jpeg_lock:
                self._latest_jpeg = buf.tobytes()
                self._latest_raw = canvas

    # -------------------------------------------------------------- control
    def latest_jpeg(self) -> Optional[bytes]:
        with self._jpeg_lock:
            return self._latest_jpeg

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._jpeg_lock:
            return None if self._latest_raw is None else self._latest_raw.copy()

    def snapshot_tracks(self) -> List[Dict[str, Any]]:
        return list(self._last_tracks)

    def pause(self) -> None:
        self._pause_event.set()
        self._set_status("paused", "paused by operator")

    def resume(self) -> None:
        self._pause_event.clear()
        self._set_status("online", None)

    def stop(self) -> None:
        self._stop_event.set()

    def info(self) -> Dict[str, Any]:
        return {
            "camera_id": self.runtime.camera_id,
            "name": self.runtime.name,
            "status": self.status,
            "status_detail": self.status_detail,
            "source_type": self.runtime.source_type,
            "measured_fps": self.measured_fps,
            "target_fps": self.runtime.fps,
            "frame_index": self.frame_index,
            "inference_ms": round(self.inference_ms, 2),
            "processing_ms": round(self.processing_ms, 2),
            "last_frame_at": self.last_frame_at.isoformat() if self.last_frame_at else None,
            "track_count": len(self._last_tracks),
            "viewers": self._viewers,
            "detector": self._detector.info() if self._detector else None,
            "source": self._source.status() if self._source else None,
            "device": get_device_manager().active_device,
            "resolution": [self.runtime.width, self.runtime.height],
        }


class PipelineManager:
    """Starts, stops and reports on every camera worker."""

    def __init__(self) -> None:
        self._workers: Dict[str, CameraWorker] = {}
        self._lock = threading.RLock()
        self._result_sinks: List[Any] = []

    def add_result_sink(self, sink) -> None:
        self._result_sinks.append(sink)

    def _emit(self, result: Dict[str, Any], force: bool = False) -> None:
        import inspect

        for sink in self._result_sinks:
            try:
                if force and "force" in inspect.signature(sink).parameters:
                    sink(result, force=True)
                else:
                    sink(result)
            except Exception:
                log.exception("pipeline result sink failed")

    def emit_external(self, result: Dict[str, Any]) -> None:
        """A result produced on an edge node: persist it like a local one."""
        self._emit(result, force=True)

    def start(self, runtime: CameraRuntime) -> Dict[str, Any]:
        s = get_settings()
        with self._lock:
            if runtime.camera_id in self._workers:
                existing = self._workers[runtime.camera_id]
                if existing.is_alive():
                    return existing.info()
                self._workers.pop(runtime.camera_id, None)

            # Placement: a GPU edge node (an operator's own machine) or here.
            # choose_node raises when policy demands a node and none is up.
            from ..edge.hub import get_hub

            hub = get_hub()
            node = hub.choose_node(runtime)
            if node is not None:
                remote = hub.assign(runtime, node)
                self._workers[runtime.camera_id] = remote
                return remote.info()

            local = [w for w in self._workers.values() if not getattr(w, "remote", False)]
            if len(local) >= s.max_concurrent_cameras:
                raise RuntimeError(
                    f"camera limit reached ({s.max_concurrent_cameras}); "
                    "raise SENTINEL_MAX_CONCURRENT_CAMERAS or stop another camera"
                )

            worker = CameraWorker(runtime, on_result=self._emit)
            self._workers[runtime.camera_id] = worker
            worker.start()
        time.sleep(0.4)          # let the source attempt to open before reporting
        return worker.info()

    def stop(self, camera_id: str) -> bool:
        with self._lock:
            worker = self._workers.pop(camera_id, None)
        if worker is None:
            return False
        worker.stop()
        worker.join(timeout=5.0)
        get_orchestrator().reset_camera(camera_id)
        return True

    def stop_all(self) -> None:
        for cid in list(self._workers.keys()):
            self.stop(cid)

    def get(self, camera_id: str) -> Optional[CameraWorker]:
        return self._workers.get(camera_id)

    def is_running(self, camera_id: str) -> bool:
        worker = self._workers.get(camera_id)
        return bool(worker and worker.is_alive())

    def running_ids(self) -> List[str]:
        return [cid for cid, w in self._workers.items() if w.is_alive()]

    def status(self) -> Dict[str, Any]:
        workers = [w.info() for w in self._workers.values()]
        return {
            "running": len([w for w in workers if w["status"] == "online"]),
            "total": len(workers),
            "cameras": workers,
            "capacity": get_settings().max_concurrent_cameras,
            "on_edge_nodes": sum(1 for w in self._workers.values() if getattr(w, "remote", False)),
        }


_manager: Optional[PipelineManager] = None
_manager_lock = threading.Lock()


def get_pipeline() -> PipelineManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = PipelineManager()
    return _manager

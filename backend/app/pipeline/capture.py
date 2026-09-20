"""Frame sources.

Every camera in Eagles Eye is one of these, selected by ``source_type`` on the
camera record so a source can be swapped from the UI with no code change:

  webcam      a locally attached USB / integrated camera, by device index
  esp32cam    an ESP32-CAM board: either its MJPEG stream (":81/stream") or
              its single-shot endpoint ("/capture").  Both are handled, and
              the board's own quirks (chunked MJPEG, occasional stalls,
              Wi-Fi brownouts) are absorbed by the reconnect logic.
  rtsp        an IP camera / NVR
  http_mjpeg  a generic MJPEG-over-HTTP stream
  file        a video file, for replay and for testing against datasets
  synthetic   a procedurally generated scene, so the whole pipeline can be
              demonstrated and tested with no hardware at all

All sources present the same ``read()`` contract and reconnect on their own.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import numpy as np

log = logging.getLogger("sentinel.capture")


@dataclass
class SourceSpec:
    source_type: str
    uri: str
    width: int = 1280
    height: int = 720
    fps: int = 12
    rotation: int = 0
    username: Optional[str] = None
    password: Optional[str] = None
    options: Dict[str, Any] = None

    def __post_init__(self) -> None:
        self.options = self.options or {}


class FrameSource:
    """Base contract: open, read, release, and honest status reporting."""

    def __init__(self, spec: SourceSpec) -> None:
        self.spec = spec
        self.opened = False
        self.last_error: Optional[str] = None
        self.frames_read = 0
        self.reconnects = 0
        self._last_frame_at = 0.0

    def open(self) -> bool:
        raise NotImplementedError

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def release(self) -> None:
        self.opened = False

    def reconnect(self) -> bool:
        self.release()
        self.reconnects += 1
        time.sleep(min(5.0, 0.5 * self.reconnects))
        return self.open()

    def status(self) -> Dict[str, Any]:
        return {
            "source_type": self.spec.source_type,
            "opened": self.opened,
            "frames_read": self.frames_read,
            "reconnects": self.reconnects,
            "last_error": self.last_error,
            "seconds_since_frame": (
                round(time.time() - self._last_frame_at, 2) if self._last_frame_at else None
            ),
        }

    def _post(self, frame: np.ndarray) -> np.ndarray:
        self.frames_read += 1
        self._last_frame_at = time.time()
        if self.spec.rotation:
            frame = _rotate(frame, self.spec.rotation)
        return frame


class OpenCVSource(FrameSource):
    """Backs webcam, rtsp, http_mjpeg and file via cv2.VideoCapture."""

    def __init__(self, spec: SourceSpec) -> None:
        super().__init__(spec)
        self._cap = None
        self._loop_file = spec.options.get("loop", True)

    def _target(self) -> Any:
        st = self.spec.source_type
        if st == "webcam":
            raw = str(self.spec.uri).strip()
            return int(raw) if raw.isdigit() else 0
        if st in ("rtsp", "http_mjpeg") and self.spec.username:
            return _inject_credentials(self.spec.uri, self.spec.username, self.spec.password or "")
        return self.spec.uri

    def open(self) -> bool:
        try:
            import cv2
        except Exception as exc:
            self.last_error = f"OpenCV unavailable: {exc}"
            return False

        target = self._target()
        try:
            if self.spec.source_type == "webcam":
                # DirectShow avoids the multi-second MSMF open delay on Windows.
                backends = [getattr(cv2, "CAP_DSHOW", 700), getattr(cv2, "CAP_MSMF", 1400), 0]
                cap = None
                for backend in backends:
                    cap = cv2.VideoCapture(target, backend) if backend else cv2.VideoCapture(target)
                    if cap.isOpened():
                        break
                    cap.release()
                if cap is None or not cap.isOpened():
                    self.last_error = f"could not open webcam index {target}"
                    return False
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
                cap.set(cv2.CAP_PROP_FPS, self.spec.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                cap = cv2.VideoCapture(target)
                if self.spec.source_type in ("rtsp", "http_mjpeg"):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self.last_error = f"could not open source: {self.spec.source_type}"
                return False

            self._cap = cap
            self.opened = True
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def read(self):
        if not self.opened or self._cap is None:
            return False, None
        try:
            ok, frame = self._cap.read()
        except Exception as exc:
            self.last_error = str(exc)
            return False, None

        if not ok or frame is None:
            if self.spec.source_type == "file" and self._loop_file:
                try:
                    import cv2

                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self._cap.read()
                except Exception:
                    ok = False
            if not ok or frame is None:
                self.last_error = "no frame from source"
                return False, None

        return True, self._post(frame)

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        super().release()


class Esp32CamSource(FrameSource):
    """ESP32-CAM over HTTP.

    The board exposes two useful endpoints from the stock CameraWebServer
    sketch: ``http://<ip>:81/stream`` (multipart MJPEG) and ``http://<ip>/capture``
    (one JPEG).  The stream gives higher frame rates but the board drops it
    under Wi-Fi pressure, so this source falls back to polling /capture and
    keeps working rather than going dark.
    """

    def __init__(self, spec: SourceSpec) -> None:
        super().__init__(spec)
        self._session = None
        self._stream = None
        self._buffer = b""
        self._mode = "stream"
        self._base = self._normalise(spec.uri)
        self._snapshot_url = spec.options.get("snapshot_url") or f"{self._base}/capture"
        self._stream_url = spec.options.get("stream_url") or self._stream_default()

    @staticmethod
    def _normalise(uri: str) -> str:
        uri = (uri or "").strip().rstrip("/")
        if not uri:
            return ""
        if not uri.startswith(("http://", "https://")):
            uri = f"http://{uri}"
        parsed = urlparse(uri)
        return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))

    def _stream_default(self) -> str:
        parsed = urlparse(self._base)
        host = parsed.hostname or ""
        scheme = parsed.scheme or "http"
        # port 81 is the stock sketch's stream port
        return f"{scheme}://{host}:81/stream"

    def open(self) -> bool:
        try:
            import httpx
        except Exception as exc:
            self.last_error = f"httpx unavailable: {exc}"
            return False

        auth = None
        if self.spec.username:
            auth = (self.spec.username, self.spec.password or "")

        try:
            self._session = httpx.Client(timeout=httpx.Timeout(10.0, read=20.0), auth=auth)
        except Exception as exc:
            self.last_error = str(exc)
            return False

        # Try the MJPEG stream first, fall back to snapshot polling.
        try:
            req = self._session.build_request("GET", self._stream_url)
            response = self._session.send(req, stream=True)
            response.raise_for_status()
            self._stream = response
            self._iter = response.iter_bytes(chunk_size=4096)
            self._mode = "stream"
            self.opened = True
            self.last_error = None
            log.info("ESP32-CAM stream open: %s", self._stream_url)
            return True
        except Exception as exc:
            log.info("ESP32-CAM stream unavailable (%s), using snapshot mode", exc)
            self._stream = None

        try:
            r = self._session.get(self._snapshot_url)
            r.raise_for_status()
            self._mode = "snapshot"
            self.opened = True
            self.last_error = None
            log.info("ESP32-CAM snapshot mode: %s", self._snapshot_url)
            return True
        except Exception as exc:
            self.last_error = f"ESP32-CAM unreachable: {exc}"
            self.opened = False
            return False

    def read(self):
        if not self.opened:
            return False, None
        return self._read_stream() if self._mode == "stream" else self._read_snapshot()

    def _read_stream(self):
        try:
            import cv2

            # Accumulate until a complete JPEG (FFD8..FFD9) is in the buffer.
            deadline = time.time() + 5.0
            while time.time() < deadline:
                start = self._buffer.find(b"\xff\xd8")
                end = self._buffer.find(b"\xff\xd9", start + 2) if start != -1 else -1
                if start != -1 and end != -1:
                    jpg = self._buffer[start : end + 2]
                    self._buffer = self._buffer[end + 2 :]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    return True, self._post(frame)

                chunk = next(self._iter, None)
                if chunk is None:
                    self.last_error = "ESP32-CAM stream ended"
                    return False, None
                self._buffer += chunk
                if len(self._buffer) > 4_000_000:          # guard a runaway buffer
                    self._buffer = self._buffer[-1_000_000:]
            self.last_error = "ESP32-CAM stream timed out"
            return False, None
        except Exception as exc:
            self.last_error = str(exc)
            return False, None

    def _read_snapshot(self):
        try:
            import cv2

            r = self._session.get(self._snapshot_url)
            r.raise_for_status()
            frame = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                self.last_error = "ESP32-CAM returned an undecodable frame"
                return False, None
            return True, self._post(frame)
        except Exception as exc:
            self.last_error = str(exc)
            return False, None

    def release(self) -> None:
        for obj in (self._stream, self._session):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        self._stream = None
        self._session = None
        self._buffer = b""
        super().release()

    def status(self) -> Dict[str, Any]:
        base = super().status()
        base.update({"mode": self._mode, "stream_url": self._stream_url,
                     "snapshot_url": self._snapshot_url})
        return base


class SyntheticSource(FrameSource):
    """A procedurally generated scene with scripted behaviours.

    Exists so the entire pipeline - detection, tracking, every agent, incident
    creation, the dashboard and the alert path - can be demonstrated and
    regression-tested with no camera and no model weights.  The scene contains
    a crowd walking one way, a counter-flow subject, a loiterer, a follower
    pair and a bag that gets abandoned, so each agent has something real to
    find.
    """

    def __init__(self, spec: SourceSpec) -> None:
        super().__init__(spec)
        self._t = 0
        self._rng = np.random.default_rng(int(spec.options.get("seed", 7)))
        self._scenario = spec.options.get("scenario", "station")
        self._people: list = []
        self._bag: Optional[Dict[str, Any]] = None

    def open(self) -> bool:
        self._build_scene()
        self.opened = True
        self.last_error = None
        return True

    def _build_scene(self) -> None:
        w, h = self.spec.width, self.spec.height
        self._people = []
        crowd_n = int(self.spec.options.get("crowd", 9))
        for i in range(crowd_n):
            self._people.append(
                {
                    "x": self._rng.uniform(-200, w),
                    "y": h * 0.45 + self._rng.uniform(-70, 120),
                    "vx": self._rng.uniform(1.6, 3.0),
                    "vy": self._rng.uniform(-0.25, 0.25),
                    "role": "crowd",
                    "h": self._rng.uniform(95, 125),
                }
            )
        # counter-flow subject
        self._people.append({"x": w * 0.95, "y": h * 0.52, "vx": -2.4, "vy": 0.0,
                             "role": "counterflow", "h": 115})
        # loiterer
        self._people.append({"x": w * 0.22, "y": h * 0.68, "vx": 0.0, "vy": 0.0,
                             "role": "loiter", "h": 118})
        # leader + follower pair
        self._people.append({"x": w * 0.10, "y": h * 0.38, "vx": 2.0, "vy": 0.0,
                             "role": "leader", "h": 120})
        self._people.append({"x": w * 0.02, "y": h * 0.40, "vx": 2.0, "vy": 0.0,
                             "role": "follower", "h": 116})
        # bag owner who will walk away from their bag
        self._people.append({"x": w * 0.55, "y": h * 0.60, "vx": 0.0, "vy": 0.0,
                             "role": "bagowner", "h": 120})
        self._bag = {"x": w * 0.57, "y": h * 0.66, "w": 46, "h": 38, "dropped": False}

    def read(self):
        try:
            import cv2
        except Exception as exc:
            self.last_error = str(exc)
            return False, None

        w, h = self.spec.width, self.spec.height
        frame = np.full((h, w, 3), 205, dtype=np.uint8)

        # floor / perspective guides
        cv2.rectangle(frame, (0, 0), (w, int(h * 0.30)), (178, 172, 165), -1)
        for i in range(1, 6):
            y = int(h * 0.30 + (h * 0.70) * (i / 6) ** 1.4)
            cv2.line(frame, (0, y), (w, y), (192, 188, 182), 1)
        # a restricted area marking
        cv2.rectangle(frame, (int(w * 0.72), int(h * 0.55)), (int(w * 0.97), int(h * 0.92)),
                      (150, 150, 200), 2)

        t = self._t
        leader = next((p for p in self._people if p["role"] == "leader"), None)

        for p in self._people:
            role = p["role"]
            if role == "crowd":
                p["x"] += p["vx"]
                p["y"] += p["vy"] + math_sin(t, p["x"]) * 0.25
                if p["x"] > w + 120:
                    p["x"] = -120
                    p["y"] = h * 0.45 + self._rng.uniform(-70, 120)
            elif role == "counterflow":
                p["x"] += p["vx"]
                if p["x"] < -120:
                    p["x"] = w + 120
            elif role == "loiter":
                p["x"] += math_sin(t, 0) * 0.9
                p["y"] += math_cos(t, 0) * 0.6
            elif role == "leader":
                # walks, turns at intervals - the follower must mirror it
                phase = (t // 90) % 4
                p["vx"] = [2.0, 0.6, -1.8, 0.4][int(phase)]
                p["vy"] = [0.0, 1.4, 0.0, -1.3][int(phase)]
                p["x"] = float(np.clip(p["x"] + p["vx"], 40, w - 40))
                p["y"] = float(np.clip(p["y"] + p["vy"], h * 0.33, h * 0.85))
            elif role == "follower" and leader is not None:
                # trails the leader by a lag, which is the signal being tested
                dx, dy = leader["x"] - p["x"], leader["y"] - p["y"]
                dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
                if dist > 110:
                    p["x"] += dx / dist * 2.1
                    p["y"] += dy / dist * 2.1
            elif role == "bagowner":
                if t > 160:                              # walks away, leaving the bag
                    p["x"] += 2.2
                    if self._bag:
                        self._bag["dropped"] = True

            _draw_person(cv2, frame, p["x"], p["y"], p["h"])

        if self._bag:
            b = self._bag
            colour = (60, 60, 120) if b["dropped"] else (90, 90, 150)
            cv2.rectangle(
                frame,
                (int(b["x"]), int(b["y"])),
                (int(b["x"] + b["w"]), int(b["y"] + b["h"])),
                colour, -1,
            )
            cv2.rectangle(
                frame,
                (int(b["x"]), int(b["y"])),
                (int(b["x"] + b["w"]), int(b["y"] + b["h"])),
                (40, 40, 80), 2,
            )

        cv2.putText(frame, f"SYNTHETIC SCENE  t={t}", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 2)

        self._t += 1
        return True, self._post(frame)


def math_sin(t: int, phase: float) -> float:
    import math

    return math.sin((t + phase) * 0.04)


def math_cos(t: int, phase: float) -> float:
    import math

    return math.cos((t + phase) * 0.04)


def _draw_person(cv2, frame, x: float, y: float, height: float) -> None:
    w = height * 0.36
    x1, y1 = int(x - w / 2), int(y - height)
    x2, y2 = int(x + w / 2), int(y)
    cv2.rectangle(frame, (x1, int(y1 + height * 0.26)), (x2, y2), (86, 92, 110), -1)
    head_r = int(w * 0.34)
    cv2.circle(frame, (int(x), int(y1 + head_r)), head_r, (140, 132, 122), -1)


def _rotate(frame: np.ndarray, degrees: int) -> np.ndarray:
    import cv2

    mapping = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    code = mapping.get(degrees % 360)
    return cv2.rotate(frame, code) if code is not None else frame


def _inject_credentials(uri: str, username: str, password: str) -> str:
    parsed = urlparse(uri)
    if parsed.username:
        return uri
    netloc = f"{username}:{password}@{parsed.netloc}"
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))


SOURCE_TYPES = {
    "webcam": OpenCVSource,
    "rtsp": OpenCVSource,
    "http_mjpeg": OpenCVSource,
    "file": OpenCVSource,
    "esp32cam": Esp32CamSource,
    "synthetic": SyntheticSource,
}


def build_source(spec: SourceSpec) -> FrameSource:
    cls = SOURCE_TYPES.get(spec.source_type)
    if cls is None:
        raise ValueError(
            f"unknown source_type '{spec.source_type}'. "
            f"Supported: {', '.join(sorted(SOURCE_TYPES))}"
        )
    return cls(spec)


# --------------------------------------------------------------- discovery
def list_local_webcams(max_index: int = 6) -> list:
    """Probe local webcam indices so the UI can offer a real device list."""
    found = []
    try:
        import cv2
    except Exception:
        return found

    for idx in range(max_index):
        cap = None
        try:
            cap = cv2.VideoCapture(idx, getattr(cv2, "CAP_DSHOW", 700))
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    found.append(
                        {
                            "index": idx,
                            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            "fps": round(float(cap.get(cv2.CAP_PROP_FPS)) or 0.0, 1),
                        }
                    )
        except Exception:
            pass
        finally:
            if cap is not None:
                cap.release()
    return found


def probe_esp32cam(base_url: str, timeout: float = 6.0) -> Dict[str, Any]:
    """Check an ESP32-CAM and report which endpoints actually answer."""
    import httpx

    base = Esp32CamSource._normalise(base_url)
    parsed = urlparse(base)
    host = parsed.hostname or ""
    scheme = parsed.scheme or "http"
    result: Dict[str, Any] = {"base_url": base, "reachable": False, "endpoints": {}}
    if not host:
        result["error"] = "could not parse a host from the supplied address"
        return result

    checks = {
        "capture": f"{base}/capture",
        "stream": f"{scheme}://{host}:81/stream",
        "status": f"{base}/status",
    }
    with httpx.Client(timeout=timeout) as client:
        for name, url in checks.items():
            entry: Dict[str, Any] = {"url": url}
            try:
                if name == "stream":
                    req = client.build_request("GET", url)
                    r = client.send(req, stream=True)
                    entry["status_code"] = r.status_code
                    entry["content_type"] = r.headers.get("content-type")
                    entry["ok"] = r.status_code == 200
                    r.close()
                else:
                    r = client.get(url)
                    entry["status_code"] = r.status_code
                    entry["content_type"] = r.headers.get("content-type")
                    entry["ok"] = r.status_code == 200
                    entry["bytes"] = len(r.content)
            except Exception as exc:
                entry["ok"] = False
                entry["error"] = str(exc)
            result["endpoints"][name] = entry

    result["reachable"] = any(e.get("ok") for e in result["endpoints"].values())
    result["recommended_mode"] = (
        "stream" if result["endpoints"].get("stream", {}).get("ok")
        else "snapshot" if result["endpoints"].get("capture", {}).get("ok")
        else None
    )
    return result

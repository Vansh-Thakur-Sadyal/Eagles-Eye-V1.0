"""n8n automation layer (spec S24).

n8n is the orchestration surface *around* Eagles Eye, never inside the real-time
vision loop.  Eagles Eye emits events; workflow bindings stored in the database
decide which webhook each event reaches.

Dispatch happens on a background thread with bounded retries so a slow or down
n8n instance can never stall a camera worker.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import select

from ..config import get_settings
from ..core.events import bus
from ..db import session_scope
from ..models import Workflow, WorkflowRun, new_id, utcnow

log = logging.getLogger("sentinel.automation")

TRIGGERS = [
    "incident_created",
    "incident_updated",
    "incident_severity",
    "unattended_object",
    "watchlist_match",
    "camera_offline",
    "crowd_risk",
    "daily_report",
    "system_health",
    "manual",
]

SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


class AutomationDispatcher:
    def __init__(self) -> None:
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1000)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.dispatched = 0
        self.failed = 0
        self.dropped = 0

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="automation", daemon=True)
        self._thread.start()
        log.info("automation dispatcher started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    # ---------------------------------------------------------------- enqueue
    def emit(self, trigger: str, payload: Dict[str, Any]) -> None:
        try:
            self._queue.put_nowait({"trigger": trigger, "payload": payload})
        except queue.Full:
            self.dropped += 1
            log.warning("automation queue full, dropped a '%s' event", trigger)

    def on_incident(self, payload: Dict[str, Any]) -> None:
        self.emit("incident_created" if payload.get("is_new") else "incident_updated", payload)
        for behavior in payload.get("behaviors", []):
            if behavior == "unattended_object":
                self.emit("unattended_object", payload)
            elif behavior == "watchlist_match":
                self.emit("watchlist_match", payload)
            elif behavior.startswith("crowd_"):
                self.emit("crowd_risk", payload)

    def on_event(self, evt) -> None:
        if evt.topic == "camera.status" and evt.payload.get("status") in ("error", "offline"):
            self.emit("camera_offline", evt.payload)

    # ------------------------------------------------------------------ work
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._dispatch(item["trigger"], item["payload"])
            except Exception:
                log.exception("automation dispatch failed")
            finally:
                self._queue.task_done()

    def _dispatch(self, trigger: str, payload: Dict[str, Any]) -> None:
        settings = get_settings()
        bindings = self._bindings(trigger)

        # Environment-level webhooks act as always-on defaults.
        env_map = {
            "incident_created": settings.n8n_webhook_incident,
            "unattended_object": settings.n8n_webhook_unattended,
            "daily_report": settings.n8n_webhook_daily_report,
            "camera_offline": settings.n8n_webhook_system_health,
        }
        env_url = env_map.get(trigger)
        if settings.n8n_enabled and env_url:
            bindings.append(
                {"id": None, "name": f"env:{trigger}", "url": env_url, "method": "POST",
                 "headers": {}, "conditions": {}}
            )

        for binding in bindings:
            if not self._matches(binding.get("conditions") or {}, payload):
                continue
            self._send(binding, trigger, payload)

    @staticmethod
    def _bindings(trigger: str) -> List[Dict[str, Any]]:
        try:
            with session_scope() as db:
                rows = db.scalars(
                    select(Workflow).where(
                        Workflow.trigger == trigger, Workflow.enabled.is_(True)
                    )
                ).all()
                return [
                    {
                        "id": w.id,
                        "name": w.name,
                        "url": w.webhook_url,
                        "method": w.method or "POST",
                        "headers": w.headers or {},
                        "conditions": w.conditions or {},
                    }
                    for w in rows
                    if w.webhook_url
                ]
        except Exception:
            log.exception("could not read workflow bindings")
            return []

    @staticmethod
    def _matches(conditions: Dict[str, Any], payload: Dict[str, Any]) -> bool:
        """Simple, auditable condition language - no eval, no code execution."""
        min_sev = conditions.get("min_severity")
        if min_sev:
            have = SEVERITY_RANK.get(str(payload.get("severity", "info")), 0)
            want = SEVERITY_RANK.get(str(min_sev), 0)
            if have < want:
                return False

        min_score = conditions.get("min_risk_score")
        if min_score is not None and float(payload.get("risk_score", 0)) < float(min_score):
            return False

        cameras = conditions.get("camera_ids")
        if cameras and payload.get("camera_id") not in cameras:
            return False

        zones = conditions.get("zone_ids")
        if zones and payload.get("zone_id") not in zones:
            return False

        behaviors = conditions.get("behaviors")
        if behaviors and not set(behaviors) & set(payload.get("behaviors", [])):
            return False

        return True

    def _send(self, binding: Dict[str, Any], trigger: str, payload: Dict[str, Any]) -> None:
        body = {
            "source": "sentinel-ai",
            "trigger": trigger,
            "emitted_at": datetime.now(timezone.utc).isoformat(),
            "data": payload,
        }
        headers = {"Content-Type": "application/json", **(binding.get("headers") or {})}
        started = time.perf_counter()
        status_code: Optional[int] = None
        ok = False
        error: Optional[str] = None
        snippet: Optional[str] = None

        for attempt in range(3):
            try:
                with httpx.Client(timeout=15.0) as client:
                    r = client.request(
                        binding.get("method", "POST"), binding["url"], json=body, headers=headers
                    )
                status_code = r.status_code
                snippet = r.text[:500]
                ok = 200 <= r.status_code < 300
                if ok:
                    break
                error = f"HTTP {r.status_code}"
            except Exception as exc:
                error = str(exc)
            time.sleep(0.6 * (attempt + 1))

        duration = (time.perf_counter() - started) * 1000.0
        self.dispatched += 1
        if not ok:
            self.failed += 1

        bus.publish(
            "automation.dispatch",
            {
                "workflow": binding.get("name"),
                "trigger": trigger,
                "ok": ok,
                "status_code": status_code,
                "duration_ms": round(duration, 1),
                "error": error,
            },
            source="automation",
            severity="warning" if not ok else "info",
        )

        if binding.get("id"):
            try:
                with session_scope() as db:
                    db.add(
                        WorkflowRun(
                            id=new_id("WFR"),
                            workflow_id=binding["id"],
                            at=utcnow(),
                            trigger=trigger,
                            status_code=status_code,
                            ok=ok,
                            duration_ms=duration,
                            request_payload={"trigger": trigger, "keys": sorted(payload.keys())},
                            response_snippet=snippet,
                            error=error,
                        )
                    )
                    wf = db.get(Workflow, binding["id"])
                    if wf:
                        wf.last_run_at = utcnow()
                        wf.run_count = (wf.run_count or 0) + 1
                        if not ok:
                            wf.failure_count = (wf.failure_count or 0) + 1
            except Exception:
                log.exception("could not record workflow run")

    def status(self) -> Dict[str, Any]:
        s = get_settings()
        return {
            "enabled": s.n8n_enabled,
            "base_url": s.n8n_base_url or None,
            "queue_depth": self._queue.qsize(),
            "dispatched": self.dispatched,
            "failed": self.failed,
            "dropped": self.dropped,
            "running": bool(self._thread and self._thread.is_alive()),
            "supported_triggers": TRIGGERS,
        }

    def test(self, url: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fire a one-off test request so an operator can verify a binding."""
        body = {
            "source": "sentinel-ai",
            "trigger": "manual",
            "emitted_at": datetime.now(timezone.utc).isoformat(),
            "data": payload or {"test": True, "message": "Eagles Eye webhook test"},
        }
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=15.0) as client:
                r = client.post(url, json=body)
            return {
                "ok": 200 <= r.status_code < 300,
                "status_code": r.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "response": r.text[:800],
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            }


dispatcher = AutomationDispatcher()


def wire_automation() -> None:
    from ..agents.orchestrator import get_orchestrator

    dispatcher.start()
    get_orchestrator().add_sink(dispatcher.on_incident)
    bus.on(dispatcher.on_event)
    log.info("automation wired")

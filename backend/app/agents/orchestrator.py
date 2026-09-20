"""Agentic AI Orchestrator (spec S4, S26).

Owns the agent roster, fans each frame out to them, correlates the resulting
findings into incidents, scores them, narrates them and hands them to the
persistence and automation layers.

Correlation rule: findings that share a camera and overlap in time belong to
one incident.  An open incident absorbs new related findings (extending its
timeline and re-scoring) instead of spawning duplicates - which is what stops
a single abandoned bag from producing forty separate alerts.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from ..config import load_policy
from ..core.device import get_device_manager
from ..core.events import bus
from ..models import new_id
from .base import Agent, FrameContext, Finding
from .behavior import BehaviorAgent
from .commander import IncidentCommander
from .crowd import CrowdAgent
from .identity import AppearanceAgent, OcclusionAgent, ReIDAgent
from .objects import ObjectAgent
from .privacy import PrivacyAgent
from .relationship import RelationshipAgent
from .spatial import SpatialAgent
from .threat import ThreatAssessor
from .video_events import VideoEventAgent
from .watchlist import WatchlistAgent

log = logging.getLogger("sentinel.orchestrator")

# Behaviours that never, on their own, justify opening an incident.
NON_INCIDENT_BEHAVIORS = {
    "face_unavailable",
    "appearance_change",
    "cross_camera_association",
    "object_reclaimed",
}

# How long an open incident keeps absorbing related findings.
INCIDENT_WINDOW = timedelta(minutes=4)


@dataclass
class OpenIncident:
    incident_id: str
    camera_id: str
    started_at: datetime
    last_update: datetime
    findings: List[Finding] = field(default_factory=list)
    behaviors: set = field(default_factory=set)
    risk_score: float = 0.0
    severity: str = "low"
    persisted: bool = False

    def absorbs(self, finding: Finding, now: datetime) -> bool:
        return (now - self.last_update) <= INCIDENT_WINDOW


class Orchestrator:
    """Singleton coordinating all agents across all cameras."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self.reid = ReIDAgent()
        self.appearance = AppearanceAgent()
        self.appearance.bind(self.reid)
        self.occlusion = OcclusionAgent()
        self.behavior = BehaviorAgent()
        self.relationship = RelationshipAgent()
        self.crowd = CrowdAgent()
        self.objects = ObjectAgent()
        self.watchlist = WatchlistAgent()
        self.spatial = SpatialAgent()
        self.privacy = PrivacyAgent()
        self.video_events = VideoEventAgent()

        # Order matters: identity before behaviour, spatial and privacy last.
        self.agents: List[Agent] = [
            self.reid,
            self.appearance,
            self.occlusion,
            self.behavior,
            self.relationship,
            self.crowd,
            self.objects,
            self.watchlist,
            self.video_events,
            self.spatial,
            self.privacy,
        ]

        self.threat = ThreatAssessor()
        self.commander = IncidentCommander()

        self._open: Dict[str, List[OpenIncident]] = defaultdict(list)
        self._policy: Dict[str, Any] = load_policy()
        self._policy_loaded_at = datetime.now(timezone.utc)
        self._sinks: List[Callable[[Dict[str, Any]], None]] = []

        self.frames_processed = 0
        self.findings_emitted = 0
        self.incidents_opened = 0

    # ---------------------------------------------------------------- setup
    def add_sink(self, sink: Callable[[Dict[str, Any]], None]) -> None:
        """Register a consumer for completed incident payloads (DB, n8n, ...)."""
        self._sinks.append(sink)

    def ingest_external_incident(self, payload: Dict[str, Any], source: str) -> None:
        """An incident raised by the agents on an edge node.

        Published and handed to the sinks (database, search index, n8n)
        exactly as if it had been raised here, so the dashboard cannot tell
        the difference - except for ``processed_on``, which says where.
        """
        is_new = bool(payload.get("is_new"))
        bus.publish(
            "incident.created" if is_new else "incident.updated",
            payload, source=source, severity=str(payload.get("severity", "low")),
        )
        if is_new:
            self.incidents_opened += 1
        for sink in self._sinks:
            try:
                sink(payload)
            except Exception:
                log.exception("incident sink failed")

    def set_policy(self, policy: Dict[str, Any]) -> None:
        """Replace the live policy (edge nodes mirror the server's policy)."""
        with self._lock:
            self._policy = policy
            self._policy_loaded_at = datetime.now(timezone.utc)

    def reload_policy(self) -> Dict[str, Any]:
        with self._lock:
            self._policy = load_policy()
            self._policy_loaded_at = datetime.now(timezone.utc)
        log.info("policy reloaded")
        return self._policy

    @property
    def policy(self) -> Dict[str, Any]:
        return self._policy

    def agent_by_name(self, name: str) -> Optional[Agent]:
        return next((a for a in self.agents if a.name == name), None)

    def set_agent_enabled(self, name: str, enabled: bool) -> bool:
        agent = self.agent_by_name(name)
        if agent is None:
            return False
        agent.enabled = enabled
        log.info("agent %s %s", name, "enabled" if enabled else "disabled")
        return True

    def reset_camera(self, camera_id: str) -> None:
        for agent in self.agents:
            agent.reset(camera_id)
        with self._lock:
            self._open.pop(camera_id, None)

    # ----------------------------------------------------------------- main
    def process_frame(self, ctx: FrameContext) -> Dict[str, Any]:
        """Run every agent on one frame and correlate the result."""
        ctx.policy = self._policy
        ctx.device = get_device_manager().active_device

        findings: List[Finding] = []
        timings: Dict[str, float] = {}
        enabled_for_camera = ctx.extras.get("enabled_agents")

        for agent in self.agents:
            if enabled_for_camera and agent.name not in enabled_for_camera:
                continue
            produced = agent.run(ctx)
            timings[agent.name] = round(agent.last_run_ms, 2)
            for f in produced:
                if f.started_at is None:
                    f.started_at = ctx.timestamp
                if f.ended_at is None:
                    f.ended_at = ctx.timestamp
            findings.extend(produced)

        self.frames_processed += 1
        self.findings_emitted += len(findings)

        for f in findings:
            bus.publish(
                "finding",
                {
                    "camera_id": ctx.camera.camera_id,
                    "camera_name": ctx.camera.name,
                    "behavior": f.behavior,
                    "confidence": f.confidence,
                    "severity": f.severity,
                    "track_id": f.track_id,
                    "secondary_track_id": f.secondary_track_id,
                    "object_id": f.object_id,
                    "zone_id": f.zone_id,
                    "explanation": f.explanation,
                    "evidence": f.evidence,
                    "agent": f.agent,
                    "at": ctx.timestamp.isoformat(),
                },
                source=f.agent,
                severity=f.severity,
            )

        incidents = self._correlate(ctx, findings)

        return {
            "camera_id": ctx.camera.camera_id,
            "frame_index": ctx.frame_index,
            "timestamp": ctx.timestamp.isoformat(),
            "track_count": len(ctx.tracks),
            "person_count": len(ctx.persons()),
            "findings": len(findings),
            "incidents": incidents,
            "agent_timings_ms": timings,
            "crowd": self.crowd.latest(ctx.camera.camera_id),
            "device": ctx.device,
        }

    # ---------------------------------------------------------- correlation
    def _correlate(self, ctx: FrameContext, findings: List[Finding]) -> List[Dict[str, Any]]:
        actionable = [f for f in findings if f.behavior not in NON_INCIDENT_BEHAVIORS]
        context_only = [f for f in findings if f.behavior in NON_INCIDENT_BEHAVIORS]

        now = ctx.timestamp
        emitted: List[Dict[str, Any]] = []

        with self._lock:
            open_list = self._open[ctx.camera.camera_id]
            # retire stale incidents
            open_list[:] = [oi for oi in open_list if (now - oi.last_update) <= INCIDENT_WINDOW]

            if not actionable:
                return emitted

            target: Optional[OpenIncident] = next(
                (oi for oi in open_list if oi.absorbs(actionable[0], now)), None
            )
            is_new = target is None
            if target is None:
                target = OpenIncident(
                    incident_id=new_id("INC"),
                    camera_id=ctx.camera.camera_id,
                    started_at=now,
                    last_update=now,
                )
                open_list.append(target)
                self.incidents_opened += 1

            before = set(target.behaviors)
            target.findings.extend(actionable)
            target.behaviors.update(f.behavior for f in actionable)
            target.last_update = now

            # Context signals inform the assessment without opening incidents.
            scoring_pool = target.findings + context_only
            changed = is_new or target.behaviors != before

        zone = self._zone_for(ctx, actionable[0].zone_id)
        assessment = self.threat.assess(
            scoring_pool,
            self._policy,
            now=now,
            zone_risk_weight=float(zone.get("risk_weight", 1.0)) if zone else 1.0,
            context={"zone_name": zone.get("name") if zone else None},
        )

        previous_band = target.severity
        target.risk_score = assessment.score
        target.severity = assessment.band

        # Re-narrate on creation, on a new behaviour, or on a band change.
        if not (changed or assessment.band != previous_band):
            return emitted

        composed = self.commander.compose(
            findings=scoring_pool,
            assessment=assessment,
            camera={
                "id": ctx.camera.camera_id,
                "name": ctx.camera.name,
                "location": ctx.camera.location,
            },
            zone=zone,
            crowd=self.crowd.latest(ctx.camera.camera_id),
            now=now,
        )

        payload = {
            "id": target.incident_id,
            "is_new": is_new,
            "camera_id": ctx.camera.camera_id,
            "camera_name": ctx.camera.name,
            "site_id": ctx.camera.site_id,
            "zone_id": actionable[0].zone_id or ctx.camera.zone_id,
            "started_at": target.started_at.isoformat(),
            "last_update_at": now.isoformat(),
            "track_ids": sorted({f.track_id for f in target.findings if f.track_id}),
            "object_ids": sorted({f.object_id for f in target.findings if f.object_id}),
            "global_ids": sorted(
                {t.global_id for t in ctx.tracks if t.global_id and str(t.track_id)
                 in {f.track_id for f in target.findings if f.track_id}}
            ),
            "behaviors": sorted(target.behaviors),
            "location": {
                "latitude": ctx.camera.latitude,
                "longitude": ctx.camera.longitude,
                "floor": ctx.camera.floor,
                "zone_name": zone.get("name") if zone else None,
            },
            **composed,
        }

        bus.publish(
            "incident.created" if is_new else "incident.updated",
            payload,
            source="orchestrator",
            severity=assessment.band,
        )
        for sink in self._sinks:
            try:
                sink(payload)
            except Exception:
                log.exception("incident sink failed")

        target.persisted = True
        emitted.append(payload)
        return emitted

    @staticmethod
    def _zone_for(ctx: FrameContext, zone_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not zone_id:
            return None
        return next((z for z in ctx.camera.zones if z.get("id") == zone_id), None)

    # -------------------------------------------------------------- reading
    def status(self) -> Dict[str, Any]:
        dm = get_device_manager()
        with self._lock:
            open_count = sum(len(v) for v in self._open.values())
        return {
            "agents": [a.status() for a in self.agents],
            "reasoners": [
                {
                    "name": self.threat.name,
                    "spec_id": self.threat.spec_id,
                    "description": self.threat.description,
                    "enabled": True,
                },
                {
                    "name": self.commander.name,
                    "spec_id": self.commander.spec_id,
                    "description": self.commander.description,
                    "enabled": True,
                    "llm_successes": self.commander.llm_successes,
                    "llm_failures": self.commander.llm_failures,
                },
            ],
            "frames_processed": self.frames_processed,
            "findings_emitted": self.findings_emitted,
            "incidents_opened": self.incidents_opened,
            "open_incidents": open_count,
            "policy_loaded_at": self._policy_loaded_at.isoformat(),
            "device": dm.active_device,
            "gpu_enabled": dm.gpu_enabled,
        }

    def open_incidents(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "id": oi.incident_id,
                    "camera_id": oi.camera_id,
                    "started_at": oi.started_at.isoformat(),
                    "last_update_at": oi.last_update.isoformat(),
                    "risk_score": round(oi.risk_score, 1),
                    "severity": oi.severity,
                    "behaviors": sorted(oi.behaviors),
                    "finding_count": len(oi.findings),
                }
                for lst in self._open.values()
                for oi in lst
            ]


_orchestrator: Optional[Orchestrator] = None
_orch_lock = threading.Lock()


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        with _orch_lock:
            if _orchestrator is None:
                _orchestrator = Orchestrator()
    return _orchestrator

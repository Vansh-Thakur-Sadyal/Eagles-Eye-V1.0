"""Agent 11 - AI Incident Commander (spec S15).

Turns the agents' findings plus a threat assessment into:
  * a human-readable incident summary,
  * an ordered timeline (spec S21),
  * ranked *recommended* actions.

The spec's boundary is enforced structurally: every action this module produces
carries ``requires_human_approval``, and actions with real-world consequence
(dispatch, access restriction, identity escalation) are always True.  Nothing
here executes anything; the API applies an action only after an operator
confirms it.

Narration prefers the configured LLM and falls back to deterministic templates,
so an unreachable model degrades wording, never capability.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..rag.llm import LLMUnavailable, get_llm
from .base import Finding
from .threat import ThreatAssessment

log = logging.getLogger("sentinel.agents.commander")

SYSTEM_PROMPT = """You are the Incident Commander module of Eagles Eye, a public-safety
video intelligence platform used by authorised security operators.

Rules you must never break:
- You describe observations and confidence. You never assert a person's identity.
- You never infer the contents of a bag, package or container.
- You never accuse anyone of a crime or use words like "stalker", "criminal" or "terrorist".
- You recommend actions for a human to approve. You never claim to have taken action.
- If evidence is weak, say so plainly.
- Write in plain operational English, 2-4 sentences, no markdown, no preamble.
"""

# Consequence -> whether a human must approve. Anything touching the physical
# world or a person's identity is always True.
ACTION_CATALOGUE: Dict[str, Dict[str, Any]] = {
    "notify_operator": {
        "label": "Notify watch operator",
        "requires_human_approval": False,
        "consequence": "informational",
    },
    "display_on_dashboard": {
        "label": "Raise on command dashboard",
        "requires_human_approval": False,
        "consequence": "informational",
    },
    "continue_correlation": {
        "label": "Continue cross-camera correlation",
        "requires_human_approval": False,
        "consequence": "analytical",
    },
    "preserve_evidence": {
        "label": "Preserve video segment as evidence",
        "requires_human_approval": False,
        "consequence": "retention",
    },
    "review_footage": {
        "label": "Queue footage for human review",
        "requires_human_approval": False,
        "consequence": "analytical",
    },
    "dispatch_team": {
        "label": "Dispatch nearest authorised team",
        "requires_human_approval": True,
        "consequence": "physical response",
    },
    "restrict_access": {
        "label": "Restrict access to the affected zone",
        "requires_human_approval": True,
        "consequence": "physical access control",
    },
    "escalate_identity": {
        "label": "Request identity escalation",
        "requires_human_approval": True,
        "consequence": "privacy escalation",
    },
    "public_announcement": {
        "label": "Request public announcement",
        "requires_human_approval": True,
        "consequence": "public communication",
    },
    "crowd_control": {
        "label": "Initiate crowd-management protocol",
        "requires_human_approval": True,
        "consequence": "physical response",
    },
}

# behaviour -> ordered action keys
PLAYBOOK: Dict[str, List[str]] = {
    "unattended_object": [
        "notify_operator", "display_on_dashboard", "preserve_evidence",
        "continue_correlation", "dispatch_team", "restrict_access",
    ],
    "object_separation": ["notify_operator", "continue_correlation", "review_footage"],
    "restricted_entry": ["notify_operator", "display_on_dashboard", "preserve_evidence", "dispatch_team"],
    "following": ["notify_operator", "review_footage", "continue_correlation", "dispatch_team"],
    "crowd_density_critical": ["notify_operator", "display_on_dashboard", "crowd_control", "public_announcement"],
    "crowd_surge": ["notify_operator", "display_on_dashboard", "crowd_control"],
    "crowd_reversal": ["notify_operator", "display_on_dashboard", "crowd_control", "dispatch_team"],
    "crowd_flow_anomaly": ["notify_operator", "review_footage"],
    "violence_detected": ["notify_operator", "display_on_dashboard", "preserve_evidence", "dispatch_team"],
    "anomalous_activity": ["notify_operator", "review_footage", "preserve_evidence"],
    "fire_smoke": ["notify_operator", "display_on_dashboard", "dispatch_team", "public_announcement"],
    "watchlist_match": ["notify_operator", "review_footage", "preserve_evidence", "escalate_identity"],
    "loitering": ["notify_operator", "review_footage"],
    "counter_flow": ["notify_operator", "review_footage"],
    "running": ["notify_operator", "review_footage"],
    "sudden_movement": ["notify_operator", "review_footage"],
    "abnormal_trajectory": ["notify_operator", "review_footage"],
}

TITLES: Dict[str, str] = {
    "unattended_object": "Potentially unattended object",
    "object_separation": "Object separated from associated subject",
    "restricted_entry": "Restricted-zone entry",
    "following": "Possible persistent following pattern",
    "loitering": "Prolonged confined presence",
    "counter_flow": "Movement against prevailing flow",
    "running": "Running detected",
    "sudden_movement": "Sudden movement detected",
    "abnormal_trajectory": "Non-goal-directed movement",
    "crowd_surge": "Rapid occupancy increase",
    "crowd_reversal": "Crowd direction reversal",
    "crowd_density_critical": "Critical crowd density",
    "crowd_flow_anomaly": "Incoherent crowd flow",
    "watchlist_match": "Possible sighting of enrolled subject",
    "violence_detected": "Possible violent activity",
    "anomalous_activity": "Anomalous scene activity",
    "fire_smoke": "Possible smoke or fire",
}


class IncidentCommander:
    name = "commander"
    spec_id = 11
    description = "Operational reasoning, incident narration and response recommendation"

    def __init__(self) -> None:
        self.llm_failures = 0
        self.llm_successes = 0

    # ---------------------------------------------------------------- build
    def compose(
        self,
        *,
        findings: List[Finding],
        assessment: ThreatAssessment,
        camera: Dict[str, Any],
        zone: Optional[Dict[str, Any]] = None,
        crowd: Optional[Dict[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        primary = self._primary(findings)
        event_type = primary.behavior if primary else "observation"

        ordered = ([primary] + [f for f in findings if f is not primary]) if primary else findings
        facts = self._facts(ordered, assessment, camera, zone, crowd)
        summary = self._narrate(facts)
        timeline = self._timeline(findings, assessment, now)
        actions = self._actions(findings, assessment)

        return {
            "event_type": event_type,
            "title": TITLES.get(event_type, event_type.replace("_", " ").capitalize()),
            "summary": summary,
            "explanation": assessment.explanation,
            "risk_score": round(assessment.score, 1),
            "severity": assessment.band,
            "risk_factors": assessment.factors,
            "confidence": round(assessment.confidence, 3),
            "recommended_actions": actions,
            "timeline": timeline,
            "facts": facts,
            "narration_source": facts.get("narration_source", "template"),
        }

    # ---------------------------------------------------------------- parts
    @staticmethod
    def _primary(findings: List[Finding]) -> Optional[Finding]:
        rank = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
        weighted = [f for f in findings if f.behavior not in ("face_unavailable", "appearance_change")]
        pool = weighted or findings
        if not pool:
            return None
        return max(pool, key=lambda f: (rank.get(f.severity, 0), f.confidence))

    @staticmethod
    def _facts(findings, assessment, camera, zone, crowd) -> Dict[str, Any]:
        return {
            "camera": {
                "id": camera.get("id"),
                "name": camera.get("name"),
                "location": camera.get("location"),
            },
            "zone": {"id": zone.get("id"), "name": zone.get("name"), "type": zone.get("zone_type")}
            if zone else None,
            "crowd": (
                {
                    "count": crowd.get("count"),
                    "density_band": crowd.get("density_band"),
                    "risk": crowd.get("risk"),
                }
                if crowd else None
            ),
            "risk_score": round(assessment.score, 1),
            "severity": assessment.band,
            "observations": [
                {
                    "behavior": f.behavior,
                    "confidence": round(f.confidence, 3),
                    "severity": f.severity,
                    "track_id": f.track_id,
                    "secondary_track_id": f.secondary_track_id,
                    "object_id": f.object_id,
                    "duration_seconds": f.duration_seconds,
                    "detail": f.explanation,
                }
                for f in findings
            ],
        }

    def _narrate(self, facts: Dict[str, Any]) -> str:
        llm = get_llm()
        if llm.enabled:
            try:
                import json

                text = llm.complete(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                "Write the incident summary for these facts.\n\n"
                                + json.dumps(facts, indent=2, default=str)
                            ),
                        },
                    ],
                    temperature=0.2,
                    max_tokens=320,
                )
                if text.strip():
                    self.llm_successes += 1
                    facts["narration_source"] = f"llm:{llm.provider}"
                    return text.strip()
            except LLMUnavailable as exc:
                self.llm_failures += 1
                log.warning("incident narration fell back to template: %s", exc)
            except Exception:
                self.llm_failures += 1
                log.exception("incident narration failed, using template")

        facts["narration_source"] = "template"
        return self._template_summary(facts)

    @staticmethod
    def _template_summary(facts: Dict[str, Any]) -> str:
        cam = facts.get("camera", {})
        where = cam.get("location") or cam.get("name") or cam.get("id") or "an unnamed camera"
        zone = facts.get("zone") or {}
        if zone.get("name"):
            where = f"{zone['name']} ({where})"

        obs = facts.get("observations", [])
        if not obs:
            return f"No weighted observations are currently active at {where}."

        lead = obs[0]
        sentences = [f"{lead['detail']}".rstrip(".") + f", observed at {where}."]

        others = [o for o in obs[1:] if o["behavior"] not in ("face_unavailable", "appearance_change")][:2]
        if others:
            extra = "; ".join(f"{o['behavior'].replace('_', ' ')} (confidence {o['confidence']:.2f})"
                              for o in others)
            sentences.append(f"Concurrent observations: {extra}.")

        crowd = facts.get("crowd")
        if crowd and crowd.get("count") is not None:
            sentences.append(
                f"Crowd density in view is {str(crowd.get('density_band', 'unknown')).upper()} "
                f"with {crowd['count']} subjects tracked."
            )

        sentences.append(
            f"Assessed risk {facts['risk_score']:.0f}% ({str(facts['severity']).upper()}). "
            "Recommendations require operator approval."
        )
        return " ".join(sentences)

    @staticmethod
    def _timeline(findings: List[Finding], assessment: ThreatAssessment, now: datetime) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for f in sorted(findings, key=lambda x: x.started_at or now):
            entries.append(
                {
                    "at": (f.started_at or now).isoformat(),
                    "kind": "detection",
                    "actor": f.agent or "agent",
                    "text": f.explanation or f.behavior.replace("_", " "),
                    "confidence": round(f.confidence, 3),
                    "payload": {
                        "behavior": f.behavior,
                        "track_id": f.track_id,
                        "object_id": f.object_id,
                    },
                }
            )
        entries.append(
            {
                "at": now.isoformat(),
                "kind": "assessment",
                "actor": "threat",
                "text": assessment.explanation,
                "confidence": round(assessment.confidence, 3),
                "payload": {"score": round(assessment.score, 1), "band": assessment.band},
            }
        )
        entries.append(
            {
                "at": now.isoformat(),
                "kind": "notification",
                "actor": "commander",
                "text": "Incident Commander generated a summary and response recommendations "
                        "for operator review.",
                "confidence": None,
                "payload": {},
            }
        )
        return entries

    @staticmethod
    def _actions(findings: List[Finding], assessment: ThreatAssessment) -> List[Dict[str, Any]]:
        keys: List[str] = []
        for f in sorted(findings, key=lambda x: -x.confidence):
            for key in PLAYBOOK.get(f.behavior, []):
                if key not in keys:
                    keys.append(key)
        if not keys:
            keys = ["notify_operator", "review_footage"]

        # High-consequence actions only surface once the score justifies them.
        band_rank = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        score_rank = band_rank.get(assessment.band, 0)

        actions: List[Dict[str, Any]] = []
        for priority, key in enumerate(keys, start=1):
            meta = ACTION_CATALOGUE.get(key)
            if meta is None:
                continue
            if meta["requires_human_approval"] and score_rank < 2:
                continue
            actions.append(
                {
                    "key": key,
                    "label": meta["label"],
                    "priority": priority,
                    "requires_human_approval": meta["requires_human_approval"],
                    "consequence": meta["consequence"],
                    "status": "proposed",
                    "rationale": f"Proposed for risk band {assessment.band.upper()}"
                                 + (f" driven by {assessment.dominant_factor}."
                                    if assessment.dominant_factor else "."),
                }
            )
        return actions

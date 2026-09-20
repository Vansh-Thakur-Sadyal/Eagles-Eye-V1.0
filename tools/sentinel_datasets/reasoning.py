"""Incident-reasoning and conversational Q&A datasets (write-up S40, S22).

Both are derived from incidents Sentinel *actually recorded*, not invented
scenarios. That matters: the reasoning targets are the system's own
explanations and the questions are answerable from its own evidence, so a model
fine-tuned on this learns the house style and the house guardrails rather than
a fictional one.

Two outputs:

  incidents/events.json      structured incident records (S40 / S45 shape)
  incidents/qa.jsonl         instruction-tuning pairs for the assistant

Every generated answer inherits the platform's rules: no identity assertions,
no contents inference, recommendations rather than actions.
"""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

BANNED = ("explosive", "bomb", "stalker", "criminal", "terrorist")

SYSTEM_RULES = (
    "You are Sentinel AI, a public-safety video intelligence assistant. "
    "Describe observations and confidence. Never assert a person's identity, "
    "never infer the contents of a bag or container, never accuse anyone. "
    "Recommend actions for a human to approve; never claim to have acted."
)


def load_incidents(database_url: Optional[str] = None, limit: int = 5000) -> List[Dict[str, Any]]:
    """Read recorded incidents straight out of the Sentinel database."""
    import sys

    backend = Path(__file__).resolve().parents[2] / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    if database_url:
        import os

        os.environ["SENTINEL_DATABASE_URL"] = database_url

    from sqlalchemy import select

    from app.db import session_scope
    from app.models import Camera, Incident, IncidentTimelineEntry

    rows: List[Dict[str, Any]] = []
    with session_scope() as db:
        cameras = {c.id: c for c in db.scalars(select(Camera)).all()}
        incidents = db.scalars(
            select(Incident).order_by(Incident.started_at.desc()).limit(limit)
        ).all()
        for incident in incidents:
            timeline = db.scalars(
                select(IncidentTimelineEntry)
                .where(IncidentTimelineEntry.incident_id == incident.id)
                .order_by(IncidentTimelineEntry.at)
            ).all()
            camera = cameras.get(incident.camera_id) if incident.camera_id else None
            rows.append(
                {
                    "incident_id": incident.id,
                    "camera_id": incident.camera_id,
                    "camera_name": camera.name if camera else None,
                    "camera_location": camera.location if camera else None,
                    "zone_id": incident.zone_id,
                    "zone_name": (incident.location or {}).get("zone_name"),
                    "timestamp": incident.started_at.isoformat(),
                    "last_update": incident.last_update_at.isoformat(),
                    "event": incident.event_type,
                    "title": incident.title,
                    "summary": incident.summary,
                    "explanation": incident.explanation,
                    "risk_score": incident.risk_score,
                    "severity": incident.severity,
                    "confidence": incident.confidence,
                    "risk_factors": incident.risk_factors or [],
                    "behaviors": (incident.meta or {}).get("behaviors", []),
                    "track_ids": incident.track_ids or [],
                    "object_ids": incident.object_ids or [],
                    "recommended_action": [
                        a.get("key") for a in (incident.recommended_actions or [])
                    ],
                    "status": incident.status,
                    "evidence": {
                        "timeline": [
                            {"at": e.at.isoformat(), "kind": e.kind, "actor": e.actor, "text": e.text}
                            for e in timeline
                        ]
                    },
                }
            )
    return rows


# ------------------------------------------------------------------- Q & A
def _clean(text: Optional[str]) -> str:
    return (text or "").strip()


def _safe(*parts: str) -> bool:
    blob = " ".join(parts).lower()
    return not any(word in blob for word in BANNED)


def qa_pairs(incident: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    """Build grounded question/answer pairs for one incident."""
    where = incident.get("zone_name") or incident.get("camera_location") or incident.get("camera_name") or "the monitored area"
    summary = _clean(incident.get("summary"))
    explanation = _clean(incident.get("explanation"))
    if not summary:
        return []

    pairs: List[Dict[str, Any]] = []

    def add(question: str, answer: str, kind: str) -> None:
        if answer and _safe(question, answer):
            pairs.append(
                {
                    "kind": kind,
                    "incident_id": incident["incident_id"],
                    "messages": [
                        {"role": "system", "content": SYSTEM_RULES},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    "grounding": {
                        "incident_id": incident["incident_id"],
                        "camera_id": incident.get("camera_id"),
                        "event": incident.get("event"),
                        "severity": incident.get("severity"),
                        "risk_score": incident.get("risk_score"),
                    },
                }
            )

    add(
        rng.choice([
            f"What happened at {where}?",
            f"Tell me about incident {incident['incident_id']}.",
            f"Summarise the {str(incident.get('event', '')).replace('_', ' ')} at {where}.",
        ]),
        summary,
        "summary",
    )

    if explanation:
        add(
            rng.choice([
                f"Why is incident {incident['incident_id']} rated {incident.get('risk_score', 0):.0f}?",
                "Why is the risk score what it is?",
                "Explain the risk assessment for this incident.",
            ]),
            explanation,
            "risk_explanation",
        )

    counted = [f for f in incident.get("risk_factors", []) if f.get("counted")]
    if counted:
        listing = "; ".join(
            f"{f['behavior'].replace('_', ' ')} (weight {f['weight']:.0f}, "
            f"confidence {f['confidence']:.2f}, contributing {f['contribution']:.1f})"
            for f in counted[:5]
        )
        add(
            "Which signals contributed to this score?",
            f"The weighted contributors were: {listing}. "
            "Observability signals such as face availability carry no weight by design.",
            "factor_breakdown",
        )

    actions = incident.get("recommended_action") or []
    if actions:
        readable = ", ".join(a.replace("_", " ") for a in actions[:5])
        add(
            "What should we do about it?",
            f"Recommended for operator approval: {readable}. "
            "Actions with a real-world consequence require a commander's authorisation; "
            "Sentinel does not carry them out.",
            "recommendation",
        )

    timeline = incident.get("evidence", {}).get("timeline", [])
    if len(timeline) >= 2:
        narrative = " ".join(
            f"{entry['at'][11:19]} - {entry['text']}" for entry in timeline[:6]
        )
        add("How did this incident develop?", narrative, "timeline")

    if "unattended_object" in incident.get("behaviors", []):
        add(
            "What is inside the bag?",
            "That cannot be determined from video, and Sentinel does not attempt to infer it. "
            "The system reports only that an object has been stationary without an associated "
            "subject nearby, and can trace where the last associated subject went.",
            "guardrail_contents",
        )

    if incident.get("track_ids"):
        add(
            "Who is the person involved?",
            f"The subject is tracked anonymously as {incident['track_ids'][0]}. "
            "Sentinel does not assert real-world identity; identity escalation requires a "
            "documented lawful basis and a commander's approval.",
            "guardrail_identity",
        )

    return pairs


def build(
    out_dir: Path,
    *,
    database_url: Optional[str] = None,
    limit: int = 5000,
    seed: int = 20260919,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    incidents = load_incidents(database_url, limit)

    folder = out_dir / "incidents"
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "events.json").write_text(json.dumps(incidents, indent=2), encoding="utf-8")

    trajectories = [
        {
            "incident_id": i["incident_id"],
            "camera_id": i["camera_id"],
            "track_ids": i["track_ids"],
            "object_ids": i["object_ids"],
            "timestamp": i["timestamp"],
        }
        for i in incidents
    ]
    (folder / "trajectories.json").write_text(json.dumps(trajectories, indent=2), encoding="utf-8")

    reports = [
        {
            "incident_id": i["incident_id"],
            "title": i["title"],
            "summary": i["summary"],
            "explanation": i["explanation"],
            "severity": i["severity"],
            "risk_score": i["risk_score"],
        }
        for i in incidents
        if i.get("summary")
    ]
    (folder / "reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")

    all_pairs: List[Dict[str, Any]] = []
    for incident in incidents:
        all_pairs.extend(qa_pairs(incident, rng))

    with (folder / "qa.jsonl").open("w", encoding="utf-8") as fh:
        for pair in all_pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

    by_kind: Dict[str, int] = {}
    for pair in all_pairs:
        by_kind[pair["kind"]] = by_kind.get(pair["kind"], 0) + 1

    return {
        "incidents": len(incidents),
        "qa_pairs": len(all_pairs),
        "qa_by_kind": by_kind,
        "files": ["events.json", "trajectories.json", "reports.json", "qa.jsonl"],
        "note": (
            "Derived from incidents this deployment actually recorded. Run the pipeline "
            "for longer, or against recorded footage, to grow the set. Guardrail pairs "
            "(contents and identity) are included so a fine-tuned model inherits the "
            "platform's limits rather than learning to overstep them."
        ),
        "warning": (
            None if len(incidents) >= 50
            else f"Only {len(incidents)} incident(s) available. This is far too few to "
                 "fine-tune on; treat it as a format check and collect more first."
        ),
    }

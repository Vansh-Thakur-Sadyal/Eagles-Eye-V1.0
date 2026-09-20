"""Agent 12 - Conversational Forensic Search (spec S16, S22, S23).

Answers natural-language operator questions against the incident record:

    "Show me all incidents near Gate 4 between 7 PM and 9 PM."
    "Show me the incident involving the unattended bag."
    "Where did the associated subject go?"

The pipeline is query -> structured filters -> vector retrieval -> metadata
filter -> grounded answer.  Filters are extracted deterministically (regex and
a controlled vocabulary) first; the LLM is used to *phrase* the answer over
retrieved evidence, never to invent facts.  With no LLM configured the answer
is composed from templates over the same evidence, so the feature works
offline and can never hallucinate an incident that is not in the store.

Every answer carries citations - the incident IDs it was built from.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from .llm import LLMUnavailable, get_llm
from .vectorstore import SearchHit, get_store

log = logging.getLogger("sentinel.forensic")

SYSTEM_PROMPT = """You are the Forensic Search module of Eagles Eye, answering an
authorised security operator's questions about recorded incidents.

Rules:
- Answer ONLY from the retrieved evidence you are given. If the evidence does not
  contain the answer, say so plainly.
- Never invent an incident, camera, time or subject.
- Refer to people as anonymous track or subject identifiers. Never assert a person's
  real-world identity.
- Never state or guess the contents of a bag or container.
- Cite the incident IDs you used.
- Be concise and operational: 2-5 sentences.
"""

BEHAVIOR_SYNONYMS: Dict[str, List[str]] = {
    "unattended_object": ["unattended", "abandoned", "left behind", "unattended bag",
                          "abandoned bag", "suitcase", "luggage", "left luggage"],
    "following": ["following", "followed", "trailing", "shadowing", "tailing"],
    "loitering": ["loitering", "waiting too long", "hanging around", "lingering"],
    "counter_flow": ["counter flow", "counter-flow", "wrong direction", "against the flow",
                     "wrong way"],
    "running": ["running", "ran", "sprinting", "fleeing"],
    "restricted_entry": ["restricted", "unauthorised", "unauthorized", "trespass",
                         "restricted area", "restricted zone", "secure area"],
    "crowd_surge": ["surge", "crowd build", "overcrowding", "crowding"],
    "crowd_density_critical": ["crowd density", "packed", "congestion", "critical density"],
    "crowd_reversal": ["crowd reversal", "crowd turned", "reversal"],
    "watchlist_match": ["watchlist", "person of interest", "sighting", "matched",
                        "missing person"],
    "abnormal_trajectory": ["erratic", "abnormal movement", "wandering"],
    "sudden_movement": ["sudden movement", "sudden"],
    "violence_detected": ["fight", "fighting", "violence", "assault", "altercation"],
    "fire_smoke": ["fire", "smoke", "flames"],
}

SEVERITIES = ["info", "low", "medium", "high", "critical"]


@dataclass
class ParsedQuery:
    text: str
    behaviors: List[str] = field(default_factory=list)
    severities: List[str] = field(default_factory=list)
    camera_hints: List[str] = field(default_factory=list)
    zone_hints: List[str] = field(default_factory=list)
    track_ids: List[str] = field(default_factory=list)
    incident_ids: List[str] = field(default_factory=list)
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    limit: int = 8
    intent: str = "search"     # search | detail | trajectory | count | timeline

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "behaviors": self.behaviors,
            "severities": self.severities,
            "camera_hints": self.camera_hints,
            "zone_hints": self.zone_hints,
            "track_ids": self.track_ids,
            "incident_ids": self.incident_ids,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "intent": self.intent,
            "limit": self.limit,
        }


class QueryParser:
    """Deterministic natural-language -> structured filters."""

    def parse(self, text: str, *, now: Optional[datetime] = None) -> ParsedQuery:
        now = now or datetime.now(timezone.utc)
        low = (text or "").lower()
        q = ParsedQuery(text=text)

        for behavior, phrases in BEHAVIOR_SYNONYMS.items():
            if any(p in low for p in phrases):
                q.behaviors.append(behavior)

        for sev in SEVERITIES:
            if re.search(rf"\b{sev}\b", low):
                q.severities.append(sev)
        if "critical" in low or "urgent" in low:
            q.severities = list(dict.fromkeys(q.severities + ["critical"]))

        q.incident_ids = [m.upper() for m in re.findall(r"\b(INC[_-][A-Za-z0-9]+)\b", text)]
        q.track_ids = [m.upper() for m in re.findall(r"\b(?:track|subject)\s*#?\s*([A-Za-z0-9_]+)", low)]

        # "near Gate 4", "at Platform 3", "in Terminal A", "camera 12"
        for pattern in (
            r"(?:near|at|in|around|by)\s+((?:gate|platform|terminal|zone|hall|concourse|entrance|exit|apron|lobby)\s*[a-z0-9]+)",
            r"\b(camera\s*[a-z0-9_]+)\b",
            r"\b(cam[_-][a-z0-9]+)\b",
        ):
            for m in re.findall(pattern, low):
                q.zone_hints.append(m.strip())

        q.start, q.end = self._time_window(low, now)

        if any(w in low for w in ("where did", "trajectory", "route", "path", "go next", "went")):
            q.intent = "trajectory"
        elif any(w in low for w in ("how many", "count", "number of")):
            q.intent = "count"
        elif any(w in low for w in ("how did", "timeline", "develop", "what happened")):
            q.intent = "timeline"
        elif q.incident_ids or "details" in low or "tell me about" in low:
            q.intent = "detail"

        m = re.search(r"\b(?:last|latest|top)\s+(\d{1,2})\b", low)
        if m:
            q.limit = max(1, min(50, int(m.group(1))))

        return q

    @staticmethod
    def _time_window(low: str, now: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
        # "between 7 pm and 9 pm"
        m = re.search(
            r"between\s+(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)?\s+and\s+(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)?",
            low,
        )
        if m:
            start = _clock(now, int(m.group(1)), int(m.group(2) or 0), m.group(3))
            end = _clock(now, int(m.group(4)), int(m.group(5) or 0), m.group(6))
            if end < start:
                end += timedelta(days=1)
            return start, end

        m = re.search(r"(?:in the )?last\s+(\d+)\s*(minute|min|hour|hr|day|week)s?", low)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            delta = {
                "minute": timedelta(minutes=n), "min": timedelta(minutes=n),
                "hour": timedelta(hours=n), "hr": timedelta(hours=n),
                "day": timedelta(days=n), "week": timedelta(weeks=n),
            }[unit]
            return now - delta, now

        if "today" in low:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return start, now
        if "yesterday" in low:
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            return start, start + timedelta(days=1)
        if "this week" in low:
            return now - timedelta(days=7), now
        if "this hour" in low or "right now" in low or "currently" in low:
            return now - timedelta(hours=1), now
        return None, None


def _passes(hit: SearchHit, where: Dict[str, Any]) -> bool:
    for key, expected in where.items():
        actual = hit.metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _clock(now: datetime, hour: int, minute: int, meridiem: Optional[str]) -> datetime:
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    return now.replace(hour=min(23, hour), minute=min(59, minute), second=0, microsecond=0)


class ForensicSearch:
    """Retrieval + grounded answer generation."""

    name = "forensic"
    spec_id = 12
    description = "Conversational, evidence-grounded investigation over the incident record"

    def __init__(self) -> None:
        self.parser = QueryParser()
        self.queries_served = 0
        self.llm_answers = 0
        self.template_answers = 0

    # ------------------------------------------------------------- indexing
    def index_incident(self, incident: Dict[str, Any]) -> None:
        from .vectorstore import incident_to_document

        try:
            get_store().upsert([incident_to_document(incident)])
        except Exception:
            log.exception("failed to index incident %s", incident.get("id"))

    def index_many(self, incidents: List[Dict[str, Any]]) -> int:
        from .vectorstore import incident_to_document

        try:
            return get_store().upsert([incident_to_document(i) for i in incidents])
        except Exception:
            log.exception("bulk index failed")
            return 0

    # -------------------------------------------------------------- retrieve
    def retrieve(self, query: ParsedQuery) -> List[SearchHit]:
        store = get_store()
        where: Dict[str, Any] = {}
        if query.severities:
            where["severity"] = query.severities
        if query.behaviors and len(query.behaviors) == 1:
            where["event_type"] = query.behaviors[0]

        search_text = query.text
        if query.behaviors:
            search_text = f"{search_text} {' '.join(query.behaviors)}"
        if query.zone_hints:
            search_text = f"{search_text} {' '.join(query.zone_hints)}"

        hits = store.search(search_text, top_k=max(query.limit * 3, 15),
                            where=where or None)

        # A query like "all incidents today" carries filters but no distinctive
        # keywords, so term search legitimately matches nothing. Fall back to a
        # filtered listing - but ONLY when the query actually carries a filter.
        # A query with neither keywords nor filters ("zeppelin collision on
        # platform 94") must return nothing rather than an arbitrary incident.
        filter_driven = bool(query.start or query.end or query.severities or query.behaviors)
        if not hits and filter_driven:
            hits = [
                hit for hit in store.all_documents(limit=500)
                if not where or _passes(hit, where)
            ]
            hits.sort(key=lambda h: float(h.metadata.get("started_ts") or 0), reverse=True)

        # Time filtering is applied after retrieval so a mismatched metadata
        # key can never silently return nothing.
        if query.start or query.end:
            start_ts = query.start.timestamp() if query.start else 0.0
            end_ts = query.end.timestamp() if query.end else float("inf")
            filtered = [
                h for h in hits
                if start_ts <= float(h.metadata.get("started_ts") or 0) <= end_ts
            ]
            if filtered or not hits:
                hits = filtered

        if query.incident_ids:
            preferred = [h for h in hits if h.doc_id in query.incident_ids]
            if preferred:
                hits = preferred

        return hits[: query.limit]

    # ---------------------------------------------------------------- answer
    def answer(
        self,
        question: str,
        *,
        now: Optional[datetime] = None,
        extra_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self.queries_served += 1
        query = self.parser.parse(question, now=now)
        hits = self.retrieve(query)

        evidence = [h.to_dict() for h in hits]
        citations = [
            {
                "incident_id": h.doc_id,
                "camera_id": h.metadata.get("camera_id"),
                "camera_name": h.metadata.get("camera_name"),
                "severity": h.metadata.get("severity"),
                "risk_score": h.metadata.get("risk_score"),
                "started_at": h.metadata.get("started_at"),
                "relevance": round(h.score, 3),
            }
            for h in hits
        ]

        text, source = self._compose(question, query, hits, extra_context or [])
        return {
            "question": question,
            "answer": text,
            "answer_source": source,
            "parsed_query": query.to_dict(),
            "citations": citations,
            "evidence": evidence,
            "result_count": len(hits),
            "store_backend": get_store().backend,
        }

    def _compose(self, question, query, hits, extra) -> Tuple[str, str]:
        llm = get_llm()
        if llm.enabled:
            try:
                import json

                payload = {
                    "question": question,
                    "parsed_filters": query.to_dict(),
                    "retrieved_incidents": [h.to_dict() for h in hits],
                    "additional_context": extra,
                }
                text = llm.complete(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload, indent=2, default=str)},
                    ],
                    temperature=0.1,
                    max_tokens=600,
                )
                if text.strip():
                    self.llm_answers += 1
                    return text.strip(), f"llm:{llm.provider}"
            except LLMUnavailable as exc:
                log.info("forensic answer fell back to template: %s", exc)
            except Exception:
                log.exception("forensic LLM answer failed")

        self.template_answers += 1
        return self._template_answer(query, hits, extra), "template"

    @staticmethod
    def _template_answer(query: ParsedQuery, hits: List[SearchHit],
                         extra: List[Dict[str, Any]]) -> str:
        if not hits:
            bits = []
            if query.behaviors:
                bits.append("event type " + ", ".join(b.replace("_", " ") for b in query.behaviors))
            if query.start and query.end:
                same_day = query.start.date() == query.end.date()
                fmt = "%H:%M" if same_day else "%Y-%m-%d %H:%M"
                bits.append(f"between {query.start:{fmt}} and {query.end:{fmt}}")
            if query.zone_hints:
                bits.append("near " + ", ".join(query.zone_hints))
            where = " matching " + " ".join(bits) if bits else ""
            return (
                f"No recorded incidents{where} were found in the indexed evidence. "
                "Widen the time window or relax the filters and try again."
            )

        if query.intent == "count":
            by_sev: Dict[str, int] = {}
            for h in hits:
                sev = str(h.metadata.get("severity", "unknown"))
                by_sev[sev] = by_sev.get(sev, 0) + 1
            breakdown = ", ".join(f"{n} {s}" for s, n in sorted(by_sev.items()))
            return f"Found {len(hits)} matching incident(s): {breakdown}."

        lead = hits[0]
        meta = lead.metadata
        lines = [
            f"Found {len(hits)} relevant incident(s). "
            f"The closest match is {lead.doc_id} - "
            f"{str(meta.get('event_type', 'incident')).replace('_', ' ')} on "
            f"{meta.get('camera_name') or meta.get('camera_id') or 'an unnamed camera'}, "
            f"severity {str(meta.get('severity', 'unknown')).upper()}, "
            f"risk {float(meta.get('risk_score') or 0):.0f}%."
        ]
        started = meta.get("started_at")
        if started:
            lines.append(f"It began at {started}.")

        summary_line = (lead.text or "").split("\n")
        if len(summary_line) > 1 and summary_line[1].strip():
            lines.append(summary_line[1].strip())

        if len(hits) > 1:
            others = ", ".join(h.doc_id for h in hits[1:5])
            lines.append(f"Other matches: {others}.")

        if query.intent == "trajectory":
            lines.append(
                "Use the incident's track IDs with the tracks endpoint to replay the "
                "associated subject's cross-camera trajectory."
            )
        if query.intent == "timeline":
            lines.append(f"Request the timeline for {lead.doc_id} to see how it developed.")

        return " ".join(lines)

    def status(self) -> Dict[str, Any]:
        store = get_store()
        llm = get_llm()
        return {
            "name": self.name,
            "spec_id": self.spec_id,
            "description": self.description,
            "queries_served": self.queries_served,
            "llm_answers": self.llm_answers,
            "template_answers": self.template_answers,
            "store": store.info(),
            "llm": llm.info(),
        }


_forensic: Optional[ForensicSearch] = None


def get_forensic() -> ForensicSearch:
    global _forensic
    if _forensic is None:
        _forensic = ForensicSearch()
    return _forensic

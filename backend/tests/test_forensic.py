"""Forensic search: query parsing, retrieval and grounding guarantees."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.rag.forensic import ForensicSearch, QueryParser
from app.rag.vectorstore import Document, MemoryStore, incident_to_document

NOW = datetime(2026, 1, 15, 20, 30, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------ parsing
@pytest.fixture
def parser():
    return QueryParser()


def test_parses_behaviour_from_plain_language(parser):
    q = parser.parse("show me unattended bags", now=NOW)
    assert "unattended_object" in q.behaviors


def test_parses_an_explicit_time_range(parser):
    q = parser.parse("incidents near Gate 4 between 7 PM and 9 PM", now=NOW)
    assert q.start and q.end
    assert q.start.hour == 19 and q.end.hour == 21
    assert any("gate 4" in hint for hint in q.zone_hints)


def test_parses_a_relative_window(parser):
    q = parser.parse("unattended objects in the last 2 hours", now=NOW)
    assert q.start == NOW - timedelta(hours=2)
    assert q.end == NOW


def test_range_crossing_midnight_rolls_forward(parser):
    q = parser.parse("anything between 11 pm and 2 am", now=NOW)
    assert q.end > q.start
    assert (q.end - q.start) == timedelta(hours=3)


def test_recognises_intent(parser):
    assert parser.parse("how many incidents today", now=NOW).intent == "count"
    assert parser.parse("where did the subject go", now=NOW).intent == "trajectory"
    assert parser.parse("how did this incident develop", now=NOW).intent == "timeline"


def test_picks_up_incident_ids(parser):
    q = parser.parse("tell me about INC_ABC123", now=NOW)
    assert q.incident_ids == ["INC_ABC123"]
    assert q.intent == "detail"


# ---------------------------------------------------------------- retrieval
def _incident(idx: int, event_type: str, severity: str, started: datetime, summary: str):
    return {
        "id": f"INC_TEST{idx:03d}",
        "title": event_type.replace("_", " ").title(),
        "summary": summary,
        "explanation": f"Risk driven by {event_type}.",
        "behaviors": [event_type],
        "camera_id": f"CAM_{idx}",
        "camera_name": f"Camera {idx}",
        "zone_id": "ZONE_A",
        "site_id": "SITE_A",
        "event_type": event_type,
        "severity": severity,
        "risk_score": 50 + idx,
        "started_at": started.isoformat(),
        "location": {"zone_name": "Gate 4"},
        "track_ids": [],
    }


@pytest.fixture
def populated(monkeypatch):
    """A ForensicSearch backed by a fresh in-memory store."""
    store = MemoryStore()
    import app.rag.forensic as forensic_module
    import app.rag.vectorstore as vs

    monkeypatch.setattr(vs, "_store", store, raising=False)
    monkeypatch.setattr(forensic_module, "get_store", lambda: store)

    incidents = [
        _incident(1, "unattended_object", "high", NOW - timedelta(minutes=30),
                  "A suitcase has been stationary near Gate 4 for 78 seconds."),
        _incident(2, "crowd_surge", "medium", NOW - timedelta(hours=3),
                  "Occupancy rose 60% in the west concourse."),
        _incident(3, "restricted_entry", "high", NOW - timedelta(days=4),
                  "Subject entered the apron, classified restricted."),
    ]
    store.upsert([incident_to_document(i) for i in incidents])
    return ForensicSearch(), store


def test_keyword_query_finds_the_right_incident(populated):
    forensic, _ = populated
    result = forensic.answer("show me the unattended suitcase", now=NOW)
    assert result["result_count"] >= 1
    assert result["citations"][0]["incident_id"] == "INC_TEST001"


def test_filter_only_query_falls_back_to_a_listing(populated):
    """'all incidents today' has filters but no distinctive keywords."""
    forensic, _ = populated
    result = forensic.answer("show me all incidents today", now=NOW)
    assert result["result_count"] >= 1, "a filter-driven query returned nothing"


def test_nonsense_query_returns_nothing(populated):
    """The fallback must not turn an unmatched query into an arbitrary hit."""
    forensic, _ = populated
    result = forensic.answer("zeppelin collision on platform 94 in 1873", now=NOW)
    assert result["result_count"] == 0
    assert "no recorded incidents" in result["answer"].lower()


def test_time_window_excludes_older_incidents(populated):
    forensic, _ = populated
    result = forensic.answer("incidents in the last 1 hour", now=NOW)
    ids = {c["incident_id"] for c in result["citations"]}
    assert "INC_TEST003" not in ids, "a four-day-old incident leaked into a one-hour window"


def test_severity_filter_is_applied(populated):
    forensic, _ = populated
    result = forensic.answer("show me high severity incidents", now=NOW)
    for citation in result["citations"]:
        assert citation["severity"] == "high"


def test_behaviour_filter_excludes_other_event_types(populated):
    forensic, _ = populated
    result = forensic.answer("restricted zone entries", now=NOW)
    ids = {c["incident_id"] for c in result["citations"]}
    assert ids <= {"INC_TEST003"}, ids


def test_answer_only_cites_retrieved_incidents(populated):
    """Grounding: every citation must correspond to a real indexed document."""
    forensic, store = populated
    result = forensic.answer("unattended", now=NOW)
    known = {hit.doc_id for hit in store.all_documents()}
    for citation in result["citations"]:
        assert citation["incident_id"] in known


def test_empty_index_says_so(monkeypatch):
    store = MemoryStore()
    import app.rag.forensic as forensic_module

    monkeypatch.setattr(forensic_module, "get_store", lambda: store)
    result = ForensicSearch().answer("anything at all today", now=NOW)
    assert result["result_count"] == 0
    assert "no recorded incidents" in result["answer"].lower()


# ------------------------------------------------------------------- store
def test_memory_store_upsert_replaces_rather_than_duplicates():
    store = MemoryStore()
    store.upsert([Document("A", "unattended suitcase at gate four")])
    store.upsert([Document("A", "crowd surge in the concourse")])
    assert store.count() == 1
    assert store.search("suitcase") == []
    assert len(store.search("crowd surge")) == 1


def test_memory_store_delete():
    store = MemoryStore()
    store.upsert([Document("A", "restricted entry apron")])
    assert store.delete("A") is True
    assert store.delete("A") is False
    assert store.count() == 0


def test_metadata_filter_narrows_results():
    store = MemoryStore()
    store.upsert([
        Document("A", "crowd surge concourse", {"severity": "high"}),
        Document("B", "crowd surge platform", {"severity": "low"}),
    ])
    hits = store.search("crowd surge", where={"severity": "high"})
    assert [h.doc_id for h in hits] == ["A"]

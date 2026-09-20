"""Conversational forensic search and the AI assistant (spec S16, S22, S23)."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...agents.orchestrator import get_orchestrator
from ...core.security import current_user, require
from ...db import get_session
from ...models import ChatMessage, Incident, User, new_id, utcnow
from ...rag.forensic import get_forensic
from ...rag.vectorstore import get_store, incident_to_document
from ..deps import record, to_dict

router = APIRouter(prefix="/api/forensic", tags=["forensic"])

SUGGESTIONS = [
    "Show me all incidents near Gate 4 between 7 PM and 9 PM",
    "Show me unattended objects detected in the last hour",
    "Which zone currently has the highest crowd risk?",
    "Find possible following patterns today",
    "How many critical incidents were raised this week?",
    "Show me restricted-zone entries in the last 24 hours",
]


class SearchBody(BaseModel):
    query: str
    limit: int = 10


@router.post("/search")
def search(
    body: SearchBody,
    request: Request,
    user: User = Depends(require("incident:read")),
    db: Session = Depends(get_session),
):
    """Natural-language search over the indexed incident record."""
    if not body.query.strip():
        raise HTTPException(400, "query cannot be empty")

    forensic = get_forensic()
    result = forensic.answer(body.query)
    record(db, request, user, "forensic.search", "incident", None,
           detail={"query": body.query[:200], "hits": result["result_count"]})

    # Attach the live incident rows so the UI can render full cards.
    ids = [c["incident_id"] for c in result["citations"]]
    if ids:
        rows = db.scalars(select(Incident).where(Incident.id.in_(ids))).all()
        by_id = {r.id: to_dict(r) for r in rows}
        result["incidents"] = [by_id[i] for i in ids if i in by_id]
    else:
        result["incidents"] = []
    return result


class ChatBody(BaseModel):
    message: str
    session_id: Optional[str] = None


@router.post("/chat")
def chat(
    body: ChatBody,
    request: Request,
    user: User = Depends(require("incident:read")),
    db: Session = Depends(get_session),
):
    """Conversational investigation, grounded in retrieved evidence."""
    if not body.message.strip():
        raise HTTPException(400, "message cannot be empty")

    session_id = body.session_id or f"CHAT_{uuid.uuid4().hex[:10].upper()}"
    started = time.perf_counter()

    db.add(
        ChatMessage(id=new_id("MSG"), session_id=session_id, at=utcnow(), role="user",
                    content=body.message, actor=user.username)
    )

    context = _live_context()
    result = get_forensic().answer(body.message, extra_context=context)
    latency = (time.perf_counter() - started) * 1000.0

    db.add(
        ChatMessage(
            id=new_id("MSG"), session_id=session_id, at=utcnow(), role="assistant",
            content=result["answer"], actor="sentinel",
            citations=result["citations"], latency_ms=latency,
        )
    )
    db.commit()
    record(db, request, user, "forensic.chat", "chat", session_id,
           detail={"message": body.message[:200]})

    return {
        "session_id": session_id,
        "answer": result["answer"],
        "answer_source": result["answer_source"],
        "citations": result["citations"],
        "parsed_query": result["parsed_query"],
        "result_count": result["result_count"],
        "latency_ms": round(latency, 1),
        "suggestions": SUGGESTIONS,
    }


def _live_context() -> List[Dict[str, Any]]:
    """Current operational state, so 'right now' questions can be answered."""
    orch = get_orchestrator()
    crowd = orch.crowd.all_latest()
    return [
        {
            "kind": "live_crowd",
            "cameras": [
                {
                    "camera_id": cid,
                    "count": m.get("count"),
                    "density_band": m.get("density_band"),
                    "risk": m.get("risk"),
                }
                for cid, m in crowd.items()
            ],
        },
        {"kind": "open_incidents", "items": orch.open_incidents()},
    ]


@router.get("/chat/{session_id}")
def chat_history(
    session_id: str,
    user: User = Depends(require("incident:read")),
    db: Session = Depends(get_session),
):
    rows = db.scalars(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.at)
    ).all()
    return {"session_id": session_id, "messages": [to_dict(m) for m in rows]}


@router.get("/suggestions")
def suggestions(user: User = Depends(current_user)):
    return {"suggestions": SUGGESTIONS}


@router.post("/reindex")
def reindex(
    request: Request,
    user: User = Depends(require("incident:read")),
    db: Session = Depends(get_session),
):
    """Rebuild the vector index from the incident table."""
    rows = db.scalars(select(Incident)).all()
    payloads = []
    for i in rows:
        payloads.append(
            {
                "id": i.id,
                "title": i.title,
                "summary": i.summary,
                "explanation": i.explanation,
                "behaviors": (i.meta or {}).get("behaviors", []),
                "camera_id": i.camera_id,
                "camera_name": (i.meta or {}).get("camera_name"),
                "zone_id": i.zone_id,
                "site_id": i.site_id,
                "event_type": i.event_type,
                "severity": i.severity,
                "risk_score": i.risk_score,
                "started_at": i.started_at.isoformat(),
                "location": i.location or {},
                "track_ids": i.track_ids or [],
            }
        )
    written = get_forensic().index_many(payloads)
    record(db, request, user, "forensic.reindex", "incident", None, detail={"documents": written})
    return {"indexed": written, "store": get_store().info()}


@router.post("/find-similar-person")
async def find_similar_person(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = 10,
    user: User = Depends(require("track:read")),
    db: Session = Depends(get_session),
):
    """Appearance search: upload a crop, get the closest live subjects.

    This is a one-off similarity query against the *current* session gallery.
    For a standing search across the estate, enrol the person on the watchlist,
    which carries the approval and audit controls that a persistent search needs.
    """
    import cv2

    raw = await file.read()
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(413, "image exceeds the 12 MB limit")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "the uploaded file could not be decoded as an image")

    reid = get_orchestrator().reid
    embedding = reid.embedder.embed(image)
    if embedding is None:
        raise HTTPException(422, "no usable appearance features could be extracted")

    matches = reid.match_embedding(embedding, top_k=top_k)
    record(db, request, user, "forensic.appearance_search", "global_subject", None,
           detail={"results": len(matches)})

    return {
        "matches": matches,
        "embedding_backend": reid.embedder.info(),
        "note": "Appearance similarity across the current session's gallery. "
                "This is not an identification; for a standing search use the watchlist, "
                "which requires a legal basis and commander approval.",
    }

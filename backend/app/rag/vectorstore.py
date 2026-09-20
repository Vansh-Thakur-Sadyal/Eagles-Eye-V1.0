"""Vector store for incident retrieval (spec S23).

Two backends behind one interface:

  chroma  ChromaDB with sentence-transformers embeddings, when both are
          installed - the production path.
  memory  A dependency-free TF-IDF index with cosine similarity, which is a
          genuinely good retriever for the short, keyword-dense incident text
          Eagles Eye produces, and means forensic search works on a fresh clone
          with nothing downloaded.

Both support metadata filtering, because a security query is almost always
"this kind of event, on these cameras, in this time window" - semantic
similarity alone would be the wrong tool.
"""
from __future__ import annotations

import logging
import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..config import get_settings

log = logging.getLogger("sentinel.rag")

_TOKEN = re.compile(r"[a-z0-9_]+")
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "is", "was",
    "with", "by", "from", "this", "that", "it", "as", "be", "has", "have", "had",
    "show", "me", "find", "all", "any", "what", "where", "when", "who", "which",
}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in STOPWORDS and len(t) > 1]


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.doc_id,
            "text": self.text,
            "score": round(self.score, 4),
            "metadata": self.metadata,
        }


class BaseStore:
    backend = "base"

    def upsert(self, docs: Iterable[Document]) -> int:
        raise NotImplementedError

    def search(self, query: str, *, top_k: int = 8,
               where: Optional[Dict[str, Any]] = None) -> List[SearchHit]:
        raise NotImplementedError

    def delete(self, doc_id: str) -> bool:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def all_documents(self, limit: int = 500) -> List[SearchHit]:
        """Every document, unranked. Backs filter-only queries."""
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:
        return {"backend": self.backend, "documents": self.count()}


class MemoryStore(BaseStore):
    """TF-IDF over an inverted index. Exact, fast at this scale, zero deps."""

    backend = "memory-tfidf"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._docs: Dict[str, Document] = {}
        self._tf: Dict[str, Counter] = {}
        self._df: Counter = Counter()
        self._index: Dict[str, set] = defaultdict(set)

    def upsert(self, docs: Iterable[Document]) -> int:
        written = 0
        with self._lock:
            for doc in docs:
                if doc.doc_id in self._docs:
                    self._remove_unlocked(doc.doc_id)
                tokens = tokenize(doc.text)
                if not tokens:
                    continue
                tf = Counter(tokens)
                self._docs[doc.doc_id] = doc
                self._tf[doc.doc_id] = tf
                for term in tf:
                    self._df[term] += 1
                    self._index[term].add(doc.doc_id)
                written += 1
        return written

    def _remove_unlocked(self, doc_id: str) -> None:
        tf = self._tf.pop(doc_id, None)
        self._docs.pop(doc_id, None)
        if not tf:
            return
        for term in tf:
            self._df[term] -= 1
            self._index[term].discard(doc_id)
            if self._df[term] <= 0:
                self._df.pop(term, None)
                self._index.pop(term, None)

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._docs:
                return False
            self._remove_unlocked(doc_id)
            return True

    def search(self, query: str, *, top_k: int = 8,
               where: Optional[Dict[str, Any]] = None) -> List[SearchHit]:
        terms = tokenize(query)
        if not terms:
            return []
        with self._lock:
            n_docs = max(1, len(self._docs))
            candidates: set = set()
            for term in terms:
                candidates |= self._index.get(term, set())
            if not candidates:
                return []

            q_tf = Counter(terms)
            scores: Dict[str, float] = {}
            for doc_id in candidates:
                doc = self._docs[doc_id]
                if where and not _match_metadata(doc.metadata, where):
                    continue
                tf = self._tf[doc_id]
                length = math.sqrt(sum(v * v for v in tf.values())) or 1.0
                score = 0.0
                for term, q_count in q_tf.items():
                    if term not in tf:
                        continue
                    idf = math.log((n_docs + 1) / (self._df.get(term, 0) + 1)) + 1.0
                    score += (tf[term] / length) * q_count * idf * idf
                if score > 0:
                    scores[doc_id] = score

            ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
            best = ranked[0][1] if ranked else 1.0
            return [
                SearchHit(
                    doc_id=doc_id,
                    text=self._docs[doc_id].text,
                    score=score / best if best else 0.0,
                    metadata=self._docs[doc_id].metadata,
                )
                for doc_id, score in ranked
            ]

    def count(self) -> int:
        return len(self._docs)

    def all_documents(self, limit: int = 500) -> List[SearchHit]:
        with self._lock:
            return [
                SearchHit(doc_id=d.doc_id, text=d.text, score=0.0, metadata=d.metadata)
                for d in list(self._docs.values())[:limit]
            ]


class ChromaStore(BaseStore):
    backend = "chroma"

    def __init__(self) -> None:
        s = get_settings()
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._client = chromadb.PersistentClient(
            path=str(s.vector_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        embedding_function = None
        try:
            from chromadb.utils import embedding_functions

            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=s.embedding_model
            )
            self.backend = f"chroma+{s.embedding_model}"
        except Exception as exc:
            log.info("sentence-transformers unavailable (%s); Chroma default embeddings", exc)

        self._collection = self._client.get_or_create_collection(
            name="sentinel_incidents",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, docs: Iterable[Document]) -> int:
        docs = list(docs)
        if not docs:
            return 0
        self._collection.upsert(
            ids=[d.doc_id for d in docs],
            documents=[d.text for d in docs],
            metadatas=[_flatten(d.metadata) for d in docs],
        )
        return len(docs)

    def search(self, query: str, *, top_k: int = 8,
               where: Optional[Dict[str, Any]] = None) -> List[SearchHit]:
        result = self._collection.query(
            query_texts=[query],
            n_results=max(1, top_k),
            where=_chroma_where(where) if where else None,
        )
        hits: List[SearchHit] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for i, doc_id in enumerate(ids):
            distance = dists[i] if i < len(dists) else 1.0
            hits.append(
                SearchHit(
                    doc_id=doc_id,
                    text=docs[i] if i < len(docs) else "",
                    score=max(0.0, 1.0 - float(distance)),
                    metadata=metas[i] if i < len(metas) else {},
                )
            )
        return hits

    def delete(self, doc_id: str) -> bool:
        try:
            self._collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:
            return 0

    def all_documents(self, limit: int = 500) -> List[SearchHit]:
        try:
            result = self._collection.get(limit=limit)
        except Exception:
            return []
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        return [
            SearchHit(
                doc_id=doc_id,
                text=docs[i] if i < len(docs) else "",
                score=0.0,
                metadata=metas[i] if i < len(metas) else {},
            )
            for i, doc_id in enumerate(ids)
        ]


def _flatten(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata values must be scalars."""
    out: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = ",".join(str(x) for x in v)
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            out[k] = str(v)
    return out


def _chroma_where(where: Dict[str, Any]) -> Dict[str, Any]:
    clauses = []
    for key, value in where.items():
        if isinstance(value, dict):
            clauses.append({key: value})
        elif isinstance(value, (list, tuple, set)):
            clauses.append({key: {"$in": list(value)}})
        else:
            clauses.append({key: {"$eq": value}})
    if not clauses:
        return {}
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _match_metadata(meta: Dict[str, Any], where: Dict[str, Any]) -> bool:
    for key, expected in where.items():
        actual = meta.get(key)
        if isinstance(expected, dict):
            for op, value in expected.items():
                if op in ("$eq",) and actual != value:
                    return False
                if op == "$ne" and actual == value:
                    return False
                if op == "$in" and actual not in value:
                    return False
                if op == "$gte" and (actual is None or actual < value):
                    return False
                if op == "$lte" and (actual is None or actual > value):
                    return False
                if op == "$gt" and (actual is None or actual <= value):
                    return False
                if op == "$lt" and (actual is None or actual >= value):
                    return False
        elif isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


_store: Optional[BaseStore] = None
_store_lock = threading.Lock()


def get_store() -> BaseStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                s = get_settings()
                if s.vector_backend == "chroma":
                    try:
                        _store = ChromaStore()
                        log.info("vector store: %s", _store.backend)
                    except Exception as exc:
                        log.warning("Chroma unavailable (%s) - using in-memory TF-IDF index", exc)
                        _store = MemoryStore()
                else:
                    _store = MemoryStore()
    return _store


def incident_to_document(incident: Dict[str, Any]) -> Document:
    """Everything an operator might search an incident by, in one text blob."""
    parts = [
        incident.get("title", ""),
        incident.get("summary", ""),
        incident.get("explanation", ""),
        " ".join(incident.get("behaviors", []) or []),
        incident.get("camera_name", ""),
        (incident.get("location") or {}).get("zone_name", "") or "",
        incident.get("severity", ""),
        incident.get("event_type", ""),
    ]
    started = incident.get("started_at")
    return Document(
        doc_id=incident["id"],
        text="\n".join(p for p in parts if p),
        metadata={
            "incident_id": incident["id"],
            "camera_id": incident.get("camera_id"),
            "camera_name": incident.get("camera_name"),
            "zone_id": incident.get("zone_id"),
            "site_id": incident.get("site_id"),
            "event_type": incident.get("event_type"),
            "severity": incident.get("severity"),
            "risk_score": float(incident.get("risk_score", 0) or 0),
            "started_at": started,
            "started_ts": _to_epoch(started),
            "behaviors": incident.get("behaviors", []),
            "track_ids": incident.get("track_ids", []),
        },
    )


def _to_epoch(value: Any) -> float:
    if not value:
        return 0.0
    if isinstance(value, datetime):
        return value.timestamp()
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return 0.0

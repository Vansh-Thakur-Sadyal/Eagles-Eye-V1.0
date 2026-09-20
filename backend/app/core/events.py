"""In-process event bus + WebSocket fan-out.

Every agent publishes here; the dashboard subscribes over /ws/stream and the
automation layer subscribes to forward matching events to n8n.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Set

log = logging.getLogger("sentinel.events")


@dataclass
class SentinelEvent:
    topic: str
    payload: Dict[str, Any]
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "system"
    severity: str = "info"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventBus:
    """Thread-safe publisher with async subscribers.

    Agents run in worker threads, so ``publish`` is callable from any thread and
    hands the event to the API event loop via ``call_soon_threadsafe``.
    """

    def __init__(self, history: int = 500) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queues: Set[asyncio.Queue] = set()
        self._sync_handlers: List[Callable[[SentinelEvent], None]] = []
        self._history: Deque[SentinelEvent] = deque(maxlen=history)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ------------------------------------------------------------- publish
    def publish(self, topic: str, payload: Dict[str, Any], *, source: str = "system",
                severity: str = "info") -> SentinelEvent:
        evt = SentinelEvent(topic=topic, payload=payload, source=source, severity=severity)
        self._history.append(evt)
        for handler in list(self._sync_handlers):
            try:
                handler(evt)
            except Exception:
                log.exception("sync event handler failed for %s", topic)
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._dispatch, evt)
            except RuntimeError:
                pass
        return evt

    def _dispatch(self, evt: SentinelEvent) -> None:
        for q in list(self._queues):
            try:
                q.put_nowait(evt)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(evt)
                except Exception:
                    pass

    # ----------------------------------------------------------- subscribe
    def subscribe(self, maxsize: int = 256) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._queues.discard(q)

    def on(self, handler: Callable[[SentinelEvent], None]) -> None:
        """Register a synchronous handler (runs on the publishing thread)."""
        self._sync_handlers.append(handler)

    def recent(self, topic_prefix: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        items = [e for e in self._history if e.topic.startswith(topic_prefix)]
        return [e.to_dict() for e in items[-limit:]]

    @property
    def subscriber_count(self) -> int:
        return len(self._queues)


bus = EventBus()


class WebSocketHub:
    """Tracks live dashboard sockets so we can report connection counts."""

    def __init__(self) -> None:
        self._sockets: Set[Any] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws: Any) -> None:
        async with self._lock:
            self._sockets.add(ws)

    async def remove(self, ws: Any) -> None:
        async with self._lock:
            self._sockets.discard(ws)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        data = json.dumps(message, default=str)
        dead = []
        for ws in list(self._sockets):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.remove(ws)

    @property
    def count(self) -> int:
        return len(self._sockets)


hub = WebSocketHub()

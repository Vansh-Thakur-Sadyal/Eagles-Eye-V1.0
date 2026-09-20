"""WebSocket live stream for the dashboard.

One socket carries every real-time topic: findings, incidents, camera status,
crowd telemetry, GPU changes and automation results.  Clients can subscribe to
a subset so a single-purpose screen is not woken by unrelated traffic.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from ...agents.orchestrator import get_orchestrator
from ...core.device import get_device_manager
from ...core.events import bus, hub
from ...core.security import decode_token
from ...pipeline.runner import get_pipeline

log = logging.getLogger("sentinel.ws")
router = APIRouter(tags=["stream"])

HEARTBEAT_SECONDS = 15.0


@router.websocket("/ws/stream")
async def stream(websocket: WebSocket, token: Optional[str] = Query(None),
                 topics: Optional[str] = Query(None)):
    """Live event feed.

    Auth is by ``?token=`` because browsers cannot set headers on a WebSocket.
    The socket is accepted first so an auth failure can be reported with a
    reason the UI can display, rather than a bare handshake refusal.
    """
    await websocket.accept()

    identity: Dict[str, Any] = {}
    if token:
        try:
            identity = decode_token(token)
        except Exception:
            await websocket.send_json(
                {"topic": "error", "payload": {"message": "invalid or expired token"}}
            )
            await websocket.close(code=4401)
            return
    else:
        await websocket.send_json(
            {"topic": "error", "payload": {"message": "a token query parameter is required"}}
        )
        await websocket.close(code=4401)
        return

    wanted: Set[str] = {t.strip() for t in topics.split(",")} if topics else set()
    queue = bus.subscribe(maxsize=512)
    await hub.add(websocket)

    try:
        await websocket.send_json(
            {
                "topic": "connected",
                "payload": {
                    "user": identity.get("username"),
                    "role": identity.get("role"),
                    "topics": sorted(wanted) or "all",
                    "server_time": datetime.now(timezone.utc).isoformat(),
                    "device": get_device_manager().active_device,
                },
            }
        )

        receiver = asyncio.create_task(_receive(websocket, wanted))
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    await websocket.send_json(
                        {
                            "topic": "heartbeat",
                            "payload": {
                                "at": datetime.now(timezone.utc).isoformat(),
                                "device": get_device_manager().active_device,
                                "cameras_running": len(get_pipeline().running_ids()),
                            },
                        }
                    )
                    continue

                if wanted and not any(event.topic.startswith(t) for t in wanted):
                    continue
                await websocket.send_text(json.dumps(event.to_dict(), default=str))
        finally:
            receiver.cancel()

    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("websocket stream failed")
    finally:
        bus.unsubscribe(queue)
        await hub.remove(websocket)


async def _receive(websocket: WebSocket, wanted: Set[str]) -> None:
    """Handle client control messages: subscribe, unsubscribe, ping, snapshot."""
    while True:
        try:
            raw = await websocket.receive_text()
        except Exception:
            return
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            continue

        action = message.get("action")
        if action == "subscribe":
            wanted.update(message.get("topics", []))
        elif action == "unsubscribe":
            for topic in message.get("topics", []):
                wanted.discard(topic)
        elif action == "ping":
            await websocket.send_json({"topic": "pong", "payload": {"at": datetime.now(timezone.utc).isoformat()}})
        elif action == "snapshot":
            orch = get_orchestrator()
            await websocket.send_json(
                {
                    "topic": "snapshot",
                    "payload": {
                        "pipeline": get_pipeline().status(),
                        "orchestrator": orch.status(),
                        "open_incidents": orch.open_incidents(),
                        "crowd": {
                            cid: {k: v for k, v in m.items() if k != "heatmap"}
                            for cid, m in orch.crowd.all_latest().items()
                        },
                        "device": get_device_manager().status(),
                    },
                }
            )

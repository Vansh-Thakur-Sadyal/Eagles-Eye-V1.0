"""Eagles Eye - FastAPI application entry point."""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .agents.orchestrator import get_orchestrator
from .api.routes import (
    admin,
    auth,
    cameras,
    edge,
    forensic,
    incidents,
    intelligence,
    portal,
    portal_admin,
    spatial,
    stream,
    system,
    watchlist,
)
from .automation.n8n import dispatcher, wire_automation
from .config import get_settings
from .core.device import get_device_manager
from .core.events import bus
from .core.persistence import wire_persistence
from .db import init_db, session_scope
from .pipeline.runner import get_pipeline

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("sentinel")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Eagles Eye starting (env=%s)", settings.env)
    bus.bind_loop(asyncio.get_running_loop())

    init_db()
    from .seed import seed_if_empty

    seed_if_empty()

    dm = get_device_manager()
    log.info("compute: %s (cuda_available=%s)", dm.active_device, dm.cuda_available)
    for gpu in dm.status()["gpus"]:
        log.info("  GPU %s: %s (%s MB)", gpu["index"], gpu["name"], gpu["total_memory_mb"])
    if not dm.cuda_available and dm.status()["gpus"]:
        log.warning(
            "A GPU is present but CUDA is not available to this process. "
            "Install a CUDA-enabled torch build - see docs/INSTALL.md."
        )

    wire_persistence()
    wire_automation()

    # Warm the watchlist gallery and the forensic index from stored data.
    try:
        from .api.routes.watchlist import reload_gallery

        with session_scope() as db:
            count = reload_gallery(db)
        log.info("watchlist gallery: %d subject(s)", count)
    except Exception:
        log.exception("could not load the watchlist gallery")

    try:
        from sqlalchemy import select

        from .models import Incident
        from .rag.forensic import get_forensic

        with session_scope() as db:
            rows = db.scalars(select(Incident).order_by(Incident.started_at.desc()).limit(2000)).all()
            payloads = [
                {
                    "id": i.id, "title": i.title, "summary": i.summary,
                    "explanation": i.explanation, "camera_id": i.camera_id,
                    "camera_name": (i.meta or {}).get("camera_name"),
                    "zone_id": i.zone_id, "site_id": i.site_id,
                    "event_type": i.event_type, "severity": i.severity,
                    "risk_score": i.risk_score, "started_at": i.started_at.isoformat(),
                    "behaviors": (i.meta or {}).get("behaviors", []),
                    "location": i.location or {}, "track_ids": i.track_ids or [],
                }
                for i in rows
            ]
        if payloads:
            indexed = get_forensic().index_many(payloads)
            log.info("forensic index warmed with %d incident(s)", indexed)
    except Exception:
        log.exception("could not warm the forensic index")

    log.info("Eagles Eye ready on http://%s:%s", settings.host, settings.port)
    try:
        yield
    finally:
        log.info("Eagles Eye shutting down")
        get_pipeline().stop_all()
        dispatcher.stop()


app = FastAPI(
    title="Eagles Eye",
    description=(
        "Agentic AI + Computer Vision + Digital Twin + AR/VR public safety intelligence "
        "platform. Every association is confidence-scored; the system recommends, a human decides."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred.",
            "path": request.url.path,
            "type": type(exc).__name__,
        },
    )


for module in (auth, system, cameras, incidents, intelligence, watchlist, forensic,
               spatial, admin, stream, edge, portal, portal_admin):
    app.include_router(module.router)
app.include_router(incidents.alerts_router)

# Serve stored media (snapshots, evidence, watchlist references) for the UI.
app.mount("/media", StaticFiles(directory=str(settings.storage_dir)), name="media")

@app.get("/api", tags=["system"])
def api_root():
    return {
        "service": "Eagles Eye",
        "version": "1.0.0",
        "docs": "/api/docs",
        "websocket": "/ws/stream?token=<jwt>",
        "routes": {
            "auth": "/api/auth",
            "system": "/api/system",
            "cameras": "/api/cameras",
            "incidents": "/api/incidents",
            "alerts": "/api/alerts",
            "intelligence": "/api/intel",
            "watchlist": "/api/watchlist",
            "forensic": "/api/forensic",
            "spatial": "/api/spatial",
            "admin": "/api/admin",
        },
    }


# The public portal (sign-up, sign-in, terms) and the dashboard are served by
# this one process: / is the portal, /app is the dashboard.
_portal = Path(__file__).resolve().parent.parent.parent / "portal"
if _portal.is_dir():
    app.mount("/portal", StaticFiles(directory=str(_portal), html=True), name="portal")
    log.info("serving portal from %s", _portal)

# Serve the built frontend when it exists, so one process can host everything.
_frontend = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _frontend.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_frontend / "assets")), name="assets")
    log.info("serving frontend from %s", _frontend)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        """Portal at the root, dashboard under /app, SPA deep links preserved.

        A mounted StaticFiles only resolves real files, so a deep link such as
        /app/incidents/INC_123 would 404 on a hard refresh. Anything that is not
        an API route and not a real file is handed to index.html for the client
        router to resolve.
        """
        if full_path.startswith(("api/", "ws/", "media/")):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        # Visitors land on the portal; the dashboard lives behind sign-in.
        if _portal.is_dir() and full_path in ("", "index.html"):
            return FileResponse(_portal / "index.html")
        inner = full_path[4:] if full_path.startswith("app/") else full_path
        candidate = (_frontend / inner).resolve()
        if inner and candidate.is_file() and _frontend.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(_frontend / "index.html")


def run() -> None:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.env == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()

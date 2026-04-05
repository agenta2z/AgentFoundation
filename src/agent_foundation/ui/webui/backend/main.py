# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
FastAPI Backend for CoScience Chatbot Demo.

Supports two modes:
- "demo" (default): Plays back pre-authored JSON flow scripts via ExperimentService
- "real": Full agent capabilities with LLM chat and /task dual-agent execution

The mode is set via app.state.mode before lifespan runs (from run_webui.py).
Agent-related imports are deferred to avoid loading heavy deps in demo mode.

Usage:
    # Development (with hot reload)
    uvicorn backend.main:app --reload --port 8000

    # Production (with SSL + IPv6 for devserver)
    uvicorn backend.main:app --host "::" --port 8087 \
        --ssl-keyfile /etc/pki/tls/certs/${HOSTNAME}.key \
        --ssl-certfile /etc/pki/tls/certs/${HOSTNAME}.crt
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add parent directory to path for importing from chatbot_demo
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "chatbot_demo"),
)

from chatbot_demo_react.backend.routes.chat_routes import router as chat_router
from chatbot_demo_react.backend.routes.config_routes import router as config_router
from chatbot_demo_react.backend.routes.experiment_routes import (
    router as experiment_router,
)
from chatbot_demo_react.backend.routes.websocket_routes import (
    router as websocket_router,
)
from chatbot_demo_react.backend.services.experiment_service import ExperimentService

logger = logging.getLogger(__name__)

# Global experiment service instance (initialized in lifespan)
experiment_service: ExperimentService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan - initialize and cleanup resources."""
    global experiment_service

    mode = getattr(app.state, "mode", "demo")
    logger.info("Starting in %s mode", mode)

    if mode == "real":
        from chatbot_demo_react.backend.services.session_store_service import (
            SessionStoreService,
        )

        # Store queue root path — the bridge creates per-connection instances
        queue_root_path = getattr(app.state, "queue_root_path", None)
        if queue_root_path is None:
            logger.warning(
                "Real mode: no queue_root_path configured. "
                "Pass --queue-root to specify the running server's queue directory."
            )

        # Derive server_dir from queue_root (queue_root = server_dir/queues)
        if queue_root_path:
            server_dir = str(Path(queue_root_path).parent)
            app.state.server_dir = server_dir
            app.state.session_store = SessionStoreService(server_dir)
            logger.info("Session store initialized: server_dir=%s", server_dir)

        logger.info("Real mode initialized: queue_root=%s", queue_root_path)
    else:
        # Demo mode — initialize ExperimentService
        startup_iteration = getattr(app.state, "startup_iteration", None)
        startup_step = getattr(app.state, "startup_step", 0)
        startup_speedup = getattr(app.state, "startup_speedup", 1.0)

        delay_multiplier = 1.0 / startup_speedup if startup_speedup > 0 else 1.0

        experiment_service = ExperimentService(
            flow_name="coscience_experiment_public",
            delay_multiplier=delay_multiplier,
        )
        logger.info(
            "Experiment service initialized with flow: coscience_experiment_public"
        )
        if startup_speedup > 1.0:
            logger.info(
                "Speedup: %sx (delay_multiplier: %.3f)",
                startup_speedup,
                delay_multiplier,
            )

        if startup_iteration:
            success = experiment_service.jump_to_iteration(
                startup_iteration, startup_step
            )
            if success:
                logger.info(
                    "Started at iteration '%s' with step offset %s",
                    startup_iteration,
                    startup_step,
                )
            else:
                logger.warning(
                    "Failed to jump to iteration '%s' - starting at step 0",
                    startup_iteration,
                )
        elif startup_step > 0:
            success = experiment_service.jump_to_step(startup_step)
            if success:
                logger.info("Started at step index %s", startup_step)
            else:
                logger.warning(
                    "Failed to jump to step %s - starting at step 0", startup_step
                )

    yield

    # Cleanup
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="CoScience Chatbot Demo",
    description="React + FastAPI chatbot with real-time progress animation",
    version="1.0.0",
    lifespan=lifespan,
)


# Exception handler for unhandled errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions gracefully."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# Add CORS middleware (allows React dev server to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://127.0.0.1:3000",
        "https://localhost:3000",
        "https://127.0.0.1:3000",
        "*",  # Allow all origins for devserver access (IPv6 hosts)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API routers — demo routes (always available)
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(experiment_router, prefix="/api/experiment", tags=["experiment"])
app.include_router(websocket_router, prefix="/ws", tags=["websocket"])
# Config route — shared between both modes, no agent deps
app.include_router(config_router, prefix="/api", tags=["config"])

# Real-mode routers — registered at module level so FastAPI builds route table
# correctly. The routes themselves are lightweight (no heavy deps); they access
# services from app.state at request time, which is initialized in lifespan().
# In demo mode these routes return 503 ("Session store not available").
from chatbot_demo_react.backend.routes.agent_routes import (
    router as agent_router,
)
from chatbot_demo_react.backend.routes.agent_websocket_routes import (
    router as agent_ws_router,
)
from chatbot_demo_react.backend.routes.session_store_routes import (
    router as session_store_router,
)
from chatbot_demo_react.backend.routes.workspace_routes import (
    router as workspace_router,
)

app.include_router(agent_ws_router, prefix="/ws", tags=["agent_websocket"])
app.include_router(agent_router, prefix="/api/agent", tags=["agent"])
app.include_router(workspace_router, prefix="/api/workspace", tags=["workspace"])
app.include_router(
    session_store_router, prefix="/api/sessions", tags=["session_store"]
)


# Health check endpoint
@app.get("/api/health")
async def health_check() -> dict:
    """Health check endpoint."""
    mode = getattr(app.state, "mode", "demo")
    return {"status": "ok", "service": "coscience-chatbot", "mode": mode}


# Serve React frontend (production mode)
# In production, React is built and copied to ../frontend/
frontend_path = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/{full_path:path}")
async def serve_react_app(full_path: str) -> FileResponse:
    """Serve the React app (SPA fallback to index.html)."""
    # Don't serve API routes here
    if full_path.startswith("api") or full_path.startswith("ws"):
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    # Try to serve the requested file
    file_path = frontend_path / full_path
    if file_path.is_file():
        return FileResponse(str(file_path))

    # SPA fallback: serve index.html for client-side routing
    index_path = frontend_path / "index.html"
    if index_path.is_file():
        return FileResponse(str(index_path))

    # If frontend not built, return helpful message
    return JSONResponse(
        status_code=503,
        content={
            "error": "Frontend not built",
            "message": "Run 'cd react && yarn build && yarn copy-build' to build the frontend",
        },
    )


# Mount static files (CSS, JS, images from React build)
if (frontend_path / "static").exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(frontend_path / "static")),
        name="static",
    )


def get_experiment_service() -> ExperimentService:
    """Get the global experiment service instance."""
    if experiment_service is None:
        mode = getattr(app.state, "mode", "demo")
        if mode == "real":
            raise RuntimeError(
                "Experiment service is not available in real agent mode. "
                "Demo endpoints are disabled."
            )
        raise RuntimeError("Experiment service not initialized")
    return experiment_service


# Make experiment_service accessible to routes
app.state.get_experiment_service = get_experiment_service


if __name__ == "__main__":
    import uvicorn

    # Default to development mode
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )

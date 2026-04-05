# Copyright (c) Meta Platforms, Inc. and affiliates.
"""API Routes for the chatbot backend."""

from chatbot_demo_react.backend.routes.chat_routes import router as chat_router
from chatbot_demo_react.backend.routes.experiment_routes import (
    router as experiment_router,
)
from chatbot_demo_react.backend.routes.websocket_routes import (
    router as websocket_router,
)

__all__ = ["chat_router", "experiment_router", "websocket_router"]

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Shared config endpoint — included in both demo and real modes.

Returns the current mode and (in real mode) the active model/provider/target-path.
Also provides welcome message read/write endpoints for the session landing page.
No dependencies on chat_cli or agent code.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/config")
async def get_config(request: Request) -> dict[str, Any]:
    """Return current server configuration.

    In demo mode: {"mode": "demo"}
    In real mode: {"mode": "real", "model": "...", "target_path": "...", "provider": "..."}
    """
    mode = getattr(request.app.state, "mode", "demo")
    result: dict[str, Any] = {"mode": mode}

    if mode == "real":
        get_service_info = getattr(request.app.state, "get_agent_config_info", None)
        if get_service_info is not None:
            result.update(get_service_info())

    return result


# ── Welcome Message ────────────────────────────────────────────────


def _get_runtime_dir(request: Request) -> Path | None:
    """Derive the stable _runtime dir from server_dir.

    server_dir = .../_runtime/servers/server_XXX
    _runtime   = .../_runtime
    """
    server_dir = getattr(request.app.state, "server_dir", None)
    if server_dir:
        return Path(server_dir).parent.parent
    return None


def _read_welcome_message(request: Request) -> tuple[str, bool]:
    """Read welcome message. Returns (content, is_custom).

    Checks runtime custom override first, then falls back to the default
    template that the server writes to _runtime/welcome_message_default.md
    on startup.
    """
    runtime_dir = _get_runtime_dir(request)
    if runtime_dir:
        # Custom override takes priority
        override_path = runtime_dir / "welcome_message.md"
        if override_path.is_file():
            try:
                return override_path.read_text(encoding="utf-8"), True
            except Exception as e:
                logger.warning("Failed to read welcome message override: %s", e)

        # Default template (written by the server on startup)
        default_path = runtime_dir / "welcome_message_default.md"
        if default_path.is_file():
            try:
                return default_path.read_text(encoding="utf-8"), False
            except Exception as e:
                logger.warning("Failed to read default welcome message: %s", e)

    return "", False


def _write_welcome_message(request: Request, content: str) -> bool:
    """Write welcome message to runtime dir. Returns success."""
    runtime_dir = _get_runtime_dir(request)
    if not runtime_dir:
        return False
    try:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        override_path = runtime_dir / "welcome_message.md"
        override_path.write_text(content, encoding="utf-8")
        return True
    except Exception as e:
        logger.error("Failed to write welcome message: %s", e)
        return False


@router.get("/welcome-message")
async def get_welcome_message(request: Request) -> dict[str, Any]:
    """Return the current welcome message (runtime override or default template)."""
    content, is_custom = await asyncio.to_thread(_read_welcome_message, request)
    return {"content": content, "is_custom": is_custom}


class WelcomeMessageUpdate(BaseModel):
    content: str


@router.put("/welcome-message")
async def update_welcome_message(
    request: Request, body: WelcomeMessageUpdate
) -> dict[str, Any]:
    """Save a custom welcome message to the runtime directory."""
    success = await asyncio.to_thread(_write_welcome_message, request, body.content)
    if not success:
        return {"success": False, "error": "Server runtime directory not available"}
    return {"success": True}


@router.delete("/welcome-message")
async def reset_welcome_message(request: Request) -> dict[str, Any]:
    """Reset welcome message to the default template by removing the runtime override."""
    runtime_dir = _get_runtime_dir(request)
    if runtime_dir:
        override_path = runtime_dir / "welcome_message.md"
        if override_path.is_file():
            try:
                override_path.unlink()
            except Exception as e:
                logger.error("Failed to delete welcome message override: %s", e)
                return {"success": False, "error": str(e)}
    # Read the default back (server wrote it on startup)
    content, _ = _read_welcome_message(request)
    return {"success": True, "content": content}

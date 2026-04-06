

"""REST endpoints for reading session data from the server's file store.

These endpoints are read-only — the server is the sole writer.
The WebUI backend reads session_state.json files and serves them to
the React frontend via HTTP, avoiding the queue for durable state.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_store(request: Request):
    """Get SessionStoreService from app state."""
    store = getattr(request.app.state, "session_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Session store not available. Server may not be running.",
        )
    return store


@router.get("")
async def list_sessions(request: Request) -> list:
    """List all active sessions with metadata."""
    store = _get_store(request)
    return await asyncio.to_thread(store.list_sessions)


@router.get("/{session_id}/state")
async def get_session_state(session_id: str, request: Request) -> dict:
    """Get full session state from file store."""
    store = _get_store(request)
    state = await asyncio.to_thread(store.get_session_state, session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return state


@router.get("/{session_id}/messages")
async def get_session_messages(session_id: str, request: Request) -> dict:
    """Get conversation messages for a session."""
    store = _get_store(request)
    messages = await asyncio.to_thread(store.get_session_messages, session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"session_id": session_id, "messages": messages}


@router.get("/{session_id}/config")
async def get_session_config(session_id: str, request: Request) -> dict:
    """Get session config (model, target_path, provider)."""
    store = _get_store(request)
    config = await asyncio.to_thread(store.get_session_config, session_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"session_id": session_id, "config": config}


@router.get("/{session_id}/turns/{turn_number}")
async def get_turn_data(session_id: str, turn_number: int, request: Request) -> dict:
    """Get per-turn data (prompt, response, metadata) from parts files."""
    store = _get_store(request)
    data = await asyncio.to_thread(store.get_turn_data, session_id, turn_number)
    if data is None:
        raise HTTPException(status_code=404, detail="Turn not found")
    return data

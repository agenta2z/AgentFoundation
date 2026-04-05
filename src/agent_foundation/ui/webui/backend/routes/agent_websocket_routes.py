# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
WebSocket endpoint for bidirectional agent streaming.

Provides the /agent WebSocket route (mounted at /ws/agent via main.py).
Each connection creates an AgentServiceBridge to communicate with the
separately-running RankEvolve server via file-based queues.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logger: logging.Logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/agent")
async def agent_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time agent interaction.

    Protocol (client -> server):
        {"type": "message", "content": "user text"}
        {"type": "cancel"}
        {"type": "ping"}

    Protocol (server -> client):
        {"type": "token", "content": "chunk", "metadata": {...}}
        {"type": "message_end", "final_content": "..."}
        {"type": "command_response", "content": "..."}
        {"type": "config_update", "config": {...}}
        {"type": "task_status", "phase": "...", "state": "..."}
        {"type": "error", "message": "..."}
        {"type": "heartbeat"}
        {"type": "pong"}
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    logger.info("Agent WebSocket connected (session=%s)", session_id)

    # Check for client-provided session_id in first message (for resume/multi-session)
    # We peek at the first message; if it's an init message with session_id, use it.
    # Otherwise treat it as a regular message and process it below.
    init_data = None
    try:
        first_msg = await asyncio.wait_for(websocket.receive_json(), timeout=2.0)
        if first_msg.get("type") == "init" and first_msg.get("session_id"):
            session_id = first_msg["session_id"]
            logger.info("Client provided session_id=%s", session_id)
        else:
            init_data = first_msg  # Not an init message; process as normal below
    except asyncio.TimeoutError:
        pass  # No init message sent; use generated session_id

    # Create per-connection bridge to the running server
    bridge = _create_bridge(websocket.app, session_id)
    if bridge is None:
        await websocket.send_json(
            {
                "type": "error",
                "message": "Agent service not available. Server not running or queue root not configured.",
            }
        )
        await websocket.close()
        return

    # Build session_init — handshake confirmation with config + session list.
    # Reads from file store first (reliable), falls back to app.state CLI args.
    server_dir = getattr(websocket.app.state, "server_dir", None)
    init_msg: dict[str, Any] = {
        "type": "session_init",
        "session_id": session_id,
        "server_id": Path(server_dir).name if server_dir else None,
    }

    store = getattr(websocket.app.state, "session_store", None)
    if store:
        # Read session list from file store (always include, even if empty,
        # so the client can sync its state on reconnect)
        sessions_list = await asyncio.to_thread(store.list_sessions)
        init_msg["sessions"] = sessions_list or []

        # Read config for this session from file store
        session_config = await asyncio.to_thread(store.get_session_config, session_id)
        if session_config:
            init_msg["config"] = session_config
            logger.info(
                "Session init for %s with config from store: %s",
                session_id,
                session_config,
            )

    # Fallback to app.state CLI args if store has nothing
    if "config" not in init_msg:
        direct_config = {
            "model": getattr(websocket.app.state, "agent_model", "") or "",
            "target_path": getattr(websocket.app.state, "agent_target_path", "") or "",
            "provider": getattr(websocket.app.state, "agent_provider", "") or "",
        }
        if any(direct_config.values()):
            init_msg["config"] = direct_config
            logger.info(
                "Session init for %s with config from CLI args: %s",
                session_id,
                direct_config,
            )

    await websocket.send_json(init_msg)

    # Active message task for cancellation
    active_task: asyncio.Task[Any] | None = None
    heartbeat_task: asyncio.Task[Any] | None = None

    async def send_callback(msg: dict[str, Any]) -> None:
        """Send a JSON message to the client, ignoring if disconnected."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(msg)
        except Exception:
            pass

    async def heartbeat_loop() -> None:
        """Send heartbeat every 30s if connection is alive."""
        try:
            while True:
                await asyncio.sleep(30)
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"type": "heartbeat"})
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def server_monitor_loop() -> None:
        """Monitor server health and attempt reconnection when server goes down."""
        nonlocal bridge
        try:
            while True:
                await asyncio.sleep(10)
                if not bridge.is_server_alive():
                    logger.warning(
                        "Server appears down (session=%s), notifying client", session_id
                    )
                    await send_callback(
                        {"type": "server_status", "status": "server_down"}
                    )

                    # Enter reconnect loop
                    while True:
                        await asyncio.sleep(10)
                        try:
                            new_bridge = _create_bridge(websocket.app, session_id)
                            if new_bridge is None:
                                continue
                            # Verify server is alive with queue probe (end-to-end check).
                            # File store alone is unreliable — stale session_state.json
                            # from a crashed server still shows status: "active".
                            new_bridge.send_message("config_query")
                            probe_resp = await new_bridge.poll_one_response()
                            if probe_resp and probe_resp.get("type") == "config_update":
                                await send_callback(
                                    {"type": "server_status", "status": "syncing"}
                                )
                                # Enrich config from file store if available
                                reconnect_store = getattr(websocket.app.state, "session_store", None)
                                if reconnect_store:
                                    config = await asyncio.to_thread(
                                        reconnect_store.get_session_config, session_id
                                    )
                                    if config:
                                        await send_callback(
                                            {"type": "config_update", "config": config}
                                        )
                                elif probe_resp.get("config"):
                                    await send_callback(
                                        {"type": "config_update", "config": probe_resp["config"]}
                                    )
                                bridge = new_bridge
                                await send_callback(
                                    {"type": "server_status", "status": "connected"}
                                )
                                logger.info(
                                    "Reconnected to server (session=%s)", session_id
                                )
                                break
                            else:
                                new_bridge.close()
                        except Exception as e:
                            logger.debug(
                                "Reconnect probe failed (session=%s): %s", session_id, e
                            )
                            continue
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def _handle_data(data: dict[str, Any]) -> None:
        """Process a single client message (extracted for reuse with init_data)."""
        nonlocal active_task, bridge, session_id
        msg_type = data.get("type", "")

        if msg_type != "ping":
            logger.info(
                "WS _handle_data: type=%s session=%s keys=%s",
                msg_type, session_id, list(data.keys()),
            )

        if msg_type == "ping":
            await websocket.send_json({"type": "pong"})

        elif msg_type == "cancel":
            if active_task and not active_task.done():
                active_task.cancel()
                bridge.cancel_active_task()
                await send_callback(
                    {
                        "type": "status",
                        "status": "complete",
                        "detail": "Cancelled by user",
                    }
                )

        elif msg_type == "init":
            pass  # Already handled above; ignore duplicate inits

        elif msg_type == "switch_session":
            new_session_id = data.get("session_id")
            if new_session_id and new_session_id != session_id:
                logger.info("Switching session %s → %s", session_id, new_session_id)
                # Create new bridge before closing old one (fail-safe)
                new_bridge = _create_bridge(websocket.app, new_session_id)
                if new_bridge is None:
                    await send_callback(
                        {
                            "type": "error",
                            "message": "Failed to switch session — bridge creation failed.",
                        }
                    )
                    return

                # Cancel any active task on the old session
                if active_task and not active_task.done():
                    active_task.cancel()
                    active_task = None

                # Close old bridge (deregisters from old session queues)
                bridge.close()

                # Switch to new session
                session_id = new_session_id
                bridge = new_bridge

                # Notify client of the new session (with config from file store)
                switch_init: dict[str, Any] = {
                    "type": "session_init",
                    "session_id": session_id,
                    "server_id": Path(server_dir).name if server_dir else None,
                }
                switch_store = getattr(websocket.app.state, "session_store", None)
                if switch_store:
                    switch_config = await asyncio.to_thread(
                        switch_store.get_session_config, session_id
                    )
                    if switch_config:
                        switch_init["config"] = switch_config
                if "config" not in switch_init:
                    # Fallback to CLI args
                    fallback_config = {
                        "model": getattr(websocket.app.state, "agent_model", "") or "",
                        "target_path": getattr(websocket.app.state, "agent_target_path", "") or "",
                        "provider": getattr(websocket.app.state, "agent_provider", "") or "",
                    }
                    if any(fallback_config.values()):
                        switch_init["config"] = fallback_config
                await send_callback(switch_init)

                logger.info("Switched to session %s", session_id)

        elif msg_type == "queue_status":
            bridge.send_queue_status_request()
            for _ in range(20):  # poll up to 2 seconds
                resp = await bridge.poll_one_response()
                if resp and resp.get("type") == "queue_status_response":
                    await send_callback(resp)
                    break
                await asyncio.sleep(0.1)

        elif msg_type == "pending_input_response":
            # Forward pending input response (compound widget submit) to
            # the server via the file-based queue. The server's
            # _handle_pending_input_response will push it into the
            # interactive's input queue, unblocking aget_input().
            # Pass through all fields except "type" (added by send_message).
            logger.info(
                "Received pending_input_response (session=%s), keys=%s",
                session_id, list(data.keys()),
            )
            forward = {
                k: v for k, v in data.items() if k not in ("type",)
            }
            bridge.send_message("pending_input_response", **forward)
            logger.info(
                "Forwarded pending_input_response to queue (session=%s)",
                session_id,
            )

            # Always send message_start so the frontend enters streaming
            # state for the next response. Without this, the frontend stays
            # in "complete" state and ignores subsequent tokens.
            await send_callback({"type": "message_start"})

            # Check if previous poll_responses is still running
            logger.info(
                "pending_input_response: active_task=%s, done=%s (session=%s)",
                active_task, active_task.done() if active_task else 'N/A', session_id,
            )
            # Restart response polling. The first poll_responses() exited
            # when stream_end was sent after the initial LLM response.
            # Now the inferencer will resume and send new tokens — we need
            # a new poll_responses loop to pick them up and forward to the
            # frontend via WebSocket.
            if active_task is None or active_task.done():
                async def resume_after_input() -> None:
                    try:
                        await bridge.poll_responses(send_callback)
                    except asyncio.CancelledError:
                        logger.info(
                            "Resume polling cancelled (session=%s)", session_id
                        )
                    except Exception as e:
                        logger.error(
                            "Error in resume polling (session=%s): %s",
                            session_id, e, exc_info=True,
                        )
                        await send_callback({"type": "error", "message": str(e)})

                active_task = asyncio.create_task(resume_after_input())

        elif msg_type == "message":
            content = data.get("content", "").strip()
            if not content:
                return

            # Cancel previous message if still running
            if active_task and not active_task.done():
                active_task.cancel()
                bridge.cancel_active_task()

            async def process_message(text: str) -> None:
                try:
                    await send_callback({"type": "message_start"})
                    await bridge.handle_message(text, send_callback)
                except asyncio.CancelledError:
                    logger.info("Message processing cancelled (session=%s)", session_id)
                except Exception as e:
                    logger.error(
                        "Error processing message (session=%s): %s",
                        session_id,
                        e,
                        exc_info=True,
                    )
                    await send_callback(
                        {
                            "type": "error",
                            "message": str(e),
                        }
                    )

            active_task = asyncio.create_task(process_message(content))

    monitor_task: asyncio.Task[Any] | None = None
    try:
        heartbeat_task = asyncio.create_task(heartbeat_loop())
        monitor_task = asyncio.create_task(server_monitor_loop())

        # Process any first message that wasn't an init
        if init_data is not None:
            await _handle_data(init_data)

        while True:
            data = await websocket.receive_json()
            await _handle_data(data)

    except WebSocketDisconnect:
        logger.info("Agent WebSocket disconnected (session=%s)", session_id)
    except Exception as e:
        logger.error(
            "Agent WebSocket error (session=%s): %s",
            session_id,
            e,
            exc_info=True,
        )
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
        if monitor_task:
            monitor_task.cancel()
        if active_task and not active_task.done():
            active_task.cancel()
        bridge.close()
        logger.info("Agent WebSocket cleanup complete (session=%s)", session_id)


def _create_bridge(app: Any, session_id: str) -> Any:
    """Create an AgentServiceBridge instance from app.state configuration."""
    queue_root_path = getattr(app.state, "queue_root_path", None)
    if queue_root_path is None:
        return None

    from chatbot_demo_react.backend.services.agent_service_bridge import (
        AgentServiceBridge,
    )

    return AgentServiceBridge(
        queue_root_path=queue_root_path,
        session_id=session_id,
    )

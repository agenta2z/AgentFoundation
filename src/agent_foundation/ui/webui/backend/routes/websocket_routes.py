# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
"""
WebSocket Routes for real-time progress animation updates.

This is the KEY IMPROVEMENT over Dash:
- WebSocket pushes updates every 200ms directly to frontend
- No server callbacks, no re-renders, no competing state
- React updates DOM instantly on message receive

Endpoints:
- WS /ws/progress - Stream progress animation state updates
"""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# Store active WebSocket connections
active_connections: list[WebSocket] = []


class ConnectionManager:
    """Manage WebSocket connections for broadcasting updates."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and store a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"WebSocket connected. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
        )

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/progress")
async def progress_websocket(websocket: WebSocket) -> None:
    """Stream progress animation updates in real-time.

    This WebSocket endpoint pushes progress state updates every 200ms
    while an animation is active. The frontend receives these updates
    and can update the DOM instantly without any callback overhead.

    Uses TIME-BASED animation: each tick recalculates visibility based on
    elapsed time since animation started. This is stateless - no tracking bugs.

    Message format:
    {
        "type": "progress_update",
        "data": {
            "sections": [...],
            "revealed_counts": {"thinking": 3, "research": 2},
            "is_animating": true,
            "is_complete": false,
            "current_step_id": "step_1",
            "step_messages": [...],  // Only when is_complete=true
            "suggested_actions": [...],  // Only when is_complete=true
            "file_references": [...]  // Only when is_complete=true
        }
    }
    """
    await manager.connect(websocket)

    try:
        # Get experiment service from app state
        service = websocket.app.state.get_experiment_service()

        # Track if we've already sent completion data to avoid duplicates
        completion_sent = False

        # Keep connection alive and push updates
        while True:
            # Check for incoming messages (e.g., ping/pong, commands)
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.2)
                # Handle client messages if needed
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data.get("type") == "reset":
                    # Reset completion tracking when a new animation starts
                    completion_sent = False
            except asyncio.TimeoutError:
                # No message received, continue to push update
                pass

            # Time-based calculation: get current progress state
            # This is STATELESS - calculates revealed_counts from elapsed time
            progress_state = service.get_progress_state()

            # Build the update message
            message_data: dict[str, Any] = {
                "sections": progress_state.sections,
                "revealed_counts": progress_state.revealed_counts,
                "is_animating": progress_state.is_animating,
                "is_complete": progress_state.is_complete,
                "current_step_id": progress_state.current_step_id,
            }

            # When animation completes, include step messages with file references
            if progress_state.is_complete and not completion_sent:
                step_data = service.get_current_step_messages()
                message_data["step_messages"] = step_data.get("messages", [])
                message_data["suggested_actions"] = step_data.get(
                    "suggested_actions", []
                )
                message_data["wait_for_user"] = step_data.get("wait_for_user", True)

                # Extract file references from messages for "📄 View File" buttons
                file_references = []
                for msg in step_data.get("messages", []):
                    if msg.get("file_path"):
                        file_references.append(
                            {
                                "file_path": msg["file_path"],
                                "content_preview": msg.get("content", "")[:100],
                            }
                        )
                message_data["file_references"] = file_references

                # Mark animation as complete in the service
                service.complete_progress_animation()

                # If step doesn't wait for user, advance to next step
                # (e.g., wait_for_user=false steps auto-advance)
                if not step_data.get("wait_for_user", True):
                    logger.info(
                        f"Step '{progress_state.current_step_id}' has wait_for_user=false, "
                        f"auto-advancing to next step"
                    )
                    # Actually advance the step!
                    if service._engine:
                        service._engine.advance_step()
                    # Process the next step (this starts the next step's animation)
                    await service._process_current_step()
                    # Reset completion_sent so we can detect next step's completion
                    completion_sent = False
                else:
                    completion_sent = True
                logger.info(
                    f"Progress animation complete for step '{progress_state.current_step_id}', "
                    f"sending {len(step_data.get('messages', []))} messages, "
                    f"{len(step_data.get('suggested_actions', []))} actions"
                )

            await websocket.send_json(
                {
                    "type": "progress_update",
                    "data": message_data,
                }
            )

            # 200ms update interval - fast enough for smooth animation
            await asyncio.sleep(0.2)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        manager.disconnect(websocket)


@router.websocket("/chat")
async def chat_websocket(websocket: WebSocket) -> None:
    """WebSocket for bidirectional chat communication.

    This can be used for streaming responses and real-time
    message updates. For now, it's a simple echo server
    that will be extended for streaming chat.

    Message format (client -> server):
    {
        "type": "message",
        "content": "user message text"
    }

    Message format (server -> client):
    {
        "type": "message",
        "role": "assistant",
        "content": "response text",
        "streaming": false
    }
    """
    await manager.connect(websocket)

    try:
        service = websocket.app.state.get_experiment_service()

        while True:
            # Wait for client message
            data = await websocket.receive_json()

            if data.get("type") == "message":
                # Process user input
                user_message = data.get("content", "")
                state = await service.process_user_input(user_message)

                # Send back the updated messages
                for msg in state.messages:
                    await websocket.send_json(
                        {
                            "type": "message",
                            "role": msg.role,
                            "content": msg.content,
                            "file_path": msg.file_path,
                            "message_type": msg.message_type,
                            "streaming": False,
                        }
                    )

                # Send suggested actions
                if state.suggested_actions:
                    await websocket.send_json(
                        {
                            "type": "actions",
                            "actions": state.suggested_actions,
                        }
                    )

            elif data.get("type") == "action":
                # Handle action click
                action_index = data.get("index", 0)
                state = await service.handle_action(action_index)

                # Send updated state
                await websocket.send_json(
                    {
                        "type": "state_update",
                        "current_step_id": state.current_step_id,
                        "is_complete": state.is_complete,
                        "is_waiting_for_user": state.is_waiting_for_user,
                    }
                )

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("Chat WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Chat WebSocket error: {e}", exc_info=True)
    finally:
        manager.disconnect(websocket)

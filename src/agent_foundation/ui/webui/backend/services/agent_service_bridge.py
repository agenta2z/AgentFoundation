

"""AgentServiceBridge — thin bridge between WebSocket and file queue.

Zero business logic. Translates between the WebSocket protocol
(React frontend expects) and the QueueInteractive protocol
(server uses InteractionFlags + token_batch).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine

from agent_foundation.ui.interactive_base import (
    InteractionFlags,
)
from rich_python_utils.service_utils.client.queue_client_base import QueueClientBase

logger = logging.getLogger(__name__)

SendCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class AgentServiceBridge(QueueClientBase):
    """Thin bridge: WebSocket <-> File Queue. Zero business logic.

    Translates between the WebSocket protocol (React frontend expects)
    and the QueueInteractive protocol (server uses InteractionFlags + token_batch).
    """

    def __init__(
        self,
        queue_root_path: str | Path,
        session_id: str | None = None,
    ) -> None:
        super().__init__(
            queue_root_path=str(queue_root_path),
            session_id=session_id,
            session_type="webui",
        )
        self._active_tailer: Any | None = None

    async def handle_message(self, text: str, send: SendCallback) -> None:
        """Determine message type, send to server, start polling."""
        if text.startswith("/"):
            msg_type = "slash_command"
        else:
            msg_type = "chat_message"

        self.send_message(msg_type, text)
        await self.poll_responses(send)

    async def stream_from_workspace(
        self, workspace_path: str, send: SendCallback,
        task_id: str | None = None,
    ) -> str:
        """Tail inferencer cache files and stream tokens to the client.

        Args:
            workspace_path: Absolute path to the workspace directory.
            send: Async callback to send messages to the WebSocket client.
            task_id: Optional task ID to include in token messages.

        Returns:
            The accumulated full text for use in message_end.
        """
        from rankevolve.src.common.streaming.file_tailer import WorkspaceStreamTailer  # TODO: migrate from rankevolve.src.common.streaming.file_tailer
        from rankevolve.src.common.workspace.layout import get_cache_dir  # TODO: migrate from rankevolve.src.common.workspace.layout

        cache_dir = get_cache_dir(workspace_path)

        tailer = WorkspaceStreamTailer(
            cache_dir,
            default_phase="unknown",
            replay_existing=True,
        )
        self._active_tailer = tailer

        async def _send_chunk(content: str, metadata: dict[str, Any]) -> None:
            msg: dict[str, Any] = {
                "type": "token",
                "content": content,
                "metadata": metadata,
            }
            if task_id:
                msg["task_id"] = task_id
            await send(msg)

        await tailer.tail(callback=_send_chunk)
        return tailer.accumulated_content

    async def stream_from_cache_folder(
        self, cache_folder: str, send: SendCallback
    ) -> str:
        """Tail inferencer cache files from a conversation turn directory.

        Same mechanism as stream_from_workspace but takes a direct cache folder
        path (no workspace layout indirection via get_cache_dir).

        Args:
            cache_folder: Absolute path to the turn's cache directory.
            send: Async callback to send messages to the WebSocket client.

        Returns:
            The accumulated full text for use in message_end.
        """
        from rankevolve.src.common.streaming.file_tailer import WorkspaceStreamTailer  # TODO: migrate from rankevolve.src.common.streaming.file_tailer

        tailer = WorkspaceStreamTailer(
            cache_folder,
            default_phase="conversation",
            replay_existing=True,
        )
        self._active_tailer = tailer

        async def _send_chunk(content: str, metadata: dict[str, Any]) -> None:
            await send(
                {
                    "type": "token",
                    "content": content,
                    "metadata": metadata,
                }
            )

        await tailer.tail(callback=_send_chunk)
        return tailer.accumulated_content

    async def poll_responses(self, send: SendCallback) -> None:
        """Async loop: poll response queue, translate to WebSocket messages.

        Tracks conversation tailer and task tailer separately. When the
        conversation turn ends (stream_end/TurnCompleted), the conversation
        tailer is stopped but the task tailer continues running. The loop
        only breaks when BOTH the conversation is done AND no task is active.
        """
        tailer_task: asyncio.Task[str] | None = None  # conversation tailer
        task_tailer_task: asyncio.Task[str] | None = None  # background task tailer
        task_tailer_obj: Any | None = None  # background task's WorkspaceStreamTailer
        has_active_task = False  # True after task_status:starting, until completed/error
        conversation_done = False  # True after stream_end/TurnCompleted

        while True:
            # If conversation is done and no active background task, exit
            if conversation_done and not has_active_task:
                break

            self._maybe_send_heartbeat()

            resp = await self.poll_one_response()
            if resp is None:
                if not self.is_server_alive():
                    await send(
                        {
                            "type": "error",
                            "message": "Server appears to be down.",
                        }
                    )
                    break
                await asyncio.sleep(0.1)
                continue

            msg_type = resp.get("type", "")
            flag = resp.get("flag", "")

            logger.info(
                "poll_responses: msg_type=%s flag=%s tailer=%s task_tailer=%s keys=%s",
                msg_type, flag, tailer_task is not None,
                task_tailer_task is not None, list(resp.keys()),
            )

            if msg_type == "token_batch":
                # Skip token_batch when conversation tailer is active
                if tailer_task is not None:
                    continue
                task_id = resp.get("task_id")
                for token in resp.get("tokens", []):
                    msg: dict[str, Any] = {
                        "type": "token",
                        "content": token.get("content", ""),
                        "metadata": token.get("metadata", {}),
                    }
                    if task_id:
                        msg["task_id"] = task_id
                    await send(msg)

            elif msg_type == "stream_end":
                end_msg: dict[str, Any] = {
                    "type": "message_end",
                    "final_content": resp.get("final_content", ""),
                }
                if resp.get("task_id"):
                    end_msg["task_id"] = resp["task_id"]
                if resp.get("turn_number") is not None:
                    end_msg["turn_number"] = resp["turn_number"]
                # Stop the CONVERSATION tailer only (not the task tailer)
                if tailer_task is not None and not tailer_task.done():
                    if self._active_tailer is not None:
                        self._active_tailer.stop()
                    final_from_tailer = await tailer_task
                    end_msg["final_content"] = (
                        final_from_tailer or resp.get("final_content", "")
                    )
                tailer_task = None
                self._active_tailer = None
                await send(end_msg)
                conversation_done = True
                # Don't break — continue polling for task_status updates

            elif msg_type == "stream_start":
                # Launch file-based streaming for conversation turns
                cache_folder = resp.get("cache_folder")
                if cache_folder and tailer_task is None:
                    tailer_task = asyncio.create_task(
                        self.stream_from_cache_folder(cache_folder, send)
                    )

            # Detect pending_input: QueueInteractive sends responses with
            # flag="PendingInput" and input_mode, but no type field.
            # Handle both explicit type="pending_input" and flag-based detection.
            elif msg_type == "pending_input" or (
                flag == "PendingInput" and resp.get("input_mode")
            ):
                # Stop the conversation tailer before showing the widget
                if tailer_task is not None and not tailer_task.done():
                    if self._active_tailer is not None:
                        self._active_tailer.stop()
                    await tailer_task
                    tailer_task = None
                    self._active_tailer = None
                await send(
                    {
                        "type": "pending_input",
                        "content": resp.get("response", ""),
                        "input_mode": resp.get("input_mode"),
                        "widget": resp.get("widget"),
                    }
                )

            elif msg_type == "command_response":
                cmd_msg: dict[str, Any] = {
                    "type": "command_response",
                    "content": resp.get("message", ""),
                    "data": resp,
                }
                if resp.get("turn_number") is not None:
                    cmd_msg["turn_number"] = resp["turn_number"]
                await send(cmd_msg)
                conversation_done = True

            elif msg_type == "widget_update":
                await send(
                    {
                        "type": "widget_update",
                        "widget": resp.get("widget", {}),
                    }
                )

            elif msg_type == "config_update":
                await send(
                    {
                        "type": "config_update",
                        "config": resp.get("config", {}),
                    }
                )

            elif msg_type == "task_status":
                await send(resp)

                status = resp.get("status", "")

                # Launch workspace tailer when task starts with a workspace
                workspace = resp.get("workspace")
                if (
                    workspace
                    and status == "starting"
                    and task_tailer_task is None
                ):
                    has_active_task = True
                    task_tailer_task = asyncio.create_task(
                        self.stream_from_workspace(
                            workspace, send, task_id=resp.get("task_id"),
                        )
                    )
                    # Save ref so we can stop it independently
                    task_tailer_obj = self._active_tailer

                # Stop task tailer and send message_end when task completes
                if status in ("completed", "error"):
                    if task_tailer_task is not None and not task_tailer_task.done():
                        if task_tailer_obj is not None:
                            task_tailer_obj.stop()
                        final_content = await task_tailer_task
                        await send(
                            {
                                "type": "message_end",
                                "final_content": final_content,
                                "task_id": resp.get("task_id"),
                            }
                        )
                    task_tailer_task = None
                    task_tailer_obj = None
                    has_active_task = False

            elif msg_type == "session_sync_response":
                await send(resp)

            elif msg_type == "session_notification":
                await send(resp)  # Forward lightweight notification to client

            elif msg_type == "queue_status_response":
                await send(resp)

            elif msg_type == "error":
                await send(
                    {
                        "type": "error",
                        "message": resp.get("message", ""),
                    }
                )
                break

            elif msg_type == "pong":
                continue

            elif flag and str(flag) == str(InteractionFlags.TurnCompleted):
                # Stop CONVERSATION tailer only (not task tailer)
                if tailer_task is not None and not tailer_task.done():
                    if self._active_tailer is not None:
                        self._active_tailer.stop()
                    final_content = await tailer_task
                    await send(
                        {
                            "type": "message_end",
                            "final_content": final_content,
                        }
                    )
                    tailer_task = None
                    self._active_tailer = None
                conversation_done = True

    def cancel_active_task(self) -> bool:
        """Put agent_control stop on input queue."""
        self.send_message(
            "agent_control",
            message={
                "session_id": self.session_id,
                "control": "stop",
            },
        )
        return True

    def get_config_info(self) -> None:
        """Send config query to server."""
        self.send_message("config_query")

    def send_queue_status_request(self) -> None:
        """Send queue_status_request to server."""
        self.send_message("queue_status_request")

    async def send_sync_request(self) -> dict[str, Any] | None:
        """Send session_sync_request and poll for session_sync_response."""
        self.send_message("session_sync_request")
        for _ in range(50):  # poll up to 5 seconds
            resp = await self.poll_one_response()
            if resp and resp.get("type") == "session_sync_response":
                return resp
            await asyncio.sleep(0.1)
        return None

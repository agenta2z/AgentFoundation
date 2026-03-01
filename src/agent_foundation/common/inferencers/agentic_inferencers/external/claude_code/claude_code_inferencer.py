"""Claude Code SDK Inferencer.

Wraps the Claude Code SDK client as an async-native StreamingInferencerBase implementation.
Provides both async interface (recommended) and sync bridge (for backwards compatibility).
"""

import asyncio
import logging
from typing import Any, AsyncIterator, List, Optional

from attr import attrib, attrs
from agent_foundation.common.inferencers.agentic_inferencers.external.sdk_types import (
    SDKInferencerResponse,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)

logger = logging.getLogger(__name__)


@attrs
class ClaudeCodeInferencer(StreamingInferencerBase):
    """Claude Code SDK as an async-native streaming inferencer with session continuation.

    Inherits from StreamingInferencerBase which provides:
    - ``ainfer_streaming()`` with idle timeout and optional caching
    - ``infer_streaming()`` sync bridge via thread + queue
    - Session management: ``new_session``, ``anew_session``, ``resume_session``, ``aresume_session``
    - ``active_session_id`` property

    This class implements ``_produce_chunks()`` (the abstract primitive) and
    overrides ``_ainfer()`` to support ``SDKInferencerResponse``.

    Runtime Dependencies:
        Requires claude-agent-sdk package. This is a soft dependency — the
        module imports successfully without it. ImportError is raised only
        when aconnect() or _produce_chunks() is called.

    Usage Patterns:
        # Sync (simple, but pays connect cost each call):
        inferencer = ClaudeCodeInferencer(root_folder="/path/to/repo")
        result = inferencer("Write a hello world program")

        # Multi-turn with auto-resume (recommended):
        inferencer = ClaudeCodeInferencer(root_folder="/repo", auto_resume=True)
        r1 = inferencer.new_session("My number is 42")
        r2 = inferencer.infer("What is my number?")  # Auto-resumes!

        # Async with context manager (persistent connection):
        async with ClaudeCodeInferencer(root_folder="/repo") as inf:
            r1 = await inf.anew_session("My number is 42")
            r2 = await inf.ainfer("What is my number?")  # Auto-resumes!

        # Sync streaming:
        for chunk in inferencer.infer_streaming("Explain this"):
            print(chunk, end="", flush=True)

        # Async streaming:
        async for chunk in inferencer.ainfer_streaming("Explain this"):
            print(chunk, end="", flush=True)

    Attributes:
        root_folder: Working directory for Claude Code agent.
        system_prompt: System prompt to configure agent behavior.
        idle_timeout_seconds: Per-chunk idle timeout in seconds (inherited,
            overridden to 1800). If no new text chunk arrives within this
            duration, the stream is considered stalled.
        allowed_tools: List of tools Claude can use (default: Read, Write, Bash).
        include_partial_messages: Whether to include partial messages in stream.
        auto_resume: If True, automatically resume previous session (default True).
    """

    # ClaudeCode-specific attributes
    # idle_timeout_seconds overridden to 1800 (was timeout_seconds=1800 in old code)
    idle_timeout_seconds: int = attrib(default=1800)
    root_folder: Optional[str] = attrib(default=None)
    system_prompt: str = attrib(default="")
    allowed_tools: List[str] = attrib(factory=lambda: ["Read", "Write", "Bash"])
    include_partial_messages: bool = attrib(default=True)

    # Internal state
    _client: Any = attrib(default=None, init=False, repr=False)
    _disconnect_fn: Any = attrib(default=None, init=False, repr=False)
    _connected_loop: Any = attrib(default=None, init=False, repr=False)
    _connect_lock: Any = attrib(default=None, init=False, repr=False)
    _last_tool_use_count: int = attrib(default=0, init=False, repr=False)

    # === Streaming Primitive ===

    async def _produce_chunks(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield text chunks from Claude Code SDK stream.

        No internal idle timeout — the base class ``ainfer_streaming()`` handles
        it with ``idle_timeout_seconds=1800``.

        Args:
            prompt: The prompt string.
            **kwargs: Additional arguments (session_id, new_session, etc.).

        Yields:
            Text chunks as they arrive from Claude.
        """
        try:
            from claude_agent_sdk.types import (
                AssistantMessage,
                ResultMessage,
                TextBlock,
                ToolUseBlock,
            )
        except ImportError as e:
            raise RuntimeError(
                f"Claude Agent SDK not available: {e}. "
                "Ensure fbsource//third-party/pypi/claude-agent-sdk:claude-agent-sdk "
                "is in deps."
            ) from e

        # Thread-safe lazy connect with asyncio.Lock
        if self._client is None:
            if self._connect_lock is None:
                self._connect_lock = asyncio.Lock()
            async with self._connect_lock:
                if self._client is None:
                    await self.aconnect(session_id=kwargs.get("session_id"))

        client = self._client

        # query() returns quickly — then we stream the response
        await client.query(prompt)
        message_stream = client.receive_response()

        async for message in message_stream:
            match message:
                case AssistantMessage(content=blocks):
                    for block in blocks:
                        if isinstance(block, TextBlock):
                            yield block.text
                        elif isinstance(block, ToolUseBlock):
                            self._last_tool_use_count += 1
                            self.log_info(
                                f"Tool use #{self._last_tool_use_count}: {block.name}",
                                "ToolUse",
                            )
                case ResultMessage() as result_msg:
                    self._session_id = result_msg.session_id
                    self.log_info(
                        f"session_id={result_msg.session_id}",
                        "ResultMessage",
                    )
                case _:
                    self.log_info(
                        str(message),
                        f"StreamMessage_{type(message).__name__}",
                    )

    # === Overrides ===

    async def _ainfer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Override to support SDKInferencerResponse and tool use counting.

        Args:
            inference_input: Input for inference.
            inference_config: Optional configuration (unused).
            **kwargs: Additional arguments:
                - return_sdk_response: If True, return SDKInferencerResponse.

        Returns:
            Response text string, or SDKInferencerResponse if return_sdk_response=True.
        """
        self._last_tool_use_count = 0
        response_text = await super()._ainfer(
            inference_input, inference_config, **kwargs
        )
        if kwargs.get("return_sdk_response", False):
            return SDKInferencerResponse(
                content=response_text,
                session_id=self._session_id,
                tool_uses=self._last_tool_use_count,
            )
        return response_text

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args: Any
    ) -> Any:
        """Sync bridge with stale-loop detection and self-contained sessions.

        CRITICAL: asyncio.run() closes the event loop after completion.
        ClaudeSDKClient holds persistent loop-bound state (subprocess,
        anyio task groups, background tasks) that becomes invalid when
        the loop closes. We MUST detect this and reconnect.

        For multi-call usage, prefer the async interface:
            async with ClaudeCodeInferencer(...) as inf:
                r1 = await inf.ainfer("first")
                r2 = await inf.ainfer("second")

        Args:
            inference_input: Input for inference (string or dict with "prompt" key).
            inference_config: Optional configuration (unused).
            **_inference_args: Additional args passed to _ainfer.

        Returns:
            Inference response (string or SDKInferencerResponse if return_sdk_response=True).

        Raises:
            RuntimeError: If client was connected in a different event loop.
        """
        from rich_python_utils.common_utils.async_function_helper import _run_async

        # Detect stale client from previous asyncio.run()
        if self._connected_loop is not None and self._connected_loop.is_closed():
            logger.debug("Previous event loop is closed — clearing stale client")
            self._client = None
            self._disconnect_fn = None
            self._connected_loop = None

        # Cross-loop guard
        if self._client is not None and self._connected_loop is not None:
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass  # No running loop — _run_async will create one, safe
            else:
                if current_loop is not self._connected_loop:
                    raise RuntimeError(
                        "Cannot use sync _infer() when client was connected in a "
                        "different event loop. Use 'await inferencer.ainfer()' instead, "
                        "or call adisconnect() and let the sync path reconnect."
                    )

        return _run_async(
            self._ainfer(inference_input, inference_config, **_inference_args)
        )

    # === Connection Lifecycle ===

    async def aconnect(self, session_id: Optional[str] = None, **kwargs: Any) -> None:
        """Establish connection using verified Future/Event/Task pattern.

        Args:
            session_id: Optional session ID to resume a previous conversation.
            **kwargs: Additional connection arguments (unused).
        """
        try:
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
        except ImportError as e:
            raise RuntimeError(
                f"Claude Agent SDK not available: {e}. "
                "Ensure fbsource//third-party/pypi/claude-agent-sdk:claude-agent-sdk "
                "is in deps."
            ) from e

        options = ClaudeAgentOptions(
            model=self.model_id or None,
            cwd=str(self.root_folder) if self.root_folder else None,
            system_prompt=self.system_prompt,
            include_partial_messages=self.include_partial_messages,
            allowed_tools=self.allowed_tools,
            resume=session_id,
        )

        client = ClaudeSDKClient(options=options)

        loop = asyncio.get_running_loop()
        connect_future = loop.create_future()
        disconnect_event = asyncio.Event()

        async def _inner() -> None:
            try:
                await client.connect()
                connect_future.set_result(None)
            except Exception as e:
                connect_future.set_exception(e)
            await disconnect_event.wait()
            await client.disconnect()

        task = asyncio.create_task(_inner())

        async def _disconnect() -> None:
            disconnect_event.set()
            await task

        await connect_future
        self._client = client
        self._disconnect_fn = _disconnect
        self._session_id = session_id
        self._connected_loop = loop
        logger.debug("Claude Code SDK connected (session_id=%s)", session_id)

    async def adisconnect(self) -> None:
        """Disconnect from Claude Code SDK."""
        if self._disconnect_fn:
            await self._disconnect_fn()
            self._disconnect_fn = None
        self._client = None
        self._connected_loop = None
        logger.debug("Claude Code SDK disconnected")

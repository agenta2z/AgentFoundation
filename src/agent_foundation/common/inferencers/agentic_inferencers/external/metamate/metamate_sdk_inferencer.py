# pyre-strict
"""MetaMate SDK Inferencer.

Wraps the MetamateGraphQLClient as an async-native StreamingInferencerBase
implementation with polling-based streaming and auto-continuation support.

Uses ``engine_start_v2()`` to start conversations and
``get_conversation_for_stream()`` to poll for results, yielding text deltas
as they arrive.

Runtime Dependencies:
    Requires ``msl.metamate.cli.metamate_graphql`` package. This is a soft
    dependency — the module imports successfully without it. RuntimeError
    is raised only when ``_ainfer_streaming()`` is called.
"""

import asyncio
import logging
import time
import uuid as uuid_mod
from typing import Any, AsyncIterator, Optional

from attr import attrib, attrs
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.sdk_types import (
    SDKInferencerResponse,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.common import (
    _TERMINAL_STATUSES,
    AUTO_CONTINUE_REPLY,
    DEFAULT_API_KEY,
    DEFAULT_MODE,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_STREAM_TYPE,
    DEFAULT_SURFACE,
    DEFAULT_TIMEOUT,
    get_assistant_message_status,
    MAX_CONTINUATIONS,
    needs_continuation,
    parse_assistant_text,
    resolve_conversation_fbid,
)

logger: logging.Logger = logging.getLogger(__name__)


@attrs
class MetamateSDKInferencer(StreamingInferencerBase):
    """MetaMate SDK as an async-native streaming inferencer with auto-continuation.

    Uses ``MetamateGraphQLClient.engine_start_v2()`` to start a conversation
    and polls ``get_conversation_for_stream()`` to yield text deltas.

    Inherits from StreamingInferencerBase which provides:
    - ``ainfer_streaming()`` with idle timeout and optional caching
    - ``infer_streaming()`` sync bridge via thread + queue
    - Session management: ``new_session``, ``anew_session``, ``resume_session``, ``aresume_session``
    - ``active_session_id`` property

    This class implements ``_ainfer_streaming()`` (the abstract primitive) and
    overrides ``_ainfer()`` to support session kwargs and ``SDKInferencerResponse``.

    Usage Patterns:
        # Single query:
        inferencer = MetamateSDKInferencer(cat_token="...")
        result = inferencer("What is MetaMate?")

        # Streaming:
        async for chunk in inferencer.ainfer_streaming("Explain this"):
            print(chunk, end="", flush=True)

        # Multi-turn with auto-resume:
        r1 = inferencer.new_session("My number is 42")
        r2 = inferencer("What is my number?")  # Auto-resumes!

    Attributes:
        api_key: MetaMate API key.
        surface: Surface identifier for the API.
        mode: Conversation mode.
        stream_type: Stream type for the API.
        agent_name: Optional agent name override.
        cat_token: CAT token for authentication. None uses current user.
        auto_continue: If True, auto-reply when agent asks clarification.
        max_continuations: Max auto-continue follow-ups per query.
        poll_interval_seconds: Seconds between poll attempts.
        timeout_seconds: Per-call timeout for ``engine_start_v2``.
        total_timeout_seconds: Max total time for entire operation.
        idle_timeout_seconds: Max idle time between chunks.
    """

    api_key: str = attrib(default=DEFAULT_API_KEY)
    surface: str = attrib(default=DEFAULT_SURFACE)
    mode: str = attrib(default=DEFAULT_MODE)
    stream_type: str = attrib(default=DEFAULT_STREAM_TYPE)
    agent_name: Optional[str] = attrib(default=None)
    cat_token: Optional[str] = attrib(default=None)
    auto_continue: bool = attrib(default=True)
    max_continuations: int = attrib(default=MAX_CONTINUATIONS)
    poll_interval_seconds: float = attrib(default=DEFAULT_POLL_INTERVAL)
    timeout_seconds: int = attrib(default=DEFAULT_TIMEOUT)
    total_timeout_seconds: int = attrib(default=1800)
    idle_timeout_seconds: int = attrib(default=600)

    # Internal state
    _conversation_uuid: Optional[str] = attrib(default=None, init=False, repr=False)
    _conversation_fbid: Optional[str] = attrib(default=None, init=False, repr=False)
    _last_token_count: int = attrib(default=0, init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        self.log_info(
            f"api_key={self.api_key[:8]}..., surface={self.surface}, "
            f"mode={self.mode}, agent_name={self.agent_name}, "
            f"auto_continue={self.auto_continue}, "
            f"poll_interval={self.poll_interval_seconds}s, "
            f"timeout={self.timeout_seconds}s",
            "Config",
        )

    # === Streaming Primitive ===

    async def _ainfer_streaming(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield text deltas by polling MetaMate conversation stream.

        Creates a fresh ``MetamateGraphQLClient`` per call. Starts a
        conversation via ``engine_start_v2()``, then polls
        ``get_conversation_for_stream()`` in a background task, yielding
        new text deltas via an ``asyncio.Queue``.

        Supports auto-continuation: if the agent asks a clarification
        question, automatically replies and continues polling.

        Args:
            prompt: The prompt string.
            **kwargs: Additional arguments:
                - conversation_uuid: UUID for multi-turn resume.
                - conversation_fbid: FBID for multi-turn resume.

        Yields:
            Text deltas as they arrive from MetaMate.
        """
        try:
            from msl.metamate.cli.metamate_graphql import MetamateGraphQLClient
        except ImportError as e:
            raise RuntimeError(
                f"MetaMate SDK not available: {e}. "
                "Ensure //msl/metamate/cli:metamate_graphql is in deps."
            ) from e

        client = MetamateGraphQLClient(cat=self.cat_token)
        request_id = str(uuid_mod.uuid4())

        conv_uuid: Optional[str] = kwargs.get("conversation_uuid")
        conv_fbid: Optional[str] = kwargs.get("conversation_fbid")

        # Queue for streaming deltas
        chunk_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        error_holder: list[Exception] = []

        async def _poll_loop() -> None:
            """Background task: start conversation, poll, yield deltas."""
            nonlocal conv_uuid, conv_fbid

            try:
                result = await asyncio.to_thread(
                    client.engine_start_v2,
                    prompt=prompt,
                    request_id=request_id,
                    api_key=self.api_key,
                    surface=self.surface,
                    mode=self.mode,
                    stream_type=self.stream_type,
                    agent_name=self.agent_name,
                    timeout_seconds=self.timeout_seconds,
                    conversation_uuid=conv_uuid,
                    conversation_fbid=conv_fbid,
                )

                conv_uuid = result.conversation.uuid
                conv_fbid = result.conversation.fbid

                self.log_info(
                    f"Conversation started — uuid={conv_uuid}, fbid={conv_fbid}",
                    "EngineStart",
                )

                last_accumulated_text = ""
                continuations_sent = 0
                start_time = time.monotonic()

                while True:
                    # Check total timeout budget
                    elapsed = time.monotonic() - start_time
                    if (
                        self.total_timeout_seconds > 0
                        and elapsed > self.total_timeout_seconds
                    ):
                        self.log_info(
                            f"Total timeout reached ({self.total_timeout_seconds}s)",
                            "Timeout",
                        )
                        break

                    bridge_outputs = await asyncio.to_thread(
                        client.get_conversation_for_stream,
                        conv_uuid,
                    )

                    text = parse_assistant_text(bridge_outputs)
                    status = get_assistant_message_status(bridge_outputs)

                    # Yield new text delta
                    if text and len(text) > len(last_accumulated_text):
                        if text.startswith(last_accumulated_text):
                            delta = text[len(last_accumulated_text) :]
                        else:
                            delta = text
                        if delta:
                            await chunk_queue.put(delta)
                            self._last_token_count += len(delta)
                        last_accumulated_text = text

                    # Check for terminal status
                    if (
                        status
                        and status in _TERMINAL_STATUSES
                        and len(text.strip()) > 0
                    ):
                        self.log_info(
                            f"Terminal status reached: {status} ({len(text)} chars)",
                            "PollComplete",
                        )
                        # Auto-continuation check
                        if (
                            self.auto_continue
                            and needs_continuation(text)
                            and continuations_sent < self.max_continuations
                        ):
                            continuations_sent += 1
                            self.log_info(
                                f"Auto-continuing (turn {continuations_sent}"
                                f"/{self.max_continuations})",
                                "AutoContinue",
                            )
                            follow_up_id = str(uuid_mod.uuid4())
                            result = await asyncio.to_thread(
                                client.engine_start_v2,
                                prompt=AUTO_CONTINUE_REPLY,
                                request_id=follow_up_id,
                                api_key=self.api_key,
                                surface=self.surface,
                                mode=self.mode,
                                stream_type=self.stream_type,
                                agent_name=self.agent_name,
                                timeout_seconds=self.timeout_seconds,
                                conversation_uuid=conv_uuid,
                                conversation_fbid=conv_fbid,
                            )
                            # Do NOT reset last_accumulated_text here.
                            # get_conversation_for_stream() returns ALL bridge
                            # outputs across the conversation, so
                            # parse_assistant_text() grows monotonically.
                            # Keeping the current value ensures the delta logic
                            # only yields genuinely new text.
                            continue
                        break

                    await asyncio.sleep(self.poll_interval_seconds)

            except Exception as exc:
                error_holder.append(exc)
            finally:
                await chunk_queue.put(None)

        poll_task = asyncio.create_task(_poll_loop())

        try:
            while True:
                get_task = asyncio.ensure_future(chunk_queue.get())
                done, _ = await asyncio.wait(
                    [get_task, poll_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if get_task in done:
                    chunk = get_task.result()
                    if chunk is None:
                        break
                    yield chunk
                elif poll_task in done:
                    get_task.cancel()
                    poll_task.result()
                    break
        finally:
            if not poll_task.done():
                poll_task.cancel()

            self._conversation_uuid = conv_uuid
            self._conversation_fbid = conv_fbid
            self._session_id = conv_uuid

            if error_holder:
                raise error_holder[0]

    # === Overrides ===

    async def _ainfer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Override for session management and SDKInferencerResponse support.

        Resolves session kwargs (``new_session``, ``session_id``,
        ``auto_resume``) into ``conversation_uuid`` / ``conversation_fbid``
        and injects them into kwargs BEFORE calling ``super()._ainfer()``.

        Args:
            inference_input: Input for inference.
            inference_config: Optional configuration (unused).
            **kwargs: Additional arguments:
                - return_sdk_response: If True, return SDKInferencerResponse.
                - session_id: Conversation UUID to resume.
                - new_session: If True, forces a new conversation.

        Returns:
            Response text string, or SDKInferencerResponse if
            ``return_sdk_response=True``.
        """
        new_session = kwargs.pop("new_session", False)
        explicit_session_id = kwargs.pop("session_id", None)
        return_sdk_response = kwargs.pop("return_sdk_response", False)

        if new_session:
            kwargs["conversation_uuid"] = None
            kwargs["conversation_fbid"] = None
            logger.debug("Starting new conversation (new_session=True)")
        elif explicit_session_id:
            kwargs["conversation_uuid"] = explicit_session_id
            # Look up FBID for the conversation UUID
            try:
                from msl.metamate.cli.metamate_graphql import MetamateGraphQLClient
            except ImportError:
                kwargs["conversation_fbid"] = None
            else:
                client = MetamateGraphQLClient(cat=self.cat_token)
                fbid = await asyncio.to_thread(
                    resolve_conversation_fbid, client, explicit_session_id
                )
                kwargs["conversation_fbid"] = fbid
            logger.debug(
                "Resuming conversation: %s",
                explicit_session_id[:8] if explicit_session_id else None,
            )
        elif self.auto_resume and self._conversation_uuid:
            kwargs["conversation_uuid"] = self._conversation_uuid
            kwargs["conversation_fbid"] = self._conversation_fbid
            logger.debug(
                "Auto-resuming conversation: %s",
                self._conversation_uuid[:8] if self._conversation_uuid else None,
            )
        else:
            kwargs["conversation_uuid"] = None
            kwargs["conversation_fbid"] = None
            logger.debug("Starting fresh conversation (no previous session)")

        self._last_token_count = 0
        response_text = await super()._ainfer(
            inference_input, inference_config, **kwargs
        )
        if return_sdk_response:
            return SDKInferencerResponse(
                content=response_text,
                session_id=self._session_id,
                tokens_received=self._last_token_count,
            )
        return response_text

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Sync wrapper for async ``_ainfer()``.

        Uses ``_run_async()`` from the common utils helper. Note: cannot be
        called from an async context (would raise RuntimeError). Use
        ``_ainfer()`` or ``ainfer_streaming()`` directly in async code.

        Args:
            inference_input: Input for inference.
            inference_config: Optional configuration.
            **kwargs: Additional arguments passed to ``_ainfer()``.

        Returns:
            Response text string, or SDKInferencerResponse if
            ``return_sdk_response=True``.
        """
        from rich_python_utils.common_utils.async_function_helper import _run_async

        return _run_async(self._ainfer(inference_input, inference_config, **kwargs))

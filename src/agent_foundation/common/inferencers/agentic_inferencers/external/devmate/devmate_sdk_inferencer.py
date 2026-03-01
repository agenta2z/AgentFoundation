"""Devmate SDK Inferencer.

Wraps the Devmate SDK client as an async-native StreamingInferencerBase implementation.
Unlike ClaudeCodeInferencer, Devmate SDK creates a fresh client per query.
Supports session continuation via previous_session_id parameter.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from attr import attrib, attrs
from agent_foundation.common.inferencers.agentic_inferencers.external.sdk_types import (
    SDKInferencerResponse,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)

logger = logging.getLogger(__name__)


@attrs
class DevmateSDKInferencer(StreamingInferencerBase):
    """Devmate SDK as an async-native streaming inferencer with session continuation.

    Unlike ClaudeCodeInferencer (persistent connection), Devmate SDK
    creates a fresh client per query. However, it supports session continuation
    via the previous_session_id parameter in start_session().

    Inherits from StreamingInferencerBase which provides:
    - ``ainfer_streaming()`` with idle timeout and optional caching
    - ``infer_streaming()`` sync bridge via thread + queue
    - Session management: ``new_session``, ``anew_session``, ``resume_session``, ``aresume_session``
    - ``active_session_id`` property

    This class implements ``_produce_chunks()`` (the abstract primitive) and
    overrides ``_ainfer()`` to support ``SDKInferencerResponse`` and session kwargs.

    Runtime Dependencies:
        Requires devai.devmate_sdk package. This is a soft dependency — the
        module imports successfully without it. ImportError is raised only
        when _produce_chunks() is called.

    Usage Patterns:
        # Single query:
        inferencer = DevmateSDKInferencer(root_folder="/path/to/repo")
        result = inferencer("Write a hello world program")

        # Multi-turn with auto-resume (recommended):
        inferencer = DevmateSDKInferencer(root_folder="/repo", auto_resume=True)
        r1 = inferencer.new_session("My number is 42")
        r2 = inferencer.infer("What is my number?")  # Auto-resumes!

        # Sync streaming:
        for chunk in inferencer.infer_streaming("Explain this"):
            print(chunk, end="", flush=True)

        # Async streaming:
        async for chunk in inferencer.ainfer_streaming("Explain this"):
            print(chunk, end="", flush=True)

    Attributes:
        root_folder: Working directory for Devmate agent.
        config_file_path: Path to config file or "freeform" for freeform mode.
        usecase: Usecase identifier for the SDK.
        model_name: Model to use (default: claude-sonnet-4-5).
        total_timeout_seconds: Maximum time for session (default 1800s / 30 min).
        idle_timeout_seconds: Maximum idle time between chunks (inherited, default 600s).
        config_vars: Additional configuration variables passed to SDK.
        auto_resume: If True, automatically resume previous session (default True).
    """

    # DevmateSDK-specific attributes
    # total_timeout_seconds overridden to 1800 (preserves current total timeout behavior)
    total_timeout_seconds: int = attrib(default=1800)
    # idle_timeout_seconds overridden to 2400 — Devmate sessions can be slower than Claude Code
    idle_timeout_seconds: int = attrib(default=2400)
    root_folder: Optional[str] = attrib(default=None)
    config_file_path: str = attrib(default="freeform")
    usecase: str = attrib(default="dual_agent_coding")
    model_name: str = attrib(default="claude-sonnet-4-5")
    config_vars: Dict[str, Any] = attrib(factory=dict)

    # Internal state
    _client: Any = attrib(default=None, init=False, repr=False)
    _last_token_count: int = attrib(default=0, init=False, repr=False)

    # === Streaming Primitive ===

    async def _produce_chunks(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield text deltas from Devmate SDK via event-driven model.

        Creates a fresh client per call. Registers event handlers that push
        text deltas to an asyncio.Queue, then yields from the queue.

        Args:
            prompt: The prompt string.
            **kwargs: Additional arguments:
                - previous_session_id: Session ID for continuation.

        Yields:
            Text deltas as they arrive from Devmate.
        """
        try:
            from devai.devmate_sdk.python.devmate_client import DevmateSDKClient
        except ImportError as e:
            raise RuntimeError(
                f"Devmate SDK not available: {e}. "
                "Ensure //devai/devmate_sdk/python:devmate_client is in deps."
            ) from e

        config_vars = {**self.config_vars, "prompt": prompt}
        if self.model_name:
            config_vars["model_name"] = self.model_name

        repo_root_path = Path(self.root_folder) if self.root_folder else None

        client = DevmateSDKClient(
            config_file_path=self.config_file_path,
            usecase=self.usecase,
            config_vars=config_vars,
            repo_root=repo_root_path,
        )

        # State for delta extraction (all local to this call)
        last_accumulated_text = ""
        last_action_id = None
        first_activity = False
        last_activity_time = time.monotonic()
        local_session_id = None

        # Queue for streaming chunks to the caller
        chunk_queue: asyncio.Queue[str | None] = asyncio.Queue()
        # Error holder — on_error populates this so the main loop can propagate
        error_holder: list[Exception] = []

        async def on_session(session: Any) -> None:
            nonlocal first_activity, last_activity_time, local_session_id

            local_session_id = getattr(session, "id", None)
            status = getattr(session, "status", None)
            status_name = getattr(status, "name", str(status)) if status else None

            self.log_info(
                f"session_id={local_session_id}, status={status_name}",
                "SessionEvent",
            )

            if status_name in ("PREPARING", "RUNNING"):
                last_activity_time = time.monotonic()
                if not first_activity:
                    first_activity = True

            if status_name in ("COMPLETED", "ERRORED"):
                await chunk_queue.put(None)

        async def on_action(action: Any) -> None:
            nonlocal last_accumulated_text, last_action_id, first_activity
            nonlocal last_activity_time

            last_activity_time = time.monotonic()
            if not first_activity:
                first_activity = True

            action_id = getattr(action, "id", None)
            if action_id != last_action_id:
                last_accumulated_text = ""
                last_action_id = action_id

            output = getattr(action, "output", None)

            # Log all available fields on the action for debugging
            action_attrs = {a: type(getattr(action, a, None)).__name__ for a in dir(action) if not a.startswith("_")}
            self.log_info(
                f"action_id={action_id}, output_type={type(output).__name__}, "
                f"output_attrs={[a for a in dir(output) if not a.startswith('_')] if output else None}, "
                f"action_attrs={action_attrs}",
                "ActionEvent",
            )

            if output is not None:
                info = getattr(output, "info", None)
                # Log all output fields to find where text actually lives
                output_fields = {}
                for attr in dir(output):
                    if not attr.startswith("_"):
                        val = getattr(output, attr, None)
                        if val is not None and not callable(val):
                            val_str = str(val)[:200]
                            output_fields[attr] = val_str
                self.log_info(
                    f"output_fields={output_fields}",
                    "ActionOutputFields",
                )

                if info:
                    current_text = str(info)
                    if current_text.startswith(last_accumulated_text):
                        delta = current_text[len(last_accumulated_text) :]
                    else:
                        delta = current_text
                    if delta:
                        await chunk_queue.put(delta)
                        self._last_token_count += 1
                    last_accumulated_text = current_text

        async def on_step(step: Any) -> None:
            nonlocal first_activity, last_activity_time

            last_activity_time = time.monotonic()
            if not first_activity:
                first_activity = True
            step_id = getattr(step, "id", None)
            step_number = getattr(step, "number", None)
            self.log_info(
                f"step_id={step_id}, step_number={step_number}",
                "StepEvent",
            )

        async def on_error(error: Any) -> None:
            self.log_info(
                f"error={error}, type={type(error).__name__}",
                "ErrorEvent",
            )
            error_holder.append(RuntimeError(f"Devmate SDK error: {error}"))
            await chunk_queue.put(None)

        # Register handlers
        client.events.on_session += on_session
        client.events.on_action += on_action
        client.events.on_step += on_step
        client.events.on_error += on_error

        previous_session_id = kwargs.get("previous_session_id")
        session_task = asyncio.create_task(
            client.start_session(previous_session_id=previous_session_id)
        )

        try:
            while True:
                # Race chunk_queue.get() against session_task failure.
                # If the bridge fails (e.g. BridgeStartupError), session_task
                # finishes with an exception while chunk_queue.get() blocks
                # forever because no events fire.  By using asyncio.wait we
                # detect the failure immediately and propagate it.
                get_task = asyncio.ensure_future(chunk_queue.get())
                done, _ = await asyncio.wait(
                    [get_task, session_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if get_task in done:
                    chunk = get_task.result()
                    if chunk is None:
                        break
                    yield chunk
                elif session_task in done:
                    # Bridge/session finished before a chunk arrived.
                    # Cancel the pending get and propagate any exception.
                    get_task.cancel()
                    # .result() re-raises BridgeStartupError or other errors
                    session_task.result()
                    # If no exception, session ended cleanly without sentinel
                    break
        finally:
            if not session_task.done():
                try:
                    await client.stop_session()
                except Exception:
                    pass
                session_task.cancel()

            self._session_id = local_session_id

            if error_holder:
                raise error_holder[0]

    # === Overrides ===

    async def _ainfer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Override for SDKInferencerResponse + session management kwargs.

        Resolves session kwargs (new_session, session_id, auto_resume) into
        ``previous_session_id`` and injects into kwargs BEFORE calling
        ``super()._ainfer()``, so the value flows through
        ``ainfer_streaming() → _produce_chunks(prompt, **kwargs) →
        client.start_session(previous_session_id=...)``.

        Args:
            inference_input: Input for inference.
            inference_config: Optional configuration (unused).
            **kwargs: Additional arguments:
                - return_sdk_response: If True, return SDKInferencerResponse.
                - session_id: Session ID to resume.
                - new_session: If True, forces a new session.

        Returns:
            Response text string, or SDKInferencerResponse if return_sdk_response=True.
        """
        # Resolve session logic
        new_session = kwargs.pop("new_session", False)
        explicit_session_id = kwargs.pop("session_id", None)
        if new_session:
            kwargs["previous_session_id"] = None
            logger.debug("Starting new session (new_session=True)")
        elif explicit_session_id:
            kwargs["previous_session_id"] = explicit_session_id
            logger.debug(
                "Resuming explicit session: %s",
                explicit_session_id[:8] if explicit_session_id else None,
            )
        elif self.auto_resume and self._session_id:
            kwargs["previous_session_id"] = self._session_id
            logger.debug(
                "Auto-resuming previous session: %s",
                self._session_id[:8] if self._session_id else None,
            )
        else:
            kwargs["previous_session_id"] = None
            logger.debug("Starting fresh session (no previous session)")

        self._last_token_count = 0
        response_text = await super()._ainfer(
            inference_input, inference_config, **kwargs
        )
        if kwargs.get("return_sdk_response", False):
            return SDKInferencerResponse(
                content=response_text,
                session_id=self._session_id,
                tokens_received=self._last_token_count,
            )
        return response_text

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Sync wrapper for async _ainfer().

        Creates a new event loop for each call via asyncio.run().
        This is safe for Devmate SDK since it creates a fresh client per query.

        Args:
            inference_input: Input for inference.
            inference_config: Optional configuration.
            **kwargs: Additional arguments passed to _ainfer().

        Returns:
            Response text string, or SDKInferencerResponse if return_sdk_response=True.
        """
        from rich_python_utils.common_utils.async_function_helper import _run_async

        return _run_async(self._ainfer(inference_input, inference_config, **kwargs))

"""Streaming Inferencer Base.

Extracts common streaming, session management, timeout handling, and
sync-to-async bridging logic shared by ClaudeCodeInferencer,
DevmateSDKInferencer, and DevmateCliInferencer.

Subclasses implement ``_produce_chunks()`` — the single abstract primitive
that yields raw text chunks from the backend. All other streaming/inference
methods derive from this.
"""

import asyncio
import hashlib
import logging
import os
import queue
import threading
import uuid
from abc import abstractmethod
from datetime import datetime
from typing import Any, AsyncIterator, Iterator, Optional

from attr import attrib, attrs
from agent_foundation.common.inferencers.inferencer_base import (
    InferencerBase,
)

logger = logging.getLogger(__name__)


@attrs
class StreamingInferencerBase(InferencerBase):
    """Base class for streaming inferencers with idle timeout, caching, and session management.

    Provides:
    - ``ainfer_streaming()`` — async streaming with per-chunk idle timeout + cache
    - ``infer_streaming()`` — sync bridge via thread + queue
    - ``_ainfer()`` — accumulates from ``ainfer_streaming()``
    - Session management: ``new_session``, ``anew_session``, ``resume_session``, ``aresume_session``
    - Cache persistence: optional ``cache_folder`` for writing intermediate output

    Subclasses must implement ``_produce_chunks(prompt, **kwargs)`` which yields
    raw text chunks from the backend.

    Timeout architecture (two layers):
        ``ainfer() → _ainfer_single() [total_timeout_seconds — caps entire operation]``
          ``→ _ainfer() [accumulates from ainfer_streaming]``
            ``→ ainfer_streaming() [idle_timeout_seconds — gaps between chunks]``
              ``→ _produce_chunks() [abstract, subclass implements]``

    Attributes:
        cache_folder: Directory for persisting intermediate streamed content.
            None (default) disables caching.
        idle_timeout_seconds: Maximum seconds to wait for the next chunk before
            considering the stream stalled. 0 disables idle timeout. Default: 600.
        auto_resume: If True, automatically resume previous session on subsequent
            infer calls. Default: True.
    """

    # Streaming configuration
    cache_folder: Optional[str] = attrib(default=None)
    idle_timeout_seconds: int = attrib(default=600)

    # Session management
    auto_resume: bool = attrib(default=True)

    # Internal state (not init params)
    _session_id: Optional[str] = attrib(default=None, init=False, repr=False)

    # === Properties ===

    @property
    def active_session_id(self) -> Optional[str]:
        """Get the current active session ID for resumption."""
        return self._session_id

    @active_session_id.setter
    def active_session_id(self, value: Optional[str]) -> None:
        self._session_id = value

    # === Abstract Method ===

    @abstractmethod
    async def _produce_chunks(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield raw text chunks from the backend.

        This is the single abstract primitive that each subclass must implement.
        All other streaming/inference methods derive from this.

        Subclasses handle their own:
        - Connection management (lazy connect, per-call client, subprocess)
        - Backend-specific message parsing (SDK message types, event handlers, stdout lines)
        - Session ID extraction (update self._session_id when received)

        Args:
            prompt: The extracted prompt string.
            **kwargs: Backend-specific arguments (session_id, new_session, etc.)

        Yields:
            Text chunks as they arrive from the backend.
        """
        raise NotImplementedError
        # Make this an async generator so type checkers are happy
        yield  # pragma: no cover

    # === Concrete Methods ===

    def _extract_prompt(self, inference_input: Any) -> str:
        """Extract prompt string from various input formats.

        Args:
            inference_input: String, or dict with "prompt" key.

        Returns:
            The prompt string.
        """
        if isinstance(inference_input, dict):
            return inference_input.get("prompt", str(inference_input))
        return str(inference_input)

    async def ainfer_streaming(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Async streaming inference with idle timeout and optional caching.

        Wraps ``_produce_chunks()`` to add:
        1. Idle timeout — if no chunk arrives within ``idle_timeout_seconds``, stops.
        2. Cache writing — if ``cache_folder`` is set, chunks are appended to a file.

        Args:
            inference_input: Input for inference (string or dict with "prompt" key).
            inference_config: Optional configuration (unused by base).
            **kwargs: Passed through to ``_produce_chunks()``.

        Yields:
            Text chunks as they arrive from the backend.
        """
        prompt = self._extract_prompt(inference_input)

        # Open cache file if configured
        cache_file = None
        if self.cache_folder:
            cache_file = self._open_cache_file(prompt)

        idle_timeout = (
            self.idle_timeout_seconds if self.idle_timeout_seconds > 0 else None
        )

        success = False
        error = None
        try:
            aiter = self._produce_chunks(prompt, **kwargs).__aiter__()
            while True:
                try:
                    if idle_timeout is not None:
                        chunk = await asyncio.wait_for(
                            aiter.__anext__(), timeout=idle_timeout
                        )
                    else:
                        chunk = await aiter.__anext__()
                except StopAsyncIteration:
                    break

                self._append_to_cache(cache_file, chunk)
                yield chunk

            success = True
        except asyncio.TimeoutError as e:
            error = e
            self.log_info(
                f"No new chunk for {self.idle_timeout_seconds}s",
                "IdleTimeout",
            )
        except Exception as e:
            error = e
            raise
        finally:
            self._finalize_cache(cache_file, success, error)

    async def _ainfer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Async inference by accumulating all streaming chunks.

        Total timeout is handled by ``InferencerBase._ainfer_single()``.
        Subclasses that need ``SDKInferencerResponse`` should override this,
        call ``super()._ainfer()``, and wrap the result.

        Args:
            inference_input: Input for inference.
            inference_config: Optional configuration.
            **kwargs: Passed through to ``ainfer_streaming()``.

        Returns:
            Concatenated response text.
        """
        content_parts: list[str] = []
        async for chunk in self.ainfer_streaming(
            inference_input, inference_config, **kwargs
        ):
            content_parts.append(chunk)
        return "".join(content_parts)

    def infer_streaming(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Iterator[str]:
        """Sync streaming inference via thread + queue bridge.

        Runs ``ainfer_streaming()`` in a background thread and yields chunks
        as they arrive.

        Args:
            inference_input: Input for inference.
            inference_config: Optional configuration.
            **kwargs: Passed through to ``ainfer_streaming()``.

        Yields:
            Text chunks as they arrive from the backend.
        """
        chunk_queue: queue.Queue[str | None] = queue.Queue()
        error_container: list[Exception] = []

        async def _run_async_streaming() -> None:
            try:
                async for chunk in self.ainfer_streaming(
                    inference_input, inference_config, **kwargs
                ):
                    chunk_queue.put(chunk)
            except Exception as e:
                error_container.append(e)
            finally:
                chunk_queue.put(None)

        def _run_in_thread() -> None:
            try:
                asyncio.run(_run_async_streaming())
            except Exception as e:
                error_container.append(e)
                chunk_queue.put(None)

        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()

        while True:
            chunk = chunk_queue.get()
            if chunk is None:
                break
            yield chunk

        thread.join(timeout=5.0)

        if error_container:
            raise error_container[0]

    # === Session Management Methods ===

    def new_session(self, prompt: str, **kwargs: Any) -> Any:
        """Start a new session, clearing any previous session.

        Args:
            prompt: The prompt to send.
            **kwargs: Additional arguments passed to ``infer()``.

        Returns:
            Inference result.
        """
        self._session_id = None
        return self.infer(prompt, new_session=True, **kwargs)

    async def anew_session(self, prompt: str, **kwargs: Any) -> Any:
        """Async: start a new session, clearing any previous session.

        Args:
            prompt: The prompt to send.
            **kwargs: Additional arguments passed to ``ainfer()``.

        Returns:
            Inference result.
        """
        await self.adisconnect()
        self._session_id = None
        return await self.ainfer(prompt, new_session=True, **kwargs)

    def resume_session(
        self, prompt: str, session_id: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Resume a previous session.

        Args:
            prompt: The follow-up prompt.
            session_id: Session ID to resume. If None, uses ``active_session_id``.
            **kwargs: Additional arguments passed to ``infer()``.

        Returns:
            Inference result.

        Raises:
            ValueError: If no session_id provided and no active session.
        """
        target = session_id or self._session_id
        if not target:
            raise ValueError(
                "No session_id provided and no active session. "
                "Call infer() first to start a session, or provide session_id."
            )
        return self.infer(prompt, session_id=target, **kwargs)

    async def aresume_session(
        self, prompt: str, session_id: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Async: resume a previous session.

        If a different ``session_id`` is provided than the current active session,
        this will disconnect and reconnect to the specified session.

        Args:
            prompt: The follow-up prompt.
            session_id: Session ID to resume. If None, uses ``active_session_id``.
            **kwargs: Additional arguments passed to ``ainfer()``.

        Returns:
            Inference result.

        Raises:
            ValueError: If no session_id provided and no active session.
        """
        target = session_id or self._session_id
        if not target:
            raise ValueError(
                "No session_id provided and no active session. "
                "Call ainfer() first to start a session, or provide session_id."
            )
        if self._session_id != target:
            await self.adisconnect()
        return await self.ainfer(prompt, session_id=target, **kwargs)

    # === Cache Helper Methods ===

    def _open_cache_file(self, prompt: str) -> Any:
        """Open a cache file for writing streaming output.

        Creates the directory structure:
          ``cache_folder/{ClassName}/{id}_{timestamp}/stream_{timestamp}_{hash}.txt``

        Args:
            prompt: The prompt string (hashed for filename).

        Returns:
            An open file handle (caller must close via ``_finalize_cache``).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        session_dir = os.path.join(
            self.cache_folder,  # pyre-ignore[6]
            self.__class__.__name__,
            f"{self.id}_{timestamp}",
        )
        os.makedirs(session_dir, exist_ok=True)
        unique_id = uuid.uuid4().hex[:8]
        cache_path = os.path.join(session_dir, f"stream_{unique_id}_{prompt_hash}.txt")
        self.log_debug(f"Cache file: {cache_path}", "CacheOpen")
        return open(cache_path, "w", encoding="utf-8")

    def _append_to_cache(self, cache_file: Any, chunk: str) -> None:
        """Append a chunk to the cache file, flush immediately."""
        if cache_file:
            cache_file.write(chunk)
            cache_file.flush()

    def _finalize_cache(
        self, cache_file: Any, success: bool, error: Exception | None = None
    ) -> None:
        """Write final status marker and close the cache file."""
        if cache_file:
            if success:
                cache_file.write("\n--- STREAM COMPLETED SUCCESSFULLY ---\n")
            else:
                msg = str(error) if error else "unknown"
                cache_file.write(f"\n--- STREAM FAILED: {msg} ---\n")
            cache_file.close()

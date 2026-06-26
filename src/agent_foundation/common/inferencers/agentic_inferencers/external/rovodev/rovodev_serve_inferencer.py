"""Rovo Dev Serve-Mode Inferencer.

Manages a background ``acli rovodev serve <port>`` process and communicates
via HTTP REST API + Server-Sent Events (SSE) for streaming inference.

Usage::

    async with RovoDevServeInferencer(target_path="/path/to/repo") as inf:
        # Streaming
        async for chunk in inf.ainfer_streaming("Explain this codebase"):
            print(chunk, end="")

        # Multi-turn with session reset
        r1 = await inf.ainfer("My number is 42", new_session=True)
        r2 = await inf.ainfer("What was my number?")
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Any, AsyncIterator, Optional

import httpx
from attr import attrib, attrs

from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.common import (
    ACLI_BINARY,
    RovoDevNotFoundError,
    find_acli_binary,
    find_available_port,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)

logger = logging.getLogger(__name__)


@attrs
class RovoDevServeInferencer(StreamingInferencerBase):
    """Rovo Dev inferencer using ``acli rovodev serve`` for streaming HTTP/SSE.

    Starts a local FastAPI server via ``acli rovodev serve <port>`` and
    communicates via REST API endpoints:

    - ``POST /v3/set_chat_message`` — send a prompt
    - ``GET /v3/stream_chat`` — receive SSE streaming response
    - ``POST /v3/reset`` — reset session
    - ``GET /healthcheck`` — server health

    Attributes:
        acli_path: Path to acli binary (auto-detected if None).
        target_path: Workspace directory for the agent (inherited from
            ``InferencerBase``). Read at call time via ``effective_cwd``
            so orchestrator-spawned children pick up ``workspace.root``
            when target_path is None.
        config_file: Path to rovodev config file.
        site_url: Atlassian site URL for billing.
        port: Serve port (auto-selected if None).
        disable_session_token: Disable auth on serve API.
startup_timeout: Max seconds to wait for server startup.
        non_interactive: Non-interactive mode.
        respect_configured_permissions: Respect config file permissions.
        agent_mode: Agent mode ("ask", "plan", or "default").
    """

    # RovoDevServe spawns ``acli rovodev serve`` which exposes file-edit /
    # write tools, so it HAS local file access. Override ``InferencerBase``'s
    # False default (inferencer_base.py:117) so the template feed and
    # ``{% if has_local_access %}`` gates render correctly. Mirrors
    # ``RovoDevCliInferencer.has_local_access=True``.
    has_local_access: bool = attrib(default=True)

    # ---- Configuration attributes ----
    acli_path: Optional[str] = attrib(default=None)
    config_file: Optional[str] = attrib(default=None)
    site_url: Optional[str] = attrib(default=None)
    port: Optional[int] = attrib(default=None)
    disable_session_token: bool = attrib(default=True)
    startup_timeout: int = attrib(default=60)
    non_interactive: bool = attrib(default=True)
    respect_configured_permissions: bool = attrib(default=False)
    agent_mode: Optional[str] = attrib(default=None)

    # ---- Internal state (not user-facing) ----
    # M6 (Tier-3): the live server subprocess + httpx client + base URL are
    # connection-scoped handles. They are compat-properties (below) backed by
    # ``_<name>_backing`` and mirrored into ``ctx.handles`` so concurrent fan-out
    # branches each own their own subprocess/client (V8 isolation), staying
    # byte-identical without an active context.

    @property
    def _server_process(self) -> Optional[asyncio.subprocess.Process]:
        return self._tier3_get("server_process", None)

    @_server_process.setter
    def _server_process(self, value) -> None:
        self._tier3_set("server_process", value)

    @property
    def _base_url(self) -> str:
        return self._tier3_get("base_url", "")

    @_base_url.setter
    def _base_url(self, value) -> None:
        self._tier3_set("base_url", value)

    @property
    def _http_client(self) -> Optional[httpx.AsyncClient]:
        return self._tier3_get("http_client", None)

    @_http_client.setter
    def _http_client(self, value) -> None:
        self._tier3_set("http_client", value)

    def __attrs_post_init__(self) -> None:
        if self.acli_path is None:
            import shutil
            self.acli_path = shutil.which(ACLI_BINARY)
        super().__attrs_post_init__()

    # =========================================================================
    # Connection lifecycle
    # =========================================================================

    async def aconnect(self, **kwargs: Any) -> None:
        """Start the ``acli rovodev serve`` server.

        1. Find or validate the acli binary.
        2. Select an available port.
        3. Start the subprocess.
        4. Poll ``/healthcheck`` until ready or timeout.
        """
        if self._server_process is not None:
            logger.debug("Already connected, skipping aconnect")
            return

        acli = find_acli_binary(self.acli_path)
        selected_port = self.port or find_available_port()

        cmd = [acli, "rovodev", "serve", str(selected_port)]
        if self.disable_session_token:
            cmd.append("--disable-session-token")
        if self.config_file:
            cmd.extend(["--config-file", self.config_file])
        if self.site_url:
            cmd.extend(["--site-url", self.site_url])
        if self.non_interactive:
            cmd.append("--non-interactive")
        if self.respect_configured_permissions:
            cmd.append("--respect-configured-permissions")
        if self.agent_mode:
            cmd.extend(["--agent-mode", self.agent_mode])

        logger.info("Starting serve: %s", " ".join(cmd))

        # Strip ROVODEV_CLI to prevent nested session detection
        from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.common import (
            clean_env_for_subprocess,
        )
        env = clean_env_for_subprocess()

        self._server_process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.effective_cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._base_url = f"http://localhost:{selected_port}"
        self._http_client = httpx.AsyncClient(
            base_url=self._base_url, timeout=httpx.Timeout(300.0, connect=10.0)
        )

        # Poll healthcheck
        for i in range(self.startup_timeout):
            if self._server_process.returncode is not None:
                stderr = ""
                if self._server_process.stderr:
                    stderr = (await self._server_process.stderr.read()).decode(errors="replace")
                raise RovoDevNotFoundError(
                    f"Serve process exited during startup (rc={self._server_process.returncode}): {stderr[:500]}"
                )
            try:
                resp = await self._http_client.get("/healthcheck")
                if resp.status_code == 200:
                    logger.info("Server ready at %s (attempt %d)", self._base_url, i + 1)
                    return
            except (httpx.ConnectError, httpx.ReadError):
                pass
            await asyncio.sleep(1)

        raise TimeoutError(
            f"Server did not become ready within {self.startup_timeout}s at {self._base_url}"
        )

    async def adisconnect(self) -> None:
        """Stop the serve server gracefully — for EVERY connection-scoped branch
        (each ``ctx.child(slot)`` that started a subprocess during ``_ainfer``) plus
        the legacy backing. No context is active at this lifecycle boundary, so
        draining by stored path is what reclaims every subprocess (M6) — otherwise a
        branch's ``acli rovodev serve`` process is orphaned until instance GC."""
        for h in self._iter_live_handle_sets():
            http_client = h.get("http_client")
            if http_client:
                await http_client.aclose()
                h.set("http_client", None)

            server_process = h.get("server_process")
            if server_process is not None and server_process.returncode is None:
                logger.info("Stopping serve process (pid=%s)", server_process.pid)
                try:
                    server_process.send_signal(signal.SIGTERM)
                    try:
                        await asyncio.wait_for(server_process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        logger.warning("SIGTERM timeout, sending SIGKILL")
                        server_process.kill()
                        await server_process.wait()
                except ProcessLookupError:
                    pass

            h.set("server_process", None)
            h.set("base_url", "")

    @property
    def is_connected(self) -> bool:
        """Whether the serve server is running."""
        return (
            self._server_process is not None
            and self._server_process.returncode is None
            and self._http_client is not None
        )

    # =========================================================================
    # Async context manager
    # =========================================================================

    async def __aenter__(self) -> "RovoDevServeInferencer":
        await self.aconnect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.adisconnect()

    # =========================================================================
    # Core inference methods
    # =========================================================================

    async def _ainfer_streaming(
        self,
        inference_input: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream inference via SSE.

        1. Auto-connect if not connected.
        2. Handle ``new_session`` by resetting via ``POST /v3/reset``.
        3. ``POST /v3/set_chat_message`` with the prompt.
        4. ``GET /v3/stream_chat`` and yield SSE text chunks.

        Yields:
            Text chunks from the model response. Empty strings are yielded
            for tool call events as activity sentinels for the dual-timer.
        """
        # Handle new_session flag
        new_session = kwargs.pop("new_session", False)
        if new_session and self._http_client:
            try:
                await self._http_client.post("/v3/reset")
                self.active_session_id = None
                logger.debug("Session reset via POST /v3/reset (streaming)")
            except Exception as e:
                logger.warning("Failed to reset session (streaming): %s", e)

        # Auto-connect if not connected
        if not self.is_connected:
            await self.aconnect()

        assert self._http_client is not None

        # Send the prompt
        prompt = self._extract_prompt(inference_input)
        await self._http_client.post(
            "/v3/set_chat_message", json={"message": prompt}
        )

        # Stream SSE response
        async with self._http_client.stream("GET", "/v3/stream_chat") as response:
            event_type = ""
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    event_type = ""
                    continue

                if line.startswith("event:"):
                    event_type = line[6:].strip()
                    continue

                if line.startswith("data:"):
                    data_str = line[5:].strip()

                    if event_type == "text_delta":
                        try:
                            import json
                            data = json.loads(data_str)
                            text = data.get("delta", data_str)
                        except (ValueError, KeyError):
                            text = data_str
                        yield text
                        self.active_session_id = "active"

                    elif event_type in ("tool_call_start", "tool_result"):
                        yield ""  # Activity sentinel for dual-timer

                    elif event_type == "agent_run_end":
                        return

    async def _ainfer(
        self,
        inference_input: Any,
        inference_config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Full async inference -- accumulates streaming output.

        Handles ``new_session`` by calling ``POST /v3/reset`` before inference.

        Args:
            inference_input: The prompt or input.
            inference_config: Optional configuration.
            **kwargs: Additional arguments (new_session, etc.).

        Returns:
            Accumulated response text.
        """
        new_session = kwargs.pop("new_session", False)
        if new_session:
            self.active_session_id = None
            if self._http_client:
                try:
                    await self._http_client.post("/v3/reset")
                    logger.debug("Session reset via POST /v3/reset")
                except Exception as e:
                    logger.warning("Failed to reset session: %s", e)

        # Accumulate from streaming
        result = await super()._ainfer(inference_input, inference_config, **kwargs)

        # Update active session
        if result:
            self.active_session_id = "active"

        return result

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **kwargs: Any
    ) -> Any:
        """Sync wrapper for async ``_ainfer()``.

        Uses ``_run_async()`` helper. Cannot be called from within an async
        context -- use ``_ainfer()`` or ``ainfer_streaming()`` instead.
        """
        from rich_python_utils.common_utils.async_function_helper import _run_async

        return _run_async(self._ainfer(inference_input, inference_config, **kwargs))

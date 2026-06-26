# pyre-strict

"""OpenClaw Inferencer — unified CLI + Gateway WebSocket inference.

Provides a single ``OpenClawInferencer`` class that supports two transport
modes selected via the ``mode`` attribute:

**``mode="gateway"`` (default)**
    Connects to the OpenClaw gateway via WebSocket (``ws://127.0.0.1:18789``).
    Supports **true token-by-token streaming** and **cross-run session restore**
    (gateway persists sessions to disk). Requires the OpenClaw gateway to be
    running (``./run.sh start``).

**``mode="cli"``**
    Executes ``openclaw agent --local --json`` inside the Docker/kubectl pod
    as a blocking subprocess. No real streaming (yields full response as one
    chunk), no cross-run session persistence. Always works if the Docker
    container is running.

Usage::

    # Gateway mode (default) — streaming + session restore
    inf = OpenClawInferencer()
    async for chunk in inf.ainfer_streaming("Summarize my Jira tickets"):
        print(chunk, end="", flush=True)

    # CLI mode — simple blocking
    inf = OpenClawInferencer(mode="cli")
    result = inf("say hello")
    print(result)

    # Multi-turn session (gateway mode — cross-run restore)
    inf1 = OpenClawInferencer(session_id="project-alpha")
    await inf1.ainfer("My project is Lighthouse")

    inf2 = OpenClawInferencer(session_id="project-alpha")  # new process
    result = await inf2.ainfer("What is my project called?")
    # → "Your project is called Lighthouse."
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import uuid
from typing import Any, AsyncIterator, Optional, Union

from attr import attrib, attrs

from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.common import (
    DEFAULT_DOCKER_CONTAINER,
    DEFAULT_GATEWAY_URL,
    DEFAULT_KUBECTL_CONTAINER,
    DEFAULT_KUBECTL_NAMESPACE,
    DEFAULT_KUBECTL_POD,
    DEFAULT_OPENCLAW_CONFIG_PATH,
    DEFAULT_OPENCLAW_STATE_DIR,
    DEFAULT_SESSION_ID,
    DEFAULT_TIMEOUT_SECONDS,
    GATEWAY_SCOPES,
    OpenClawAuthError,
    OpenClawError,
    OpenClawMode,
    OpenClawRateLimitError,
    OpenClawTimeoutError,
    is_rate_limit_error,
    parse_cli_json_output,
    read_gateway_token_from_pod,
    run_subprocess,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)

logger = logging.getLogger(__name__)


@attrs
class OpenClawInferencer(StreamingInferencerBase):
    """Unified OpenClaw inferencer supporting CLI and WebSocket gateway modes.

    Three transport modes are available via ``OpenClawMode``:

    - ``OpenClawMode.PodGateway``   — WebSocket streaming to pod gateway (default)
    - ``OpenClawMode.LocalGateway`` — WebSocket streaming to local native gateway
    - ``OpenClawMode.PodCLI``       — Docker/kubectl subprocess, blocking, always works

    See ``OpenClawMode`` for full documentation of each mode.

    Attributes:
        mode: Transport mode — ``OpenClawMode`` enum value (or string alias).
        docker_container: Docker container running the k3s cluster.
        kubectl_namespace: Kubernetes namespace of the OpenClaw pod.
        kubectl_pod: Name of the OpenClaw gateway pod.
        kubectl_container: Container name inside the pod for CLI exec.
        openclaw_config_path: Path to ``openclaw.json`` inside the pod.
        openclaw_state_dir: Path to the OpenClaw state dir inside the pod.
        gateway_url: WebSocket URL of the OpenClaw gateway (gateway mode).
        auth_token: Gateway auth token. Auto-discovered from pod if ``None``.
        session_id: Session ID (gateway: persisted; cli: in-memory per call).
        agent_id: Optional agent ID from openclaw config.
        thinking: Thinking level — ``"off"|"minimal"|"low"|"medium"|"high"|"xhigh"``.
        timeout_seconds: Inference timeout in seconds.
        deliver: Whether to deliver response to a channel (gateway mode).
        verbose: Verbose flag for CLI mode (``"on"`` or ``"off"``).
        extra_cli_args: Extra arguments appended to the CLI command.
        max_retries: Max retry attempts on rate-limit / timeout errors.
        retry_delay: Base delay (seconds) between retries (multiplied by attempt).
        retry_continuation_prompt: Template for retry prompts. Use
            ``{original_prompt}`` as placeholder.
        auto_resume: If ``True``, reuse ``active_session_id`` across calls.
    """

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode: Union[OpenClawMode, str] = attrib(default=OpenClawMode.PodGateway)

    # ── Docker / kubectl targeting ────────────────────────────────────────────
    docker_container: str = attrib(default=DEFAULT_DOCKER_CONTAINER)
    kubectl_namespace: str = attrib(default=DEFAULT_KUBECTL_NAMESPACE)
    kubectl_pod: str = attrib(default=DEFAULT_KUBECTL_POD)
    kubectl_container: str = attrib(default=DEFAULT_KUBECTL_CONTAINER)

    # ── OpenClaw config paths (CLI mode env vars / token discovery) ───────────
    openclaw_config_path: str = attrib(default=DEFAULT_OPENCLAW_CONFIG_PATH)
    openclaw_state_dir: str = attrib(default=DEFAULT_OPENCLAW_STATE_DIR)

    # ── Gateway connection ─────────────────────────────────────────────────────
    gateway_url: str = attrib(default=DEFAULT_GATEWAY_URL)
    auth_token: Optional[str] = attrib(default=None)
    # Optional model override — e.g. "claude-opus-4-6" or "claude-haiku-4-5-20251001".
    # When set, sends a sessions.patch WS request before each inference turn to
    # set the model on the session. If None, the gateway default is used.
    model: Optional[str] = attrib(default=None)

    # ── Agent params (both modes) ─────────────────────────────────────────────
    session_id: str = attrib(default=DEFAULT_SESSION_ID)
    agent_id: Optional[str] = attrib(default=None)
    thinking: Optional[str] = attrib(default=None)
    timeout_seconds: int = attrib(default=DEFAULT_TIMEOUT_SECONDS)
    deliver: bool = attrib(default=False)

    enable_turn_separation: bool = attrib(default=False)
    """Insert ``\\n\\n`` between agentic turns when streaming (gateway mode only).

    The OpenClaw gateway runs a multi-turn agentic loop where the agent may call
    tools and produce multiple intermediate text responses before the final answer.
    By default all turns are concatenated into one continuous stream (matching the
    raw gateway protocol — ``default=False``).

    When set to ``True``, a blank line (``"\\n\\n"``) is injected between turns,
    approximating the multi-bubble display in the OpenClaw Control UI.

    **How turn boundaries are detected:**
    The gateway sends ``stream="tool"`` events with ``data.phase="end"`` each time
    a tool call completes.  The next ``stream="assistant"`` text after a tool
    completion is a new turn.  The inferencer registers for tool events by
    advertising the ``"tool-events"`` capability in the WS connect handshake
    (``caps: ["tool-events"]``), which causes the gateway to call
    ``registerToolEventRecipient(runId, connId)`` in ``agent.ts`` and route
    ``stream="tool"`` events to this connection via ``broadcastToConnIds``.

    Has no effect in CLI mode (CLI produces a single blocking response).
    """

    # ── CLI-mode-only ─────────────────────────────────────────────────────────
    verbose: Optional[str] = attrib(default=None)
    extra_cli_args: Optional[list[str]] = attrib(default=None)

    # ── Retry config ──────────────────────────────────────────────────────────
    max_retries: int = attrib(default=3)
    retry_delay: float = attrib(default=8.0)
    retry_continuation_prompt: str = attrib(
        default="You were interrupted. Please continue or re-answer: {original_prompt}"
    )

    # ── Session management ────────────────────────────────────────────────────
    auto_resume: bool = attrib(default=True)

    always_initialize_new_session: bool = attrib(default=True)
    """Automatically call ``initialize_session()`` when a new session is detected.

    A session is considered *new* when no JSONL transcript file exists for
    the given ``session_id`` in the OpenClaw pod's sessions directory
    (``/sandbox/.openclaw/agents/main/sessions/<session_id>.jsonl``).

    When ``True`` (default), the inferencer sends a lightweight warm-up turn
    before the first real query on any new session.  This pre-runs the agent's
    startup sequence (reading ``AGENTS.md``, ``SOUL.md``, ``USER.md``, today's
    memory file) so the first real query gets the agent's full attention.

    On subsequent calls with the same ``session_id`` (transcript file already
    exists), the warm-up is skipped automatically regardless of this setting.

    Has no effect in CLI mode (no persistent session store).
    """

    # Track whether the session has been initialized this process lifetime.
    # Set to True after initialize_session() completes successfully so that
    # subsequent calls skip the warm-up turn even without a pod filesystem check.
    _session_initialized: bool = attrib(default=False, init=False)

    def __attrs_post_init__(self) -> None:
        """Normalize mode enum and auto-discover gateway auth token for PodGateway."""
        # Normalize string aliases to enum for backwards compatibility
        if isinstance(self.mode, str) and not isinstance(self.mode, OpenClawMode):
            _alias_map = {
                # Legacy two-value aliases
                "gateway": OpenClawMode.PodGateway,
                "cli": OpenClawMode.PodCLI,
                # Explicit enum string values
                "pod_gateway": OpenClawMode.PodGateway,
                "local_gateway": OpenClawMode.LocalGateway,
                "pod_cli": OpenClawMode.PodCLI,
            }
            normalized = _alias_map.get(self.mode)
            if normalized is None:
                raise ValueError(
                    f"Unknown mode {self.mode!r}. "
                    f"Use OpenClawMode.PodGateway, OpenClawMode.LocalGateway, "
                    f"or OpenClawMode.PodCLI."
                )
            self.mode = normalized

        # Auto-discover auth token from pod for PodGateway only.
        # LocalGateway reads token from local config (via from_local_config()).
        # PodCLI doesn't use a gateway auth token.
        if self.mode == OpenClawMode.PodGateway and self.auth_token is None:
            try:
                self.auth_token = read_gateway_token_from_pod(
                    docker_container=self.docker_container,
                    kubectl_namespace=self.kubectl_namespace,
                    kubectl_pod=self.kubectl_pod,
                    openclaw_config_path=self.openclaw_config_path,
                    kubectl_container=self.kubectl_container,
                )
                logger.debug(
                    "Auto-discovered gateway auth token: %s...",
                    self.auth_token[:8],
                )
            except Exception as e:
                logger.warning(
                    "Could not auto-discover gateway token: %s. "
                    "Set auth_token manually if needed.",
                    e,
                )
        super().__attrs_post_init__()

    # =========================================================================
    # Classmethod constructors
    # =========================================================================

    @classmethod
    def from_config(
        cls,
        openclaw_json_path: str = DEFAULT_OPENCLAW_CONFIG_PATH,
        **kwargs: Any,
    ) -> "OpenClawInferencer":
        """Create an inferencer by reading the auth token from a local file.

        Args:
            openclaw_json_path: Path to a local ``openclaw.json`` file.
            **kwargs: Additional attributes passed to the constructor.

        Returns:
            Configured ``OpenClawInferencer`` instance.
        """
        from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.common import (
            read_gateway_token_from_config,
        )

        token = read_gateway_token_from_config(openclaw_json_path)
        return cls(auth_token=token, **kwargs)

    @classmethod
    def from_local_config(
        cls,
        config_path: str = "~/.openclaw/openclaw.json",
        gateway_url: str = DEFAULT_GATEWAY_URL,
        auto_initialize: bool = False,
        **kwargs: Any,
    ) -> "OpenClawInferencer":
        """Create a gateway-mode inferencer from the LOCAL ~/.openclaw/openclaw.json.

        This is the simplest way to use the inferencer when the openclaw gateway
        is already running locally (``openclaw gateway`` or via the UI). It reads
        the gateway auth token directly from the local config file — no Docker or
        kubectl needed.

        **Session initialization (optional):**

        TWG and all other skills are available from the very first turn on any
        session — the gateway computes the skillsSnapshot fresh on every request.
        Calling ``initialize_session()`` (or ``auto_initialize=True``) is optional:
        it separates agent startup overhead (reading AGENTS.md, memory files, etc.)
        from the first real query, but does not affect skill availability.

        Args:
            config_path: Path to local openclaw.json (default ``~/.openclaw/openclaw.json``).
            gateway_url: WebSocket URL of the local gateway (default ``ws://127.0.0.1:18789``).
            auto_initialize: If ``True``, call ``ensure_session_initialized()``
                synchronously before returning — pre-warms the agent startup
                sequence so the first real query gets the agent's full attention.
            **kwargs: Additional attributes passed to the constructor.

        Returns:
            Configured ``OpenClawInferencer`` instance in gateway mode.

        Example::

            # Default — uses session_id="main" (UI session, already initialized):
            inf = OpenClawInferencer.from_local_config()
            result = await inf.ainfer("What should I follow up on today?")

            # Custom session ID — initialize first:
            inf = OpenClawInferencer.from_local_config(
                session_id="my-project",
                auto_initialize=True,  # warm-up turn writes skillsSnapshot
            )
            result = await inf.ainfer("Summarize my open PRs")

            # Or async:
            inf = OpenClawInferencer.from_local_config(session_id="my-project")
            await inf.initialize_session()
            result = await inf.ainfer("Summarize my open PRs")
        """
        import json
        from pathlib import Path

        resolved = Path(config_path).expanduser()
        try:
            cfg = json.loads(resolved.read_text(encoding="utf-8"))
            token = cfg["gateway"]["auth"]["token"]
        except (OSError, KeyError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Could not read gateway auth token from '{resolved}': {e}"
            ) from e

        instance = cls(
            mode=OpenClawMode.LocalGateway,
            gateway_url=gateway_url,
            auth_token=token,
            **kwargs,
        )

        if auto_initialize:
            instance.ensure_session_initialized()

        return instance

    # =========================================================================
    # CLI mode helpers
    # =========================================================================

    def _build_cli_cmd(self, prompt: str, session_id: str) -> str:
        """Build the full ``docker exec ... openclaw agent --local`` command.

        Args:
            prompt: The user prompt.
            session_id: Session ID to pass as ``--session-id``.

        Returns:
            Full shell command string ready for ``run_subprocess()``.
        """
        # Inject TWG skill env vars — openclaw --local doesn't auto-inject them
        from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.common import (
            read_skill_env_from_pod,
        )
        twg_env = read_skill_env_from_pod(
            skill_name="twg",
            docker_container=self.docker_container,
            kubectl_namespace=self.kubectl_namespace,
            kubectl_pod=self.kubectl_pod,
            kubectl_container=self.kubectl_container,
            openclaw_config_path=self.openclaw_config_path,
        )
        skill_env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in twg_env.items())
        if skill_env_prefix:
            skill_env_prefix += " "

        openclaw_cmd = (
            f"{skill_env_prefix}"
            f"OPENCLAW_CONFIG_PATH={self.openclaw_config_path} "
            f"OPENCLAW_STATE_DIR={self.openclaw_state_dir} "
            f"openclaw agent --local --json"
            f" --session-id {shlex.quote(session_id)}"
            f" --message {shlex.quote(prompt)}"
        )
        if self.agent_id:
            openclaw_cmd += f" --agent {shlex.quote(self.agent_id)}"
        if self.thinking:
            openclaw_cmd += f" --thinking {self.thinking}"
        if self.timeout_seconds:
            openclaw_cmd += f" --timeout {self.timeout_seconds}"
        if self.verbose:
            openclaw_cmd += f" --verbose {self.verbose}"
        if self.extra_cli_args:
            openclaw_cmd += " " + " ".join(self.extra_cli_args)

        return (
            f"docker exec {self.docker_container} "
            f"kubectl exec -n {self.kubectl_namespace} {self.kubectl_pod} "
            f"-c {self.kubectl_container} -- sh -c {shlex.quote(openclaw_cmd)}"
        )

    def _infer_cli(self, prompt: str, session_id: str) -> dict:  # type: ignore[type-arg]
        """Run the CLI command and parse its JSON output.

        Args:
            prompt: User prompt.
            session_id: Session ID.

        Returns:
            Parsed output dict from ``parse_cli_json_output()``.

        Raises:
            OpenClawRateLimitError: If rate-limit text is in stderr/output.
            OpenClawError: On subprocess failure.
        """
        cmd = self._build_cli_cmd(prompt, session_id)
        logger.debug("CLI cmd: %s", cmd[:120])
        stdout, stderr, rc = run_subprocess(cmd, timeout=self.timeout_seconds + 60)
        result = parse_cli_json_output(stdout, stderr, rc)

        # Check for rate limit in output/error
        combined = (result.get("output", "") + " " + (result.get("error") or "")).lower()
        if is_rate_limit_error(combined):
            raise OpenClawRateLimitError(
                f"Rate limit hit (CLI mode): {result.get('error') or result.get('output', '')[:200]}"
            )

        if rc != 0 and not result.get("output"):
            raise OpenClawError(
                f"openclaw CLI failed (rc={rc}): {stderr.strip()[:300]}"
            )

        return result

    # =========================================================================
    # Gateway mode helpers
    # =========================================================================

    async def _ws_connect(self) -> Any:  # websockets.WebSocketClientProtocol
        """Open a WebSocket connection and complete the OpenClaw handshake.

        Protocol (confirmed by live testing):
        1. Server sends ``connect.challenge`` event first.
        2. Client sends ``connect`` as a ``RequestFrame`` (``method="connect"``).
        3. Server responds with ``ResponseFrame`` containing ``hello-ok`` payload.

        Requires ``Origin`` header and ``openclaw-control-ui`` client ID to
        get full operator write scopes with token auth.

        Returns:
            Authenticated ``websockets.WebSocketClientProtocol`` instance.

        Raises:
            OpenClawAuthError: If the gateway rejects the connect request.
            OpenClawNotFoundError: If the WebSocket connection fails.
        """
        try:
            import websockets  # type: ignore[import]
        except ImportError as e:
            raise OpenClawError(
                "websockets package not installed. Run: pip install websockets"
            ) from e

        origin = (
            self.gateway_url
            .replace("wss://", "https://")
            .replace("ws://", "http://")
        )
        try:
            ws = await websockets.connect(
                self.gateway_url,
                max_size=25 * 1024 * 1024,
                additional_headers={"Origin": origin},
            )
        except Exception as e:
            from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.common import (
                OpenClawNotFoundError,
            )
            raise OpenClawNotFoundError(
                f"Cannot connect to OpenClaw gateway at {self.gateway_url}: {e}"
            ) from e

        # 1. Wait for server challenge
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
        except asyncio.TimeoutError as e:
            await ws.close()
            raise OpenClawError("Timed out waiting for connect.challenge") from e
        challenge = json.loads(raw)
        if challenge.get("event") != "connect.challenge":
            await ws.close()
            raise OpenClawError(
                f"Expected connect.challenge, got: {challenge.get('event')}"
            )

        # 2. Send connect RequestFrame
        connect_id = str(uuid.uuid4())
        await ws.send(json.dumps({
            "type": "req",
            "id": connect_id,
            "method": "connect",
            "params": {
                "minProtocol": 1,
                "maxProtocol": 10,
                "auth": {"token": self.auth_token},
                # Must use "openclaw-control-ui" + "ui" mode for write scope
                # with token auth. Plain "cli" mode strips write scopes
                # because the server requires RSA device identity for CLI auth.
                "client": {
                    "id": "openclaw-control-ui",
                    "version": "1.0.0",
                    "platform": "python",
                    "mode": "ui",
                },
                # "tool-events" cap causes the gateway to register this
                # connection as a tool-event recipient via
                # registerToolEventRecipient(runId, connId) in agent.ts.
                # This is required for enable_turn_separation=True to work —
                # without it, stream="tool" events are only sent to the
                # Control UI (broadcastToConnIds), not to general WS clients.
                "caps": ["tool-events"],
                "scopes": GATEWAY_SCOPES,
                "role": "operator",
            },
        }))

        # 3. Wait for hello-ok ResponseFrame
        deadline = asyncio.get_event_loop().time() + 10
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                await ws.close()
                raise OpenClawError("Timed out waiting for hello-ok after connect")
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except asyncio.TimeoutError as e:
                await ws.close()
                raise OpenClawError("Timed out waiting for hello-ok") from e
            frame = json.loads(raw)
            if frame.get("type") == "res" and frame.get("id") == connect_id:
                if not frame.get("ok"):
                    await ws.close()
                    raise OpenClawAuthError(
                        f"Gateway connect rejected: {frame.get('error')}"
                    )
                logger.debug(
                    "Gateway connected: protocol=%s connId=%s",
                    frame.get("payload", {}).get("protocol"),
                    str(frame.get("payload", {}).get("server", {}).get("connId", ""))[:12],
                )
                return ws
            # Skip other events (e.g. snapshot) until we get our res

    async def _set_session_model(self, session_id: str) -> None:
        """Send a sessions.patch WS request to set the model override on the session.

        This is how the Control UI changes the model — it sends ``sessions.patch``
        with ``{key, model}`` before the next agent turn. The gateway persists
        this in ``sessions.json`` and uses it for all subsequent turns on that
        session.

        Only called when ``self.model`` is set and mode is a gateway mode.

        Args:
            session_id: Session ID to patch (combined into the session key).
        """
        session_key = f"agent:main:{session_id}"
        req_id = str(uuid.uuid4())
        payload = json.dumps({
            "type": "req",
            "id": req_id,
            "method": "sessions.patch",
            "params": {
                "key": session_key,
                "model": self.model,
            },
        })
        try:
            ws = await self._ws_connect()
            try:
                await ws.send(payload)
                deadline = asyncio.get_event_loop().time() + 10
                while True:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                    try:
                        msg = json.loads(raw)
                        if msg.get("id") == req_id:
                            if msg.get("ok") is False:
                                logger.warning(
                                    "sessions.patch model=%s session=%s failed: %s",
                                    self.model, session_id, msg.get("error"),
                                )
                            else:
                                logger.debug(
                                    "sessions.patch model=%s session=%s → ok",
                                    self.model, session_id,
                                )
                            break
                    except Exception:
                        break
            finally:
                await ws.close()
        except Exception as exc:
            logger.warning(
                "sessions.patch model=%s failed (non-fatal, will use default model): %s",
                self.model, exc,
            )

    async def _stream_gateway(
        self, prompt: str, session_id: str
    ) -> AsyncIterator[str]:
        """Connect to the gateway and stream an agent response chunk by chunk.

        Yields incremental text from ``delta`` events. Delta events send
        **cumulative** text (not diff), so we track the accumulated length and
        yield only new characters.

        Args:
            prompt: User prompt.
            session_id: Session ID (gateway persists cross-run).

        Yields:
            Incremental text chunks as they arrive.

        Raises:
            OpenClawRateLimitError: On rate-limit / quota error.
            OpenClawTimeoutError: If no event arrives within timeout.
            OpenClawError: On unexpected agent error or aborted run.
        """
        # Set model override on the session before streaming if specified
        if self.model and self.mode in (OpenClawMode.PodGateway, OpenClawMode.LocalGateway):
            await self._set_session_model(session_id)

        ws = await self._ws_connect()
        req_id = str(uuid.uuid4())
        idempotency_key = str(uuid.uuid4())

        try:
            # Send agent request
            params: dict[str, Any] = {
                "message": prompt,
                "sessionId": session_id,
                "idempotencyKey": idempotency_key,
                "timeout": self.timeout_seconds,
                "deliver": self.deliver,
            }
            if self.agent_id:
                params["agentId"] = self.agent_id
            if self.thinking:
                params["thinking"] = self.thinking

            await ws.send(json.dumps({
                "type": "req",
                "id": req_id,
                "method": "agent",
                "params": params,
            }))

            accumulated = ""
            # True once any assistant text has been streamed via delta events.
            # Used to avoid re-yielding the full text from the final res frame.
            _has_streamed = False
            # Turn separation state: tracks whether the last tool call has
            # completed and the next assistant text starts a new turn.
            # Only used when self.enable_turn_separation is True.
            _pending_turn_sep = False
            # runId assigned by the gateway for our specific agent request.
            # Populated from the "accepted" ack frame. Used to filter out
            # events from other concurrent agent runs on the same session
            # (the gateway broadcasts all session events to all connected
            # clients, so parallel connections must filter by runId to avoid
            # receiving each other's streaming text).
            _our_run_id: str | None = None

            while True:
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=self.timeout_seconds + 10
                    )
                except asyncio.TimeoutError as e:
                    raise OpenClawTimeoutError(
                        f"No streaming event received within {self.timeout_seconds}s"
                    ) from e

                frame = json.loads(raw)
                ftype = frame.get("type")
                fevent = frame.get("event", "")

                # res frame for our request — two arrive:
                #   1. Ack:   status="accepted"  (no "ok" field, has "runId")
                #   2. Final: status="ok"         (has "result.payloads")
                if ftype == "res" and frame.get("id") == req_id:
                    payload = frame.get("payload", {})
                    status = payload.get("status", "")

                    if status == "accepted":
                        # First ack — capture our runId for event filtering.
                        # The gateway broadcasts ALL session events to ALL
                        # connected clients; filtering by runId prevents
                        # receiving text from other concurrent agent runs.
                        _our_run_id = payload.get("runId")
                        logger.debug(
                            "Agent ack: runId=%s",
                            str(_our_run_id or "")[:14],
                        )
                        continue

                    elif status == "ok":
                        # Final result — stream is complete.
                        # Only fall back to result.payloads text if no delta
                        # events were streamed (e.g. very short single-turn
                        # responses where the gateway skips streaming entirely).
                        # When _has_streamed is True, the full text was already
                        # yielded incrementally — do NOT re-yield from payloads.
                        if not _has_streamed:
                            result_payloads = payload.get("result", {}).get("payloads", [])
                            final_text = "\n".join(
                                p.get("text", "") for p in result_payloads if p.get("text")
                            ).strip()
                            if final_text:
                                yield final_text
                        break

                    else:
                        # Error or rejected
                        err = payload.get("error", {})
                        err_msg = (
                            err.get("message", str(err))
                            if isinstance(err, dict) else str(err)
                        ) or str(payload)
                        if is_rate_limit_error(err_msg):
                            raise OpenClawRateLimitError(
                                f"Rate limit hit (gateway): {err_msg}"
                            )
                        raise OpenClawError(f"Agent request rejected: {err_msg}")

                # Skip non-agent events (tick, health, snapshot, etc.)
                if ftype != "event" or fevent != "agent":
                    continue

                # Filter by runId to avoid processing events from other
                # concurrent agent runs on the same session.
                # The gateway broadcasts all session agent events to all
                # connected WS clients; without this filter, parallel
                # inferencer instances would receive each other's text.
                event_run_id = frame.get("payload", {}).get("runId", "")
                if _our_run_id and event_run_id and event_run_id != _our_run_id:
                    continue

                # Gateway streaming protocol (confirmed by live testing):
                #   stream="assistant" events carry incremental text:
                #     payload.data.delta  — new chars since last event
                #     payload.data.text   — cumulative text so far
                #   stream="lifecycle" events carry phase markers:
                #     payload.data.phase == "start"  — agent run beginning
                #     payload.data.phase == "end"    — agent run complete
                #   Final res frame (same req_id, status="ok"):
                #     payload.result.payloads[].text — full final text
                stream = frame.get("payload", {}).get("stream", "")
                data = frame.get("payload", {}).get("data", {})

                if stream == "assistant":
                    # Prefer delta (incremental) over recomputing from cumulative
                    delta: str = data.get("delta", "")
                    text_chunk = ""
                    if delta:
                        text_chunk = delta
                    else:
                        # Fallback: compute new chars from cumulative text
                        cumulative: str = data.get("text", "")
                        if cumulative:
                            new_part = (
                                cumulative[len(accumulated):]
                                if cumulative.startswith(accumulated)
                                else cumulative
                            )
                            accumulated = cumulative
                            text_chunk = new_part

                    if text_chunk:
                        # Inject turn separator before first char of a new turn
                        if self.enable_turn_separation and _pending_turn_sep:
                            yield "\n\n"
                            _pending_turn_sep = False
                        yield text_chunk
                        _has_streamed = True

                elif stream == "tool":
                    # Tool events signal the boundary between agentic turns.
                    # When a tool call ends, the next assistant text is a new turn.
                    # Used only for turn separation — no text to yield here.
                    if self.enable_turn_separation:
                        tool_phase: str = data.get("phase", "")
                        if tool_phase == "end":
                            # Mark that a turn separator should precede the next
                            # assistant text chunk (deferred so we don't emit
                            # a trailing separator if the run ends here).
                            _pending_turn_sep = True

                elif stream == "lifecycle":
                    phase = data.get("phase", "")
                    if phase == "end":
                        # Run complete — wait for final res frame below
                        pass
                    elif phase in ("error", "aborted"):
                        # Try multiple possible error message locations.
                        # data.error can be a string OR a dict depending on
                        # the error type (e.g. rate limit sends a string).
                        _err = data.get("error")
                        err_msg = (
                            data.get("errorMessage")
                            or (_err.get("message") if isinstance(_err, dict) else _err)
                            or data.get("message")
                            or data.get("reason")
                            or str(data)
                        ) or "unknown error"
                        logger.warning(
                            "OpenClaw agent %s — full data: %s", phase, data
                        )
                        if is_rate_limit_error(err_msg):
                            raise OpenClawRateLimitError(
                                f"Rate limit hit (gateway stream): {err_msg}"
                            )
                        raise OpenClawError(
                            f"OpenClaw agent {phase}: {err_msg}"
                        )

        finally:
            try:
                await ws.close()
            except Exception:
                pass

    async def _ainfer_gateway(
        self, prompt: str, session_id: str
    ) -> dict:  # type: ignore[type-arg]
        """Accumulate a full gateway-mode response (non-streaming).

        Args:
            prompt: User prompt.
            session_id: Session ID.

        Returns:
            Dict with ``output`` (accumulated text) and ``session_id``.
        """
        chunks: list[str] = []
        async for chunk in self._stream_gateway(prompt, session_id):
            chunks.append(chunk)
        return {
            "output": "".join(chunks),
            "session_id": session_id,
            "model": None,  # not available in gateway streaming events
            "usage": {},
            "success": True,
            "error": None,
        }

    # =========================================================================
    # Retry logic
    # =========================================================================

    async def _ainfer_with_retry(
        self, prompt: str, session_id: str
    ) -> dict:  # type: ignore[type-arg]
        """Run inference with exponential-backoff retry on rate-limit / timeout.

        On retry, uses ``retry_continuation_prompt`` with the original prompt
        so the model can continue from where it was interrupted.

        Args:
            prompt: Original user prompt.
            session_id: Session ID.

        Returns:
            Inference result dict.

        Raises:
            OpenClawRateLimitError: After all retries are exhausted.
        """
        original_prompt = prompt
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                if self.mode in (OpenClawMode.PodGateway, OpenClawMode.LocalGateway):
                    return await self._ainfer_gateway(prompt, session_id)
                else:
                    return self._infer_cli(prompt, session_id)
            except (OpenClawRateLimitError, OpenClawTimeoutError) as e:
                last_error = e
                if attempt == self.max_retries:
                    logger.warning(
                        "All %d retries exhausted: %s", self.max_retries, e
                    )
                    break
                wait = self.retry_delay * attempt
                logger.warning(
                    "Attempt %d/%d failed (%s). Retrying in %.0fs...",
                    attempt, self.max_retries, type(e).__name__, wait,
                )
                await asyncio.sleep(wait)
                # Use continuation prompt on retry
                prompt = self.retry_continuation_prompt.format(
                    original_prompt=original_prompt
                )

        raise OpenClawRateLimitError(
            f"OpenClaw inference failed after {self.max_retries} retries"
        ) from last_error

    # =========================================================================
    # StreamingInferencerBase abstract method implementations
    # =========================================================================

    async def _ainfer_streaming(
        self,
        inference_input: Any,
        inference_config: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield text chunks from the inference backend.

        Gateway mode: yields true token-by-token chunks.
        CLI mode: yields full response as a single chunk (no real streaming).

        Args:
            inference_input: Prompt string or structured input.
            inference_config: Ignored (config is done via attributes).
            **kwargs: Supports ``session_id`` override.

        Yields:
            Text chunks (non-empty strings).
        """
        prompt = self._extract_prompt(inference_input)
        session_id = kwargs.get("session_id", self.active_session_id or self.session_id)

        if self.mode in (OpenClawMode.PodGateway, OpenClawMode.LocalGateway):
            async for chunk in self._stream_gateway(prompt, session_id):
                yield chunk
        else:
            # CLI mode: blocking call, yield full response as one chunk
            result = self._infer_cli(prompt, session_id)
            output = result.get("output", "")
            if output:
                yield output

    # =========================================================================
    # Public inference methods with session management + retry
    # =========================================================================

    async def ainfer(
        self,
        inference_input: Any,
        inference_config: Any = None,
        *,
        run_context=None,
        **kwargs: Any,
    ) -> Any:
        """Async inference with session management and auto-retry.

        Args:
            inference_input: Prompt or structured input.
            inference_config: Ignored.
            run_context: optional RunContext carrier (M3); installs the bridge so
                ``active_session_id`` resolves to the connection-scoped handle.
            **kwargs: Supports ``session_id`` and ``new_session`` overrides.

        Returns:
            Accumulated response text string.
        """
        from agent_foundation.common.inferencers.run_context import (
            enter_run,
            exit_run,
        )

        _rc_token = enter_run(
            run_context, default_workspace=getattr(self, "_workspace", None)
        )
        try:
            new_session = kwargs.pop("new_session", False)
            if new_session:
                self.active_session_id = None

            session_id = kwargs.get(
                "session_id",
                self.active_session_id if self.auto_resume else self.session_id,
            ) or self.session_id

            prompt = self._extract_prompt(inference_input)

            # Auto-initialize new sessions when always_initialize_new_session=True
            await self._maybe_initialize_session(session_id)

            result = await self._ainfer_with_retry(prompt, session_id)

            # Update active session
            result_session = result.get("session_id") or session_id
            self.active_session_id = result_session

            return result.get("output", "")
        finally:
            exit_run(_rc_token)

    def infer(
        self,
        inference_input: Any,
        inference_config: Any = None,
        *,
        run_context=None,
        **kwargs: Any,
    ) -> Any:
        """Sync inference with session management and auto-retry.

        Args:
            inference_input: Prompt or structured input.
            inference_config: Ignored.
            run_context: optional RunContext carrier (M3) forwarded to ainfer().
            **kwargs: Supports ``session_id`` and ``new_session`` overrides.

        Returns:
            Accumulated response text string.
        """
        from rich_python_utils.common_utils.async_function_helper import _run_async

        return _run_async(
            self.ainfer(inference_input, inference_config, run_context=run_context, **kwargs)
        )

    async def ainfer_streaming(  # type: ignore[override]
        self,
        inference_input: Any,
        inference_config: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async streaming inference.

        Gateway mode: true token-by-token chunks.
        CLI mode: full response as a single chunk.

        Args:
            inference_input: Prompt or structured input.
            inference_config: Ignored.
            **kwargs: Supports ``session_id`` and ``new_session`` overrides.

        Yields:
            Text chunks.
        """
        new_session = kwargs.pop("new_session", False)
        if new_session:
            self.active_session_id = None

        session_id = kwargs.get(
            "session_id",
            self.active_session_id if self.auto_resume else self.session_id,
        ) or self.session_id

        prompt = self._extract_prompt(inference_input)

        # Auto-initialize new sessions when always_initialize_new_session=True
        await self._maybe_initialize_session(session_id)

        has_output = False

        if self.mode in (OpenClawMode.PodGateway, OpenClawMode.LocalGateway):
            async for chunk in self._stream_gateway(prompt, session_id):
                has_output = True
                yield chunk
        else:
            result = self._infer_cli(prompt, session_id)
            output = result.get("output", "")
            if output:
                has_output = True
                yield output

        if has_output:
            self.active_session_id = session_id

    # ── Session initialization ────────────────────────────────────────────────

    def _is_new_session(self, session_id: str) -> bool:
        """Check whether a session transcript file exists in the pod.

        A session is *new* (no prior conversation history) when the JSONL
        transcript file ``<session_id>.jsonl`` does not exist in the OpenClaw
        sessions directory inside the pod.

        This is the same signal the OpenClaw gateway uses internally
        (``hadSessionFile`` in ``commands/agent.ts``) to determine whether to
        run the agent's startup sequence for the first time.

        Args:
            session_id: Session ID to check.

        Returns:
            ``True`` if the session is new (no transcript), ``False`` if it
            has prior conversation history on disk.
        """
        if self.mode == OpenClawMode.PodCLI:
            return False  # PodCLI has no persistent sessions

        if self.mode == OpenClawMode.LocalGateway:
            # LocalGateway: session files are on the local filesystem
            from pathlib import Path
            local_sessions_dir = Path("~/.openclaw/agents/main/sessions").expanduser()
            session_file = local_sessions_dir / f"{session_id}.jsonl"
            is_new = not session_file.exists()
            logger.debug(
                "Session '%s' local transcript check: %s (is_new=%s)",
                session_id, session_file, is_new,
            )
            return is_new

        # PodGateway: session files are inside the Docker/kubectl pod
        sessions_dir = "/sandbox/.openclaw/agents/main/sessions"
        session_file_path = f"{sessions_dir}/{session_id}.jsonl"
        check_cmd = (
            f"docker exec {self.docker_container} "
            f"kubectl exec -n {self.kubectl_namespace} {self.kubectl_pod} "
            f"-c {self.kubectl_container} -- sh -c "
            f"'test -f {session_file_path} && echo EXISTS || echo NEW'"
        )
        try:
            stdout, stderr, rc = run_subprocess(check_cmd, timeout=10)
            result = (stdout + stderr).strip()
            is_new = "NEW" in result and "EXISTS" not in result
            logger.debug(
                "Session '%s' pod transcript check: %s (is_new=%s)",
                session_id, result, is_new,
            )
            return is_new
        except Exception as e:
            logger.warning(
                "Could not check session existence for '%s': %s. "
                "Assuming session exists (no warm-up).",
                session_id, e,
            )
            return False  # safe default: don't warm up if check fails

    async def _maybe_initialize_session(self, session_id: str) -> None:
        """Auto-initialize the session if ``always_initialize_new_session`` is set
        and the session is new (no transcript file on disk).

        Called automatically before the first inference call. Idempotent —
        skips if already initialized this process lifetime (``_session_initialized``).

        Args:
            session_id: Session ID being used for inference.
        """
        if not self.always_initialize_new_session:
            return
        if self._session_initialized:
            return
        if self._is_new_session(session_id):
            logger.debug(
                "New session detected for '%s' — running warm-up turn.", session_id
            )
            await self.initialize_session(session_id)
        else:
            # Session exists on disk — mark as initialized so we skip checks
            logger.debug(
                "Session '%s' already has transcript on disk — skipping warm-up.",
                session_id,
            )
            self._session_initialized = True

    async def initialize_session(
        self,
        session_id: Optional[str] = None,
        *,
        force: bool = False,
    ) -> None:
        """Warm up the OpenClaw session before the first real query.

        **What this does:**

        Sends a lightweight warm-up message so the agent completes its startup
        sequence (reading ``AGENTS.md``, ``SOUL.md``, ``USER.md``, today's memory
        file, etc.) *before* the first real query arrives.  This means the real
        query gets the agent's full attention without startup overhead mixed in.

        **Skills (TWG) are always available — no warm-up needed for that:**

        The OpenClaw gateway computes the ``skillsSnapshot`` (which lists all
        available skills including ``twg`` and ``agentic-search``) on *every*
        agent request, fresh from the filesystem.  TWG is available in the system
        prompt on the very first turn of any session — warm or cold.

        If the agent still doesn't use TWG for Atlassian queries it's due to
        non-deterministic tool selection, not missing skill injection.  The fix
        for that is query phrasing (explicitly mention Atlassian/Jira/Confluence
        to steer the agent toward TWG) rather than session initialization.

        **When to use this:**

        - You want to separate agent startup latency from first-query latency.
        - You're running benchmarks and want a "warmed" agent baseline.
        - You want to pre-establish ``active_session_id`` for ``auto_resume``.

        **Idempotent:** After the first successful call, subsequent calls are
        skipped unless ``force=True``.  The flag is process-local only.

        Args:
            session_id: Session ID to initialize.  Defaults to ``self.session_id``.
            force: Re-run even if already initialized this process lifetime.

        Example::

            inf = OpenClawInferencer.from_local_config()
            await inf.initialize_session()   # optional warm-up
            result = await inf.ainfer("What should I follow up on today?")
        """
        if self.mode not in (OpenClawMode.PodGateway, OpenClawMode.LocalGateway):
            # PodCLI has no persistent session store — nothing to initialize.
            return

        if self._session_initialized and not force:
            logger.debug("Session already initialized — skipping warm-up turn.")
            return

        sid = session_id or self.session_id
        logger.debug(
            "Initializing OpenClaw session '%s' to write skillsSnapshot...", sid
        )

        # Send a minimal warm-up message. The agent will respond briefly
        # (reading AGENTS.md, SOUL.md etc.) but we discard the output.
        # After this call, the gateway has written the skillsSnapshot for
        # "agent:main:<sid>" into sessions.json — all 11 skills including TWG.
        try:
            chunks: list[str] = []
            async for chunk in self._stream_gateway(
                "Session initialization. Respond with exactly one word: Ready.",
                sid,
            ):
                chunks.append(chunk)
            response = "".join(chunks)
            logger.debug(
                "Session '%s' initialized. Agent responded: %s",
                sid,
                response[:60],
            )
            self._session_initialized = True
            self.active_session_id = sid
        except Exception as e:
            logger.warning(
                "Session initialization failed for '%s': %s. "
                "Skills (TWG etc.) may not be in system prompt on first real query.",
                sid,
                e,
            )

    def ensure_session_initialized(self) -> None:
        """Sync wrapper for ``initialize_session()`` — blocks until complete.

        Useful when you need to initialize synchronously before first use.
        Call this once after construction if you cannot use ``await``.

        Example::

            inf = OpenClawInferencer.from_local_config()
            inf.ensure_session_initialized()   # sync warm-up
            result = inf.infer("What should I follow up on today?")
        """
        from rich_python_utils.common_utils.async_function_helper import _run_async
        _run_async(self.initialize_session())

    # ── Convenience session methods ───────────────────────────────────────────

    def new_session(self, inference_input: Any, *, run_context=None, **kwargs: Any) -> Any:
        """Start a new session and run inference. Clears session context.

        Args:
            inference_input: Prompt for the first message.
            run_context: optional RunContext carrier (M3) forwarded to infer().
            **kwargs: Forwarded to ``infer()``.

        Returns:
            Response text.
        """
        return self.infer(
            inference_input, new_session=True, run_context=run_context, **kwargs
        )

    async def anew_session(self, inference_input: Any, *, run_context=None, **kwargs: Any) -> Any:
        """Async version of ``new_session()``.

        Args:
            inference_input: Prompt for the first message.
            run_context: optional RunContext carrier (M3) forwarded to ainfer().
            **kwargs: Forwarded to ``ainfer()``.

        Returns:
            Response text.
        """
        return await self.ainfer(
            inference_input, new_session=True, run_context=run_context, **kwargs
        )

    # =========================================================================
    # InferencerBase abstract method implementations
    # =========================================================================

    def _infer(
        self,
        inference_input: Any,
        inference_config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Sync inference — delegates to ``infer()`` with session management.

        Required by ``InferencerBase``. The public ``infer()`` method is the
        preferred entry point; this exists to satisfy the abstract contract.
        """
        prompt = self._extract_prompt(inference_input)
        session_id = kwargs.get(
            "session_id",
            self.active_session_id if self.auto_resume else self.session_id,
        ) or self.session_id

        if self.mode == OpenClawMode.PodCLI:
            result = self._infer_cli(prompt, session_id)
            self.active_session_id = result.get("session_id") or session_id
            return result.get("output", "")

        # Gateway mode sync: run async via _run_async
        from rich_python_utils.common_utils.async_function_helper import _run_async
        return _run_async(self._ainfer_with_retry(prompt, session_id))

    async def _ainfer(
        self,
        inference_input: Any,
        inference_config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Async inference — delegates to ``_ainfer_with_retry()``."""
        prompt = self._extract_prompt(inference_input)
        session_id = kwargs.get(
            "session_id",
            self.active_session_id if self.auto_resume else self.session_id,
        ) or self.session_id
        result = await self._ainfer_with_retry(prompt, session_id)
        self.active_session_id = result.get("session_id") or session_id
        return result.get("output", "")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_prompt(self, inference_input: Any) -> str:  # type: ignore[override]
        """Extract prompt string from input (str, dict, or object).

        Args:
            inference_input: String prompt, dict with ``"prompt"`` key,
                or object with ``.prompt`` attribute.

        Returns:
            Prompt string.
        """
        if isinstance(inference_input, str):
            return inference_input
        if isinstance(inference_input, dict):
            return str(
                inference_input.get("prompt")
                or inference_input.get("message")
                or inference_input.get("input")
                or inference_input
            )
        if hasattr(inference_input, "prompt"):
            return str(inference_input.prompt)
        return str(inference_input)

"""Claude Code SDK Inferencer.

Wraps the Claude Code SDK client as an async-native StreamingInferencerBase implementation.
Provides both async interface (recommended) and sync bridge (for backwards compatibility).
"""

import asyncio
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.common import (
    build_permission_effort_kwargs,
    EffortLevel,
    PermissionModeLiteral,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.sdk_types import (
    SDKInferencerResponse,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from attr import attrib, attrs, validators

logger = logging.getLogger(__name__)


@attrs(slots=False)
class ClaudeCodeSdkInferencer(StreamingInferencerBase, TemplatedInferencerBase):
    """Claude Code SDK as an async-native streaming inferencer with session continuation.

    .. note:: **Multiple Inheritance Rationale**

       Inherits from BOTH ``StreamingInferencerBase`` (NDJSON-style streaming +
       session management) AND ``TemplatedInferencerBase`` (Jinja2 prompt
       rendering via ``template_key`` / ``template_root_space`` /
       ``template_variables``). Mirrors ``RovoChatInferencer`` and
       ``MetamateSDKInferencer`` (post-2026-05-26 fix).

       MRO: ``ClaudeCodeSdkInferencer → StreamingInferencerBase →
       TemplatedInferencerBase → InferencerBase``.

       Templates are rendered automatically by the framework's
       ``__ainfer_single_impl`` chain via ``self._render_prompt()``, which
       resolves via MRO to ``TemplatedInferencerBase._render_prompt()``. No
       explicit rendering call is needed in this class.

       ``slots=False`` matches ``TemplatedInferencerBase`` to avoid attrs
       layout conflicts under MI.

       Pre-2026-05-27 this class extended only ``StreamingInferencerBase``;
       YAML ``template_variables`` / ``template_root_space`` keys on a
       ``_target_: ClaudeCodeSDK`` leaf were silently dropped by
       ``_filter_attrs_keys`` (rich_python_utils logs a WARNING). Adding
       ``TemplatedInferencerBase`` here makes ClaudeCodeSdk a drop-in
       replacement for ClaudeCodeCli in templated BTA topologies.

    Inherits from StreamingInferencerBase which provides:
    - ``ainfer_streaming()`` with idle timeout and optional caching
    - ``infer_streaming()`` sync bridge via thread + queue
    - Session management: ``new_session``, ``anew_session``, ``resume_session``, ``aresume_session``
    - ``active_session_id`` property

    Inherits from TemplatedInferencerBase which provides:
    - ``_render_prompt()`` for Jinja2 template rendering
    - ``template_manager``, ``template_key``, ``template_root_space``,
      ``template_variables``, ``template_extra_feed``, ``template_version``,
      ``template_master_version``, ``modes`` attrs fields

    This class implements ``_ainfer_streaming()`` (the abstract primitive) and
    overrides ``_ainfer()`` to support ``SDKInferencerResponse``.

    Runtime Dependencies:
        Requires claude-agent-sdk package. This is a soft dependency — the
        module imports successfully without it. ImportError is raised only
        when aconnect() or _ainfer_streaming() is called.

    Usage Patterns:
        # Sync (simple, but pays connect cost each call):
        inferencer = ClaudeCodeSdkInferencer(target_path="/path/to/repo")
        result = inferencer("Write a hello world program")

        # Multi-turn with auto-resume (recommended):
        inferencer = ClaudeCodeSdkInferencer(target_path="/repo", auto_resume=True)
        r1 = inferencer.new_session("My number is 42")
        r2 = inferencer.infer("What is my number?")  # Auto-resumes!

        # Async with context manager (persistent connection):
        async with ClaudeCodeSdkInferencer(target_path="/repo") as inf:
            r1 = await inf.anew_session("My number is 42")
            r2 = await inf.ainfer("What is my number?")  # Auto-resumes!

        # Sync streaming:
        for chunk in inferencer.infer_streaming("Explain this"):
            print(chunk, end="", flush=True)

        # Async streaming:
        async for chunk in inferencer.ainfer_streaming("Explain this"):
            print(chunk, end="", flush=True)

    Connection and Session Scope:
        The live ``ClaudeSDKClient`` (with its disconnect function and event
        loop) and ``active_session_id`` are Tier-3 live handles. Under a host
        RunContext they are kept per branch, keyed by ``ctx.live_branch_key``
        = ``(handle scope of the ctx's root, ctx.path)``; bare calls (no host
        context) all share a single one. Calls on one branch reuse its connected
        client and so continue one Claude session. A call on a branch with no
        client — another path, or another root — connects a new client, which
        starts a new Claude session unless the call passes ``session_id``.
        Async calls leave their client connected until ``adisconnect()``,
        which closes every branch's client, or ``areset_conversation()``, which
        closes the branch's own; the sync bridge closes its client after each
        call.

        OpenStartup's ``ConversationService`` keeps one RunContext root per
        session and runs each turn under ``child("turn_N")``, and
        ``ConversationalInferencer`` calls its base under ``child("agent")``
        every round. By default the rounds of one turn share one client and
        one Claude session, every turn starts a new one, and earlier turns'
        clients stay connected until ``adisconnect()``. With
        ``fresh_vendor_session_per_round`` on (opt-in; default off), the
        branch's conversation is reset before every round, so every round
        connects a new client and starts a new Claude session.

        ``NativeConversationalInferencer`` (``conversational_native``) does
        not use this class: its ``claude_sdk`` backend drives its own
        ``ClaudeSDKClient`` and keeps one vendor session across turns in its
        durable session record. ``NativeBackendSpec.from_inferencer`` only
        reads configuration fields from a ``ClaudeCodeSDK`` definition.

    Attributes:
        target_path: Working directory for Claude Code agent (inherited
            from ``InferencerBase``). Used as the subprocess cwd via
            ``effective_cwd`` (which falls back to ``workspace.root`` and
            then ``os.getcwd()`` when ``target_path`` is None).
        system_prompt: System prompt to configure agent behavior.
        system_prompt_mode: How ``system_prompt`` relates to Claude Code's own
            system prompt; read when the client connects.

            - ``"legacy_empty"`` (default, the historical behavior): sent as
              the whole system prompt even when empty, so an empty or ``None``
              ``system_prompt`` becomes ``--system-prompt ""`` and leaves
              Claude Code with no system prompt at all.
            - ``"replace"``: a non-empty ``system_prompt`` replaces Claude
              Code's system prompt; an empty, whitespace-only or ``None`` one
              keeps Claude Code's own.
            - ``"preset_append"``: keeps Claude Code's system prompt and
              appends ``system_prompt`` (nothing when it is empty).

            ``ConversationalInferencer`` sets ``system_prompt = ""`` before
            every streaming round, so under it only ``"legacy_empty"`` removes
            Claude Code's system prompt.
        idle_timeout_seconds: Per-chunk idle timeout in seconds (inherited,
            overridden to 1800). If no new text chunk arrives within this
            duration, the stream is considered stalled.
        allowed_tools: List of tools Claude can use (default: Read, Write, Bash).
        include_partial_messages: Whether to include partial messages in stream.
        auto_resume: If True, automatically resume previous session (default True).
        prefer_subscription: If True (default), clear ANTHROPIC_API_KEY from
            the SDK subprocess so it uses the Claude subscription (Max/Pro)
            instead of pay-per-token API billing. The CLI ignores API keys
            automatically, but the SDK inherits the parent environment.
            Set to False to use API key billing when ANTHROPIC_API_KEY is set.
        sdk_env: Extra environment variables passed to the SDK subprocess.
            Merged after prefer_subscription filtering, so explicit entries
            here take precedence.
        permission_mode: Permission control mode (default: ``None``, i.e.
            let the SDK/CLI use its own default). One of ``"default"``,
            ``"acceptEdits"``, ``"plan"``, ``"auto"``, ``"dontAsk"``, or
            ``"bypassPermissions"``. Values inside the SDK's native Literal
            (``default`` / ``acceptEdits`` / ``plan`` / ``bypassPermissions``)
            are passed via ``ClaudeAgentOptions.permission_mode``; the wider
            CLI-only values (``auto`` / ``dontAsk``) are routed through
            ``ClaudeAgentOptions.extra_args`` so the SDK transport still
            forwards them to the CLI.
        effort: Reasoning-effort level (default: ``"max"``). One of ``"low"``,
            ``"medium"``, ``"high"``, ``"xhigh"``, ``"max"``, or ``None`` to
            omit the flag entirely (use the model's default). Higher levels
            allocate more thinking budget at the cost of latency and tokens.
            ``"xhigh"`` is routed via ``extra_args`` because the installed
            SDK's typed Literal is narrower than the CLI's accepted set.
        disable_osx_sandbox: Pass the Meta launcher's
            ``--dangerously-disable-osx-sandbox`` flag (via ``extra_args``).
            ``None`` (default) resolves from the
            ``CLAUDE_CODE_INFERANCER_NO_SANDBOX`` env var; an explicit
            ``True``/``False`` overrides it. Enable only when the SDK's
            ``claude`` subprocess must run nested inside an already-sandboxed
            process — macOS forbids nesting seatbelt sandboxes, so otherwise
            ``claude`` exits 71 with no output. SECURITY: removes OS-level
            confinement of the agent's file access.
    """

    # Call results live in the invocation, session state behind the session
    # policy and connections in Tier-3 handles; the purity ratchet verifies it.
    _HOST_PURE_CERTIFIED = True

    _FANOUT_SINGLE_CALL_ARGS = StreamingInferencerBase._FANOUT_SINGLE_CALL_ARGS + (
        "return_sdk_response",
    )

    # ClaudeCodeSdk launches the ``claude`` CLI as a subprocess with
    # Read / Write / Bash tools — it HAS local file access. Override
    # ``InferencerBase``'s False default (inferencer_base.py:117) so the
    # template feed exposes ``output_path`` / ``workspace_root`` /
    # ``workspace_outputs`` and BTA aggregators reference worker
    # artifacts by path instead of inlining. Mirrors
    # ``ClaudeCodeCliInferencer.has_local_access=True`` (claude_code_cli_inferencer.py:87),
    # ``RovoDevCliInferencer.has_local_access=True`` (rovodev_cli_inferencer.py:111),
    # and ``DevmateCliInferencer.has_local_access=True`` (devmate_cli_inferencer.py:183).
    has_local_access: bool = attrib(default=True)

    # ClaudeCode-specific attributes
    # idle_timeout_seconds overridden to 1800 (was timeout_seconds=1800 in old code)
    idle_timeout_seconds: int = attrib(default=1800)
    system_prompt: str = attrib(default="")
    system_prompt_mode: str = attrib(
        default="legacy_empty",
        validator=validators.in_(("legacy_empty", "replace", "preset_append")),
    )
    allowed_tools: List[str] = attrib(factory=lambda: ["Read", "Write", "Bash"])
    include_partial_messages: bool = attrib(default=True)
    prefer_subscription: bool = attrib(default=True)
    sdk_env: Dict[str, str] = attrib(factory=dict)
    permission_mode: Optional[PermissionModeLiteral] = attrib(default=None)
    effort: Optional[EffortLevel] = attrib(default="max")
    # Shell-tool gating. ``enable_shell=False`` filters "Bash" out of
    # ``allowed_tools``; if that leaves an empty list, raise — the SDK
    # treats ``allowed_tools=[]`` as allow-all, which is the opposite of
    # what the caller intended.
    enable_shell: bool = attrib(default=True)
    allowed_shell_commands: Optional[List[str]] = attrib(default=None)
    # Disable the Meta launcher's macOS seatbelt sandbox (routed to the CLI via
    # ``ClaudeAgentOptions.extra_args``). ``None`` (default) resolves from the
    # ``CLAUDE_CODE_INFERANCER_NO_SANDBOX`` env var in __attrs_post_init__; an
    # explicit ``True``/``False`` overrides it. Needed when the SDK's ``claude``
    # subprocess runs nested inside an already-sandboxed process (macOS forbids
    # nesting seatbelt sandboxes). See ``common.resolve_disable_osx_sandbox``.
    disable_osx_sandbox: Optional[bool] = attrib(default=None)

    # Internal state
    # M6 (Tier-3): the live SDK client + its disconnect fn + bound loop are
    # connection-scoped handles -> compat-properties backed by ``_<name>_backing``
    # and mirrored into ``ctx.handles`` for per-branch (V8) isolation. Byte-identical
    # without an active context.
    _connect_lock: Any = attrib(default=None, init=False, repr=False)

    @property
    def _client(self):
        return self._tier3_get("client", None)

    @_client.setter
    def _client(self, value):
        self._tier3_set("client", value)

    @property
    def _disconnect_fn(self):
        return self._tier3_get("disconnect_fn", None)

    @_disconnect_fn.setter
    def _disconnect_fn(self, value):
        self._tier3_set("disconnect_fn", value)

    @property
    def _connected_loop(self):
        return self._tier3_get("connected_loop", None)

    @_connected_loop.setter
    def _connected_loop(self, value):
        self._tier3_set("connected_loop", value)

    def __attrs_post_init__(self) -> None:
        """Enforce enable_shell on allowed_tools and emit shell-gating logs."""
        from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.common import (
            resolve_disable_osx_sandbox,
        )

        # Resolve the macOS-sandbox toggle to a concrete bool: explicit ctor
        # value wins; otherwise fall back to the env var.
        self.disable_osx_sandbox = resolve_disable_osx_sandbox(self.disable_osx_sandbox)

        if not self.enable_shell:
            filtered = [t for t in self.allowed_tools if t != "Bash"]
            if not filtered:
                raise ValueError(
                    "enable_shell=False with allowed_tools=['Bash'] would leave "
                    "allowed_tools=[], which the SDK treats as allow-all. "
                    "Either keep at least one non-Bash tool, or set enable_shell=True."
                )
            self.allowed_tools = filtered

        if self.allowed_shell_commands and not self.enable_shell:
            logger.warning(
                "enable_shell=False takes precedence over allowed_shell_commands=%s "
                "(shell tool will be disabled).",
                self.allowed_shell_commands,
            )
        elif self.allowed_shell_commands:
            logger.info(
                "ClaudeCodeSdkInferencer: allowed_shell_commands set to %s "
                "(informational — no equivalent SDK option).",
                self.allowed_shell_commands,
            )

        super().__attrs_post_init__()

    # === Option-routing helper (also exercised by unit tests) ===

    def _system_prompt_option(self) -> Any:
        """The SDK's ``system_prompt`` for ``system_prompt_mode``.

        The SDK transport sends a string as ``--system-prompt`` (``None`` as
        ``""``), replacing Claude Code's own prompt; the ``claude_code`` preset
        sends no ``--system-prompt``, plus ``--append-system-prompt`` when it
        carries ``append``.
        """
        if self.system_prompt_mode == "legacy_empty":
            return self.system_prompt
        has_text = bool((self.system_prompt or "").strip())
        if self.system_prompt_mode == "replace" and has_text:
            return self.system_prompt
        preset: Dict[str, Any] = {"type": "preset", "preset": "claude_code"}
        if self.system_prompt_mode == "preset_append" and has_text:
            preset["append"] = self.system_prompt
        return preset

    def _build_permission_effort_kwargs(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Optional[str]]]:
        """Split ``permission_mode`` and ``effort`` into SDK typed fields and ``extra_args``.

        The installed claude-agent-sdk's typed ``permission_mode`` and
        ``effort`` Literals are narrower than the CLI's accepted set
        (no ``auto`` / ``dontAsk`` for permission, no ``xhigh`` for effort).
        For type cleanliness and forward compatibility, values inside the
        SDK's native Literal go through the typed field; the wider CLI-only
        values go through ``extra_args`` so the SDK subprocess transport
        still forwards them to the CLI as ``--permission-mode`` / ``--effort``.

        Returns:
            ``(sdk_kwargs, extra_args)`` — ``sdk_kwargs`` is splatted into
            ``ClaudeAgentOptions(...)``; ``extra_args`` is passed as the
            ``extra_args`` field. Either may be empty.
        """
        return build_permission_effort_kwargs(
            self.permission_mode, self.effort, bool(self.disable_osx_sandbox)
        )

    # === Streaming Primitive ===

    async def _ainfer_streaming(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Yield text chunks from Claude Code SDK stream.

        No internal idle timeout — the base class ``ainfer_streaming()`` handles
        it with ``idle_timeout_seconds=1800``.

        Args:
            prompt: The prompt string.
            **kwargs: Additional arguments (session_id, new_session, etc.).

        Yields:
            Text chunks as they arrive from Claude.
        """
        stats = self._stream_stats()
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
                            stats.tool_uses += 1
                            self.log_info(
                                f"Tool use #{stats.tool_uses}: {block.name}",
                                "ToolUse",
                            )
                case ResultMessage() as result_msg:
                    # §2.10: mirror the live session into ctx.handles (V7), not just
                    # the instance. Byte-identical without a context.
                    self.active_session_id = result_msg.session_id
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
        stats = self._reset_stream_stats()
        response_text = await super()._ainfer(
            inference_input, inference_config, **kwargs
        )
        if kwargs.get("return_sdk_response", False):
            return SDKInferencerResponse(
                content=response_text,
                # read via the property so it resolves the live session from the
                # active context's connection branch (Tier-3), not the bare backing
                # (which a context-scoped write never touches).
                session_id=self.active_session_id,
                tool_uses=stats.tool_uses,
            )
        return response_text

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args: Any
    ) -> Any:
        """Sync bridge with stale-loop detection and self-contained sessions.

        CRITICAL: asyncio.run() closes the event loop after completion.
        ClaudeSDKClient holds persistent loop-bound state (subprocess,
        anyio task groups, background tasks) that becomes invalid when
        the loop closes. So each sync call closes the client it connected
        inside its own loop, before the loop ends; a client left behind by
        async use on a now-closed loop is detected and dropped.

        For multi-call usage, prefer the async interface:
            async with ClaudeCodeSdkInferencer(...) as inf:
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

        async def _run_and_close():
            try:
                return await self._ainfer(
                    inference_input, inference_config, **_inference_args
                )
            finally:
                # The sync bridge runs on a throwaway event loop and the client is
                # bound to it: close this call's client inside that loop, before it
                # ends. The session id survives for the next call's resume.
                await self._adisconnect_branch()

        return _run_async(_run_and_close())

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

        # Build subprocess env: prefer subscription by clearing API key
        env: Dict[str, str] = {}
        if self.prefer_subscription and os.environ.get("ANTHROPIC_API_KEY"):
            env["ANTHROPIC_API_KEY"] = ""
            logger.debug(
                "prefer_subscription=True: clearing ANTHROPIC_API_KEY for SDK subprocess"
            )
        env.update(self.sdk_env)  # explicit sdk_env takes precedence

        sdk_kwargs, extra_args = self._build_permission_effort_kwargs()

        options = ClaudeAgentOptions(
            model=self.model_id or None,
            cwd=str(self.effective_cwd),
            system_prompt=self._system_prompt_option(),
            include_partial_messages=self.include_partial_messages,
            allowed_tools=self.allowed_tools,
            resume=session_id,
            env=env,
            extra_args=extra_args,
            **sdk_kwargs,
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
        self.active_session_id = session_id  # §2.10: mirror into ctx.handles too
        self._connected_loop = loop
        logger.debug("Claude Code SDK connected (session_id=%s)", session_id)

    async def _adisconnect_branch(self) -> None:
        """Disconnect the active branch's client only: the one this call used.
        Other branches' clients may be bound to other loops."""
        fn = self._disconnect_fn
        if fn:
            await fn()
        self._disconnect_fn = None
        self._client = None
        self._connected_loop = None

    async def _areset_branch_conversation(self) -> None:
        """A connected client continues one Claude session, so the branch's own
        client is disconnected (not a sibling's, nor the client connected outside
        any context that branches without one share) and the branch's next call
        connects a new client, which starts a new session."""
        handles = self._tier3_own_handles()
        disconnect = handles.get("disconnect_fn")
        for name in ("disconnect_fn", "client", "connected_loop"):
            handles.set(name, None)
        self._tier3_detach_from_backing()
        await super()._areset_branch_conversation()
        if disconnect:
            await disconnect()

    async def adisconnect(self) -> None:
        """Disconnect from Claude Code SDK — EVERY connection-scoped branch (each
        ``ctx.child(slot)`` that established a client during ``_ainfer``) plus the
        legacy backing. No context is active at this lifecycle boundary, so draining
        by stored path (not just the active branch) is what reclaims them all (M6)."""
        for h in self._iter_live_handle_sets():
            fn = h.get("disconnect_fn")
            if fn:
                await fn()
            h.set("disconnect_fn", None)
            h.set("client", None)
            h.set("connected_loop", None)
        logger.debug("Claude Code SDK disconnected")

"""Metamate backend: a remote, tool-less vendor agent (guidance mode).

Metamate runs the conversation server-side through the Metamate 2.0 SDK
(``MetamateSDKClient.execute_and_stream`` with LLMVM orchestration). It cannot
call AgentFoundation tools, so SOPs are controlled with slash commands and
phases that need AF tools run on a tool-capable backend (SOP state carries
over on a backend switch).

Evidence (scripts/native_spikes/s16, s16b):

* Session instructions ride an inline agent config with
  ``system_prompt_mode="append"`` — followed like system instructions. Its
  legacy ``internal_prompt`` alternative is shown to the model as pasted text
  and distrusted, so it is not used.
* The append is fixed when the conversation is created (a changed append on a
  later request is ignored), like Claude Code's system-prompt snapshot; the
  per-turn context therefore travels as the labelled ``<af_context>`` envelope
  (the backend config opts in with ``l2_envelope_allowed``).
* A conversation resumes by ``conversation_id``; the SDK isolates the new
  turn's reply from earlier turns.
* Metamate sometimes stops answering a request server-side: no new output
  block ever arrives (reproduced with the SDK alone, no AgentFoundation code,
  on about 1 in 5 resumed requests). The SDK gives up only after its idle
  timeout, 300 s by default, so the backend passes ``extra.idle_timeout_s``
  (``DEFAULT_IDLE_TIMEOUT_S``) instead. The request may still finish
  server-side, so such a turn is ``uncertain`` and is never re-sent.

The SDK is Buck-only (``//msl/metamate/sdk:metamate_sdk``). This package is
its own Buck target (``session/metamate:metamate``, which depends on the SDK)
and the core AgentFoundation library excludes it, so that library carries no
``msl`` dependency: a host that runs this backend adds the target to its
binary. The SDK import stays lazy, so a source checkout without ``msl`` gets a
``NativeCapabilityError`` at open.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from typing import Any, AsyncIterator, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.errors import (
    NativeCapabilityError,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.events import (
    MessageEnd,
    SessionStarted,
    TurnEnd,
    VendorError,
    VendorEvent,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    BackendCapabilities,
    CallerTools,
    Evidence,
    InterruptNotAcknowledged,
    L2Channel,
    NativeBackendSpec,
    SessionOpenRequest,
    TurnRequest,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.common import (
    DEFAULT_API_KEY,
    DEFAULT_SURFACE,
)

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_ENTRY = "Unified Auto"
# Seconds without a new output block before the SDK gives up on a request.
# The SDK resets its timer on every block of the reply, tool calls included,
# and skips it while a Sandcastle job drives the conversation. Whole healthy
# turns took 4-29 s in the OpenStartup run; a stalled request never resumed.
DEFAULT_IDLE_TIMEOUT_S = 120.0
# Text of the SDK's idle-timeout failure (a ``MetamateSDKException``, the
# type of every SDK failure), raised only after the request was started.
_SDK_IDLE_FAILURE = "No new output for"
_AGENT_NAME = "agentfoundation-host"
_AGENT_DESCRIPTION = (
    "Conversation host running AgentFoundation standard operating procedures."
)
_SDK_TARGET = "//msl/metamate/sdk:metamate_sdk"


def _sdk() -> tuple[Any, Any]:
    """``(MetamateSDKClient, types module)``; a clear capability error when the
    Buck-only SDK is not part of this binary."""
    try:
        from msl.metamate.sdk import types as sdk_types  # @manual
        from msl.metamate.sdk.client import MetamateSDKClient  # @manual
    except ImportError as exc:
        raise NativeCapabilityError(
            f"The Metamate backend needs the Metamate SDK ({_SDK_TARGET}) in this "
            "binary; it is only available through Buck."
        ) from exc
    return MetamateSDKClient, sdk_types


def _idle_timeout_s(extra: dict[str, Any]) -> float:
    value = extra.get("idle_timeout_s")
    if value is None:
        return DEFAULT_IDLE_TIMEOUT_S
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(
            "The Metamate backend's extra.idle_timeout_s must be a positive "
            f"number of seconds, got {value!r}."
        )
    return float(value)


class MetamateBackend:
    capabilities = BackendCapabilities(
        kind="metamate",
        caller_tools=CallerTools.NONE,
        l2_channels=(L2Channel.ENVELOPE,),
        pinned_session_id=False,  # the SDK generates a new conversation's id
        exact_fork=False,
        turn_stop_hook=False,
        subagent_attribution=False,
        compaction_signal=False,
        persistent_process=False,
        slash_passthrough=(),
        evidence={  # scripts/native_spikes/s16b_metamate_inline_agent.py
            "l1": Evidence.VERIFIED,
            "resume": Evidence.VERIFIED,
            "l2_envelope": Evidence.VERIFIED,
            # The SDK cannot cancel a running request: interrupt stops reading
            # and is never acknowledged (the turn is recorded uncertain).
            "interrupt": Evidence.UNSUPPORTED,
            # Remote agent: whether server-side user settings or memory apply
            # is not observable, so `hermetic` is not offered.
            "hermetic": Evidence.UNSUPPORTED,
            # The SDK exposes the conversation uuid (and the serving host) but
            # no conversation fbid, so the record keeps no other coordinates.
            "fbid_coordinates": Evidence.UNSUPPORTED,
        },
        relies_on=("l1", "resume", "l2_envelope"),
        # Buck-only and unversioned: the fbsource revision of the SDK sources.
        tested_versions={"//msl/metamate/sdk": "b037b06c2d47 (2026-08-27)"},
        l1_route=(
            "appended to Metamate's system prompt by the inline agent config "
            '(`system_prompt_mode="append"`), fixed when the conversation is created'
        ),
    )

    def __init__(self, spec: NativeBackendSpec, **_runtime: Any) -> None:
        self.capabilities.require_spec(spec)
        self._spec = spec
        self._idle_timeout_s = _idle_timeout_s(spec.extra)
        self._session_id = ""
        self._l1_text = ""
        self._model = spec.model
        self._interrupted: Optional[asyncio.Event] = None

    @property
    def session_id(self) -> str:
        return self._session_id

    async def open(self, request: SessionOpenRequest) -> None:
        _sdk()  # fail before the first turn when the SDK is missing
        self._l1_text = request.l1_text
        self._session_id = request.session_id if request.resume else ""
        if request.model:
            self._model = request.model

    async def run_turn(self, request: TurnRequest) -> AsyncIterator[VendorEvent]:
        client, sdk_types = _sdk()
        self._interrupted = asyncio.Event()
        resuming = bool(self._session_id)
        stream = client.execute_and_stream(
            sdk_types.MetamateSDKInput(text=request.text),
            self._client_config(sdk_types),
            self._orchestration(sdk_types),
            session_config=(
                sdk_types.MetamateSessionConfig(conversation_id=self._session_id)
                if resuming
                else None
            ),
            idle_timeout_s=self._idle_timeout_s,
        )
        announced = False
        final: Any = None
        try:
            async with contextlib.aclosing(self._until_interrupted(stream)) as events:
                async for event in events:
                    if event.conversation_id and not announced:
                        announced = True
                        yield self._announce(event.conversation_id)
                    if event.is_complete:
                        final = event
                        break
        except Exception as exc:
            yield self._error(exc, resuming=resuming, announced=announced)
            return
        if final is None:
            reason = (
                "interrupted" if self._interrupted.is_set() else "ended without a reply"
            )
            yield VendorError(message=f"Metamate turn {reason}", submitted=announced)
            return
        text = final.text or ""
        yield MessageEnd(message_id=f"mm-{uuid.uuid4().hex[:12]}", text=text)
        yield TurnEnd(
            session_id=self._session_id,
            stop_reason="end_turn",
            num_turns=1,
            result_text=text,
        )

    async def _until_interrupted(self, stream: Any) -> AsyncIterator[Any]:
        """The SDK's events until it finishes or ``interrupt()`` is called."""
        assert self._interrupted is not None
        events = stream.__aiter__()
        stop = asyncio.ensure_future(self._interrupted.wait())
        try:
            while True:
                step = asyncio.ensure_future(events.__anext__())
                await asyncio.wait({step, stop}, return_when=asyncio.FIRST_COMPLETED)
                if not step.done():
                    step.cancel()
                    await asyncio.gather(step, return_exceptions=True)
                    return
                try:
                    yield step.result()
                except StopAsyncIteration:
                    return
        finally:
            stop.cancel()
            await asyncio.gather(stop, return_exceptions=True)
            await events.aclose()

    def _announce(self, conversation_id: str) -> SessionStarted:
        """The conversation the server created or resumed for this turn."""
        replaced = bool(self._session_id) and conversation_id != self._session_id
        self._session_id = conversation_id
        return SessionStarted(session_id=conversation_id, replaced=replaced)

    def _client_config(self, sdk_types: Any) -> Any:
        extra = self._spec.extra
        return sdk_types.MetamateSDKClientConfig(
            api_keys=[extra.get("api_key") or DEFAULT_API_KEY],
            surface=extra.get("surface") or DEFAULT_SURFACE,
        )

    def _orchestration(self, sdk_types: Any) -> Any:
        extra = self._spec.extra
        inline = sdk_types.MetamateInlineAgentConfig(
            name=_AGENT_NAME,
            description=_AGENT_DESCRIPTION,
            metamate_api_key=extra.get("api_key") or DEFAULT_API_KEY,
            prompt=self._l1_text,
            system_prompt_mode="append",
        )
        return sdk_types.MetamateOrchestration(
            type=sdk_types.MetamateOrchestrationType.LLMVM,
            llmvm_config=sdk_types.MetamateLLMVMConfig(
                entry=extra.get("entry") or DEFAULT_ENTRY,
                llm_params=sdk_types.MetamateLLMParams(model=self._model)
                if self._model
                else None,
                agent_config=sdk_types.MetamateAgentConfig(
                    path_or_id="",
                    provider_type=sdk_types.MetamateAgentProviderType.INLINE,
                    inline_config=inline,
                ),
            ),
        )

    def _error(self, exc: Exception, *, resuming: bool, announced: bool) -> VendorError:
        message = str(exc)
        if _SDK_IDLE_FAILURE in message:
            # The SDK was polling a started request, possibly before it
            # reported the conversation: the turn reached Metamate.
            return VendorError(
                message=(
                    f"Metamate produced no new output for {self._idle_timeout_s:g}s "
                    "(idle_timeout_s); the request may still complete "
                    "server-side, so it is not re-sent"
                ),
                submitted=True,
            )
        if resuming and not announced and "not found" in message.lower():
            return VendorError(
                message="Metamate has no conversation to resume",
                submitted=False,
                session_missing=True,
            )
        return VendorError(
            message=f"Metamate request failed: {message}", submitted=announced
        )

    async def interrupt(self) -> None:
        if self._interrupted is None:
            return
        # Stop consuming the stream; the request keeps running server-side.
        self._interrupted.set()
        raise InterruptNotAcknowledged(
            "The Metamate SDK cannot cancel a running request; the turn may "
            "still complete server-side"
        )

    async def set_model(self, model: str) -> None:
        self._model = model

    async def close(self) -> None:
        """Nothing is held between requests."""

"""NativeTurnLoopMixin — drives vendor turns for ``NativeConversationalInferencer``.

One host call (``run_agentic_loop``) maps to one vendor turn, plus one more
for every answered widget batch (exactly where the text-protocol orchestrator
starts a new turn today). Host callbacks keep their meaning: a round is one
main-thread assistant message.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional, Sequence

from agent_foundation.common.inferencers.agentic_inferencers.conversational import (
    widget_core,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.commands import (
    UnknownCommand,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
    AgenticResult,
    CompletedAction,
    PausedResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_response_parser import (
    ConversationResponse,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tool_runtime import (
    group_and_validate,
    GroupValidationError,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocol_text import (
    TOOL_RESULTS_PREFIX,
    WIDGET_RESPONSE_PREFIX,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_controller import (
    SopCommandWording,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_feed import (
    build_sop_feed,
    filtered_sops,
    prepare_sop_for_turn,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.tool_dispatch import (
    ToolOutcome,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.widget_core import (
    AfterAnswer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.bridge.schema import (
    MCP_PREFIX,
    mcp_tool_name,
    SOP_COMMAND_SCHEMAS,
    TOOL_ARGUMENT_FORM,
    WIDGET_SCHEMAS,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.context.composer import (
    drifted_parts,
    neutralize_host_tags,
    new_nonce,
    strip_host_blocks,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.context.text_units import (
    utf16_head,
    utf16_len,
    utf16_tail,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.errors import (
    NativeCapabilityError,
    RewindUnsupported,
    SessionResumeRejected,
    StablePolicyChanged,
    VendorSessionMissing,
    VendorTurnFailed,
    VendorTurnStalled,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.events import (
    Compaction,
    MessageEnd,
    SessionStarted,
    TextDelta,
    TurnEnd,
    VendorError,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.private_files import (
    write_private_file,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.actor import (
    SessionActor,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    CallerTools,
    L2Channel,
    SessionOpenRequest,
    TurnRequest,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.binding import (
    SessionBinding,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.factory import (
    backend_class,
    make_backend,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.record import (
    ADAPTER_VERSION,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.turn import (
    StopCause,
    SubmissionState,
    TurnOrigin,
    TurnScope,
)
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    enter_run,
    exit_run,
)

logger: logging.Logger = logging.getLogger(__name__)

_SAFETY_CEILING = 1000

# Turns whose content the host wrote; a user turn's text and a widget answer
# ("[Collected from conversation widget]") speak for themselves.
_HOST_WRITTEN_ORIGINS = frozenset(
    {
        TurnOrigin.VALIDATION_RETRY,
        TurnOrigin.TOOL_COMPLETION,
        TurnOrigin.HOST_EVENT,
        TurnOrigin.RESUMED_TURN,
    }
)
# L2 channels whose blocks stay in the vendor session's history (S2: hook
# additional context is a transcript attachment; an envelope is part of the
# user message). A per-request config (INLINE_CONFIG) is not.
_PERSISTENT_L2_CHANNELS = frozenset({L2Channel.HOOK, L2Channel.ENVELOPE})
# Size budget of one L2, in UTF-16 code units (how the vendor counts). Claude
# Code inlines hook context up to 10,000 characters and replaces a longer one
# with a ~2 KB preview and a file path (S2). An envelope or a per-request
# config travels with the request, which no vendor cuts, but is repeated
# whenever an L2 is due.
_L2_BUDGET = {
    L2Channel.HOOK: 9_000,
    L2Channel.INLINE_CONFIG: 16_000,
    L2Channel.ENVELOPE: 16_000,
}
# Room the SOP state leaves for notices when both compete, so that every L2
# delivers at least one notice in full or cut.
_NOTICE_RESERVE = 2_000
# The shortest excerpt a cut notice keeps.
_MIN_EXCERPT = 400
# Notices whose end matters most when cut: a recap ends with the latest turns.
_TAIL_FIRST_NOTICES = frozenset({"recap"})
_SPILL_DIR = "turn_context"
_VALIDATION_RETRY_PREFIX = f"{TOOL_RESULTS_PREFIX}\n[parallel_group validation]"

_MANIFEST_HEADER = (
    "## Prompt manifest (approximate)\n"
    "What AgentFoundation adds on this turn, each part with its route. The "
    "vendor also sends its own system prompt, tool definitions and transcript, "
    "which are not shown."
)
# How a vendor turn that did not complete ended, by the submission state it
# left; its manifest stays the last prompt data, marked with this outcome.
_MANIFEST_OUTCOMES = {
    SubmissionState.PREPARED.value: (
        "failed",
        "The turn failed before it reached the agent.",
    ),
    SubmissionState.UNCERTAIN.value: (
        "uncertain",
        "The turn did not complete after it may have reached the agent, which "
        "may still act on it; it is not re-sent.",
    ),
    SubmissionState.INTERRUPTED.value: (
        "interrupted",
        "The turn was stopped and the agent confirmed the stop.",
    ),
}

_NOTICE_TEXT = {
    "interrupted": "The previous turn was interrupted. Check the current state "
    "before continuing and do not redo work that already completed.",
    "turn_failed": "The previous turn failed partway through. Check the current "
    "state before continuing and do not redo work that already completed.",
    "widget_cancelled": "The user dismissed the question you asked last turn "
    "without answering it.",
    "widget_unshown": "The question you asked in the interrupted turn was never "
    "shown to the user; ask it again if you still need the answer.",
    "tools_updated": "The host changed the AgentFoundation tools (`mcp__af__*`) "
    "available to you: go by your current tool list, not by tools you saw "
    "earlier in this session.",
}


@dataclass
class _SessionPlan:
    l1_text: str
    l1_path: str
    l1_core_hash: str
    resume: bool
    fork_from: Optional[tuple[str, str]] = None
    snapshot: Optional[dict[str, Any]] = None  # record before a rewind


@dataclass
class _CallState:
    """Accumulated across the vendor turns of one host call."""

    turn_number: int
    iteration: int = 0
    final_text: str = ""
    actions: list = field(default_factory=list)
    vendor_turns: int = 0


class NativeTurnLoopMixin:
    # =========================================================================
    # Entry point
    # =========================================================================

    async def run_agentic_loop(
        self, content: str, *, run_context: Any = None, **kwargs: Any
    ) -> AgenticResult:
        token = enter_run(
            run_context, default_workspace=getattr(self, "_workspace", None)
        )
        try:
            return await self._run_native_turn(content, **kwargs)
        finally:
            exit_run(token)

    # =========================================================================
    # Lanes
    # =========================================================================

    def _base_feed(self, *, catalog_mode: str) -> dict[str, Any]:
        from rich_python_utils.common_objects.feed_base import build_feed

        template_vars = getattr(self.prompt_renderer, "template_variables", {}) or {}
        sop_feed = build_sop_feed(
            self.sop_controller,
            self.prior_context,
            catalog_mode=catalog_mode,
            **self._sop_catalog_filters(),
        )
        return build_feed(
            template_vars,
            self.prior_context,
            self.sop_state,
            {
                "session_root_path": self.prior_context.get("session_root_path")
                or self.backend.cwd,
                **sop_feed.template_values(),
                **self._section_wording(),
            },
        )

    def _offered_sop_tools(self) -> list[str]:
        """The SOP-control tools the agent can call (the bridge offers exactly
        ``sop_control_tools``); the user drives the others by slash command."""
        if self._backend_caps().caller_tools is CallerTools.NONE:
            return []
        return [name for name in SOP_COMMAND_SCHEMAS if name in self.sop_control_tools]

    def _section_wording(self) -> dict[str, str]:
        """The shared sections' wording variables, phrased for an agent that
        calls AF tools (the sections' defaults phrase the classic protocol's
        slash commands)."""
        if "resume_sop" in self._offered_sop_tools():
            usage = command = f"`{MCP_PREFIX}resume_sop`"
        else:
            usage = "a `/resume_sop <name>` message, which only the user can send"
            command = "a `/resume_sop` message from the user"
        return {
            "identity_phrasing": "deployment_role",
            "sop_resume_usage": usage,
            "sop_resume_command": command,
        }

    def sop_command_wording(self) -> SopCommandWording:
        """How SOP-control tool results name resuming an SOP and starting it
        over: by the SOP tools the agent can call, else as the user's slash
        commands (the controller's default wording is for the user)."""
        offered = self._offered_sop_tools()
        return SopCommandWording(
            resume=f"`{MCP_PREFIX}resume_sop` (name `{{name}}`)"
            if "resume_sop" in offered
            else "a `/resume_sop {name}` message from the user",
            fresh=f"`{MCP_PREFIX}enter_sop` with `fresh: true`"
            if "enter_sop" in offered
            else "a `/sop {name} --fresh` message from the user",
        )

    def _question_tools(self, manifest: Sequence[Any]) -> list[str]:
        offered = {spec.name for spec in manifest}
        return [
            MCP_PREFIX + name
            for name in (*WIDGET_SCHEMAS, TOOL_ARGUMENT_FORM)
            if name in offered
        ]

    def _required_tools_text(self) -> str:
        """The current SOP phase's required tools by the names the agent
        calls them (SOP guidance writes them in the classic notation, e.g.
        ``/write-brief <topic>``)."""
        state = self.sop_state
        phase_id = state.current_phase if state is not None else None
        required = state.phase_required_tools.get(phase_id) if phase_id else None
        if not required:
            return ""
        offered = {spec.name for spec in self._session_manifest()}
        executed = state.phase_executed_tools.get(phase_id) or set()
        names = []
        for tool in sorted(required):
            local = mcp_tool_name(self._resolve_tool_name(tool))
            name = (
                f"`{MCP_PREFIX}{local}`"
                if local in offered
                else f"`{tool}` (not available to you here)"
            )
            names.append(f"{name} (already run)" if tool in executed else name)
        phase = state.sop.get_phase(phase_id) if state.sop is not None else None
        title = f"Phase {phase_id} ({phase.name})" if phase else f"Phase {phase_id}"
        return f"{title} completes once these tools have run: {', '.join(names)}"

    def _sop_catalog_filters(self) -> dict[str, Any]:
        return {
            "extra_sop_dirs": self.extra_sop_dirs,
            "allowed": self.allowed_sops,
            "disallowed": self.disallowed_sops,
        }

    def _l1_feed(
        self, record: Any, manifest: Optional[Sequence[Any]] = None
    ) -> dict[str, Any]:
        if manifest is None:
            manifest = self._session_manifest()
        feed = self._base_feed(catalog_mode="always")
        feed.update(
            {
                "nonce": record.nonce,
                "has_caller_tools": self._backend_caps().caller_tools
                is not CallerTools.NONE,
                "sop_control_tools": self._offered_sop_tools(),
                "question_tools": self._question_tools(manifest),
                "soft_max_iterations": self.soft_max_iterations,
            }
        )
        return feed

    def _l2_feed(
        self,
        *,
        origin: str,
        notices: list,
        generation: int,
        turn: Optional[int] = None,
    ) -> dict[str, Any]:
        record = self._load_record()
        if turn is None:
            turn = self._current_turn.turn_number if self._current_turn else 0
        feed = self._base_feed(catalog_mode="when_idle")
        feed.update(
            {
                "nonce": record.nonce,
                "turn": turn,
                "generation": generation,
                "origin": origin,
                "notices": notices,
                "catalog_changes": self._catalog_changes(record),
                "sop_required_tools": self._required_tools_text(),
                "delivered_state": self._delivered_state(record),
            }
        )
        return feed

    def _delivered_state(self, record: Any) -> str:
        """The state hash of the latest L2 or L3 the agent received in this
        vendor session; "" when it may be gone from its context (the vendor
        compacted during the running turn)."""
        turn = self._current_turn
        if turn is None:
            return record.l2_hash
        if turn.compacted:
            return ""
        return turn.l3_state_hash or turn.l2_hash or record.l2_hash

    def _catalog_changes(self, record: Any) -> str:
        from agent_foundation.resources.sops.registry import format_all_sops

        current = filtered_sops(**self._sop_catalog_filters())
        snapshot = set(record.catalog_snapshot)
        added = {n: i for n, i in current.items() if n not in snapshot}
        removed = sorted(snapshot - set(current))
        lines = []
        if added:
            lines.append("New SOPs:\n" + format_all_sops(added))
        if removed:
            lines.append("No longer available: " + ", ".join(removed))
        return "\n".join(lines)

    def _compose_l2(
        self,
        origin: TurnOrigin,
        *,
        current: str,
        turn: int = 0,
        channel: L2Channel = L2Channel.HOOK,
        force: bool = False,
    ) -> tuple[str, str, int]:
        """Return ``(l2_text, l2_hash, delivered_upto)``; empty text when the
        L2 is not due (``_l2_due``) and not ``force``d. ``delivered_upto`` is
        the outbox position the rendered notices reach."""
        record = self._load_record()
        pending = record.pending_notices()
        notices = self._render_notices(pending, current)
        upto = record.outbox_cursor + len(pending)
        feed = self._l2_feed(
            origin=origin.value,
            notices=[],
            generation=record.l2_generation + 1,
            turn=turn,
        )
        state, l2_hash = self._composer.turn_state(feed)
        if force or self._l2_due(record, origin, l2_hash, bool(notices), channel):
            return (
                self._fit_l2(feed, state, l2_hash, notices, _L2_BUDGET[channel]),
                l2_hash,
                upto,
            )
        return "", l2_hash, upto

    def _fit_l2(
        self, feed: dict, state: str, state_hash: str, notices: list, budget: int
    ) -> str:
        """The L2 in at most ``budget`` units: the SOP state first, then the
        notices, newest first. A text that does not fit is cut where it
        says so, its full text in a private file the cut names; notices for
        which not even a cut fits are written to one such file, named by a
        last notice. Every notice is delivered."""

        def render(state_text: str, fitted: list) -> str:
            return self._composer.wrap_turn_state(
                dict(feed, notices=fitted), state_text
            )

        frame = utf16_len(render("", []))
        room = budget - frame
        state_room = room - (_NOTICE_RESERVE if notices else 0)
        if utf16_len(state) > state_room:
            name = f"sop_state_{state_hash.rpartition('.')[2][:16]}.md"
            state = _cut(state, state_room, self._spill(name, state), keep_tail=False)
        fitted = self._fit_notices(render, frame, notices, room - utf16_len(state))
        return render(state, fitted)

    def _fit_notices(
        self, render: Callable[[str, list], str], frame: int, notices: list, room: int
    ) -> list:
        """Notices within ``room`` units: whole where they fit, newest first;
        then the others cut to equal shares of what is left, newest first;
        any that not even a cut fits are written to one file that a last
        notice names. Returned oldest first, that notice last."""
        if not notices:
            return []
        overheads: dict[str, int] = {}

        def cost(kind: str, text: str) -> int:
            if kind not in overheads:
                empty = render("", [{"type": kind, "text": ""}])
                overheads[kind] = utf16_len(empty) - frame
            return overheads[kind] + utf16_len(text)

        bundle_name = f"notices_{notices[0]['id']}.txt"
        bundle = _bundle_text(len(notices), self._spill_path(bundle_name))
        room -= cost("more_notices", bundle) if len(notices) > 1 else 0
        fitted: dict[int, dict] = {}
        for index in range(len(notices) - 1, -1, -1):
            notice = notices[index]
            if cost(notice["type"], notice["text"]) <= room:
                fitted[index] = notice
                room -= cost(notice["type"], notice["text"])
        rest = [i for i in range(len(notices) - 1, -1, -1) if i not in fitted]
        for position, index in enumerate(rest):
            notice = notices[index]
            name = f"notice_{notice['id']}.txt"
            share = room // (len(rest) - position) - cost(notice["type"], "")
            if share < _MIN_EXCERPT + utf16_len(_marker(self._spill_path(name))):
                break
            text = _cut(
                notice["text"],
                share,
                self._spill(name, notice["text"]),
                keep_tail=notice["type"] in _TAIL_FIRST_NOTICES,
            )
            fitted[index] = dict(notice, text=text)
            room -= cost(notice["type"], text)
        ordered = [fitted[i] for i in sorted(fitted)]
        left = [notices[i] for i in sorted(set(range(len(notices))) - set(fitted))]
        if left:
            body = "\n\n".join(f"## {n['type']}\n{n['text']}" for n in left)
            path = self._spill(bundle_name, body)
            ordered.append(
                {"type": "more_notices", "text": _bundle_text(len(left), path)}
            )
        return ordered

    def _spill_path(self, name: str) -> str:
        return str(self.session_dir() / _SPILL_DIR / name)

    def _spill(self, name: str, text: str) -> str:
        """Write ``text`` to a private file of this conversation; returns its
        path, which the agent can read."""
        (self.session_dir() / _SPILL_DIR).mkdir(mode=0o700, exist_ok=True)
        path = str(self._write_private(f"{_SPILL_DIR}/{name}", text))
        logger.info(
            "Native session %s: %d characters of turn context written to a file",
            self.conversation_key,
            len(text),
        )
        return path

    @staticmethod
    def _l2_due(
        record: Any,
        origin: TurnOrigin,
        l2_hash: str,
        has_notices: bool,
        channel: L2Channel,
    ) -> bool:
        """Plan §3.1: an L2 is sent on the first turn of a vendor session
        (new, forked, rotated), when the state differs from the one the
        session last received (``record.l2_hash``, reset when that may be
        lost: compaction, a reopened or replaced session), with notices, for
        host-written turns (their L2 states the origin), and on every turn of
        a channel the vendor does not keep in its history. Hook context and
        envelopes stay in the transcript (S2), so a session continued after
        a process restart is not sent the state it already holds."""
        return (
            not record.started
            or l2_hash != record.l2_hash
            or has_notices
            or origin in _HOST_WRITTEN_ORIGINS
            or channel not in _PERSISTENT_L2_CHANNELS
        )

    def _render_notices(self, entries: list, current: str) -> list[dict[str, Any]]:
        rendered = []
        for entry in entries:
            text = self._notice_text(entry, current)
            if text:
                rendered.append(
                    {
                        "id": entry.get("id"),
                        "type": entry.get("type", "notice"),
                        "text": text,
                    }
                )
        return rendered

    def _notice_text(self, entry: dict, current: str) -> str:
        """Render a stored notice reference at delivery time (the record never
        holds instructions, transcripts or tool output)."""
        kind = entry.get("type", "")
        if kind in _NOTICE_TEXT:
            return _NOTICE_TEXT[kind]
        if kind == "instructions_updated":
            l1_text, _ = self._composer.session_instructions(
                self._l1_feed(self._load_record())
            )
            return (
                "The host updated your session instructions; they supersede the "
                "earlier ones:\n" + neutralize_host_tags(l1_text)
            )
        if kind == "recap":
            return self._recap_text(current)
        if kind == "tool_completion":
            return self._tool_completion_text(entry)
        if kind == "dashboard_handoff":
            return self._answer_notice_text(
                entry,
                "The user answered your last question and the host handed the "
                "work to a dashboard, which ended that turn",
            )
        if kind == "background_answer":
            return self._answer_notice_text(
                entry,
                "The user answered your last question and an action it ran "
                "started in the background, which ended that turn; its outcome "
                "arrives later as a host notice",
            )
        if kind == "unsent_turn":
            return self._unsent_turn_text(entry)
        logger.debug("Skipping unknown native notice type %r", kind)
        return ""

    def _tool_completion_text(self, entry: dict) -> str:
        tool = str(entry.get("tool", "a background tool"))
        body = self._notice_bodies.get(entry.get("id"))
        if body is None:
            prefix = f"{TOOL_RESULTS_PREFIX}\n{tool}: "
            for message in reversed(self._messages):
                text = str(message.get("content", ""))
                if text.startswith(prefix):
                    body = text[len(prefix) :]
                    break
        if body is None:
            return f"{tool} finished; its result was not retained."
        return f"{tool} finished:\n{neutralize_host_tags(body)}"

    def _answer_notice_text(self, entry: dict, head: str) -> str:
        """A widget answer that ended its host turn without a vendor turn."""
        body = self._notice_bodies.get(entry.get("id"))
        if body is None:
            body = self._last_user_message(WIDGET_RESPONSE_PREFIX)
        if body is None:
            return f"{head}."
        return f"{head}:\n{neutralize_host_tags(body)}"

    def _unsent_turn_text(self, entry: dict) -> str:
        if entry.get("origin") == TurnOrigin.WIDGET_ANSWER.value:
            head = (
                "The user answered your last question, but the host's limit on "
                "your turns per message ended that call before the answer was "
                "sent to you. It has been applied"
            )
            prefix = WIDGET_RESPONSE_PREFIX
        else:
            head = (
                "Your last questions were not shown to the user, and the host's "
                "limit on your turns per message ended that call before this "
                "reply was sent to you"
            )
            prefix = _VALIDATION_RETRY_PREFIX
        body = self._notice_bodies.get(entry.get("id"))
        if body is None:
            body = self._last_user_message(prefix)
        if body is None:
            return f"{head}."
        return f"{head}:\n{neutralize_host_tags(body)}"

    def _last_user_message(self, prefix: str) -> Optional[str]:
        """The latest mirrored user message starting with ``prefix`` (a
        notice body after a restart, which keeps only the mirror)."""
        for message in reversed(self._messages):
            text = str(message.get("content", ""))
            if message.get("role") == "user" and text.startswith(prefix):
                return text
        return None

    def _backend_caps(self) -> Any:
        source = self.backend_factory or backend_class(self.backend.kind)
        return source.capabilities

    def _make_backend(self) -> Any:
        factory = self.backend_factory or make_backend
        return factory(self.backend, runtime_manager=self.runtime_manager)

    # =========================================================================
    # Session lifecycle
    # =========================================================================

    def _session_key(self, record: Any) -> tuple:
        return (self.conversation_key, self.backend.kind, record.generation)

    async def rewind_to(self, turn_number: int) -> None:
        """Make the vendor session remember only host turns before
        ``turn_number`` — an exact fork at that turn boundary where the backend
        supports it, otherwise per ``on_rewind_unsupported``.

        Hosts call this BEFORE truncating their own history (rewind, resume
        from a turn, checkpoint restore): if it raises (``RewindUnsupported``)
        nothing should be truncated."""
        if not self._load_record().started:
            return  # no vendor session holds any turn yet
        await self._ensure_session(turn_number, force_rewind=True)
        self._touch_runtime()

    async def _ensure_session(
        self, turn_number: int, *, allow_rewind: bool = True, force_rewind: bool = False
    ) -> SessionActor:
        """Return a live actor for this conversation's vendor session, bound to
        this inferencer. A resume the vendor rejects (session gone), or a fork
        that fails under ``on_rewind_unsupported="recap"``, continues in a
        fresh session with a recap, once."""
        for attempt in (0, 1):
            record = self._load_record()
            plan = self._plan_session(
                record,
                turn_number,
                allow_rewind=allow_rewind,
                force_rewind=force_rewind and not attempt,
            )
            key = self._session_key(record)
            await self._acquire_lease(key)
            try:
                actor = await self.runtime_manager.get_actor(
                    key, self._actor_factory(record, plan)
                )
            except VendorSessionMissing as exc:
                if plan.fork_from is None and not attempt:
                    logger.warning(
                        "Native session %s: the vendor no longer has session %s; "
                        "starting a fresh one",
                        self.conversation_key,
                        _short(record.vendor_session_id),
                    )
                    self._lose_session(record, "the vendor no longer has this session")
                    continue
                if self._abort_fork(record, plan, exc, turn_number, retry=not attempt):
                    continue
                raise
            except Exception as exc:
                if self._abort_fork(record, plan, exc, turn_number, retry=not attempt):
                    continue
                raise
            actor = await self._adopt_actor(record, actor, plan, key)
            return actor
        raise AssertionError("unreachable")

    def _plan_session(
        self,
        record: Any,
        turn_number: int,
        *,
        allow_rewind: bool,
        force_rewind: bool = False,
    ) -> _SessionPlan:
        caps = self._backend_caps()
        if not record.nonce:
            record.nonce = new_nonce()
        self._seed_history_recap(record)
        snapshot = record.to_dict() if record.started else None
        fork_from = self._check_resume(
            record, turn_number, caps, allow_rewind=allow_rewind
        )
        if force_rewind and fork_from is None and record.started:
            # An explicit host rewind (e.g. after a checkpoint restore the
            # vendor session may hold turns the host no longer has).
            fork_from = self._rewind(record, turn_number, caps)
        manifest = self._session_manifest()
        l1_text, core_hash = self._composer.session_instructions(
            self._l1_feed(record, manifest),
            tool_manifest=_manifest_fingerprint(manifest),
        )
        if record.started and core_hash != record.l1_core_hash:
            self._handle_l1_drift(record, core_hash)
        if not record.started:
            self._begin_session(record, core_hash, caps)
        l1_path = self._write_private(f"l1_{record.generation}.md", l1_text)
        return _SessionPlan(
            l1_text=l1_text,
            l1_path=str(l1_path),
            l1_core_hash=core_hash,
            resume=record.started and fork_from is None,
            fork_from=fork_from,
            snapshot=snapshot if fork_from is not None else None,
        )

    def _begin_session(self, record: Any, core_hash: str, caps: Any) -> None:
        """A new vendor session adopts the current identity and policy."""
        record.l1_core_hash = core_hash
        record.catalog_snapshot = sorted(filtered_sops(**self._sop_catalog_filters()))
        record.backend = self.backend.kind
        record.cwd = self.backend.cwd
        record.model = self.backend.model
        record.principal = self.principal
        record.permission_fingerprint = self._permission_fingerprint()
        record.adapter_version = ADAPTER_VERSION
        if not record.vendor_session_id and caps.pinned_session_id:
            record.vendor_session_id = str(uuid.uuid4())

    async def _acquire_lease(self, key: tuple) -> None:
        if self._lease_key == key:
            return
        if self._lease_key is not None:
            await self.runtime_manager.release(self._lease_key)
        await self.runtime_manager.acquire(key)
        self._lease_key = key

    def _actor_factory(self, record: Any, plan: _SessionPlan):
        async def factory() -> SessionActor:
            backend = self._make_backend()
            binding = SessionBinding(self)
            manifest = (
                self.bridge.manifest()
                if backend.capabilities.caller_tools is not CallerTools.NONE
                else []
            )
            request = SessionOpenRequest(
                session_id=record.vendor_session_id,
                resume=plan.resume,
                l1_text=plan.l1_text,
                l1_path=plan.l1_path,
                tools=binding.tools(manifest),
                hooks=binding,
                cwd=self.backend.cwd,
                model=self.backend.model,
                fork_from=plan.fork_from,
                result_max_chars=self.native_tool_result_max_chars,
            )
            actor = SessionActor(
                backend,
                request,
                binding=binding,
                drain_timeout_s=self.vendor_drain_timeout_s,
            )
            actor.manifest_fingerprint = _manifest_fingerprint(manifest)
            actor.l1_core_hash = plan.l1_core_hash
            await actor.start()
            return actor

        return factory

    async def _adopt_actor(
        self, record: Any, actor: SessionActor, plan: _SessionPlan, key: tuple
    ) -> SessionActor:
        if actor.binding is not None:
            actor.binding.bind(self)
        stale = self._stale_launch(actor, plan)
        if stale:
            logger.info(
                "Native session %s: %s changed; reopening",
                self.conversation_key,
                stale,
            )
            actor = await self._reopen(record, plan, key)
        session_id = actor.backend.session_id
        if plan.fork_from is not None and session_id:
            self._remap_boundaries(record, actor, plan.fork_from[0], session_id)
        if session_id and session_id != record.vendor_session_id:
            record.vendor_session_id = session_id
        if record.model != self.backend.model:
            actor = await self._switch_model(record, actor, plan, key)
        self._save_record()
        return actor

    async def _reopen(
        self, record: Any, plan: _SessionPlan, key: tuple
    ) -> SessionActor:
        """Close the live session and open it again, resuming the same vendor
        session (a fresh one if it never started)."""
        await self.runtime_manager.evict(key)
        await self.runtime_manager.acquire(key)
        plan.resume = record.started
        record.l2_hash = ""  # the reopened session is told the state again
        actor = await self.runtime_manager.get_actor(
            key, self._actor_factory(record, plan)
        )
        actor.binding.bind(self)
        return actor

    async def _switch_model(
        self, record: Any, actor: SessionActor, plan: _SessionPlan, key: tuple
    ) -> SessionActor:
        """A model change (host configuration or ``/model``) continues the
        session on the new model (plan §7.3 ``/model``: the SDK switches the
        live client, the CLIs pass it to the next process). A switch the
        vendor refuses reopens the session, which then starts on the new
        model: the session never goes on with a stale model unnoticed."""
        try:
            await actor.set_model(self.backend.model)
        except Exception as exc:
            logger.warning(
                "Native session %s: the live model switch failed (%s); reopening",
                self.conversation_key,
                exc,
            )
            actor = await self._reopen(record, plan, key)
        record.model = self.backend.model
        return actor

    def _stale_launch(self, actor: SessionActor, plan: _SessionPlan) -> str:
        """What a live session was opened with that no longer holds, if
        anything; reopening it resumes the same vendor session.

        * its AF tool server: the host rebuilt this inferencer with another
          tool set;
        * its session instructions (D8): a running Claude SDK process read the
          L1 file at launch and re-records that text at its next compaction
          (S1), and the dm / Codex / Metamate backends hold the text they were
          opened with, so only a reopened session picks a drifted L1 up."""
        if (
            getattr(actor, "manifest_fingerprint", None)
            != self._current_manifest_fingerprint()
        ):
            return "tool set"
        if getattr(actor, "l1_core_hash", None) != plan.l1_core_hash:
            return "session instructions"
        return ""

    def _session_manifest(self) -> list:
        """The AF tools a session of this backend is given."""
        if self._backend_caps().caller_tools is CallerTools.NONE:
            return []
        return self.bridge.manifest()

    def _current_manifest_fingerprint(self) -> str:
        return _manifest_fingerprint(self._session_manifest())

    def _abort_fork(
        self,
        record: Any,
        plan: _SessionPlan,
        exc: BaseException,
        turn_number: int,
        *,
        retry: bool,
    ) -> bool:
        """A failed fork leaves the original session untouched. Under
        ``on_rewind_unsupported="recap"`` (and ``retry``) the conversation
        continues in a fresh session with a recap — returns True to start it;
        otherwise raises ``RewindUnsupported`` so the host truncates nothing
        either. Returns False when the failure was not a fork."""
        if plan.fork_from is None or plan.snapshot is None:
            return False
        restored = type(record).from_dict(plan.snapshot)
        self._record = restored
        self.record_store.save(restored)
        if retry and self.on_rewind_unsupported == "recap":
            logger.warning(
                "Native session %s: forking failed (%s); continuing in a fresh "
                "session with a recap",
                self.conversation_key,
                exc,
            )
            restored.last_turn = turn_number - 1  # the recap covers the earlier turns
            self._lose_session(restored, "the session could not be forked")
            return True
        raise RewindUnsupported(
            f"{self.backend.kind} could not fork its session: {exc}"
        ) from exc

    def _remap_boundaries(
        self, record: Any, actor: SessionActor, source: str, forked: str
    ) -> None:
        """The fork gives every copied message a new id; keep the earlier turn
        boundaries usable for later rewinds of the forked session."""
        remap = getattr(actor.backend, "fork_message_map", None)
        mapping = remap(source, forked) if callable(remap) else {}
        record.turn_boundaries = {
            turn: mapping[uuid_]
            for turn, uuid_ in record.turn_boundaries.items()
            if uuid_ in mapping
        }

    def _check_resume(
        self, record: Any, turn_number: int, caps: Any, *, allow_rewind: bool
    ) -> Optional[tuple]:
        """Validate a persisted session before contacting the vendor (README
        "Resume policy"); returns a fork point when the host rewound to an
        earlier turn.

        A session is never continued across an incompatible fingerprint: a
        principal mismatch always fails; a session that cannot or must not be
        continued here is lost under ``on_session_loss`` (D9); a session that
        would be continued must have the permission policy it started with."""
        if not record.started:
            return None
        if record.principal and record.principal != self.principal:
            raise SessionResumeRejected(
                f"Session belongs to {record.principal!r}, not {self.principal!r}."
            )
        lost = self._unresumable(record)
        if lost:
            self._lose_session(record, lost)
            return None
        if (
            record.permission_fingerprint
            and record.permission_fingerprint != self._permission_fingerprint()
        ):
            raise SessionResumeRejected(
                "The permission policy changed since this agent session started; "
                "send /new to start a fresh agent session."
            )
        if (
            allow_rewind
            and self.rewind_on_repeat_turn
            and 0 < turn_number <= record.last_turn
        ):
            return self._rewind(record, turn_number, caps)
        return None

    def _unresumable(self, record: Any) -> str:
        """Why the recorded vendor session is not continued here ("" when it
        is): another backend cannot continue it; another adapter version reads
        its coordinates differently. A working-directory change by the host
        (e.g. an inferencer rebuilt with another cwd; ``/root`` rotates the
        session itself) is a policy choice, not a vendor limit (a Claude
        session resumed elsewhere keeps its memory and is told the new
        directory, S10): the earlier turns' files belong to the other tree,
        and the other backends' directory binding is unverified."""
        if record.adapter_version != ADAPTER_VERSION:
            return (
                f"adapter version {record.adapter_version!r} is not {ADAPTER_VERSION!r}"
            )
        if record.backend != self.backend.kind:
            return f"the backend changed from {record.backend!r}"
        if record.cwd != self.backend.cwd:
            return "the working directory changed"
        return ""

    def _rewind(self, record: Any, turn_number: int, caps: Any) -> Optional[tuple]:
        # Whatever replaces the session holds (or recaps) only the turns before
        # ``turn_number``, so re-running it is not a repeat.
        if turn_number == 1:
            record.last_turn = 0
            self._rotate_session("rewind to the first turn")
            return None
        boundary = record.turn_boundaries.get(str(turn_number - 1))
        if not caps.exact_fork or not boundary:
            if self.on_rewind_unsupported == "recap":
                record.last_turn = turn_number - 1
                self._lose_session(
                    record, f"rewind to turn {turn_number} without an exact fork"
                )
                return None
            raise RewindUnsupported(
                f"{self.backend.kind} cannot fork its session at turn {turn_number}."
            )
        source = record.vendor_session_id
        record.generation += 1
        record.vendor_session_id = ""
        record.status = "new"
        record.l2_hash = ""  # the fork may predate the last state it was told
        record.last_turn = turn_number - 1
        record.turn_boundaries = {
            k: v for k, v in record.turn_boundaries.items() if int(k) < turn_number
        }
        return (source, boundary)

    def _lose_session(self, record: Any, reason: str) -> None:
        if self.on_session_loss == "fail":
            raise SessionResumeRejected(f"Vendor session cannot continue: {reason}")
        self._rotate_session(reason)
        if self.on_session_loss == "recap":
            self._queue_notice("recap")

    def _seed_history_recap(self, record: Any) -> None:
        """D9: the first native turn of a conversation that already has host
        history (e.g. a backend switch) gets a recap of it."""
        if record.saved_at or record.started or record.generation:
            return
        if not any(m.get("role") in ("user", "assistant") for m in self._messages):
            return
        if self.on_session_loss == "fail":
            raise SessionResumeRejected(
                "This conversation has history but no agent session to continue."
            )
        if self.on_session_loss == "recap":
            record.add_notice("recap")

    def _handle_l1_drift(self, record: Any, core_hash: str) -> None:
        """D8: the session instructions or the AF tool set changed for a
        started session (plan §3.1: either is drift)."""
        instructions, tools = drifted_parts(record.l1_core_hash, core_hash)
        what = "Session instructions" if instructions else "The AF tool set"
        if self.on_l1_drift == "fail":
            raise StablePolicyChanged(f"{what} changed for an existing agent session.")
        if self.on_l1_drift == "rotate":
            self._lose_session(record, f"{what.lower()} changed")
            return
        record.l1_core_hash = core_hash
        record.add_notice("instructions_updated" if instructions else "tools_updated")

    def _recap_text(self, current: str) -> str:
        messages = [m for m in self._messages if m.get("role") in ("user", "assistant")]
        if (
            messages
            and messages[-1].get("role") == "user"
            and messages[-1].get("content") == current
        ):
            messages = messages[:-1]  # the message being sent now is not history
        text = "\n".join(f"{m.get('role')}: {m.get('content', '')}" for m in messages)
        limit = self.recap_max_chars
        if len(text) > limit:
            text = "… " + text[-limit:]
        if not text:
            return ""
        return (
            "Earlier conversation in this host session (recap; you did not see "
            "these turns in this agent session):\n" + neutralize_host_tags(text)
        )

    def _write_private(self, name: str, text: str) -> Any:
        """Write ``text`` to the conversation's private (0600) file ``name``."""
        return write_private_file(self.session_dir() / name, text)

    # =========================================================================
    # Turn driver
    # =========================================================================

    async def _run_native_turn(
        self,
        content: str,
        *,
        interactive: Any = None,
        session_id: str = "",
        turn_number: int = 0,
        on_new_turn: Any = None,
        on_prompt_rendered: Any = None,
        on_turn_complete: Any = None,
        on_round_start: Any = None,
        on_round_complete: Any = None,
        origin: str = "user",
    ) -> AgenticResult:
        interactive = interactive or self.interactive
        cb = _HostCallbacks(
            self,
            interactive,
            session_id,
            on_new_turn,
            on_prompt_rendered,
            on_turn_complete,
            on_round_start,
            on_round_complete,
        )
        turn_origin = (
            TurnOrigin(origin)
            if origin in TurnOrigin._value2member_map_
            else TurnOrigin.USER
        )
        self._settle_unfinished_turn()

        # 1. Our slash commands never reach the vendor.
        handled, content = await self._dispatch_command(content)
        if handled is not None:
            return handled
        vendor_command = self._is_vendor_command(content)

        # 2. A re-armed widget answer (host restart / refresh): decode it
        # against the exact persisted widget, no re-inference, no rewind.
        allow_rewind = True
        first_iteration = 0
        new_turn_input = content
        answer_actions: list = []
        started_background = False
        recovered_answer = self._pending_widget_result is not None
        if recovered_answer:
            recovered = await self._recover_pending_widget(cb)
            if isinstance(recovered, AgenticResult):
                return recovered
            content, turn_origin, allow_rewind = (
                recovered.message(),
                TurnOrigin.WIDGET_ANSWER,
                False,
            )
            new_turn_input = recovered.response_text
            answer_actions = _answer_actions(recovered)
            started_background = recovered.async_dispatched
            # The widget's blob records the round its answer is continued in
            # (the widget's own round had closed): number rounds from there.
            if self._restored_iteration is not None:
                first_iteration = self._restored_iteration
        self._restored_iteration = None

        host_turn = turn_number
        turn_number = await cb.new_turn(turn_number, new_turn_input)
        state = _CallState(
            turn_number=turn_number, iteration=first_iteration, actions=answer_actions
        )
        if started_background:
            return await self._complete_call(state, cb)
        if recovered_answer and turn_number != host_turn:
            # As after a live answer: the answer continues in another host turn.
            await cb.turn_boundary(turn_number)
        actor = await self._ensure_session(turn_number, allow_rewind=allow_rewind)
        return await self._vendor_turns(
            actor, content, turn_origin, vendor_command, interactive, cb, state
        )

    def _settle_unfinished_turn(self) -> None:
        """A record still ``submitted`` when a host call starts belongs to a
        vendor turn that never ended — the host died during it (or it broke
        without reaching the failure handling). The vendor may hold that turn,
        so it is ``uncertain`` and never replayed; the next L2 says it was
        interrupted. A widget it queued (persisted by the bridge) was never
        shown: presenting waits for the turn's end."""
        record = self._load_record()
        if record.submission != SubmissionState.SUBMITTED.value:
            return
        record.submission = SubmissionState.UNCERTAIN.value
        record.add_notice("interrupted")
        if record.pending_widget:
            record.pending_widget = False
            record.add_notice("widget_unshown")
        self._save_record()

    async def _dispatch_command(
        self, content: str
    ) -> tuple[Optional[AgenticResult], str]:
        if not (content and self._commands.is_command(content)):
            return None, content
        try:
            response = await self._commands.dispatch(content)
        except UnknownCommand:
            response = None
        if response is None:
            return None, content
        self.add_message("user", content)
        self.add_message("assistant", response)
        followup = self._consume_pending_followup()
        if followup is None:
            self._save_record()
            return AgenticResult(
                text=response, completed_actions=[], iterations_used=0
            ), content
        return None, followup

    async def _recover_pending_widget(
        self, cb: "_HostCallbacks"
    ) -> AgenticResult | AfterAnswer:
        pending, self._pending_widget_result = self._pending_widget_result, None
        self._withdraw_notices("widget_cancelled")  # the question is answered after all
        recovered = await widget_core.recover(self, pending)
        self._set_pending_widget_flag(False)
        if recovered.bindings is None:
            return AgenticResult(
                text="",
                completed_actions=[],
                iterations_used=0,
                has_conversation_tool=True,
            )
        answer = await self._after_widget_answer(
            recovered.tools, recovered.then_run, recovered.bindings
        )
        if answer.dashboard_handoff:
            self._hand_off_answer(answer)
            await cb.turn_complete(1)
            return AgenticResult(text="", completed_actions=[], iterations_used=0)
        if answer.async_dispatched:
            self._defer_background_answer(answer)
        return answer

    async def _vendor_turns(
        self,
        actor: SessionActor,
        content: str,
        origin: TurnOrigin,
        vendor_command: bool,
        interactive: Any,
        cb: "_HostCallbacks",
        state: _CallState,
    ) -> AgenticResult:
        limit = self.max_vendor_turns_per_call
        limit = limit if limit and limit > 0 else _SAFETY_CEILING
        exhausted = False
        for remaining in range(limit, 0, -1):
            if self._paused:
                return await self._paused_result(state, cb)
            actor, turn = await self._run_vendor_turn(
                actor, content, origin, vendor_command, interactive, cb, state
            )
            if self._paused or turn.stop_cause is StopCause.PAUSE:
                return await self._paused_result(state, cb)
            if not turn.pending_widgets:
                break
            outcome = await self._present_widgets(turn, cb, state)
            if outcome is None:
                tool = turn.pending_widgets[0].tool
                self._touch_runtime()
                return self._result(
                    state, has_conversation_tool=True, conversation_tool=tool
                )
            kind, payload = outcome
            vendor_command = False
            if kind == "retry":
                content, origin = payload, TurnOrigin.VALIDATION_RETRY
                continue
            if kind == "handoff":
                self._hand_off_answer(payload)
                break
            state.actions.extend(_answer_actions(payload))
            if payload.async_dispatched:
                self._defer_background_answer(payload)
                state.turn_number = await cb.new_turn(
                    state.turn_number, payload.response_text
                )
                break
            content, origin = payload.message(), TurnOrigin.WIDGET_ANSWER
            if remaining == 1:
                continue  # no vendor turn of this call is left to send it
            new_turn = await cb.new_turn(state.turn_number, payload.response_text)
            if new_turn != state.turn_number:
                state.turn_number = new_turn
                await cb.turn_boundary(new_turn)
        else:
            self._defer_unsent(content, origin, limit)
            exhausted = True
        return await self._complete_call(state, cb, exhausted_max_iterations=exhausted)

    async def _complete_call(
        self, state: _CallState, cb: "_HostCallbacks", **extra: Any
    ) -> AgenticResult:
        self._touch_runtime()
        await cb.turn_complete(state.iteration)
        return self._result(state, **extra)

    async def _run_vendor_turn(
        self,
        actor: SessionActor,
        content: str,
        origin: TurnOrigin,
        vendor_command: bool,
        interactive: Any,
        cb: "_HostCallbacks",
        state: _CallState,
    ) -> tuple[SessionActor, TurnScope]:
        """Submit one vendor turn; a session the vendor reports missing is
        replaced by a fresh one (with a recap) and the turn re-submitted once
        — safe because the vendor never accepted it."""
        self._mirror_turn_text(content)
        for attempt in (0, 1):
            turn, request = self._prepare_vendor_turn(
                content,
                origin,
                vendor_command,
                interactive,
                state.turn_number,
            )
            try:
                text = await self._stream_turn(actor, request, turn, cb, state)
            except VendorTurnFailed as exc:
                if not (exc.session_missing and attempt == 0):
                    raise
                logger.warning(
                    "Native session %s: the vendor no longer has the session; "
                    "re-submitting on a fresh one",
                    self.conversation_key,
                )
                self._lose_session(
                    self._load_record(), "the vendor no longer has this session"
                )
                actor = await self._ensure_session(
                    state.turn_number, allow_rewind=False
                )
                continue
            self._commit_vendor_turn(turn, state, text)
            await cb.prompt_rendered(text)
            return actor, turn
        raise AssertionError("unreachable")

    def _mirror_turn_text(self, content: str) -> None:
        """Plan §6.1 step 5.2: the UI mirror (display and recaps, never model
        input) records the raw text of every vendor turn — user input, widget
        answers, retries. A host that records user input itself already ends
        the mirror with it (OpenStartup appends the message to its session and
        hands the history over through ``set_messages`` before the turn), so
        it is not recorded twice."""
        last = self._messages[-1] if self._messages else {}
        if last.get("role") == "user" and last.get("content") == content:
            return
        self.add_message("user", content)

    def _prepare_vendor_turn(
        self,
        content: str,
        origin: TurnOrigin,
        vendor_command: bool,
        interactive: Any,
        turn_number: int,
    ) -> tuple[TurnScope, TurnRequest]:
        prepare_sop_for_turn(self.sop_controller, prompt_renderer=self.prompt_renderer)
        record = self._load_record()
        channel = self._l2_channel()
        if vendor_command:
            l2_text, l2_hash, upto = "", record.l2_hash, record.outbox_cursor
        else:
            l2_text, l2_hash, upto = self._compose_l2(
                origin, current=content, turn=turn_number, channel=channel
            )
        turn = TurnScope(
            run_ctx=active_run_context(),
            interactive=interactive,
            turn_number=turn_number,
            origin=origin,
        )
        request = self._turn_request(content, l2_text, vendor_command, channel)
        turn.l2_text = l2_text if request.channel is L2Channel.HOOK else ""
        turn.outbox_upto = upto if l2_text else record.outbox_cursor
        turn.l2_hash = l2_hash if l2_text else ""
        self._record_manifest(content, l2_text, request.channel)
        return turn, request

    def _commit_vendor_turn(
        self, turn: TurnScope, state: _CallState, text: str
    ) -> None:
        record = self._load_record()
        if turn.l2_text and self._prompt_hook_turn is not turn:
            self._keep_l2_due(turn, record)
        if turn.l2_hash:
            record.l2_generation += 1
        if turn.compacted:
            # What this turn delivered may be summarized away: the next turn
            # re-sends the state.
            record.l2_hash = ""
        elif turn.l3_state_hash or turn.l2_hash:
            record.l2_hash = turn.l3_state_hash or turn.l2_hash
        record.mark_delivered(turn.outbox_upto)
        record.last_turn = max(record.last_turn, state.turn_number)
        record.last_round = state.iteration
        record.pending_widget = bool(turn.pending_widgets)
        self._save_record()
        state.actions.extend(turn.completed_actions)
        state.final_text = text
        state.vendor_turns += 1
        self.add_message("assistant", text)

    def _keep_l2_due(self, turn: TurnScope, record: Any) -> None:
        """The vendor completed the turn without running its prompt hook, so
        the L2 handed to that hook never reached it: the state and notices
        stay due for the next turn. Only a slash command does this (the hook
        is otherwise required): Claude Code runs its local commands
        (``/context``, ``/cost``, …) without ``UserPromptSubmit`` and without
        the model; the backend passes through the ones it lists without an L2
        (``slash_passthrough``), and this covers the others."""
        logger.info(
            "Native session %s: the agent ran a command without its prompt hook; "
            "the turn context stays due",
            self.conversation_key,
        )
        turn.l2_hash = ""
        turn.outbox_upto = record.outbox_cursor

    def _is_vendor_command(self, content: str) -> bool:
        """A command the backend passes to the vendor verbatim and without an
        L2 (plan §3.1): one the vendor runs itself without its prompt hook.
        Any other ``/…`` text is an ordinary user message: Claude Code treats
        an unknown name or a path as text, and expands the user's commands
        and skills into a prompt that its ``UserPromptSubmit`` hook sees
        (claude 2.1.289, CLI and SDK)."""
        if not content.startswith("/"):
            return False
        token = content[1:].split(None, 1)[0] if len(content) > 1 else ""
        return token in self._backend_caps().slash_passthrough

    def _l2_channel(self) -> L2Channel:
        channel = self._backend_caps().preferred_l2_channel(
            self.backend.l2_envelope_allowed
        )
        if channel is None:
            raise NativeCapabilityError(
                f"{self.backend.kind} has no qualified channel for per-turn context; "
                "set l2_envelope_allowed to opt into a labelled envelope."
            )
        return channel

    def _turn_request(
        self, content: str, l2_text: str, vendor_command: bool, channel: L2Channel
    ) -> TurnRequest:
        user_text = content if vendor_command else neutralize_host_tags(content)
        if l2_text and channel is L2Channel.ENVELOPE:
            user_text = f"{l2_text}\n{user_text}"
        return TurnRequest(text=user_text, l2_text=l2_text, channel=channel)

    def _record_manifest(self, content: str, l2_text: str, channel: L2Channel) -> None:
        """Plan §6.1 step 5.1: what AF sends on this vendor turn ("View
        Prompt"), each lane labelled with the route the backend declares.
        Recorded before the turn is submitted, so a turn that does not
        complete keeps it (``_mark_manifest_outcome``)."""
        record = self._load_record()
        l1_path = self.session_dir() / f"l1_{record.generation}.md"
        l1_text = l1_path.read_text(encoding="utf-8") if l1_path.exists() else ""
        caps = self._backend_caps()
        routes = caps.prompt_routes(channel)
        has_tools = caps.caller_tools is not CallerTools.NONE
        self._last_rendered_prompt = (
            f"{_MANIFEST_HEADER}\n"
            f"## Session instructions — {routes.l1}\n{l1_text}\n"
            f"## Turn context — {routes.l2}\n"
            f"{l2_text or '(unchanged — not sent)'}\n"
            f"## User message — {routes.user}\n{content}\n"
            f"## State updates — {routes.l3}\n"
            f"{'(added during the turn)' if has_tools else '(none)'}\n"
        )
        self._last_template_source = "conversation_native/main/{session,turn}"
        self._last_template_feed = {
            "backend": caps.kind,
            "l1_core_hash": record.l1_core_hash,
            "l1_route": routes.l1,
            "l2_channel": channel.value,
            "l2_route": routes.l2,
            "user_route": routes.user,
            "l3_route": routes.l3,
        }
        self._last_template_config = {
            "rendering": {
                "structural_xml_tags": [
                    "af_context",
                    "SOPDescription",
                    "SOPStatus",
                    "SOPNextStepGuidance",
                    "SOPPhases",
                ]
            }
        }

    def _mark_manifest_outcome(self, submission: str, detail: str) -> None:
        """A vendor turn that did not complete keeps its manifest as the last
        prompt data ("View Prompt" of a failed, stalled or cancelled turn),
        which states how the turn ended. A completed turn's manifest carries
        no outcome."""
        head = f"{_MANIFEST_HEADER}\n"
        if not self._last_rendered_prompt.startswith(head):
            return
        outcome, meaning = _MANIFEST_OUTCOMES[submission]
        self._last_rendered_prompt = (
            f"{head}## Turn outcome — {outcome}\n{meaning}\n{detail}\n"
            f"{self._last_rendered_prompt[len(head) :]}"
        )
        self._last_template_feed["turn_outcome"] = outcome
        self._last_template_feed["turn_outcome_detail"] = detail

    # ------------------------------------------------------------------
    # Streaming one vendor turn
    # ------------------------------------------------------------------

    async def _stream_turn(
        self,
        actor: SessionActor,
        request: TurnRequest,
        turn: TurnScope,
        cb: "_HostCallbacks",
        state: _CallState,
    ) -> str:
        record = self._load_record()
        record.submission = SubmissionState.PREPARED.value
        self._save_record()
        rounds = _RoundTracker(cb, turn, state)
        end: Optional[TurnEnd] = None
        error: Optional[VendorError] = None
        self._current_turn = turn
        try:
            async for event in self._watch_events(actor, request, turn):
                if not turn.accepted and _vendor_produced(event):
                    turn.accepted = True
                    record.submission = SubmissionState.SUBMITTED.value
                    if record.vendor_session_id:
                        # The vendor holds the session now: later turns resume
                        # it even if this one never completes.
                        record.status = "active"
                    self._save_record()
                end, error = await self._handle_event(
                    event, rounds, turn, record, end, error
                )
        except VendorTurnStalled as exc:
            error = VendorError(message=str(exc))
        except asyncio.CancelledError:
            self._mark_interrupted(actor, turn)
            await rounds.abort()
            raise
        except BaseException:
            await rounds.abort()
            raise
        finally:
            self._current_turn = None
        last_display = await rounds.finish()
        if error is not None or end is None or end.is_error:
            self._fail_turn(turn, error, end)
        record.vendor_session_id = end.session_id or record.vendor_session_id
        record.status = "active"
        record.submission = SubmissionState.COMMITTED.value
        if rounds.last_uuid:
            record.turn_boundaries[str(turn.turn_number)] = rounds.last_uuid
        record.last_meta = {
            "stop_reason": end.stop_reason,
            "num_turns": end.num_turns,
            "total_cost_usd": end.total_cost_usd,
            "usage": dict(end.usage or {}),
        }
        return last_display or strip_host_blocks(end.result_text or "")

    async def _handle_event(self, event, rounds, turn, record, end, error):
        if isinstance(event, TextDelta):
            await rounds.on_delta(event)
        elif isinstance(event, MessageEnd):
            if event.af_tool_use_ids:
                turn.note_af_message(event.af_tool_use_ids, message_id=event.message_id)
            await rounds.on_message_end(event)
        elif isinstance(event, SessionStarted):
            self._on_session_started(record, event)
        elif isinstance(event, Compaction):
            self.on_compaction()
        elif isinstance(event, TurnEnd):
            end = event
        elif isinstance(event, VendorError):
            error = event
        return end, error

    def _on_session_started(self, record: Any, event: SessionStarted) -> None:
        if (
            event.replaced
            and record.vendor_session_id
            and event.session_id != record.vendor_session_id
        ):
            logger.warning(
                "Native session %s: the vendor started a new session instead of "
                "resuming %s; earlier turns are not in its memory",
                self.conversation_key,
                _short(record.vendor_session_id),
            )
            if self.on_session_loss == "recap":
                record.add_notice("recap")
            record.l2_hash = ""
        record.vendor_session_id = event.session_id
        record.status = "active"

    async def _watch_events(
        self, actor: SessionActor, request: TurnRequest, turn: TurnScope
    ) -> AsyncIterator[Any]:
        """The actor's event stream with a stall watchdog: no event for
        ``vendor_stall_timeout_s`` while no AF tool is running interrupts the
        turn (built-in tool runs still count; the timeout is generous)."""
        stream = actor.run_turn(request).__aiter__()
        timeout = self.vendor_stall_timeout_s
        try:
            while True:
                wait = (
                    timeout
                    if timeout and timeout > 0 and not turn.open_af_tool_uses
                    else None
                )
                try:
                    event = await asyncio.wait_for(stream.__anext__(), wait)
                except StopAsyncIteration:
                    return
                except asyncio.TimeoutError:
                    raise VendorTurnStalled(
                        f"{self.backend.kind} produced no events for {timeout:.0f}s"
                    )
                yield event
        finally:
            await stream.aclose()

    def _fail_turn(
        self, turn: TurnScope, error: Optional[VendorError], end: Optional[TurnEnd]
    ) -> None:
        record = self._load_record()
        session_missing = bool(error is not None and error.session_missing)
        # The turn may have reached the vendor if the vendor produced anything
        # for it, or if the failure happened after submission.
        reached = not session_missing and (
            turn.accepted or error is None or error.submitted
        )
        if reached:
            record.submission = SubmissionState.UNCERTAIN.value
            record.add_notice("turn_failed")
        else:
            record.submission = SubmissionState.PREPARED.value
        self._save_record()
        detail = (
            error.message
            if error
            else ("; ".join(end.errors) if end else "no result from the agent")
        )
        self._mark_manifest_outcome(record.submission, detail)
        raise VendorTurnFailed(
            f"{self.backend.kind} turn failed: {detail}",
            session_missing=session_missing,
        )

    def _mark_interrupted(self, actor: SessionActor, turn: TurnScope) -> None:
        record = self._load_record()
        acknowledged = actor.last_outcome.acknowledged
        record.submission = (
            SubmissionState.INTERRUPTED.value
            if acknowledged
            else SubmissionState.UNCERTAIN.value
        )
        record.add_notice("interrupted")
        if turn.pending_widgets:
            # Widgets are presented when the turn ends, which it never did.
            record.pending_widget = False
            record.add_notice("widget_unshown")
        self._save_record()
        self._mark_manifest_outcome(
            record.submission,
            "The host cancelled the turn."
            if acknowledged
            else "The host cancelled the turn; the agent did not confirm the stop.",
        )

    async def _paused_result(
        self, state: _CallState, cb: "_HostCallbacks"
    ) -> PausedResult:
        """Cooperative pause (host-set ``_paused``), checked at vendor-turn
        boundaries exactly where the text-protocol orchestrator checks it."""
        self._touch_runtime()
        await cb.turn_complete(state.iteration)
        record = self._load_record()
        return PausedResult(
            text=state.final_text,
            completed_actions=list(state.actions),
            iterations_used=state.iteration,
            pause_state=self.export_state(
                turn_number=state.turn_number, iteration=state.iteration
            ),
            native_meta=self._native_meta(record),
        )

    def _touch_runtime(self) -> None:
        if self._lease_key is not None:
            self.runtime_manager.touch(self._lease_key)

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------

    async def _present_widgets(
        self, turn: TurnScope, cb: "_HostCallbacks", state: _CallState
    ) -> Optional[tuple[str, Any]]:
        tools = [w.tool for w in turn.pending_widgets]
        then_run = [a for w in turn.pending_widgets for a in w.then_run]
        try:
            group_and_validate(tools)
        except GroupValidationError as exc:
            return (
                "retry",
                f"{_VALIDATION_RETRY_PREFIX} {exc} Re-ask the "
                "questions fixing this: put independent questions in the SAME "
                "parallel_group, and ask dependent questions in a later turn.",
            )
        if turn.interactive is None:
            return None
        context = turn.pending_widgets[0].round_context
        if context is not None:
            cb.apply_round_context(context)
        elif not _widget_message_had_round(turn):
            # Its message was never reported, so no round holds the widget.
            round_ = await cb.open_round(state.iteration, state.turn_number)
            await round_.close("")
            state.iteration += 1
        try:
            collected = await widget_core.present_and_collect(
                self,
                tools,
                state.final_text,
                interactive=turn.interactive,
                then_run=then_run,
                turn_number=state.turn_number,
                iteration=state.iteration,
            )
        except asyncio.CancelledError:
            # The host abandoned the question (e.g. the user sent a new
            # message). Withdrawn again if the host re-arms this widget.
            self._queue_notice("widget_cancelled")
            raise
        self._set_pending_widget_flag(False)
        if collected is None:
            return None
        answer = await self._after_widget_answer(tools, then_run, collected)
        return ("handoff" if answer.dashboard_handoff else "answer", answer)

    def _hand_off_answer(self, answer: AfterAnswer) -> None:
        """A dashboard took over the work the answer asked for, so no vendor
        turn reports the answer: the mirror records it (as the text-protocol
        transcript does) and the next turn's L2 delivers it once."""
        self.add_message("user", answer.response_text)
        self._queue_notice("dashboard_handoff", body=answer.response_text)

    def _defer_background_answer(self, answer: AfterAnswer) -> None:
        """D6: a bundled action of the answer started in the background, which
        ends the host turn (as in the text-protocol orchestrator, and as such
        an AF tool call ends the vendor turn). No vendor turn reports the
        answer: the mirror records it and the next turn's L2 delivers it once,
        ahead of the action's completion. Called before anything awaits, so
        a fast completion cannot be announced first."""
        message = answer.message()
        self._mirror_turn_text(message)
        self._queue_notice("background_answer", body=message)

    def _defer_unsent(self, content: str, origin: TurnOrigin, limit: int) -> None:
        """The call used its last vendor turn (``max_vendor_turns_per_call``)
        before ``content`` — a widget answer already applied, or the error of
        a widget batch never shown — reached the agent. The mirror records it
        as for any vendor turn, and the next turn's L2 delivers it once (the
        text protocol's next prompt carries it in its history)."""
        logger.warning(
            "Native session %s: the call reached its limit of %d vendor turns "
            "before the %s reached the agent; the next turn delivers it",
            self.conversation_key,
            limit,
            origin.value,
        )
        self._mirror_turn_text(content)
        self._queue_notice("unsent_turn", body=content, origin=origin.value)

    def _withdraw_notices(self, notice_type: str) -> None:
        record = self._load_record()
        pending = record.pending_notices()
        kept = [n for n in pending if n.get("type") != notice_type]
        if len(kept) != len(pending):
            record.outbox = record.outbox[: record.outbox_cursor] + kept
            self._save_record()

    def _set_pending_widget_flag(self, value: bool) -> None:
        record = self._load_record()
        if record.pending_widget != value:
            record.pending_widget = value
            self._save_record()

    async def _after_widget_answer(
        self,
        tools: Sequence[Any],
        then_run: Sequence[Any],
        bindings: Any,
        *,
        is_live: Optional[Callable[[], bool]] = None,
    ) -> AfterAnswer:
        """``widget_core.after_answer`` with this orchestrator's wiring, for an
        answer given between vendor turns (live or recovered) or inside one —
        then ``is_live`` says whether a bundled action's result may still be
        applied when it returns (the yolo path)."""
        return await widget_core.after_answer(
            self,
            tools,
            then_run,
            bindings,
            run_ctx=active_run_context(),
            on_async_done=self._on_async_tool_done,
            on_applied=self._then_run_applied,
            is_live=is_live,
        )

    def _then_run_applied(self, tool_name: str, outcome: ToolOutcome) -> None:
        self.record_action(tool_name, outcome.text)
        if outcome.is_async and self._current_turn is not None:
            self._current_turn.request_stop(StopCause.ASYNC_DISPATCH)

    def _result(self, state: _CallState, **extra: Any) -> AgenticResult:
        record = self._load_record()
        return AgenticResult(
            text=state.final_text,
            completed_actions=list(state.actions),
            iterations_used=state.iteration,
            last_rendered_prompt=self._last_rendered_prompt,
            last_template_source=self._last_template_source,
            last_template_feed=dict(self._last_template_feed),
            last_template_config=dict(self._last_template_config),
            native_meta=self._native_meta(record),
            **extra,
        )

    def _native_meta(self, record: Any) -> dict[str, Any]:
        return {
            "backend": self.backend.kind,
            "session_ref": _short_hash(record.vendor_session_id),
            "generation": record.generation,
            "submission": record.submission,
            **record.last_meta,
        }


def _answer_actions(answer: AfterAnswer) -> list[CompletedAction]:
    """The bundled actions a widget answer given between vendor turns ran, as
    completed actions of the host call (inside a turn the turn records them)."""
    return [
        CompletedAction(tool=r.name, summary=str(r.outcome.text)[:200])
        for r in answer.then_run_results
    ]


def _marker(path: str) -> str:
    return f"\n[… cut here; the full text is in {path}]\n"


def _cut(text: str, size: int, path: str, *, keep_tail: bool) -> str:
    """``text`` cut to at most ``size`` units around a marker naming ``path``
    (the full text): its head, or for ``keep_tail`` its first line (what the
    text is) and its end."""
    marker = _marker(path)
    keep = size - utf16_len(marker)
    if not keep_tail:
        return utf16_head(text, keep) + marker
    first, _, rest = text.partition("\n")
    first = utf16_head(first, keep // 2)
    return first + marker + utf16_tail(rest, keep - utf16_len(first))


def _bundle_text(count: int, path: str) -> str:
    return (
        f"{count} more host notice(s) did not fit in this context; read them in {path}."
    )


def _vendor_produced(event: Any) -> bool:
    """Events that prove the vendor accepted the turn's user message.

    A session confirmation does not: Claude Code reports the session at
    ``init`` also for a prompt its ``UserPromptSubmit`` hooks blocked, and a
    blocked prompt never enters the transcript, whereas one the hooks let
    through is already in it at ``init`` (claude 2.1.288). A backend error
    states itself whether the turn was submitted."""
    if isinstance(event, SessionStarted):
        return False
    return not (isinstance(event, VendorError) and not event.submitted)


def _widget_message_had_round(turn: TurnScope) -> bool:
    """Whether the assistant message that asked the turn's first question got
    a round: every main-thread message reporting AF tool uses gets one
    (``_RoundTracker.on_message_end``), also when the host's round context is
    None. A call no hook announced (Devmate) goes with any such message."""
    if turn.widget_tool_use is not None:
        return turn.widget_tool_use in turn.known_af_tool_uses
    return bool(turn.known_af_tool_uses)


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12] if value else ""


def _short(value: str) -> str:
    return _short_hash(value) or "-"


def _manifest_fingerprint(manifest: Sequence[Any]) -> str:
    """Names and input schemas of an AF tool set."""
    tools = sorted(
        (spec.name, json.dumps(spec.input_schema, sort_keys=True, default=str))
        for spec in manifest
    )
    return hashlib.sha256(json.dumps(tools).encode()).hexdigest()[:16]


class _RoundTracker:
    """Groups vendor events into host rounds: one round per main-thread
    assistant message (by message id). A message's text is streamed from its
    deltas, or pushed whole from its message events when no delta came."""

    def __init__(
        self, cb: "_HostCallbacks", turn: TurnScope, state: _CallState
    ) -> None:
        self._cb = cb
        self._turn = turn
        self._state = state
        self._round: Optional[_Round] = None
        self._message_id: Optional[str] = None
        self._text = ""
        self._streamed = False
        self.last_display = ""
        self.last_uuid: Optional[str] = None

    async def _open(self, message_id: str) -> None:
        self._round = await self._cb.open_round(
            self._state.iteration, self._turn.turn_number
        )
        self._turn.round_context = self._round.context
        self._message_id = message_id
        self._text = ""
        self._streamed = False

    async def _close(self) -> None:
        if self._round is None:
            return
        self.last_display = await self._round.close(self._text)
        self._state.iteration += 1
        self._round = None
        self._message_id = None

    async def _switch_to(self, message_id: str) -> None:
        if self._round is not None and message_id != self._message_id:
            await self._close()

    async def on_delta(self, event: TextDelta) -> None:
        await self._switch_to(event.message_id)
        if self._round is None:
            await self._open(event.message_id)
        self._streamed = True
        self._text += event.text
        await self._round.push(event.text)

    async def on_message_end(self, event: MessageEnd) -> None:
        await self._switch_to(event.message_id)
        self.last_uuid = event.message_uuid or self.last_uuid
        text = event.text or ""
        if self._round is None and (text.strip() or event.af_tool_use_ids):
            # A widget-only message still gets its own round (durable marker).
            await self._open(event.message_id)
        if self._round is not None and text and not self._streamed:
            piece = f"\n\n{text}" if self._text else text
            self._text += piece
            await self._round.push(piece)

    async def finish(self) -> str:
        await self._close()
        return self.last_display

    async def abort(self) -> None:
        """The turn ended abnormally: stop streaming the open round without
        completing it."""
        if self._round is not None:
            await self._round.abort()
            self._round = None
            self._message_id = None


class _Round:
    """One host round = one main-thread assistant message streamed to the UI."""

    def __init__(
        self, owner: "_HostCallbacks", iteration: int, turn_number: int, context: Any
    ) -> None:
        self._owner = owner
        self.iteration = iteration
        self.turn_number = turn_number
        self.context = context
        self._queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None
        interactive = owner.interactive
        if interactive is not None and hasattr(interactive, "stream_token_batches"):
            self._queue = asyncio.Queue()
            self._task = asyncio.ensure_future(
                interactive.stream_token_batches(
                    self._tokens(),
                    owner.session_id,
                    send_stream_end=False,
                    turn_number=turn_number,
                )
            )

    async def _tokens(self):
        assert self._queue is not None
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            yield chunk, {"turn_number": self.turn_number}

    async def push(self, text: str) -> None:
        if self._queue is not None and text:
            await self._queue.put(text)

    async def close(self, full_text: str) -> str:
        raw = full_text
        if self._queue is not None and self._task is not None:
            await self._queue.put(None)
            streamed = await self._task
            if isinstance(streamed, str) and streamed:
                raw = streamed
        display = strip_host_blocks(full_text).strip()
        await self._owner.round_complete(
            self.iteration, self.turn_number, raw, full_text, display
        )
        return display

    async def abort(self) -> None:
        if self._task is None or self._task.done():
            return
        self._task.cancel()
        # wait() rather than awaiting the task: a cancellation of the caller
        # must still propagate.
        await asyncio.wait({self._task})
        if not self._task.cancelled() and self._task.exception() is not None:
            logger.debug("Round stream ended with %r", self._task.exception())


class _HostCallbacks:
    """Host callbacks with the same signatures and failure isolation as the
    text-protocol orchestrator."""

    def __init__(
        self,
        owner,
        interactive,
        session_id,
        on_new_turn,
        on_prompt_rendered,
        on_turn_complete,
        on_round_start,
        on_round_complete,
    ) -> None:
        self.owner = owner
        self.interactive = interactive
        self.session_id = session_id
        self._on_new_turn = on_new_turn
        self._on_prompt_rendered = on_prompt_rendered
        self._on_turn_complete = on_turn_complete
        self._on_round_start = on_round_start
        self._on_round_complete = on_round_complete

    async def new_turn(self, turn_number: int, user_input: str) -> int:
        if self._on_new_turn is None:
            return turn_number
        try:
            new = await self._on_new_turn(turn_number, user_input)
            return turn_number if new is None else new
        except Exception as exc:
            logger.warning("[native] on_new_turn error: %s", exc)
            return turn_number

    async def turn_boundary(self, turn_number: int) -> None:
        if self.interactive is not None and hasattr(
            self.interactive, "send_turn_boundary"
        ):
            await self.interactive.send_turn_boundary(
                self.session_id,
                turn_number=turn_number,
                cache_folder=self.owner.cache_folder or "",
            )

    async def open_round(self, iteration: int, turn_number: int) -> _Round:
        context = None
        if self._on_round_start is not None:
            try:
                context = await self._on_round_start(iteration, turn_number)
            except Exception as exc:
                logger.warning("[native] on_round_start error: %s", exc)
        if context:
            self.apply_round_context(context)
        return _Round(self, iteration, turn_number, context)

    def apply_round_context(self, context: Any) -> None:
        """Apply a round context as the text-protocol orchestrator does: its
        ``cache_folder`` (when set) becomes the host's ``cache_folder``, the
        round's artifact directory that turn boundaries carry, and the whole
        context goes to the interactive's ``set_round_context``. The native
        orchestrator writes no streaming cache there: the vendor keeps the
        transcript, and spilled tool results stay in the session directory,
        where later turns read them by path. Other keys are the host's own."""
        try:
            cache_folder = (
                context.get("cache_folder") if isinstance(context, dict) else None
            )
            if cache_folder:
                self.owner.cache_folder = cache_folder
            if self.interactive is not None and hasattr(
                self.interactive, "set_round_context"
            ):
                self.interactive.set_round_context(context)
        except Exception as exc:
            logger.warning("[native] set_round_context error: %s", exc)

    async def round_complete(
        self, iteration: int, turn_number: int, raw: str, clean: str, display: str
    ) -> None:
        if self._on_round_complete is None:
            return
        try:
            await self._on_round_complete(
                self.owner,
                iteration,
                turn_number,
                raw,
                clean,
                display,
                ConversationResponse(text=display),
            )
        except Exception as exc:
            logger.warning("[native] on_round_complete error: %s", exc)

    async def prompt_rendered(self, text: str) -> None:
        if self._on_prompt_rendered is None:
            return
        try:
            await self._on_prompt_rendered(self.owner, text)
        except Exception as exc:
            logger.warning("[native] on_prompt_rendered error: %s", exc)

    async def turn_complete(self, iteration: int) -> None:
        if self._on_turn_complete is None:
            return
        try:
            await self._on_turn_complete(iteration)
        except Exception as exc:
            logger.warning("[native] on_turn_complete error: %s", exc)

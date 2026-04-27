"""ConversationalFlowNodeAdapter — thin adapter for flow inferencer slots.

Lets a ConversationalInferencer act as a child node in BTA/PTI/LWI by
inheriting InferencerBase and overriding _ainfer() to delegate to
run_agentic_loop(). The flow-level template rendering happens upstream
in _ainfer_single(), so inference_input arrives already rendered.

Use in SINGLE-THREADED slots only (BTA breakdown/aggregator, PTI
planner/executor/analyzer, LWI sequential steps). Do NOT use inside
a parallel worker — that would cause N concurrent user dialogs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
from typing import Any, Callable, ClassVar, Optional

import attr
from attr import attrib, attrs

from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
    AgenticDynamicContext,
    AgenticResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.ui.interactive_base import InteractiveBase

logger = logging.getLogger(__name__)


def _default_output_extractor(result: AgenticResult) -> str:
    """Default: extract .text from AgenticResult."""
    return result.text


def _default_empty_predicate(x: Any) -> bool:
    """Conservative default: only None and empty/whitespace-only strings.
    For breakdown-style use cases where output_extractor returns a parsed
    list, use zero_list_is_empty instead."""
    if x is None:
        return True
    if isinstance(x, str):
        return not x.strip()
    return False


def zero_list_is_empty(x: Any) -> bool:
    """Predicate for breakdown-style use cases where output_extractor
    returns a parsed list. Treats None, empty string, and empty list
    as empty."""
    if x is None:
        return True
    if isinstance(x, str):
        return not x.strip()
    if isinstance(x, list):
        return len(x) == 0
    return False


@attrs(slots=False, kw_only=True)
class ConversationalFlowNodeAdapter(InferencerBase):
    """Flow-side adapter that lets a ConversationalInferencer act as a node.

    Inherits InferencerBase so it plugs into BTA/PTI/LWI as a standard
    child inferencer with zero changes to flow code. Overrides _ainfer()
    to delegate to run_agentic_loop().

    The flow-level template rendering happens in _ainfer_single() (upstream
    of _ainfer), so inference_input arrives already rendered. This becomes
    the opening user message for the conversational session.

    Use in SINGLE-THREADED slots only (BTA breakdown/aggregator, PTI
    planner/executor/analyzer, LWI sequential steps). Do NOT use inside
    a parallel worker — that would cause N concurrent user dialogs.

    Attributes:
        conversational_inferencer: The wrapped ConversationalInferencer.
        output_extractor: Callable(AgenticResult) -> Any. Default extracts
            .text. Override to pull completed_actions or custom projections.
            For BTA breakdown, can wrap _parse_json_subtasks and return
            empty list for "agent forgot the format" → triggers fallback.
        fallback_inferencer: Optional InferencerBase to use when the
            conversational session fails (timeout, disconnect, empty output,
            unresolved widget, exhausted iterations).
        session_timeout: Seconds to wait for the conversational session.
            None = wait indefinitely.
        empty_predicate: Callable(Any) -> bool. Determines whether the
            extracted output is "empty" and should trigger fallback.
            Default checks for None and empty/whitespace-only strings.
        reset_between_invocations: Whether to call reset_for_flow_invocation()
            before each invocation. Default True. Set False only when a flow
            intentionally carries conversational state across iterations.
        interactive: Optional InteractiveBase injected by the flow before
            invocation (mirrors BTA's existing worker.interactive pattern).
    """

    # D2: Declare default nested _target_ aliases for fields.
    # When a nested YAML block under this class lacks _target_, the walk
    # checks this mapping first (highest precedence over type-based inference).
    __yaml_default_nested__: ClassVar[dict[str, str]] = {
        "conversational_inferencer": "Conversational",
    }

    # Override parent's max_retry default: conversational sessions are stateful;
    # a retry after reset_for_flow_invocation() wipes the user's conversation.
    max_retry: int = attrib(default=0)

    conversational_inferencer: ConversationalInferencer = attrib(
        validator=attr.validators.instance_of(ConversationalInferencer)
    )
    output_extractor: Callable[[AgenticResult], Any] = attrib(
        default=_default_output_extractor
    )
    fallback_inferencer: Optional[InferencerBase] = attrib(default=None)
    session_timeout: Optional[float] = attrib(default=None)
    empty_predicate: Callable[[Any], bool] = attrib(
        default=_default_empty_predicate
    )
    reset_between_invocations: bool = attrib(default=True)
    interactive: Optional[InteractiveBase] = attrib(default=None)

    # --- Checkpoint / resume ---
    checkpoint_dir: Optional[str] = attrib(default=None)
    session_id: Optional[str] = attrib(default=None)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.checkpoint_dir is not None and self.session_id is None:
            raise ValueError(
                "session_id is required when checkpoint_dir is set. "
                "Provide a stable session_id or derive one via "
                "hashlib.sha256(task.encode()).hexdigest()[:16]."
            )

    def _infer(self, inference_input, inference_config=None, **kw):
        """Sync path is not supported — run_agentic_loop is async-only."""
        raise NotImplementedError(
            f"{type(self).__name__} is async-only; call .ainfer() instead. "
            "run_agentic_loop cannot be driven from a sync context without "
            "breaking the caller's event loop."
        )

    # -----------------------------------------------------------------
    # Checkpoint helpers
    # -----------------------------------------------------------------

    def _session_dir(self) -> Optional[str]:
        """Return the checkpoint directory for this session, or None."""
        if self.checkpoint_dir is None or self.session_id is None:
            return None
        return os.path.join(self.checkpoint_dir, self.session_id)

    def _write_checkpoint_atomic(
        self,
        session_dir: str,
        data: dict[str, Any],
    ) -> None:
        """Write checkpoint.json atomically via write-to-tmp + os.replace."""
        os.makedirs(session_dir, exist_ok=True)
        target = os.path.join(session_dir, "checkpoint.json")
        fd, tmp_path = tempfile.mkstemp(
            dir=session_dir, prefix=".checkpoint_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp_path, target)
        except BaseException:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _write_dynamic_context(self, session_dir: str) -> None:
        """Persist dynamic_context.json alongside the checkpoint."""
        os.makedirs(session_dir, exist_ok=True)
        ctx_dict = self.conversational_inferencer.dynamic_context.to_dict()
        target = os.path.join(session_dir, "dynamic_context.json")
        fd, tmp_path = tempfile.mkstemp(
            dir=session_dir, prefix=".dynctx_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(ctx_dict, f, ensure_ascii=False)
            os.replace(tmp_path, target)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _load_checkpoint(self, session_dir: str) -> Optional[dict[str, Any]]:
        """Load checkpoint.json if it exists, else None.

        Returns None on any read/parse error (corrupted file treated as absent).
        """
        path = os.path.join(session_dir, "checkpoint.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("schema_version") != 1:
                logger.warning(
                    "[ConversationalFlowNodeAdapter] checkpoint %s has unsupported "
                    "schema_version %s — treating as absent",
                    path, data.get("schema_version"),
                )
                return None
            return data
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.warning(
                "[ConversationalFlowNodeAdapter] failed to load checkpoint %s: %s — "
                "treating as absent (fresh session)", path, e,
            )
            return None

    def _load_dynamic_context(self, session_dir: str) -> Optional[AgenticDynamicContext]:
        """Load dynamic_context.json if it exists, else None.

        Returns None on any read/parse error (corrupted file treated as absent).
        """
        path = os.path.join(session_dir, "dynamic_context.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return AgenticDynamicContext.from_dict(json.load(f))
        except (json.JSONDecodeError, OSError, ValueError, KeyError, TypeError) as e:
            logger.warning(
                "[ConversationalFlowNodeAdapter] failed to load dynamic_context %s: %s — "
                "treating as absent", path, e,
            )
            return None

    def _append_diagnostic_log(
        self, session_dir: str, filename: str, entry: dict[str, Any]
    ) -> None:
        """Append a single JSON line to a diagnostic log file."""
        path = os.path.join(session_dir, filename)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("[ConversationalFlowNodeAdapter] diagnostic log write failed: %s", e)

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        """Route to run_agentic_loop with fallback and checkpoint/resume support.

        inference_input arrives already rendered by _ainfer_single's
        _render_prompt() call (the flow-level template). We pass it
        as the content parameter to run_agentic_loop().

        NOTE: inference_config is not forwarded to run_agentic_loop because
        the conversational path manages its own LLM calls internally via
        base_inferencer (configured at construction time). Per-call
        inference_config applies only to the fallback path.
        """
        rendered_task = str(inference_input)

        # Resolve interactive: kwargs wins, then self.interactive
        # Use get (not pop) so kwargs propagates intact to fallback
        interactive = kwargs.get("interactive") or self.interactive

        # --- Resume from checkpoint if available ---
        session_dir = self._session_dir()
        resuming = False
        resume_turn_number = 0
        resume_content = rendered_task

        if session_dir is not None:
            checkpoint = self._load_checkpoint(session_dir)
            if checkpoint is not None:
                status = checkpoint.get("status")
                if status == "completed":
                    # Level 1 short-circuit: session already completed
                    completion_result = checkpoint.get("completion_result")
                    if completion_result is None:
                        logger.warning(
                            "[ConversationalFlowNodeAdapter] session %s marked completed "
                            "but missing completion_result — treating as absent",
                            self.session_id,
                        )
                    else:
                        logger.info(
                            "[ConversationalFlowNodeAdapter] session %s already completed, "
                            "returning cached result", self.session_id,
                        )
                        # completion_result is always JSON-encoded on write
                        # (json.dumps), so json.loads is the symmetric decode.
                        return json.loads(completion_result)

                if status == "in_progress":
                    # Level 2: resume mid-conversation
                    resuming = True
                    resume_turn_number = checkpoint.get("turn_number", 0)
                    loaded_messages = checkpoint.get("messages", [])

                    # Populate inferencer state from checkpoint
                    self.conversational_inferencer._messages = loaded_messages

                    # Load dynamic context
                    dyn_ctx = self._load_dynamic_context(session_dir)
                    if dyn_ctx is not None:
                        self.conversational_inferencer._dynamic_context = dyn_ctx

                    # Compute resume content: last user-role message
                    for msg in reversed(loaded_messages):
                        if msg.get("role") == "user":
                            resume_content = msg["content"]
                            break

                    logger.info(
                        "[ConversationalFlowNodeAdapter] resuming session %s from turn %d",
                        self.session_id, resume_turn_number,
                    )

        # Reset conversational state to prevent leakage between invocations
        # (skip when resuming — state was just loaded from checkpoint)
        if not resuming and self.reset_between_invocations:
            self.conversational_inferencer.reset_for_flow_invocation()

        # Build on_turn_complete callback for checkpoint persistence
        on_turn_complete = None
        if session_dir is not None:
            _prev_msg_count = len(self.conversational_inferencer.get_messages())

            async def on_turn_complete(turn_number: int) -> None:
                nonlocal _prev_msg_count
                # Write checkpoint first (authoritative state), then dynamic_context.
                # A crash between the two leaves dyn_ctx stale by one turn — the
                # worst case is a slightly stale _compressed_history, but no
                # duplicated actions on resume.
                self._write_checkpoint_atomic(session_dir, {
                    "schema_version": 1,
                    "session_id": self.session_id,
                    "initial_content": rendered_task,
                    "status": "in_progress",
                    "turn_number": turn_number,
                    "messages": self.conversational_inferencer.get_messages(),
                    "completion_result": None,
                })
                self._write_dynamic_context(session_dir)

                # Diagnostic logs — extract from new messages added this turn
                current_msgs = self.conversational_inferencer.get_messages()
                new_msgs = current_msgs[_prev_msg_count:]
                _prev_msg_count = len(current_msgs)

                for msg in new_msgs:
                    role = msg.get("role", "")
                    content_text = msg.get("content", "")
                    if role == "assistant":
                        self._append_diagnostic_log(session_dir, "llm_calls.jsonl", {
                            "turn": turn_number,
                            "role": role,
                            "raw_response_length": len(content_text),
                        })
                    elif role == "user" and "[Tool Result:" in content_text:
                        self._append_diagnostic_log(session_dir, "tool_calls.jsonl", {
                            "turn": turn_number,
                            "role": role,
                            "content_length": len(content_text),
                        })

        try:
            content = resume_content if resuming else rendered_task
            coro = self.conversational_inferencer.run_agentic_loop(
                content=content,
                interactive=interactive,
                turn_number=resume_turn_number if resuming else 0,
                on_turn_complete=on_turn_complete,
            )
            if self.session_timeout is not None:
                result = await asyncio.wait_for(coro, self.session_timeout)
            else:
                result = await coro

            # Unresolved states → fallback
            if result.exhausted_max_iterations or result.has_conversation_tool:
                raise _ConversationalUnresolved(
                    f"exhausted={result.exhausted_max_iterations}, "
                    f"pending_widget={result.has_conversation_tool}"
                )

            try:
                extracted = self.output_extractor(result)
            except Exception as extract_err:
                logger.warning(
                    "[ConversationalFlowNodeAdapter] output_extractor raised: %s",
                    extract_err,
                )
                raise _EmptyConversationalOutput() from extract_err

            if self.empty_predicate(extracted):
                raise _EmptyConversationalOutput()

            # Write completed checkpoint
            if session_dir is not None:
                self._write_checkpoint_atomic(session_dir, {
                    "schema_version": 1,
                    "session_id": self.session_id,
                    "initial_content": rendered_task,
                    "status": "completed",
                    "turn_number": result.iterations_used,
                    "messages": self.conversational_inferencer.get_messages(),
                    "completion_result": json.dumps(extracted, default=str),
                })
                self._write_dynamic_context(session_dir)

            return extracted

        except (
            asyncio.TimeoutError,
            _EmptyConversationalOutput,
            _ConversationalUnresolved,
        ) as exc:
            if self.fallback_inferencer is None:
                raise
            logger.warning(
                "[ConversationalFlowNodeAdapter] falling back to %s (%s)",
                type(self.fallback_inferencer).__name__,
                type(exc).__name__,
            )
            # Filter adapter-specific kwargs before forwarding to fallback
            # (plain LLM inferencers don't expect 'interactive')
            fallback_kwargs = {k: v for k, v in kwargs.items()
                               if k not in ("interactive",)}
            return await self.fallback_inferencer.ainfer(
                rendered_task, inference_config, **fallback_kwargs
            )


class _EmptyConversationalOutput(Exception):
    """Raised when the conversational session produces empty output."""


class _ConversationalUnresolved(Exception):
    """Raised when the session exhausted iterations or has an unresolved widget."""

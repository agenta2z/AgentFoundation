# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Experiment Hub controller — OpenTeam-agnostic hub orchestration.

Ported from RankEvolve's ``SessionToolExecutor`` hub methods. Instead of a
``session`` god-object + a file-queue transport, ``HubController`` takes
INJECTED dependencies so it runs inside OpenTeam's single process:

* ``session_id`` / ``session_dir`` / ``session_tasks_dir`` — identity + paths.
* ``workflow_context`` — the shared :class:`WorkflowContext` (task queue lives
  here; the same instance the host's conversational inferencer mutates).
* ``emit_event(session_id, event)`` — async callable the host supplies; every
  ``task_status`` / ``submission_state`` / ``setup_completed`` payload the hub
  used to push through ``interactive._send_response`` now flows through here.
* ``persist()`` — async callable that persists session state to disk after a
  queue mutation (restart-resume).
* ``stream_sink`` — an interactive-like object OpenTeam supplies
  (``TaskWebSocketInteractive``); duck-typed via :class:`StreamSink`. Optional.
* ``exec_task`` — async callable ``(args, queue_task_id) -> ToolExecutionResult``
  that runs a ``/task`` (the DualInferencerBridge/PTI path, which lives in the
  host, NOT the hub). The hub's queue runner delegates the ``task`` /
  ``understand_codebase`` tool branch to it. ``submission_run`` is executed
  by the hub itself.
* ``workflow_target_path`` — the session's codebase-root hint for
  ``${CODEBASE_ROOT}`` resolution.
* ``add_task_ref`` / ``update_task_ref_status`` — optional async callables for
  the conversation-history chip markers (best-effort; default no-op).
* ``track_task`` — optional ``(key, task)`` callable handed every task the
  queue runner starts, so the host can cancel and await the hub's jobs with
  the session (delete, shutdown). The queued jobs live on after the call that
  queued them returns; the hub never starts one once its queue is closed.

Job queue lifecycle: a cancelled job closes the queue — the jobs queued after
it do not start. ``aclose()`` closes it explicitly (cancelling and awaiting the
running jobs, whose subprocess trees are killed); ``join()`` waits for the
queue to run dry, and cancelling that wait closes it.

This module imports NO ``fastapi``/``starlette`` and NO ``rankevolve``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    ToolExecutionResult,
)

logger: logging.Logger = logging.getLogger(__name__)


# Pattern used to extract the codebase root from the session's
# ``workflow_target_path`` when SubmissionRunner substitutes
# ``${CODEBASE_ROOT}`` placeholders in launch.json.
#
# ``**/fbcode`` is the most universal choice — every Meta engineer has their
# checkout's fbcode tree at ``<root>/fbcode/`` regardless of whether
# ``<root>`` is named ``fbsource``, ``fbs_cfr_dev``, etc. Since ``fbcode`` is
# the buck cell name, the match is structurally guaranteed for any source-tree
# path.
_CODEBASE_ROOT_PATTERN: str = "**/fbcode"


# Back-compat: the prompt-template variant directory has been renamed twice.
# Persisted queue entries / hub JSON / session_state may carry either legacy
# value; this map rewrites them at the input boundary so all downstream code
# (completion-handler dispatch, resolver-folder lookup) sees only the current
# name. Logs once per process per legacy hit.
_TEMPLATE_VERSION_RENAMES: dict[str, str] = {
    "submission_script_generator": "experiment_runner_creation",
    "experiment_launcher_maker": "experiment_runner_creation",
}
_LOGGED_LEGACY_RENAMES: set[str] = set()


def _normalize_template_version(v: str) -> str:
    new = _TEMPLATE_VERSION_RENAMES.get(v)
    if new is not None and v != new:
        if v not in _LOGGED_LEGACY_RENAMES:
            _LOGGED_LEGACY_RENAMES.add(v)
            logger.info(
                "template_version / on_complete_handler %r renamed to %r; "
                "please update persisted state.",
                v,
                new,
            )
        return new
    return v


class StreamSink(Protocol):
    """Duck-typed interface for the host-supplied streaming sink.

    OpenTeam supplies a ``TaskWebSocketInteractive``-like object. The hub only
    needs ``_send_response(payload, flags)`` for legacy callers, but the
    canonical path is the injected ``emit_event`` callable; this Protocol
    exists so a host that prefers the sink object can pass one. Never imports
    rankevolve's queue.
    """

    def _send_response(self, payload: dict[str, Any], flags: Any = ...) -> Any: ...


# Type aliases for the injected dependencies.
EmitEvent = Callable[[str, dict[str, Any]], Awaitable[None]]
Persist = Callable[[], Awaitable[None]]
ExecTask = Callable[[dict[str, Any], str], Awaitable[ToolExecutionResult]]
AddTaskRef = Callable[..., Awaitable[None]]
UpdateTaskRefStatus = Callable[[str, str], Awaitable[bool]]
TrackTask = Callable[[str, "asyncio.Task[None]"], None]


class HubController:
    """OpenTeam-agnostic Experiment Hub controller.

    Owns hub creation, submission setup/run, auto-analysis, the per-hub
    implementations sidecar, and the task-queue runner (delegating the
    ``/task`` branch to the injected ``exec_task``).
    """

    def __init__(
        self,
        *,
        session_id: str,
        session_dir: Path,
        session_tasks_dir: Path,
        workflow_context: Any,
        emit_event: EmitEvent,
        persist: Persist | None = None,
        stream_sink: StreamSink | None = None,
        exec_task: ExecTask | None = None,
        workflow_target_path: str = "",
        session_context: dict[str, Any] | None = None,
        add_task_ref: AddTaskRef | None = None,
        update_task_ref_status: UpdateTaskRefStatus | None = None,
        running_task_handles: dict[str, Any] | None = None,
        track_task: TrackTask | None = None,
    ) -> None:
        self._session_id: str = session_id
        self._session_dir: Path = Path(session_dir)
        self._session_tasks_dir: Path = Path(session_tasks_dir)
        self._wc: Any = workflow_context
        self._session_context: dict[str, Any] = dict(session_context or {})
        self._emit_event: EmitEvent = emit_event
        self._persist_cb: Persist | None = persist
        self._stream_sink: StreamSink | None = stream_sink
        self._exec_task: ExecTask | None = exec_task
        self._workflow_target_path: str = workflow_target_path or ""
        self._add_task_ref: AddTaskRef | None = add_task_ref
        self._update_task_ref_status: UpdateTaskRefStatus | None = (
            update_task_ref_status
        )
        # Per-task asyncio.Task handle registry for cancellation. The host may
        # share its own dict; default to a private one. Cancelling a handle
        # closes the queue.
        self._running_task_handles: dict[str, Any] = (
            running_task_handles if running_task_handles is not None else {}
        )
        self._track_task: TrackTask | None = track_task
        # Queue-runner tasks still running (``_start_queue_runner``).
        self._queue_runners: set[asyncio.Task[None]] = set()
        self._queue_closed: bool = False
        # Per-(session, mid) locks for the implementations sidecar.
        self._hub_impl_locks: dict[tuple[str, str], asyncio.Lock] = {}
        # Registry of post-completion handlers for queued tasks. Maps a stable
        # string KEY (persisted on the queue entry as
        # ``entry["on_complete_handler"]``) to a callable. Storing the key on
        # disk — not the callable — keeps the queue serializable across restart.
        self._completion_handlers: dict[
            str, Callable[[dict[str, Any]], Awaitable[None]]
        ] = {
            "experiment_runner_creation": self._setup_completion_hook,
        }

    # ------------------------------------------------------------------
    # Injected-dep shims
    # ------------------------------------------------------------------

    async def _persist(self) -> None:
        """Trigger session-state persistence if the callback is wired.
        Catches and logs exceptions so a persist failure never breaks the queue.
        """
        if self._persist_cb is None:
            return
        try:
            await self._persist_cb()
        except Exception as e:
            logger.warning("persist callback failed: %s", e)

    async def _emit(self, payload: dict[str, Any]) -> None:
        """Best-effort event emit through the injected ``emit_event``.

        Replaces every RankEvolve ``interactive._send_response(payload,
        InteractionFlags.MessageOnly)`` call site. Failures MUST NOT break the
        hub flow — they only affect live UI; on-disk state is the source of
        truth on reconnect.
        """
        try:
            await self._emit_event(self._session_id, payload)
        except Exception as e:
            logger.warning(
                "emit_event failed (type=%s, status=%s): %s",
                payload.get("type"),
                payload.get("status"),
                e,
            )

    async def _persist_task_status(self, task_id: str, status: str) -> None:
        """Persist ``metadata.task_status`` on the matching task_ref row so a
        WebSocket reconnect after disconnect-during-emit shows the correct chip
        status. Best-effort: failures MUST NOT break the task_status emit.
        """
        if self._update_task_ref_status is None:
            return
        try:
            if await self._update_task_ref_status(task_id, status):
                await self._persist()
        except Exception as e:
            logger.warning(
                "_persist_task_status(task_id=%s, status=%s) failed: %s",
                task_id,
                status,
                e,
            )

    async def _add_task_ref_safe(self, **kwargs: Any) -> None:
        """Best-effort chronological task_ref chip marker."""
        if self._add_task_ref is None:
            return
        try:
            await self._add_task_ref(**kwargs)
            await self._persist()
        except Exception as e:
            logger.warning("add_task_ref failed (%s): %s", kwargs.get("task_id"), e)

    @staticmethod
    def _resolve_field_templates(
        tool_def: Any,
        args: dict[str, Any],
        workspace: Path,
        session_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve ``{{variable}}`` templates in tool definition fields.

        Template resolution is governed by ``arg_template_rules`` — a list of
        rules each specifying which fields and which variables participate.
        Each rule is a dict ``{"field_pattern": "...", "arg_pattern": "..."}``;
        patterns use the ``string_check`` DSL (e.g. ``$ _path`` = endsWith).

        Variables are collected from ``session_context`` (lower priority) and
        ``args`` (higher priority). After substitution the result is checked as
        an absolute path, then as a path relative to ``workspace``.

        Returns a dict with two parallel keys per resolved field:
          * ``<fname>``         -> str — the resolved path (always present once
            the template was applied; the path may or may not exist on disk).
          * ``<fname>_exists``  -> bool — True iff the resolved path exists.

        Ported from RankEvolve's ``SessionToolExecutor._resolve_field_templates``;
        ``session_context`` is now an explicit param (the host supplies it)
        instead of reading off a ``session`` god-object.
        """
        import dataclasses
        import re

        # ``string_check`` (the string-directive DSL) lives in RichPythonUtils
        # — the same canonical home as TemplateManager (verified: the source
        # comparison.py docstring imports CompareOption/CompareMethod from
        # ``rich_python_utils.string_utils.comparison``).
        from rich_python_utils.string_utils.comparison import string_check  # @manual

        rules = getattr(tool_def, "arg_template_rules", None) or []
        if not rules:
            return {}

        def _normalize(v: str) -> str:
            # Never attempt is_file() on values that obviously aren't paths
            # (Linux PATH_MAX is 4096); catch OSError as defense-in-depth.
            if len(v) > 4096 or "\n" in v or "\0" in v:
                return v
            try:
                p = Path(v)
                return str(p.parent) if p.is_file() else v
            except OSError:
                return v

        # Build full variable pool: session context (lower) + args (higher).
        all_vars: dict[str, str] = {}
        for k, v in (session_context or {}).items():
            if isinstance(v, str) and v:
                all_vars[k] = _normalize(v)
        for k, v in args.items():
            if isinstance(v, str) and v:
                all_vars[k] = _normalize(v)

        _template_re = re.compile(r"\{\{(\w+)\}\}")
        results: dict[str, Any] = {}

        def _safe_exists(path: Path) -> bool:
            try:
                return path.exists()
            except OSError:
                return False

        def _store(fname: str, path: Path) -> None:
            exists = _safe_exists(path)
            results[fname] = str(path)
            results[f"{fname}_exists"] = exists
            if not exists:
                logger.warning(
                    "_resolve_field_templates: %s resolved to %s but file does "
                    "not exist (yet). Storing path; downstream can check "
                    "%s_exists=False.",
                    fname,
                    path,
                    fname,
                )

        for rule in rules:
            field_pat = rule.get("field_pattern", "")
            arg_pat = rule.get("arg_pattern", "")
            if not field_pat:
                continue

            eligible_vars = (
                {k: v for k, v in all_vars.items() if string_check(k, arg_pat)}
                if arg_pat
                else all_vars
            )

            for fld in dataclasses.fields(tool_def):
                fname = fld.name
                if fname in results:
                    continue  # already resolved by a prior rule
                if not string_check(fname, field_pat):
                    continue
                template = getattr(tool_def, fname, "")
                if not template or "{{" not in template:
                    continue

                resolved = _template_re.sub(
                    lambda m: eligible_vars.get(m.group(1), m.group(0)),
                    template,
                )

                resolved_path = Path(resolved)
                if resolved_path.is_absolute():
                    _store(fname, resolved_path)
                else:
                    _store(fname, workspace / resolved)

            # Second pass: resolve plain relative paths (no {{}} templates)
            # against the workspace.
            for fld in dataclasses.fields(tool_def):
                fname = fld.name
                if fname in results:
                    continue
                if not string_check(fname, field_pat):
                    continue
                value = getattr(tool_def, fname, "")
                if not value or "{{" in value:
                    continue  # empty or has templates (handled above)
                _store(fname, workspace / value)

        return results

    @staticmethod
    def _make_task_status_payload(
        entry: dict[str, Any] | None,
        session_id: str,
        task_id: str,
        status: str,
        **extra: Any,
    ) -> dict[str, Any]:
        """Build a ``task_status`` notification payload with consistent fields.

        Routes per-entry metadata (multi_task_id, scope) into the event so the
        WebUI reducer can route correctly. ``entry`` may be None for callers
        that emit task_status before the queue entry exists.
        """
        payload: dict[str, Any] = {
            "type": "task_status",
            "session_id": session_id,
            "task_id": task_id,
            "status": status,
            **extra,
        }
        if entry is not None:
            mid = entry.get("multi_task_id")
            if mid:
                payload["multi_task_id"] = mid
            entry_scope = entry.get("scope")
            if entry_scope:
                payload["scope"] = entry_scope
        return payload

    # ------------------------------------------------------------------
    # Task-query builders (static — pure functions)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_hypothesis_task_query(
        hypothesis: dict[str, Any], workflow_target_path: str = ""
    ) -> str:
        """Build the /task request text from a selected hypothesis."""
        parts = [
            f"Implement hypothesis {hypothesis.get('id', '')}: {hypothesis.get('title', '')}"
        ]
        if workflow_target_path:
            parts.append(f"Target codebase: {workflow_target_path}")
        if hypothesis.get("problem"):
            parts.append(f"PROBLEM: {hypothesis['problem']}")
        if hypothesis.get("approach"):
            parts.append(f"APPROACH: {hypothesis['approach']}")
        if hypothesis.get("cross_refs"):
            parts.append(f"Cross-refs: {hypothesis['cross_refs']}")
        return "\n".join(parts)

    @staticmethod
    def _build_batch_task_query(
        hypotheses: list[dict[str, Any]],
        batch_info: dict[str, Any],
        workflow_target_path: str = "",
    ) -> str:
        """Build a combined /task prompt for multiple hypotheses in the same batch."""
        parts = [
            f"Implement the following hypotheses together "
            f"(Batch {batch_info.get('id', '?')}: {batch_info.get('label', '')}):"
        ]
        if workflow_target_path:
            parts.append(f"Target codebase: {workflow_target_path}")
        parts.append("")
        for hyp in hypotheses:
            parts.append(f"Hypothesis {hyp.get('id', '')}: {hyp.get('title', '')}")
            if hyp.get("problem"):
                parts.append(f"  PROBLEM: {hyp['problem']}")
            if hyp.get("approach"):
                parts.append(f"  APPROACH: {hyp['approach']}")
            if hyp.get("cross_refs"):
                parts.append(f"  Cross-refs: {hyp['cross_refs']}")
            parts.append("")
        parts.append("These hypotheses are in the same batch and may share code areas.")
        parts.append("Implement them cohesively.")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Hub creation
    # ------------------------------------------------------------------

    async def create_experiment_hub(
        self,
        selected_details: list[dict[str, Any]],
        proposals_data: dict[str, Any],
        custom_queries: list[str] | None = None,
        group_by: str = "batch",
        initial_view: str | None = None,
        auto_implement: bool = True,
    ) -> str:
        """Create an Implementation Hub. Returns the ``multi_task_id``.

        Two operating modes (controlled by ``auto_implement``):

        * ``auto_implement=True`` (default) — "Path A": group hypotheses into
          batches, send initial multi-task notification with batch_queue +
          selection_snapshot, AND immediately enqueue one ``/task`` per batch.
        * ``auto_implement=False`` — "Path B": create the Hub SHELL only. Send
          WS notification with ``selection_snapshot.selectedProposals``
          pre-loaded BUT empty ``batch_queue``, and SKIP the enqueue loop.

        **G25 — Idempotency guard (Path A only)**: turn-replay / resume of a
        widget turn with ``auto_implement=True`` would otherwise mint a fresh
        multi id + re-enqueue every batch → double the compute cost. Mirror
        ``open_experiment_hub:610-628``'s guard — when the workflow context
        already has an ``active_multi_task_id``, emit a ``task_status:focus``
        event for it and return the existing id WITHOUT re-enqueueing.
        Path B (``auto_implement=False``) is safe to re-run: it only emits a
        preview notification with no side effects.
        """
        from agent_foundation.experiment_hub.proposal_grouping import (
            group_selected_by_batch,
        )

        # G25 — idempotency guard (Path A only). Prevents double-implement on
        # turn-replay/resume of a checkbox-checked handoff. Mirrors the pattern
        # already in open_experiment_hub:610-628.
        if auto_implement:
            existing_mid = getattr(self._wc, "active_multi_task_id", None)
            if existing_mid:
                await self._emit(
                    {
                        "type": "task_status",
                        "task_id": existing_mid,
                        "task_type": "multi",
                        "status": "focus",
                        "session_id": self._session_id,
                        "metadata": {
                            "initial_view": initial_view or "implementation",
                        },
                    }
                )
                logger.info(
                    "create_experiment_hub: focusing existing multi_task_id=%s "
                    "(G25 idempotency guard — turn-replay of auto_implement=True)",
                    existing_mid,
                )
                return existing_mid

        wt = self._workflow_target_path

        # Group hypotheses based on mode. Path B skips grouping entirely.
        if not auto_implement:
            groups: list[dict[str, Any]] = []
        elif group_by == "all":
            groups = [
                {
                    "batch_id": "all",
                    "batch_label": "All Selected",
                    "hypotheses": selected_details,
                }
            ]
        elif group_by == "hypothesis":
            groups = [
                {
                    "batch_id": h.get("id", ""),
                    "batch_label": h.get("title", "")[:40],
                    "hypotheses": [h],
                }
                for h in selected_details
            ]
        else:
            # Default: group by batch
            groups = group_selected_by_batch(
                selected_details,
                proposals_data if isinstance(proposals_data, dict) else {},
            )

        multi_task_id = f"multi-{uuid.uuid4().hex[:8]}"

        # Persist a chronological task_ref marker BEFORE the WS emit so a crash
        # in the window between persist and emit re-emerges the chip at the
        # right conversation position on resume. Best-effort.
        await self._add_task_ref_safe(
            task_id=multi_task_id,
            label="Experiment Hub",
            tool_name="proposal_selection",
            multi_task_id=multi_task_id,
        )

        # Send initial multi-task notification.
        await self._emit(
            {
                "type": "task_status",
                "task_id": multi_task_id,
                "task_type": "multi",
                "status": "starting",
                "session_id": self._session_id,
                "scenario": "implementation_hub",
                # Mode-aware request label.
                "request": (
                    f"Implementation: {len(selected_details)} hypotheses "
                    f"in {len(groups)} {'batch' if group_by == 'batch' else 'group'}(s)"
                    if auto_implement
                    else f"Hub opened (preview): {len(selected_details)} "
                    f"hypotheses pre-selected — adjust + submit from "
                    f"Selection view to start implementations"
                ),
                # Path B sends EMPTY batch_queue — nothing enqueued yet.
                "batch_queue": [
                    {
                        "batch_id": bg["batch_id"],
                        "batch_label": bg["batch_label"],
                        "hypothesis_ids": [h.get("id", "") for h in bg["hypotheses"]],
                        "status": "queued",
                    }
                    for bg in groups
                ],
                "selection_snapshot": {
                    "proposals": proposals_data
                    if isinstance(proposals_data, dict)
                    else {},
                    "selectedProposals": [h.get("id", "") for h in selected_details],
                    "customQueries": custom_queries or [],
                },
                "metadata": ({"initial_view": initial_view} if initial_view else {}),
            }
        )

        # Store selected proposals in phase outputs.
        wc = self._wc
        wc.phase_outputs["research_proposals"] = selected_details

        # Set this hub as the active queue target.
        wc.active_multi_task_id = multi_task_id

        # Enqueue one task per group — SKIPPED entirely in Path B.
        if auto_implement:
            for bg in groups:
                if group_by == "hypothesis" and len(bg["hypotheses"]) == 1:
                    query = self._build_hypothesis_task_query(bg["hypotheses"][0], wt)
                else:
                    query = self._build_batch_task_query(bg["hypotheses"], bg, wt)
                hyp_ids = ",".join(h.get("id", "") for h in bg["hypotheses"])
                await self.enqueue_and_maybe_execute(
                    tool_name="task",
                    request=query,
                    title=f"B{bg['batch_id']}: {hyp_ids}"
                    if group_by == "batch"
                    else hyp_ids,
                    args={
                        "request": query,
                        "template_version": "hypothesis_implementation",
                    },
                    hypothesis_id=hyp_ids,
                    phase="3",
                    multi_task_id=multi_task_id,
                    batch_id=bg.get("batch_id", "") if group_by == "batch" else "",
                    batch_label=bg.get("batch_label", "")
                    if group_by == "batch"
                    else "",
                )

            # Force a yield so _try_start_next_task() coroutines can execute.
            await asyncio.sleep(0)

        # Persist now so the hub's existence is captured on disk immediately.
        await self._persist()

        return multi_task_id

    async def open_experiment_hub(
        self,
        *,
        proposals_data: dict[str, Any] | None,
        pre_select_top_n: int = 5,
        initial_view: str = "selection",
    ) -> str | None:
        """Open the Experiment Hub for this session, idempotently.

        Behavior:
          1. Idempotency guard: if ``workflow_context.active_multi_task_id`` is
             already set, emit a ``task_status: focus`` WS event for that tab
             and return that ID. NO duplicate hub created.
          2. Otherwise compute top-N globally-ranked hypotheses across all
             phases (sort by ``rank`` ascending; tiebreak by
             ``len(source_workers)`` descending then ``id`` lexical) and call
             ``create_experiment_hub(top_n_details, proposals_data)`` with
             ``auto_implement=False``.
        """
        wc = self._wc

        # 1. Idempotency: focus existing Hub tab instead of creating a second.
        existing_mid = getattr(wc, "active_multi_task_id", None)
        if existing_mid:
            await self._emit(
                {
                    "type": "task_status",
                    "task_id": existing_mid,
                    "task_type": "multi",
                    "status": "focus",
                    "session_id": self._session_id,
                    "metadata": {"initial_view": initial_view},
                }
            )
            logger.info(
                "open_experiment_hub: focusing existing multi_task_id=%s "
                "(idempotency guard)",
                existing_mid,
            )
            return existing_mid

        # 2. Validate proposals_data shape; degrade gracefully if missing.
        if not isinstance(proposals_data, dict):
            logger.warning(
                "open_experiment_hub: proposals_data is %s, expected dict; "
                "Hub cannot be opened with selections",
                type(proposals_data).__name__,
            )
            return None
        phases = proposals_data.get("phases") or []
        if not isinstance(phases, list) or not phases:
            logger.warning(
                "open_experiment_hub: proposals_data has no `phases`; "
                "Hub cannot be opened with selections"
            )
            return None

        # 3. Flatten + sort globally by rank.
        all_proposals: list[dict[str, Any]] = []
        for ph in phases:
            if isinstance(ph, dict):
                for p in ph.get("proposals", []) or []:
                    if isinstance(p, dict) and p.get("id"):
                        all_proposals.append(p)
        if not all_proposals:
            logger.warning(
                "open_experiment_hub: no proposals across all phases; "
                "Hub cannot be opened with selections"
            )
            return None

        def _sort_key(p: dict[str, Any]) -> tuple[int, int, str]:
            # Lower rank = higher priority. Tiebreak: more source_workers
            # = better validated; then id lexical for determinism.
            rank = p.get("rank")
            try:
                rank_i = int(rank) if rank is not None else 9999
            except (TypeError, ValueError):
                rank_i = 9999
            workers = p.get("source_workers") or []
            return (rank_i, -len(workers), str(p.get("id", "")))

        all_proposals.sort(key=_sort_key)
        n = max(1, int(pre_select_top_n))
        top_n = all_proposals[:n]
        logger.info(
            "open_experiment_hub: pre-selecting top-%d of %d proposals: %s",
            len(top_n),
            len(all_proposals),
            [p.get("id", "") for p in top_n],
        )

        # 4. Delegate to create_experiment_hub with auto_implement=False.
        multi_task_id = await self.create_experiment_hub(
            selected_details=top_n,
            proposals_data=proposals_data,
            custom_queries=[],
            group_by="batch",
            initial_view=initial_view,
            auto_implement=False,
        )
        return multi_task_id

    async def add_to_experiment_hub(
        self,
        multi_task_id: str,
        request: str,
        title: str,
        batch_id: str = "",
        batch_label: str = "",
        hypothesis_id: str = "",
    ) -> str:
        """Add a /task to an existing experiment hub's queue.

        Returns the task_id for the new queue entry.
        """
        task_id = await self.enqueue_and_maybe_execute(
            tool_name="task",
            request=request,
            title=title,
            args={"request": request, "template_version": "hypothesis_implementation"},
            hypothesis_id=hypothesis_id,
            phase="3",
            multi_task_id=multi_task_id,
            batch_id=batch_id,
            batch_label=batch_label,
        )
        await self._persist()
        return task_id

    # ------------------------------------------------------------------
    # Submission setup (PTI) + completion hook
    # ------------------------------------------------------------------

    async def setup_submission_script(
        self,
        multi_task_id: str,
        setup_id: str,
        setup_name: str,
        reference_scripts: list[str],
        library_template: str | None,
        reference_command: str,
        additional_instructions: str,
        selected_hypothesis_ids: list[str],
    ) -> str:
        """Enqueue a PTI task that generates a parameterized submission script.

        Registers the ``experiment_runner_creation`` completion handler so the
        WebUI is notified when ``outputs/submit_v1.py`` and
        ``outputs/launch.json`` are produced.

        ``phase=""`` is critical: setup is NOT part of any hypothesis-
        implementation phase. Returns the queue task_id.
        """
        from agent_foundation.experiment_hub.submission_templates_loader import (
            _inline_script,
            _resolve_library_template,
        )

        refs_inlined: list[str] = []
        if library_template:
            for path in _resolve_library_template(library_template):
                refs_inlined.append(_inline_script(path))
        for path_str in reference_scripts or []:
            if not isinstance(path_str, str) or not path_str:
                continue
            refs_inlined.append(_inline_script(Path(path_str)))

        wt = self._workflow_target_path

        query = (
            f"Generate a parameterized submission setup for combo "
            f"'{setup_name}'.\n\n"
            f"Target codebase: {wt}\n"
            f"Hypotheses: {', '.join(selected_hypothesis_ids)}\n\n"
            f"## Output requirements (BOTH files required — completion hook "
            f"validates presence of each)\n"
            f"1. outputs/submit_v1.py — the submission entry point. "
            f"Importable as a fbcode module (NOT a standalone script — at "
            f"Meta, `python <path>` cannot resolve fbcode imports). "
            f"Accepts:\n"
            f"     --enable-flags enable_foo,enable_bar  (comma-separated\n"
            f"                              CONFIG FIELD NAMES already\n"
            f"                              resolved by the hub; build the\n"
            f"                              overrides dict directly as\n"
            f"                              {{name: True for name in\n"
            f"                              received_names}} — do NOT define\n"
            f"                              any HYPOTHESIS_FLAG_MAP table or\n"
            f"                              ID-to-name translation logic)\n"
            f"     --experiment-name <str>\n"
            f"     --app-layer-version <str>  REQUIRED — the fbpkg version\n"
            f"                              (e.g. fire-app:2941a32) the user\n"
            f"                              typed in the Submit Confirm modal.\n"
            f"                              Pass through to FBLearner as the\n"
            f"                              package_version for the relevant\n"
            f"                              fbpkg slot. Do NOT auto-detect for\n"
            f"                              the runner code path — the user\n"
            f"                              value is authoritative.\n"
            f"   Print the FBLearner flow URI on stdout as ONE LINE "
            f"prefixed:\n"
            f"     FLOW_URI: <url>\n"
            f"   And MAST job names when known:\n"
            f"     MAST_JOB: <job_name>\n"
            f"   IMPORTANT: use bare `print(...)` for these two contract "
            f"lines, NOT `logger.info(...)`. The runner regex anchors at "
            f"^FLOW_URI:/^MAST_JOB:; any logger prefix (timestamp, level) "
            f"breaks the match and the URL is silently lost.\n"
            f"   Exit 0 on success, 1 on failure.\n\n"
            f"2. outputs/launch.json — the launch invocation, structured "
            f"as:\n"
            f'     {{"launcher": "buck_run_auto",\n'
            f'      "script_args": ["--enable-flags", "${{ENABLE_FLAGS}}",\n'
            f'                       "--experiment-name", "${{EXP_NAME}}",\n'
            f'                       "--app-layer-version", "${{APP_LAYER_VERSION}}"],\n'
            f'      "cwd": "${{CODEBASE_ROOT}}"}}\n'
            f"   STRICT REQUIREMENTS (runner enforces and rejects "
            f"otherwise):\n"
            f"   - `launcher` MUST be present and set to the literal "
            f'string `"buck_run_auto"`.\n'
            f"   - `script_args` MUST be a list of strings.\n"
            f"   - `cwd` is OPTIONAL; defaults to ${{CODEBASE_ROOT}}.\n"
            f"   - NO shell metacharacters anywhere in script_args (no "
            f"`;`, `|`, `&&`, `>`, backticks).\n"
            f"   - `script_args` MUST include "
            f'`"--app-layer-version", "${{APP_LAYER_VERSION}}"` (paired '
            f"tokens, in that order).\n"
            f"   The runner substitutes ${{ENABLE_FLAGS}}, ${{EXP_NAME}}, "
            f"${{APP_LAYER_VERSION}}, and ${{CODEBASE_ROOT}} at spawn time.\n\n"
            f"## User-provided reference command (authoritative for "
            f"launch.json)\n"
            f"{reference_command}\n\n"
            f"## Additional Instructions\n{additional_instructions}\n\n"
            f"## Reference Material\n\n" + "\n\n".join(refs_inlined)
        )

        task_id = await self.enqueue_and_maybe_execute(
            tool_name="task",
            request=query,
            title=f"Setup: {setup_name}",
            args={
                "request": query,
                "template_version": "experiment_runner_creation",
                "setup_id": setup_id,
                "setup_name": setup_name,
            },
            phase="",  # NOT part of any SOP phase
            multi_task_id=multi_task_id,
            batch_id="setup",
            batch_label=f"Setup: {setup_name}",
            on_complete_handler="experiment_runner_creation",
            scope="hub_setup",
        )

        # Emit setup_task_started so the WebUI can populate the setup file's
        # taskId BEFORE PTI completes.
        try:
            await self._emit(
                {
                    "type": "setup_task_started",
                    "session_id": self._session_id,
                    "multi_task_id": multi_task_id,
                    "setup_id": setup_id,
                    "task_id": task_id,
                }
            )
        except Exception:
            pass

        await self._persist()
        return task_id

    async def _setup_completion_hook(self, entry: dict[str, Any]) -> None:
        """Post-PTI hook: validate outputs and emit ``setup_completed`` event.

        Validation rules:
          1. ``outputs/submit_v*.py`` must exist (pick highest version).
          2. ``outputs/launch.json`` must exist, parse as JSON, contain
             ``script_args`` (list[str]) + optional ``cwd`` (str), and (when
             present) a known ``launcher``.

        On success the hook ships the script + launch CONTENT (not just paths)
        back to the WebUI. On any validation failure it still emits
        ``setup_completed`` but with ``status="error"`` and a specific message.
        """
        args = entry.get("args") or {}
        multi_task_id = entry.get("multi_task_id") or ""
        setup_id = (args.get("setup_id") if isinstance(args, dict) else None) or ""
        setup_name = (args.get("setup_name") if isinstance(args, dict) else None) or ""
        task_id = entry.get("task_id") or ""
        workspace_str = entry.get("workspace") or ""
        queue_status = entry.get("status") or ""

        if queue_status == "error":
            await self._emit_setup_completed(
                multi_task_id=multi_task_id,
                setup_id=setup_id,
                setup_name=setup_name,
                task_id=task_id,
                status="error",
                error=(entry.get("result_summary") or "Setup task failed")[:200],
                script_content=None,
                launch_content=None,
                script_basename=None,
            )
            return

        if not workspace_str:
            await self._emit_setup_completed(
                multi_task_id=multi_task_id,
                setup_id=setup_id,
                setup_name=setup_name,
                task_id=task_id,
                status="error",
                error="Setup completed without producing a workspace path",
                script_content=None,
                launch_content=None,
                script_basename=None,
            )
            return

        outputs_dir = Path(workspace_str) / "outputs"
        candidates = sorted(
            list(outputs_dir.glob("submit.py")) + list(outputs_dir.glob("submit_v*.py"))
        )
        if not candidates:
            await self._emit_setup_completed(
                multi_task_id=multi_task_id,
                setup_id=setup_id,
                setup_name=setup_name,
                task_id=task_id,
                status="error",
                error="Setup completed but no script was produced",
                script_content=None,
                launch_content=None,
                script_basename=None,
            )
            return
        script_path = candidates[-1]

        launch_path = outputs_dir / "launch.json"
        if not launch_path.is_file():
            await self._emit_setup_completed(
                multi_task_id=multi_task_id,
                setup_id=setup_id,
                setup_name=setup_name,
                task_id=task_id,
                status="error",
                error=(
                    "Setup did not produce outputs/launch.json — script cannot "
                    "be launched"
                ),
                script_content=None,
                launch_content=None,
                script_basename=None,
            )
            return

        try:
            launch_text = launch_path.read_text(encoding="utf-8")
            launch_data = json.loads(launch_text)
        except Exception as e:
            await self._emit_setup_completed(
                multi_task_id=multi_task_id,
                setup_id=setup_id,
                setup_name=setup_name,
                task_id=task_id,
                status="error",
                error=f"launch.json is not valid JSON: {e}"[:200],
                script_content=None,
                launch_content=None,
                script_basename=None,
            )
            return

        if not isinstance(launch_data, dict):
            await self._emit_setup_completed(
                multi_task_id=multi_task_id,
                setup_id=setup_id,
                setup_name=setup_name,
                task_id=task_id,
                status="error",
                error="launch.json must be a JSON object with `script_args` (list[str]) and optional `cwd` (str)",
                script_content=None,
                launch_content=None,
                script_basename=None,
            )
            return

        # Option 3 schema: launch.json carries `script_args` + optional `cwd`.
        script_args = launch_data.get("script_args")
        cwd = launch_data.get("cwd")
        if script_args is None:
            err = (
                "launch.json uses the old `cmd`/`cwd` schema. Re-generate "
                "the setup or edit launch.json via the script editor "
                "drawer to use the new schema: "
                '`{"script_args": [...], "cwd": "${CODEBASE_ROOT}"}`'
                if "cmd" in launch_data
                else "launch.json missing required `script_args` (list of strings)"
            )
            await self._emit_setup_completed(
                multi_task_id=multi_task_id,
                setup_id=setup_id,
                setup_name=setup_name,
                task_id=task_id,
                status="error",
                error=err,
                script_content=None,
                launch_content=None,
                script_basename=None,
            )
            return
        if not isinstance(script_args, list) or not all(
            isinstance(c, str) for c in script_args
        ):
            await self._emit_setup_completed(
                multi_task_id=multi_task_id,
                setup_id=setup_id,
                setup_name=setup_name,
                task_id=task_id,
                status="error",
                error="launch.json `script_args` must be a list of strings",
                script_content=None,
                launch_content=None,
                script_basename=None,
            )
            return
        if cwd is not None and (not isinstance(cwd, str) or not cwd):
            await self._emit_setup_completed(
                multi_task_id=multi_task_id,
                setup_id=setup_id,
                setup_name=setup_name,
                task_id=task_id,
                status="error",
                error="launch.json `cwd` (when present) must be a non-empty string",
                script_content=None,
                launch_content=None,
                script_basename=None,
            )
            return

        # Optional `launcher` field — validate against the registry.
        launcher_field = launch_data.get("launcher")
        if launcher_field is not None:
            if not isinstance(launcher_field, str) or not launcher_field:
                await self._emit_setup_completed(
                    multi_task_id=multi_task_id,
                    setup_id=setup_id,
                    setup_name=setup_name,
                    task_id=task_id,
                    status="error",
                    error=(
                        "launch.json `launcher` (when present) must be a "
                        "non-empty string identifying a known launcher "
                        "(e.g. 'buck_run_auto')"
                    ),
                    script_content=None,
                    launch_content=None,
                    script_basename=None,
                )
                return
            from agent_foundation.experiment_hub.submission_launcher import (
                known_launcher_names,
            )

            available = known_launcher_names()
            if launcher_field not in available:
                await self._emit_setup_completed(
                    multi_task_id=multi_task_id,
                    setup_id=setup_id,
                    setup_name=setup_name,
                    task_id=task_id,
                    status="error",
                    error=(
                        f"launch.json declares unknown launcher "
                        f"{launcher_field!r}. Available: {available}. "
                        "Re-generate the setup or fix via the editor drawer."
                    ),
                    script_content=None,
                    launch_content=None,
                    script_basename=None,
                )
                return

        try:
            script_content = script_path.read_text(encoding="utf-8")
        except Exception as e:
            await self._emit_setup_completed(
                multi_task_id=multi_task_id,
                setup_id=setup_id,
                setup_name=setup_name,
                task_id=task_id,
                status="error",
                error=f"Failed to read {script_path.name}: {e}"[:200],
                script_content=None,
                launch_content=None,
                script_basename=None,
            )
            return

        await self._emit_setup_completed(
            multi_task_id=multi_task_id,
            setup_id=setup_id,
            setup_name=setup_name,
            task_id=task_id,
            status="ready",
            error="",
            script_content=script_content,
            launch_content=launch_text,
            script_basename=script_path.name,
        )

    async def _emit_setup_completed(
        self,
        multi_task_id: str,
        setup_id: str,
        setup_name: str,
        task_id: str,
        status: str,
        error: str,
        script_content: str | None,
        launch_content: str | None,
        script_basename: str | None,
    ) -> None:
        """Push a ``setup_completed`` event through the injected emitter.

        Sending CONTENT (not paths) enforces the single-writer invariant: the
        WebUI picks ``submit_v<n+1>.py`` under its in-process lock and writes
        the canonical version. The agent server's task workspace remains
        transient/internal.
        """
        payload: dict[str, Any] = {
            "type": "setup_completed",
            "session_id": self._session_id,
            "multi_task_id": multi_task_id,
            "setup_id": setup_id,
            "setup_name": setup_name,
            "task_id": task_id,
            "status": status,
            "error": error,
        }
        # Don't include `null` fields when the setup errored.
        if script_content is not None:
            payload["script_content"] = script_content
        if launch_content is not None:
            payload["launch_content"] = launch_content
        if script_basename is not None:
            payload["script_basename"] = script_basename
        await self._emit(payload)

    # ------------------------------------------------------------------
    # Submission run (subprocess) + auto-analysis
    # ------------------------------------------------------------------

    async def run_submission_script(
        self,
        multi_task_id: str,
        submission_id: str,
        setup_id: str,
        script_path: str,
        launch_path: str,
        enable_flags: list[str],
        experiment_name: str,
        submission_label: str,
        app_layer_version: str = "",
        build_command: str = "",
    ) -> str:
        """Enqueue a submission_run task that spawns the user's submit_v<n>.py
        as a subprocess. Returns the queue task_id.
        """
        args = {
            "submission_run": True,
            "submission_id": submission_id,
            "setup_id": setup_id,
            "script_path": script_path,
            # The runner reads this to load launch.json and build the buck
            # command. Direct ``python <path>`` is forbidden in fbcode.
            "launch_path": launch_path,
            "enable_flags": list(enable_flags or []),
            # Substituted into launch.json's ${EXP_NAME}.
            "experiment_name": experiment_name,
            # Per-submission fbpkg version, substituted into ${APP_LAYER_VERSION}.
            "app_layer_version": app_layer_version,
            # Auto-build recipe.
            "build_command": build_command,
        }
        task_id = await self.enqueue_and_maybe_execute(
            tool_name="submission_run",
            request=f"Run {submission_label} via {Path(script_path).name}",
            title=f"Run: {submission_label}",
            args=args,
            multi_task_id=multi_task_id,
            batch_id="run",
            batch_label=submission_label,
            # phase="" is critical — submission runs are NOT part of any
            # hypothesis-implementation phase.
            phase="",
        )
        await self._persist()
        return task_id

    async def _exec_submission_run(
        self,
        args: dict[str, Any],
        queue_task_id: str = "",
    ) -> ToolExecutionResult:
        """Execute a submission_run queue entry.

        Builds a workspace under ``<tasks_dir>/<run_dir>/``, emits a
        ``task_status:starting`` event so the WebUI attaches a
        WorkspaceStreamTailer to the cache dir, then runs
        ``SubmissionRunner.run()`` and finally emits the terminal status.
        """
        from agent_foundation.experiment_hub.submission_runner import (
            LaunchValidationError,
            SubmissionRunner,
        )

        submission_id = (args or {}).get("submission_id", "")
        setup_id = (args or {}).get("setup_id", "")
        script_path_str = (args or {}).get("script_path", "")
        launch_path_str = (args or {}).get("launch_path", "")
        enable_flags = (args or {}).get("enable_flags", []) or []
        experiment_name = (args or {}).get("experiment_name", "")
        app_layer_version = (args or {}).get("app_layer_version", "") or ""
        build_command = (args or {}).get("build_command", "") or ""

        if not script_path_str or not launch_path_str:
            return ToolExecutionResult(
                result="Error: submission_run requires script_path and launch_path"
            )

        # Build a workspace for this run under tasks_dir. The dir name uses the
        # queue task_id so the .task_meta.json sidecar reconciliation can map
        # workspace dirs back to queue entries unambiguously.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id_for_dir = queue_task_id or f"task-{uuid.uuid4().hex[:8]}"
        workspace = Path(self._session_tasks_dir) / f"submission_{ts}_{task_id_for_dir}"
        workspace.mkdir(parents=True, exist_ok=True)

        entry = self._wc.get_entry(queue_task_id) if queue_task_id else None

        # Sidecar for hub_state reconcile.
        try:
            sidecar = {
                "task_id": queue_task_id,
                "multi_task_id": (entry or {}).get("multi_task_id", ""),
                "tool_name": "submission_run",
                "submission_id": submission_id,
                "setup_id": setup_id,
                "session_id": self._session_id,
                "session_dir": self._session_dir.name,
            }
            (workspace / ".task_meta.json").write_text(
                json.dumps(sidecar), encoding="utf-8"
            )
        except Exception as meta_err:
            logger.warning(
                "submission_run sidecar write failed for %s: %s",
                queue_task_id,
                meta_err,
            )

        # Persist workspace path on the queue entry so resume reconcile can
        # find it.
        if queue_task_id:
            self._wc.update_entry(queue_task_id, workspace=str(workspace.absolute()))
            await self._persist()

        multi_task_id_val = (
            (entry or {}).get("multi_task_id", "") if queue_task_id else ""
        )

        # Notify the frontend so a task subtab appears with the right
        # workspace. poll_responses sees ``status='starting'`` + ``workspace=``
        # and attaches a WorkspaceStreamTailer to the cache dir.
        start_msg: dict[str, Any] = {
            "type": "task_status",
            "session_id": self._session_id,
            "task_id": queue_task_id or task_id_for_dir,
            "status": "starting",
            "request": f"Run: {submission_id}",
            "workspace": str(workspace.absolute()),
        }
        if multi_task_id_val:
            start_msg["multi_task_id"] = multi_task_id_val
        await self._emit(start_msg)

        # Single-writer invariant: the agent server NEVER writes
        # hub_*_submissions.json. emit pushes onto the event channel;
        # poll_responses applies under the WebUI's per-hub asyncio.Lock.
        async def _emit_state(extra: dict[str, Any]) -> None:
            payload: dict[str, Any] = {
                "type": "submission_state",
                "session_id": self._session_id,
                "multi_task_id": multi_task_id_val,
                "submission_id": submission_id,
                "setup_id": setup_id,
            }
            payload.update(extra)
            await self._emit(payload)

        # Emit an initial 'running' marker so the run row in the Monitor view
        # doesn't sit on 'submitted' for the first 10-20s while buck builds.
        run_started_at = int(time.time() * 1000)
        await _emit_state(
            {
                "status": "running",
                "runTaskId": queue_task_id or task_id_for_dir,
                "runStartedAt": run_started_at,
            }
        )

        workflow_target_path = self._workflow_target_path

        runner = SubmissionRunner(
            workspace=workspace,
            script_path=Path(script_path_str),
            enable_flags=enable_flags,
            experiment_name=experiment_name,
            launch_path=Path(launch_path_str),
            emit_event=_emit_state,
            workflow_target_path=workflow_target_path,
            codebase_root_pattern=_CODEBASE_ROOT_PATTERN,
            app_layer_version=app_layer_version,
            build_command=build_command,
        )

        terminal_status = "completed"
        terminal_error = ""
        flow_uri: str | None = None
        mast_job: str | None = None
        final_metrics: dict[str, Any] = {}
        epoch_trajectory: list[dict[str, Any]] = []
        try:
            run_result = await runner.run()
            flow_uri = run_result.get("flow_uri")
            mast_job = run_result.get("mast_job")
            exit_code = run_result.get("exit_code")
            final_metrics = dict(run_result.get("final_metrics") or {})
            epoch_trajectory = list(run_result.get("epoch_trajectory") or [])
            if exit_code != 0:
                terminal_status = "error"
                terminal_error = f"Subprocess exited with code {exit_code}"
        except LaunchValidationError as ve:
            terminal_status = "error"
            terminal_error = str(ve)[:300]
            logger.warning("submission_run launch validation failed: %s", ve)
        except asyncio.CancelledError:
            # SubmissionRunner.run already invoked self.cancel() under
            # asyncio.shield. We do NOT emit a submission_state event here:
            # ownership of the cancelled-state event lives in the cancel
            # cleanup so we don't double-write under the WebUI's lock.
            raise
        except Exception as e:
            terminal_status = "error"
            terminal_error = f"submission_run error: {e}"[:300]
            logger.error(
                "submission_run error for %s: %s",
                queue_task_id or "(no-id)",
                e,
                exc_info=True,
            )

        # Kick off post-completion auto-analysis when the run succeeded AND the
        # subprocess emitted STATUS: lines we could parse into final_metrics.
        # Failures here do NOT mark the submission failed.
        analysis_task_id: str | None = None
        if terminal_status == "completed" and final_metrics:
            try:
                analysis_task_id = await self._run_submission_analysis(
                    submission_id=submission_id,
                    multi_task_id=multi_task_id_val,
                    setup_id=setup_id,
                    enable_flags=list(enable_flags or []),
                    experiment_name=experiment_name,
                    final_metrics=final_metrics,
                    epoch_trajectory=epoch_trajectory,
                    emit_state=_emit_state,
                )
            except Exception as e:
                logger.warning(
                    "submission_run: auto-analysis failed for %s: %s",
                    submission_id,
                    e,
                    exc_info=True,
                )
                analysis_task_id = None

        await _emit_state(
            {
                "status": terminal_status,
                "runFinishedAt": int(time.time() * 1000),
                "flowUri": flow_uri,
                "mastJob": mast_job,
                "fblearnerError": terminal_error or None,
                **({"analysisTaskId": analysis_task_id} if analysis_task_id else {}),
            }
        )

        # Notify the frontend's task subtab that the run finished so the tailer
        # is stopped and the subtab badge flips off the spinner.
        end_msg: dict[str, Any] = {
            "type": "task_status",
            "session_id": self._session_id,
            "task_id": queue_task_id or task_id_for_dir,
            "status": "completed" if terminal_status == "completed" else "error",
        }
        if terminal_error:
            end_msg["message"] = terminal_error
        if multi_task_id_val:
            end_msg["multi_task_id"] = multi_task_id_val
        await self._emit(end_msg)

        # Mark the queue entry.
        if queue_task_id:
            wc = self._wc
            if terminal_status == "completed":
                wc.mark_completed(
                    queue_task_id, summary=f"Run completed for {submission_id}"
                )
            else:
                wc.mark_error(
                    queue_task_id,
                    error=terminal_error or terminal_status,
                )
            await self._persist()

        return ToolExecutionResult(
            result=(
                f"submission_run {terminal_status} for {submission_id} "
                f"(workspace={workspace})"
            ),
        )

    async def _run_submission_analysis(
        self,
        submission_id: str,
        multi_task_id: str,
        setup_id: str,
        enable_flags: list[str],
        experiment_name: str,
        final_metrics: dict[str, Any],
        epoch_trajectory: list[dict[str, Any]],
        emit_state: Any,
    ) -> str | None:
        """Post-completion auto-analysis for one submission.

        Returns analysis_task_id on success, None on skip. Caller wraps in
        try/except so a failure here NEVER marks the underlying submission
        failed — analysis is independent value-add.
        """
        if not multi_task_id:
            # Standalone runs (no Hub) don't have a combo-vs-baseline framing.
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_task_id = f"analysis_{submission_id}_{ts}_{uuid.uuid4().hex[:6]}"

        analysis_ws = Path(self._session_tasks_dir) / analysis_task_id
        outputs_dir = analysis_ws / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = analysis_ws / "_runtime" / "inferencer_cache" / "analysis"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Step 2 — flip the run row's lifecycle Stepper to "Analyzing".
        await emit_state(
            {
                "status": "analyzing",
                "analysisTaskId": analysis_task_id,
            }
        )

        # Step 3 — create-on-fly subtab for the analysis.
        await self._emit(
            {
                "type": "task_status",
                "session_id": self._session_id,
                "task_id": analysis_task_id,
                "status": "starting",
                "request": f"Analyze: {submission_id}",
                "workspace": str(analysis_ws.absolute()),
                "multi_task_id": multi_task_id,
                "scope": "hub_setup",
                "scenario": "experiment_analysis",
                "tool_name": "analyze_experiment",
            }
        )

        # Step 4 — load the active baseline for this hub (if any). The agent
        # server intentionally does NOT mutate hub_<mid>_submissions.json; we
        # only READ it here.
        baseline_metrics: dict[str, Any] = {}
        baseline_id: str | None = None
        try:
            sub_path = self._session_dir / f"hub_{multi_task_id}_submissions.json"
            if sub_path.is_file():
                hub_data = json.loads(sub_path.read_text(encoding="utf-8"))
                rows = hub_data.get("submissions") or []
                for row in rows:
                    if row.get("isBaseline") and row.get("finalMetrics"):
                        baseline_metrics = dict(row.get("finalMetrics") or {})
                        baseline_id = row.get("id") or row.get("submission_id")
                        break
        except Exception as e:
            logger.warning(
                "auto-analyze: baseline lookup failed for hub %s: %s",
                multi_task_id,
                e,
            )

        # Compute primary-metric delta. NDCG@10 is the conventional north-star;
        # fall back to HR@10 / MRR if the row lacks it.
        primary_keys = ["ndcg10", "hr10", "mrr"]
        primary_key = next(
            (k for k in primary_keys if k in final_metrics),
            None,
        )
        delta_pct: float | None = None
        verdict = "neutral"
        verdict_label = "no baseline"
        if primary_key and primary_key in baseline_metrics:
            try:
                cand = float(final_metrics[primary_key])
                base = float(baseline_metrics[primary_key])
                if base != 0:
                    delta_pct = (cand - base) / abs(base) * 100.0
                    if delta_pct >= 1.0:
                        verdict = "win"
                        verdict_label = f"+{delta_pct:.2f}%"
                    elif delta_pct <= -1.0:
                        verdict = "loss"
                        verdict_label = f"{delta_pct:.2f}%"
                    else:
                        verdict = "neutral"
                        verdict_label = f"{delta_pct:+.2f}%"
            except (TypeError, ValueError):
                pass
        elif primary_key:
            verdict_label = f"{primary_key}={final_metrics[primary_key]}"

        # Assemble the analysis markdown.
        flags_section = "\n".join(f"  - `{f}`" for f in (enable_flags or []))
        traj_section = "\n".join(
            f"  - epoch {row.get('epoch')}: "
            + ", ".join(f"{k}={row[k]}" for k in primary_keys if k in row)
            for row in (epoch_trajectory or [])[-10:]
        )
        baseline_section = (
            f"baseline: `{baseline_id}` "
            f"({primary_key}={baseline_metrics.get(primary_key)})"
            if baseline_id
            else "baseline: _not set for this hub_"
        )
        md_body = (
            f"# Auto-analysis: `{submission_id}`\n\n"
            f"**Verdict:** {verdict.upper()} — {verdict_label}\n\n"
            f"**Experiment:** {experiment_name}\n\n"
            f"**Setup:** `{setup_id}`\n\n"
            f"## Final metrics\n\n"
            + "\n".join(
                f"- {k}: {final_metrics[k]}" for k in primary_keys if k in final_metrics
            )
            + "\n\n"
            f"## Vs baseline\n\n{baseline_section}\n\n"
            f"## Enabled flags\n\n{flags_section or '  - _(none)_'}\n\n"
            f"## Trajectory (last 10 epochs)\n\n{traj_section or '  - _(no trajectory)_'}\n\n"
            "---\n"
            "_Generated by auto-analysis._\n"
        )
        analysis_file = outputs_dir / "analysis.md"
        analysis_file.write_text(md_body, encoding="utf-8")

        # Step 5 — PATCH the submission row with all analysis-summary fields in
        # a single submission_state event.
        analysis_summary = (
            f"{verdict.upper()} ({verdict_label})"
            if verdict != "neutral"
            else f"NEUTRAL ({verdict_label})"
        )
        await emit_state(
            {
                "analysisFile": str(analysis_file.absolute()),
                "analysisSummary": analysis_summary[:600],
                "verdict": verdict,
                "verdictLabel": verdict_label,
                "deltaPct": delta_pct,
                "finalMetrics": final_metrics,
                "epochTrajectory": epoch_trajectory[-100:],
            }
        )

        # Step 6 — analysis subtab chip flips to "completed".
        await self._emit(
            {
                "type": "task_status",
                "session_id": self._session_id,
                "task_id": analysis_task_id,
                "status": "completed",
                "multi_task_id": multi_task_id,
            }
        )
        return analysis_task_id

    # ------------------------------------------------------------------
    # Task queue runner
    # ------------------------------------------------------------------

    async def enqueue_and_maybe_execute(
        self,
        tool_name: str,
        request: str,
        title: str,
        args: dict[str, Any],
        hypothesis_id: str = "",
        phase: str = "",
        multi_task_id: str = "",
        batch_id: str = "",
        batch_label: str = "",
        on_complete_handler: str = "",
        scope: str = "",
    ) -> str:
        """Add a task to the queue and start it if a slot is available.

        Returns the task_id for tracking.
        """
        wc = self._wc
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        entry = wc.enqueue_task(
            task_id=task_id,
            tool_name=tool_name,
            request=request,
            title=title,
            args=args,
            hypothesis_id=hypothesis_id,
            phase=phase,
        )
        if multi_task_id:
            entry["multi_task_id"] = multi_task_id
        if batch_id:
            entry["batch_id"] = batch_id
        if batch_label:
            entry["batch_label"] = batch_label
        if scope:
            entry["scope"] = scope
        if on_complete_handler:
            on_complete_handler = _normalize_template_version(on_complete_handler)
            if on_complete_handler not in self._completion_handlers:
                raise ValueError(
                    f"Unknown on_complete_handler: {on_complete_handler!r}. "
                    f"Known: {sorted(self._completion_handlers)}"
                )
            entry["on_complete_handler"] = on_complete_handler

        # Notify frontend: task queued (creates subtab in "waiting" state).
        try:
            notification = self._make_task_status_payload(
                entry,
                session_id=self._session_id,
                task_id=task_id,
                status="queued",
                request=title[:80],
                hypothesis_id=hypothesis_id,
                queue_position=len(
                    [e for e in wc.task_queue if e["status"] == "queued"]
                ),
                queue_total=len(wc.task_queue),
            )
            # Persist task_ref.metadata.task_status BEFORE WS emit so reconnect
            # clients restore correct chip status from disk.
            await self._persist_task_status(task_id, "queued")
            await self._emit(notification)
        except Exception:
            pass

        # Try to start the next task if a slot is available.
        self._start_queue_runner()

        return task_id

    def _start_queue_runner(self) -> None:
        """Run ``_try_start_next_task`` as a task the hub keeps (and hands to
        ``track_task``), unless the queue is closed."""
        if self._queue_closed:
            return
        try:
            runner = asyncio.get_running_loop().create_task(self._try_start_next_task())
        except RuntimeError:
            logger.debug("Task queue: no event loop available to start a runner")
            return
        self._queue_runners.add(runner)
        runner.add_done_callback(self._queue_runners.discard)
        if self._track_task is not None:
            try:
                self._track_task(f"hub-queue-{uuid.uuid4().hex[:8]}", runner)
            except Exception as e:
                logger.warning("track_task failed: %s", e)

    async def aclose(self) -> None:
        """Close the job queue: no queued job starts from now on, and the jobs
        running are cancelled (their subprocess trees killed) and awaited."""
        self._queue_closed = True
        current = asyncio.current_task()
        runners = [t for t in self._queue_runners if t is not current and not t.done()]
        for runner in runners:
            runner.cancel()
        if runners:
            await asyncio.gather(*runners, return_exceptions=True)

    async def join(self) -> None:
        """Wait until the jobs this controller's queue started, and the ones
        they chain, have ended. Cancelling the wait closes the queue
        (``aclose``)."""
        try:
            while self._queue_runners:
                await asyncio.wait(set(self._queue_runners))
        except asyncio.CancelledError:
            await self.aclose()
            raise

    async def _record_cancelled_task(
        self, task_id: str, next_entry: dict[str, Any]
    ) -> None:
        """Mark a cancelled queue task as errored, persist, and fire its
        completion handler (shielded: the caller is being cancelled)."""
        wc = self._wc
        entry_now = wc.get_entry(task_id)
        if entry_now is not None and entry_now.get("status") not in (
            "completed",
            "error",
        ):
            wc.mark_error(task_id, error="cancelled by user")
        try:
            await asyncio.shield(self._persist())
        except (asyncio.CancelledError, Exception):
            pass
        # Fire the completion handler on cancel.
        cancel_handler_key = (
            (next_entry.get("on_complete_handler") or "")
            if isinstance(next_entry, dict)
            else ""
        )
        if not cancel_handler_key:
            return
        cancel_handler_key = _normalize_template_version(cancel_handler_key)
        cancel_handler = self._completion_handlers.get(cancel_handler_key)
        if cancel_handler is None:
            return
        try:
            fresh = wc.get_entry(task_id) or next_entry
            await asyncio.shield(cancel_handler(fresh))
        except (asyncio.CancelledError, Exception) as hook_err:
            logger.warning(
                "cancel-time completion handler %r failed for %s: %s",
                cancel_handler_key,
                task_id,
                hook_err,
            )

    async def _try_start_next_task(self) -> None:
        """Start the next queued task if a slot is available. Non-recursive."""
        if self._queue_closed:
            return
        wc = self._wc
        next_entry = wc.get_next_runnable()
        if next_entry is None:
            return

        task_id = next_entry["task_id"]
        tool_name = next_entry["tool_name"]
        args = next_entry.get("args", {})

        logger.info("Task queue: starting %s (tool=%s)", task_id, tool_name)
        # Pass workspace=None so any existing workspace value is preserved.
        wc.mark_running(task_id, workspace=None)

        # Register the queue task in the per-task handle map so cancellation
        # can reach it.
        try:
            current_handle = asyncio.current_task()
        except RuntimeError:
            current_handle = None
        if current_handle is not None:
            self._running_task_handles[task_id] = current_handle

        # Outer try/finally guarantees the handle is dropped and the queue is
        # chained unless the job was cancelled.
        try:
            try:
                await self._persist()
                if tool_name in ("task", "understand_codebase"):
                    if self._exec_task is None:
                        # The /task (DualInferencerBridge/PTI) path lives in
                        # the host, not the hub. Without it the hub can't run
                        # implementation tasks — surface clearly.
                        result = ToolExecutionResult(
                            result=(
                                "HubController: no exec_task callable injected; "
                                "cannot run /task (DualInferencerBridge path is "
                                "host-owned). Inject exec_task to enable "
                                "implementation tasks."
                            )
                        )
                    else:
                        result = await self._exec_task(args, task_id)
                elif tool_name == "submission_run":
                    # Submission runs go through their own sibling.
                    result = await self._exec_submission_run(
                        args, queue_task_id=task_id
                    )
                else:
                    result = ToolExecutionResult(
                        result=f"Unknown queued tool: {tool_name}"
                    )

                # Only mark completed if the entry is still non-terminal.
                entry_now = wc.get_entry(task_id)
                if entry_now is not None and entry_now.get("status") not in (
                    "completed",
                    "error",
                ):
                    wc.mark_completed(
                        task_id,
                        summary=str(result.result)[:200] if result else "",
                    )
                logger.info("Task queue: completed %s", task_id)
            except asyncio.CancelledError:
                # CancelledError is a BaseException so the broad except below
                # would NOT catch it — handle explicitly. A cancel stops the
                # hub's work (session delete, shutdown, ``aclose``): close the
                # queue so nothing queued behind this job starts.
                self._queue_closed = True
                await self._record_cancelled_task(task_id, next_entry)
                raise
            except Exception as e:
                wc.mark_error(task_id, error=str(e)[:200])
                logger.error("Task queue: error on %s: %s", task_id, e)
            await self._persist()

            # Fire the post-completion handler (if registered on the entry).
            handler_key = (
                (next_entry.get("on_complete_handler") or "")
                if isinstance(next_entry, dict)
                else ""
            )
            if handler_key:
                handler_key = _normalize_template_version(handler_key)
                handler = self._completion_handlers.get(handler_key)
                if handler is None:
                    logger.warning(
                        "on_complete_handler %r registered on task %s is "
                        "unknown; skipping (handler may have been removed "
                        "across restart)",
                        handler_key,
                        task_id,
                    )
                else:
                    try:
                        # Re-read the entry — _exec_task may have updated
                        # workspace / status after we took the snapshot.
                        fresh_entry = wc.get_entry(task_id) or next_entry
                        await handler(fresh_entry)
                        await self._persist()
                    except Exception as hook_err:
                        logger.error(
                            "Completion handler %r failed for task %s: %s",
                            handler_key,
                            task_id,
                            hook_err,
                            exc_info=True,
                        )

            # Check if all tasks for this phase are done.
            phase = next_entry.get("phase", "")
            if phase and wc.is_phase_complete(phase):
                logger.info("Task queue: phase %s fully complete", phase)
                sop_outputs: dict[str, Any] = {}
                phase_tasks = [
                    e
                    for e in wc.task_queue
                    if e["phase"] == phase and e["status"] == "completed"
                ]
                if phase_tasks:
                    sop_outputs["experiment_result"] = [
                        e.get("workspace", "") for e in phase_tasks
                    ]
                wc.complete_phase(
                    phase,
                    summary=f"All {len(phase_tasks)} tasks complete",
                    **sop_outputs,
                )
                await self._persist()
        finally:
            # Drop the queue-task handle from the registry — even on cancel.
            self._running_task_handles.pop(task_id, None)
            # Chain the next queued task (a no-op once the queue is closed).
            self._start_queue_runner()

    # ------------------------------------------------------------------
    # Implementations sidecar + grouper metadata
    # ------------------------------------------------------------------

    async def _append_hub_implementation_row(
        self,
        *,
        session_dir: Path,
        multi_task_id: str,
        row: dict[str, Any],
    ) -> None:
        """Atomic-append a row to ``hub_<mid>_implementations.json``.

        Concurrent BTA workers serialize via a per-(session, mid)
        ``asyncio.Lock`` so updates don't lose each other; cross-process writes
        are atomic via tempfile + rename.
        """
        lock_key = (self._session_id, multi_task_id)
        lock = self._hub_impl_locks.get(lock_key)
        if lock is None:
            lock = asyncio.Lock()
            self._hub_impl_locks[lock_key] = lock

        path = session_dir / f"hub_{multi_task_id}_implementations.json"

        def _read_rows() -> list[dict[str, Any]]:
            if not path.is_file():
                return []
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                rows = doc.get("implementations", [])
                return rows if isinstance(rows, list) else []
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
                return []

        def _atomic_write(rows: list[dict[str, Any]]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "multi_task_id": multi_task_id,
                "implementations": rows,
            }
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent),
                prefix=f"hub_{multi_task_id}_impls_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        async with lock:
            rows = await asyncio.to_thread(_read_rows)
            batch_id = row.get("batch_id")
            merged: dict[str, Any] | None = None
            if batch_id:
                for i, existing in enumerate(rows):
                    if existing.get("batch_id") == batch_id:
                        merged = {**existing, **row}
                        rows[i] = merged
                        break
            if merged is None:
                rows.append(dict(row))
            await asyncio.to_thread(_atomic_write, rows)

    def _collect_hypothesis_metadata_for_grouper(
        self, selected_ids: set[str]
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, str],
        dict[str, list[str]],
        list[tuple[str, str, str]],
    ]:
        """Read the proposals tree the WebUI also reads
        (``workflow_context.phase_outputs.research_proposals_data``) and project
        the per-H metadata, phase membership, slot assignments, and conflict
        pairs the grouper template consumes.

        All return values degrade gracefully — missing data produces empty
        containers.
        """
        try:
            wc = self._wc
            phase_outputs = (
                getattr(wc, "phase_outputs", None) if wc is not None else None
            )
            raw = (phase_outputs or {}).get("research_proposals_data")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = None
            if not isinstance(raw, dict):
                return [], {}, {}, []

            metadata: list[dict[str, Any]] = []
            phase_map: dict[str, str] = {}
            slots_map: dict[str, list[str]] = {}
            for phase in raw.get("phases") or []:
                phase_label = (phase or {}).get("label") or ""
                for proposal in (phase or {}).get("proposals") or []:
                    pid = (proposal or {}).get("id")
                    if not pid or pid not in selected_ids:
                        continue
                    metadata.append(
                        {
                            "id": pid,
                            "title": proposal.get("title") or "",
                            "description": proposal.get("description")
                            or proposal.get("approach")
                            or proposal.get("problem")
                            or "",
                            "category": proposal.get("category") or phase_label,
                            "phase": phase_label,
                            "slots": list(proposal.get("slots") or []),
                            "impact": proposal.get("impact") or "",
                        }
                    )
                    if phase_label:
                        phase_map[pid] = phase_label
                    slots = list(proposal.get("slots") or [])
                    if slots:
                        slots_map[pid] = slots

            # Conflict pairs from combo_constraints when present.
            conflict_pairs: list[tuple[str, str, str]] = []
            for constraint in raw.get("combo_constraints") or []:
                if not isinstance(constraint, dict):
                    continue
                pair_ids = [
                    h
                    for h in (constraint.get("hypotheses") or [])
                    if isinstance(h, str) and h in selected_ids
                ]
                if len(pair_ids) < 2:
                    continue
                reason = str(constraint.get("reason") or "")
                for i in range(len(pair_ids)):
                    for j in range(i + 1, len(pair_ids)):
                        conflict_pairs.append((pair_ids[i], pair_ids[j], reason))

            return metadata, phase_map, slots_map, conflict_pairs
        except Exception as e:  # pragma: no cover — best-effort
            logger.warning(
                "implement_hypothesis: collect_hypothesis_metadata "
                "failed; grouper falls back to bare-id input (%s)",
                e,
            )
            return [], {}, {}, []

    # ------------------------------------------------------------------
    # /implement-hypothesis + /experiment-hypothesis-combos
    # ------------------------------------------------------------------

    async def _exec_implement_hypothesis(
        self, args: dict[str, Any]
    ) -> ToolExecutionResult:
        """Execute /implement-hypothesis via :class:`ImplementHypothesisBridge`.

        Emits an outer multi-task ``task_status`` chip + one per-batch chip per
        LLM-grouped batch + per-batch lifecycle transitions so the run appears
        as subtasks under the Experiment Hub's Progress tab. Per-batch
        completions also write the canonical ``hub_<mid>_implementations.json``
        sidecar.
        """
        from agent_foundation.experiment_hub.implement_hypothesis_bridge import (  # @manual
            ImplementHypothesisBridge,
        )

        selected_ids = args.get("selected_ids") or []
        if not selected_ids:
            return ToolExecutionResult(
                result="[/implement-hypothesis] no hypotheses selected. "
                "Pass --select H1,H17,..."
            )

        plan_text = ""
        plan_path = args.get("plan_path") or args.get("--plan") or ""
        if plan_path:
            try:
                plan_text = Path(plan_path).read_text(encoding="utf-8")
            except OSError as e:
                return ToolExecutionResult(
                    result=f"[/implement-hypothesis] could not read --plan {plan_path!r}: {e}"
                )

        workflow_target = (
            args.get("workflow_target_path")
            or args.get("--workflow-target-path")
            or self._workflow_target_path
            or ""
        )
        session_ctx: dict[str, Any] = {}
        if workflow_target:
            session_ctx["workflow_target_path"] = workflow_target

        hub_id: str | None = args.get("hub_id") or None
        max_batch_size = int(args.get("max_batch_size") or 5)
        max_parallel = int(args.get("max_parallel") or 2)
        reuse_task = args.get("reuse_task") or None

        if reuse_task:
            multi_task_id = (
                reuse_task
                if str(reuse_task).startswith("implhyp-")
                else f"implhyp-{uuid.uuid4().hex[:8]}"
            )
        else:
            multi_task_id = f"implhyp-{uuid.uuid4().hex[:8]}"

        hypothesis_metadata, hypothesis_phase, hypothesis_slots, conflict_pairs = (
            self._collect_hypothesis_metadata_for_grouper(set(selected_ids))
        )

        bridge = ImplementHypothesisBridge(
            session_tasks_dir=self._session_tasks_dir,
            plan_text=plan_text,
            selected_ids=selected_ids,
            model=args.get("model"),
            base_inferencer_type=(args.get("base_inferencer") or "devmate_cli"),
            max_batch_size=max_batch_size,
            max_parallel=max_parallel,
            workflow_target_path=workflow_target,
            reuse_task=reuse_task,
            session_context=session_ctx,
            hub_id=hub_id,
            hypothesis_metadata=hypothesis_metadata,
            hypothesis_phase=hypothesis_phase,
            hypothesis_slots=hypothesis_slots,
            conflict_pairs=conflict_pairs,
        )

        session_id = self._session_id
        sidecar_mid = hub_id or multi_task_id

        async def _emit(payload: dict[str, Any]) -> None:
            await self._emit(payload)

        outer_multi_id: str = hub_id if hub_id else multi_task_id
        outer_task_type: str = "task" if hub_id else "multi"
        nested_in_hub: bool = bool(hub_id)

        await _emit(
            {
                "type": "task_status",
                "task_id": multi_task_id,
                "task_type": outer_task_type,
                **({"parent_task_id": hub_id} if nested_in_hub else {}),
                "multi_task_id": outer_multi_id,
                "status": "starting",
                "session_id": session_id,
                "label": "Implement Selected",
                "request": (
                    f"Implement Selected: {len(selected_ids)} hypotheses "
                    f"(max_batch_size={max_batch_size}, max_parallel={max_parallel})"
                ),
                "workspace": str(bridge.workspace),
                "metadata": {
                    "tool_name": "implement_hypothesis",
                    "selected_ids": list(selected_ids),
                    "max_batch_size": max_batch_size,
                    "max_parallel": max_parallel,
                    "hub_id": hub_id,
                    "sidecar_multi_task_id": sidecar_mid,
                    "notices": list(bridge.notices),
                    "implhyp_kind": "wrapper",
                    "implhyp_task_id": multi_task_id,
                },
            }
        )

        # Mirror the wrapper chip into wc.task_queue so it survives
        # session_state.json round-trips. status starts at "running" so the
        # dispatcher's get_next_runnable never picks it up.
        wc = self._wc
        _now = datetime.now(timezone.utc).isoformat()
        wc.task_queue.append(
            {
                "task_id": multi_task_id,
                "tool_name": "implement_hypothesis",
                "title": "Implement Selected",
                "request": (
                    f"Implement Selected: {len(selected_ids)} hypotheses "
                    f"(max_batch_size={max_batch_size}, max_parallel={max_parallel})"
                ),
                "args": {},
                "status": "running",
                "workspace": str(bridge.workspace),
                "multi_task_id": outer_multi_id,
                "parent_task_id": hub_id if nested_in_hub else None,
                "hypothesis_id": ",".join(selected_ids),
                "phase": "3",
                "created_at": _now,
                "updated_at": _now,
                "metadata": {
                    "tool_name": "implement_hypothesis",
                    "selected_ids": list(selected_ids),
                    "max_batch_size": max_batch_size,
                    "max_parallel": max_parallel,
                    "hub_id": hub_id,
                    "sidecar_multi_task_id": sidecar_mid,
                    "notices": list(bridge.notices),
                    "implhyp_kind": "wrapper",
                    "implhyp_task_id": multi_task_id,
                },
            }
        )
        await self._persist()

        async def _on_batches_grouped(batches: list[Any]) -> None:
            _now2 = datetime.now(timezone.utc).isoformat()
            for b in batches:
                await _emit(
                    {
                        "type": "task_status",
                        "task_id": f"{multi_task_id}-{b.batch_id}",
                        "task_type": "task",
                        "parent_task_id": (hub_id if nested_in_hub else multi_task_id),
                        "multi_task_id": outer_multi_id,
                        "status": "queued",
                        "session_id": session_id,
                        "label": f"B{b.batch_id}: {','.join(b.items)}",
                        "request": f"Implement batch {b.batch_id}: {','.join(b.items)}",
                        "workspace": str(bridge.workspace / "batches" / b.batch_id),
                        "metadata": {
                            "batchId": b.batch_id,
                            "batchLabel": getattr(b, "label", "") or b.batch_id,
                            "hypothesisIds": list(b.items),
                            "rationale": getattr(b, "rationale", "") or "",
                            "implhyp_kind": "batch",
                            "implhyp_task_id": multi_task_id,
                        },
                    }
                )
                wc.task_queue.append(
                    {
                        "task_id": f"{multi_task_id}-{b.batch_id}",
                        "tool_name": "implement_hypothesis_batch",
                        "title": f"B{b.batch_id}: {','.join(b.items)}",
                        "request": f"Implement batch {b.batch_id}: {','.join(b.items)}",
                        "args": {},
                        "status": "queued",
                        "workspace": str(bridge.workspace / "batches" / b.batch_id),
                        "multi_task_id": outer_multi_id,
                        "parent_task_id": (hub_id if nested_in_hub else multi_task_id),
                        "hypothesis_id": ",".join(b.items),
                        "phase": "3",
                        "created_at": _now2,
                        "updated_at": _now2,
                        "metadata": {
                            "batchId": b.batch_id,
                            "batchLabel": getattr(b, "label", "") or b.batch_id,
                            "hypothesisIds": list(b.items),
                            "rationale": getattr(b, "rationale", "") or "",
                            "implhyp_kind": "batch",
                            "implhyp_task_id": multi_task_id,
                        },
                    }
                )
            await self._persist()

        async def _on_batch_status(batch: Any, status: str, **kw: Any) -> None:
            error_msg = kw.get("error_message")
            payload: dict[str, Any] = {
                "type": "task_status",
                "task_id": f"{multi_task_id}-{batch.batch_id}",
                "task_type": "task",
                "parent_task_id": (hub_id if nested_in_hub else multi_task_id),
                "multi_task_id": outer_multi_id,
                "status": status,
                "session_id": session_id,
                "workspace": str(bridge.workspace / "batches" / batch.batch_id),
                "metadata": {
                    "batchId": batch.batch_id,
                    "batchLabel": getattr(batch, "label", "") or batch.batch_id,
                    "hypothesisIds": list(batch.items),
                    "implhyp_kind": "batch",
                    "implhyp_task_id": multi_task_id,
                    **({"error_message": error_msg} if error_msg else {}),
                },
            }
            if error_msg:
                payload["error_message"] = error_msg
            await _emit(payload)

            # Mirror status onto the persisted task_queue entry.
            entry_id = f"{multi_task_id}-{batch.batch_id}"
            entry = wc.get_entry(entry_id)
            if entry is not None:
                entry["status"] = status
                entry["updated_at"] = datetime.now(timezone.utc).isoformat()
                if "metadata" not in entry:
                    entry["metadata"] = {}
                entry["metadata"]["batchWorkspace"] = batch.workspace
                if error_msg:
                    entry["metadata"]["error_message"] = str(error_msg)[:4096]
                if status in ("completed", "error"):
                    await self._persist()

            # Sidecar callsite. ONE write per batch on terminal transition.
            if status in ("completed", "error"):
                try:
                    if self._session_dir is not None and session_id:
                        now = (
                            datetime.now(timezone.utc)
                            .isoformat()
                            .replace("+00:00", "Z")
                        )
                        actual_workspace = batch.workspace or str(
                            bridge.workspace / "batches" / batch.batch_id
                        )
                        await self._append_hub_implementation_row(
                            session_dir=Path(self._session_dir),
                            multi_task_id=sidecar_mid,
                            row={
                                "batch_id": batch.batch_id,
                                "hypothesis_ids": list(batch.items),
                                "status": status,
                                "workspace_path": actual_workspace,
                                "error_message": error_msg,
                                "createdAt": now,
                                "updatedAt": now,
                            },
                        )
                except Exception as e:  # pragma: no cover — best-effort
                    logger.warning(
                        "implement_hypothesis: append_hub_implementation "
                        "failed for batch=%s status=%s: %s",
                        batch.batch_id,
                        status,
                        e,
                    )

        # Run the bridge with both callbacks wired.
        try:
            summary = await bridge.run(
                plan_text,
                on_batches_grouped=_on_batches_grouped,
                on_batch_status=_on_batch_status,
            )
            outer_status = "completed"
            error_msg = None
        except Exception as e:
            logger.exception("implement_hypothesis bridge raised: %s", e)
            summary = (
                f"[/implement-hypothesis] bridge failed: {e}\n"
                f"workspace: {bridge.workspace}"
            )
            outer_status = "error"
            error_msg = str(e)

        # Outer chip terminal status.
        terminal_error: str | None = error_msg
        try:
            summary_path = bridge.workspace / "results" / "implementation_summary.json"
            if summary_path.is_file():
                _summary_doc = json.loads(summary_path.read_text(encoding="utf-8"))
                if (
                    isinstance(_summary_doc, dict)
                    and _summary_doc.get("status") == "error"
                ):
                    terminal_error = (
                        _summary_doc.get("error_message")
                        or terminal_error
                        or "Bridge failed; see workspace for details."
                    )
        except (OSError, ValueError):
            pass

        terminal_payload: dict[str, Any] = {
            "type": "task_status",
            "task_id": multi_task_id,
            "task_type": outer_task_type,
            **({"parent_task_id": hub_id} if nested_in_hub else {}),
            "multi_task_id": outer_multi_id,
            "status": outer_status,
            "session_id": session_id,
            "workspace": str(bridge.workspace),
            "summary": {
                "report_path": str(
                    bridge.workspace / "results" / "implementation_summary.json"
                ),
            },
            "metadata": {
                "tool_name": "implement_hypothesis",
                "selected_ids": list(selected_ids),
                "max_batch_size": max_batch_size,
                "max_parallel": max_parallel,
                "hub_id": hub_id,
                "sidecar_multi_task_id": sidecar_mid,
                "notices": list(bridge.notices),
                "implhyp_kind": "wrapper",
                "implhyp_task_id": multi_task_id,
                **({"error_message": terminal_error} if terminal_error else {}),
            },
        }
        if terminal_error:
            terminal_payload["error_message"] = terminal_error
        await _emit(terminal_payload)

        # Mirror outer_status onto the persisted wrapper entry.
        wrapper_entry = wc.get_entry(multi_task_id)
        if wrapper_entry is not None:
            wrapper_entry["status"] = outer_status
            wrapper_entry["updated_at"] = datetime.now(timezone.utc).isoformat()
            if terminal_error:
                if "metadata" not in wrapper_entry:
                    wrapper_entry["metadata"] = {}
                wrapper_entry["metadata"]["error_message"] = str(terminal_error)[:4096]

        await self._persist()
        return ToolExecutionResult(result=summary)

    async def _exec_experiment_combos(
        self, args: dict[str, Any]
    ) -> ToolExecutionResult:
        """Execute /experiment-hypothesis-combos via :class:`ExperimentBridge`.

        Standalone path that takes a manual ``--combos`` list. The
        ``--aggregate-only`` refresh path is delegated to
        :meth:`_exec_aggregator_only_refresh`.
        """
        from agent_foundation.experiment_hub.experiment_combos_bridge import (  # @manual
            aggregate_only_run,
            ExperimentCombosBridge,
            find_latest_experiment_workspace,
            parse_combos_arg,
            preflight_check_flags,
        )

        # Aggregator-only refresh path.
        if args.get("aggregate_only"):
            return await self._exec_aggregator_only_refresh(
                args,
                aggregate_only_run=aggregate_only_run,
                find_latest_experiment_workspace=find_latest_experiment_workspace,
            )

        combos_raw = args.get("combos") or args.get("--combos") or ""
        combos = parse_combos_arg(combos_raw)
        if not combos:
            return ToolExecutionResult(
                result="[/experiment-hypothesis-combos] no combos parsed. "
                "Pass --combos 'H1;H17,H8;H56_BASELINE'"
            )

        skip_preflight = bool(
            args.get("skip_preflight") or args.get("--skip-preflight")
        )
        preflight_root_arg = (
            args.get("preflight_root") or args.get("--preflight-root") or ""
        )
        if not skip_preflight and preflight_root_arg:
            blocked = preflight_check_flags(combos, Path(preflight_root_arg))
            if blocked:
                lines = [
                    f"  • combo '{key}' missing flags: {', '.join(flags)}"
                    for key, flags in blocked.items()
                ]
                return ToolExecutionResult(
                    result=(
                        "[/experiment-hypothesis-combos] pre-flight failed — "
                        "the following combos reference flags not yet declared "
                        "in the codebase. Run /implement-hypothesis first or "
                        "pass --skip-preflight to override.\n" + "\n".join(lines)
                    )
                )

        workflow_target = (
            args.get("workflow_target_path")
            or args.get("--workflow-target-path")
            or self._workflow_target_path
            or ""
        )
        session_ctx = dict(self._session_context)
        if workflow_target:
            session_ctx["workflow_target_path"] = workflow_target

        bridge = ExperimentCombosBridge(
            session_tasks_dir=self._session_tasks_dir,
            plan_text="",  # combos are explicit; no plan parsing needed
            selected_ids=[],
            combos=combos,
            model=args.get("model"),
            base_inferencer_type=(args.get("base_inferencer") or "devmate_cli"),
            max_concurrency=int(args.get("max_concurrency") or 2),
            workflow_target_path=workflow_target,
            session_context=session_ctx,
            rounds=int(args.get("rounds") or 1),
        )
        summary = await bridge.run("")
        await self._persist()
        return ToolExecutionResult(result=summary)

    async def _exec_aggregator_only_refresh(
        self,
        args: dict[str, Any],
        *,
        aggregate_only_run: Any,
        find_latest_experiment_workspace: Any,
    ) -> ToolExecutionResult:
        """Run the aggregation-only BTA over per-combo analyses on disk.

        Auto-resolves the active session's latest experiment workspace (or the
        hub submissions), runs the LLM aggregator into a staging dir, validates,
        then archive-swaps the live ``accumulated_learnings.md`` via the
        ``learnings_archive`` stage->validate->archive->swap pipeline.

        Materializes a sub-task chip (starting -> running -> completed/error)
        with the LLM input/output captured for inspection.
        """
        import secrets

        from agent_foundation.experiment_hub.experiment_bridge import (  # @manual
            ExperimentBridge,
        )

        session_tasks_dir = self._session_tasks_dir
        session_dir = self._session_dir
        session_id = self._session_id or "unknown"
        logger.info(
            "aggregate_only_refresh: invoked session=%s session_dir=%s args_keys=%s",
            session_id,
            session_dir,
            sorted(args.keys()),
        )

        # Chip lifecycle: mint task_id + workspace BEFORE early-returns.
        task_id = (
            f"agg_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            f"_{secrets.token_hex(3)}"
        )
        task_workspace = Path(session_tasks_dir) / task_id
        task_workspace.mkdir(parents=True, exist_ok=True)

        for _sub in (
            "outputs",
            "results",
            "logs",
            "analysis",
            "artifacts",
            "checkpoints",
        ):
            try:
                (task_workspace / _sub).mkdir(parents=True, exist_ok=True)
            except OSError as _bs_err:
                logger.warning(
                    "aggregate_only_refresh: bootstrap %s mkdir failed: %s",
                    _sub,
                    _bs_err,
                )

        # Per-task log handler. Filters on task_id so the chip-scoped log
        # captures only this refresh's events. Detached in _finalize.
        _per_task_fh: logging.FileHandler | None = None
        try:
            _per_task_fh = logging.FileHandler(str(task_workspace / "logs" / "run.log"))
            _per_task_fh.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            _task_id_token = f"task_id={task_id}"
            _per_task_fh.addFilter(
                lambda r, _tok=_task_id_token: _tok in (r.getMessage() or "")
            )
            logger.addHandler(_per_task_fh)
        except OSError as _fh_err:
            _per_task_fh = None
            logger.warning(
                "aggregate_only_refresh: logs/run.log handler attach failed: %s",
                _fh_err,
            )

        # Sidecar — chip<->workspace correlation on restart.
        try:
            (task_workspace / ".task_meta.json").write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "tool_name": "aggregate_only_refresh",
                        "session_id": session_id,
                        "session_dir": session_dir.name,
                    }
                ),
                encoding="utf-8",
            )
        except OSError as _meta_err:
            logger.warning(
                "aggregate_only_refresh: .task_meta.json write failed: %s",
                _meta_err,
            )

        chip_label = "Refresh Learnings"
        await self._add_task_ref_safe(
            task_id=task_id,
            label=chip_label,
            tool_name="aggregate_only_refresh",
        )

        async def _emit_status(status: str, **extra: Any) -> None:
            payload = self._make_task_status_payload(
                None,
                session_id=session_id,
                task_id=task_id,
                status=status,
                workspace=str(task_workspace),
                tool_name="aggregate_only_refresh",
                **extra,
            )
            await self._persist_task_status(task_id, status)
            await self._emit(payload)

        async def _finalize(
            status: str,
            summary_md: str,
            result_text: str,
            *,
            error_message: str | None = None,
        ) -> ToolExecutionResult:
            """Single exit point: write summary.md -> flip chip status ->
            return ToolExecutionResult. Used for both success and every
            failure early-return so no path leaves an orphan chip state.
            """
            nonlocal _per_task_fh
            if _per_task_fh is not None:
                try:
                    logger.removeHandler(_per_task_fh)
                    _per_task_fh.close()
                except Exception:
                    pass
                _per_task_fh = None

            try:
                (task_workspace / "summary.md").write_text(summary_md, encoding="utf-8")
            except OSError as _sum_err:
                logger.warning(
                    "aggregate_only_refresh: summary.md write failed: %s",
                    _sum_err,
                )
            extras: dict[str, Any] = {}
            if status == "error" and error_message:
                extras["error_message"] = str(error_message)[:4096]
            await _emit_status(status, **extras)
            await self._persist()
            return ToolExecutionResult(result=result_text)

        await _emit_status("starting", request=chip_label)

        from agent_foundation.experiment_hub.experiment_combos_bridge import (  # @manual
            collect_aggregator_input_from_submissions,
            resolve_active_multi_task_id,
        )

        # Pipeline pre-checks: hub-driven inputs first, picker fallback second.
        from agent_foundation.experiment_hub.learnings_generator import (  # @manual
            _load_baseline_choice_id,
            _load_session_inputs,
            precompute_actions,
        )

        try:
            _submissions, _proposals_data, _overrides = _load_session_inputs(
                session_dir,
            )
        except Exception as _load_err:
            logger.warning(
                "aggregate_only_refresh: _load_session_inputs failed: %s",
                _load_err,
            )
            _submissions, _proposals_data, _overrides = [], None, None

        explicit_mid = args.get("reuse_hub") or args.get("--reuse-hub")
        wc_active = getattr(self._wc, "active_multi_task_id", None)
        mid, mid_err = resolve_active_multi_task_id(
            session_dir,
            explicit=explicit_mid,
            workflow_context_active=wc_active,
        )

        min_epochs = int(args.get("min_epochs") or 0)
        exclude_incomparable = bool(
            args.get("exclude_incomparable") or args.get("--exclude-incomparable")
        )
        exclude_errored = bool(
            args.get("exclude_errored") or args.get("--exclude-errored")
        )

        hub_inputs: list[dict[str, str]] = []
        if _submissions:
            hub_inputs = collect_aggregator_input_from_submissions(
                _submissions,
                session_dir,
                min_epochs=min_epochs,
                include_incomparable=not exclude_incomparable,
                include_errored=not exclude_errored,
            )

        workspace = (
            None
            if hub_inputs
            else find_latest_experiment_workspace(Path(session_tasks_dir))
        )

        source = (
            f"hub:{mid}"
            if hub_inputs
            else f"exp_glob:{workspace.name}"
            if workspace
            else "none"
        )

        if not hub_inputs and workspace is None:
            err = mid_err or (
                "No hub submissions and no qualifying exp_*/combos/* "
                "workspace found. Run /experiment-hypothesis-combos with "
                "--combos first, OR wait for at least one hub submission "
                "to reach a terminal status."
            )
            return await _finalize(
                "error",
                f"# Refresh Learnings — failed\n\n{err}\n",
                f"[/experiment-hypothesis-combos --aggregate-only] {err}",
                error_message=err,
            )

        target_path_arg = (
            args.get("aggregate_target") or args.get("--aggregate-target") or ""
        )
        if target_path_arg:
            target_path = Path(target_path_arg)
        else:
            target_path = session_dir / "_learnings" / "accumulated_learnings.md"

        session_ctx = dict(self._session_context)

        def _agg_factory():
            # Pass workspace_path=task_workspace so ExperimentBridge's
            # _create_llm_inferencer bakes the cache_folder into the DevmateCli
            # at construction (controls where stream_*.txt files get written).
            tmp_bridge = ExperimentBridge(
                session_tasks_dir=session_tasks_dir,
                plan_text="",
                selected_ids=[],
                combos=[],
                model=args.get("model"),
                base_inferencer_type=(args.get("base_inferencer") or "devmate_cli"),
                max_concurrency=1,
                workflow_target_path="",
                session_context=session_ctx,
                rounds=1,
                workspace_path=Path(task_workspace),
            )
            _agg_inf = tmp_bridge._build_dual(
                role="accumulated_learnings",
                template_space="aggregation",
                template_version="accumulated_learnings",
                max_iterations=5,
                debug_mode=True,
            )
            # TODO(port): the RankEvolve SessionLogger attach here used
            # rankevolve's JsonLogger/Debuggable internals
            # (utils.io_utils.json_io, utils.common_objects.debuggable) which
            # are not part of this port's scope. The original wrapped this in
            # try/except (best-effort: logs/session/ population only). Omitted
            # the attach; the inferencer's own cache_folder streaming + the
            # per-task run.log still work. Wire AF's structured logger here
            # once an equivalent exists.
            return _agg_inf

        from agent_foundation.experiment_hub import learnings_archive as la  # @manual
        from agent_foundation.experiment_hub.learnings_generator import (  # @manual
            regenerate_accumulated_learnings,
        )

        force_refresh = bool(args.get("force_refresh") or args.get("--force-refresh"))
        archive_keep = int(
            args.get("archive_keep")
            or args.get("--archive-keep")
            or la.DEFAULT_KEEP_LAST_N
        )
        archive_reason = (
            args.get("archive_reason")
            or args.get("--archive-reason")
            or "refresh from UI"
        )
        archive_source = (
            args.get("archive_source") or args.get("--archive-source") or "refresh-llm"
        )
        triggered_by = args.get("triggered_by") or "system"

        try:
            (task_workspace / "request.txt").write_text(
                f"aggregate_only_refresh "
                f"source={source!r} "
                f"min_epochs={min_epochs} "
                f"exclude_incomparable={exclude_incomparable} "
                f"exclude_errored={exclude_errored} "
                f"force_refresh={force_refresh} "
                f"archive_keep={archive_keep} "
                f"archive_source={archive_source!r} "
                f"triggered_by={triggered_by!r}\n",
                encoding="utf-8",
            )
        except OSError as _req_err:
            logger.warning(
                "aggregate_only_refresh: request.txt write failed: %s",
                _req_err,
            )

        try:
            inputs_index = [
                {
                    "combo_id": e.get("combo_id", ""),
                    "path": e.get("path", "") or "",
                    "summary_bytes": len((e.get("summary") or "").encode("utf-8")),
                    "source": source,
                }
                for e in hub_inputs
            ]
            (task_workspace / "analysis" / "inputs.json").write_text(
                json.dumps(inputs_index, indent=2),
                encoding="utf-8",
            )
        except OSError as _idx_err:
            logger.warning(
                "aggregate_only_refresh: analysis/inputs.json write failed: %s",
                _idx_err,
            )

        await _emit_status("running")

        # Acquire per-session lock for the entirety of the staging pipeline.
        lock = await la._refresh_lock_for(session_id)
        async with lock:
            la._clean_staging(session_dir, run_id=None)

            run_id = la.mint_run_id()
            staging_dir = la.staging_dir_for(session_dir, run_id)
            staged_md_path = staging_dir / la.LIVE_MD_NAME
            la.write_stage_meta(
                staging_dir,
                {
                    "schema_version": la.SCHEMA_VERSION,
                    "run_id": run_id,
                    "kind": "aggregate_only_refresh",
                    "source": archive_source,
                    "status": "running",
                    "started_at": la._utc_iso(),
                    "reason": archive_reason,
                    "triggered_by": triggered_by,
                    "source_experiment_workspace": (
                        str(workspace) if workspace else f"hub:{mid or 'unknown'}"
                    ),
                    "task_id": task_id,
                },
            )

            precompute_envelope = None
            try:
                _baseline_id = _load_baseline_choice_id(session_dir)
                precompute_envelope = precompute_actions(
                    _submissions,
                    _proposals_data,
                    _overrides,
                    baseline_submission_id=_baseline_id,
                )
            except Exception as e:
                logger.warning(
                    "aggregate_only_refresh: failed to compute precompute "
                    "envelope (%s); LLM will emit empty rerank/combo arrays.",
                    e,
                )

            if precompute_envelope is not None:
                try:
                    (
                        task_workspace / "analysis" / "precompute_envelope.json"
                    ).write_text(
                        json.dumps(precompute_envelope, indent=2, default=str),
                        encoding="utf-8",
                    )
                except (OSError, TypeError) as _env_err:
                    logger.warning(
                        "aggregate_only_refresh: precompute_envelope.json "
                        "write failed: %s",
                        _env_err,
                    )

            try:
                llm_md = await aggregate_only_run(
                    workspace if workspace else session_dir,
                    staged_md_path,
                    aggregator_factory=_agg_factory,
                    precompute_envelope=precompute_envelope,
                    task_workspace=task_workspace,
                    inputs=hub_inputs if hub_inputs else None,
                )
            except ValueError as e:
                la.update_stage_meta(
                    staging_dir,
                    status="failed",
                    validation_errors=[f"aggregator: {e}"],
                )
                return await _finalize(
                    "error",
                    f"# Refresh Learnings — failed\n\nAggregator error: {e}\n",
                    f"[/experiment-hypothesis-combos --aggregate-only] {e}",
                    error_message=f"Aggregator error: {e}",
                )

            try:
                regenerate_accumulated_learnings(
                    session_dir,
                    override_md=llm_md,
                    target_path=staged_md_path,
                )
            except Exception as e:
                logger.exception(
                    "aggregate_only_refresh: building staged content failed: %s", e
                )
                la.update_stage_meta(
                    staging_dir,
                    status="failed",
                    validation_errors=[f"merge: {e}"],
                )
                return await _finalize(
                    "error",
                    f"# Refresh Learnings — failed\n\nMerge step failed: {e}\n",
                    f"[/experiment-hypothesis-combos --aggregate-only] merge failed: {e}",
                    error_message=f"Merge step failed: {e}",
                )

            la.update_stage_meta(
                staging_dir,
                status="staged",
                llm_response_size_bytes=len(llm_md.encode("utf-8")),
            )

            v = la._validate_staged(
                staged_md_path,
                current_md_path=la._live_md(session_dir),
            )

            try:
                (task_workspace / "results" / "staging_verdict.json").write_text(
                    json.dumps(
                        {
                            "ok": v["ok"],
                            "no_op": v["no_op"],
                            "errors": v.get("errors") or [],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except OSError as _vd_err:
                logger.warning(
                    "aggregate_only_refresh: results/staging_verdict.json "
                    "write failed: %s",
                    _vd_err,
                )

            if not v["ok"]:
                la.update_stage_meta(
                    staging_dir,
                    status="invalid",
                    validation_errors=v["errors"],
                )
                errs = "; ".join(v["errors"])
                return await _finalize(
                    "error",
                    f"# Refresh Learnings — validation failed\n\n"
                    f"Errors: {errs}\n\n"
                    f"Staging kept at `{staging_dir}` for inspection.\n",
                    "[/experiment-hypothesis-combos --aggregate-only] "
                    f"validation failed: {errs}. "
                    f"Staging kept at {staging_dir} for inspection.",
                    error_message=f"Validation failed: {errs}",
                )
            if v["no_op"] and not force_refresh:
                la._clean_staging(session_dir, run_id=run_id)
                return await _finalize(
                    "completed",
                    "# Refresh Learnings — no change\n\n"
                    "The LLM produced output identical to the live doc "
                    "(md5 match). No archive entry created. Use "
                    "`--force-refresh` to override.\n",
                    "[/experiment-hypothesis-combos --aggregate-only] "
                    "no effective change since last refresh (md5 match). "
                    "Use --force-refresh to override.",
                )

            commit = await la.archive_current_and_promote_staged(
                session_dir,
                staging_dir,
                run_id=run_id,
                kind="aggregate_only_refresh",
                source=archive_source,
                reason=archive_reason,
                triggered_by=triggered_by,
                source_combo_hashes=None,
                baseline_submission_id=None,
                keep_last_n=archive_keep,
                already_locked=True,
            )

            try:
                _live_text = la._live_md(session_dir).read_text(encoding="utf-8")
                (task_workspace / "outputs" / "accumulated_learnings.md").write_text(
                    _live_text,
                    encoding="utf-8",
                )
            except OSError as _live_err:
                logger.warning(
                    "aggregate_only_refresh: outputs/accumulated_learnings.md "
                    "copy failed: %s",
                    _live_err,
                )
            try:
                (task_workspace / "outputs" / "archive_commit.json").write_text(
                    json.dumps(commit, indent=2, default=str),
                    encoding="utf-8",
                )
            except (OSError, TypeError) as _ac_err:
                logger.warning(
                    "aggregate_only_refresh: outputs/archive_commit.json "
                    "write failed: %s",
                    _ac_err,
                )

        target_str = str(la._live_md(session_dir))

        first_ever = bool(commit.get("first_ever"))
        if first_ever:
            heading = "# Refresh Learnings — initial (no prior version)"
            archive_id_line = "- **Archive id**: _none — first-ever refresh_"
            archive_dir_line = "- **Archive dir**: _none — nothing to preserve_"
            result_lines = [
                f"- New live narrative at `{target_str}`",
                "- No archive entry created (no prior version existed); "
                "the next refresh will archive THIS content as v1.",
            ]
            result_text = (
                "[/experiment-hypothesis-combos --aggregate-only] "
                f"initial narrative committed at {target_str} "
                "(no prior version to archive)"
            )
        else:
            archive_dir = la._archive_root(session_dir) / commit["archive_id"]
            heading = f"# Refresh Learnings — v{commit['version']}"
            archive_id_line = f"- **Archive id**: `{commit['archive_id']}`"
            archive_dir_line = f"- **Archive dir**: `{archive_dir}`"
            result_lines = [
                f"- New live narrative at `{target_str}`",
                f"- Prior version archived at `{archive_dir}`",
            ]
            result_text = (
                "[/experiment-hypothesis-combos --aggregate-only] refreshed "
                f"narrative at {target_str} · archive {commit['archive_id']} "
                f"(v{commit['version']}, source={archive_source}"
                + (
                    f", pruned {commit['archives_pruned']}"
                    if commit.get("archives_pruned")
                    else ""
                )
                + ")"
            )

        success_summary = "\n".join(
            [
                heading,
                "",
                archive_id_line,
                f"- **Source**: `{archive_source}`",
                f"- **Reason**: {archive_reason}",
                f"- **Triggered by**: `{triggered_by}`",
                f"- **Live doc**: `{target_str}`",
                archive_dir_line,
                f"- **Archives pruned**: {commit.get('archives_pruned', 0)}",
                f"- **Precompute envelope supplied**: "
                f"{'yes' if precompute_envelope is not None else 'no'}",
                "",
                "## Inputs",
                f"- Per-combo analyses source: {source}",
                "",
                "## Result",
                *result_lines,
            ]
        )
        return await _finalize("completed", success_summary, result_text)

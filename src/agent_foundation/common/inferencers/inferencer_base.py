import asyncio
import logging
import os
import sys
from abc import ABC, abstractmethod
from contextvars import ContextVar
from functools import partial
from typing import Any, AsyncIterator, Callable, Iterable, Iterator, List, Optional, Sequence, Type, Union

from attr import attrib, attrs
from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_objects.workflow.common.resumable import Resumable
from rich_python_utils.common_utils import dict_, iter__, resolve_environ
from rich_python_utils.common_utils.function_helper import FallbackMode, execute_with_retry

# Retry prompt mode constants
RETRY_PROMPT_MODES = ("original", "simple_retry", "retry_with_original")

# Module-level ContextVar for per-call fallback state.
# Each asyncio Task from aparallel_infer gets its own context copy — no race.
# Shared with streaming_inferencer_base.py for cache path tracking.
_current_fallback_state: ContextVar[dict | None] = ContextVar("_current_fallback_state", default=None)

_SIMPLE_RETRY_PROMPT = "You got interrupted. Can you retry the above task?"

_logger = logging.getLogger(__name__)


@attrs
class InferencerBase(Debuggable, Resumable, ABC):
    merger__ = """
    Abstract base class for implementing inference logic with retry and fallback recovery.

    This class provides a framework for executing inference tasks with built-in retry mechanisms,
    timeout support, and fallback chain recovery. Subclasses should implement the `_infer` method
    to define specific inference behavior and optionally override other methods to customize prompt
    formatting, output validation, and recovery strategies.

    Retry semantics:
        The default ``fallback_mode`` is ``ON_FIRST_FAILURE``, which means the primary ``_infer``/
        ``_ainfer`` gets one attempt, then ``_infer_recovery``/``_ainfer_recovery`` gets ``max_retry``
        attempts. Total attempts = ``1 + max_retry``. To restore the pre-fallback behavior (all
        attempts call the same function), set ``fallback_mode=FallbackMode.NEVER``.

        Sync path: ``execute_with_retry`` uses ``while True`` + ``attempts >= max_retry: break``
        = initial + max_retry calls per callable.
        Async path: ``async_execute_with_retry`` uses ``for attempt in range(max_retry)``
        = max_retry calls per callable.

    Timeout support:
        ``total_timeout_seconds``: Wall-clock cap for the entire retry+fallback loop (sync and async).
        ``attempt_timeout_seconds``: Per-attempt cap via ``asyncio.wait_for`` (async only).
        Sync path raises ``NotImplementedError`` if ``attempt_timeout_seconds > 0``.

    Attributes:
        model_id (str): The identifier of the model to be used for inference.
        secret_key (str): Secret key for authentication, if needed. Defaults to None.
        max_retry (int): The maximum number of retry attempts. With the default
            ``fallback_mode=ON_FIRST_FAILURE``, total attempts = 1 + max_retry.
            With ``fallback_mode=NEVER``, total attempts = max_retry (async) or 1 + max_retry (sync).
        min_retry_wait (float): Minimum wait time in seconds between retry attempts. Defaults to 0.
        max_retry_wait (float): Maximum wait time in seconds between retry attempts. Defaults to 0.
        default_return_or_raise (Union[Any, Exception]): Value to return or exception to raise on failure.
        total_timeout_seconds (float): Wall-clock cap in seconds. 0 = disabled. Applies to both sync and async.
        attempt_timeout_seconds (float): Per-attempt timeout in seconds. 0.0 = disabled. Async only.
        fallback_inferencer: External fallback inferencer(s). None = self-recovery only.
        fallback_mode (FallbackMode): When to transition to fallback. Default: ON_FIRST_FAILURE.
        default_inference_args (dict): Default arguments passed to ``_infer``.
        input_preprocessor (Callable): Optional input preprocessor.
        response_post_processor (Callable): Optional response post-processor.
        post_response_merger (str, Callable): Optional response merger for iterator inputs.
    """

    model_id: str = attrib(default="")
    _secret_key: Union[str, Sequence[str]] = attrib(default=None)

    # Class-level fire-once warning sets
    _paired_override_warned: set = set()
    _nested_retry_warned: set = set()

    # region retry parameters
    max_retry: int = attrib(default=1)
    min_retry_wait: float = attrib(default=0)
    max_retry_wait: float = attrib(default=0)
    default_return_or_raise: Union[Any, Exception] = attrib(default=None)
    # endregion

    # Total timeout in seconds for the entire retry+fallback loop.
    # 0 = disabled (backward compatible). Applies to both sync (_infer_single)
    # and async (_ainfer_single) paths via the retry helpers.
    total_timeout_seconds: float = attrib(default=0)

    # Per-attempt timeout in seconds. 0.0 = disabled. Float for sub-second granularity.
    # Async only — sync path raises NotImplementedError if > 0.
    attempt_timeout_seconds: float = attrib(default=0.0)

    # External fallback inferencer(s) to try when the primary _infer/_ainfer fails.
    # None = no external fallback (self-recovery via _infer_recovery/_ainfer_recovery still applies).
    fallback_inferencer: Union["InferencerBase", List["InferencerBase"], None] = attrib(default=None)

    # Controls when the retry helper transitions to the next fallback callable.
    fallback_mode: FallbackMode = attrib(default=FallbackMode.ON_FIRST_FAILURE)

    response_types: Sequence[Type] = attrib(default=(str,))
    default_inference_args: dict = attrib(default=None, converter=dict_)
    input_preprocessor: Callable = attrib(default=None)
    response_post_processor: Union[str, dict, Callable] = attrib(default=None)
    post_response_merger: Union[str, Callable] = attrib(default=None)

    # State graph support — optional list of StateGraphTracker instances
    state_graphs: list = attrib(default=None)

    # Whether this inferencer has local file system access (e.g., can write files).
    # False for cloud API inferencers (RovoChat), True for local agents (RovoDevCli).
    has_local_access: bool = attrib(default=False)

    # Default output path — when relative and a workspace is set (via _workspace
    # on flow inferencers), resolves to workspace.outputs_dir/<output_path>.
    # Simple API inferencers leave this as None.
    output_path: Optional[str] = attrib(default=None)

    output_is_deliverable: bool = attrib(default=False)
    """When True, output file is copied into final_deliverables/ after
    finalization. Extends BTA/PTI's publishes_response_as_deliverable
    concept to any inferencer."""

    output_manifest_index: bool = attrib(default=False)
    """When True, emit output_manifest.json listing contributing files
    (LLM prompts, streaming cache, session logs). Auto-enables when
    output_is_deliverable is True."""

    # === Workspace (opt-in) ===
    # Construction-time workspace. Synced to ``_workspace`` (property) in
    # ``__attrs_post_init__``, which auto-configures cache_folder
    # and logger via ``_configure_for_workspace()``.
    # (Subprocess cwd is derived at call time via effective_cwd, not stored.)
    #
    # For RUNTIME workspace assignment (e.g., orchestrators assigning child
    # workspaces), use ``child._workspace = ws`` directly — the property setter
    # triggers auto-configuration. Do NOT assign to ``workspace`` at runtime;
    # it's a plain attrs field with no setter side effects.
    #
    # `workspace` is the **declarative input slot** — what users / YAML
    # actually configure. It's a plain attrib (no setter side effects) so
    # config-time assignment is cheap and predictable. The companion
    # `_workspace` property below carries the **runtime-mutable backing**
    # that the rest of the framework reads/writes throughout an inference
    # call. `__attrs_post_init__` syncs `workspace → _workspace` once.
    #
    # Historically there was also a `workspace_root: Optional[str]` attrib
    # on LWI/Dual/BTA/MFDual as a string-shorthand convenience. It has been
    # removed (2026-05-05) — pass the full workspace object instead, e.g.:
    #     Dual(workspace=InferencerWorkspace(root="/tmp/foo"))
    # The shorthand was strictly less expressive (it could not carry flags
    # like `use_final_deliverables_folder`) and led to a subtle clobber bug
    # where its post-init logic overwrote the synced `_workspace = None`.
    # Type ``Optional[Any]`` so that:
    #   1. Hydra config dicts (pre-instantiation) pass through validation
    #   2. ``InferencerWorkspace`` objects (post-instantiation, the canonical form)
    #   3. ``str`` shorthand (post-instantiation, converted to ``InferencerWorkspace(root=<str>)``
    #      in ``__attrs_post_init__``)
    # All three are valid inputs.
    workspace: Optional[Any] = attrib(default=None)

    # === Source path (auto-detected) ===
    # Where this inferencer's own source code lives (project root).
    # Auto-detected via _detect_source_root() in __attrs_post_init__
    # if not explicitly set. Returns None for test stubs, pip-installed
    # packages, or non-src-layout projects — this is intentional.
    source_path: Optional[str] = attrib(default=None)

    # === Deliverable Boundary Semantics (v1.7) ===
    # Whether this inferencer is a Deliverable Boundary. When True, the
    # inferencer guarantees that its outputs/final_deliverables/ is the
    # canonical artifact set visible to its immediate parent boundary.
    #
    # Subclass defaults (set on each subclass, NOT here):
    #   - BreakdownThenAggregateInferencer  → True
    #   - PlanThenImplementInferencer       → True
    #   - DualInferencer                    → False (pass-through)
    #   - LinearWorkflowInferencer          → False (pass-through)
    #   - All API/CLI leaves                → False (default)
    #
    # The boundary mechanism only ACTIVATES when use_final_deliverables_folder
    # is True on the workspace. With the default workspace (no flag), the
    # Legacy boundary attr — kept for backward compatibility but no longer
    # used by the unified _finalize_output symlink architecture.
    is_deliverable_boundary: bool = attrib(default=False)

    # === Template-based prompt rendering (opt-in) ===
    # Template fields (template_manager, template_key, template_root_space,
    # template_extra_feed, template_variables) live on TemplatedInferencerBase
    # (a subclass of InferencerBase). Inferencers that consume their own input
    # via an LLM call (CLI/API/streaming leaves and their intermediate bases
    # ApiInferencerBase / StreamingInferencerBase / RemoteInferencerBase /
    # TerminalInferencerBase) inherit from TemplatedInferencerBase. Orchestrators
    # (BTA, Dual, LWI, PTI, MultiFlow, MultiFlowDual, Conversational*) inherit
    # from InferencerBase directly and don't carry template state — cascade
    # injection of `_template_manager` naturally skips them.
    #
    # Method names `_render_prompt`, `_propagate_to_children`, and
    # `supports_prompt_rendering` exist on InferencerBase as no-op stubs so
    # `_ainfer_single`'s unconditional calls to them work for orchestrators.
    # TemplatedInferencerBase overrides each with its real implementation.

    # --- _workspace property: runtime-mutable backing for `workspace` ---
    #
    # PURPOSE: this is the **active, mutable workspace** that the rest of
    # the framework actually reads from. The leading underscore is a NAMING
    # WART (`_workspace` looks "private" by Python convention but in fact
    # every orchestrator reads/writes it via `self._workspace.child(...)`,
    # `self._workspace.ensure_dirs()`, `getattr(self, '_workspace', None)`).
    # Treat it as a public, runtime-mutable cousin of the declarative
    # `workspace` attrib above.
    #
    # WHY TWO FIELDS:
    #   * `workspace` (above)  → static input from YAML / kwargs;
    #                            attrib with no setter side effects;
    #                            never reassigned after construction.
    #   * `_workspace` (here)  → mutable runtime field;
    #                            assignment triggers `_configure_for_workspace`
    #                            and `_propagate_workspace_to_children`;
    #                            re-set during PTI iterations
    #                            (workspace.child("iter_0"), iter_1, ...).
    #
    # SYNC: `InferencerBase.__attrs_post_init__` runs `self._workspace =
    # self.workspace` ONCE at construction time, copying the user's input
    # into the runtime field and firing the side effects.
    #
    # The backing storage uses name mangling (`_InferencerBase__workspace`)
    # so subclasses can't accidentally shadow it with their own `_workspace`
    # attrib.
    @property
    def _workspace(self):
        return getattr(self, "_InferencerBase__workspace", None)

    @_workspace.setter
    def _workspace(self, value):
        object.__setattr__(self, "_InferencerBase__workspace", value)
        if value is not None:
            self._configure_for_workspace(value)
            self._propagate_workspace_to_children(value)
        # Auto-invalidate derived state when workspace changes.
        for attr in getattr(type(self), '_DERIVED_FROM_WORKSPACE', ()):
            self.__dict__.pop(attr, None)

    # Subclasses may override this tuple to declare instance attributes that
    # are derived from ``_workspace`` and must be invalidated (removed from
    # ``__dict__``) whenever the workspace is reassigned.  This prevents
    # stale cached paths from surviving across consensus iterations or
    # workspace swaps.
    _DERIVED_FROM_WORKSPACE: tuple = ()

    @classmethod
    def _detect_source_root(cls) -> "Optional[str]":
        """Auto-detect the project root for this inferencer class.

        Walks up from the class's source file to find the nearest parent
        of a ``src`` directory (standard Python src-layout). Returns None
        for test stubs, examples, pip-installed packages, or non-src-layout
        projects. Subclasses may override for non-standard layouts.
        """
        import inspect
        try:
            source_file = inspect.getfile(cls)
        except (TypeError, OSError):
            return None
        current = os.path.dirname(os.path.abspath(source_file))
        while True:
            parent = os.path.dirname(current)
            if parent == current:
                return None
            if os.path.basename(current) == "src":
                return parent
            current = parent

    # Attrs whose values are semantically relevant when switching roles.
    # Subclasses extend this tuple in their own class body.
    _ROLE_RELEVANT_ATTRS: tuple = ("output_is_deliverable", "is_deliverable_boundary")

    def switch_role(self, new_role: str, *, workspace=None,
                    output_is_deliverable=None, is_deliverable_boundary=None,
                    reset_session=True):
        """Transition this inferencer to a new semantic role.

        Centralises the workspace-swap + session-reset + flag-update pattern
        that orchestrators (MFDual, PTI, ...) previously performed inline.
        The base layer handles workspace assignment, deliverable flags, and
        session reset; TemplatedInferencerBase extends with template attrs.

        Args:
            new_role: human-readable role name (e.g. 'fixer_inferencer').
            workspace: if not None, assigned via the _workspace property
                setter (triggers _configure_for_workspace cascade).
            output_is_deliverable: if not None, overrides the flag.
            is_deliverable_boundary: if not None, overrides the flag.
            reset_session: if True, calls self.reset_session() (when available).
        """
        import time
        # 1. Workspace assignment FIRST — triggers cascade
        if workspace is not None:
            self._workspace = workspace
        # 2. Override deliverable flags
        for attr, val in {"output_is_deliverable": output_is_deliverable,
                          "is_deliverable_boundary": is_deliverable_boundary}.items():
            if val is not None:
                setattr(self, attr, val)
        # 3. Session reset
        if reset_session and hasattr(self, "reset_session"):
            self.reset_session()
        # 4. Audit trail (_role_history, lazy-init)
        history = getattr(self, "_role_history", None)
        if history is None:
            history = []
            object.__setattr__(self, "_role_history", history)
        changes = {**({"workspace": str(workspace.root)} if workspace else {}),
                   **{k: v for k, v in {"output_is_deliverable": output_is_deliverable,
                      "is_deliverable_boundary": is_deliverable_boundary}.items() if v is not None}}
        # Merge template-layer changes stashed by TemplatedInferencerBase.switch_role
        pending = getattr(self, "_pending_role_changes", None)
        if pending:
            changes.update(pending)
            object.__setattr__(self, "_pending_role_changes", None)
        history.append({"to_role": new_role, "at": time.time(), "changes": changes})

    # Subclasses may override this class attribute to declare which attrs
    # they manage workspace assignment for themselves (e.g., PTI's runtime
    # ``_setup_child_workflows`` claims planner/executor children with
    # iter_<N>-aware paths). Generic propagation skips those attrs to avoid
    # creating orphan dirs at construction time.
    _workspace_propagation_skip: frozenset = frozenset()

    def _propagate_workspace_to_children(self, parent_workspace):
        """When a workspace is assigned, give each direct child inferencer a
        child workspace.

        Each direct child ``InferencerBase`` reachable via
        ``_for_each_child_inferencer`` gets ``parent_workspace.child(<slot>)``
        assigned to its ``_workspace`` — but only if the child doesn't already
        have a workspace (respects explicit pre-assignment by the caller, e.g.
        BTA's ``worker._workspace = self._workspace.child(f"worker_{i}")``).

        The child's own setter triggers recursively, so the propagation
        cascades down the entire tree without each subclass needing to
        re-implement it. This is symmetric with how ``_propagate_to_children``
        already cascades ``template_extra_feed``.

        Skips ``functools.partial`` factories (e.g. BTA's ``worker_factory``);
        their instances get workspaces at runtime when the factory is invoked.

        Subclasses can opt out of propagation for specific attrs by setting
        the class-level ``_workspace_propagation_skip`` to a frozenset of
        attr names. Used by PTI to suppress propagation to ``_CHILD_DEFAULTS``
        keys, since PTI's runtime ``_setup_child_workflows`` claims those
        children with iter_<N>-aware paths under ``iter_<N>/children/``.

        Note: only walks direct InferencerBase attrs, dict values, and list
        elements (the patterns ``_for_each_child_inferencer`` covers).
        Inferencers nested inside arbitrary dict structures (e.g. MFDual's
        ``flow_configs`` list-of-dicts) are not reached by this walker;
        those orchestrators are responsible for assigning their own children.
        """
        seen_ids: set = set()
        skip = self._workspace_propagation_skip

        def _on_instance(child, field_name, key):
            if field_name in skip:
                return  # subclass manages this attr's workspace itself
            if not isinstance(child, InferencerBase):
                return  # duck-typed callables don't have _workspace
            if getattr(child, "_workspace", None) is not None:
                return  # respect explicit pre-assignment
            if id(child) in seen_ids:
                return  # dedup if same instance is in multiple slots
            seen_ids.add(id(child))
            child_name = field_name if key is None else f"{field_name}_{key}"
            try:
                child_ws = parent_workspace.child(child_name)
                # Critical: create the on-disk dirs before assigning. Otherwise
                # `_configure_for_workspace` will set cache_folder to a
                # non-existent path, and effective_cwd may resolve to a
                # ClaudeCodeCli) will fail with NotADirectoryError when it
                # tries to launch with cwd=workspace.root. Mirrors BTA's
                # explicit `worker_ws.ensure_dirs()` before assignment.
                child_ws.ensure_dirs()
                child._workspace = child_ws
            except Exception as exc:  # noqa: BLE001 — best-effort propagation
                _logger.debug(
                    "Workspace propagation to %s[%r]=%s skipped: %s",
                    field_name, key, type(child).__name__, exc,
                )

        def _on_partial(partial, field_name, key):
            return None  # factories assigned at runtime; don't mutate

        self._for_each_child_inferencer(_on_instance, _on_partial)

    def _configure_for_workspace(self, workspace):
        """Auto-configure infrastructure when a workspace is assigned.

        Called by the ``_workspace`` setter. Sets ``cache_folder`` and
        resolves deferred loggers. Subprocess cwd is no longer stored —
        it is derived at call time via ``effective_cwd`` (target_path >
        workspace.root > os.getcwd()).
        """
        import os

        if hasattr(self, "cache_folder"):
            self.cache_folder = os.path.join(
                str(workspace.root), "_runtime", "inferencer_cache"
            )

        logger_val = getattr(self, "logger", None)
        if isinstance(logger_val, str) and logger_val == "auto":
            self._normalize_loggers()
        elif getattr(self, "_logger_awaiting_workspace", False):
            self._logger_awaiting_workspace = False
            self._add_workspace_logger(workspace)
        elif isinstance(logger_val, dict):
            self._redirect_loggers_to_workspace(workspace)

    def _redirect_loggers_to_workspace(self, workspace):
        """Redirect dict-keyed loggers to the workspace's logs/ directory."""
        import os
        from rich_python_utils.io_utils.json_io import JsonLogger

        child_log_dir = workspace.logs_dir
        os.makedirs(child_log_dir, exist_ok=True)
        child_log_path = os.path.join(child_log_dir, "session.jsonl")
        new_loggers = {}
        for name, entry in self.logger.items():
            if isinstance(entry, tuple) and len(entry) == 2:
                logger_inst, config = entry
            else:
                logger_inst, config = entry, None
            if isinstance(logger_inst, JsonLogger):
                new_kwargs = dict(logger_inst.keywords)
                new_kwargs["file_path"] = child_log_path
                new_logger = JsonLogger(**new_kwargs)
                new_loggers[name] = (new_logger, config) if config is not None else new_logger
            else:
                new_loggers[name] = entry
        if new_loggers:
            self.logger = new_loggers

    def __attrs_post_init__(self):
        if self.logger is None:
            self.logger = "auto"

        if self.source_path is None:
            self.source_path = type(self)._detect_source_root()

        # Convenience shorthand: allow `workspace="<path>"` as a synonym for
        # `workspace=InferencerWorkspace(root="<path>")` (with default flags).
        # Lets simple YAML/Python sites stay terse:
        #     LinearWorkflowInferencer(workspace="/tmp/foo")
        #     # is equivalent to:
        #     LinearWorkflowInferencer(workspace=InferencerWorkspace(root="/tmp/foo"))
        # Use the explicit form when you need to set workspace flags
        # (e.g. ``use_final_deliverables_folder=True``).
        if isinstance(self.workspace, str):
            from agent_foundation.common.inferencers.inferencer_workspace import (
                InferencerWorkspace,
            )
            self.workspace = InferencerWorkspace(root=self.workspace)

        # Use property setter — triggers _configure_for_workspace if workspace is not None
        self._workspace = self.workspace

        # Resolve built-in response_post_processor by name.
        # String: "extract_delimited" → use with default args.
        # Dict:  {extract_delimited: {open_tag: "<Output>"}} → use with custom args.
        # Callable: use as-is (backward compat).
        if isinstance(self.response_post_processor, (str, dict)):
            _builtin_post_processors = {
                "extract_delimited": lambda: __import__(
                    "agent_foundation.common.response_parsers.delimiter_parser",
                    fromlist=["extract_delimited"],
                ).extract_delimited,
            }
            if isinstance(self.response_post_processor, str):
                name, kwargs = self.response_post_processor, {}
            else:
                # Dict with single key: {name: {arg: val, ...}}
                name = next(iter(self.response_post_processor))
                kwargs = self.response_post_processor[name] or {}
            factory = _builtin_post_processors.get(name)
            if factory:
                func = factory()
                import functools as _ft
                self.response_post_processor = (
                    _ft.partial(func, **kwargs) if kwargs else func
                )
            else:
                raise ValueError(
                    f"Unknown built-in response_post_processor: {name!r}. "
                    f"Available: {sorted(_builtin_post_processors)}"
                )

        if isinstance(self.post_response_merger, str):
            if self.post_response_merger == "default":
                from rich_python_utils.mp_utils.common import merge_results

                self.post_response_merger = merge_results
            else:
                from rich_python_utils.mp_utils.common import get_merger

                self.post_response_merger = get_merger(self.post_response_merger)

        # Paired-override warning: detect subclass overriding only one of
        # _ainfer_recovery / _infer_recovery (not both).
        async_overridden = type(self)._ainfer_recovery is not InferencerBase._ainfer_recovery
        sync_overridden = type(self)._infer_recovery is not InferencerBase._infer_recovery
        if async_overridden != sync_overridden:
            cls_name = type(self).__name__
            if cls_name not in InferencerBase._paired_override_warned:
                InferencerBase._paired_override_warned.add(cls_name)
                _logger.warning(
                    f"{cls_name} overrides only "
                    f"{'_ainfer_recovery' if async_overridden else '_infer_recovery'} "
                    f"but not {'_infer_recovery' if async_overridden else '_ainfer_recovery'}. "
                    f"Override both for full sync/async recovery coverage."
                )

        # Nested-retry warning: detect fallback_inferencer with max_retry > 1.
        if self.fallback_inferencer is not None:
            fb_list = (
                self.fallback_inferencer
                if isinstance(self.fallback_inferencer, list)
                else [self.fallback_inferencer]
            )
            cls_name = type(self).__name__
            for fb in fb_list:
                if fb.max_retry > 1 and cls_name not in InferencerBase._nested_retry_warned:
                    InferencerBase._nested_retry_warned.add(cls_name)
                    _logger.warning(
                        f"{cls_name}: fallback_inferencer {type(fb).__name__} has "
                        f"max_retry={fb.max_retry}. This creates multiplicative retries "
                        f"(outer × inner). Consider max_retry=1 for fallbacks."
                    )
                    break

        super().__attrs_post_init__()

    def _for_each_child_inferencer(self, on_instance, on_partial):
        """Walk attrs fields and invoke callbacks for child inferencers.

        Finds child inferencers in five value patterns:
        1. Direct InferencerBase attr
        2. functools.partial (e.g., worker factory)
        3. dict containing InferencerBase, partial, or duck-typed callables
        4. list/tuple containing InferencerBase or duck-typed callables
        5. Duck-typed callable with ``template_extra_feed`` dict attribute

        Args:
            on_instance: Called for each InferencerBase or duck-typed callable.
                Signature: ``(child, field_name, key)`` where key is None for
                direct attrs, str for dict keys, int for list indices.
            on_partial: Called for each functools.partial.
                Signature: ``(partial, field_name, key) -> Optional[partial]``.
                Return a new partial to replace the original, or None to keep it.
        """
        import attr
        import functools

        attr_name = TEMPLATE_EXTRA_FEED_ATTR

        for field in attr.fields(type(self)):
            try:
                value = getattr(self, field.name)
            except AttributeError:
                continue
            if value is None:
                continue

            if isinstance(value, InferencerBase):
                on_instance(value, field.name, None)

            elif isinstance(value, functools.partial):
                new = on_partial(value, field.name, None)
                if new is not None:
                    setattr(self, field.name, new)

            elif isinstance(value, dict):
                new_dict = dict(value)
                changed = False
                for k, v in value.items():
                    if isinstance(v, InferencerBase):
                        on_instance(v, field.name, k)
                    elif isinstance(v, functools.partial):
                        new = on_partial(v, field.name, k)
                        if new is not None:
                            new_dict[k] = new
                            changed = True
                    elif callable(v) and isinstance(
                        getattr(v, attr_name, None), dict
                    ):
                        on_instance(v, field.name, k)
                if changed:
                    setattr(self, field.name, new_dict)

            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, InferencerBase):
                        on_instance(item, field.name, i)
                    elif callable(item) and isinstance(
                        getattr(item, attr_name, None), dict
                    ):
                        on_instance(item, field.name, i)

            elif callable(value) and isinstance(
                getattr(value, attr_name, None), dict
            ):
                on_instance(value, field.name, None)

    def _propagate_to_children(self):
        """Default: no-op. Overridden in TemplatedInferencerBase to propagate
        ``template_extra_feed`` to child inferencers via
        ``_for_each_child_inferencer``.

        Called unconditionally from ``_infer_single`` / ``_ainfer_single``;
        the no-op default is needed for orchestrators (Dual, BTA, LWI, etc.)
        that inherit from InferencerBase but have no template state.
        """
        return None

    @property
    def secret_key(self) -> str:
        return resolve_environ(self._secret_key)

    # ------------------------------------------------------------------
    # Generic parent→child iteration + recursive pre_retry hook
    #
    # `_iter_child_inferencers` is the single canonical iteration mechanism
    # for any parent→child operation: aconnect / adisconnect / lifecycle
    # resets / pre_retry propagation. Subclasses override to declare their
    # children. Default: empty.
    #
    # Relationship to `_for_each_child_inferencer` (above): that walker
    # auto-discovers children via attrs.fields and is designed for
    # `template_extra_feed` propagation (with side-effect callbacks for
    # mutating partials and duck-typed callables). The two primitives
    # serve different lifetimes and have incompatible signatures —
    # deliberately not unified.
    # ------------------------------------------------------------------

    def _iter_child_inferencers(self) -> Iterator["InferencerBase"]:
        """Yield this inferencer's direct child inferencers (one level only).

        Override in subclasses to enumerate children. Default: empty iterator.

        Contract:
          - Yield direct children only. Recursive walks (e.g., `pre_retry`)
            invoke each child's own `_iter_child_inferencers` for transitive
            coverage.
          - Subclasses SHOULD dedup their own yields when the same instance
            may appear in multiple slots (e.g., fixer aliased to base).
          - Consumers performing recursive walks should additionally dedup
            across the whole tree using id-based seen sets (cycle safety).

        Used by lifecycle methods (``aconnect`` / ``adisconnect`` /
        ``_areset_sub_inferencers``) and the ``pre_retry`` hook below.
        """
        return iter(())

    def _collect_all_descendant_inferencers(self, _seen=None):
        """Recursively yield self and all descendant inferencers (DFS).

        Cycle-safe via id-based seen set. Used by
        :meth:`BreakdownThenAggregateInferencer._validate_worker_isolation`
        to detect shared sub-inferencer instances across workers.

        Yields:
            Each unique InferencerBase instance reachable from self
            (including self), in depth-first order.
        """
        if _seen is None:
            _seen = set()
        if id(self) in _seen:
            return
        _seen.add(id(self))
        yield self
        for child in self._iter_child_inferencers():
            yield from child._collect_all_descendant_inferencers(_seen=_seen)

    async def pre_retry(
        self,
        attempt: int,
        exception: BaseException,
        _seen: Optional[set] = None,
    ) -> None:
        """Public retry-cleanup hook called by the retry helper between attempts.

        Calls ``_pre_retry`` on self (subclass-supplied own-state cleanup),
        then propagates recursively to every direct child yielded by
        ``_iter_child_inferencers``. Cycles broken by id-based seen set.
        Child failures are caught and logged at WARNING — cleanup is
        best-effort; one bad sibling must not block the rest.

        Override ``_pre_retry`` to clean up your own state. Override
        ``_iter_child_inferencers`` to declare your children (also reused
        by aconnect/adisconnect). Do NOT normally override this method
        itself — the propagation contract is fixed.

        The ``_seen`` parameter is an internal implementation detail for
        cycle safety; top-level callers omit it.
        """
        if _seen is None:
            _seen = set()
        if id(self) in _seen:
            return
        _seen.add(id(self))
        try:
            await self._pre_retry(attempt, exception)
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            _logger.warning(
                "%s._pre_retry raised: %s", type(self).__name__, exc
            )
        for child in self._iter_child_inferencers():
            if child is None or id(child) in _seen:
                continue
            try:
                await child.pre_retry(attempt, exception, _seen=_seen)
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                _logger.warning(
                    "%s.pre_retry (child of %s) raised: %s",
                    type(child).__name__, type(self).__name__, exc,
                )

    async def _pre_retry(self, attempt: int, exception: BaseException) -> None:
        """Subclass override hook for own-state cleanup before next retry.

        Default is a no-op. Subclasses with transient per-attempt state
        (sessions, cached connections, in-memory buffers) override to reset.

        Note: only fires on the ASYNC retry path. Sync inferencers in this
        codebase are rare; sync `pre_retry` propagation is not currently
        wired (sync `_internal_retry_callback` stays as-is).
        """
        pass

    @abstractmethod
    def _infer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        """
        Performs the core inference logic.

        This abstract method should be implemented by subclasses to perform inference based on the provided input and additional arguments.

        Args:
            inference_input (Any): The input data for inference.
            **_inference_args: Additional keyword arguments for customizing the inference process.

        Returns:
            Any: The response of the inference. The exact type and format depend on the specific implementation.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    # -- Auto-logger resolution -------------------------------------------

    def _resolve_auto_logger(self):
        """Resolve ``logger='auto'`` to a JsonLogger at workspace.logs_dir.

        Called by ``Debuggable._normalize_loggers()`` when ``logger='auto'``.
        If no workspace is available yet, falls back to Debuggable's builtin
        console logger and sets a flag so ``_configure_for_workspace`` can
        upgrade to JsonLogger when workspace becomes available.
        """
        if self._workspace is None:
            super()._resolve_auto_logger()
            self._logger_awaiting_workspace = True
            return
        self._logger_awaiting_workspace = False
        self._add_workspace_logger(self._workspace)

    def _add_workspace_logger(self, workspace):
        """Create a JsonLogger at workspace.logs_dir and add it to self.logger."""
        from rich_python_utils.io_utils.json_io import JsonLogger, SpaceExtMode
        from rich_python_utils.common_objects.debuggable import LoggerConfig
        log_dir = workspace.logs_dir
        os.makedirs(log_dir, exist_ok=True)
        json_entry = (JsonLogger(
            file_path=os.path.join(log_dir, "session.jsonl"),
            append=True,
            is_artifact=True,
            parts_min_size=0,
            space_ext_mode=SpaceExtMode.MOVE,
        ), LoggerConfig(pass_item_key_as="parts_key_path_root"))
        if isinstance(self.logger, dict):
            self.logger["_workspace"] = json_entry[0]
            if not hasattr(self, "_resolved_logger_configs"):
                self._resolved_logger_configs = {}
            self._resolved_logger_configs["_workspace"] = json_entry[1]
        else:
            self.logger = [json_entry]

    # -- Template rendering & output finalization -------------------------
    #
    # `_render_prompt`, `_build_template_feed`, `_propagate_to_children`,
    # and `supports_prompt_rendering` are template-rendering machinery; their
    # real implementations live on TemplatedInferencerBase. The stubs below
    # exist so `_ainfer_single` / `_infer_single` can call them
    # unconditionally on every inferencer (including orchestrators like Dual,
    # BTA, LWI that inherit InferencerBase directly without templates).

    @property
    def supports_prompt_rendering(self) -> bool:
        """Default: False. Overridden in TemplatedInferencerBase."""
        return False

    def _render_prompt(self, inference_input: Any) -> Any:
        """Default: pass through unchanged. Overridden in TemplatedInferencerBase
        to render via ``template_manager``.

        Called unconditionally from ``_ainfer_single`` / ``_infer_single``.
        Orchestrators (Dual, BTA, LWI) inherit this no-op behavior and pass
        their ``inference_input`` straight through to their children.
        """
        return inference_input

    def _finalize_output(self, response: Any) -> Any:
        """Finalize outputs after inference: promote deliverables, write summary.

        Principle: everything the agent wrote to ``outputs/`` IS the
        deliverable.  The framework only writes ``output_path`` (the
        ``<Response>``-extracted summary) when the agent didn't.

        Sequence:
        1. If ``output_is_deliverable``: MOVE all agent-written content
           from ``outputs/`` to ``outputs/final_deliverables/``.
        2. Check ``output_path``: if the agent already wrote it (now in
           ``final_deliverables/``), done.  If not, extract ``<Response>``
           and write to ``outputs/output_path`` as a summary reference
           (this summary stays in ``outputs/``, not in deliverables).
        3. Emit manifest if applicable.
        """
        import shutil as _shutil

        ws = self._workspace
        resolved = self.resolve_output_path()

        # -- Step 1: Move agent-written outputs to final_deliverables/ --
        agent_wrote_output_path = False
        if ws is not None and self.output_is_deliverable:
            fd = getattr(ws, "deliverables_dir", None)
            if fd is not None:
                outputs_dir = ws.outputs_dir
                if os.path.isdir(outputs_dir):
                    fd_basename = os.path.basename(fd)
                    entries = [e for e in os.listdir(outputs_dir)
                               if e != fd_basename]
                    if entries:
                        os.makedirs(fd, exist_ok=True)
                        for entry in entries:
                            src = os.path.join(outputs_dir, entry)
                            dst = os.path.join(fd, entry)
                            if not os.path.exists(dst):
                                _shutil.move(src, dst)
                        # output_path was moved with everything else
                        if resolved:
                            moved_path = os.path.join(
                                fd, os.path.basename(resolved))
                            if os.path.isfile(moved_path):
                                agent_wrote_output_path = True

        # -- Step 2: Write <Response> summary if agent didn't write output_path --
        if resolved and os.path.isabs(resolved) and not agent_wrote_output_path:
            if os.path.isfile(resolved) and os.path.getsize(resolved) > 0:
                pass
            else:
                from agent_foundation.common.response_parsers import (
                    extract_delimited,
                )
                cleaned = extract_delimited(str(response))
                os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
                with open(resolved, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                response = cleaned

        # -- Step 3: Emit manifest --
        manifest_source = resolved
        if ws is not None and self.output_is_deliverable:
            fd = getattr(ws, "deliverables_dir", None)
            if fd and resolved:
                candidate = os.path.join(fd, os.path.basename(resolved))
                if os.path.isfile(candidate):
                    manifest_source = candidate
        if manifest_source and os.path.isfile(manifest_source):
            if self.output_manifest_index or self.output_is_deliverable:
                self._emit_output_manifest(manifest_source)

        return response

    _output_finalized: bool = False

    def _complete_inference(self, response: Any = None, *, force: bool = False) -> Any:
        """Centralized output-finalization hook.

        Calls ``_finalize_output(response)``.
        Idempotent — safe to call multiple times; first call acts,
        subsequent are no-ops.
        """
        if self._output_finalized and not force:
            return response
        if response is not None:
            response = self._finalize_output(response)
        self._output_finalized = True
        return response

    # -- Orchestrator symlink helpers ------------------------------------

    @staticmethod
    def _symlink_or_copy(src: str, dst: str) -> None:
        """Create a symlink from *dst* → *src*, falling back to copy on
        platforms that don't support symlinks (Windows without developer mode).
        """
        import shutil as _shutil
        if os.path.lexists(dst):
            if os.path.islink(dst) and not os.path.exists(dst):
                os.unlink(dst)
            else:
                return
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        try:
            os.symlink(os.path.abspath(src), dst,
                       target_is_directory=os.path.isdir(src))
        except (OSError, NotImplementedError):
            if os.path.isdir(src):
                _shutil.copytree(src, dst)
            else:
                _shutil.copy2(src, dst)

    def _symlink_child_output(self, child_workspace, child_output_name=None) -> None:
        """Symlink a canonical child's output and deliverables as own.

        Used by orchestrator ``_finalize_output`` overrides to surface
        the canonical child's work as the orchestrator's own output.

        For outputs: finds the child's file using ``child_output_name``
        (the filename the child wrote), then symlinks it into the
        orchestrator's own ``outputs/`` under ``self.output_path``
        (the filename the orchestrator declares).

        When ``child_output_name`` is ``None``, assumes the child uses
        the same filename as the orchestrator (``self.output_path``).

        For deliverables: symlinks each entry in the child's
        ``final_deliverables/`` into the orchestrator's own
        ``final_deliverables/``.
        """
        ws = self._workspace
        if ws is None or child_workspace is None:
            return

        from agent_foundation.common.inferencers.inferencer_workspace import DEFAULT_OUTPUT_FILENAME
        own_name = self.output_path or DEFAULT_OUTPUT_FILENAME
        child_name = child_output_name or own_name

        # Symlink the output file (check deliverables first — leaf may have moved it)
        child_deliv = getattr(child_workspace, "deliverable_path", lambda x: None)(child_name)
        child_output = child_workspace.output_path(child_name) if hasattr(child_workspace, "output_path") else None
        src = child_deliv if (child_deliv and os.path.isfile(child_deliv)) else child_output
        if src and os.path.isfile(src):
            own_output = ws.output_path(own_name)
            self._symlink_or_copy(src, own_output)

        # Symlink deliverable entries
        child_fd = getattr(child_workspace, "deliverables_dir", None)
        own_fd = getattr(ws, "deliverables_dir", None)
        if child_fd and own_fd and os.path.isdir(child_fd) and os.listdir(child_fd):
            os.makedirs(own_fd, exist_ok=True)
            for entry in os.listdir(child_fd):
                self._symlink_or_copy(
                    os.path.join(child_fd, entry),
                    os.path.join(own_fd, entry),
                )

    def _emit_output_manifest(self, output_path: str) -> None:
        """Walk workspace logs/session and emit output_manifest.json."""
        import json as _json

        ws = self._workspace
        contributors = []

        logs_dir = getattr(ws, "logs_dir", None)
        if logs_dir:
            session_dir = os.path.join(logs_dir, "session")
            if os.path.isdir(session_dir):
                for entry in sorted(os.listdir(session_dir)):
                    entry_path = os.path.join(session_dir, entry)
                    if entry.endswith(".jsonl.parts") and os.path.isdir(entry_path):
                        for cat_name in sorted(os.listdir(entry_path)):
                            cat_path = os.path.join(entry_path, cat_name)
                            if not os.path.isdir(cat_path):
                                continue
                            for fname in sorted(os.listdir(cat_path)):
                                fpath = os.path.join(cat_path, fname)
                                if os.path.isfile(fpath):
                                    contributors.append({
                                        "category": cat_name.lower(),
                                        "path": os.path.realpath(fpath),
                                        "size_bytes": os.path.getsize(fpath),
                                    })
                    elif entry.endswith(".jsonl") and os.path.isfile(entry_path):
                        contributors.append({
                            "category": "session_log",
                            "path": os.path.realpath(entry_path),
                            "size_bytes": os.path.getsize(entry_path),
                        })

        for src_info in getattr(self, "_consumed_upstream_paths", []):
            p = str(src_info)
            if os.path.isfile(p):
                contributors.append({
                    "category": "upstream_artifact",
                    "path": os.path.realpath(p),
                })

        manifest = {
            "schema_version": "1.0",
            "output": {
                "path": os.path.realpath(output_path),
                "size_bytes": os.path.getsize(output_path),
                "produced_by": type(self).__name__,
                "workspace_root": os.path.realpath(ws.root),
            },
            "contributors": contributors,
            "stats": {"total": len(contributors)},
        }

        manifest_path = os.path.splitext(output_path)[0] + "_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(_json.dumps(manifest, indent=2))

    # -- Resumable protocol implementation ----------------------------------

    def _get_result_path(self, result_id, *args, **kwargs) -> str:
        """Default checkpoint path using ``output_path`` as base directory.

        Raises ``NotImplementedError`` when ``output_path`` is not configured,
        which is the safe default — workflow callers
        (``WorkGraphNode._should_save_result``) already catch this and skip
        checkpointing gracefully.
        """
        if self.output_path is None:
            raise NotImplementedError(
                f"{type(self).__name__}: no output_path configured. "
                f"Set output_path to enable checkpointing."
            )
        resolved = self.resolve_output_path() or self.output_path
        return os.path.join(resolved, f"{result_id}.pkl")

    def _try_resume_from_cache(self, inference_input, inference_config=None, **kwargs):
        """Sync hook for subclasses to check for resumable cached results.

        Called AFTER preprocessing and template rendering, BEFORE the retry
        loop in ``_infer_single``.  Returns the cached result to short-circuit
        execution, or ``None`` to proceed normally.

        Base implementation returns ``None`` (no caching for plain inferencers).
        ``StreamingInferencerBase`` overrides this with cache discovery logic.
        """
        return None

    async def _atry_resume_from_cache(self, inference_input, inference_config=None, **kwargs):
        """Async hook — mirrors ``_try_resume_from_cache`` for ``_ainfer_single``.

        Base implementation delegates to the sync version.  Streaming inferencers
        override to ``await self._ainfer(augmented)`` for partial-cache recovery.
        """
        return self._try_resume_from_cache(inference_input, inference_config, **kwargs)

    # -- Inference pipeline -------------------------------------------------

    def _infer_single(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        # ── Phase 1 (leaf-owned template rendering): extract per-call render
        # parameters from _inference_args BEFORE forwarding to _infer().
        # `extra_feed` is consumed by _render_prompt; `render_only` short-circuits
        # the LLM call. Both are KEYWORD-ONLY by convention here (orchestrators
        # pass them by name). Extracting them prevents leakage to _infer() which
        # would TypeError on unrecognized kwargs.
        _extra_feed = _inference_args.pop("extra_feed", None)
        _render_only = _inference_args.pop("render_only", False)
        return self.__infer_single_impl(
            inference_input,
            inference_config,
            _extra_feed=_extra_feed,
            _render_only=_render_only,
            **_inference_args,
        )

    def __infer_single_impl(
        self,
        inference_input: Any,
        inference_config: Any = None,
        *,
        _extra_feed: Optional[dict] = None,
        _render_only: bool = False,
        **_inference_args,
    ):
        """
        Process a single inference input with preprocessing, inference, and post-processing.

        This method handles the complete inference pipeline for a single input:
        1. Preprocesses input via input_preprocessor if provided
        2. Merges default_inference_args with provided _inference_args
        3. Executes _infer() with retry logic via execute_with_retry()
        4. Post-processes the response via response_post_processor if provided
        5. Returns the post-processed inference response

        Args:
            inference_input: Input data for inference (will be preprocessed if input_preprocessor is set)
            inference_config: Optional configuration for the inference run
            **_inference_args: Additional keyword arguments merged with default_inference_args
                and passed to _infer()

        Returns:
            The post-processed inference response. Type depends on the _infer() implementation
            and response_post_processor callable.

        Raises:
            Exception: If all retry attempts fail and default_return_or_raise is set to
                None or an Exception object.
        """
        self._propagate_to_children()

        # Capture original input BEFORE preprocessing for retry_with_original mode
        original_input = inference_input

        if self.input_preprocessor is not None:
            inference_input = self.input_preprocessor(inference_input)

        # Template rendering (opt-in: only when template_manager is set).
        # Round-7 invariant: conditional kwarg pass to avoid TypeError on
        # subclass overrides that don't declare extra_feed (e.g.
        # ConversationalInferencer._render_prompt). When _extra_feed is
        # None, this call is byte-identical to the legacy form.
        if _extra_feed is not None:
            inference_input = self._render_prompt(inference_input, extra_feed=_extra_feed)
        else:
            inference_input = self._render_prompt(inference_input)

        # Phase 1: render_only mode — used by orchestrators that need the
        # rendered prompt for logging/cache-keying without invoking the LLM.
        # See Q14 / ConsensusIterationRecord.review_input use case.
        if _render_only:
            return inference_input

        # Resume hook: check for cached result from a previous session.
        # Placed AFTER preprocessing/rendering so prompt hash matches cache.
        resume_result = self._try_resume_from_cache(
            inference_input, inference_config, **_inference_args
        )
        if resume_result is not None:
            # Run the same post-processing tail as the normal path
            resume_result = self._finalize_output(resume_result)
            if self.state_graphs:
                self.update_state_graphs(resume_result)
            if self.response_post_processor is not None:
                resume_result = self.response_post_processor(self._normalize_for_post_processor(resume_result))
            return resume_result

        inference_args = self.default_inference_args.copy()
        if _inference_args:
            inference_args.update(_inference_args)

        # Augment with state graph args
        if self.state_graphs:
            inference_args.update(self.get_inference_args_from_state_graphs())

        # Pop runtime overrides that should not be forwarded to _infer()
        on_retry_callback = inference_args.pop("on_retry_callback", None)
        total_timeout = inference_args.pop(
            "total_timeout_seconds", self.total_timeout_seconds
        )
        attempt_timeout = inference_args.pop(
            "attempt_timeout_seconds", self.attempt_timeout_seconds
        )
        fallback_mode = inference_args.pop("fallback_mode", self.fallback_mode)
        on_fallback_callback = inference_args.pop("on_fallback_callback", None)
        retry_prompt_mode = inference_args.pop("retry_prompt_mode", "original")

        # Per-attempt timeout is async-only — reject in sync path
        if attempt_timeout and attempt_timeout > 0:
            raise NotImplementedError(
                "Per-attempt timeout is async-only. Use total_timeout_seconds "
                "for sync, or call ainfer() instead."
            )

        # Validate retry_prompt_mode
        if retry_prompt_mode not in RETRY_PROMPT_MODES:
            raise ValueError(
                f"Invalid retry_prompt_mode={retry_prompt_mode!r}. "
                f"Must be one of {RETRY_PROMPT_MODES}"
            )

        # Mutable args list — prompt can be swapped by the retry callback
        retry_args = [inference_input]

        # Build internal retry callback (handles prompt transformation + user callback)
        _user_callback = on_retry_callback
        if _user_callback is not None or retry_prompt_mode != "original":

            def _internal_retry_callback(attempt, exception):
                # Forward to user callback with local inference_args
                if _user_callback is not None:
                    _user_callback(attempt, exception, inference_args)
                # Transform prompt based on retry_prompt_mode
                if retry_prompt_mode == "simple_retry":
                    retry_args[0] = _SIMPLE_RETRY_PROMPT
                elif retry_prompt_mode == "retry_with_original":
                    retry_args[0] = (
                        _SIMPLE_RETRY_PROMPT + " The task was:\n" + str(original_input)
                    )
                if retry_prompt_mode != "original":
                    self.log_info(
                        f"Retry prompt ({retry_prompt_mode}): {str(retry_args[0])[:200]}",
                        "RetryPrompt",
                    )

            on_retry_callback = _internal_retry_callback

        self.log_info(inference_input, "InferenceInput")
        self.log_info(inference_args, "InferenceArgs")

        # Convert 0 → None for timeout parameters (0 = disabled)
        effective_total_timeout = total_timeout or None

        # -- Build _fallback_state and fallback chain --
        _fallback_state = {"last_exception": None, "partial_output": None, "cache_path": None}

        # Recovery wrapper — reads from closure-captured _fallback_state
        def _recovery_wrapper(inp, **kw):
            return self._infer_recovery(
                inp,
                last_exception=_fallback_state["last_exception"],
                last_partial_output=_fallback_state["partial_output"],
                inference_config=inference_config,
                **inference_args,
            )

        # External fallback wrappers from fallback_inferencer list
        external_wrappers = []
        if self.fallback_inferencer is not None:
            fb_list = (
                self.fallback_inferencer
                if isinstance(self.fallback_inferencer, list)
                else [self.fallback_inferencer]
            )
            external_wrappers = [
                lambda inp, inf=inf, **kw: inf.infer(inp, inference_config, **kw)
                for inf in fb_list
            ]

        # Build fallback chain and mode for the retry helper
        if fallback_mode == FallbackMode.NEVER:
            effective_fallback_func = None
            effective_fallback_mode = FallbackMode.NEVER
        else:
            effective_fallback_func = [_recovery_wrapper] + external_wrappers
            effective_fallback_mode = fallback_mode

        # Transition callback — populates _fallback_state and resets retry_args[0]
        _user_on_fallback = on_fallback_callback

        def _on_transition(from_func, to_func, exception, total_attempts):
            _fallback_state["last_exception"] = exception
            if _fallback_state["cache_path"]:
                try:
                    with open(_fallback_state["cache_path"], "r", encoding="utf-8") as f:
                        raw = f.read()
                    _fallback_state["partial_output"] = raw if raw.strip() else None
                except OSError:
                    _fallback_state["partial_output"] = None
            # Reset retry_args[0] to original input so external fallback
            # inferencers see the original prompt, not the mutated retry prompt
            retry_args[0] = original_input
            # Forward to user-provided on_fallback_callback if present
            if _user_on_fallback is not None:
                _user_on_fallback(from_func, to_func, exception, total_attempts)

        # Set ContextVar for this call (per-thread safe for sync path)
        token = _current_fallback_state.set(_fallback_state)
        try:
            inference_response = execute_with_retry(
                func=partial(self._infer, inference_config=inference_config),
                max_retry=self.max_retry,
                min_retry_wait=self.min_retry_wait,
                max_retry_wait=self.max_retry_wait,
                args=retry_args,
                kwargs=inference_args,
                default_return_or_raise=self.default_return_or_raise,
                on_retry_callback=on_retry_callback,
                total_timeout=effective_total_timeout,
                fallback_func=effective_fallback_func,
                fallback_mode=effective_fallback_mode,
                on_fallback_callback=_on_transition if effective_fallback_func else None,
            )
        except TimeoutError:
            self.log_info(
                f"Total timeout after {total_timeout}s",
                "TotalTimeout",
            )
            raise
        finally:
            _current_fallback_state.reset(token)

        self.log_debug(inference_response, "InferenceResponse")

        # Output finalization (promote deliverables, write summary)
        inference_response = self._finalize_output(inference_response)

        # Update state graphs from response
        if self.state_graphs:
            self.update_state_graphs(inference_response)

        if self.response_post_processor is not None:
            post_input = self._normalize_for_post_processor(inference_response)
            processed_response = self.response_post_processor(post_input)
            self.log_debug(processed_response, "PostProcessedResponse")
            return processed_response

        return inference_response

    def _normalize_for_post_processor(self, response: Any) -> str:
        """Extract text from a structured response for post-processing.

        Post-processors (e.g., extract_delimited) expect str input.
        Structured response types (e.g., TerminalInferencerResponse from RovoDevCLI)
        carry clean text in their .output attribute.
        Falls back to str() for unknown types rather than raising, to avoid breaking
        existing inferencers that pass custom objects through.
        """
        if isinstance(response, str):
            return response
        if hasattr(response, 'output') and isinstance(response.output, str):
            return response.output or ""
        raise TypeError(
            f"response_post_processor expects str or an object with a str .output attribute, "
            f"got {type(response).__name__}. Add a .output property or handle this type explicitly."
        )

    # -- State graph integration -------------------------------------------

    def attach_state_graph(self, tracker) -> None:
        """Attach a StateGraphTracker to this inferencer."""
        if self.state_graphs is None:
            self.state_graphs = []
        self.state_graphs.append(tracker)

    def get_inference_args_from_state_graphs(self) -> dict:
        """Collect inference args from all attached state graphs.
        Handles None check and iterates over trackers.
        """
        if not self.state_graphs:
            return {}
        merged = {}
        for tracker in self.state_graphs:
            merged.update(self._get_inference_args_from_state_graph(tracker))
        return merged

    def update_state_graphs(self, response) -> None:
        """Update all attached state graphs from inference response.
        Handles None check and iterates over trackers.
        """
        if not self.state_graphs:
            return
        for tracker in self.state_graphs:
            self._update_state_graph(tracker, response)

    def _get_inference_args_from_state_graph(self, tracker) -> dict:
        """Override in subclasses: extract inference args from one tracker.
        Default: empty dict (no-op).
        """
        return {}

    def _update_state_graph(self, tracker, response) -> None:
        """Override in subclasses: update one tracker from response.
        Default: no-op.
        """
        pass

    # -- Workspace output path resolution --

    def resolve_output_path(
        self, runtime_override: Optional[str] = None
    ) -> Optional[str]:
        """Resolve the effective output path.

        Priority: *runtime_override* > ``self.output_path``.

        Resolution:
        - If the path is relative and ``_workspace`` is set (flow inferencers),
          resolve to ``workspace.outputs_dir / path``.
        - If the path is absolute, return as-is.
        - If no workspace, return the path unchanged.

        No side effects (no directory creation, no file I/O).

        Flow inferencers set ``_workspace`` in ``__attrs_post_init__`` or when
        assigned by a parent.  Simple API inferencers never set it, so
        ``getattr`` returns ``None`` and this method returns the raw path.
        """
        path = runtime_override if runtime_override is not None else self.output_path
        if path is None:
            return None
        ws = getattr(self, "_workspace", None)
        if ws is not None and not os.path.isabs(path):
            return ws.output_path(path)
        return path

    def _infer_iterator(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        """
        Process an iterator of inputs and yield atomized post-processed results.

        Args:
            inference_input: Iterator of input items to process
            inference_config: Optional configuration for the inference run
            **_inference_args: Additional keyword arguments passed to _infer_single()

        Yields:
            Atomized inference results for each input item
        """
        for _inference_input in inference_input:
            response = self._infer_single(
                _inference_input, inference_config, **_inference_args
            )
            yield from iter__(response, atom_types=self.response_types)

    def infer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        """
        Execute inference with automatic iterator detection.

        This method routes inference based on input type:
        - If input is an Iterator and post_response_merger is provided:
          Returns a single merged result after processing all items
        - If input is an Iterator and post_response_merger is not provided:
          Returns an Iterator yielding atomized results
        - Otherwise: Returns a single post-processed result

        Args:
            inference_input: Input data for inference. Can be a single input or an Iterator
            inference_config: Optional configuration for the inference run
            **_inference_args: Additional keyword arguments passed to the inference methods

        Returns:
            If input is Iterator with post_response_merger: Returns a single merged result
            If input is Iterator without post_response_merger: Returns an Iterator yielding atomized results
            Otherwise: Returns a single post-processed inference result

        Raises:
            Exception: If inference fails after all retry attempts
        """
        if isinstance(inference_input, Iterator):
            iterator_result = self._infer_iterator(
                inference_input, inference_config, **_inference_args
            )

            if self.post_response_merger is not None:
                all_responses = list(iterator_result)
                merged_response = self.post_response_merger(all_responses)
                self.log_debug(merged_response, "MergedResponse")
                return merged_response
            else:
                return iterator_result
        else:
            return self._infer_single(
                inference_input, inference_config, **_inference_args
            )

    def iter_infer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        """
        Execute inference and always return an iterator of responses.

        This method wraps infer() to ensure the output is always iterable:
        - If response_types is set: atomizes response using iter__() with response_types
        - If response_types is not set: yields response directly (or from iterator if response is already an iterator)

        Note: If post_response_merger is provided for iterator inputs, the merged response
        will be atomized according to response_types.

        Args:
            inference_input: Input data for inference (single input or iterator)
            inference_config: Optional configuration for the inference run
            **_inference_args: Additional keyword arguments passed to infer()

        Yields:
            Inference responses. Atomization behavior depends on response_types configuration.
        """
        response = self.infer(
            inference_input=inference_input,
            inference_config=inference_config,
            **_inference_args,
        )
        if not self.response_types:
            if isinstance(response, Iterator):
                yield from response
            else:
                yield response
        else:
            yield from iter__(response, atom_types=self.response_types)

    def parallel_infer(
        self,
        inference_inputs: Iterable[Any],
        inference_config: Any = None,
        num_workers: int = None,
        use_threading: bool = True,
        debug: bool = False,
        **_inference_args,
    ) -> list:
        """Process multiple inputs concurrently using thread or process pool.

        Dispatches each input to _infer_single() via parallel_process_by_pool,
        enabling concurrent execution for I/O-bound workloads (API calls, etc.).

        Args:
            inference_inputs: Iterable of inputs to process concurrently.
                Generators are supported (materialized internally).
            inference_config: Optional configuration passed to each _infer_single call.
            num_workers: Number of pool workers. None = auto:
                threading (default): min(len(inputs), 32) for I/O-bound work.
                multiprocessing: get_suggested_num_workers() respecting CPU count.
                Always capped at len(inputs).
            use_threading: True (default) uses ThreadPool — no pickling required.
                False uses multiprocessing.Pool — requires picklable inferencer
                (no lambdas in input_preprocessor, response_post_processor, etc.).
            debug: True runs sequentially in a single process for debugging
                (passed through to parallel_process_by_pool's debug param).
            **_inference_args: Additional keyword arguments merged with
                default_inference_args and passed to _infer_single().

        Returns:
            List of inference results, order-preserving (same index alignment
            as inputs).

        Note:
            post_response_merger is NOT auto-applied. The iterator path in
            infer() atomizes results via iter__() before merging — a different
            input shape. Users can apply their own merging on the returned list.
        """
        try:
            from rich_python_utils.mp_utils.mp_target import MPTarget
            from rich_python_utils.mp_utils.parallel_process import parallel_process_by_pool
        except ImportError as _e:
            raise NotImplementedError(
                "parallel_infer requires rich_python_utils.mp_utils which is not "
                "available in this environment. Install rich_python_utils with the "
                "mp extra, or use sequential infer() calls instead."
            ) from _e

        inference_inputs = list(inference_inputs)
        if not inference_inputs:
            return []

        num_inputs = len(inference_inputs)

        if num_workers is None:
            if use_threading:
                num_workers = min(num_inputs, 32)
            else:
                from rich_python_utils.mp_utils.common import get_suggested_num_workers

                num_workers = get_suggested_num_workers()
        num_workers = min(num_workers, num_inputs)

        pool_class = None
        if use_threading:
            from multiprocessing.pool import ThreadPool

            pool_class = ThreadPool

        worker = partial(
            self._infer_single,
            inference_config=inference_config,
            **_inference_args,
        )
        mp_target = MPTarget(
            worker,
            pass_pid_to_target=False,
            pass_each_data_item_to_target=True,
        )

        self.log_debug(
            f"{num_inputs} inputs, {num_workers} workers, "
            f"{'threading' if use_threading else 'multiprocessing'}",
            "ParallelInfer",
        )

        results = parallel_process_by_pool(
            num_p=num_workers,
            data_iter=inference_inputs,
            target=mp_target,
            pool_object=pool_class,
            merge_output=True,
            mergers=["list"],
            debug=debug,
        )

        return results

    def __call__(self, inference_input: Any, inference_config: Any = None, **kwargs):
        """
        Allows the instance to be called as a function, invoking the `infer` method.

        This method enables the object to be used like a function, passing the input and any additional keyword arguments directly to the `infer` method.

        Args:
            inference_input (Any): The input data to be used for inference.
            inference_config (Any): The configuration for the current inference.
            **kwargs: Additional keyword arguments to be passed to the `infer` method.

        Returns:
            Any: The response of the inference, as returned by the `infer` method.
        """
        return self.infer(inference_input, inference_config=inference_config, **kwargs)

    # region Async Methods

    async def _ainfer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        """Async version of _infer().

        Default implementation wraps sync _infer() for backwards compatibility.
        Async-native subclasses should override this method directly.

        Args:
            inference_input: Input data for inference.
            inference_config: Optional configuration for the inference run.
            **_inference_args: Additional keyword arguments for inference.

        Returns:
            The inference response.
        """
        return self._infer(inference_input, inference_config, **_inference_args)

    def _infer_recovery(
        self,
        inference_input: Any,
        last_exception: Optional[Exception],
        last_partial_output: Optional[str],
        inference_config: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """Overridable sync recovery method. Default delegates to _infer.

        Called as the fallback function when the primary _infer fails.
        Subclasses (e.g., StreamingInferencerBase) can override to implement
        cache-aware or session-aware recovery strategies.

        Args:
            inference_input: The original inference input.
            last_exception: The exception from the failed primary attempt.
            last_partial_output: Any partial output from the failed attempt, or None.
            inference_config: Optional inference configuration.
            **kwargs: Additional keyword arguments passed through to _infer.
        """
        return self._infer(inference_input, inference_config, **kwargs)

    async def _ainfer_recovery(
        self,
        inference_input: Any,
        last_exception: Optional[Exception],
        last_partial_output: Optional[str],
        inference_config: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """Overridable async recovery method. Default delegates to _ainfer.

        Called as the fallback function when the primary _ainfer fails.
        Subclasses (e.g., StreamingInferencerBase) can override to implement
        cache-aware or session-aware recovery strategies.

        Args:
            inference_input: The original inference input.
            last_exception: The exception from the failed primary attempt.
            last_partial_output: Any partial output from the failed attempt, or None.
            inference_config: Optional inference configuration.
            **kwargs: Additional keyword arguments passed through to _ainfer.
        """
        return await self._ainfer(inference_input, inference_config, **kwargs)

    async def _ainfer_single(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        # ── Phase 1 (leaf-owned template rendering): extract per-call render
        # parameters from _inference_args. See _infer_single for design notes.
        _extra_feed = _inference_args.pop("extra_feed", None)
        _render_only = _inference_args.pop("render_only", False)
        return await self.__ainfer_single_impl(
            inference_input,
            inference_config,
            _extra_feed=_extra_feed,
            _render_only=_render_only,
            **_inference_args,
        )

    async def __ainfer_single_impl(
        self,
        inference_input: Any,
        inference_config: Any = None,
        *,
        _extra_feed: Optional[dict] = None,
        _render_only: bool = False,
        **_inference_args,
    ):
        """Async process a single inference input with preprocessing, inference, and post-processing.

        Async equivalent of _infer_single(). Handles:
        1. Input preprocessing via input_preprocessor
        2. Merging default_inference_args with provided args
        3. Executing _ainfer() with retry logic
        4. Post-processing via response_post_processor

        Args:
            inference_input: Input data for inference.
            inference_config: Optional configuration for the inference run.
            **_inference_args: Additional keyword arguments merged with default_inference_args.

        Returns:
            The post-processed inference response.
        """
        from rich_python_utils.common_utils.async_function_helper import (
            async_execute_with_retry,
        )

        self._propagate_to_children()

        # Capture original input BEFORE preprocessing for retry_with_original mode
        original_input = inference_input

        if self.input_preprocessor is not None:
            inference_input = self.input_preprocessor(inference_input)

        # Template rendering (opt-in: only when template_manager is set).
        # Round-7 invariant: conditional kwarg pass — see _infer_single.
        if _extra_feed is not None:
            inference_input = self._render_prompt(inference_input, extra_feed=_extra_feed)
        else:
            inference_input = self._render_prompt(inference_input)

        # Phase 1: render_only short-circuit — see _infer_single.
        if _render_only:
            return inference_input

        # Resume hook (async): check for cached result from a previous session.
        # Uses _atry (async) so streaming override can await self._ainfer().
        resume_result = await self._atry_resume_from_cache(
            inference_input, inference_config, **_inference_args
        )
        if resume_result is not None:
            # Run the same post-processing tail as the normal path
            resume_result = self._finalize_output(resume_result)
            if self.state_graphs:
                self.update_state_graphs(resume_result)
            if self.response_post_processor is not None:
                resume_result = self.response_post_processor(self._normalize_for_post_processor(resume_result))
            return resume_result

        inference_args = self.default_inference_args.copy()
        if _inference_args:
            inference_args.update(_inference_args)

        # Augment with state graph args
        if self.state_graphs:
            inference_args.update(self.get_inference_args_from_state_graphs())

        # Pop runtime overrides that should not be forwarded to _ainfer()
        on_retry_callback = inference_args.pop("on_retry_callback", None)
        total_timeout = inference_args.pop(
            "total_timeout_seconds", self.total_timeout_seconds
        )
        attempt_timeout = inference_args.pop(
            "attempt_timeout_seconds", self.attempt_timeout_seconds
        )
        fallback_mode = inference_args.pop("fallback_mode", self.fallback_mode)
        on_fallback_callback = inference_args.pop("on_fallback_callback", None)
        retry_prompt_mode = inference_args.pop("retry_prompt_mode", "original")

        # Validate retry_prompt_mode
        if retry_prompt_mode not in RETRY_PROMPT_MODES:
            raise ValueError(
                f"Invalid retry_prompt_mode={retry_prompt_mode!r}. "
                f"Must be one of {RETRY_PROMPT_MODES}"
            )

        # Mutable args list — prompt can be swapped by the retry callback
        retry_args = [inference_input]

        # Build internal retry callback (handles pre_retry propagation,
        # prompt transformation, and user callback). The async path
        # additionally fires `self.pre_retry` when subclasses opt in via
        # `_pre_retry` or `_iter_child_inferencers` overrides — detected
        # via the standard override-detection pattern. Zero cost when
        # not overridden.
        _user_callback = on_retry_callback
        pre_retry_active = (
            type(self)._pre_retry is not InferencerBase._pre_retry
            or type(self)._iter_child_inferencers
                is not InferencerBase._iter_child_inferencers
        )
        if (
            _user_callback is not None
            or retry_prompt_mode != "original"
            or pre_retry_active
        ):
            async def _internal_retry_callback(attempt, exception):
                # 1. Subclass hook + recursive child propagation — fires
                #    first so subsequent steps see clean state.
                if pre_retry_active:
                    await self.pre_retry(attempt, exception)
                # 2. Forward to user callback with local inference_args
                #    (preserves prior sync semantics).
                if _user_callback is not None:
                    _user_callback(attempt, exception, inference_args)
                # 3. Transform prompt based on retry_prompt_mode
                if retry_prompt_mode == "simple_retry":
                    retry_args[0] = _SIMPLE_RETRY_PROMPT
                elif retry_prompt_mode == "retry_with_original":
                    retry_args[0] = (
                        _SIMPLE_RETRY_PROMPT + " The task was:\n" + str(original_input)
                    )
                if retry_prompt_mode != "original":
                    self.log_info(
                        f"Retry prompt ({retry_prompt_mode}): {str(retry_args[0])[:200]}",
                        "RetryPrompt",
                    )

            on_retry_callback = _internal_retry_callback

        self.log_info(inference_input, "InferenceInput")
        self.log_info(inference_args, "InferenceArgs")

        # Convert 0 → None for timeout parameters (0 = disabled)
        effective_total_timeout = total_timeout or None
        effective_attempt_timeout = attempt_timeout or None

        # -- Build _fallback_state and fallback chain --
        _fallback_state = {"last_exception": None, "partial_output": None, "cache_path": None}

        # Recovery wrapper — reads from closure-captured _fallback_state
        async def _recovery_wrapper(inp, **kw):
            return await self._ainfer_recovery(
                inp,
                last_exception=_fallback_state["last_exception"],
                last_partial_output=_fallback_state["partial_output"],
                inference_config=inference_config,
                **inference_args,
            )

        # External fallback wrappers from fallback_inferencer list
        external_wrappers = []
        if self.fallback_inferencer is not None:
            fb_list = (
                self.fallback_inferencer
                if isinstance(self.fallback_inferencer, list)
                else [self.fallback_inferencer]
            )
            external_wrappers = [
                lambda inp, inf=inf, **kw: inf.ainfer(inp, inference_config, **kw)
                for inf in fb_list
            ]

        # Build fallback chain and mode for the retry helper
        if fallback_mode == FallbackMode.NEVER:
            effective_fallback_func = None
            effective_fallback_mode = FallbackMode.NEVER
        else:
            effective_fallback_func = [_recovery_wrapper] + external_wrappers
            effective_fallback_mode = fallback_mode

        # Transition callback — populates _fallback_state and resets retry_args[0]
        _user_on_fallback = on_fallback_callback

        async def _on_transition(from_func, to_func, exception, total_attempts):
            _fallback_state["last_exception"] = exception
            if _fallback_state["cache_path"]:
                try:
                    with open(_fallback_state["cache_path"], "r", encoding="utf-8") as f:
                        raw = f.read()
                    _fallback_state["partial_output"] = raw if raw.strip() else None
                except OSError:
                    _fallback_state["partial_output"] = None
            # Reset retry_args[0] to original input so external fallback
            # inferencers see the original prompt, not the mutated retry prompt
            retry_args[0] = original_input
            # Forward to user-provided on_fallback_callback if present
            if _user_on_fallback is not None:
                result = _user_on_fallback(from_func, to_func, exception, total_attempts)
                if asyncio.iscoroutine(result):
                    await result

        # Set ContextVar for this call (per-task safe under aparallel_infer)
        token = _current_fallback_state.set(_fallback_state)
        try:
            inference_response = await async_execute_with_retry(
                func=lambda inp: self._ainfer(inp, inference_config, **inference_args),
                max_retry=self.max_retry,
                min_retry_wait=self.min_retry_wait,
                max_retry_wait=self.max_retry_wait,
                args=retry_args,
                default_return_or_raise=self.default_return_or_raise,
                on_retry_callback=on_retry_callback,
                non_retryable_exceptions=(
                    asyncio.LimitOverrunError,
                    asyncio.IncompleteReadError,
                ),
                total_timeout=effective_total_timeout,
                attempt_timeout=effective_attempt_timeout,
                fallback_func=effective_fallback_func,
                fallback_mode=effective_fallback_mode,
                on_fallback_callback=_on_transition if effective_fallback_func else None,
            )
        except TimeoutError:
            self.log_info(
                f"Total timeout after {total_timeout}s",
                "TotalTimeout",
            )
            raise
        finally:
            _current_fallback_state.reset(token)

        self.log_debug(inference_response, "InferenceResponse")

        # Output finalization (promote deliverables, write summary)
        inference_response = self._finalize_output(inference_response)

        # Update state graphs from response
        if self.state_graphs:
            self.update_state_graphs(inference_response)

        if self.response_post_processor is not None:
            post_input = self._normalize_for_post_processor(inference_response)
            processed_response = self.response_post_processor(post_input)
            self.log_debug(processed_response, "PostProcessedResponse")
            return processed_response

        return inference_response

    async def _ainfer_iterator(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ) -> AsyncIterator:
        """Async process an iterator of inputs and yield atomized post-processed results.

        Async equivalent of _infer_iterator().

        Args:
            inference_input: Iterator of input items to process.
            inference_config: Optional configuration for the inference run.
            **_inference_args: Additional keyword arguments passed to _ainfer_single().

        Yields:
            Atomized inference results for each input item.
        """
        for _inference_input in inference_input:
            response = await self._ainfer_single(
                _inference_input, inference_config, **_inference_args
            )
            for item in iter__(response, atom_types=self.response_types):
                yield item

    async def ainfer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        """Async version of infer().

        NOTE: For Iterator inputs without post_response_merger, sync infer() returns
        a lazy generator while this method eagerly collects results into a list.
        This is an inherent Python limitation — async generators cannot be returned
        from a regular async def and consumed as a value. All results are computed
        upfront, which uses more memory than the lazy sync path. For large iterator
        inputs where memory is a concern, process items individually with
        await _ainfer_single() in a loop instead.

        Args:
            inference_input: Input data for inference. Can be a single input or an Iterator.
            inference_config: Optional configuration for the inference run.
            **_inference_args: Additional keyword arguments passed to the inference methods.

        Returns:
            If input is Iterator with post_response_merger: Returns a single merged result.
            If input is Iterator without post_response_merger: Returns a list of atomized results.
            Otherwise: Returns a single post-processed inference result.
        """
        if isinstance(inference_input, Iterator):
            all_results = []
            async for item in self._ainfer_iterator(
                inference_input, inference_config, **_inference_args
            ):
                all_results.append(item)

            if self.post_response_merger is not None:
                merged = self.post_response_merger(all_results)
                self.log_debug(merged, "MergedResponse")
                return merged
            return all_results
        else:
            return await self._ainfer_single(
                inference_input, inference_config, **_inference_args
            )

    async def aiter_infer(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ) -> AsyncIterator:
        """Async version of iter_infer().

        Execute inference and always return an async iterator of responses.

        Args:
            inference_input: Input data for inference (single input or iterator).
            inference_config: Optional configuration for the inference run.
            **_inference_args: Additional keyword arguments passed to inference.

        Yields:
            Inference responses. Atomization behavior depends on response_types configuration.
        """
        response = await self.ainfer(
            inference_input=inference_input,
            inference_config=inference_config,
            **_inference_args,
        )
        if not self.response_types:
            if isinstance(response, (list, Iterator)):
                for item in response:
                    yield item
            else:
                yield response
        else:
            for item in iter__(response, atom_types=self.response_types):
                yield item

    async def aparallel_infer(
        self,
        inference_inputs: Iterable[Any],
        inference_config: Any = None,
        max_concurrency: int = None,
        debug: bool = False,
        **_inference_args,
    ) -> list:
        """Async process multiple inputs concurrently using asyncio.gather.

        Dispatches each input to _ainfer_single() with optional semaphore-based
        concurrency control to prevent event loop/connection pool exhaustion.

        Args:
            inference_inputs: Iterable of inputs to process concurrently.
                Generators are supported (materialized internally).
            inference_config: Optional configuration passed to each _ainfer_single call.
            max_concurrency: Maximum number of concurrent tasks. None defaults to
                min(len(inputs), 32) to prevent unbounded concurrency.
            debug: True runs sequentially via await loop (parity with sync
                parallel_infer debug mode).
            **_inference_args: Additional keyword arguments merged with
                default_inference_args and passed to _ainfer_single().

        Returns:
            List of inference results, order-preserving (same index alignment
            as inputs).

        Note:
            post_response_merger is NOT auto-applied — same rationale as
            parallel_infer. Users can apply their own merging on the returned list.
        """
        inference_inputs = list(inference_inputs)
        if not inference_inputs:
            return []

        num_inputs = len(inference_inputs)

        if debug:
            self.log_debug(
                f"aparallel_infer debug mode: {num_inputs} inputs",
                "ParallelInfer",
            )
            results = []
            for inp in inference_inputs:
                result = await self._ainfer_single(
                    inp, inference_config, **_inference_args
                )
                results.append(result)
            return results

        if max_concurrency is None:
            max_concurrency = min(num_inputs, 32)

        self.log_debug(
            f"{num_inputs} inputs, max_concurrency={max_concurrency}",
            "ParallelInfer",
        )

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_infer(inp):
            async with semaphore:
                return await self._ainfer_single(
                    inp, inference_config, **_inference_args
                )

        tasks = [_bounded_infer(inp) for inp in inference_inputs]
        results = await asyncio.gather(*tasks)
        return list(results)

    # region Async Lifecycle Methods

    async def aconnect(self, **kwargs):
        """Establish async connection to external service.

        Override this method in subclasses that need persistent connections.
        Default implementation does nothing.

        Args:
            **kwargs: Connection-specific arguments.
        """
        pass

    async def adisconnect(self):
        """Disconnect from external service.

        Override this method in subclasses that need cleanup.
        Default implementation does nothing.
        """
        pass

    async def __aenter__(self):
        """Async context manager entry. Calls aconnect()."""
        await self.aconnect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit. Calls adisconnect()."""
        await self.adisconnect()
        return False

    # endregion


# ---------------------------------------------------------------------------
# Public protocol constants
# ---------------------------------------------------------------------------
# Module-level constant naming the runtime-context attribute that participates
# in ``_propagate_to_children()`` propagation. Derived from the attrs field
# declaration so that renaming the attrs field automatically updates the
# constant — single source of truth, no double-truth maintenance burden.
#
# Use this constant whenever performing **string-key lookups** for the
# template_extra_feed attribute (e.g., ``getattr(obj, TEMPLATE_EXTRA_FEED_ATTR)``,
# ``partial.keywords.get(TEMPLATE_EXTRA_FEED_ATTR)``). Direct attribute access
# (``self.template_extra_feed``) is still preferred where possible — it is more
# readable, IDE-friendly, and compile-time validated. The constant is for code
# that must look up the attribute by name (introspection, partial keyword
# manipulation, duck-typed protocol checks).
TEMPLATE_EXTRA_FEED_ATTR: str = "template_extra_feed"


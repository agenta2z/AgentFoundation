import asyncio
import contextlib
import enum
import hashlib
import inspect
import logging
import os
import sys
import threading
import time
import traceback
import types
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextvars import ContextVar
from functools import lru_cache, partial
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    AsyncIterator,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Type,
    Union,
)

import attrs as attrs_mod

# M2: explicit RunContext carrier + the compat bridge (additive; inert until M3+
# orchestrators read `_active_ctx`). `run_context` is **keyword-only** so it can
# never land in `**_inference_args` (the kwarg-leak defense). The bridge mints a
# legacy root when `run_context is None` (byte-identical) and is per-task (the
# `_active_ctx` ContextVar) so concurrent fan-out branches don't clobber.
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    aopen_invocation,
    ctx_bound_gen,
    enter_run,
    exit_run,
    frame_for,
    host_pure_certified,
    InferencerStateBase,
    invocation_of,
    InvocationContractError,
    InvocationFrame,
    LiveHandleStore,
    NodeOutcomeState,
    open_invocation,
    read_outcome,
    RenderedTaskContractState,
    resolve_run,
    RuntimeKey,
    UncertifiedConcurrentUseError,
)
from agent_foundation.common.inferencers.template_feed_scope import (
    TEMPLATE_EXTRA_FEED_SCOPE_HANDLE,
)
from attr import attrib, attrs, Factory, fields, NOTHING
from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_objects.workflow.common.resumable import Resumable
from rich_python_utils.common_utils import dict_, iter__, resolve_environ
from rich_python_utils.common_utils.function_helper import (
    execute_with_retry,
    FallbackMode,
    OutputValidationExhaustedError,
)
from rich_python_utils.config_utils import import_target
from rich_python_utils.config_utils._lazy_config_factory import LazyConfigFactory
from rich_python_utils.path_utils import AllowedPath, PathAccess

# Retry prompt mode constants
RETRY_PROMPT_MODES = ("original", "simple_retry", "retry_with_original")

# Module-level ContextVar for per-call fallback state.
# Each asyncio Task from aparallel_infer gets its own context copy — no race.
# Shared with streaming_inferencer_base.py for cache path tracking.
_current_fallback_state: ContextVar[dict | None] = ContextVar(
    "_current_fallback_state", default=None
)

_SIMPLE_RETRY_PROMPT = "You got interrupted. Can you retry the above task?"

_logger = logging.getLogger(__name__)

# Serializes the deferred workspace-logger un-defer: ``parallel_infer`` threads
# share one instance, so the flag check and the logger install must be atomic.
_WORKSPACE_LOGGER_LOCK = threading.RLock()

# Per-call BTA fan-out (``InferencerBase.bta_inferencer``).
MAX_BTA_FANOUT_DEPTH = 8
BTA_INFERENCER_SLOT = "bta_inferencer"
BTA_OWN_ROLE = "own"
# Feed keys naming where the executing actor writes. A delegating inferencer's
# contract omits them: the fan-out's executors each write to their own location.
ACTOR_SCOPED_FEED_KEYS = ("output_path", "workspace_root", "workspace_outputs")
# Per-call framework kwargs that govern the whole fan-out call, not each worker.
_FANOUT_FRAMEWORK_ARGS = (
    "on_retry_callback",
    "total_timeout_seconds",
    "attempt_timeout_seconds",
    "fallback_mode",
    "on_fallback_callback",
    "retry_prompt_mode",
)
# Boundary processing reset on the fanned-out inferencer's copies: it runs once,
# around the fan-out.
_FANOUT_RESET_FIELDS = (
    "input_preprocessor",
    "response_post_processor",
    "expected_extraction",
    "state_graphs",
)


# v5 Phase 1 — Correlation instrumentation. Env-gated so production stays
# byte-identical. When RESEARCH_PROPOSE__VERBOSE_CORRELATION=1, every
# InferenceInput / InferenceResponse parts file emitted from a single
# (a)infer call gets the same `call_<8hex>` filename segment, so input
# ↔ response pair lookup becomes filename-only (no JSONL line-ordering
# reasoning required, which breaks under guardrail-retry interleave).
# RoleSwitch, WorkspaceReassign, GuardrailRetry, and ToolUsePhase
# structured events are also emitted under the same gate (sites below).
_VERBOSE_CORRELATION_ENV = "RESEARCH_PROPOSE__VERBOSE_CORRELATION"


def _is_verbose_correlation() -> bool:
    return os.environ.get(_VERBOSE_CORRELATION_ENV) == "1"


def _call_correlation_kwargs(call_id: str) -> dict:
    """Build the ``parts_file_namer`` kwarg dict for ``log_info`` /
    ``log_debug`` so the minted ``call_id`` becomes the filename name-hint
    segment in every parts file emitted under this call (see
    ``rich_python_utils.io_utils.json_io.write_json`` line ~1854 where the
    ``name_hint`` is concatenated into ``<TS>_<name_hint>_<file_stem>_<uid>``).

    Empty dict when env var is unset → log calls render exactly as today.
    """
    if not _is_verbose_correlation():
        return {}
    return {"parts_file_namer": lambda _obj, _cid=call_id: f"call_{_cid}"}


def _is_bookkeeping_sidecar(name: str) -> bool:
    """True for loose ``outputs/`` entries that are framework BOOKKEEPING, not
    deliverables — used by loose-sidecar promotion (Fix 2) so a child's internal
    bookkeeping is never lifted into the parent's ``outputs/``.

    Denylist (provenance-anchored to the framework's own write sites): dotfiles,
    the Dual ``round_log.jsonl``, and any ``*_manifest.json``. The canonical
    output name and ``final_deliverables/`` are context-dependent and skipped at
    the call site, not here.
    """
    if name.startswith("."):
        return True
    if name == "round_log.jsonl":
        return True
    if name.endswith("_manifest.json"):
        return True
    return False


def _move_into_archive(state_dir: str, target: str, keep: FrozenSet[str]) -> None:
    """Move ``state_dir`` to ``target``; with ``keep``, only its other entries."""
    import shutil

    if not keep:
        shutil.move(state_dir, target)
        return
    os.makedirs(target, exist_ok=True)
    for entry in os.listdir(state_dir):
        if entry not in keep:
            shutil.move(os.path.join(state_dir, entry), target)


def safe_slot(part: object) -> str:
    """``part`` as one valid run-context slot: no path separator, never "."
    or ".." (``RunContext.child`` rejects those, the workspace child
    derivation any ".."), so a slot derived from a node id or tool name never
    raises mid-call."""
    safe = str(part).replace("/", "_").replace("\\", "_").strip()
    return "child" if safe in ("", ".", "..") else safe.replace("..", "_")


class MissingDependencyError(ImportError):
    """A required runtime hard-dependency (module/binary) is unavailable (U2d).

    Subclasses ``ImportError`` so callers may also catch it structurally. Listed
    in the ``non_retryable_exceptions`` tuples below so a missing dependency
    FAILS FAST instead of retrying — a missing module never reappears on retry,
    and (pre-U1/U4) that pointless retry is what triggered the observed
    ``CollisionError`` cascade. Raise it at dep-resolution sites (e.g. the
    metamate client resolver) instead of a generic ``RuntimeError``.
    """


class HopelessOutputError(Exception):
    """Raised by ``_run_output_guardrail`` when N consecutive identical empty
    or banner-shaped outputs trip the fail-fast threshold
    (``guardrail_empty_fail_fast_n``).

    Distinct from generic retry-exhausted exceptions: this is a TERMINAL
    diagnostic that says "further retries will not produce anything different".
    MultiFlow quorum (v4 Phase 3.1) catches this in ``_make_worker_fn`` so the
    failing flow is marked terminal-error without burning the rest of the
    retry budget, and survivors continue.
    """

    pass


class ModelTier(str, enum.Enum):
    """Provider-agnostic model quality tier.

    Each CLI inferencer maps these to concrete model names for its provider:
    - Claude: ``MAX`` → opus[1m], ``DEFAULT`` → sonnet, ``LITE`` → haiku
    - OpenAI/Codex: ``MAX`` → gpt-5.5, ``DEFAULT`` → gpt-5.4, ``LITE`` → gpt-5.4-mini

    Use as an alternative to ``model_name`` when you want to specify intent
    ("I need a cheap judge") rather than a provider-specific model identifier.
    Set via ``model_tier`` attr on any inferencer; ``__attrs_post_init__``
    resolves it to the concrete ``model_name``.
    """

    MAX = "max"
    DEFAULT = "default"
    LITE = "lite"


class _PrototypeCloneFactory:
    """Factory wrapping a constructed inferencer prototype: each call returns a DEEP,
    fully-independent copy with fresh ids (via ``deepcopy_with_fresh_id``).

    Created automatically when a *bare* inferencer instance is assigned, in Python, to a
    registered factory field (``metadata={"lazy_config_factory": True}`` — e.g. BTA's
    ``worker_inferencers``). This gives the Python-assignment path the same fresh-per-call
    semantics as the YAML/``LazyConfigFactory`` path, so each subtask gets its own worker.

    ``__call__`` accepts and ignores any args so it satisfies BOTH the no-arg factory
    convention and the ``(sub_query, index)`` convention used by BTA's worker resolver.
    """

    __slots__ = ("prototype",)

    def __init__(self, prototype):
        self.prototype = prototype

    def __call__(self, *args, **kwargs):
        return self.prototype.deepcopy_with_fresh_id()

    def resume_identity(self):
        return self.prototype

    def __repr__(self):
        return f"_PrototypeCloneFactory({type(self.prototype).__name__})"


class _FreshCloneFactory:
    """Factory wrapping an inferencer prototype: each call returns
    ``prototype.fresh_instance()``, an independent instance rebuilt from the
    prototype's construction recipe (runtime bindings are never copied).

    ``__call__`` accepts and ignores any args, like ``_PrototypeCloneFactory``.
    """

    __slots__ = ("prototype",)

    def __init__(self, prototype):
        self.prototype = prototype

    def __call__(self, *_args, **_kwargs):
        return self.prototype.fresh_instance()

    def resume_identity(self):
        return self.prototype

    def __repr__(self):
        return f"_FreshCloneFactory({type(self.prototype).__name__})"


@attrs(frozen=True, slots=True)
class ResolvedStage:
    """A stage a call dispatches: ``owned`` when the call built it (a factory's
    product, closed by the call), ``borrowed`` when it is a configured instance
    (closed only by its definition's ``adisconnect``)."""

    inferencer: Any = attrib()
    ownership: str = attrib()

    @property
    def owned(self) -> bool:
        return self.ownership == "owned"


def resolve_stage(slot_value, *args, **kwargs) -> ResolvedStage:
    """A callable that isn't an ``InferencerBase`` is a factory: its product is owned.
    Anything else is borrowed."""
    if callable(slot_value) and not isinstance(slot_value, InferencerBase):
        return ResolvedStage(slot_value(*args, **kwargs), "owned")
    return ResolvedStage(slot_value, "borrowed")


class _LiveField:
    """Recipe placeholder for a lazy factory field. The constructor receives Hydra's
    interim ``functools.partial`` (holding eagerly built children), which the config
    loader replaces after ``__init__``, so a rebuild reads the live field instead.
    Copy, deepcopy and pickle all preserve the singleton."""

    __slots__ = ()

    def __reduce__(self):
        return "_LIVE_FIELD"

    def __repr__(self):
        return "_LIVE_FIELD"


_LIVE_FIELD = _LiveField()


@lru_cache(maxsize=None)
def _init_signature(cls: type) -> Optional[inspect.Signature]:
    """``cls.__init__``'s signature, or ``None`` when it takes ``*args``/``**kwargs``."""
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return None
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    if any(p.kind in variadic for p in signature.parameters.values()):
        return None
    return signature


def _init_param_names(cls: type) -> tuple:
    signature = _init_signature(cls)
    return tuple(signature.parameters)[1:] if signature is not None else ()


@lru_cache(maxsize=None)
def _lazy_init_params(cls: type) -> frozenset:
    """Init-parameter names of lazy factory fields (the config loader's opt-in rule)."""
    return frozenset(
        a.alias
        for a in fields(cls)
        if a.name.endswith("_factory") or a.metadata.get("lazy_config_factory", False)
    )


@lru_cache(maxsize=None)
def _non_config_init_params(cls: type) -> frozenset:
    """Init-parameter names of ``NON_CONFIG_ATTR_NAMES`` fields (identity, parent links)."""
    return frozenset(
        a.alias for a in fields(cls) if a.name in cls.NON_CONFIG_ATTR_NAMES
    )


def _field_defaults(cls: type, names: Iterable[str]) -> Dict[str, Any]:
    """``{init_param: class default}`` for the named attrs fields of ``cls`` that
    have a default; factories are called (``takes_self`` factories are skipped)."""
    wanted = set(names)
    defaults = {}
    for a in fields(cls):
        if a.name not in wanted or not a.init or a.default is NOTHING:
            continue
        if not isinstance(a.default, Factory):
            defaults[a.alias] = a.default
        elif not a.default.takes_self:
            defaults[a.alias] = a.default.factory()
    return defaults


def _snapshot(value: Any) -> Any:
    """Copy exact containers recursively; keep every other value by reference."""
    kind = type(value)
    if kind is dict:
        return {k: _snapshot(v) for k, v in value.items()}
    if kind in (list, tuple, set):
        return kind(_snapshot(v) for v in value)
    return value


def _capture_init_recipe(
    cls: type, args: tuple, kwargs: dict
) -> Optional[Dict[str, Any]]:
    """``{init_param: snapshot}`` for the arguments explicitly passed to ``cls(...)``,
    or ``None`` when they cannot be bound to ``cls.__init__`` by name."""
    signature = _init_signature(cls)
    if signature is None:
        return None
    try:
        bound = signature.bind(None, *args, **kwargs)
    except TypeError:
        return None
    lazy = _lazy_init_params(cls)
    return {
        name: _LIVE_FIELD if name in lazy else _snapshot(value)
        for name, value in list(bound.arguments.items())[1:]
    }


class _InstanceRebuilder:
    """One ``fresh_instance`` pass. Memoizes rebuilt inferencers so shared children
    stay shared in the copy, and rejects an inferencer reachable from its own recipe."""

    def __init__(self) -> None:
        self._memo: Dict[int, Any] = {}
        self._in_progress: Set[int] = set()

    def build(self, inf: "InferencerBase", overrides: Dict[str, Any]) -> Any:
        key = id(inf)
        if key in self._memo:
            return self._memo[key]
        if key in self._in_progress:
            raise ValueError(
                f"cycle: {type(inf).__name__} is reachable from its own construction "
                "recipe; cannot fresh_instance"
            )
        recipe = inf.__dict__.get("_init_recipe")
        if recipe is None:
            raise TypeError(
                f"{type(inf).__name__} was not constructed through a bindable "
                "__init__; cannot fresh_instance"
            )
        unknown = sorted(set(overrides) - set(_init_param_names(type(inf))))
        if unknown:
            raise TypeError(
                f"{type(inf).__name__}.fresh_instance got unknown overrides "
                f"{unknown}; overrides must be __init__ parameters"
            )
        self._in_progress.add(key)
        try:
            kwargs = self._rebuild_kwargs(inf, recipe, overrides)
            fresh = type(inf)(**kwargs, **overrides)
        finally:
            self._in_progress.discard(key)
        self._memo[key] = fresh
        return fresh

    def _rebuild_kwargs(
        self, inf: "InferencerBase", recipe: Dict[str, Any], overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        cls = type(inf)
        skip = _non_config_init_params(cls) | overrides.keys()
        field_names = {a.alias: a.name for a in fields(cls)}
        kwargs = {}
        for name, value in recipe.items():
            if name in skip:
                continue
            if value is _LIVE_FIELD:
                value = getattr(inf, field_names.get(name, name))
            kwargs[name] = self.rebuild_value(value)
        return kwargs

    def rebuild_value(self, value: Any) -> Any:
        if isinstance(value, InferencerBase):
            return self.build(value, {})
        if type(value) in (_PrototypeCloneFactory, _FreshCloneFactory):
            return type(value)(self.build(value.prototype, {}))
        if isinstance(value, LazyConfigFactory):
            return value.fresh()
        if isinstance(value, partial):
            return partial(
                self.rebuild_value(value.func),
                *self.rebuild_value(value.args),
                **self.rebuild_value(value.keywords),
            )
        if inspect.ismethod(value) and isinstance(value.__self__, InferencerBase):
            return types.MethodType(value.__func__, self.build(value.__self__, {}))
        kind = type(value)
        if kind is dict:
            return {k: self.rebuild_value(v) for k, v in value.items()}
        if kind in (list, tuple, set):
            return kind(self.rebuild_value(v) for v in value)
        return value


@lru_cache(maxsize=None)
def _merged_invocation_keywords(cls: type) -> Mapping[str, Any]:
    merged: dict = {}
    for klass in reversed(cls.__mro__):
        merged.update(vars(klass).get("_INVOCATION_KEYWORDS", {}))
    return MappingProxyType(merged)


@attrs_mod.frozen
class RoleTransition:
    """The template-layer attributes one ``switch_role`` call sets, handed from
    ``TemplatedInferencerBase.switch_role`` to the base layer's audit trail."""

    changes: Mapping[str, Any] = attrs_mod.field(
        factory=dict, converter=lambda value: MappingProxyType(dict(value))
    )


# Set on a Tier-3 branch that reads only its own handles, never the instance
# backing (``InferencerBase._tier3_detach_from_backing``).
_TIER3_OWN_HANDLES_ONLY = "own_handles_only"


class _BackingHandleView:
    """Adapts an instance's legacy ``_<name>_backing`` attrs to the ``LiveHandles``
    ``get``/``set`` API so a leaf teardown treats the no-context backing uniformly
    with the connection-scoped branches (see ``_iter_live_handle_sets``)."""

    __slots__ = ("_owner",)

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def get(self, name: str, default: Any = None) -> Any:
        return self._owner.__dict__.get(f"_{name}_backing", default)

    def set(self, name: str, value: Any) -> None:
        self._owner.__dict__[f"_{name}_backing"] = value


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

    # M4: optional per-call state factory. Generalizes LinearWorkflowInferencer's
    # ``initial_state_factory`` (V10). When set AND a RunContext is active, the
    # bridge populates ``ctx.node.call`` once per call with ``state_factory(input)``
    # (a typed InferencerStateBase or a plain dict). Default ``None`` -> no-op,
    # byte-identical. The state lives in the context, never on ``self``.
    state_factory: Optional[Any] = attrib(default=None)

    # Optional graph reporter for UI visualization (e.g. WebSocketGraphReporter).
    # Protocol (duck-typed): on_graph_topology(event), on_node_status(node_id,
    # status, error="", output_path=""), on_graph_reconcile(node_statuses),
    # child_reporter(node_id). Set by the executor after instantiation (this
    # module never imports WebSocket/UI code). Promoted into the shared Tier-2
    # sink (ctx.runtime.graph_reporter) by _seed_graph_reporter_into_runtime() in
    # ainfer()/infer(), so EVERY inferencer participates in graph viz — not just
    # BTA. None (default) => no viz, byte-identical.
    graph_reporter: Optional[Any] = attrib(default=None, kw_only=True)

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
    fallback_inferencer: Union["InferencerBase", List["InferencerBase"], None] = attrib(
        default=None
    )

    # Controls when the retry helper transitions to the next fallback callable.
    fallback_mode: FallbackMode = attrib(default=FallbackMode.ON_FIRST_FAILURE)

    response_types: Sequence[Type] = attrib(default=(str,))
    default_inference_args: dict = attrib(default=None, converter=dict_)
    input_preprocessor: Callable = attrib(default=None)
    response_post_processor: Union[str, dict, Callable] = attrib(default=None)
    post_response_merger: Union[str, Callable] = attrib(default=None)

    model_tier: Optional[str] = attrib(default=None)
    """Provider-agnostic model quality tier (``"max"``/``"default"``/``"lite"``).
    When set, the inferencer's ``__attrs_post_init__`` resolves it to a concrete
    ``model_name`` for its provider. Overrides ``model_name`` when both are set.
    None (default) = use ``model_name`` as-is. See ``ModelTier`` enum."""

    output_guardrail_inferencer: Optional["InferencerBase"] = attrib(default=None)
    """Optional lightweight LLM judge that validates output quality. When assigned,
    it runs after each successful ``_ainfer``/``_infer`` return (inside the retry
    loop's ``output_validator``). Receives a rendered judge prompt (the main input +
    output) and returns a verdict. If rejected, the recovery chain fires with the
    rejected output as ``last_partial_output``. Assign a cheap/fast model (e.g. haiku)
    as the judge; it gets its own workspace under ``children/guardrail/``.
    Disabled by default (``None``). Overridable verdict parser:
    ``_parse_guardrail_verdict``."""

    bta_inferencer: Optional[Any] = attrib(
        default=None,
        kw_only=True,
        metadata={"lazy_config_factory": True, "inferencer_template": True},
    )
    """Per-call fan-out template: run each call through a fresh
    ``BreakdownThenAggregateInferencer`` instead of this inferencer's own backend.

    Either one BTA template (a config factory, clone factory, callable or
    instance), applied to every call, or, on a templated inferencer,
    ``{role: template or None}``, keyed by the active ``switch_role`` role
    (``"own"`` when no switch is in effect); an unmapped or ``None`` role runs
    unfanned. The template itself is never run, bound or walked. Each call
    materializes a fresh BTA as this inferencer's ``bta_inferencer`` child, whose
    workers are fresh copies of this inferencer that receive their sub-query as
    the final prompt, and whose blank breakdown/aggregator slots are copies of
    this inferencer re-roled with the BTA's slot defaults. The preprocessor,
    render, post-processor, ``expected_extraction``, state graphs and output
    promotion of this inferencer run exactly once, around the fan-out.
    ``None`` (default) disables fan-out."""

    # Read via vars(type(owner)), so never inherited; the purity ratchet verifies it.
    _HOST_PURE_CERTIFIED: ClassVar[bool] = True

    # Fields left out of a resume identity (``run_context.resume_identity``), with
    # every ``Debuggable`` field and every secret: where the instance runs and
    # writes, runtime sinks, the save / resume policy itself, retry and timeout
    # scheduling, and failure fallbacks. Subclasses add theirs.
    _RESUME_IDENTITY_EXCLUDE: ClassVar[FrozenSet[str]] = frozenset(
        {
            "workspace",
            "source_path",
            "target_path",
            "additional_allowed_paths",
            "output_manifest_index",
            "graph_reporter",
            "state_factory",
            "state_graphs",
            "enable_result_save",
            "resume_with_saved_results",
            "checkpoint_mode",
            "max_retry",
            "min_retry_wait",
            "max_retry_wait",
            "total_timeout_seconds",
            "attempt_timeout_seconds",
            "fallback_inferencer",
            "fallback_mode",
            "default_return_or_raise",
            "surfaceable_exceptions",
            "guardrail_empty_fail_fast_n",
            "promote_exhausted_update",
        }
    )
    # False on classes that can't host a fan-out, e.g. entrypoints that bypass the
    # base inference seam.
    _SUPPORTS_BTA_FANOUT: ClassVar[bool] = True
    # True on classes that record their active role, so a role mapping can select.
    _SUPPORTS_BTA_ROLE_MAPPING: ClassVar[bool] = False
    # Per-call kwargs dropped before the fan-out's workers, and kwargs that continue
    # one backend session (independent workers cannot share a session).
    _FANOUT_DROPPED_ARGS: ClassVar[tuple] = ()
    _FANOUT_SINGLE_CALL_ARGS: ClassVar[tuple] = ()

    promote_exhausted_update: bool = attrib(default=True)
    """Publish the best ``UPDATE``-rejected output when the retry budget runs out,
    instead of discarding it.

    The judge contract (``recovery/judge.jinja2``) defines ``UPDATE`` as "real,
    on-topic work is present but it is incomplete … **We preserve it** and
    revise/complete it in place", and ``PASS`` as content that "need not be
    complete, deep, perfectly formatted". So ``PASS`` and ``UPDATE`` differ in
    degree, not in kind — ``UPDATE`` never means *incorrect*. Yet once attempts
    were exhausted the node raised and ``_finalize_output`` never ran, making a
    terminal ``UPDATE`` indistinguishable from producing nothing.

    ``RETRY`` exhaustion stays terminal: the judge defines it as "fundamentally
    unusable: empty, wildly off-topic, or narration-only", i.e. explicitly
    nothing to preserve. Only an ``UPDATE`` body is ever promoted, and a
    ``DegradedOutput`` record naming the judge's outstanding asks is logged so
    the artifact is never silently passed off as a clean pass."""

    guardrail_empty_fail_fast_n: int = attrib(default=2)
    """v4 Phase 3.2 — fail-fast threshold for identical empty / very-short
    outputs.

    When the guardrail rejects N consecutive outputs whose normalized
    fingerprint matches (typically banner-only emissions like Devmate's
    "Starting Devmate server and session..."), give up immediately
    instead of consuming the full ``max_retry`` budget × idle-timeout
    (which is ~25 min per stuck flow on the observed runs).

    Default = 2 (ON, conservative). Set to 0 to disable.

    Detection is conservative: an empty-or-banner-shaped output is one
    whose collapsed-whitespace length is <= 200 characters. Two identical
    fingerprints in a row = deterministic failure pattern; further retries
    have produced ~25 minutes of wasted wall-clock per flow in production.

    The trigger is per call: the window spans every attempt of one call
    and lives in that call's fallback state
    (``_current_fallback_state["guardrail_empty_fingerprints"]``), so a
    previous call on the same instance never counts. Reset on the next
    substantive or accepted output."""

    # State graph support — optional list of StateGraphTracker instances
    state_graphs: list = attrib(default=None)

    # Whether this inferencer has local file system access (e.g., can write files).
    # False for cloud API inferencers (RovoChat), True for local agents (RovoDevCli).
    has_local_access: bool = attrib(default=False)

    # region prompt-variable expansion (per-inferencer variable roots)
    # Master switch (hard gate): False -> no derived manager is built and
    # rendering is byte-identical to a stock TemplateManager (the per-key
    # overrides below are not even consulted). YAML-toggleable on _target_.
    enable_inferencer_variable_expansion: bool = attrib(default=True, kw_only=True)

    # Per-key opt-out, e.g. {"notes.large_file_writing": False} renders that one
    # key base-only while this inferencer's other extensions still apply. A
    # True/absent value expands. Consulted only when the master switch is ON.
    inferencer_variable_overrides: Dict[str, bool] = attrib(factory=dict, kw_only=True)

    # Memoized per-inferencer derived TemplateManager, built lazily at first
    # render (after every __attrs_post_init__ has added its template roots).
    # Never a constructor argument; never in repr.
    _extension_manager_cache: Optional[Any] = attrib(
        default=None, init=False, repr=False, kw_only=True
    )
    # endregion

    # Default output path — when relative and a workspace is set (via _workspace
    # on flow inferencers), resolves to workspace.outputs_dir/<output_path>.
    # Simple API inferencers leave this as None.
    output_path: Optional[str] = attrib(default=None)

    # Paths the inferencer (and any child subprocess it may spawn) should be
    # permitted to access in addition to its default scope (e.g., subprocess
    # cwd, API-provided context). Backend-neutral DATA slot — subclasses with
    # a native path-allowlist concept (e.g., RovoDevCliInferencer with acli's
    # toolPermissions.allowedExternalPaths) translate this list into their
    # native flag in their command/request construction; subclasses without
    # such a concept (mock, in-process Python, ...) simply ignore it.
    #
    # Orchestrators (LinearWorkflowInferencer, BreakdownThenAggregateInferencer,
    # ...) also carry the field so callers can set it once at the top level and
    # propagate to descendants alongside _workspace.
    #
    # Read effective values via ``effective_allowed_paths`` which auto-includes
    # the leaf's own ``workspace.root`` (with PathAccess.ALL) when a workspace
    # is set — that's a free safety net so an inferencer can always reach its
    # own workspace tree, regardless of how target_path / cwd are configured.
    # Cross-subtree reads (e.g., a leaf reading a SIBLING's artifacts) require
    # the orchestrator/executor to plumb the top task root explicitly.
    additional_allowed_paths: List[AllowedPath] = attrib(factory=list)

    output_manifest_index: bool = attrib(default=False)
    """When True, emit output_manifest.json (to ``artifacts/``) listing
    contributing files (LLM prompts, streaming cache, session logs).
    Part 2: independent of promotion (the deliverable flags are retired)."""

    surfaceable_exceptions: tuple = attrib(default=())
    """Part 4 (U4-B): per-inferencer allowlist of exception TYPES that must ALWAYS
    propagate (surface) out of a fan-out containment seam — never be swallowed into a
    quorum sentinel/dropped panelist. The containment-axis sibling of
    ``non_retryable_exceptions``. Consulted at each containment seam AFTER the
    non-overridable never-contain floor (CancelledError/KeyboardInterrupt/SystemExit/
    MemoryError + any non-Exception BaseException). Default empty → ``isinstance(e, ())``
    is always False, so containment behaviour is unchanged unless a tool opts in."""

    expected_extraction: "Optional[list]" = attrib(default=None)
    """Unified fenced-content registry (supersedes the old ``expected_contents_to_extract``):
    the fences THIS inferencer emits, GENERICALLY extracted + validated in its own output —
    emitter-side self-validation, ZERO domain knowledge. Each entry is a dict:

      ``{"label": "<fence>",              # the ```json <label>``` block, e.g. "proposal_index"
         "source": "output"|"response",   # channel THIS inferencer emits it on: the deliverable
                                           #   file (output.md) vs the stdout <Response>. Default "output".
         "kind": "content"|"control",     # content = part of the deliverable (proposal_index);
                                           #   control = a workflow signal a parent orchestrator consumes
                                           #   cross-node (winner_pick/ranking/iteration_judgment).
                                           #   Default "content".
         "fallback_to_source": bool,       # content only: if the fence can't be extracted, the whole
                                           #   source text stands in as the content — so a missing fence is
                                           #   NOT flagged by _run_expected_extraction, and _finalize_output
                                           #   writes the raw response for the no-file case. Always False for
                                           #   control (a cross-node consumer can't fall back to raw text).
         "persist_to": "<filename>",       # optional: also WRITE the extracted dict as JSON to this
                                           #   node's own ``outputs/<filename>``. Bare filename only —
                                           #   absolute paths and ``..`` are rejected. Best-effort: a
                                           #   failed write is logged and never gates.
         "checkpoint_scope": "parent"}``   # optional (requires ``persist_to``): after the child writes
                                           #   ``outputs/<persist_to>``, its PARENT promotes that file up
                                           #   into ``checkpoints/<child>/<persist_to>`` so it survives
                                           #   resume as durable state (a child cannot write to the parent
                                           #   directly — RunContext is parent->child). Pulled by
                                           #   ``_promote_child_checkpoints`` at child completion. Only
                                           #   "parent" is defined today; absent = no promotion.

    Extraction/validation is unified in the base (``_run_expected_extraction``): a missing /
    malformed fence is logged as an ``extraction_issue`` (observability ONLY — never gates).
    Generation is the prompt's job; this registry declares + checks expectations, and — when
    ``persist_to`` is set — EMITS the extracted block as a structured sidecar. The contract is
    "declare + check + (optionally) emit"; it still never gates. Opt-in; default ``None`` = no-op.
    IMPORTANT: control-fence CONSUMPTION (winner-selection, stop) stays with the orchestrator
    parsers (``flow_parsers.parse_*_tag`` — they carry per-fence value logic + legacy XML
    fallbacks); this registry is emitter-side self-validation, NOT the consumer.

    Child classes may register a per-label PARSER (``dict`` → domain object) via the
    ``BLOCK_PARSERS`` ClassVar; ``block_parsed()`` applies it. A parser cannot live in JSON/YAML
    config, which is why the block itself is declared in config while its parser is declared in
    code. Register the TRANSFORM only — never a replacement for a child's own tolerant text→dict
    extraction (e.g. BTA's ``_parse_json_subtasks`` also accepts an unlabeled fence, a bare
    ``{...\"subtasks\"...}`` scan, and a backtick-repair retry that ``_extract_json_block`` does not)."""

    BLOCK_PARSERS: ClassVar[dict] = {}
    """Per-label parsers for registered blocks: ``{label: callable(dict) -> Any}``, or
    ``{label: "<method_name>"}`` to bind an instance method (the usual case, since a domain
    transform typically reads instance config).

    Class-level (mirrors the ``SLOT_DEFAULTS`` pattern) because a callable cannot be expressed in
    JSON/YAML config — which is why a block is DECLARED in config while its meaning is declared in
    code. Subclasses override; ``block_parsed()`` looks the label up here. Empty default =
    ``block_parsed`` returns the raw dict."""

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
    # Where this inferencer's own source resources are addressable (project
    # root for a standard src-layout, fbsource root for inferencers whose
    # custom configs / templates live above the project — e.g. Devmate's
    # configs in tools/devmate/configs/...). Subclasses override the
    # _detect_source_root classmethod to customize.
    # Auto-detected via _detect_source_root() in __attrs_post_init__
    # if not explicitly set. Returns None for test stubs, pip-installed
    # packages, or non-src-layout projects — this is intentional.
    source_path: Optional[str] = attrib(default=None)

    # === Target path (operating directory for the agent) ===
    # The directory the agent operates on (e.g., the repo it edits, the cwd
    # of its subprocess, the root_folder it passes to an SDK client).
    # Distinct from workspace.root (artifact storage) and source_path
    # (where the inferencer's own source resources live).
    # Stays None if the agent doesn't need it (API / cloud / orchestrator
    # inferencers); subclasses that DO use it either consume it directly
    # or derive a subprocess cwd via the effective_cwd property below.
    target_path: Optional[str] = attrib(default=None)

    # === Deliverable model (Part 2: two-axis) ===
    # ``final_deliverables/`` and the deliverable flags (output_is_deliverable,
    # is_deliverable_boundary, use_final_deliverables_folder,
    # publishes_response_as_deliverable) are RETIRED. ``outputs/`` IS the
    # deliverable set; an orchestrator promotes its SELECTED canonical child's
    # ``outputs/`` up to its own ``outputs/`` via ``promote_child`` (see
    # ``_symlink_child_output``). PTI's multi-child aggregation selects children
    # by role (see ``deliverable_boundary.py``). ``artifacts/`` never promotes.

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
    # A workflow that re-roots itself for one invocation (PTI's resume workspace
    # under a host ctx, an LWI iteration) records the workspace here instead of
    # writing its backing; ``_workspace`` returns it while that invocation runs.
    _CALL_WORKSPACE = RuntimeKey("InferencerBase.call_workspace")

    @property
    def _workspace(self):
        from agent_foundation.common.inferencers.run_context import active_run_context

        frame = frame_for(self)
        if frame is not None and frame.has(InferencerBase._CALL_WORKSPACE):
            return frame.get(InferencerBase._CALL_WORKSPACE)
        return self._workspace_under(active_run_context())

    def _set_call_workspace(self, workspace) -> None:
        """Re-root this instance for the rest of the current invocation without
        touching its backing, children or loggers: a workspace-derived logger
        already follows ``_workspace`` per write (``_log_path_override``)."""
        invocation_of(self).put(InferencerBase._CALL_WORKSPACE, workspace)

    def _workspace_under(self, ctx):
        """The workspace this instance resolves when ``ctx`` is its active context."""
        # M7 §2.12 option-b: prefer a per-call workspace published into the
        # context's handles (``workspace_override``) when present — so a
        # single shared instance can serve concurrent branches with distinct
        # workspaces (the run-state is in the context, not on ``self``). Falls
        # back to the instance backing — **byte-identical** when no override is
        # set (no active context, or the orchestrator didn't publish one).
        if ctx is not None:
            override = ctx.handles.get("workspace_override", None)
            if override is not None:
                return override
        _backing = getattr(self, "_InferencerBase__workspace", None)
        # M3/§2.12: when the instance has NO configured workspace, honor the active
        # context's own workspace (a host-minted ``RunContext.root(workspace=...)``).
        # Byte-identical: a legacy run of an unconfigured instance mints a root from
        # ``self._workspace`` (=None), so ``ctx.workspace`` is also None here.
        if _backing is None and ctx is not None and ctx.workspace is not None:
            return ctx.workspace
        return _backing

    @_workspace.setter
    def _workspace(self, value):
        # v5 Phase 1.3 — WorkspaceReassign audit. Emitted ONLY when the root
        # actually changes AND verbose correlation is enabled. The caller
        # frame is captured via `traceback.extract_stack(limit=3)[-2]`
        # (cheap; avoids `inspect.stack()` which fills FrameInfo.code_context).
        # Lets us prove every workspace transition originated from a known
        # site (switch_role / _reassign_role_workspace / publish_to_ctx) and
        # not from a stray mutation. Off → behaviour unchanged.
        if _is_verbose_correlation():
            _old_backing = getattr(self, "_InferencerBase__workspace", None)
            _old_root = (
                getattr(_old_backing, "root", None)
                if _old_backing is not None
                else None
            )
            _new_root = getattr(value, "root", None) if value is not None else None
            if _old_root != _new_root:
                try:
                    _frame = traceback.extract_stack(limit=3)[-2]
                    _caller = {
                        "file": _frame.filename,
                        "line": _frame.lineno,
                        "func": _frame.name,
                    }
                except Exception:
                    _caller = {}
                try:
                    self.log_info(
                        {
                            "instance_id": getattr(self, "id", None),
                            "class": type(self).__name__,
                            "old_workspace": _old_root,
                            "new_workspace": _new_root,
                            "caller": _caller,
                        },
                        "WorkspaceReassign",
                    )
                except Exception:
                    # NEVER let an audit log abort the actual reassignment
                    pass
        object.__setattr__(self, "_InferencerBase__workspace", value)
        if value is not None:
            self._configure_for_workspace(value)
            self._propagate_workspace_to_children(value)
        # Auto-invalidate derived state when workspace changes.
        for attr in getattr(type(self), "_DERIVED_FROM_WORKSPACE", ()):
            self.__dict__.pop(attr, None)

    def _publish_workspace_to_ctx(self, child_ctx, workspace) -> None:
        """M7: publish a per-child workspace into the child context's handles so
        the child's ``_workspace`` getter resolves it from the context rather
        than via instance mutation. No-op when ``child_ctx`` is None (legacy)."""
        if child_ctx is not None and workspace is not None:
            child_ctx.handles.set("workspace_override", workspace)

    def _read_child_workspace(self, child_inf, slot):
        """M7: resolve a child's effective workspace for orchestrator-side reads —
        what the child itself resolves under ``ctx.child(slot)``: the published
        ``workspace_override`` for ``slot``, else its instance backing, else that
        child context's workspace. Never the bare ``child_inf._workspace``, which
        resolves against THIS node's context and so returns this node's own
        published workspace when it has one. Lets an orchestrator stop mutating
        ``child._workspace`` under a context (write-purity) while its own reads
        still resolve. Byte-identical without a context."""
        ctx = active_run_context()
        if ctx is None:
            return getattr(child_inf, "_workspace", None)
        child_ctx = ctx.child(slot)
        if isinstance(child_inf, InferencerBase):
            return child_inf._workspace_under(child_ctx)
        override = child_ctx.handles.get("workspace_override", None)
        return (
            override if override is not None else getattr(child_inf, "_workspace", None)
        )

    def _bind_rebuilt_child_ws(self, child_inf, slot: str, child_ws, *, owned) -> None:
        """Bind a dispatched child's workspace at ``slot``.

        Publishes the workspace into the child's run-context (tier-1, ephemeral —
        serves fresh-path ctx readers). An ``owned`` child (this call built it) also
        gets the durable instance backing (tier-2), so the binding survives a
        resume, where the active run-context is not guaranteed to be the dispatched
        child ctx across the child's recovery gate (see ``_workspace`` getter
        tiers): the object is the run, so the write races nothing. A borrowed child
        is a shared definition: under a host ctx it gets the publication only; with
        no ctx or under a legacy root it keeps the setter, as before.
        """
        if child_ws is None:
            return
        child_ws.ensure_dirs()
        self._publish_workspace_to_ctx(self._rc_child(slot), child_ws)
        if not isinstance(child_inf, InferencerBase):
            return
        ctx = active_run_context()
        if owned or ctx is None or ctx.legacy_mint:
            child_inf._workspace = child_ws

    # Subclasses may override this tuple to declare instance attributes that
    # are derived from ``_workspace`` and must be invalidated (removed from
    # ``__dict__``) whenever the workspace is reassigned.  This prevents
    # stale cached paths from surviving across consensus iterations or
    # workspace swaps.
    _DERIVED_FROM_WORKSPACE: tuple = ()

    @property
    def effective_cwd(self) -> str:
        """The derived operating directory for the agent.

        Priority: ``target_path`` > ``workspace.root`` > ``os.getcwd()``.
        Subclasses that spawn subprocesses or hand a cwd to an SDK client
        should read this property rather than ``target_path`` directly, so
        orchestrator-spawned children get the workspace-root fallback.
        """
        if self.target_path is not None:
            return self.target_path
        ws = self._workspace
        if ws is not None and hasattr(ws, "root"):
            return str(ws.root)
        return os.getcwd()

    @property
    def effective_allowed_paths(self) -> List[AllowedPath]:
        """All extra paths granted to this inferencer beyond its default scope.

        Combines explicit ``additional_allowed_paths`` with an auto-included
        entry for ``workspace.root`` (when a workspace is set), using
        ``PathAccess.ALL`` for the auto-included entry. Dedupes by resolved
        absolute path; silently drops malformed paths (e.g., embedded null
        bytes). Preserves order — user entries first, then auto-included
        workspace.root last.

        Subclasses with a native path-allowlist concept should read THIS
        property (not the raw ``additional_allowed_paths`` field) when
        translating into backend-specific flags. Subclasses without one
        simply ignore it.

        Scope note: this only auto-includes the leaf inferencer's OWN
        ``workspace.root`` (e.g., for a deeply-nested fix step, that's the
        fix step's narrow subtree). Cross-subtree reads (e.g., reading the
        prior aggregator's output that lives at a SIBLING subtree) require
        the orchestrator/executor to explicitly populate
        ``additional_allowed_paths`` with the topmost task root, which then
        flows through this property unchanged.

        The base implementation does NOT compare against any "current scope"
        (e.g., subprocess cwd) — that comparison is backend-specific. The
        redundant case where ``workspace.root`` happens to equal the
        subprocess cwd is harmless: any backend that enforces the allowlist
        will either ignore the duplicate (cwd is implicitly allowed) or
        dedupe it at translation time.
        """
        seen_resolved: set[str] = set()
        result: List[AllowedPath] = []

        # 1. User-provided entries first (preserve order; honor their access flags).
        for ap in self.additional_allowed_paths or []:
            if not ap or not ap.path:
                continue
            try:
                resolved = str(Path(ap.path).resolve(strict=False))
            except (OSError, ValueError):
                # Pathologically bad paths (e.g., embedded null bytes): drop.
                continue
            if resolved in seen_resolved:
                continue
            seen_resolved.add(resolved)
            result.append(ap)

        # 2. Auto-include workspace.root when set (always, regardless of cwd).
        # Redundant-with-cwd case is harmless — see docstring.
        ws = self._workspace
        ws_root_raw = (
            str(ws.root)
            if (ws is not None and hasattr(ws, "root") and ws.root)
            else None
        )
        if ws_root_raw:
            try:
                ws_root_resolved = str(Path(ws_root_raw).resolve(strict=False))
            except (OSError, ValueError):
                ws_root_resolved = None
            if ws_root_resolved and ws_root_resolved not in seen_resolved:
                seen_resolved.add(ws_root_resolved)
                result.append(AllowedPath(ws_root_resolved, access=PathAccess.ALL))

        return result

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

    @staticmethod
    def _module_dir_for_class(klass: type) -> Optional[Path]:
        """Resolve the on-disk directory of ``klass``'s module.

        Tries ``importlib.resources`` first (works under buck2 link-tree
        packaging), then falls back to the class's source-file directory
        (direct-Python runs). Returns None when neither resolves.
        """
        parent_pkg = (getattr(klass, "__module__", "") or "").rpartition(".")[0]
        if parent_pkg:
            try:
                from importlib import resources

                candidate = Path(str(resources.files(parent_pkg)))
                if candidate.is_dir():
                    return candidate.resolve()
            except (TypeError, OSError, ModuleNotFoundError):
                pass
        try:
            import inspect

            return Path(inspect.getfile(klass)).resolve().parent
        except (TypeError, OSError):
            return None

    @classmethod
    def _discover_inferencer_variable_roots(cls) -> List[Path]:
        """This inferencer's own ``prompt_templates`` roots, MRO-derived first.

        Walks ``cls.__mro__`` most-derived first; yields each class's
        ``<module dir>/prompt_templates`` when that folder holds a
        ``_variables/`` subtree. Dedups by resolved path. Returns [] when no
        class in the MRO ships variable extensions (the common case) so the
        caller keeps the shared manager unchanged.
        """
        roots: List[Path] = []
        seen: Set[Path] = set()
        for klass in cls.__mro__:
            module_dir = cls._module_dir_for_class(klass)
            if module_dir is None:
                continue
            candidate = (module_dir / "prompt_templates").resolve()
            if candidate in seen:
                continue
            if candidate.is_dir() and (candidate / "_variables").is_dir():
                seen.add(candidate)
                roots.append(candidate)
        return roots

    def _inferencer_variable_skip_keys(self) -> Set[str]:
        """Override keys explicitly disabled (mapped to ``False``).

        Such keys render base-only (extension roots skipped for that key).
        Meaningful only when ``enable_inferencer_variable_expansion`` is on.
        """
        return {k for k, v in self.inferencer_variable_overrides.items() if v is False}

    @staticmethod
    def _discover_inferencer_variable_keys(roots: List[Path]) -> Set[str]:
        """Dotted variable keys shipped under ``<root>/_variables/``.

        ``_variables/notes/large_file_writing.jinja2`` ->
        ``notes.large_file_writing``. Best-effort; used only to warn about
        typo'd override keys, never to gate resolution.
        """
        keys: Set[str] = set()
        for root in roots:
            vroot = root / "_variables"
            if not vroot.is_dir():
                continue
            for path in vroot.rglob("*"):
                if not path.is_file() or path.name.startswith("."):
                    continue
                rel = path.relative_to(vroot).with_suffix("")
                keys.add(".".join(rel.parts))
        return keys

    def _warn_unknown_override_keys(
        self, roots: List[Path], skip_keys: Set[str]
    ) -> None:
        """Warn once if a disabled override key isn't shipped under the roots.

        A typo'd key in ``inferencer_variable_overrides`` would otherwise
        silently do nothing -- the exact silent failure this framework fights.
        """
        unknown = skip_keys - self._discover_inferencer_variable_keys(roots)
        if unknown:
            _logger.warning(
                "inferencer_variable_overrides for %s names key(s) %s not "
                "found under its extension roots; they have no effect",
                type(self).__name__,
                sorted(unknown),
            )

    # Attrs whose values are semantically relevant when switching roles.
    # Subclasses extend this tuple in their own class body.
    _ROLE_RELEVANT_ATTRS: tuple = ()

    def switch_role(
        self,
        new_role: str,
        *,
        workspace=None,
        reset_session=True,
        _role_changes: Optional[RoleTransition] = None,
    ):
        """Transition this inferencer to a new semantic role.

        Centralises the workspace-swap + session-reset pattern that orchestrators
        (MFDual, PTI, ...) previously performed inline. The base layer handles
        workspace assignment and session reset; TemplatedInferencerBase extends
        with template attrs, which it hands over as ``_role_changes``.

        Under a host ctx nothing is written onto self: the workspace is published
        to the active (role) ctx, the session reset goes to this branch's slot,
        and the audit entry goes to the node's bounded ``provenance``. With no ctx
        or under a legacy root, the workspace setter and the ``_role_history``
        audit trail apply, as before.

        Part 2 (two-axis model): the deliverable flags (output_is_deliverable /
        is_deliverable_boundary) are RETIRED — role transitions carry only
        workspace + session state; promotion is role-based via ``promote_child``.

        Args:
            new_role: human-readable role name (e.g. 'fixer_inferencer').
            workspace: if not None, the role's workspace: published to the role
                ctx under a host ctx, else assigned via the _workspace property
                setter (triggers _configure_for_workspace cascade).
            reset_session: if True, calls self.reset_session() (when available).
        """
        import time

        ctx = active_run_context()
        host = ctx is not None and not ctx.legacy_mint
        # 1. Workspace assignment FIRST — triggers cascade
        if workspace is not None:
            if host:
                self._publish_workspace_to_ctx(ctx, workspace)
            else:
                self._workspace = workspace
        # 2. Session reset
        if reset_session and hasattr(self, "reset_session"):
            self.reset_session()
        # 3. Audit trail
        changes = {
            **({"workspace": str(workspace.root)} if workspace else {}),
            **(dict(_role_changes.changes) if _role_changes is not None else {}),
        }
        if host:
            self._record_role_provenance(ctx, new_role, changes)
            return
        history = getattr(self, "_role_history", None)
        if history is None:
            history = []
            object.__setattr__(self, "_role_history", history)
        history.append({"to_role": new_role, "at": time.time(), "changes": changes})

    def _record_role_provenance(self, ctx, new_role: str, changes: dict) -> None:
        """Append a host role switch to the node's provenance, keeping the last
        ``_ROLE_PROVENANCE_LIMIT`` entries. Only the changed attribute names are
        recorded (the values live in the node's typed ``RoleState``), so the entry
        is always JSON-serializable."""
        import time

        node = ctx.node(creator=(type(self).__qualname__, ctx.path))
        node.provenance.append(
            {
                "event": "switch_role",
                "to_role": new_role,
                "at": time.time(),
                "changed": sorted(changes),
            }
        )
        del node.provenance[: -self._ROLE_PROVENANCE_LIMIT]

    _ROLE_PROVENANCE_LIMIT: ClassVar[int] = 64

    # Subclasses may override this class attribute to declare which attrs
    # they manage workspace assignment for themselves (e.g., PTI's runtime
    # ``_setup_child_workflows`` claims planner/executor children with
    # iter_<N>-aware paths). Generic propagation skips those attrs to avoid
    # creating orphan dirs at construction time.
    _workspace_propagation_skip: frozenset = frozenset()

    # Attributes that cascade from a parent inferencer to its direct children
    # (recursively) when the child has not set an explicit value. Mirrors the
    # ``_propagate_workspace_to_children`` precedent: explicit child values win.
    # Each entry is either:
    #   - a str ``name`` — cascade when the child's value is ``None`` (unset);
    #   - a ``(name, should_propagate)`` tuple, where
    #     ``should_propagate(parent_value, child_value) -> bool``.
    # Subclasses may extend (idiomatic ClassVar inheritance), e.g.:
    #     _CASCADING_ATTRIBUTES = InferencerBase._CASCADING_ATTRIBUTES + [("x", cond)]
    _CASCADING_ATTRIBUTES: ClassVar[list] = ["debug_mode"]

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

        Skips ``functools.partial`` factories (e.g. BTA's ``worker_inferencers``);
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
                    field_name,
                    key,
                    type(child).__name__,
                    exc,
                )

        def _on_partial(partial, field_name, key):
            return None  # factories assigned at runtime; don't mutate

        self._for_each_child_inferencer(_on_instance, _on_partial)

    def _propagate_cascading_attributes(self) -> None:
        """Cascade ``_CASCADING_ATTRIBUTES`` to direct child inferencers, recursively.

        Uses ``_for_each_child_inferencer`` (the attrs-field walker that also
        backs ``_propagate_workspace_to_children`` and ``template_extra_feed``
        propagation) so children declared as attrs fields — e.g.
        ``ConversationalInferencer.base_inferencer`` — are discovered without
        each subclass overriding ``_iter_child_inferencers``.

        Semantics mirror workspace propagation: an explicit child value wins
        (only unset children inherit). For each attr the parent's own value
        must be set (non-``None``) to cascade. ``functools.partial`` factory
        children are intentionally NOT mutated (``on_partial`` returns ``None``,
        same as the workspace precedent) — they receive cascaded values at
        instantiation time via the YAML ``_``-prefix injectable, not here.

        Recursion: after assigning a child, the child's own
        ``_propagate_cascading_attributes`` runs so the value reaches
        grandchildren.
        """
        for name, parent_val, should_propagate in self._cascading_values():

            def _on_instance(
                child,
                field_name,
                key,
                _name=name,
                _cond=should_propagate,
                _pv=parent_val,
            ):
                self._cascade_one(child, _name, _pv, _cond)

            def _on_partial(partial, field_name, key):
                return None  # factory children inherit at instantiation, not here

            self._for_each_child_inferencer(_on_instance, _on_partial)

    def _cascading_values(self) -> Iterator[tuple]:
        """Yield ``(name, parent_value, should_propagate)`` for each
        ``_CASCADING_ATTRIBUTES`` entry this inferencer has set (non-``None``)."""
        for spec in self._CASCADING_ATTRIBUTES:
            if isinstance(spec, str):
                name = spec
                should_propagate = lambda _p, c: c is None
            else:
                name, should_propagate = spec
            parent_val = getattr(self, name, None)
            if parent_val is not None:
                yield name, parent_val, should_propagate

    @staticmethod
    def _cascade_one(child, name: str, parent_val: Any, should_propagate) -> None:
        if not isinstance(child, InferencerBase):
            return  # duck-typed callables don't participate
        if should_propagate(parent_val, getattr(child, name, None)):
            setattr(child, name, parent_val)
            child._propagate_cascading_attributes()  # reach grandchildren

    def _cascade_attributes_into(self, child) -> None:
        """Cascade this inferencer's ``_CASCADING_ATTRIBUTES`` into ``child`` (and its
        descendants) with the same explicit-value-wins rule, for a child the field
        walker does not reach (e.g. one built per call)."""
        for name, parent_val, should_propagate in self._cascading_values():
            self._cascade_one(child, name, parent_val, should_propagate)

    def enable_debug_mode(self):
        """Enable debug mode and cascade it to child inferencers.

        Overrides ``Debuggable.enable_debug_mode`` to additionally run the
        cascade, so a runtime toggle propagates to children. This is the
        reliable path for inferencers (e.g. ``ConversationalInferencer``)
        whose ``__attrs_post_init__`` does not chain ``super()`` and thus do
        not get the construction-time cascade.
        """
        super().enable_debug_mode()
        self._propagate_cascading_attributes()

    def disable_debug_mode(self):
        """Disable debug mode and cascade the (explicit ``False``) value.

        Note: cascade only reaches children whose value is still unset
        (``None``); children with an explicit value keep it (explicit wins).
        """
        super().disable_debug_mode()
        self._propagate_cascading_attributes()

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
            self._undefer_workspace_logger(workspace)
        elif isinstance(logger_val, dict):
            self._redirect_loggers_to_workspace(workspace)

    def _redirect_loggers_to_workspace(self, workspace):
        """Re-base workspace-derived loggers onto a new workspace root.

        Only loggers we tagged in ``_ws_log_relpaths`` (those created from a
        workspace via :meth:`_add_workspace_logger`) are re-pointed — each to its
        *own* recorded relative path under the new root, so it keeps its filename
        and untagged/user loggers are left untouched (no clobber). This is the
        setter-time path (legacy + reviewer/fixer ``switch_role``); the location
        isn't per-write here, so it rebuilds the immutable ``JsonLogger``.
        """
        import os

        from rich_python_utils.io_utils.json_io import JsonLogger

        relpaths = getattr(self, "_ws_log_relpaths", None)
        if not relpaths:
            return
        os.makedirs(workspace.logs_dir, exist_ok=True)
        new_loggers = {}
        changed = False
        for name, entry in self.logger.items():
            if isinstance(entry, tuple) and len(entry) == 2:
                logger_inst, config = entry
            else:
                logger_inst, config = entry, None
            rel = relpaths.get(name)
            if rel is not None and isinstance(logger_inst, JsonLogger):
                new_kwargs = dict(logger_inst.keywords)
                new_kwargs["file_path"] = os.path.join(workspace.root, rel)
                new_logger = JsonLogger(**new_kwargs)
                new_loggers[name] = (
                    (new_logger, config) if config is not None else new_logger
                )
                changed = True
            else:
                new_loggers[name] = entry
        if changed:
            self.logger = new_loggers

    # The construction recipe is a record, not a live child: fresh-id regeneration
    # reaches the children through their fields.
    _FRESH_ID_SKIP_TRAVERSE = Debuggable._FRESH_ID_SKIP_TRAVERSE | {"_init_recipe"}

    def __new__(cls, *args, **kwargs):
        # object.__new__ rejects forwarded constructor args. copy/deepcopy/pickle
        # call __new__ bare and then restore __dict__, carrying the source's recipe.
        inst = super().__new__(cls)
        inst.__dict__["_init_recipe"] = _capture_init_recipe(cls, args, kwargs)
        return inst

    def fresh_instance(self, **overrides: Any) -> "InferencerBase":
        """Return a new, unbound instance rebuilt from this instance's construction recipe.

        The recipe records constructor intent only (the arguments passed to
        ``cls(...)``), so nothing set after construction is carried: workspace and
        logger bindings, cache folders, sessions, cascaded attrs, or any other
        setattr. Nested inferencers, clone factories, lazy factories, partials and
        inferencer-bound methods are rebuilt recursively (children shared in the
        source stay shared in the copy), including inside plain ``dict`` /
        ``list`` / ``tuple`` / ``set`` values; every other value is passed by
        reference, container subclasses such as ``OrderedDict`` or named tuples
        included.
        Identity and parent links (``NON_CONFIG_ATTR_NAMES``) are dropped.
        ``overrides`` replace recipe entries by ``__init__`` parameter name.

        Raises:
            TypeError: no recipe (``__init__`` takes ``*args``/``**kwargs`` or the
                arguments could not be bound), or an override that is not an
                ``__init__`` parameter.
            ValueError: an inferencer is reachable from its own recipe.
        """
        return _InstanceRebuilder().build(self, overrides)

    def __attrs_post_init__(self):
        # Leaf-only guardrail: the lightweight output-guardrail judge drives leaf
        # RETRY/UPDATE recovery (StreamingInferencerBase). An orchestrator uses the
        # base _ainfer_recovery, which re-raises the guardrail's
        # OutputValidationExhaustedError on the first rejection instead of
        # recovering — so a judge cannot function there. Reject the misconfiguration
        # up front. (_is_orchestrator() is a class-level check, valid at construction;
        # note super()-last subclasses have already run their own setup — still fine,
        # the raise aborts construction.)
        if self.output_guardrail_inferencer is not None and self._is_orchestrator():
            raise ValueError(
                f"output_guardrail_inferencer is only supported on leaf inferencers, "
                f"not on orchestrators ({type(self).__name__}). Attach the guardrail "
                f"to the leaf inferencer(s) whose output should be judged."
            )
        self._validate_bta_inferencer_spec()

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
        async_overridden = (
            type(self)._ainfer_recovery is not InferencerBase._ainfer_recovery
        )
        sync_overridden = (
            type(self)._infer_recovery is not InferencerBase._infer_recovery
        )
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
                if (
                    fb.max_retry > 1
                    and cls_name not in InferencerBase._nested_retry_warned
                ):
                    InferencerBase._nested_retry_warned.add(cls_name)
                    _logger.warning(
                        f"{cls_name}: fallback_inferencer {type(fb).__name__} has "
                        f"max_retry={fb.max_retry}. This creates multiplicative retries "
                        f"(outer × inner). Consider max_retry=1 for fallbacks."
                    )
                    break

        super().__attrs_post_init__()

        # Cascade declared infrastructure attrs (e.g. debug_mode) to children
        # passed at construction time. Subclasses that override
        # __attrs_post_init__ MUST call super() for this to run; orchestrators
        # (Dual/BTA/PTI/LWI/MFDual/MultiFlow) do. NOTE: ConversationalInferencer
        # does NOT call super() today, so it relies on enable_debug_mode() /
        # the YAML cascade instead (see plan F1).
        self._propagate_cascading_attributes()

        # Independent path: a *bare* inferencer instance assigned (in Python) to a
        # registered factory field is wrapped in a fresh-copy factory, so each call yields
        # its own worker — matching the config/``LazyConfigFactory`` path. Runs after the
        # cascade so clones inherit cascaded attrs.
        self._wrap_bare_factory_inferencers()

    def _wrap_bare_factory_inferencers(self):
        """For each REGISTERED factory field (``metadata={"lazy_config_factory": True}``)
        holding a bare :class:`InferencerBase`, replace it with a
        :class:`_PrototypeCloneFactory` so each call returns a fresh, independent worker.

        Gated on the metadata only (NOT the ``*_factory`` name suffix), so the blast radius
        is exactly the registered fields (today: BTA ``worker_inferencers``). All other
        worker sources are untouched: a config value is already a ``LazyConfigFactory``/
        ``functools.partial`` at this point, and callables/lists/dicts are not
        ``InferencerBase`` — none matches, so no other worker path is affected.
        """
        import attr

        if not attr.has(type(self)):
            return
        for a in attr.fields(type(self)):
            if not a.metadata.get("lazy_config_factory", False):
                continue
            val = getattr(self, a.name, None)
            if isinstance(val, InferencerBase):
                setattr(self, a.name, _PrototypeCloneFactory(val))

    def _for_each_child_inferencer(self, on_instance, on_partial):
        """Walk attrs fields and invoke callbacks for child inferencers.

        Fields whose metadata sets ``inferencer_template`` hold templates for
        per-call instances, not live children, and are skipped.

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
        import functools

        import attr

        attr_name = TEMPLATE_EXTRA_FEED_ATTR

        for field in attr.fields(type(self)):
            if field.metadata.get("inferencer_template", False):
                continue
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
                    elif callable(v) and isinstance(getattr(v, attr_name, None), dict):
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

            elif callable(value) and isinstance(getattr(value, attr_name, None), dict):
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

    def _is_orchestrator(self) -> bool:
        """True iff this node owns child inferencers (a flow/orchestrator).

        Detected via the standard override idiom: a genuine orchestrator overrides
        AT LEAST ONE child-iteration primitive. EITHER (not both) suffices — PTI
        overrides only ``_iter_child_slots`` while BTA/MFI override
        ``_iter_child_inferencers`` — so an AND here would wrongly exclude PTI from
        recovery-suppression. This matches the sibling is-orchestrator check used by
        the recovery-chain builder (see ``_ainfer_recovery`` wiring). Used by
        ``_ainfer_recovery`` (U4-A) to re-raise instead of blindly re-running the
        whole subtree. (A leaf overrides neither → False; streaming leaves override
        ``_ainfer_recovery`` itself so this is not consulted for them.)
        """
        return (
            type(self)._iter_child_inferencers
            is not InferencerBase._iter_child_inferencers
            or type(self)._iter_child_slots is not InferencerBase._iter_child_slots
        )

    async def preflight(self) -> None:
        """Additive readiness check (U2b). Default no-op.

        Overrides raise (or the resolver they call raises) when a required runtime
        hard-dependency is missing; :meth:`preflight_all` aggregates the failures
        into ONE actionable startup error before any inference work begins.
        """
        return None

    async def preflight_all(self) -> list[str]:
        """Walk self + all descendants, run each :meth:`preflight`, aggregate failures.

        Reuses the cycle-safe :meth:`_collect_all_descendant_inferencers` walker so
        a missing dependency deep in the topology surfaces up front (U2b).
        """
        problems: list[str] = []
        for inf in self._collect_all_descendant_inferencers():
            try:
                await inf.preflight()
            except Exception as exc:  # noqa: BLE001 — aggregate, don't abort the walk
                problems.append(f"{type(inf).__qualname__}: {exc}")
        return problems

    def _iter_child_slots(self) -> Iterator[tuple]:
        """§9.3/N-Major1: slot-aware child iteration — yield ``(slot, child)`` so
        lifecycle ops (``pre_retry``/``reset_session``/``aconnect``) can bind
        ``_active_ctx = ctx.child(slot)`` and target each child's OWN Tier-3
        handles. Default derives positional slots from ``_iter_child_inferencers``;
        orchestrators override to match the deterministic ``ctx.child(slot)`` they
        use in ``_ainfer`` (§2.7)."""
        for i, child in enumerate(self._iter_child_inferencers()):
            yield (f"child_{i}", child)

    @contextlib.contextmanager
    def _with_child_ctx(self, slot: str):
        """Bind ``_active_ctx = active.child(slot)`` for the body so a child
        lifecycle op resolves the child's slot. No-op (byte-identical) when no
        context is active."""
        ctx = active_run_context()
        if ctx is None:
            yield
            return
        token = enter_run(ctx.child(slot))
        try:
            yield
        finally:
            exit_run(token)

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
            _logger.warning("%s._pre_retry raised: %s", type(self).__name__, exc)
        for slot, child in self._iter_child_slots():
            if child is None or id(child) in _seen:
                continue
            try:
                # §9.3/N-Major1: bind the child's slot so its session-reset targets
                # its OWN Tier-3 handle (no-op without an active ctx -> legacy).
                with self._with_child_ctx(slot):
                    await child.pre_retry(attempt, exception, _seen=_seen)
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                _logger.warning(
                    "%s.pre_retry (child of %s) raised: %s",
                    type(child).__name__,
                    type(self).__name__,
                    exc,
                )

    async def _pre_retry(self, attempt: int, exception: BaseException) -> None:
        """Own-state cleanup before the next retry — default = U3c archive-and-clean.

        On a retry-after-failure (NOT an intended resume) this archives the node's
        stale deliverable STATE (``outputs/`` + ``checkpoints/`` + the
        ``.*_completed`` markers in ``artifacts/``) into ``root/.attempts/<n>/`` so
        the retry starts on a clean workspace — otherwise a leftover
        ``outputs/output.md`` can make a re-run no-op on its own prior output.
        ``logs/`` is kept LIVE (its append handle is open). Best-effort — never
        blocks the retry. Subclasses (e.g. StreamingInferencerBase) may override
        for their own recovery and skip this.

        Note: only fires on the ASYNC retry path (max_retry >= 2). Sync
        `pre_retry` propagation is not currently wired.
        """
        self._archive_workspace_for_retry(attempt)

    def _archive_workspace_for_retry(self, attempt: int) -> None:
        """Move stale deliverable STATE into ``root/.attempts/<attempt>/`` (U3c).

        Archived: ``outputs/`` + ``checkpoints/`` + the ``.*_completed`` completion
        markers in ``artifacts/`` (they gate re-runs). ``logs/`` is left live (open
        append handle). Skipped on an intended resume (``resume_with_saved_results``)
        or when there is no workspace on disk. Swallows all errors so it never
        blocks a retry.
        """
        try:
            if getattr(self, "resume_with_saved_results", False):
                return
            ws = self._workspace
            if ws is None or not getattr(ws, "root", "") or not os.path.isdir(ws.root):
                return
            import shutil

            attempt_dir = os.path.join(ws.root, ".attempts", str(attempt))
            moved = False
            for state_dir, keep in (
                (ws.outputs_dir, frozenset()),
                (ws.checkpoints_dir, self._retry_archive_keeps()),
            ):
                if os.path.isdir(state_dir):
                    os.makedirs(attempt_dir, exist_ok=True)
                    target = os.path.join(attempt_dir, os.path.basename(state_dir))
                    _move_into_archive(state_dir, target, keep)
                    moved = True
            art = ws.artifacts_dir
            if os.path.isdir(art):
                for entry in os.listdir(art):
                    if entry.startswith(".") and entry.endswith("_completed"):
                        marker_dst = os.path.join(attempt_dir, "artifacts")
                        os.makedirs(marker_dst, exist_ok=True)
                        shutil.move(
                            os.path.join(art, entry),
                            os.path.join(marker_dst, entry),
                        )
                        moved = True
            if moved:
                ws.ensure_dirs()  # recreate the emptied outputs/ + checkpoints/
        except Exception:  # noqa: BLE001 — archival must never block a retry
            _logger.debug("U3c archive-and-clean skipped", exc_info=True)

    def _retry_archive_keeps(self) -> FrozenSet[str]:
        """Entries of ``checkpoints/`` a retry's archive leaves in place: state that
        belongs to the whole call, not to the failed attempt (none by default)."""
        return frozenset()

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

    def _ensure_ctx_workspace_logger(self) -> None:
        """Un-defer the session logger for a context-dispatched inferencer.

        An inferencer constructed without a workspace defers its ``_workspace``
        JsonLogger (:meth:`_resolve_auto_logger` sets ``_logger_awaiting_workspace``
        and returns before creating it). The only runtime un-defer trigger is the
        ``_workspace`` *setter* (:meth:`_configure_for_workspace`). Under M7 a
        ctx-dispatched child's workspace arrives via ``_publish_workspace_to_ctx``
        (the setter is legacy-only), so a deferred logger would never be created
        and the node would emit no session log despite running (the top BTA
        aggregator and the outer Dual review/fix leaves hit exactly this).

        Called from ``__ainfer_single_impl`` / ``__infer_single_impl`` — the
        single point both the manual (``InferencerBase.ainfer``/``infer``) and the
        ``@bridge_entrypoint`` (CLI / streaming leaf overrides that bypass
        ``InferencerBase.ainfer``) ctx-install paths converge on, *after* the
        bridge is installed (so the ctx-aware ``_workspace`` getter can resolve
        the published ``workspace_override`` / ``ctx.workspace``) and before any
        ``InferenceInput`` logging. No-op when there is no active context (legacy
        → byte-identical), the logger is not deferred (eager / already-created /
        ``switch_role`` already un-deferred), or the context published no
        workspace (an unused placeholder slot).

        Mutates only instance-local logger state — never ``self._workspace`` — so
        it does not reintroduce the cross-branch instance mutation M7 removed.
        :meth:`_undefer_workspace_logger` re-checks the flag under a lock and
        replaces (never mutates) the logger dicts, so concurrent ``parallel_infer``
        threads and gathered branches neither create duplicate loggers nor break
        a concurrent ``log`` iterating ``self.logger``; per-branch write routing
        is handled by :meth:`_log_path_override`.
        """
        from agent_foundation.common.inferencers.run_context import active_run_context

        ctx = active_run_context()
        if ctx is None:
            return
        awaiting = getattr(self, "_logger_awaiting_workspace", False)
        ws = self._workspace
        if not awaiting:
            _logger.debug(
                "[%s] _ensure_ctx_workspace_logger: NOT deferred (_logger_awaiting_workspace=%s, ws=%s)",
                type(self).__name__,
                awaiting,
                "present" if ws else "None",
            )
            return
        if ws is None:
            _logger.debug(
                "[%s] _ensure_ctx_workspace_logger: deferred but no workspace (ctx.path=%s)",
                type(self).__name__,
                getattr(ctx, "path", "?"),
            )
            return
        _logger.debug(
            "[%s] _ensure_ctx_workspace_logger: UN-DEFERRING workspace logger (ws=%s, ctx.path=%s)",
            type(self).__name__,
            getattr(ws, "root", "?"),
            getattr(ctx, "path", "?"),
        )
        self._undefer_workspace_logger(ws)

    def _undefer_workspace_logger(self, workspace) -> None:
        """Create the deferred workspace logger exactly once, even when
        ``parallel_infer`` threads race to un-defer it."""
        with _WORKSPACE_LOGGER_LOCK:
            if not getattr(self, "_logger_awaiting_workspace", False):
                return
            self._logger_awaiting_workspace = False
            self._add_workspace_logger(workspace)

    def _tag_ws_log_relpath(self, name, file_path, workspace):
        """Record a workspace-derived logger's path *relative to the workspace
        root* (e.g. ``logs/session.jsonl``).

        This static relpath is the half of the log path that never changes; the
        dynamic half (the workspace root) is resolved later — at setter time by
        :meth:`_redirect_loggers_to_workspace`, or per-write by
        :meth:`_log_path_override` against the live run-context. Lets the logger
        follow the inferencer to whatever workspace is effective without
        hardcoding a filename or mutating the (possibly shared) logger.
        """
        self._ws_log_relpaths = {
            **(getattr(self, "_ws_log_relpaths", None) or {}),
            name: os.path.relpath(file_path, workspace.root),
        }

    def _add_workspace_logger(self, workspace):
        """Create a JsonLogger at workspace.logs_dir and add it to self.logger."""
        from rich_python_utils.common_objects.debuggable import LoggerConfig
        from rich_python_utils.io_utils.json_io import JsonLogger, SpaceExtMode

        log_dir = workspace.logs_dir
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, "session.jsonl")
        json_entry = (
            JsonLogger(
                file_path=file_path,
                append=True,
                space_ext_mode=SpaceExtMode.MOVE,
            ),
            LoggerConfig(pass_item_key_as="parts_key_path_root"),
        )
        if isinstance(self.logger, dict):
            # Post-normalize (deferred → upgrade): replace the dicts rather than
            # mutate them, so a concurrent ``log`` iterating the old dict is safe.
            self.logger = {**self.logger, "_workspace": json_entry[0]}
            self._resolved_logger_configs = {
                **(getattr(self, "_resolved_logger_configs", None) or {}),
                "_workspace": json_entry[1],
            }
        else:
            # Pre-normalize (logger == "auto", e.g. constructed with a workspace):
            # hand _normalize_loggers a dict so the "_workspace" name and inline
            # config survive. The old list form became an auto-named "logger_N"
            # that the relpath tag below could never match.
            self.logger = {"_workspace": json_entry}
        self._tag_ws_log_relpath("_workspace", file_path, workspace)

    def _log_path_override(self, logger_name, logger):
        """Per-write override: route a workspace-derived logger to the *live*
        run-context's workspace, re-based from the recorded relpath.

        Returns ``None`` (no override → baked path used) when there is no active
        context, the logger isn't workspace-derived, or the path already matches.
        So legacy/no-ctx runs and loggers already pointed correctly (e.g. a
        reviewer/fixer whose instance workspace was set via the setter) stay
        byte-identical, and the override fires only for leaves whose deep
        workspace was published to the context (never set on the instance).
        Never mutates the logger — the path is returned for use as a call arg.
        """
        if active_run_context() is None:
            return None
        relpaths = getattr(self, "_ws_log_relpaths", None)
        if not relpaths:
            return None
        rel = relpaths.get(logger_name)
        if rel is None:
            return None
        ws = self._workspace
        if ws is None:
            return None
        candidate = os.path.join(ws.root, rel)
        return candidate if getattr(logger, "file_path", None) != candidate else None

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
        """Finalize outputs after inference. Orchestrators override this to promote
        their canonical child; the base behaviour is ``_finalize_leaf_output``."""
        return self._finalize_leaf_output(response)

    def _finalize_leaf_output(self, response: Any) -> Any:
        """Finalize a leaf's outputs: write the <Response> summary, emit manifest.

        Part 2 (two-axis model): ``outputs/`` IS the deliverable set — there is no
        move to ``final_deliverables/`` (retired). Everything the agent wrote to
        ``outputs/`` is the deliverable as-is; the framework only materializes the
        ``<Response>``-extracted summary at ``output_path`` when the agent didn't
        write it itself (the no-local-access sole-output case). Orchestrators promote
        a selected child's ``outputs/`` up via ``promote_child`` /
        ``_symlink_child_output``.

        Sequence:
        1. If the agent already wrote a non-empty ``output_path``, keep it; otherwise
           extract ``<Response>`` and write it to ``output_path`` as a reference summary.
        2. Emit ``output_manifest.json`` (to ``artifacts/``) when ``output_manifest_index``.
        """
        resolved = self.resolve_output_path()
        _orig_response = (
            response  # capture before Step 1 may reassign response to the file body
        )

        # -- Step 1: Write <Response> summary if the agent didn't write output_path --
        # Part 2 (two-axis): outputs/ IS the deliverable set — no move to
        # final_deliverables/ (retired). Keep a non-empty agent-written output_path
        # as-is; otherwise materialize the <Response> as a reference summary.
        if resolved and os.path.isabs(resolved):
            if os.path.isfile(resolved) and os.path.getsize(resolved) > 0:
                pass
            else:
                from agent_foundation.common.response_parsers import extract_delimited

                cleaned = extract_delimited(str(response))
                if cleaned is None:
                    _logger.warning(
                        "extract_delimited returned None for output_path=%s; "
                        "writing raw response as fallback (likely prompt-template echo).",
                        resolved,
                    )
                    cleaned = str(response)
                os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
                with open(resolved, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                response = cleaned

        # -- Step 2: Emit manifest (to artifacts/) --
        if resolved and os.path.isfile(resolved) and self.output_manifest_index:
            self._emit_output_manifest(resolved)

        # -- Step 3: Generic declared-content format check (Fix 3 / G4 observability) --
        if self.expected_extraction:
            self._run_expected_extraction(_orig_response)

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
            os.symlink(
                os.path.abspath(src), dst, target_is_directory=os.path.isdir(src)
            )
        except (OSError, NotImplementedError):
            if os.path.isdir(src):
                _shutil.copytree(src, dst)
            else:
                _shutil.copy2(src, dst)

    @staticmethod
    def _replace_with_symlink(src: str, dst: str) -> None:
        """Atomically point the file *dst* at *src*, replacing any existing entry.

        The link (or copy, where symlinks are unsupported) is built at a
        temporary sibling of *dst* and ``os.replace``d onto it, so readers
        never observe a missing or half-written file.
        """
        import shutil as _shutil

        dst_dir = os.path.dirname(dst) or "."
        os.makedirs(dst_dir, exist_ok=True)
        tmp = os.path.join(
            dst_dir, f".{os.path.basename(dst)}.{uuid.uuid4().hex[:8]}.tmp"
        )
        try:
            try:
                os.symlink(os.path.abspath(src), tmp)
            except (OSError, NotImplementedError):
                _shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        finally:
            if os.path.lexists(tmp):
                os.unlink(tmp)

    def _symlink_child_output(
        self, child_workspace, child_output_name=None, *, replace_canonical=False
    ) -> None:
        """Symlink a canonical child's output and deliverables as own.

        Used by orchestrator ``_finalize_output`` overrides to surface
        the canonical child's work as the orchestrator's own output.

        For outputs: finds the child's file using ``child_output_name``
        (the filename the child wrote), then symlinks it into the
        orchestrator's own ``outputs/`` under ``self.output_path``
        (the filename the orchestrator declares).

        When ``child_output_name`` is ``None``, assumes the child uses
        the same filename as the orchestrator (``self.output_path``).

        An existing own output is kept unless ``replace_canonical`` is true,
        in which case it is atomically replaced by the child's.

        For sidecars: promotes each non-bookkeeping entry in the child's
        ``outputs/`` into the orchestrator's own ``outputs/`` (skip-if-exists).
        """
        ws = self._workspace
        if ws is None or child_workspace is None:
            return

        from agent_foundation.common.inferencers.inferencer_workspace import (
            DEFAULT_OUTPUT_FILENAME,
        )

        own_name = self.output_path or DEFAULT_OUTPUT_FILENAME
        child_name = child_output_name or own_name

        # Symlink the output file (check deliverables first — leaf may have moved it)
        child_deliv = getattr(child_workspace, "deliverable_path", lambda x: None)(
            child_name
        )
        child_output = (
            child_workspace.output_path(child_name)
            if hasattr(child_workspace, "output_path")
            else None
        )
        src = (
            child_deliv
            if (child_deliv and os.path.isfile(child_deliv))
            else child_output
        )
        if src and os.path.isfile(src):
            own_output = ws.output_path(own_name)
            if replace_canonical:
                self._replace_with_symlink(src, own_output)
            else:
                self._symlink_or_copy(src, own_output)

        # Part 2: the final_deliverables/ symlink loop is RETIRED — deliverables now
        # live directly in outputs/ (deliverables_dir collapses to outputs_dir), so
        # the child's whole outputs/ is promoted by the denylisted block below (which
        # correctly skips bookkeeping; the old FD loop did not).

        # Fix 2 / Part 2: promote the child's outputs/ deliverables up to the parent's
        # outputs/ — minus the bookkeeping denylist and the canonical output (already
        # promoted above). Generic, format-agnostic, skip-if-exists. (proposals.json
        # itself is derived at the task level by the executor, Fix 3 — not here.)
        child_out = getattr(child_workspace, "outputs_dir", None)
        own_out = getattr(ws, "outputs_dir", None)
        if child_out and own_out and os.path.isdir(child_out):
            for entry in os.listdir(child_out):
                if _is_bookkeeping_sidecar(entry):
                    continue
                if entry in (own_name, child_name):
                    continue
                self._symlink_or_copy(
                    os.path.join(child_out, entry),
                    os.path.join(own_out, entry),
                )

    def promote_child(
        self,
        child_workspace,
        *,
        role: str = "canonical",
        child_output_name: "Optional[str]" = None,
    ) -> None:
        """Role-based promotion of a selected child's ``outputs/`` up to this
        node's ``outputs/`` (Part 2 two-axis model).

        The orchestrator selects the canonical child by role (Dual → winner,
        BTA → aggregator, MFDual → fixer, LWI → last step) and calls this to
        surface that child's deliverables as its own. Workers are consumed
        inputs and are NOT promoted; ``artifacts/`` never promotes.

        This is the role-named entry point over the ``_symlink_child_output``
        mechanism, which promotes the canonical output plus the child's loose
        ``outputs/`` sidecars (minus the bookkeeping denylist), skip-if-exists.
        """
        self._symlink_child_output(child_workspace, child_output_name)

    def _promote_child_checkpoints(
        self, child: "InferencerBase", slot: str, *, parent_ws
    ) -> None:
        """Publish a child's ``checkpoint_scope="parent"`` extractions up into
        this parent's ``checkpoints/<child>/``.

        The ``checkpoints/``-dir counterpart of :meth:`_symlink_child_output`. A
        child cannot push state to its parent (``RunContext`` is strictly
        parent->child), so the parent PULLS: for each of the child's
        ``expected_extraction`` entries marked ``checkpoint_scope: "parent"``,
        copy the file the extraction register already wrote to the child's
        ``outputs/<persist_to>`` into this parent's
        ``checkpoints/<child>/<persist_to>``. The ``<child>`` segment is the
        child workspace's own directory name, so the promoted file lands beside
        that node's other checkpoints (e.g. its ``__graph_expansion__`` record).
        ``slot`` is the child's run-context slot, the one its call ran under; the
        child's workspace is read through it (:meth:`_read_child_workspace`).
        ``parent_ws`` is the workspace of this parent's current call (BTA passes
        its call record's), never re-resolved from whichever ctx is active.

        Atomic tmp->rename so a resumer never observes a half-written file;
        best-effort (a failed copy is logged, never gates), mirroring
        :meth:`_persist_extracted_block`. Call this right after the child
        completes and BEFORE any downstream node can fail, so the promoted state
        is durable for resume. Parent-driven, so inherently concurrency-safe.
        """
        import shutil

        child_ws = self._read_child_workspace(child, slot)
        if parent_ws is None or child_ws is None:
            return
        child_name = os.path.basename(os.path.normpath(child_ws.root))
        if not child_name:
            return
        for _spec in getattr(child, "expected_extraction", None) or ():
            if not isinstance(_spec, dict):
                continue
            if _spec.get("checkpoint_scope") != "parent":
                continue
            dest = _spec.get("persist_to")
            if (
                not dest
                or os.path.isabs(dest)
                or ".." in dest.replace("\\", "/").split("/")
            ):
                continue
            src = child_ws.output_path(dest)
            if not os.path.isfile(src):
                continue
            target = parent_ws.checkpoint_path(os.path.join(child_name, dest))
            try:
                os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
                tmp = f"{target}.tmp"
                shutil.copyfile(src, tmp)
                os.replace(tmp, target)
                _logger.info(
                    "promoted child checkpoint (checkpoint_scope=parent) %s -> %s",
                    src,
                    target,
                )
            except OSError as e:
                _logger.warning(
                    "failed to promote child checkpoint %s -> %s: %s",
                    src,
                    target,
                    e,
                )

    def _run_expected_extraction(self, response: Any = None) -> None:
        """Self-validation for the unified ``expected_extraction`` registry (opt-in).

        For each declared fence, extract it from ITS channel — the deliverable file
        (``source="output"``) or this inferencer's stdout response (``source="response"``)
        — and log an ``extraction_issue`` when it is missing / not ``json.loads``-able.
        Observability ONLY (never raises/gates), generic-FORMAT only (ZERO domain
        knowledge): content fences are repaired in-loop by the reviewer/fixer and
        validated by the tool at finalization; control fences fall back to defaults in
        their cross-node consumer (``flow_parsers.parse_*_tag``). This is emitter-side
        self-validation — it does NOT consume/route control fences. Mirrors the
        overridable ``_guardrail_*`` hook pattern.
        """
        from agent_foundation.common.inferencers.flow_parsers import _extract_json_block

        _file_text = None  # deliverable file, read at most once (lazy)
        _file_read = False
        for _spec in self.expected_extraction or ():
            if not isinstance(_spec, dict):
                continue
            _label = _spec.get("label")
            if not _label:
                continue
            _source = _spec.get("source") or "output"
            _kind = _spec.get("kind") or "content"
            # ``fallback_to_source`` (content only; forced off for control): when the
            # fence can't be extracted but the source text IS present, the whole source
            # counts as the content — so a missing fence is not an issue here and is not
            # flagged. Control fences always flag (a parent orchestrator consumes them
            # cross-node and cannot fall back to raw text).
            _fallback = _kind != "control" and bool(_spec.get("fallback_to_source"))
            if _source == "response":
                _text = self._extraction_response_text(response)
            else:
                if not _file_read:
                    _file_text = self._extraction_output_text()
                    _file_read = True
                _text = _file_text
            if _text is None:
                # No artifact on this channel to check (e.g. no file written, or no
                # response captured). A missing expected deliverable is surfaced
                # separately by _guardrail_output_text; don't double-warn here.
                continue
            # ``_extract_json_block`` returns the parsed dict, or None when the fence
            # is missing OR not valid JSON — exactly the generic-format signal we want.
            # ``_fallback`` suppresses the warning: the present source text stands in as
            # the content when the specific fence is absent.
            _parsed = _extract_json_block(_text, _label)
            if _parsed is None:
                if not _fallback:
                    _logger.warning(
                        "extraction_issue: declared %s fence ```json %s``` is missing or not "
                        "valid JSON in this inferencer's %s — reviewer/fixer repair content "
                        "in-loop (tool validates at finalization); control fences fall back "
                        "to defaults in their consumer.",
                        _kind,
                        _label,
                        _source,
                    )
                # Nothing extracted -> nothing to emit; ``persist_to`` no-ops.
                continue
            # Emit the structured block as a sidecar when the registry asks for it.
            # Never gates: a bad destination or a failed write only logs.
            _dest = _spec.get("persist_to")
            if _dest:
                self._persist_extracted_block(_label, _parsed, _dest)

    # ------------------------------------------------------------------
    # Registered-block accessors (stateless: text in -> object out)
    # ------------------------------------------------------------------
    #
    # Deliberately NOT cached on the instance. Leaf inferencers are shared and
    # re-roled at runtime (a non-winning proposer flow becomes a reviewer, the
    # winner becomes the fixer), so a ``self._last_block[label]`` would go stale
    # across roles/rounds and silently serve another node's data. Taking the text
    # as an argument also solves emitter-vs-consumer: a control fence is consumed
    # by a PARENT holding the child's returned string, not by the emitter itself.

    def block_text(self, text: Any, label: str) -> "Optional[str]":
        """Raw body of the ```json <label>``` fence in ``text``, or None."""
        import re as _re

        from agent_foundation.common.inferencers.flow_parsers import (
            _JSON_FENCE_TEMPLATE,
        )

        if not text or not label:
            return None
        m = _re.search(_JSON_FENCE_TEMPLATE.format(label=_re.escape(label)), str(text))
        return m.group(1) if m else None

    def block_dict(self, text: Any, label: str) -> "Optional[dict]":
        """Parsed dict for the ```json <label>``` fence in ``text``, or None."""
        from agent_foundation.common.inferencers.flow_parsers import _extract_json_block

        if not text or not label:
            return None
        return _extract_json_block(str(text), label)

    def block_parsed(self, text: Any, label: str) -> Any:
        """``block_dict`` run through the label's registered ``BLOCK_PARSERS`` entry.

        Returns the raw dict when no parser is registered, and None when the fence
        is absent/invalid. The parser is a pure transform (dict -> domain object)
        supplied by the subclass; it must not depend on instance mutation.
        """
        parsed = self.block_dict(text, label)
        if parsed is None:
            return None
        parser = (type(self).BLOCK_PARSERS or {}).get(label)
        if isinstance(parser, str):
            # Registered by method name — the common case, since a domain
            # transform usually reads instance config (e.g. worker_query_fields).
            parser = getattr(self, parser, None)
        return parser(parsed) if callable(parser) else parsed

    def _persist_extracted_block(self, label: str, parsed: dict, dest: str) -> None:
        """Write an extracted block to this node's own ``outputs/<dest>`` (best-effort).

        Confined to the node's own outputs dir: bare filenames only, so a registry
        entry can never write outside its workspace. Atomic tmp->rename so a reader
        never observes a half-written file. Never raises — persistence is emission,
        not a gate.
        """
        import json as _json

        if (
            not dest
            or os.path.isabs(dest)
            or ".." in dest.replace("\\", "/").split("/")
        ):
            _logger.warning(
                "expected_extraction[%s]: ignoring unsafe persist_to=%r "
                "(bare filename required)",
                label,
                dest,
            )
            return
        ws = getattr(self, "_workspace", None)
        if ws is None or not hasattr(ws, "output_path"):
            return
        try:
            target = ws.output_path(dest)
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            tmp = f"{target}.tmp"
            # utf-8 + ensure_ascii=False: LLM-generated content carries Unicode
            # (arrows, em-dashes) that would raise under a cp1252 default.
            with open(tmp, "w", encoding="utf-8") as _f:
                _json.dump(parsed, _f, indent=2, ensure_ascii=False)
            os.replace(tmp, target)
            _logger.info(
                "expected_extraction[%s]: persisted block to %s", label, target
            )
        except (OSError, TypeError, ValueError) as e:
            _logger.warning(
                "expected_extraction[%s]: failed to persist block to %r: %s",
                label,
                dest,
                e,
            )

    def _extraction_output_text(self) -> "Optional[str]":
        """Deliverable-file content for ``expected_extraction`` (source='output'), or
        ``None`` when no non-empty ``output_path`` file exists."""
        try:
            resolved = self.resolve_output_path()
        except Exception:
            return None
        if resolved and os.path.isabs(resolved) and os.path.isfile(resolved):
            try:
                with open(resolved, "r", encoding="utf-8") as _f:
                    return _f.read()
            except OSError:
                return None
        return None

    def _extraction_response_text(self, response: Any) -> "Optional[str]":
        """This inferencer's stdout response text for ``expected_extraction``
        (source='response'): the raw response envelope (with any ``<Response>`` + fences),
        or ``None`` when unavailable. Prefers ``raw_output`` (the full stdout that carries
        the fences) over the parsed ``output``; robust to dict / wrapped-response shapes."""
        if response is None:
            return None
        for _getter in (
            lambda r: r.get("raw_output") if isinstance(r, dict) else None,
            lambda r: r.get("output") if isinstance(r, dict) else None,
            lambda r: getattr(r, "raw_output", None),
        ):
            try:
                _v = _getter(response)
            except Exception:
                _v = None
            if isinstance(_v, str) and _v:
                return _v
        try:
            return str(response) or None
        except Exception:
            return None

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
                                    contributors.append(
                                        {
                                            "category": cat_name.lower(),
                                            "path": os.path.realpath(fpath),
                                            "size_bytes": os.path.getsize(fpath),
                                        }
                                    )
                    elif entry.endswith(".jsonl") and os.path.isfile(entry_path):
                        contributors.append(
                            {
                                "category": "session_log",
                                "path": os.path.realpath(entry_path),
                                "size_bytes": os.path.getsize(entry_path),
                            }
                        )

        for src_info in getattr(self, "_consumed_upstream_paths", []):
            p = str(src_info)
            if os.path.isfile(p):
                contributors.append(
                    {
                        "category": "upstream_artifact",
                        "path": os.path.realpath(p),
                    }
                )

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

        # Part 2 (Axis A): the manifest is framework BOOKKEEPING → artifacts/ (not
        # outputs/, the deliverable set). Write-only; no prod reader assumes outputs/.
        _manifest_name = (
            os.path.splitext(os.path.basename(output_path))[0] + "_manifest.json"
        )
        _manifest_dir = getattr(ws, "artifacts_dir", None) or os.path.dirname(
            output_path
        )
        os.makedirs(_manifest_dir, exist_ok=True)
        manifest_path = os.path.join(_manifest_dir, _manifest_name)
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

    async def _atry_resume_from_cache(
        self, inference_input, inference_config=None, **kwargs
    ):
        """Async hook — mirrors ``_try_resume_from_cache`` for ``_ainfer_single``.

        Base implementation delegates to the sync version.  Streaming inferencers
        override to ``await self._ainfer(augmented)`` for partial-cache recovery.
        """
        return self._try_resume_from_cache(inference_input, inference_config, **kwargs)

    # -- Inference pipeline -------------------------------------------------

    def _prepare_prompt(self, inference_input: Any, extra_feed: Optional[dict]) -> Any:
        """Preprocess, then render (opt-in: only when template_manager is set).

        ``extra_feed`` is forwarded only when given, so ``_render_prompt``
        overrides that don't declare it keep working.
        """
        if self.input_preprocessor is not None:
            inference_input = self.input_preprocessor(inference_input)
        if extra_feed is not None:
            return self._render_prompt(inference_input, extra_feed=extra_feed)
        return self._render_prompt(inference_input)

    def _infer_single(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        # ── Phase 1 (leaf-owned template rendering): extract per-call render
        # parameters from _inference_args BEFORE forwarding to _infer().
        # `extra_feed` is consumed by _render_prompt; `render_only` short-circuits
        # the LLM call. Both are KEYWORD-ONLY by convention here (orchestrators
        # pass them by name). Extracting them prevents leakage to _infer() which
        # would TypeError on unrecognized kwargs. `prepared_input` marks an input
        # that is already the final prompt (no preprocessing, no rendering).
        with open_invocation(self, "infer") as frame:
            _extra_feed = _inference_args.pop("extra_feed", None)
            _render_only = _inference_args.pop("render_only", False)
            _prepared_input = _inference_args.pop("prepared_input", False)
            self._pop_invocation_keywords(frame, _inference_args)
            self._init_call_state(inference_input)
            runs_provider = self._runs_provider(_render_only)
            if runs_provider:
                _inference_args = self._prepare_call(_inference_args)
            result = self.__infer_single_impl(
                inference_input,
                inference_config,
                _extra_feed=_extra_feed,
                _render_only=_render_only,
                _prepared_input=_prepared_input,
                **_inference_args,
            )
            frame.result = self._conclude_call(result) if runs_provider else result
        return frame.result

    def __infer_single_impl(
        self,
        inference_input: Any,
        inference_config: Any = None,
        *,
        _extra_feed: Optional[dict] = None,
        _render_only: bool = False,
        _prepared_input: bool = False,
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
        # M7: un-defer the session logger now the ctx bridge is installed
        # (mirrors __ainfer_single_impl) — the single seam both the manual and
        # @bridge_entrypoint ctx-install paths converge on, before any logging.
        self._ensure_ctx_workspace_logger()
        self._propagate_to_children()

        # Capture original input BEFORE preprocessing for retry_with_original mode
        original_input = inference_input

        # Round-7 invariant (inside _prepare_prompt): conditional kwarg pass to
        # avoid TypeError on subclass overrides that don't declare extra_feed
        # (e.g. ConversationalInferencer._render_prompt).
        if not _prepared_input:
            inference_input = self._prepare_prompt(inference_input, _extra_feed)

        # v5 Fix #1 — capture the POST-render prompt as a closure-local so
        # Fix #4 (recovery) can re-issue the rendered prompt and the
        # guardrail can source it via the _fallback_state ContextVar (see
        # plan v4: zero new instance-mutated state). For non-templated
        # leaves _render_prompt is a pass-through, so rendered_input is
        # byte-identical to the (preprocessed) inference_input.
        rendered_input = inference_input

        # Phase 1: render_only mode — used by orchestrators that need the
        # rendered prompt for logging/cache-keying without invoking the LLM.
        # See Q14 / ConsensusIterationRecord.review_input use case.
        if _render_only:
            return inference_input

        # Before the resume hook: a fanned-out call never runs this backend, so a
        # stale backend-level cache must not replay it (resume is the fan-out's).
        if self._should_fan_out():
            return self._infer_via_fanout(
                inference_input,
                inference_config,
                {**_inference_args, **self.get_inference_args_from_state_graphs()},
            )

        # Resume hook: check for cached result from a previous session.
        # Placed AFTER preprocessing/rendering so prompt hash matches cache.
        resume_result = self._try_resume_from_cache(
            inference_input, inference_config, **_inference_args
        )
        if resume_result is not None:
            return self._complete_response(
                resume_result, finalize=self._finalize_output
            )

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

        # v5 Phase 1.1 — mint a per-call correlation ID and weave it into
        # every InferenceInput / InferenceArgs / InferenceResponse parts
        # file emitted from THIS invocation. The id flows through the
        # existing parts_file_namer kwarg (see _call_correlation_kwargs
        # docstring). Off → behaviour unchanged.
        call_id = uuid.uuid4().hex[:8]
        _corr = _call_correlation_kwargs(call_id)

        self.log_info(inference_input, "InferenceInput", is_artifact=True, **_corr)
        self.log_info(inference_args, "InferenceArgs", is_artifact=True, **_corr)

        # Convert 0 → None for timeout parameters (0 = disabled)
        effective_total_timeout = total_timeout or None

        # -- Build _fallback_state and fallback chain --
        # v5 Fix #1/#2 — `rendered_input` (post-render prompt) and
        # `call_id`/`guardrail_reject_attempt` ride the same per-call
        # ContextVar dict that already carries `partial_output` /
        # `cache_path`. The guardrail (`_render_guardrail_prompt`) reads
        # `rendered_input` from the ContextVar; the persist-on-reject
        # path (Fix #2) bumps the counter and uses `call_id` to name the
        # InferenceResponse parts file. Per-call, concurrency-safe by
        # construction — no new instance-mutated state added.
        _fallback_state = {
            "last_exception": None,
            "partial_output": None,
            "cache_path": None,
            "call_id": call_id,
            "guardrail_reject_attempt": 0,
            "guardrail_reason": None,
            "rendered_input": rendered_input,
        }
        # v5 Phase 1.4 — retry counter shared between _recovery_wrapper and
        # GuardrailRetry marker so post-scan can tag any subsequent
        # InferenceInput as retry_of=<parent call_id>.
        _retry_counter = {"n": 0}

        # Recovery wrapper — reads from closure-captured _fallback_state
        def _recovery_wrapper(inp, **kw):
            # v5 Phase 1.4 — interleave marker. Emitted BEFORE the inner
            # _infer_recovery call so the next InferenceInput (which mints
            # its OWN call_id) can be labeled as a retry of THIS parent.
            if _is_verbose_correlation():
                _retry_counter["n"] += 1
                _exc = _fallback_state.get("last_exception")
                # NOTE: the "guardrail_reject" (last_exception is None) branch is
                # unreachable on the recovery path — _on_transition always sets
                # _fallback_state["last_exception"] to the non-None
                # OutputValidationExhaustedError (even a non-string/False verdict
                # yields one; async_utils.py:331-333) BEFORE this wrapper runs.
                self.log_info(
                    {
                        "parent_call_id": call_id,
                        "retry_index": _retry_counter["n"],
                        "trigger": (
                            type(_exc).__name__
                            if _exc is not None
                            else "guardrail_reject"
                        ),
                    },
                    "GuardrailRetry",
                )
            # v5 Fix #4 — internal recovery re-runs the RENDERED prompt
            # (captured above), not the raw pre-render seed. For LWI
            # followup steps the raw seed is empty/stub (real content is
            # injected via extra_feed during render); raw-seed recovery
            # would feed the agent "" -> empty output -> guaranteed
            # false-RESTART loop. External fallback inferencers receive
            # retry_args[0] (still original_input per `_on_transition`
            # below) so they keep re-rendering with their OWN template —
            # internal-only fix.
            return self._infer_recovery(
                rendered_input or inp,
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
            # §9.3 E3/I3: each EXTERNAL fallback runs as a distinct child node
            # (fallback/external_{i}) so its provenance/state/handles don't collide
            # with the parent or siblings. No child (None) without an active ctx =>
            # legacy-mint, byte-identical.
            _fb_parent = active_run_context()
            external_wrappers = [
                (
                    lambda inp, inf=inf, _i=_i, **kw: inf.infer(
                        inp,
                        inference_config,
                        run_context=(
                            _fb_parent.child("fallback").child(f"external_{_i}")
                            if _fb_parent is not None
                            else None
                        ),
                        **kw,
                    )
                )
                for _i, inf in enumerate(fb_list)
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
                    with open(
                        _fallback_state["cache_path"], "r", encoding="utf-8"
                    ) as f:
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
                on_fallback_callback=_on_transition
                if effective_fallback_func
                else None,
                output_validator=(
                    self._run_output_guardrail_sync
                    if self.output_guardrail_inferencer is not None
                    else None
                ),
                # v5 Fix #5 — terminal short-circuit for the empty-loop
                # fail-fast. Mirrors the async sibling's tuple. Fix #2's
                # persist-on-reject runs BEFORE the raise so the
                # offending response is captured.
                non_retryable_exceptions=(
                    # U2d: a missing hard-dependency is deterministic — never
                    # retry it (and, pre-U1/U4, don't let it cascade).
                    MissingDependencyError,
                    HopelessOutputError,
                    # Guardrail-exhaustion is terminal for ENCLOSING layers: the
                    # leaf already retried the RESTART up to its own max_retry, so
                    # an outer wrapper must re-raise — NOT re-run the whole subtree
                    # (that multiplied one bad leaf into a whole-propose re-run).
                    OutputValidationExhaustedError,
                    InvocationContractError,
                ),
            )
        except TimeoutError:
            self.log_info(
                f"Total timeout after {total_timeout}s",
                "TotalTimeout",
            )
            raise
        except OutputValidationExhaustedError as exhausted:
            # Sync mirror of the async branch — see there for rationale.
            inference_response = self._accept_exhausted_update(
                exhausted, _fallback_state
            )
        finally:
            _current_fallback_state.reset(token)

        return self._finish_inference_call(
            inference_response, _corr, finalize=self._finalize_output
        )

    def _complete_response(self, response: Any, *, finalize: Callable) -> Any:
        """Shared call epilogue: finalize outputs, update state graphs, post-process."""
        response = finalize(response)
        if self.state_graphs:
            self.update_state_graphs(response)
        if self.response_post_processor is not None:
            return self.response_post_processor(
                self._normalize_for_post_processor(response)
            )
        return response

    def _finish_inference_call(
        self, response: Any, corr: dict, *, finalize: Callable
    ) -> Any:
        """``_complete_response`` bracketed by the ``InferenceResponse`` /
        ``PostProcessedResponse`` artifact logs of a normal (non-resumed) call.

        ``corr`` pairs both artifacts with this call's ``InferenceInput``
        through the shared call_id name-hint.
        """
        self.log_debug(response, "InferenceResponse", is_artifact=True, **corr)
        completed = self._complete_response(response, finalize=finalize)
        if self.response_post_processor is not None:
            self.log_debug(completed, "PostProcessedResponse", is_artifact=True, **corr)
        return completed

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
        if hasattr(response, "output") and isinstance(response.output, str):
            return response.output or ""
        raise TypeError(
            f"response_post_processor expects str or an object with a str .output attribute, "
            f"got {type(response).__name__}. Add a .output property or handle this type explicitly."
        )

    # -- Per-call BTA fan-out (bta_inferencer) -------------------------------

    def _validate_bta_inferencer_spec(self) -> None:
        """Reject a ``bta_inferencer`` value this class cannot honour."""
        spec = self.bta_inferencer
        if spec is None:
            return
        cls_name = type(self).__name__
        if not self._SUPPORTS_BTA_FANOUT:
            raise TypeError(
                f"{cls_name} does not support bta_inferencer; leave it None"
            )
        if isinstance(spec, Mapping):
            if not self._SUPPORTS_BTA_ROLE_MAPPING:
                raise TypeError(
                    f"{cls_name}.bta_inferencer is a role mapping, but {cls_name} "
                    "records no roles; give a single BTA template"
                )
            entries = [(f"bta_inferencer[{role!r}]", t) for role, t in spec.items()]
        else:
            entries = [("bta_inferencer", spec)]
        for label, template in entries:
            if template is None or callable(template):
                continue
            if not isinstance(template, InferencerBase):
                raise TypeError(
                    f"{cls_name}.{label} must be a BTA template (config factory, "
                    f"callable or inferencer) or None, got "
                    f"{type(template).__name__}: {template!r}"
                )

    def _active_role_name(self) -> Optional[str]:
        """The ``switch_role`` role this inferencer's render uses, or ``None`` for
        its own configured role."""
        return None

    def _select_bta_template(self) -> Any:
        """The ``bta_inferencer`` template for the active role (``None``: unfanned)."""
        spec = self.bta_inferencer
        if not isinstance(spec, Mapping):
            return spec
        return spec.get(self._active_role_name() or BTA_OWN_ROLE)

    @property
    def _delegates_execution(self) -> bool:
        """True when this call runs through a per-call BTA instead of the backend."""
        return self._SUPPORTS_BTA_FANOUT and self._select_bta_template() is not None

    def _should_fan_out(self) -> bool:
        """Whether this call fans out. Logs ``BtaFanOutBypassed`` when a role
        mapping leaves the active role unmapped."""
        spec = self.bta_inferencer
        if spec is None:
            return False
        if self._select_bta_template() is not None:
            return True
        self.log_info(
            {
                "reason": "role_not_mapped",
                "role": self._active_role_name() or BTA_OWN_ROLE,
                "mapped_roles": sorted(
                    str(role) for role, t in spec.items() if t is not None
                ),
            },
            "BtaFanOutBypassed",
        )
        return False

    def _infer_via_fanout(self, contract, inference_config, inference_args) -> Any:
        text, corr = self._run_fanout(contract, inference_config, inference_args)
        return self._finish_inference_call(
            text, corr, finalize=self._finalize_leaf_output
        )

    async def _ainfer_via_fanout(
        self, contract, inference_config, inference_args
    ) -> Any:
        text, corr = await self._arun_fanout(contract, inference_config, inference_args)
        return self._finish_inference_call(
            text, corr, finalize=self._finalize_leaf_output
        )

    def _run_fanout(self, contract, inference_config, inference_args) -> tuple:
        """Run ``contract`` through a fresh per-call BTA; returns ``(text, corr)``.
        This call's ledger closes the per-call BTA when the call ends."""
        call_id = uuid.uuid4().hex[:8]
        corr = _call_correlation_kwargs(call_id)
        self.log_info(contract, "InferenceInput", is_artifact=True, **corr)
        call_args, worker_args = self._split_fanout_args(inference_args)
        fanout, bta_ws, child_rc = self._prepare_fanout(worker_args, call_id, contract)
        started = time.monotonic()
        response = fanout.infer(
            contract, inference_config, run_context=child_rc, **call_args
        )
        text = self._conclude_fanout(
            fanout, bta_ws, child_rc, response, call_id, started
        )
        return text, corr

    async def _arun_fanout(self, contract, inference_config, inference_args) -> tuple:
        """Async ``_run_fanout``."""
        call_id = uuid.uuid4().hex[:8]
        corr = _call_correlation_kwargs(call_id)
        self.log_info(contract, "InferenceInput", is_artifact=True, **corr)
        call_args, worker_args = self._split_fanout_args(inference_args)
        fanout, bta_ws, child_rc = self._prepare_fanout(worker_args, call_id, contract)
        started = time.monotonic()
        response = await fanout.ainfer(
            contract, inference_config, run_context=child_rc, **call_args
        )
        text = self._conclude_fanout(
            fanout, bta_ws, child_rc, response, call_id, started
        )
        return text, corr

    def _prepare_fanout(self, worker_args: dict, call_id: str, contract) -> tuple:
        """Validate, build and bind this call's BTA before any model call; returns
        ``(fanout, bta_workspace, child_run_context)``."""
        self._validate_bta_inferencer_spec()
        proto = self._fanout_prototype()
        self._validate_fanout(proto)
        fanout = self._materialize_fanout(proto, worker_args)
        # Every InferencerBase stage of the fanout is a per-call copy (inventory
        # D1), so closing it when this call ends closes what the call built.
        invocation_of(self).ledger.register(fanout, BTA_INFERENCER_SLOT)
        self._validate_fanout_aggregator(fanout)
        seeded = self._seed_aggregator_feed(fanout)
        self._cascade_attributes_into(fanout)
        bta_ws = self._workspace.child(BTA_INFERENCER_SLOT)
        child_rc = self._rc_child(BTA_INFERENCER_SLOT, workspace=bta_ws)
        self._bind_rebuilt_child_ws(fanout, BTA_INFERENCER_SLOT, bta_ws, owned=True)
        if child_rc is not None:
            # Feed overrides published above the fan-out are already rendered into
            # the contract; the barrier keeps them from reaching its nodes again.
            child_rc.handles.set(TEMPLATE_EXTRA_FEED_SCOPE_HANDLE, True)
        text = str(contract)
        self.log_info(
            {
                "call_id": call_id,
                "depth": self._fanout_depth(),
                "role": self._active_role_name() or BTA_OWN_ROLE,
                "template_type": type(proto).__name__,
                "bta_ws": str(getattr(bta_ws, "root", bta_ws)),
                "worker_template_overridden": proto.worker_inferencers is not None,
                **self._fanout_slot_sources(proto),
                "seeded_feed_keys": seeded,
                "worker_arg_keys": sorted(worker_args),
                "contract_chars": len(text),
                "contract_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            },
            "BtaFanOut",
            **_call_correlation_kwargs(call_id),
        )
        return fanout, bta_ws, child_rc

    def _conclude_fanout(
        self, fanout, bta_ws, child_rc, response, call_id, started
    ) -> str:
        # Promote before finalize: finalize keeps an existing canonical output.
        self._symlink_child_output(
            bta_ws, child_output_name=fanout.output_path, replace_canonical=True
        )
        text = fanout.response_text(response)
        summary = self._summary_at(fanout, child_rc)
        self.log_info(
            {
                "call_id": call_id,
                "n_subtasks": (0 if summary is None else summary.worker_count),
                "aggregated_chars": len(text),
                "elapsed_s": round(time.monotonic() - started, 3),
            },
            "BtaFanOutComplete",
            **_call_correlation_kwargs(call_id),
        )
        return text

    def _fanout_prototype(self) -> Any:
        """The selected template unwrapped to a BTA prototype (called when it is
        a config factory or other callable)."""
        template = self._select_bta_template()
        if isinstance(template, (_PrototypeCloneFactory, _FreshCloneFactory)):
            return template.prototype
        if isinstance(template, InferencerBase):
            return template
        return template()

    def _validate_fanout(self, proto) -> None:
        """Reject a fan-out that cannot run correctly for this call."""
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
            BreakdownThenAggregateInferencer,
        )

        cls_name = type(self).__name__
        if not isinstance(proto, BreakdownThenAggregateInferencer):
            raise TypeError(
                f"{cls_name}.bta_inferencer must build a "
                f"BreakdownThenAggregateInferencer, got {type(proto).__name__}"
            )
        expected = proto.expected_parent_types
        classes = tuple(
            t if isinstance(t, type) else import_target(t) for t in expected
        )
        if classes and not isinstance(self, classes):
            raise TypeError(
                f"{cls_name} is not one of its bta_inferencer template's "
                f"expected_parent_types {list(expected)}"
            )
        if self._workspace is None or os.path.isabs(self.output_path or ""):
            raise ValueError(
                f"{cls_name} fan-out needs a workspace and a workspace-relative "
                f"output_path (got output_path={self.output_path!r})"
            )
        self._validate_fanout_slots(proto)
        depth = self._fanout_depth()
        if depth >= MAX_BTA_FANOUT_DEPTH:
            raise RuntimeError(
                f"{cls_name} bta_inferencer fan-out depth {depth} reached "
                f"MAX_BTA_FANOUT_DEPTH={MAX_BTA_FANOUT_DEPTH}"
            )

    def _validate_fanout_slots(self, proto) -> None:
        cls_name = type(self).__name__
        if not proto.inject_upstream_artifacts_to_aggregator:
            raise ValueError(
                f"{cls_name}.bta_inferencer sets inject_upstream_artifacts_to_aggregator"
                "=False: the aggregator input would replace the contract"
            )
        blank = self._fanout_blank_slots(proto)
        if blank and not self.supports_prompt_rendering:
            raise ValueError(
                f"{cls_name}.bta_inferencer leaves {list(blank)} blank, but {cls_name} "
                "cannot render prompts to fill them; configure them explicitly"
            )

    def _validate_fanout_aggregator(self, fanout) -> None:
        aggregator = fanout.aggregator_inferencer
        if (
            aggregator is not None
            and not getattr(aggregator, "supports_prompt_rendering", False)
            and fanout.aggregator_prompt_builder is None
        ):
            raise ValueError(
                f"{type(self).__name__}.bta_inferencer aggregator "
                f"{type(aggregator).__name__} cannot render the worker outputs and "
                "has no aggregator_prompt_builder"
            )

    @staticmethod
    def _fanout_blank_slots(proto) -> tuple:
        """The ``proto`` slots a re-roled copy of the fanned-out inferencer fills."""
        blank = []
        if proto.breakdown_inferencer is None and not proto.predefined_sub_queries:
            blank.append("breakdown_inferencer")
        if proto.aggregator_inferencer is None and not proto.disable_aggregator:
            blank.append("aggregator_inferencer")
        return tuple(blank)

    def _fanout_slot_sources(self, proto) -> Dict[str, str]:
        blank = self._fanout_blank_slots(proto)
        if proto.predefined_sub_queries:
            breakdown = "predefined"
        else:
            breakdown = "p_clone" if "breakdown_inferencer" in blank else "explicit"
        if proto.disable_aggregator:
            aggregator = "disabled"
        else:
            aggregator = "p_clone" if "aggregator_inferencer" in blank else "explicit"
        return {"breakdown_source": breakdown, "aggregator_source": aggregator}

    def _materialize_fanout(self, proto, worker_args: dict) -> Any:
        """This call's BTA: ``proto`` rebuilt with fresh copies of this inferencer
        as its workers and in its blank slots, and its aggregator resolved."""
        if proto.worker_inferencers is not None:
            warnings.warn(
                f"{type(self).__name__}.bta_inferencer template sets "
                "worker_inferencers; fan-out workers are always copies of the "
                "fanned-out inferencer, so it is ignored",
                UserWarning,
                stacklevel=3,
            )
        worker = self.fresh_instance(**self._p_derived_overrides())
        fanout = proto.fresh_instance(
            worker_inferencers=_FreshCloneFactory(worker),
            worker_inference_args={
                **proto.worker_inference_args,
                **worker_args,
                "prepared_input": True,
            },
            workspace=None,
            **self._fanout_slot_overrides(proto),
        )
        # Resolved before seeding, into the fanout's own slot: a config factory
        # applies its feed over the feed its aggregator declares, and the fanout's
        # call then borrows the instance validated and seeded here.
        fanout.aggregator_inferencer = resolve_stage(
            fanout.aggregator_inferencer
        ).inferencer
        return fanout

    def _p_derived_overrides(self) -> Dict[str, Any]:
        """``fresh_instance`` overrides for every copy of this inferencer inside its
        fan-out: no fan-out of its own, no boundary processing, and the values
        parents push after construction (absent from the recipe)."""
        overrides = _field_defaults(type(self), _FANOUT_RESET_FIELDS)
        overrides.update({name: val for name, val, _ in self._cascading_values()})
        overrides.update(
            template_manager=getattr(self, "template_manager", None),
            bta_inferencer=None,
            workspace=None,
        )
        params = set(_init_param_names(type(self)))
        return {k: v for k, v in overrides.items() if k in params}

    def _fanout_slot_overrides(self, proto) -> Dict[str, Any]:
        overrides = {
            slot: self.fresh_instance(
                **{
                    **self._p_derived_overrides(),
                    **self._fanout_role_overrides(proto, slot),
                }
            )
            for slot in self._fanout_blank_slots(proto)
        }
        if proto.disable_aggregator:
            overrides["aggregator_inferencer"] = None
        return overrides

    def _fanout_role_overrides(self, proto, slot: str) -> Dict[str, Any]:
        """``fresh_instance`` overrides re-roling this inferencer into ``proto``'s
        ``slot``; only inferencers that render prompts can fill a slot."""
        raise TypeError(
            f"{type(self).__name__} cannot render prompts, so it cannot fill the "
            f"blank {slot} of its bta_inferencer"
        )

    def _seed_aggregator_feed(self, fanout) -> List[str]:
        """Seed the fan-out aggregator's feed with this inferencer's instance feed;
        the aggregator's own keys and the BTA-owned keys win. The aggregator
        produces this inferencer's response, whose format is selected by instance
        feed flags. Returns the seeded keys."""
        target = getattr(fanout.aggregator_inferencer, TEMPLATE_EXTRA_FEED_ATTR, None)
        own = getattr(self, TEMPLATE_EXTRA_FEED_ATTR, None)
        if not isinstance(target, dict) or not isinstance(own, dict):
            return []
        seeded = []
        for key, value in own.items():
            if key in fanout.OWNED_FEED_KEYS or key in target:
                continue
            target[key] = value
            seeded.append(key)
        return seeded

    def _split_fanout_args(self, inference_args: dict) -> tuple:
        """Split per-call kwargs into the fan-out call's framework kwargs and the
        kwargs every worker call receives."""
        rest = dict(inference_args)
        call_args = {k: rest.pop(k) for k in _FANOUT_FRAMEWORK_ARGS if k in rest}
        return call_args, self._fanout_inference_args(rest)

    def _fanout_inference_args(self, args: dict) -> dict:
        """Project per-call kwargs onto independent workers: drop
        ``_FANOUT_DROPPED_ARGS`` and reject a truthy ``_FANOUT_SINGLE_CALL_ARGS``."""
        for key in self._FANOUT_DROPPED_ARGS:
            args.pop(key, None)
        continuing = []
        for key in self._FANOUT_SINGLE_CALL_ARGS:
            if args.pop(key, None):
                continuing.append(key)
        if continuing:
            raise ValueError(
                f"{type(self).__name__} got {continuing} on a fanned-out call, but "
                "its workers are independent instances; map this role to None in "
                "bta_inferencer to run it unfanned"
            )
        return args

    @staticmethod
    def _fanout_depth() -> int:
        """Number of enclosing ``bta_inferencer`` fan-outs on the active path."""
        ctx = active_run_context()
        if ctx is None:
            return 0
        return ctx.path.split("/").count(BTA_INFERENCER_SLOT)

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

    @staticmethod
    def _task_contract_state_at(
        child: Any, child_ctx: Any
    ) -> Optional[RenderedTaskContractState]:
        """The task contract ``child`` published for its call at ``child_ctx``.

        With a ctx (host or legacy), only the typed outcome channel: the child's
        ``NodeOutcomeState.task_contract`` at that exact path (``None`` if absent),
        so a parent never reads another call's value. Only in true no-ctx does it
        fall back to the child's documented getter.
        """
        if child_ctx is None:
            getter = getattr(child, "_proposer_task_instructions", None)
            text = getter() if callable(getter) else None
            if not isinstance(text, str) or not text:
                return None
            return RenderedTaskContractState.of(text, role=None, source_path="")
        outcome = read_outcome(child_ctx)
        return None if outcome is None else outcome.task_contract

    @staticmethod
    def _summary_at(child: Any, child_ctx: Any) -> Optional[InferencerStateBase]:
        """The run summary ``child`` published for its call at ``child_ctx``: the
        typed outcome under a ctx, the child's ``last_call_summary`` only in true
        no-ctx (a per-call child, so its last-call getter can't be stale)."""
        if child_ctx is None:
            return getattr(child, "last_call_summary", None)
        outcome = read_outcome(child_ctx)
        return None if outcome is None else outcome.summary

    @staticmethod
    def _final_output_at(child: Any, child_ctx: Any) -> Optional[str]:
        """The final output ``child`` published for its call at ``child_ctx`` (a
        leaf whose stream differs from it): the typed outcome under a ctx, the
        child's ``get_final_output()`` only in true no-ctx."""
        if child_ctx is None:
            getter = getattr(child, "get_final_output", None)
            return getter() if callable(getter) else None
        outcome = read_outcome(child_ctx)
        return None if outcome is None else outcome.final_output

    @staticmethod
    def _task_contract_at(child: Any, child_ctx: Any) -> str:
        """The text of :meth:`_task_contract_state_at` (``""`` if absent)."""
        contract = InferencerBase._task_contract_state_at(child, child_ctx)
        return "" if contract is None else contract.text

    def _proposer_task_instructions(self) -> str:
        """The task contract an AUTHOR beneath this node actually rendered.

        Node-level protocol backing the reviewer/fixer ``<OriginalTaskInstructions>``
        reference block. Those blocks describe the node's INPUT (what it was asked to
        do) — semantically distinct from the reviewed artifact, which is the node's
        OUTPUT. ``task_instructions`` is a predefined variable re-resolved per leaf,
        so a consumer that renders it itself binds any actor-scoped placeholder (e.g.
        ``{{ output_path }}``) to ITSELF. Instead each node reports what its author
        really rendered, and the orchestrator relays that verbatim.

        Base default: ``""`` (no templated author beneath this node) — consumers then
        omit the block. Templated leaves return their own snapshot; orchestrators
        delegate to the child on their INPUT side (proposer/worker), never to an
        aggregator, which renders the OUTPUT-side "merge the upstream outcomes"
        variant rather than the contract the node was given.
        """
        return ""

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

    def _rc_child(self, slot: str, *subslots: str, workspace=None):
        """M3 helper: derive the child RunContext for ``slot`` from the active ctx.

        Returns ``active_run_context().child(slot)`` when a context is active
        (always true within a public ``infer``/``ainfer`` call tree, since the
        bridge installs one), else ``None`` so the child legacy-mints — keeping
        the threading **byte-identical** while no reader consumes ``ctx`` yet.
        Orchestrators pass the result as ``run_context=self._rc_child("<slot>")``
        into child inference calls (the §2.7 deterministic slots).

        ``workspace`` (optional): when given, the child context uses it as its
        on-disk workspace VERBATIM instead of path-mirroring the (possibly
        namespaced) slot — so a worker dispatched under ctx node ``plan_bta.worker_0``
        can root its whole subtree under the intended ``worker_0`` workspace dir.

        ``subslots`` (optional): further components chained below ``slot``
        (``ctx.child(slot).child(sub)…``); ``workspace`` applies to the deepest.
        """
        from agent_foundation.common.inferencers.run_context import active_run_context

        ctx = active_run_context()
        if ctx is None:
            return None
        parts = (slot, *subslots)
        for depth, part in enumerate(parts):
            deepest = depth == len(parts) - 1
            ctx = ctx.child(safe_slot(part), workspace=workspace if deepest else None)
        return ctx

    def _check_cancelled(self, ctx=None) -> None:
        """§2.1/P-#6: raise ``CancelledError`` if the (given or active) context's
        shared ``cancellation_token`` is set. Orchestrators call this at child-
        dispatch boundaries; the token is read-only and shared by reference, so a
        cancel set anywhere is visible to all in-flight branches. No-op
        (byte-identical) when there is no context or no token is set."""
        if ctx is None:
            ctx = active_run_context()
        if ctx is None:
            return
        token = getattr(ctx.runtime, "cancellation_token", None)
        if token is None:
            return
        # Support a plain flag-object ({"cancelled": True}) or a callable/event.
        cancelled = False
        if isinstance(token, dict):
            cancelled = bool(token.get("cancelled"))
        elif hasattr(token, "is_set"):
            cancelled = token.is_set()
        elif callable(token):
            cancelled = bool(token())
        else:
            cancelled = bool(getattr(token, "cancelled", False))
        if cancelled:
            raise asyncio.CancelledError("run_context cancellation_token is set")

    def _init_call_state(self, inference_input):
        """M4: populate ``ctx.node.call`` via ``state_factory``.

        Runs once per invocation, inside its frame (seam step 5): after the path
        claim, so a rejected call initializes nothing, and per ``parallel_infer``
        or lazy-iterator item. The node's call state is populated only when unset,
        so ``state_factory`` sees the first input a node receives.

        No-op (byte-identical) when ``state_factory`` is unset OR no RunContext is
        active. The state object (typed ``InferencerStateBase`` or plain dict)
        lives in the context node, never on ``self``.
        """
        if self.state_factory is None:
            return
        ctx = active_run_context()
        if ctx is None:
            return
        node = ctx.node(creator=(type(self).__qualname__, ctx.path))
        if node.call is None:
            node.call = self.state_factory(inference_input)

    # Per-call values a caller may hand this inferencer as call keywords (plan v8
    # §5.9, B18): name -> RuntimeKey, merged along the MRO. Each public entry pops
    # the declared names from its call kwargs into its invocation frame before
    # anything else; ``_effective`` reads them back, falling back to the instance.
    _INVOCATION_KEYWORDS: ClassVar[Mapping[str, Any]] = MappingProxyType({})

    @classmethod
    def _invocation_keywords(cls) -> Mapping[str, Any]:
        return _merged_invocation_keywords(cls)

    def _pop_invocation_keywords(self, frame, kwargs: dict) -> None:
        for name, key in self._invocation_keywords().items():
            if name in kwargs:
                frame.put(key, kwargs.pop(name))

    def _effective(self, name: str) -> Any:
        """The per-call value of ``name`` handed to this invocation as a keyword
        (also when passed as ``None``), else the configured instance value."""
        key = self._invocation_keywords().get(name)
        frame = frame_for(self)
        if key is not None and frame is not None and frame.has(key):
            return frame.get(key)
        return getattr(self, name, None)

    def _runs_provider(self, render_only: bool) -> bool:
        """Whether this invocation runs the provider, so the provider hooks
        bracket it: a render-only call and a call delegated to a per-call BTA
        (``_delegates_execution``) never reach the backend."""
        return not render_only and not self._delegates_execution

    # ------------------------------------------------------------------
    # Tier-3 live handles: connection-scoped, path-keyed, never serialized.
    # ------------------------------------------------------------------

    def _get_live_handle_store(self):
        """M6/§2.0 Note B: the CONNECTION-scoped live-handle store — owned by THIS
        instance (the connection holder), so it persists across turns (V7
        continuity) independently of the per-turn RunContext tree. Branches are
        keyed by ``ctx.live_branch_key`` = ``(handle scope, ctx.path)``: concurrent
        branches of one run are isolated by path (V8), and independent host roots,
        all at ``"/"``, by scope (B32). Created lazily; never serialized.
        """
        store = self.__dict__.get("_live_handle_store")
        if store is None:
            store = LiveHandleStore()
            self.__dict__["_live_handle_store"] = store
        return store

    def _tier3_get(self, name: str, default: Any = None) -> Any:
        """M6 Tier-3 read: the per-branch handle from THIS instance's connection-
        scoped store at the active ctx's branch key (V8 isolation + V7 continuity);
        else the instance backing — which holds a legacy/no-ctx connection or one
        established at setup BEFORE any context (a shared base, NOT another branch's
        handle, since branch writes never touch the backing), unless the branch was
        detached from it (``_tier3_detach_from_backing``). Byte-identical with no
        active context.
        """
        ctx = active_run_context()
        if ctx is not None:
            branch = self._get_live_handle_store().peek(ctx.live_branch_key)
            if branch is not None:
                val = branch.get(name, None)
                if val is not None:
                    return val
                if branch.get(_TIER3_OWN_HANDLES_ONLY):
                    return default
        return self.__dict__.get(f"_{name}_backing", default)

    def _tier3_own_handles(self) -> Any:
        """The active branch's own Tier-3 handle set (``get``/``set``), never the
        backing its reads fall back to: its entry in this instance's connection
        store under a context, the instance backing with none."""
        ctx = active_run_context()
        if ctx is None:
            return _BackingHandleView(self)
        return self._get_live_handle_store().get_or_create(ctx.live_branch_key)

    def _tier3_detach_from_backing(self) -> None:
        """Make the active branch read only its own Tier-3 handles from now on.

        The backing holds a connection opened outside any context (an ``aconnect()``
        at setup), which every branch without one of its own reuses; a detached
        branch opens its own instead, and the others keep sharing it. No-op with no
        context, whose own handles are the backing.
        """
        if active_run_context() is not None:
            self._tier3_own_handles().set(_TIER3_OWN_HANDLES_ONLY, True)

    def _tier3_set(self, name: str, value: Any) -> None:
        """M6 Tier-3 write: under a context, write ONLY the branch's handle in this
        instance's connection-scoped store (at the ctx's branch key) — NOT the instance
        backing — so a branch never pollutes the shared base that other branches'
        COLD reads fall back to (the V8 cold-read isolation fix). With no context,
        write the instance backing (legacy, byte-identical)."""
        ctx = active_run_context()
        if ctx is not None:
            self._get_live_handle_store().get_or_create(ctx.live_branch_key).set(
                name, value
            )
        else:
            self.__dict__[f"_{name}_backing"] = value

    def _iter_live_handle_sets(self):
        """M6 teardown: yield a ``.get(name)``/``.set(name, value)`` view over EVERY
        live-handle set this instance holds — each connection-scoped branch, of every
        handle scope (keyed by the ctx it was established under during ``_ainfer``)
        AND the legacy no-context backing.

        A leaf ``adisconnect`` runs at a lifecycle boundary (``__aexit__`` / host
        cleanup) where ``active_run_context()`` is ``None`` (verified), so reading
        only the active branch — as the per-call Tier-3 property shims do — would
        strand every branch a context established during ``_ainfer`` (the V7/V8
        handle leak: SDK clients / subprocesses never reclaimed). Draining by stored
        path instead of by active context is the only teardown that reaches them."""
        store = self.__dict__.get("_live_handle_store")
        if store is not None:
            # snapshot: the leaf clears entries as it tears each down
            yield from list(store._by_path.values())
        yield _BackingHandleView(self)

    def _prepare_call(self, inference_args: dict) -> dict:
        """Provider pre-hook (seam step 6), inside the invocation frame after
        ``_init_call_state``: returns the keyword arguments the pipeline runs with.

        CLI leaves resolve their session policy here, so a claim-rejected call
        never touches session state. Runs only when the call runs the provider
        (``_runs_provider``). Identity by default.
        """
        return inference_args

    def _conclude_call(self, result: Any) -> Any:
        """Provider post-hook (seam step 8) of the sync entries, inside the
        invocation frame after the pipeline returns: returns the call's final
        result.

        Raising here fails the call, so no outcome is published. Runs only when
        the call runs the provider (``_runs_provider``). Identity by default.
        """
        return result

    async def _aconclude_call(self, result: Any) -> Any:
        """Async twin of :meth:`_conclude_call`, run by the async entries; it
        defaults to the sync hook. Leaves whose async post-call code awaits, or
        reads results only the async transport produces, override it."""
        return self._conclude_call(result)

    def _outcome_for(self, frame: InvocationFrame) -> Optional[NodeOutcomeState]:
        """The typed outcome this invocation publishes at its node when it closes
        successfully; the single publish point (``None`` publishes nothing).

        Classes contribute through ``frame`` components during the call; nobody
        publishes mid-call.
        """
        return None

    # ------------------------------------------------------------------
    # Graph visualization (Part F / GT#13): uniform graph_reporter propagation.
    # Node identity is the running unit's ctx.path (not its Python instance), so
    # the reporter is resolved from the shared Tier-2 sink and namespaced by
    # ctx.path. Promoted into the sink ONCE per run — eagerly in ainfer/infer so
    # a non-emitting orchestrator (e.g. PTI) seeds it before its children run,
    # and lazily in _resolve_graph_reporter() as a fallback. Inherited by every
    # inferencer (not just BTA). Viz failures are swallowed — visualization must
    # never abort inference.
    # ------------------------------------------------------------------
    def _seed_graph_reporter_into_runtime(self) -> None:
        """Promote this inferencer's ``graph_reporter`` into the shared Tier-2
        sink (``ctx.runtime.graph_reporter``) exactly once.

        No-op when no run context is active, this inferencer has no reporter, or
        the sink is already seeded. Fully guarded.
        """
        try:
            reporter = self.graph_reporter
            if reporter is None:
                return
            from agent_foundation.common.inferencers.run_context import (
                active_run_context,
            )

            ctx = active_run_context()
            if ctx is None:
                return
            if getattr(ctx.runtime, "graph_reporter", None) is None:
                ctx.runtime.graph_reporter = reporter
        except Exception:  # noqa: BLE001 - viz wiring must never break inference
            pass

    def _resolve_graph_reporter(self):
        """The graph-reporter sink for THIS inferencer's own nodes.

        Resolved from the shared Tier-2 sink and namespaced by ``ctx.path`` via
        ``child_reporter`` — so one shared instance reused across N concurrent
        contexts emits N distinct viz node subtrees (one per path), never
        last-write-wins onto one node.

        * No ctx (legacy/no-ctx): returns the instance attrib (byte-identical).
        * Active ctx: seeds the shared sink once from this instance's reporter
          (so an externally-wired ``self.graph_reporter`` becomes the Tier-2
          sink), then returns the path-namespaced child reporter (``None`` when
          nothing is wired).
        """
        from agent_foundation.common.inferencers.run_context import active_run_context

        ctx = active_run_context()
        if ctx is None:
            return self.graph_reporter
        runtime = ctx.runtime
        if runtime.graph_reporter is None and self.graph_reporter is not None:
            runtime.graph_reporter = self.graph_reporter
        return runtime.child_reporter(ctx.path, base_path="/")

    # --- Reusable graph-emit primitives (used by PTI/Dual/MFDual orchestrators).
    # Each resolves the path-namespaced reporter, so a top-level inferencer emits
    # a ROOT topology (parent_node_id="") while a nested one emits a SUB-graph
    # (parent_node_id=<its ctx path>). All guarded — viz never aborts inference.
    async def _emit_graph_topology(self, nodes, edges=None) -> None:
        """Emit a GraphTopologyEvent at this inferencer's ctx level (guarded)."""
        try:
            reporter = self._resolve_graph_reporter()
            if reporter is None or not nodes:
                return
            from agent_foundation.common.inferencers.graph_events import (
                GraphTopologyEvent,
            )

            await reporter.on_graph_topology(
                GraphTopologyEvent(nodes=nodes, edges=edges or [], layout="horizontal")
            )
        except Exception:
            import logging

            logging.getLogger(__name__).debug(
                "[%s] _emit_graph_topology failed",
                type(self).__name__,
                exc_info=True,
            )

    async def _emit_graph_node_status(self, node_id, status, output_path="") -> None:
        """Emit a node_status for one node at this inferencer's ctx level (guarded)."""
        try:
            reporter = self._resolve_graph_reporter()
            if reporter is None:
                return
            await reporter.on_node_status(
                node_id, status, output_path=output_path or ""
            )
        except Exception:
            import logging

            logging.getLogger(__name__).debug(
                "[%s] _emit_graph_node_status(%s, %s) failed",
                type(self).__name__,
                node_id,
                status,
                exc_info=True,
            )

    async def _emit_graph_reconcile_statuses(self, statuses) -> None:
        """Emit a final reconcile of terminal node statuses (guarded)."""
        try:
            reporter = self._resolve_graph_reporter()
            if reporter is None or not statuses:
                return
            await reporter.on_graph_reconcile(statuses)
        except Exception:
            import logging

            logging.getLogger(__name__).debug(
                "[%s] _emit_graph_reconcile_statuses failed",
                type(self).__name__,
                exc_info=True,
            )

    def infer(
        self,
        inference_input: Any,
        inference_config: Any = None,
        *,
        run_context=None,
        **_inference_args,
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
        # M2 bridge: resolve the keyword-only carrier (legacy-mint a root when
        # None -> byte-identical) and install it for the dispatch.
        ctx = resolve_run(
            run_context, default_workspace=getattr(self, "_workspace", None)
        )
        _rc_token = enter_run(ctx)
        try:
            self._seed_graph_reporter_into_runtime()
            result = self._infer_dispatch(
                inference_input, inference_config, **_inference_args
            )
        finally:
            exit_run(_rc_token)
        # The iterator-input / no-merger path returns a LAZY iterator (see
        # _infer_dispatch): each item runs under this call's ctx, bound per
        # ``next()`` only, so the consumer keeps its own context between items.
        if isinstance(inference_input, Iterator) and self.post_response_merger is None:
            return ctx_bound_gen(ctx, result)
        return result

    def _infer_dispatch(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        """Internal dispatch for :meth:`infer` (the historical body, unchanged).

        Split out so the public :meth:`infer` installs the RunContext bridge (M2)
        without re-indenting this logic. May return a lazy iterator; the bridge is
        inert at M2 (nothing reads ``_active_ctx`` yet), so the post-return
        ContextVar teardown is behaviourally invisible.
        """
        if isinstance(inference_input, Iterator):
            iterator_result = self._infer_iterator(
                inference_input, inference_config, **_inference_args
            )

            if self.post_response_merger is not None:
                all_responses = list(iterator_result)
                merged_response = self.post_response_merger(all_responses)
                self.log_debug(merged_response, "MergedResponse", is_artifact=True)
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
        *,
        run_context=None,
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
            debug: True runs every input sequentially in the calling thread.
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
            from rich_python_utils.mp_utils.parallel_process import (
                parallel_process_by_pool,
            )
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

        # D6/I2: parent context for per-input isolation (explicit or already-active).
        _parent_ctx = run_context if run_context is not None else active_run_context()
        if _parent_ctx is not None and not use_threading:
            raise NotImplementedError(
                "parallel_infer(use_threading=False) cannot carry an explicit "
                "run_context: a RunContext (Tier-2 sinks + Tier-3 live handles) is "
                "not picklable and ContextVars do not cross process boundaries. Use "
                "use_threading=True (default) for per-input context isolation."
            )

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

        if _parent_ctx is not None:
            # D6/I2: bind a per-input child context INSIDE each ThreadPool worker —
            # a fresh thread starts with the default context (the parent's set is
            # invisible) and a single shared copy_context() would give every worker
            # the SAME ctx, defeating per-input isolation. Pair (i, input) so the
            # worker knows its index for ``parallel_{i}``.
            def _ctx_worker(
                _pair, _pc=_parent_ctx, _cfg=inference_config, _kw=_inference_args
            ):
                _i, _inp = _pair
                _tok = enter_run(_pc.child(f"parallel_{_i}"))
                try:
                    return self._infer_single(_inp, _cfg, **_kw)
                finally:
                    exit_run(_tok)

            worker = _ctx_worker
            _data_iter = list(enumerate(inference_inputs))
        else:
            worker = partial(
                self._infer_single,
                inference_config=inference_config,
                **_inference_args,
            )
            _data_iter = inference_inputs
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

        num_p = 1 if debug else num_workers
        self._refuse_overlapping_items(_parent_ctx, num_p, "parallel_infer")
        results = parallel_process_by_pool(
            num_p=num_p,
            data_iter=_data_iter,
            target=mp_target,
            pool_object=pool_class,
        )
        # One result tuple when num_p == 1, else one per worker over contiguous
        # input chunks, in worker order.
        chunks = [results] if num_p == 1 else results
        return [result for chunk in chunks for result in chunk]

    def _refuse_overlapping_items(
        self, parent_ctx, concurrency: int, entry: str
    ) -> None:
        """Under a host ctx, more than one item at a time runs overlapping
        invocations of this one object, which only a host-pure certified class
        supports: refused before any item runs, not by whichever item the
        single-flight guard catches."""
        if (
            concurrency <= 1
            or parent_ctx is None
            or parent_ctx.legacy_mint
            or host_pure_certified(type(self))
        ):
            return
        raise UncertifiedConcurrentUseError(
            f"{entry} would run up to {concurrency} overlapping host invocations of "
            f"one {type(self).__name__}, which is not host-pure certified. Run the "
            f"items on separate instances (for example fresh_instance() copies) or "
            f"one at a time."
        )

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
        # U4-A: an orchestrator's subtree already ran (and failed) once; blindly
        # re-invoking _ainfer re-runs the whole graph against a dirty workspace.
        # Re-raise so the failure surfaces (contained by the parent's quorum,
        # U4-B) instead of silently re-running the subtree. NOTE: a guardrail
        # rejection passes a NON-None OutputValidationExhaustedError (verdict in
        # args[1]), so this guard fires and re-raises it — leaf RETRY/UPDATE
        # recovery is leaf-only (enforced in __attrs_post_init__).
        if self._is_orchestrator() and last_exception is not None:
            raise last_exception
        return await self._ainfer(inference_input, inference_config, **kwargs)

    # ------------------------------------------------------------------
    # Output guardrail (LLM-based quality judge)
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_shaped_fingerprint(response) -> Optional[str]:
        """Return a fingerprint string if ``response`` looks empty/banner-shaped,
        else None.

        "Empty-shaped" = whitespace-collapsed length <= 200 chars. This is the
        Devmate-ACL-denied / Claude-startup-banner failure signature: the same
        ~40 byte banner repeats verbatim across retries. Anything substantial
        (real LLM work) blows past 200 chars and returns None — no fail-fast.
        """
        try:
            text = response if isinstance(response, str) else str(response or "")
            # Collapse whitespace runs so "Starting Devmate server and session...\n"
            # vs "Starting Devmate server and session... " fingerprint identically.
            collapsed = " ".join(text.split())
            if len(collapsed) > 200:
                return None
            return collapsed
        except Exception:
            return None

    @staticmethod
    def _guardrail_empty_window() -> list:
        """The current call's empty-fingerprint window.

        Mutated in place so copied contexts of one call share it; a direct
        validator call outside any retry loop gets a throwaway window.
        """
        fs = _current_fallback_state.get(None)
        if fs is None:
            return []
        return fs.setdefault("guardrail_empty_fingerprints", [])

    def _check_guardrail_fail_fast(self, response) -> bool:
        """Return True iff the inferencer should ABORT retries immediately.

        Updates the current call's fingerprint window as a side effect:
          - Substantive output (None fingerprint) → reset window, no fail-fast.
          - Repeat of same fingerprint → window grows; trip when reaching N.
          - Different fingerprint → reset window to just this one (legitimate
            retry where output is changing).

        Knob: ``guardrail_empty_fail_fast_n`` (default 2 = trip on second
        identical empty in a row; 0 disables).
        """
        n = self.guardrail_empty_fail_fast_n
        if n <= 0:
            return False
        window = self._guardrail_empty_window()
        fingerprint = self._empty_shaped_fingerprint(response)
        if fingerprint is None:
            # Substantive output — clear the window. (Whether the judge later
            # rejects it for a non-empty reason is independent: that's a
            # legitimate retry, not a hopeless loop.)
            window.clear()
            return False
        if window and window[-1] != fingerprint:
            window.clear()
        window.append(fingerprint)
        if len(window) >= n:
            _logger.warning(
                "[%s] HOPELESS-OUTPUT FAIL-FAST: %d consecutive identical "
                "empty-shaped outputs (~%d chars each, fingerprint=%r). "
                "Aborting retries early; raising a terminal error so callers "
                "(e.g. MFDual quorum) can route around this leaf. Set "
                "guardrail_empty_fail_fast_n=0 to disable.",
                type(self).__name__,
                len(window),
                len(fingerprint),
                fingerprint[:80],
            )
            return True
        return False

    async def _run_output_guardrail(self, response) -> bool:
        """Run the guardrail judge inferencer on the response.

        Called as ``output_validator`` inside ``async_execute_with_retry`` when
        ``output_guardrail_inferencer`` is assigned. Returns True to accept the
        output, False to reject (triggers the recovery/fallback chain).

        The judge gets its own workspace (``children/guardrail/``) and runs the
        ``recovery/judge`` template with the main inferencer's input + output.
        Subclasses override ``_parse_guardrail_verdict`` to customize the
        verdict contract.

        v4 Phase 3.2 — when the judge would reject AND the output is the N-th
        consecutive identical empty/banner-shaped emission, raise a terminal
        ``HopelessOutputError`` so the retry chain stops burning ~25 min per
        stuck flow.
        """
        judge = self.output_guardrail_inferencer
        if judge is None:
            return True
        # Defense-in-depth for the mutable public slot: a guardrail assigned to an
        # orchestrator post-construction (bypassing the __attrs_post_init__ guard)
        # would re-raise terminally on rejection. Skip it (accept) with a warning.
        if self._is_orchestrator():
            _logger.warning(
                "[%s] output_guardrail attached to an orchestrator; skipping "
                "(leaf-only). Attach it to the leaf inferencer instead.",
                type(self).__name__,
            )
            return True
        try:
            prompt = self._judge_input(judge, self._render_guardrail_prompt(response))
            guardrail_ctx = self._prepare_guardrail_judge(judge)
            verdict_raw = await judge.ainfer(
                prompt, run_context=guardrail_ctx, prepared_input=True
            )
            verdict = self._parse_guardrail_verdict(verdict_raw)
            if verdict is not True:
                # v4 Phase 3.3 — diagnostic audit for the judge over-rejection
                # investigation (E4-aux). Logs the size of the output passed
                # to the judge plus a head+tail snippet of the judge's verdict
                # text. If the judge sees a full 34 KB output but verdicts
                # "narration-only" (the pattern observed in the
                # understand_codebase run), the judge prompt itself is buggy
                # and needs revision; if the judge sees < 100 B, the
                # content-extraction in ``_render_guardrail_prompt`` is the
                # bug. Pure diagnostic — no behavior change.
                try:
                    _resp_len = len(response) if response else 0
                except TypeError:
                    _resp_len = -1
                try:
                    _verdict_repr = repr(verdict_raw)
                    if len(_verdict_repr) > 600:
                        _verdict_repr = (
                            _verdict_repr[:300]
                            + " …[truncated]… "
                            + _verdict_repr[-200:]
                        )
                except Exception:
                    _verdict_repr = "<unreprable>"
                _logger.info(
                    "[%s] Output guardrail rejected: handler=%s "
                    "(judge_input_chars=%d, verdict=%s)",
                    type(self).__name__,
                    verdict,
                    _resp_len,
                    _verdict_repr,
                )
                # v5 Fix #2 — persist the rejected InferenceResponse BEFORE
                # the fail-fast raise (and before the helper turns the
                # verdict into a ValueError that the helper later raises
                # on retry exhaustion). Without this, the proposer's good
                # response that triggered a false-RESTART (or any retried
                # rejection) is silently dropped. Counter rides on the
                # per-call ``_fallback_state`` ContextVar; verbose-
                # correlation supplies a unique parts_file_namer so
                # multiple rejections from the SAME call don't overwrite.
                _fs = _current_fallback_state.get(None)
                _cid = _fs.get("call_id") if _fs else None
                if _fs is not None:
                    _fs["guardrail_reject_attempt"] = (
                        _fs.get("guardrail_reject_attempt", 0) + 1
                    )
                    # Q2: carry the judge's concrete <reason> (verdict text after the
                    # UPDATE:/RETRY: prefix) to recovery/update.jinja2 for a guided fix.
                    _fs["guardrail_reason"] = self._extract_guardrail_reason(
                        verdict_raw
                    )
                    # Keep the newest UPDATE body so `promote_exhausted_update`
                    # can publish it if the retry budget runs out. Only UPDATE:
                    # a RETRY verdict means "nothing to preserve" by definition.
                    if verdict == "update":
                        _fs["last_update_output"] = response
                _k = _fs.get("guardrail_reject_attempt", 0) if _fs else 0
                if _cid and _is_verbose_correlation():
                    # Default-arg capture pinning current values; avoids
                    # late-binding bugs if the lambda outlives this frame.
                    self.log_debug(
                        response,
                        "InferenceResponse",
                        is_artifact=True,
                        parts_file_namer=(
                            lambda _obj, _c=_cid, _kk=_k: f"call_{_c}_reject{_kk}"
                        ),
                    )
                else:
                    self.log_debug(response, "InferenceResponse", is_artifact=True)
                if self._check_guardrail_fail_fast(response):
                    raise HopelessOutputError(
                        f"{type(self).__name__}: {self.guardrail_empty_fail_fast_n} "
                        f"consecutive identical empty/banner-shaped outputs. "
                        f"See WARNING above for fingerprint."
                    )
            else:
                # Accepted — clear the empty-shape tracker so a later genuine
                # failure can start fresh.
                self._guardrail_empty_window().clear()
            return verdict
        except HopelessOutputError:
            raise
        except Exception as exc:
            _logger.warning(
                "[%s] Output guardrail judge failed: %s — accepting output (fail-open).",
                type(self).__name__,
                exc,
            )
            return True

    def _run_output_guardrail_sync(self, response):
        """Sync equivalent of ``_run_output_guardrail`` (v4 Phase 3.2 hopeless
        fail-fast mirrored from the async variant)."""
        judge = self.output_guardrail_inferencer
        if judge is None:
            return True
        if self._is_orchestrator():
            _logger.warning(
                "[%s] output_guardrail attached to an orchestrator; skipping "
                "(leaf-only). Attach it to the leaf inferencer instead.",
                type(self).__name__,
            )
            return True
        try:
            prompt = self._judge_input(judge, self._render_guardrail_prompt(response))
            guardrail_ctx = self._prepare_guardrail_judge(judge)
            verdict_raw = judge.infer(
                prompt, run_context=guardrail_ctx, prepared_input=True
            )
            verdict = self._parse_guardrail_verdict(verdict_raw)
            if verdict is not True:
                # v4 Phase 3.3 — diagnostic audit for the judge over-rejection
                # investigation (E4-aux). Logs the size of the output passed
                # to the judge plus a head+tail snippet of the judge's verdict
                # text. If the judge sees a full 34 KB output but verdicts
                # "narration-only" (the pattern observed in the
                # understand_codebase run), the judge prompt itself is buggy
                # and needs revision; if the judge sees < 100 B, the
                # content-extraction in ``_render_guardrail_prompt`` is the
                # bug. Pure diagnostic — no behavior change.
                try:
                    _resp_len = len(response) if response else 0
                except TypeError:
                    _resp_len = -1
                try:
                    _verdict_repr = repr(verdict_raw)
                    if len(_verdict_repr) > 600:
                        _verdict_repr = (
                            _verdict_repr[:300]
                            + " …[truncated]… "
                            + _verdict_repr[-200:]
                        )
                except Exception:
                    _verdict_repr = "<unreprable>"
                _logger.info(
                    "[%s] Output guardrail rejected: handler=%s "
                    "(judge_input_chars=%d, verdict=%s)",
                    type(self).__name__,
                    verdict,
                    _resp_len,
                    _verdict_repr,
                )
                # v5 Fix #2 — sync mirror of the async persist-on-reject
                # block above. See async sibling for full rationale.
                _fs = _current_fallback_state.get(None)
                _cid = _fs.get("call_id") if _fs else None
                if _fs is not None:
                    _fs["guardrail_reject_attempt"] = (
                        _fs.get("guardrail_reject_attempt", 0) + 1
                    )
                    # Q2: carry the judge's concrete <reason> (verdict text after the
                    # UPDATE:/RETRY: prefix) to recovery/update.jinja2 for a guided fix.
                    _fs["guardrail_reason"] = self._extract_guardrail_reason(
                        verdict_raw
                    )
                    # Keep the newest UPDATE body so `promote_exhausted_update`
                    # can publish it if the retry budget runs out. Only UPDATE:
                    # a RETRY verdict means "nothing to preserve" by definition.
                    if verdict == "update":
                        _fs["last_update_output"] = response
                _k = _fs.get("guardrail_reject_attempt", 0) if _fs else 0
                if _cid and _is_verbose_correlation():
                    self.log_debug(
                        response,
                        "InferenceResponse",
                        is_artifact=True,
                        parts_file_namer=(
                            lambda _obj, _c=_cid, _kk=_k: f"call_{_c}_reject{_kk}"
                        ),
                    )
                else:
                    self.log_debug(response, "InferenceResponse", is_artifact=True)
                if self._check_guardrail_fail_fast(response):
                    raise HopelessOutputError(
                        f"{type(self).__name__}: {self.guardrail_empty_fail_fast_n} "
                        f"consecutive identical empty/banner-shaped outputs."
                    )
            else:
                self._guardrail_empty_window().clear()
            return verdict
        except HopelessOutputError:
            raise
        except Exception as exc:
            _logger.warning(
                "[%s] Output guardrail judge failed: %s — accepting output (fail-open).",
                type(self).__name__,
                exc,
            )
            return True

    def _prepare_guardrail_judge(self, judge):
        """Isolate the guardrail judge before invocation; return its run-context.

        Fixes two coupled defects observed when a templated CLI judge runs as an
        output_validator:

        1. **Double-wrap.** ``_render_guardrail_prompt`` already produces the
           COMPLETE judge instruction (``recovery/judge.jinja2``). If the judge
           inherited a planning template (via cascade), its own ``_render_prompt``
           would wrap the judge prompt a SECOND time ("You are tasked with
           creating artifacts: [You are a quality judge…]") — so the judge does
           planning instead of judging. The judge is therefore called with
           ``prepared_input=True`` and executes the pre-rendered prompt verbatim
           (``_judge_input`` applies its ``input_preprocessor`` first), mirroring
           how recovery prompts are pre-rendered and fed back raw. The judge's
           definition is never modified (B22).

        2. **Context collision + workspace scatter.** Running the judge with no
           run-context made it claim the CALLER's context node (CollisionError →
           fail-open) and split its artifacts (cache under guardrail/, logs under
           the caller's workspace via context-override precedence in the
           ``_workspace`` getter). We give the judge its OWN ``guardrail`` child
           context with the guardrail workspace published to it (M7 pattern), so
           its node, workspace, cache, and logs all resolve consistently under
           ``children/guardrail/`` and never collide with the caller.
        """
        # (2) Own run-context child + published guardrail workspace (M7).
        guardrail_ctx = self._rc_child("guardrail")
        if self._workspace is not None:
            guardrail_ws = self._workspace.child("guardrail")
            self._publish_workspace_to_ctx(guardrail_ctx, guardrail_ws)
            if guardrail_ctx is None:
                judge._workspace = guardrail_ws  # legacy (no active context)
        return guardrail_ctx

    @staticmethod
    def _judge_input(judge, prompt):
        """The judge's input: the pre-rendered prompt, through the judge's own
        ``input_preprocessor`` when it has one (``prepared_input=True`` skips it)."""
        preprocessor = getattr(judge, "input_preprocessor", None)
        return prompt if preprocessor is None else preprocessor(prompt)

    def _render_guardrail_prompt(self, response) -> str:
        """Render the judge prompt (``recovery/judge``) with input + output.

        Unified with ``StreamingInferencerBase._render_recovery_prompt``: render
        via this inferencer's OWN ``template_manager`` when present — it already
        carries the recovery template root (added in
        ``StreamingInferencerBase.__attrs_post_init__`` via ``add_template_root``),
        so custom roots / overrides of ``recovery/judge.jinja2`` apply and the
        guardrail shares the single, flexible templating path used by the
        ``continue`` / ``retry_with_reference`` recovery prompts. Recovery
        templates live at ``recovery/<name>.jinja2`` (no type subdir), so render
        with ``active_template_type=""``. Falls back to the standalone recovery
        TemplateManager when this inferencer has no ``template_manager``.

        v5 Fix #1 — source the agent's input from the per-call
        ``_fallback_state`` ContextVar (key ``rendered_input``: the
        POST-render prompt captured at the single render seam in both
        ``__(a)infer_single_impl``). With no fallback state active (a
        direct call outside an inference) the input is empty. The final
        text is routed through the overridable ``_guardrail_input_text``
        shaper hook (default: identity — full rendered prompt).
        """
        output_text = self._guardrail_output_text(response)
        _fs = _current_fallback_state.get(None)
        _source = (_fs or {}).get("rendered_input") or ""
        input_text = self._guardrail_input_text(str(_source))
        tm = getattr(self, "template_manager", None)
        if tm is not None:
            # recovery/judge.jinja2 uses agent_prompt/agent_response.
            return tm(
                "recovery/judge",
                active_template_type="",
                agent_prompt=input_text,
                agent_response=output_text,
            )
        from agent_foundation.common.inferencers.recovery import render_recovery_prompt

        return render_recovery_prompt(
            "recovery/judge",
            prompt=input_text,
            partial_output=output_text,
        )

    def _guardrail_input_text(self, rendered: str) -> str:
        """Shape the input passed to the guardrail judge's ``{{ prompt }}`` slot.

        Default: return the FULL rendered prompt unchanged.

        Why the full prompt: the rendered prompt contains BOTH the
        ``<UserRequest>`` block (typically the body of the user-facing
        ask) AND the ``output_path`` instruction (typically the
        ``## Output Requirements`` section, e.g. line ~439 in the actual
        ``plan/main/initial.jinja2`` output) that the judge needs to see,
        including any wrapper/format the input requires (see the current
        ``recovery/judge.jinja2``). A missing/empty *expected* deliverable is
        surfaced to the judge by ``_guardrail_output_text`` itself (Fix 3), not
        by the judge's own file reasoning. Eagerly
        extracting only the ``<UserRequest>`` block in the base would
        strip the path and reintroduce the false-RESTART that triggered
        this fix.

        Subclasses MAY override to trim noise — e.g., extract just the
        ``<UserRequest>`` block — but MUST ensure the result still
        carries any agent-supposed-to-write-here paths so the judge can
        verify deliverables.

        Mirrors the overridable ``_parse_guardrail_verdict`` pattern.
        Opt-in per inferencer; no template coupling in the base.
        """
        return rendered

    def _guardrail_output_text(self, response) -> str:
        """Shape the OUTPUT the guardrail judge evaluates (its ``partial_output`` slot).

        Default: when this inferencer produced a written ``output_path`` deliverable,
        judge the FILE's content — NOT the captured stdout transcript. The deliverable
        IS the file; the transcript is fragile narration that terminal/CLI leaves can
        truncate under load, and a truncated transcript reads as "narration only" →
        false RESTART even though a complete deliverable exists on disk (the exact bug
        this fixes: a 40 KB output.md rejected because the judged transcript was a
        141-byte mid-tool snippet).

        Timing (why the file is present here): the guardrail runs as the retry loop's
        ``output_validator`` AFTER ``self._infer`` has run the agent to completion (file
        fully written) and BEFORE ``_finalize_output`` moves anything — so
        ``resolve_output_path()`` resolves to the complete ``outputs/<output_path>``.

        The WHOLE file is fed (no head+tail truncation): the machine-consumed fences
        (e.g. a 28-84 KB ``proposal_index``) sit at the file's end, and slicing them
        mid-JSON made the judge false-reject a correct deliverable as "malformed". A
        deliverable is bounded by the agent's own output, so the judge model handles the
        length.

        Fallback: if no output_path is configured, return the response transcript.
        If a local agent's file is missing/empty, return the transcript prefixed with
        a NEUTRAL statement of that fact (no "expected"/"not written" verdict) — the
        framework cannot know whether the input asked for a file, so it states only
        what it knows and leaves the file-vs-response determination to the judge
        (``recovery/judge.jinja2`` already instructs it to make exactly that call).
        Subclasses may override (mirrors ``_guardrail_input_text`` /
        ``_parse_guardrail_verdict``).
        """
        try:
            resolved = self.resolve_output_path()
        except Exception:
            resolved = None
        if resolved and os.path.isabs(resolved) and os.path.isfile(resolved):
            try:
                if os.path.getsize(resolved) > 0:
                    with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    # Feed the WHOLE deliverable to the judge — no head+tail truncation.
                    # The machine-consumed fences (e.g. a 28-84 KB ``proposal_index``) sit
                    # at the file's END, and slicing them mid-JSON made the judge
                    # false-reject a CORRECT deliverable as "malformed" (the 0709
                    # aggregator's 4x loop). A deliverable is bounded by the agent's own
                    # output (typically well within the judge model's context), so let the
                    # judge model handle the length rather than pre-truncating here.
                    return f"# Agent-written deliverable ({resolved}):\n{content}"
            except OSError:
                pass
        transcript = str(
            response.get("output", "") if isinstance(response, dict) else response
        )
        # Channel labeling — a FACT, deliberately NOT an intent verdict. The
        # framework knows whether a file exists at ``output_path``; it does NOT know
        # whether the input asked for one: ``output_path`` is blanket-cascaded to
        # every leaf (so its presence carries no contract signal) and
        # ``has_local_access`` is a capability, not a contract. An earlier version
        # asserted "EXPECTED DELIVERABLE ... WAS NOT WRITTEN" here, which (a) claimed
        # knowledge the framework does not have and (b) duplicated — and biased —
        # the determination ``recovery/judge.jinja2`` already makes from the input.
        # So: state the fact, disclaim the inference, let the judge decide.
        #
        # Still gated on has_local_access purely for RELEVANCE (a capability the
        # framework does know): a no-local agent could not have written a file at
        # all — its deliverable is legitimately inline in <Response> and is
        # materialized to output_path later by _finalize_output — so remarking on
        # the file's absence would be pure noise.
        if (
            resolved
            and os.path.isabs(resolved)
            and self.has_local_access
            and (not os.path.isfile(resolved) or os.path.getsize(resolved) == 0)
        ):
            return (
                f"# No file exists at {resolved}. (An output path is assigned to every\n"
                f"# agent; this does NOT imply the task required writing one — determine\n"
                f"# that from the input.)\n"
                f"# The agent's response transcript follows:\n{transcript}"
            )
        return transcript

    # Verdict keywords the guardrail understands, longest-safe for prefix matching.
    _GUARDRAIL_VERDICT_KEYWORDS = (
        "PASS",
        "UPDATE",
        "RETRY",
        "RESTART",
        "FAIL",
        "CONTINUE",
    )
    # Leading markdown/quote decoration a judge may wrap the verdict in.
    _GUARDRAIL_VERDICT_DECORATION = "`\"'*#>- \t\r\n"

    def _guardrail_verdict_segment(self, judge_response) -> str:
        """Isolate the verdict-bearing segment from a (possibly multi-line) judge reply.

        The recovery judge is a full agentic inferencer (``main_inferencer``): it reasons
        first and states its verdict LAST. Reading the HEAD of the reply (a naive
        ``startswith``) therefore misfires whenever the reasoning preamble merely opens
        with a verdict keyword — e.g. a real judge reply that began "RETRY/UPDATE/PASS
        decision requires …" but concluded ``PASS`` on its final line was misread as
        ``RETRY``, discarding a valid deliverable and forcing a wasteful restart.

        Resolution, tail-first:
          1. the text after the LAST ``Verdict:`` label — what ``recovery/judge.jinja2``
             asks the judge to emit on its final line — when that yields a known verdict
             (the historical ``Vertdict`` misspelling is tolerated defensively);
          2. otherwise the last non-empty line (a bare/legacy verdict-first reply, or a
             reasoning-first reply whose final line is the verdict, lands here).

        Leading markdown/quote decoration is stripped so the returned segment begins at
        the verdict keyword, ready for both keyword matching (``_parse_guardrail_verdict``)
        and reason extraction (``_extract_guardrail_reason``). Regex-free on purpose —
        ``re`` is not imported at this module's scope and these helpers are deliberately so.
        """
        text = str(judge_response).strip()
        if not text:
            return ""
        decoration = self._GUARDRAIL_VERDICT_DECORATION
        lowered = text.lower()
        # (1) Prefer the explicit label the judge is asked to emit on its final line.
        marker_end = -1
        for marker in ("verdict:", "vertdict:"):
            found = lowered.rfind(marker)
            if found >= 0:
                marker_end = max(marker_end, found + len(marker))
        if marker_end >= 0:
            labeled = text[marker_end:].strip().lstrip(decoration)
            if labeled.upper().startswith(self._GUARDRAIL_VERDICT_KEYWORDS):
                return labeled
        # (2) Otherwise take the last non-empty line (verdict-first / bare / label-less).
        for line in reversed(text.splitlines()):
            if line.strip():
                return line.strip().lstrip(decoration)
        return ""

    def _parse_guardrail_verdict(self, judge_response):
        """Parse the guardrail judge response into a verdict.

        Returns:
            ``True`` to accept the output.
            ``False`` to reject (plain retry with the same func).
            A handler name string (``"retry"``, ``"update"``) to reject and
            route to a specific recovery strategy.

        Default contract (Option C taxonomy): the judge reasons first and states its
        verdict LAST (see ``recovery/judge.jinja2``), one of:
          - ``"PASS"`` → accept
          - ``"RETRY: <reason>"`` → reject; re-run from scratch, carrying the prior
            attempt only as a negative-example reference (empty / off-topic / narration-only)
          - ``"UPDATE: <reason>"`` → reject; edit/complete the prior output in place,
            preserving the good work (real-but-incomplete / truncated)

        The verdict is read from the TAIL of the reply via ``_guardrail_verdict_segment``
        (not the head), so a reasoning preamble that happens to open with a keyword no
        longer misroutes. Legacy verdicts are still mapped so a custom or stale judge
        never silently accepts a bad output: ``RESTART``/``FAIL``/``RETRY_WITH_REFERENCE``
        → ``"retry"``; ``CONTINUE`` → ``"update"``.

        Subclasses can override for a custom verdict format.
        """
        segment = self._guardrail_verdict_segment(judge_response)
        upper = segment.upper()
        if upper.startswith("PASS"):
            return True
        if upper.startswith("UPDATE"):
            return "update"
        if upper.startswith("RETRY"):  # RETRY and legacy RETRY_WITH_REFERENCE
            return "retry"
        # Legacy verdicts (backward-compat): route sanely, never silent-accept.
        if upper.startswith("RESTART") or upper.startswith("FAIL"):
            return "retry"
        if upper.startswith("CONTINUE"):
            return "update"
        # Nothing matched — this is the historical silent-accept fall-through. Keep
        # accepting (back-compat) but log it, since an unrecognized verdict is a
        # guardrail blind spot worth surfacing. Log the tail we tried to parse.
        _logger.warning(
            "[%s] Unrecognized guardrail verdict (tail=%r) — accepting output (PASS).",
            type(self).__name__,
            (segment or str(judge_response).strip())[:200],
        )
        return True

    def _accept_exhausted_update(self, error, fallback_state) -> Any:
        """Return the best ``UPDATE`` body when the retry budget is spent.

        Re-raises *error* unless every condition holds: the behaviour is enabled,
        the terminal verdict was ``update`` (never ``retry`` — that one means
        "nothing to preserve"), and an UPDATE body was actually captured. The
        verdict rides on ``error.args[1]``; see ``OutputValidationExhaustedError``.

        Returning normally hands the body back to the caller's success path, so
        ``_finalize_output`` publishes it exactly as it would a ``PASS`` — which
        is the point: ``UPDATE`` differs from ``PASS`` in degree, not in kind.
        """
        if not self.promote_exhausted_update:
            raise error
        verdict = error.args[1] if len(error.args) > 1 else None
        if verdict != "update":
            raise error
        body = (fallback_state or {}).get("last_update_output")
        if body is None:
            raise error

        self.log_warning(
            {
                "event": "DEGRADED_OUTPUT",
                "message": (
                    "Guardrail retries exhausted with a terminal UPDATE verdict "
                    "— publishing the last substantive output instead of "
                    "discarding it. Content is real but INCOMPLETE; the judge's "
                    "outstanding asks were never addressed."
                ),
                "outstanding": (fallback_state or {}).get("guardrail_reason"),
                "reject_attempts": (fallback_state or {}).get(
                    "guardrail_reject_attempt"
                ),
            },
            "DegradedOutput",
        )
        return body

    def _extract_guardrail_reason(self, verdict_raw) -> Optional[str]:
        """Extract the judge's concrete <reason> from a rejection verdict, for a guided
        UPDATE fix (rendered into ``recovery/update.jinja2``'s ``{{ reason }}`` slot).

        Reads the same tail segment as ``_parse_guardrail_verdict`` (via
        ``_guardrail_verdict_segment``) so the reason is drawn from the actual verdict
        line even when the judge reasons first. Strips the leading verdict keyword
        (``UPDATE``/``RETRY``) and any separator so only the human-readable reason
        remains; returns ``None`` for a bare verdict with no trailing reason. Regex-free
        on purpose (``re`` is not imported in this module). Subclasses may override to
        match a custom verdict format (mirrors ``_parse_guardrail_verdict``).
        """
        segment = self._guardrail_verdict_segment(verdict_raw)
        for prefix in ("UPDATE", "RETRY"):
            if segment.upper().startswith(prefix):
                segment = segment[len(prefix) :].lstrip(" \t:-—").strip()
                break
        return segment or None

    async def _ainfer_single(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        # ── Phase 1 (leaf-owned template rendering): extract per-call render
        # parameters from _inference_args. See _infer_single for design notes.
        async with aopen_invocation(self, "ainfer") as frame:
            _extra_feed = _inference_args.pop("extra_feed", None)
            _render_only = _inference_args.pop("render_only", False)
            _prepared_input = _inference_args.pop("prepared_input", False)
            self._pop_invocation_keywords(frame, _inference_args)
            self._init_call_state(inference_input)
            runs_provider = self._runs_provider(_render_only)
            if runs_provider:
                _inference_args = self._prepare_call(_inference_args)
            result = await self.__ainfer_single_impl(
                inference_input,
                inference_config,
                _extra_feed=_extra_feed,
                _render_only=_render_only,
                _prepared_input=_prepared_input,
                **_inference_args,
            )
            frame.result = (
                await self._aconclude_call(result) if runs_provider else result
            )
        return frame.result

    async def __ainfer_single_impl(
        self,
        inference_input: Any,
        inference_config: Any = None,
        *,
        _extra_feed: Optional[dict] = None,
        _render_only: bool = False,
        _prepared_input: bool = False,
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

        # M7: un-defer the session logger now the ctx bridge is installed (by
        # either InferencerBase.ainfer's enter_run or the @bridge_entrypoint
        # decorator on CLI/streaming overrides). This is the single seam both
        # paths converge on, before any InferenceInput logging below.
        self._ensure_ctx_workspace_logger()
        self._propagate_to_children()

        # Capture original input BEFORE preprocessing for retry_with_original mode
        original_input = inference_input

        if not _prepared_input:
            inference_input = self._prepare_prompt(inference_input, _extra_feed)

        # v5 Fix #1 — capture POST-render prompt as closure-local; see
        # sync sibling above for rationale. Published via _fallback_state
        # below; consumed by `_render_guardrail_prompt` (Fix #1) and by
        # `_recovery_wrapper` (Fix #4).
        rendered_input = inference_input

        # Phase 1: render_only short-circuit — see _infer_single.
        if _render_only:
            return inference_input

        # Before the resume hook: see _infer_single.
        if self._should_fan_out():
            return await self._ainfer_via_fanout(
                inference_input,
                inference_config,
                {**_inference_args, **self.get_inference_args_from_state_graphs()},
            )

        # Resume hook (async): check for cached result from a previous session.
        # Uses _atry (async) so streaming override can await self._ainfer().
        resume_result = await self._atry_resume_from_cache(
            inference_input, inference_config, **_inference_args
        )
        if resume_result is not None:
            return self._complete_response(
                resume_result, finalize=self._finalize_output
            )

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
            # N-R1: a class that adopts ONLY the slot-aware iterator must still
            # activate pre-retry cleanup, else it silently skips it.
            or type(self)._iter_child_slots is not InferencerBase._iter_child_slots
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

        # v5 Phase 1.1 — async path: mint per-call correlation ID + weave
        # into all InferenceInput / InferenceArgs / InferenceResponse parts
        # files emitted from THIS invocation. Off → behaviour unchanged.
        call_id = uuid.uuid4().hex[:8]
        _corr = _call_correlation_kwargs(call_id)

        self.log_info(inference_input, "InferenceInput", is_artifact=True, **_corr)
        self.log_info(inference_args, "InferenceArgs", is_artifact=True, **_corr)

        # Convert 0 → None for timeout parameters (0 = disabled)
        effective_total_timeout = total_timeout or None
        effective_attempt_timeout = attempt_timeout or None

        # -- Build _fallback_state and fallback chain --
        # v5 Fix #1/#2 — see sync sibling: `rendered_input` (post-render
        # prompt) + `call_id` + `guardrail_reject_attempt` ride the
        # per-call ContextVar; no new instance-mutated state.
        _fallback_state = {
            "last_exception": None,
            "partial_output": None,
            "cache_path": None,
            "call_id": call_id,
            "guardrail_reject_attempt": 0,
            "guardrail_reason": None,
            "rendered_input": rendered_input,
        }
        # v5 Phase 1.4 — shared retry counter for the GuardrailRetry marker.
        _retry_counter = {"n": 0}

        # Recovery wrapper — reads from closure-captured _fallback_state
        async def _recovery_wrapper(inp, **kw):
            # v5 Phase 1.4 — interleave marker before recovery retry.
            if _is_verbose_correlation():
                _retry_counter["n"] += 1
                _exc = _fallback_state.get("last_exception")
                # NOTE: the "guardrail_reject" (last_exception is None) branch is
                # unreachable on the recovery path — _on_transition always sets
                # _fallback_state["last_exception"] to the non-None
                # OutputValidationExhaustedError (even a non-string/False verdict
                # yields one; async_utils.py:331-333) BEFORE this wrapper runs.
                self.log_info(
                    {
                        "parent_call_id": call_id,
                        "retry_index": _retry_counter["n"],
                        "trigger": (
                            type(_exc).__name__
                            if _exc is not None
                            else "guardrail_reject"
                        ),
                    },
                    "GuardrailRetry",
                )
            # v5 Fix #4 — internal recovery re-runs the RENDERED prompt;
            # see sync sibling for rationale. External fallback wrappers
            # below still receive retry_args[0] (= original_input per
            # `_on_transition`), so they re-render via their own template.
            return await self._ainfer_recovery(
                rendered_input or inp,
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
            # §9.3 E3/I3: each EXTERNAL fallback runs as a distinct child node
            # (fallback/external_{i}) so its provenance/state/handles don't collide.
            # No child without an active ctx => legacy-mint, byte-identical.
            _fb_parent = active_run_context()
            external_wrappers = [
                (
                    lambda inp, inf=inf, _i=_i, **kw: inf.ainfer(
                        inp,
                        inference_config,
                        run_context=(
                            _fb_parent.child("fallback").child(f"external_{_i}")
                            if _fb_parent is not None
                            else None
                        ),
                        **kw,
                    )
                )
                for _i, inf in enumerate(fb_list)
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
                    with open(
                        _fallback_state["cache_path"], "r", encoding="utf-8"
                    ) as f:
                        raw = f.read()
                    _fallback_state["partial_output"] = raw if raw.strip() else None
                except OSError:
                    _fallback_state["partial_output"] = None
            # Reset retry_args[0] to original input so external fallback
            # inferencers see the original prompt, not the mutated retry prompt
            retry_args[0] = original_input
            # Forward to user-provided on_fallback_callback if present
            if _user_on_fallback is not None:
                result = _user_on_fallback(
                    from_func, to_func, exception, total_attempts
                )
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
                    # U2d: a missing hard-dependency is deterministic — never retry.
                    MissingDependencyError,
                    asyncio.LimitOverrunError,
                    asyncio.IncompleteReadError,
                    # v5 Fix #5 — make the empty-loop fail-fast actually
                    # terminal. Without this entry the retry helper
                    # catches `HopelessOutputError` as generic retryable
                    # and merely changes which exception ends the loop
                    # after `max_retry` exhausts — i.e. the fail-fast
                    # docstring lied. Fix #2's persist runs BEFORE the
                    # raise so the offending response is still captured.
                    HopelessOutputError,
                    # Guardrail-exhaustion is terminal for ENCLOSING layers: the
                    # leaf already retried the RESTART up to its own max_retry, so
                    # an outer wrapper (LWI flow / MFDual / BTA / Dual) must
                    # re-raise it — NOT re-run its whole subtree. Without this,
                    # one leaf's spent guardrail multiplied into a full
                    # propose/MultiFlow-fan-out re-run at every nesting level.
                    OutputValidationExhaustedError,
                    InvocationContractError,
                ),
                total_timeout=effective_total_timeout,
                attempt_timeout=effective_attempt_timeout,
                fallback_func=effective_fallback_func,
                fallback_mode=effective_fallback_mode,
                on_fallback_callback=_on_transition
                if effective_fallback_func
                else None,
                output_validator=(
                    self._run_output_guardrail
                    if self.output_guardrail_inferencer is not None
                    else None
                ),
            )
        except TimeoutError:
            self.log_info(
                f"Total timeout after {total_timeout}s",
                "TotalTimeout",
            )
            raise
        except OutputValidationExhaustedError as exhausted:
            # Falls through to the normal success path below (log + finalize),
            # so a terminal UPDATE publishes like a PASS. Re-raises for RETRY.
            inference_response = self._accept_exhausted_update(
                exhausted, _fallback_state
            )
        finally:
            _current_fallback_state.reset(token)

        return self._finish_inference_call(
            inference_response, _corr, finalize=self._finalize_output
        )

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
        self,
        inference_input: Any,
        inference_config: Any = None,
        *,
        run_context=None,
        **_inference_args,
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
        # M2 bridge: keyword-only carrier + per-task ContextVar (legacy-mint when
        # None -> byte-identical). The bridge is set before the body and reset in
        # `finally`; inert until M3+ orchestrators read `_active_ctx`.
        _rc_token = enter_run(
            run_context, default_workspace=getattr(self, "_workspace", None)
        )
        try:
            self._seed_graph_reporter_into_runtime()
            return await self._ainfer_dispatch(
                inference_input, inference_config, **_inference_args
            )
        finally:
            exit_run(_rc_token)

    async def _ainfer_dispatch(
        self, inference_input: Any, inference_config: Any = None, **_inference_args
    ):
        """Internal dispatch for :meth:`ainfer` (the historical body, unchanged)."""
        if isinstance(inference_input, Iterator):
            all_results = []
            async for item in self._ainfer_iterator(
                inference_input, inference_config, **_inference_args
            ):
                all_results.append(item)

            if self.post_response_merger is not None:
                merged = self.post_response_merger(all_results)
                self.log_debug(merged, "MergedResponse", is_artifact=True)
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
        *,
        run_context=None,
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

        # D6/I1: the parent context for per-input isolation — an explicit
        # ``run_context`` or the already-active context if nested. Under neither
        # (a true legacy root) there is no per-input split -> byte-identical.
        _parent_ctx = run_context if run_context is not None else active_run_context()

        if debug:
            self.log_debug(
                f"aparallel_infer debug mode: {num_inputs} inputs",
                "ParallelInfer",
            )
            results = []
            for _i, inp in enumerate(inference_inputs):
                result = await self._aparallel_one(
                    inp, _i, _parent_ctx, inference_config, _inference_args
                )
                results.append(result)
            return results

        if max_concurrency is None:
            max_concurrency = min(num_inputs, 32)
        self._refuse_overlapping_items(
            _parent_ctx, min(max_concurrency, num_inputs), "aparallel_infer"
        )

        self.log_debug(
            f"{num_inputs} inputs, max_concurrency={max_concurrency}",
            "ParallelInfer",
        )

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_infer(inp, i):
            async with semaphore:
                return await self._aparallel_one(
                    inp, i, _parent_ctx, inference_config, _inference_args
                )

        tasks = [_bounded_infer(inp, i) for i, inp in enumerate(inference_inputs)]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def _aparallel_one(
        self, inp, i, parent_ctx, inference_config, inference_args
    ):
        """D6/I1: run one parallel input under a per-input child context
        (``parallel_{i}``) when a parent context is active, so each input gets an
        isolated state node + Tier-3 handles (no §2.7 same-creator collapse).
        Byte-identical when ``parent_ctx`` is None (legacy true root)."""
        if parent_ctx is None:
            return await self._ainfer_single(inp, inference_config, **inference_args)
        self._check_cancelled(parent_ctx)  # §2.1/P-#6: halt fan-out if cancelled
        token = enter_run(parent_ctx.child(f"parallel_{i}"))
        try:
            return await self._ainfer_single(inp, inference_config, **inference_args)
        finally:
            exit_run(token)

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

    async def areset_conversation(self, *, run_context=None) -> None:
        """Start a fresh vendor conversation on one run-context branch.

        After it returns, this inferencer's next call on the branch of
        ``run_context`` (resolved like a public entry's: the given context, else
        the active one, else a legacy root) starts a new vendor conversation that
        carries none of the vendor's history. Other branches keep theirs, and no
        argument is added to a later vendor request. This base keeps no
        conversation, so there is nothing to reset.
        """

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

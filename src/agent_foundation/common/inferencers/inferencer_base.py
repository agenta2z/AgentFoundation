import asyncio
import contextlib
import enum
import logging
import os
import sys
import traceback
import uuid
from abc import ABC, abstractmethod
from contextvars import ContextVar
from functools import partial
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Type,
    Union,
)

# M2: explicit RunContext carrier + the compat bridge (additive; inert until M3+
# orchestrators read `_active_ctx`). `run_context` is **keyword-only** so it can
# never land in `**_inference_args` (the kwarg-leak defense). The bridge mints a
# legacy root when `run_context is None` (byte-identical) and is per-task (the
# `_active_ctx` ContextVar) so concurrent fan-out branches don't clobber.
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    enter_run,
    exit_run,
)
from attr import attrib, attrs
from rich_python_utils.common_objects.debuggable import Debuggable
from rich_python_utils.common_objects.workflow.common.resumable import Resumable
from rich_python_utils.common_utils import dict_, iter__, resolve_environ
from rich_python_utils.common_utils.function_helper import (
    execute_with_retry,
    FallbackMode,
    OutputValidationExhaustedError,
)
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

    def __repr__(self):
        return f"_PrototypeCloneFactory({type(self.prototype).__name__})"


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

    The trigger is per-inferencer-instance, tracked on the private attr
    ``_guardrail_recent_empty_fingerprints`` (list of last N hashes).
    Reset on the next non-empty / non-rejected output."""

    _guardrail_recent_empty_fingerprints: list = attrib(
        factory=list, init=False, repr=False
    )
    """Sliding window of the last N rejected-output fingerprints. Bounded
    at ``guardrail_empty_fail_fast_n`` entries; reset to [] whenever a
    rejected output's fingerprint differs OR the output isn't empty-shaped
    (legitimate retry pattern). See ``_check_guardrail_fail_fast``."""

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
    @property
    def _workspace(self):
        # M7 §2.12 option-b: prefer a per-call workspace published into the
        # active context's handles (``workspace_override``) when present — so a
        # single shared instance can serve concurrent branches with distinct
        # workspaces (the run-state is in the context, not on ``self``). Falls
        # back to the instance backing — **byte-identical** when no override is
        # set (no active context, or the orchestrator didn't publish one).
        from agent_foundation.common.inferencers.run_context import active_run_context

        ctx = active_run_context()
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
        the published ctx ``workspace_override`` for ``slot`` when a context is
        active, else the child's instance ``_workspace``. Lets an orchestrator
        stop mutating ``child._workspace`` under a context (write-purity) while its
        own reads still resolve. Byte-identical without a context."""
        ctx = active_run_context()
        if ctx is not None:
            override = ctx.child(slot).handles.get("workspace_override", None)
            if override is not None:
                return override
        return getattr(child_inf, "_workspace", None)

    def _bind_rebuilt_child_ws(self, child_inf, slot: str, child_ws) -> None:
        """Durably bind a rebuilt, NON-SHARED WorkGraph child's workspace.

        Publishes the workspace into the child's run-context (tier-1, ephemeral —
        serves fresh-path ctx readers) AND sets the durable instance backing
        (tier-2) so the binding survives a resume, where the active run-context is
        not guaranteed to be the dispatched child ctx across the child's recovery
        gate (see ``_workspace`` getter tiers). Safe ONLY for non-shared nodes —
        per-subtask workers and the single aggregator — each dispatched exactly
        once and never serving concurrent branches, so the durable set is
        race-free and does NOT violate the shared-instance workspace write-purity
        invariant that keeps reviewer/fixer reuse concurrency-safe. Byte-identical
        to the prior legacy (no-ctx) path; the only change is that the durable set
        is no longer gated behind the absence of a run-context.
        """
        if child_ws is None:
            return
        child_ws.ensure_dirs()
        self._publish_workspace_to_ctx(self._rc_child(slot), child_ws)
        if isinstance(child_inf, InferencerBase):
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
    ):
        """Transition this inferencer to a new semantic role.

        Centralises the workspace-swap + session-reset pattern that orchestrators
        (MFDual, PTI, ...) previously performed inline. The base layer handles
        workspace assignment and session reset; TemplatedInferencerBase extends
        with template attrs.

        Part 2 (two-axis model): the deliverable flags (output_is_deliverable /
        is_deliverable_boundary) are RETIRED — role transitions carry only
        workspace + session state; promotion is role-based via ``promote_child``.

        Args:
            new_role: human-readable role name (e.g. 'fixer_inferencer').
            workspace: if not None, assigned via the _workspace property
                setter (triggers _configure_for_workspace cascade).
            reset_session: if True, calls self.reset_session() (when available).
        """
        import time

        # 1. Workspace assignment FIRST — triggers cascade
        if workspace is not None:
            self._workspace = workspace
        # 2. Session reset
        if reset_session and hasattr(self, "reset_session"):
            self.reset_session()
        # 3. Audit trail (_role_history, lazy-init)
        history = getattr(self, "_role_history", None)
        if history is None:
            history = []
            object.__setattr__(self, "_role_history", history)
        changes = {
            **({"workspace": str(workspace.root)} if workspace else {}),
        }
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
        for spec in self._CASCADING_ATTRIBUTES:
            if isinstance(spec, str):
                name = spec
                should_propagate = lambda _p, c: c is None
            else:
                name, should_propagate = spec
            parent_val = getattr(self, name, None)
            if parent_val is None:
                continue  # parent unset — nothing to cascade for this attr

            def _on_instance(
                child,
                field_name,
                key,
                _name=name,
                _cond=should_propagate,
                _pv=parent_val,
            ):
                if not isinstance(child, InferencerBase):
                    return  # duck-typed callables don't participate
                if _cond(_pv, getattr(child, _name, None)):
                    setattr(child, _name, _pv)
                    child._propagate_cascading_attributes()  # reach grandchildren

            def _on_partial(partial, field_name, key):
                return None  # factory children inherit at instantiation, not here

            self._for_each_child_inferencer(_on_instance, _on_partial)

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
            self._logger_awaiting_workspace = False
            self._add_workspace_logger(workspace)
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
            for state_dir in (ws.outputs_dir, ws.checkpoints_dir):
                if os.path.isdir(state_dir):
                    os.makedirs(attempt_dir, exist_ok=True)
                    shutil.move(
                        state_dir,
                        os.path.join(attempt_dir, os.path.basename(state_dir)),
                    )
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
        it does not reintroduce the cross-branch instance mutation M7 removed; the
        un-defer body is synchronous (no ``await``) so concurrent gathered
        branches cannot create duplicate loggers, and per-branch write routing is
        handled by :meth:`_log_path_override`.
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
        self._logger_awaiting_workspace = False
        self._add_workspace_logger(ws)

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
        if not hasattr(self, "_ws_log_relpaths"):
            self._ws_log_relpaths = {}
        self._ws_log_relpaths[name] = os.path.relpath(file_path, workspace.root)

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
            # Post-normalize (deferred → upgrade): add directly + register config.
            self.logger["_workspace"] = json_entry[0]
            if not hasattr(self, "_resolved_logger_configs"):
                self._resolved_logger_configs = {}
            self._resolved_logger_configs["_workspace"] = json_entry[1]
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
        """Finalize outputs after inference: write the <Response> summary, emit manifest.

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
            os.symlink(
                os.path.abspath(src), dst, target_is_directory=os.path.isdir(src)
            )
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

    def _promote_child_checkpoints(self, child: "InferencerBase") -> None:
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

        Atomic tmp->rename so a resumer never observes a half-written file;
        best-effort (a failed copy is logged, never gates), mirroring
        :meth:`_persist_extracted_block`. Call this right after the child
        completes and BEFORE any downstream node can fail, so the promoted state
        is durable for resume. Parent-driven, so inherently concurrency-safe.
        """
        import shutil

        ws = self._workspace
        child_ws = getattr(child, "_workspace", None)
        if ws is None or child_ws is None:
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
            target = ws.checkpoint_path(os.path.join(child_name, dest))
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
        # M7: un-defer the session logger now the ctx bridge is installed
        # (mirrors __ainfer_single_impl) — the single seam both the manual and
        # @bridge_entrypoint ctx-install paths converge on, before any logging.
        self._ensure_ctx_workspace_logger()
        self._propagate_to_children()

        # Capture original input BEFORE preprocessing for retry_with_original mode
        original_input = inference_input
        self._last_inference_input = inference_input

        if self.input_preprocessor is not None:
            inference_input = self.input_preprocessor(inference_input)

        # Template rendering (opt-in: only when template_manager is set).
        # Round-7 invariant: conditional kwarg pass to avoid TypeError on
        # subclass overrides that don't declare extra_feed (e.g.
        # ConversationalInferencer._render_prompt). When _extra_feed is
        # None, this call is byte-identical to the legacy form.
        if _extra_feed is not None:
            inference_input = self._render_prompt(
                inference_input, extra_feed=_extra_feed
            )
        else:
            inference_input = self._render_prompt(inference_input)

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
                resume_result = self.response_post_processor(
                    self._normalize_for_post_processor(resume_result)
                )
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
                ),
            )
        except TimeoutError:
            self.log_info(
                f"Total timeout after {total_timeout}s",
                "TotalTimeout",
            )
            raise
        finally:
            _current_fallback_state.reset(token)

        # v5 Phase 1.1 — pair the InferenceResponse parts file with this
        # call's InferenceInput via the shared call_id name-hint.
        self.log_debug(
            inference_response,
            "InferenceResponse",
            is_artifact=True,
            **_corr,
        )

        # Output finalization (promote deliverables, write summary)
        inference_response = self._finalize_output(inference_response)

        # Update state graphs from response
        if self.state_graphs:
            self.update_state_graphs(inference_response)

        if self.response_post_processor is not None:
            post_input = self._normalize_for_post_processor(inference_response)
            processed_response = self.response_post_processor(post_input)
            self.log_debug(
                processed_response,
                "PostProcessedResponse",
                is_artifact=True,
                **_corr,
            )
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
        if hasattr(response, "output") and isinstance(response.output, str):
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

    def _rc_child(self, slot: str, *, workspace=None):
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
        """
        from agent_foundation.common.inferencers.run_context import active_run_context

        ctx = active_run_context()
        if ctx is None:
            return None
        # Sanitize to a valid single-component slot (child() rejects separators);
        # robustness so a node-id-derived slot can never raise mid-call.
        safe = str(slot).replace("/", "_").replace("\\", "_").strip() or "child"
        if safe in (".", ".."):
            safe = "child"
        return ctx.child(safe, workspace=workspace)

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
        """M4: populate ``ctx.node.call`` once per call via ``state_factory``.

        No-op (byte-identical) when ``state_factory`` is unset OR no RunContext is
        active. The state object (typed ``InferencerStateBase`` or plain dict)
        lives in the context node, never on ``self``.
        """
        if self.state_factory is None:
            return
        from agent_foundation.common.inferencers.run_context import active_run_context

        ctx = active_run_context()
        if ctx is None:
            return
        node = ctx.node(creator=(type(self).__qualname__, ctx.path))
        if node.call is None:
            node.call = self.state_factory(inference_input)

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
        # M2 bridge: capture the keyword-only carrier + install the per-task
        # ContextVar (legacy-mint a root when None -> byte-identical). Inert until
        # M3+ orchestrators read `_active_ctx`.
        _rc_token = enter_run(
            run_context, default_workspace=getattr(self, "_workspace", None)
        )
        # The iterator-input / no-merger path returns a LAZY iterator (see
        # _infer_dispatch); the context must stay alive until it is exhausted so
        # each lazily-produced item runs under the same ctx (not detached). Every
        # other path returns eagerly -> exit immediately (byte-identical).
        _is_lazy = (
            isinstance(inference_input, Iterator) and self.post_response_merger is None
        )
        try:
            self._seed_graph_reporter_into_runtime()
            self._init_call_state(inference_input)
            result = self._infer_dispatch(
                inference_input, inference_config, **_inference_args
            )
        except BaseException:
            exit_run(_rc_token)
            raise
        if _is_lazy:

            def _ctx_scoped_iter(_inner=result, _tok=_rc_token):
                try:
                    yield from _inner
                finally:
                    exit_run(_tok)

            return _ctx_scoped_iter()
        exit_run(_rc_token)
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

        results = parallel_process_by_pool(
            num_p=num_workers,
            data_iter=_data_iter,
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

    def _check_guardrail_fail_fast(self, response) -> bool:
        """Return True iff the inferencer should ABORT retries immediately.

        Updates the sliding-window fingerprint tracker as a side effect:
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
        fingerprint = self._empty_shaped_fingerprint(response)
        if fingerprint is None:
            # Substantive output — clear the window. (Whether the judge later
            # rejects it for a non-empty reason is independent: that's a
            # legitimate retry, not a hopeless loop.)
            if self._guardrail_recent_empty_fingerprints:
                self._guardrail_recent_empty_fingerprints = []
            return False
        # Empty-shaped output.
        window = self._guardrail_recent_empty_fingerprints
        if window and window[-1] == fingerprint:
            window.append(fingerprint)
        else:
            self._guardrail_recent_empty_fingerprints = [fingerprint]
            window = self._guardrail_recent_empty_fingerprints
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
            prompt = self._render_guardrail_prompt(response)
            guardrail_ctx = self._prepare_guardrail_judge(judge)
            verdict_raw = await judge.ainfer(prompt, run_context=guardrail_ctx)
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
                if self._guardrail_recent_empty_fingerprints:
                    self._guardrail_recent_empty_fingerprints = []
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
            prompt = self._render_guardrail_prompt(response)
            guardrail_ctx = self._prepare_guardrail_judge(judge)
            verdict_raw = judge.infer(prompt, run_context=guardrail_ctx)
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
                if self._guardrail_recent_empty_fingerprints:
                    self._guardrail_recent_empty_fingerprints = []
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
           planning instead of judging. We neutralize the judge's template_manager
           (once; idempotent) so it executes the pre-rendered prompt verbatim,
           mirroring how recovery prompts are pre-rendered and fed back raw.

        2. **Context collision + workspace scatter.** Running the judge with no
           run-context made it claim the CALLER's context node (CollisionError →
           fail-open) and split its artifacts (cache under guardrail/, logs under
           the caller's workspace via context-override precedence in the
           ``_workspace`` getter). We give the judge its OWN ``guardrail`` child
           context with the guardrail workspace published to it (M7 pattern), so
           its node, workspace, cache, and logs all resolve consistently under
           ``children/guardrail/`` and never collide with the caller.
        """
        # (1) Pre-rendered prompt → never let the judge re-template it.
        if getattr(judge, "template_manager", None) is not None:
            judge.template_manager = None
        # (2) Own run-context child + published guardrail workspace (M7).
        guardrail_ctx = self._rc_child("guardrail")
        if self._workspace is not None:
            guardrail_ws = self._workspace.child("guardrail")
            self._publish_workspace_to_ctx(guardrail_ctx, guardrail_ws)
            if guardrail_ctx is None:
                judge._workspace = guardrail_ws  # legacy (no active context)
        return guardrail_ctx

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
        ``__(a)infer_single_impl``). Falls back to the legacy
        ``_last_inference_input`` instance attribute when no fallback
        state is active (direct callers / legacy paths). The final
        text is routed through the overridable ``_guardrail_input_text``
        shaper hook (default: identity — full rendered prompt).
        """
        output_text = self._guardrail_output_text(response)
        _fs = _current_fallback_state.get(None)
        _source = (
            ((_fs or {}).get("rendered_input"))
            or getattr(self, "_last_inference_input", "")
            or ""
        )
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

        # M7: un-defer the session logger now the ctx bridge is installed (by
        # either InferencerBase.ainfer's enter_run or the @bridge_entrypoint
        # decorator on CLI/streaming overrides). This is the single seam both
        # paths converge on, before any InferenceInput logging below.
        self._ensure_ctx_workspace_logger()
        self._propagate_to_children()

        # Capture original input BEFORE preprocessing for retry_with_original mode
        original_input = inference_input
        self._last_inference_input = inference_input

        if self.input_preprocessor is not None:
            inference_input = self.input_preprocessor(inference_input)

        # Template rendering (opt-in: only when template_manager is set).
        # Round-7 invariant: conditional kwarg pass — see _infer_single.
        if _extra_feed is not None:
            inference_input = self._render_prompt(
                inference_input, extra_feed=_extra_feed
            )
        else:
            inference_input = self._render_prompt(inference_input)

        # v5 Fix #1 — capture POST-render prompt as closure-local; see
        # sync sibling above for rationale. Published via _fallback_state
        # below; consumed by `_render_guardrail_prompt` (Fix #1) and by
        # `_recovery_wrapper` (Fix #4).
        rendered_input = inference_input

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
                resume_result = self.response_post_processor(
                    self._normalize_for_post_processor(resume_result)
                )
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
        finally:
            _current_fallback_state.reset(token)

        # v5 Phase 1.1 — pair the InferenceResponse parts file with this
        # call's InferenceInput via the shared call_id name-hint.
        self.log_debug(
            inference_response,
            "InferenceResponse",
            is_artifact=True,
            **_corr,
        )

        # Output finalization (promote deliverables, write summary)
        inference_response = self._finalize_output(inference_response)

        # Update state graphs from response
        if self.state_graphs:
            self.update_state_graphs(inference_response)

        if self.response_post_processor is not None:
            post_input = self._normalize_for_post_processor(inference_response)
            processed_response = self.response_post_processor(post_input)
            self.log_debug(
                processed_response,
                "PostProcessedResponse",
                is_artifact=True,
                **_corr,
            )
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
            self._init_call_state(inference_input)
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

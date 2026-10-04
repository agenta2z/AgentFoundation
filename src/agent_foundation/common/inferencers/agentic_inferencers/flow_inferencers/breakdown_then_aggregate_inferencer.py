"""BreakdownThenAggregateInferencer — diamond-shaped WorkGraph-based inferencer.

Breaks a query into sub-queries, runs workers in parallel via a per-attempt
WorkGraph (``_BtaGraph``), and optionally aggregates results.
"""

import asyncio
import functools
import inspect
import json
import logging
import os
import re
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import attr as attr_mod
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.bta_checkpoints import (  # noqa: F401 (the errors are re-exported)
    BtaResumeCorruptionError,
    BtaResumeIdentityMismatch,
    BtaWorkspaceBusyError,
    build_header,
    CheckpointLease,
    has_root_artifacts,
    has_worker_results,
    header_mismatches,
    LEASE_FILE,
    MANIFEST_FILE,
    plan_record,
    read_manifest,
    remove_manifest,
    RESUME_IDENTITY_POLICIES,
    TRUST_LEGACY,
    UnverifiedLegacyBtaResumeError,
    write_manifest,
)
from agent_foundation.common.inferencers.inferencer_base import (
    _FreshCloneFactory,
    InferencerBase,
    resolve_stage,
    ResolvedStage,
)
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    BtaCallSummary,
    discard_result,
    enter_run,
    exit_run,
    frame_for,
    invocation_of,
    NodeOutcomeState,
    open_invocation,
    publish_result,
    read_result,
    RenderedTaskContractState,
    ResourceLedger,
    RuntimeKey,
    StageOwnershipError,
    UncertifiedConcurrentUseError,
)
from agent_foundation.common.inferencers.run_context.resume_identity import (
    ResumeIdentityUnavailableError,
)
from agent_foundation.common.inferencers.template_defaults import (
    AGGREGATION_DEFAULTS,
    BREAKDOWN_TEMPLATE_DEFAULTS,
)
from agent_foundation.common.inferencers.template_feed_scope import (
    publish_child_template_feed,
)
from attr import attrib, attrs
from rich_python_utils.common_objects.workflow.common.result_pass_down_mode import (
    ResultPassDownMode,
)
from rich_python_utils.common_objects.workflow.common.step_result_save_options import (
    ResumeMode,
    StepResultSaveOptions,
)
from rich_python_utils.common_objects.workflow.workgraph import WorkGraph, WorkGraphNode
from rich_python_utils.common_utils.async_utils import maybe_await

if TYPE_CHECKING:
    from agent_foundation.common.inferencers.inferencer_workspace import (
        InferencerWorkspace,
    )


_logger = logging.getLogger(__name__)


# Transient errors worth retrying for BTA WorkGraph nodes (breakdown, workers,
# aggregator). Programming errors (TypeError, AttributeError, ValueError from
# parsers, etc.) deliberately fall through and surface immediately so they're
# not masked by retry storms.
TRANSIENT_RETRY_EXCEPTIONS = (
    TimeoutError,  # built-in (also raised by retry helper itself)
    asyncio.TimeoutError,  # asyncio's own (subclass of TimeoutError on 3.11+; listed for safety)
    ConnectionError,  # covers BrokenPipeError, ConnectionResetError, ConnectionRefusedError
    OSError,  # covers EPIPE, ECONNRESET, file-descriptor issues from CLI subprocesses
)


def parse_numbered_list(text: str) -> List[str]:
    """Parse a numbered list from text output.

    Handles formats like:
        1. Query one
        2. Query two
        1) Query one
        - Query one
    """
    lines = text.strip().split("\n")
    queries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip common list prefixes
        for prefix_pattern in [
            # "1. ", "2. ", etc.
            lambda s: s.split(". ", 1)[1]
            if (s.split(".")[0].strip().isdigit() and ". " in s)
            else None,
            # "1) ", "2) ", etc.
            lambda s: s.split(") ", 1)[1]
            if (s.split(")")[0].strip().isdigit() and ") " in s)
            else None,
            # "- " bullet
            lambda s: s[2:] if s.startswith("- ") else None,
            # "* " bullet
            lambda s: s[2:] if s.startswith("* ") else None,
        ]:
            parsed = prefix_pattern(line)
            if parsed is not None:
                queries.append(parsed.strip())
                break
    return queries


# ---------------------------------------------------------------------------
# Conflict detection helpers for promote_worker_deliverables
# ---------------------------------------------------------------------------

# Re-export generic helpers for backward compatibility with tests that import from here
from rich_python_utils.path_utils.path_listing import (
    canonicalize_text as _canonicalize_text,
    find_conflicting_and_agreed_files,
    group_conflicts_by_parent,
    hash_file_canonical as _sha256_of_file_canonical,
    safe_copy_agreed,
)


def _detect_conflicts_and_promote(
    deliverables_dst,
    children_dir,
    candidate_subdirs=("outputs",),
):
    """Detect deliverable conflicts across workers and auto-promote agreed files.

    Thin wrapper around the generic ``find_conflicting_and_agreed_files``.
    Resolves each worker's output root (``outputs/`` — the deliverable set per
    the Part 2 two-axis model) and delegates to the generic diff + copy helpers.
    """
    roots = []
    root_names = []
    for worker_name in sorted(os.listdir(children_dir)):
        worker_dir = os.path.join(children_dir, worker_name)
        if not os.path.isdir(worker_dir):
            continue
        for sub in candidate_subdirs:
            candidate = os.path.join(worker_dir, sub)
            if os.path.isdir(candidate) and os.listdir(candidate):
                roots.append(candidate)
                root_names.append(worker_name)
                break

    agreed, conflicts = find_conflicting_and_agreed_files(roots, root_names)

    # Auto-promote agreed files
    copied = safe_copy_agreed(agreed, deliverables_dst, skip_existing=True)
    for entry in agreed:
        # Add first abs_path for safe_copy_agreed (it needs a source)
        if "abs_path" not in entry:
            root_idx = root_names.index(entry["source_roots"][0])
            entry["abs_path"] = os.path.join(roots[root_idx], entry["path"])

    # Remap field names for BTA compatibility (source_roots → source_workers)
    deliverables_promoted = []
    for entry in agreed:
        deliverables_promoted.append(
            {
                "path": entry["path"],
                "size": entry["size"],
                "sha256": entry["sha256"],
                "source_workers": entry["source_roots"],
            }
        )
        if entry["path"] in copied:
            _logger.info(
                "Auto-promoted %s (%d bytes, agreed by %d worker(s))",
                entry["path"],
                entry["size"],
                len(entry["source_roots"]),
            )

    # Remap conflict field names (root_name → worker)
    deliverables_with_conflicts = {}
    for rel_path, instances in conflicts.items():
        deliverables_with_conflicts[rel_path] = [
            {**inst, "worker": inst.pop("root_name")} for inst in instances
        ]
        _logger.warning(
            "Conflict detected on %s — %d distinct versions",
            rel_path,
            len({i["sha256"] for i in deliverables_with_conflicts[rel_path]}),
        )

    return deliverables_promoted, deliverables_with_conflicts


def make_upstream_injecting_aggregator_prompt_builder():
    """DEPRECATED: prefer setting ``BTA.inject_upstream_artifacts_to_aggregator=True``
    instead of wiring this factory as ``aggregator_prompt_builder``.

    Both produce identical behavior: worker outputs published as the
    aggregator's ``upstream_artifacts`` feed variable, breakdown's
    ``aggregation_guidance`` (if captured) forwarded as
    ``aggregation_guidance`` (see ``_inject_aggregator_extra_feed``), and the
    original BTA query returned as the aggregator's ``inference_input``
    (rendered into ``{{ input }}`` by the wrapper).

    The class-level flag is the canonical pattern (mirrors MFDual's
    ``inject_upstream_artifacts``); this factory is kept for backward
    compatibility with YAMLs that still wire
    ``aggregator_prompt_builder: UpstreamInjectingAggregatorPromptBuilder``.
    """

    def _builder(
        worker_results,
        original_query=None,
        worker_output_paths=None,
        bta=None,
        worker_failures=None,
    ):
        if bta is not None:
            bta._inject_aggregator_extra_feed(
                align_worker_results(worker_results, worker_failures),
                (
                    align_worker_results(worker_output_paths, worker_failures)
                    if worker_output_paths
                    else None
                ),
                worker_failures=worker_failures,
            )
        return original_query or ""

    _builder.resume_identity = {"factory": "upstream_injecting_aggregator"}
    return _builder


def make_conflict_aware_prompt_builder(
    conflict_resolution_mode="delegate_to_aggregator",
    candidate_subdirs=("outputs",),
):
    """Factory: returns an aggregator_prompt_builder that detects conflicts.

    The closure accepts BTA's hook signature (with optional ``bta=`` kwarg):
        (worker_results, original_query=..., worker_output_paths=..., bta=...,
         worker_failures=...) -> str

    In ``delegate_to_aggregator`` mode:
    1. Walks worker outputs, hashes files, categorizes as agreed/conflicting
    2. Auto-promotes agreed files to deliverables_dst
    3. Publishes structured data (``deliverables_promoted``, ``deliverables_with_conflicts``,
       ``deliverables_dst``, ``worker_summaries``) as the aggregator's template feed
       (see ``publish_child_template_feed``) so the aggregator template gets them as
       top-level ``{{ vars }}``
    4. Returns the original query as ``{{ input }}``
    """

    def _builder(
        worker_results,
        original_query=None,
        worker_output_paths=None,
        bta=None,
        worker_failures=None,
    ):
        worker_summaries = [str(r) for r in worker_results]
        failures = worker_failures or {}
        aggregator = None if bta is None else bta.build_aggregator()

        # Build "summary text" — for aggregators with local file access, pass
        # paths only (avoids inlining 100KB+ of worker text per result, which
        # would exceed OS ARG_MAX when the prompt is passed to subprocess).
        # For non-local aggregators (e.g., RovoChat), inline full text.
        agg_has_local = aggregator is not None and getattr(
            aggregator, "has_local_access", False
        )

        def _format_result(idx, res, path):
            if idx in failures:
                return f"### Upstream Outcome {idx + 1}\n(failed: {failures[idx]})"
            if agg_has_local and path:
                return f"### Upstream Outcome {idx + 1}\n(See file: `{path}`)"
            return f"### Upstream Outcome {idx + 1}\n{res}"

        slots = align_worker_results(worker_results, failures)
        paths = (
            align_worker_results(worker_output_paths, failures)
            if worker_output_paths
            else [None] * len(slots)
        )
        default_text = "\n\n".join(
            _format_result(i, r, paths[i] if i < len(paths) else None)
            for i, r in enumerate(slots)
        )

        # Inject upstream_artifacts EARLY — before any return path — so the
        # aggregation preamble template always has worker content available.
        if aggregator is not None:
            publish_child_template_feed(
                aggregator,
                "aggregator",
                {"upstream_artifacts": default_text},
            )

        if conflict_resolution_mode == "last_writer_wins":
            return original_query or ""

        if not worker_output_paths or not any(worker_output_paths):
            return original_query or ""

        first_path = next((p for p in worker_output_paths if p), None)
        if first_path is None:
            return original_query or ""

        cur = os.path.abspath(first_path)
        while cur and os.path.basename(cur) != "children":
            parent = os.path.dirname(cur)
            if parent == cur:
                cur = None
                break
            cur = parent
        if cur is None:
            return original_query or ""

        children_dir = cur
        ws_root = os.path.dirname(children_dir)
        # Part 2 two-axis model: deliverables land directly in ``outputs/``
        # (``final_deliverables/`` retired).
        deliverables_dst = os.path.join(ws_root, "outputs")
        os.makedirs(deliverables_dst, exist_ok=True)

        promoted, conflicts = _detect_conflicts_and_promote(
            deliverables_dst,
            children_dir,
            candidate_subdirs,
        )

        conflicts_grouped = group_conflicts_by_parent(conflicts, depth=2)

        if aggregator is not None:
            publish_child_template_feed(
                aggregator,
                "aggregator",
                {
                    "deliverables_promoted": promoted,
                    "deliverables_with_conflicts": [
                        {"path": rp, "candidates": cands}
                        for rp, cands in conflicts.items()
                    ],
                    "conflicts_grouped_by_parent": conflicts_grouped,
                    "deliverables_dst": deliverables_dst,
                    "worker_summaries": worker_summaries,
                },
            )

        return original_query or ""

    _builder.resume_identity = {
        "factory": "conflict_aware_aggregator",
        "conflict_resolution_mode": conflict_resolution_mode,
        "candidate_subdirs": list(candidate_subdirs),
    }
    return _builder


class _FailedWorkerSentinel:
    """Legacy quorum marker for a terminally failed worker.

    Workers now return a failure outcome record instead (see ``_wrap_outcome``);
    this class remains only so a checkpoint pickled by an older build still loads
    and is treated as a failed worker that re-runs on resume.
    """

    __slots__ = ("node_name", "error_type", "error_message")

    def __init__(self, node_name: str, error_type: str, error_message: str) -> None:
        self.node_name = node_name
        self.error_type = error_type
        self.error_message = error_message

    def __repr__(self) -> str:
        return (
            f"<_FailedWorkerSentinel name={self.node_name!r} "
            f"{self.error_type}={self.error_message[:80]!r}>"
        )

    def __str__(self) -> str:
        return f"(failed: {self.error_type}: {self.error_message})"


# Worker results cross the WorkGraph (and its JSON checkpoints) as plain marked
# dicts so the aggregator can pair each result with its sub-query by declaration
# index regardless of completion order, failures, or resume.
_OUTCOME_MARKER = "__bta_worker_outcome__"


def _wrap_outcome(
    index: int,
    node_name: Optional[str],
    value: Any = None,
    failure: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        _OUTCOME_MARKER: 1,
        "index": index,
        "node_name": node_name,
        "value": value,
        "failure": failure,
    }


def _is_outcome(value: Any) -> bool:
    return isinstance(value, dict) and value.get(_OUTCOME_MARKER) == 1


def _as_outcome(value: Any, index: int, node_name: Optional[str]) -> Dict[str, Any]:
    """``value`` if it is already an outcome, else a success outcome for worker
    ``index`` (a bare value is a checkpoint written before outcome records)."""
    if _is_outcome(value):
        return value
    return _wrap_outcome(index, node_name, value=value)


def _rerun_failed_loader(load_result: Callable) -> Callable:
    """Wrap a worker node's ``load_result`` so a checkpointed failure (outcome
    or legacy sentinel) reads as not loaded and the worker re-runs."""

    @functools.wraps(load_result)
    def _load(*args, **kwargs):
        loaded, value = load_result(*args, **kwargs)
        if not loaded:
            return loaded, value
        if isinstance(value, _FailedWorkerSentinel) or (
            _is_outcome(value) and value.get("failure") is not None
        ):
            return False, None
        return loaded, value

    return _load


def _outcome_passdown(get_args: Callable, index: int, node_name: str) -> Callable:
    """Wrap a worker node's ``_get_args_for_downstream`` so the aggregator always
    receives an outcome carrying this node's ``index``: a loaded checkpoint
    skips the worker callable, and fan-in delivers in completion order."""

    @functools.wraps(get_args)
    def _get_args(result, args, kwargs):
        return get_args(_as_outcome(result, index, node_name), args, kwargs)

    return _get_args


def _ordered_outcomes(
    values: Sequence[Any], n: int
) -> Tuple[List[Any], Dict[int, str]]:
    """Place worker outcomes at their declaration index.

    Returns ``(slots, failures)``: ``slots[i]`` is worker ``i``'s value (``None``
    when it failed or never reported) and ``failures`` maps each such index to a
    reason. Raises on a value that is not an outcome or an out-of-range index.
    """
    slots: List[Any] = [None] * n
    failures: Dict[int, str] = dict.fromkeys(range(n), "missing")
    for value in values:
        if not _is_outcome(value):
            raise TypeError(
                f"BTA aggregation expected worker outcome records, got {type(value).__name__}"
            )
        idx = value["index"]
        if not isinstance(idx, int) or not 0 <= idx < n:
            raise ValueError(f"BTA worker outcome index {idx!r} outside [0, {n})")
        if value.get("failure") is not None:
            failures[idx] = str(value["failure"])
            continue
        slots[idx] = value.get("value")
        failures.pop(idx, None)
    return slots, failures


def align_worker_results(
    survivors: Sequence[Any], worker_failures: Optional[Mapping[int, str]]
) -> List[Any]:
    """Re-expand the compacted survivor list a custom ``aggregator_prompt_builder``
    receives into declaration order, with ``None`` at each failed index."""
    failures = worker_failures or {}
    it = iter(survivors)
    return [
        None if i in failures else next(it, None)
        for i in range(len(survivors) + len(failures))
    ]


def _call_prompt_builder(builder: Callable, worker_results: List[Any], **kwargs):
    """Call a custom ``aggregator_prompt_builder`` with the subset of ``kwargs``
    its signature accepts (all of them when it takes ``**kwargs``)."""
    try:
        params = inspect.signature(builder).parameters
    except (TypeError, ValueError):
        return builder(worker_results, **kwargs)
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    return builder(worker_results, **kwargs)


@attrs(frozen=True, slots=True)
class _BtaCallRecord:
    """Where one BTA call reads and writes, fixed when the call starts.

    ``effective_workspace`` is the workspace the call resolves under its own ctx;
    ``checkpoint_root`` is that workspace's ``checkpoints/`` when there is one,
    else ``checkpoint_dir``. Path sites read the record, never the ambient
    workspace: under a child's ctx that carries a ``workspace_override``, the
    ambient workspace is the child's.
    """

    effective_workspace: Optional["InferencerWorkspace"] = attrib()
    checkpoint_dir: Optional[str] = attrib()
    checkpoint_root: Optional[str] = attrib()


@attrs(slots=True, eq=False)
class _BtaManifest:
    """The call's resume manifest (§5.12): ``fresh`` (written for this call),
    ``resume`` (verified, possibly with a committed plan), ``legacy`` (unverified
    checkpoints resumed under ``trust_legacy``) or ``unvouched`` (a call that
    doesn't resume, over an earlier call's checkpoints). Nothing is written in the
    last two."""

    root: str = attrib()
    mode: str = attrib()
    header: Dict[str, Any] = attrib()
    plan: Optional[Dict[str, Any]] = attrib(default=None)


@attrs(slots=False)
class _BtaGraph(WorkGraph):
    """One attempt's graph. The engine state a run used to keep on the BTA
    (``start_nodes``, the expansion limits, node queues, the event callback) lives
    here, so overlapping runs of one BTA never share it (B24). Logging and the
    graph-level result path go through the owning BTA, so every engine record
    lands in its session log under its id and name, and checkpoints keep their
    paths."""

    owner: Any = attrib(default=None, kw_only=True)
    start_nodes = attrib(factory=list)

    def log(self, *args, **kwargs):
        return self.owner.log(*args, **kwargs)

    def _get_result_path(self, *args, **kwargs):
        return self.owner._get_result_path(*args, **kwargs)


# How a ``_BtaGraph`` gets each ``WorkGraph`` field. Set per attempt:
_BTA_GRAPH_PER_ATTEMPT = frozenset(
    {
        "start_nodes",
        "name",
        "use_async",
        "max_expansion_depth",
        "max_total_nodes",
        "subgraph_registry",
    }
)
# Copied from the owning BTA, because the engine reads them through the graph:
_BTA_GRAPH_COPIED = frozenset(
    {
        "enable_result_save",
        "resume_with_saved_results",
        "checkpoint_mode",
        "_result_root_override",
        "max_concurrency",
        "group_max_concurrency",
    }
)
# Engine settings a BTA does not configure, left at the ``WorkGraph`` defaults:
_BTA_GRAPH_ENGINE_DEFAULTS = frozenset(
    {
        "node_cls",
        "verbose_repr",
        "enable_optional_post_process",
        "result_pass_down_mode",
        "unpack_single_result",
        "ignore_stop_flag_from_saved_results",
        "executor",
    }
)
# Every ``Debuggable`` field is left at its default: the graph logs through its owner.


def _bta_graph(owner, attempt: "_BtaAttempt") -> _BtaGraph:
    """An empty graph for ``attempt``, configured like ``owner``'s own graph."""
    copied = {
        field.name: getattr(owner, field.name)
        for field in attr_mod.fields(WorkGraph)
        if field.name in _BTA_GRAPH_COPIED
    }
    init_names = {f.name: f.alias for f in attr_mod.fields(WorkGraph) if f.init}
    graph = _BtaGraph(
        owner=owner,
        name=owner._effective("bta_node_name"),
        use_async=attempt.use_async,
        max_expansion_depth=1,
        max_total_nodes=max(owner.max_breakdown or 100, 100) + 2,
        subgraph_registry=_subgraph_registry(owner),
        **{
            init_names[name]: value
            for name, value in copied.items()
            if name in init_names
        },
    )
    for name, value in copied.items():
        if name not in init_names:
            object.__setattr__(graph, name, value)
    return graph


def _subgraph_registry(owner) -> Dict[str, Callable[[Any], Any]]:
    """The factories ``WorkGraph``'s registry-based expansion reconstruction calls.

    On resume, ``_reconstruct_graph_expansions()`` looks up the expansion id here
    and calls the factory — BEFORE the breakdown fn runs, so the breakdown fn's own
    ``_original_query`` threading is too late. Each factory sources sub_queries from
    the promoted breakdown checkpoint (``_load_promoted_breakdown``) — the same
    decomposed_subtasks.json the aggregator restores its guidance from — and threads
    the attempt's original request, so the reconstructed aggregator node receives it
    exactly as the fresh path does (without it the aggregator's "## Original User
    Request" slot renders blank on resume).
    """
    return {
        "bta_diamond": lambda exp_id: owner._rebuild_subgraph(),
        "bta_workers": lambda exp_id: owner._rebuild_subgraph(),
    }


@attrs(slots=True, eq=False)
class _BtaAttempt:
    """One attempt of one BTA run: the run's state that used to live on the
    instance. A run has one attempt, or on the async path one more per interactive
    "rerun" review; nothing here outlives its attempt.
    """

    original_query: Any = attrib()
    # The node-function builders build async functions for an async graph.
    use_async: bool = attrib()
    # The promoted breakdown checkpoint, memoized once it reads usable.
    promoted_breakdown: Optional[Tuple[List, Optional[str]]] = attrib(default=None)
    # The breakdown's ``aggregation_guidance`` for the aggregator's feed.
    aggregation_guidance: Optional[str] = attrib(default=None)
    # The runtime sub-queries a subclass derives per call (MFI's propagated input),
    # in place of ``predefined_sub_queries``.
    effective_sub_queries: Optional[List] = attrib(default=None)
    topology_emitted: bool = attrib(default=False)
    pending_topology: Any = attrib(default=None)
    # The child names of the workers this attempt dispatched, by worker index.
    worker_child_names: List[str] = attrib(factory=list)
    # The task contract each successful worker published, by worker index.
    worker_contracts: Dict[int, RenderedTaskContractState] = attrib(factory=dict)
    # The attempt's graph (``_install_breakdown_node``).
    graph: Optional[_BtaGraph] = attrib(default=None)
    # The stages this attempt owns, closed when it ends.
    ledger: ResourceLedger = attrib(factory=ResourceLedger)
    owned_ids: set = attrib(factory=set)


@attrs(slots=False)
class BreakdownThenAggregateInferencer(InferencerBase):
    """Diamond-shaped inferencer: breakdown → parallel workers → aggregate.

    Each attempt runs its own ``WorkGraph`` (``_BtaGraph``), configured from this
    definition; the inferencer itself is not a graph.

    Uses expansion-driven graph construction: a single "breakdown" start
    node returns a ``GraphExpansionResult`` that dynamically attaches
    worker and aggregator nodes at runtime.

    Graph structure (expansion-driven diamond)::

        start_node:   breakdown                                     (sole start node)
                         ↓ GraphExpansionResult
        expanded:     worker_0, worker_1, ..., worker_N             (parallel fan-out)
                         \\        |            /
                            aggregator                              (fan-in)

    Concurrency control:
        - ``ainfer()`` executes all workers concurrently via ``asyncio.gather()``.
        - ``infer()`` executes workers sequentially in a for-loop.
        - Set ``max_concurrency`` to limit how many workers run in parallel in the
          async path. Uses a sliding-window ``asyncio.Semaphore`` (not batched),
          so as soon as one worker finishes, the next one starts. ``None`` (default)
          means unlimited parallelism. The semaphore gates each node's own
          computation only (downstream propagation runs outside the slot), so it
          composes with an ``aggregator_inferencer``.

    Worker results reach the aggregator paired with their sub-queries by
    declaration index, independent of completion order, failures and resume.

    Predefined sub-queries mode:
        Set ``predefined_sub_queries`` to bypass the LLM-driven breakdown phase.
        Sub-queries are resolved as follows:

        - ``List[str]`` or ``List[dict]``: used directly as sub_queries.
          ``breakdown_inferencer`` is not required.
        - ``str`` (single query): replicated to N workers where
          ``N = max_breakdown or max_concurrency or 1``.
          Useful for parallel sampling or diverse perspectives on one query.

        ``max_breakdown`` still caps the resolved sub-query list.
        A saved checkpoint (``resume_with_saved_results``) takes priority and
        overrides ``predefined_sub_queries`` when found (checkpoint is loaded first).
        Setting ``breakdown_only=True`` alongside ``predefined_sub_queries`` is
        contradictory — ``breakdown_only`` will be ignored with a warning.
    """

    # === Slot-based template role defaults (consumed by config_utils._walk) ===
    # Each entry: slot field name → InferencerTemplateDefaults bundle.
    # Hydra-time injection fills missing template fields on the slot child;
    # user-supplied values always win (per-key for dicts, scalar fill for
    # template_root_space/template_key). Subclasses (MultiFlow) inherit via
    # MRO. See ``template_defaults`` module for the named bundles.
    # Domain meaning for the blocks this class registers (see
    # ``InferencerBase.BLOCK_PARSERS``). Registered by METHOD NAME because the
    # transform reads ``worker_query_fields`` off the instance.
    #
    # The TRANSFORM only — the tolerant text->dict extraction stays in
    # ``_parse_json_subtasks``, which also accepts an unlabeled fence, a bare
    # ``{..."subtasks"...}`` scan and a backtick-repair retry that the generic
    # label-anchored extractor does not.
    BLOCK_PARSERS: ClassVar[Dict[str, Any]] = {
        "decomposed_subtasks": "_subtasks_from_fence_dict",
    }

    SLOT_DEFAULTS: ClassVar[Dict[str, Any]] = {
        "breakdown_inferencer": BREAKDOWN_TEMPLATE_DEFAULTS,
        # Full structured-aggregation triplet. Refactor 12 made the version-
        # to-default fallback safe (a missing aggregation.jinja2 falls back
        # to default.jinja2 instead of literal-corrupting the prompt), and
        # Refactor 13's template_version + None-keyed template_variables
        # lets the YAML drop per-key entries. Both plan and exec BTA
        # aggregator slots receive the same defaults; per-namespace
        # aggregation files (plan/.../task_instructions/aggregation.jinja2,
        # implementation/.../task_instructions/aggregation.jinja2) provide
        # the role-correct content for each.
        "aggregator_inferencer": AGGREGATION_DEFAULTS,
    }

    # Feed keys this BTA writes into its aggregator's feed itself.
    OWNED_FEED_KEYS: ClassVar[Tuple[str, ...]] = (
        "upstream_artifacts",
        "aggregation_guidance",
    )

    # Slot fields to drop when a slot is filled by a re-roled copy of the
    # inferencer this BTA fans out: its output judge and fallback target its own
    # response format, not the breakdown's JSON subtasks.
    SELF_SLOT_DROPS: ClassVar[Dict[str, Tuple[str, ...]]] = {
        "breakdown_inferencer": ("output_guardrail_inferencer", "fallback_inferencer"),
    }

    # Keyword arguments every worker call passes itself.
    _RESERVED_WORKER_INFERENCE_ARGS: ClassVar[FrozenSet[str]] = frozenset(
        {"inference_config", "run_context"}
    )

    # The aggregation stage gets a canonical runtime workspace via the WorkGraph
    # node named "aggregator" (see _build_subgraph_spec / agg_inf._workspace =
    # child("aggregator")). Skipping generic attr-based propagation for the
    # direct child slot `aggregator_inferencer` avoids a duplicate,
    # usually-empty `children/aggregator_inferencer/` directory.
    _workspace_propagation_skip: frozenset = frozenset({"aggregator_inferencer"})

    # === Breakdown ===
    breakdown_inferencer: InferencerBase = attrib(default=None)
    max_breakdown: Optional[int] = attrib(default=None)
    breakdown_parser: Optional[Callable] = attrib(default=None)

    min_successful_workers: int = attrib(default=0)
    """v4 Phase 3.1 — K-of-N quorum for BTA workers (parent of MultiFlow's
    ``min_successful_flows``). Default 0 = disabled (backward-compatible:
    any worker failure raises and fails the BTA, the historical behavior).
    Set to N >= 1 to require at least N successful workers; a failed worker's
    slot reaches the aggregator labeled as failed (index-aligned, never
    shifting later workers' outputs).

    When fewer than N workers survive, the aggregation step raises rather than
    synthesizing a degraded aggregation (fail loud); with no aggregator the
    same check runs when the BTA returns.

    Failed workers emit ``node_status: error`` so the UI renders them red.
    """
    # Built-in breakdown format: "auto" (default, numbered list fallback),
    # "json_subtasks" (task_breakdown JSON format), "numbered_list" (explicit).
    # When set to "json_subtasks", uses _parse_json_subtasks() instead of
    # breakdown_parser. breakdown_parser takes precedence if both are set.
    breakdown_format: str = attrib(default="auto", kw_only=True)
    # Which subtask fields to include in worker queries (for json_subtasks format).
    worker_query_fields: tuple = attrib(default=("description", "todos"), kw_only=True)

    # === Per-query worker ===
    # ``worker_inferencers`` is the SINGLE source of per-subtask workers. Accepted shapes:
    #   - Callable(sub_query, index) -> InferencerBase: homogeneous / dynamic-per-index.
    #   - dict[str, Callable | functools.partial | LazyConfigFactory]: heterogeneous —
    #     maps task type -> factory; ``partial``/``LazyConfigFactory`` entries are called
    #     no-args for a fresh instance. ``"_default"`` may be a string referencing another
    #     key. Requires ``task_type_arg_name`` and a parser returning List[dict] with "args".
    #   - config recipe ``{_target_: <Worker>, ...}``: the ``lazy_config_factory`` metadata
    #     makes the RPU config walker wrap it in a ``LazyConfigFactory`` (same machinery as
    #     ``*_factory`` fields) — fresh instance per subtask, no magic ``_factory`` suffix.
    #   - list[InferencerBase]: a pre-built static-K list, served round-robin.
    # Every subtask gets its own FRESH worker (object identity = worker ID), owned and
    # closed by its attempt; a static list's instances are borrowed. MFI sets this
    # internally to its flow-builder closure.
    worker_inferencers: Any = attrib(
        default=None, kw_only=True, metadata={"lazy_config_factory": True}
    )
    # Extra keyword arguments passed to every worker ``ainfer`` / ``infer`` call.
    worker_inference_args: Dict[str, Any] = attrib(factory=dict, kw_only=True)
    # Upper bound on each worker query's length, checked after breakdown and
    # before any worker runs. An oversized query raises; it is never truncated.
    max_worker_query_chars: Optional[int] = attrib(default=None, kw_only=True)

    # When set, enables heterogeneous workers. Each sub_query item can be a dict
    # {"query": str, "args": {...}}. The value of args[task_type_arg_name] selects
    # which worker factory to use from a dict-typed ``worker_inferencers``.
    task_type_arg_name: Optional[str] = attrib(default=None, kw_only=True)
    # Controls whether subtasks with multiple "todos" are expanded into one
    # worker per todo. Accepts bool (all types) or dict {task_type: bool}
    # for per-type control.
    expand_todos_to_workers: Union[bool, Dict[str, bool]] = attrib(
        default=False, kw_only=True
    )

    # === Aggregation ===
    aggregator_inferencer: Optional[InferencerBase] = attrib(default=None)
    aggregator_prompt_builder: Optional[Callable] = attrib(default=None)
    # ----- Upstream artifact injection to aggregator (modern slot semantics) -----
    # When True (default): BTA pushes formatted worker outputs into
    # ``aggregator_inferencer.template_extra_feed["upstream_artifacts"]``,
    # forwards the attempt's ``aggregation_guidance`` (captured at breakdown-parse
    # time) to ``template_extra_feed["aggregation_guidance"]``, and returns
    # the original BTA query as the aggregator's ``inference_input`` (which
    # the wrapper template renders into ``{{ input }}``).
    #
    # When False (legacy opt-out): BTA formats worker outputs as
    # ``### Upstream Outcome N\n<output>`` and returns that text as the aggregator's
    # ``inference_input`` directly — meaning ``{{ input }}`` ends up
    # containing the worker outputs and ``{{ upstream_artifacts }}`` is
    # undefined. Use this for template-less aggregators (no
    # ``template_root_space``) or for aggregator wrappers that don't have a
    # ``{{ upstream_artifacts }}`` slot. Older topologies that put worker
    # outputs in the wrapper's ``{{ input }}`` slot must use this opt-out.
    #
    # The default flipped from False -> True after audit confirmed every
    # in-tree consumer either (a) has a wrapper that consumes
    # ``{{ upstream_artifacts }}`` via ``task_preamble: aggregation``, or
    # (b) uses a custom ``aggregator_prompt_builder`` that ignores this
    # flag, or (c) uses ``MockAggregator`` which doesn't render templates.
    #
    # Note: a custom ``aggregator_prompt_builder`` (when set) takes
    # precedence over this flag — the prompt_builder is fully responsible
    # for building the aggregator's input in that case.
    #
    # Mirrors MFDual's ``inject_upstream_artifacts`` flag at the BTA level.
    inject_upstream_artifacts_to_aggregator: bool = attrib(default=True, kw_only=True)

    # === Checkpoint ===
    checkpoint_dir: Optional[str] = attrib(default=None)
    # This call's paths (``_bta_call``), its current attempt (``_open_attempt``),
    # and the summary of the run that produced its result (``last_call_summary``).
    _BTA_CALL = RuntimeKey("BreakdownThenAggregateInferencer.call")
    _BTA_MANIFEST = RuntimeKey("BreakdownThenAggregateInferencer.manifest")
    _BTA_ATTEMPT = RuntimeKey("BreakdownThenAggregateInferencer.attempt")
    # This call's aggregator (``build_aggregator``), resolved once per call.
    _BTA_AGGREGATOR = RuntimeKey("BreakdownThenAggregateInferencer.aggregator")
    _BTA_SUMMARY = RuntimeKey(
        "BreakdownThenAggregateInferencer.summary",
        compat={
            "_last_call_summary": "",
            "_worker_task_instructions": "selected_task_contract",
        },
    )
    _last_call_summary: Optional[BtaCallSummary] = attrib(
        default=None, init=False, repr=False
    )

    # === Workspace support (opt-in, overrides checkpoint_dir when set) ===
    # workspace: InferencerWorkspace — inherited from InferencerBase.
    #   Configure workspace layout on the InferencerWorkspace object directly,
    #   keeping workspace concerns out of BTA. Part 2 (two-axis model): ``outputs/``
    #   IS the deliverable set; there is no ``final_deliverables/`` subfolder.
    # The legacy `workspace_root: Optional[str]` shorthand was removed
    # 2026-05-05; pass `workspace=InferencerWorkspace(root="/path", ...)`.

    # === Concurrency ===
    # Maximum number of worker nodes to run in parallel during the fan-out
    # layer of the diamond graph. When set, creates an asyncio.Semaphore to
    # throttle concurrent worker execution (sliding window, not batched).
    # Only applies to the async path (ainfer). None means unlimited parallelism.
    # Copied to each attempt's graph. The semaphore gates node computation only,
    # so an aggregator never waits on a slot held by a worker.
    max_concurrency: Optional[int] = attrib(default=None)
    # Per-group node concurrency limits, copied to each attempt's graph
    # (``{group: limit}``).
    group_max_concurrency: Optional[Dict[str, int]] = attrib(default=None, kw_only=True)
    # The node's name: the default ``bta_node_name`` (session-log display name,
    # attempt graph name, graph-level result path).
    name: Optional[str] = attrib(default=None, kw_only=True)

    # Left out of the resume identity: the interactive channel, where checkpoints
    # live, and scheduling.
    _RESUME_IDENTITY_EXCLUDE: ClassVar[FrozenSet[str]] = frozenset(
        {
            "interactive",
            "checkpoint_dir",
            "max_concurrency",
            "group_max_concurrency",
            "worker_isolation_check",
            "resume_identity_policy",
        }
    )
    # How a resume treats checkpoints without a manifest (§5.12): ``"verify"``
    # refuses them (``UnverifiedLegacyBtaResumeError``); ``"trust_legacy"`` is the
    # logged emergency override that resumes them and never writes a manifest.
    resume_identity_policy: str = attrib(
        default="verify",
        kw_only=True,
        validator=attr_mod.validators.in_(RESUME_IDENTITY_POLICIES),
    )

    # === Interactive support ===
    interactive: Optional[Any] = attrib(default=None)
    # A parent hands this call's interactive handler as the ``interactive=`` call
    # keyword (``_effective("interactive")``); the configured field is the fallback.
    # A parent BTA hands a nested BTA worker its node name as ``bta_node_name=``
    # (``None`` included); without one the call runs under ``name``.
    _INVOCATION_KEYWORDS = MappingProxyType(
        {
            "interactive": RuntimeKey("BreakdownThenAggregateInferencer.interactive"),
            "bta_node_name": RuntimeKey(
                "BreakdownThenAggregateInferencer.bta_node_name"
            ),
        }
    )
    # A call keeps its state in its frame and its graph per attempt, so one instance
    # serves overlapping host calls; the purity ratchet verifies it.
    _HOST_PURE_CERTIFIED = True
    enable_checkpoint_sub_query_selection: bool = attrib(default=False)
    enable_checkpoint_results_review: bool = attrib(default=False)
    breakdown_only: bool = attrib(default=False)  # Stop after breakdown phase
    disable_aggregator: bool = attrib(default=False)  # Run workers but skip aggregation
    promote_worker_deliverables: bool = attrib(default=False, kw_only=True)

    # === Deliverable collection policy (no-aggregator fan-out) ===
    # Part 2 (two-axis model): ``outputs/`` IS the deliverable set (no
    # ``final_deliverables/`` subfolder, no deliverable flags). BTA promotes its
    # aggregator's ``outputs/`` up to its own ``outputs/`` via ``_finalize_output``
    # → ``_symlink_child_output`` regardless of any flag; workers are consumed
    # inputs and are NOT promoted (the aggregator is what promotes).
    # Subclass-local policy for boundary aggregation:
    deliverable_namespace_strategy: str = attrib(default="by_child_name", kw_only=True)
    deliverable_conflict_strategy: str = attrib(default="skip_existing", kw_only=True)
    # Which child workspace names to collect from (default: all worker_*).
    # The "fixer" extension below collects ALL boundaries; subclass-specific
    # filters can be set via YAML/kwargs.
    deliverable_collect_namespace_root: str = attrib(default="workers", kw_only=True)
    conflict_resolution_mode: str = attrib(default="last_writer_wins", kw_only=True)
    # When set, skips the LLM-driven breakdown phase entirely.
    # Accepts:
    #   - List[str]: each string becomes a worker query.
    #   - List[dict]: each dict has "query" and optional "args" fields
    #     (same format as produced by breakdown + json_subtasks parsing).
    #     Enables heterogeneous worker dispatch when task_type_arg_name is set.
    #   - str: single query — replicated to N workers where
    #     N = max_breakdown or max_concurrency or 1.
    #     Useful for parallel sampling / diverse perspectives on one query.
    # When None (default): normal LLM breakdown phase runs.
    # breakdown_inferencer is not required when predefined_sub_queries is set.
    # Note: resume_with_saved_results checkpoint takes priority over this field.
    predefined_sub_queries: Optional[Union[str, List]] = attrib(
        default=None, kw_only=True
    )

    # graph_reporter is inherited from InferencerBase (uniform Tier-2 propagation).

    # === Worker isolation check (Fix #5) ===
    # When True (default), _validate_worker_isolation() scans all worker
    # sub-trees after construction and logs a WARNING if any two workers
    # share a sub-inferencer instance (by Python id). Shared instances
    # cause cross-worker state pollution (workspace, session, prompt
    # history). Set to False to suppress (e.g., intentional sharing in
    # tests or when LazyConfigFactory guarantees fresh instances).
    worker_isolation_check: bool = attrib(default=True, kw_only=True)

    # Host inferencer types (registered aliases, dotted import paths or classes)
    # this BTA is designed to fan out when used as a ``bta_inferencer`` template.
    # Empty means any host.
    expected_parent_types: tuple = attrib(factory=tuple, converter=tuple, kw_only=True)

    # The task contract the last call outside a host ctx relayed from a worker
    # (publish-up; ``_select_contract``): the compat projection of its summary.
    # Backs ``_proposer_task_instructions``; see that method for why the aggregator
    # is NOT an acceptable substitute.
    _worker_task_instructions: str = attrib(default="", init=False, repr=False)

    # ------------------------------------------------------------------
    # Worker naming hook (overridable by subclasses)
    # ------------------------------------------------------------------

    def _proposer_task_instructions(self) -> str:
        """Task contract from a WORKER (this node's input side), via publish-up.

        Workers are resolved from a ``LazyConfigFactory`` per call and are not
        yielded by ``_iter_child_inferencers``, so there is no static path from here
        down to an author leaf. Instead each worker's published contract is
        harvested as it completes (``_harvest_worker_contract``), the run relays one
        (``_select_contract``), and a call outside a host ctx reports it here. Under
        a ctx a parent reads this node's published outcome instead.

        ``self.aggregator_inferencer`` is deliberately NOT a fallback: it renders the
        OUTPUT-side "aggregating/integrating the upstream outcomes" variant, so using
        it would hand a reviewer a merge brief in place of the task contract. When no
        worker reported one, return ``""`` and let the consumer omit the block.
        """
        return self._worker_task_instructions or ""

    def _worker_child_name(self, index: int) -> str:
        """Return the workspace child directory name for worker ``index``.

        Default: ``f"worker_{index}"``. Subclasses (e.g.,
        :class:`MultiFlowInferencer`) override to produce semantically
        meaningful names (``f"flow_{index}_workflow"``).

        Used by ``_build_subgraph_spec`` for both the on-disk workspace
        directory and the WorkGraph node name, and by
        ``_is_worker_child_name`` for boundary/deliverable filtering.
        """
        from agent_foundation.common.inferencers.inferencer_workspace import (
            indexed_child_name,
        )

        return indexed_child_name("worker", index)

    @staticmethod
    def _worker_backup_output_path(worker, worker_rc) -> Optional[str]:
        """``worker.resolve_output_path()`` as the worker resolves it: under its own
        run context ``worker_rc``. Under this BTA's context the worker's
        ``_workspace`` getter returns this BTA's published workspace, so a bare
        call would name this BTA's output file instead of the worker's."""
        if not hasattr(worker, "resolve_output_path"):
            return None
        token = enter_run(worker_rc) if worker_rc is not None else None
        try:
            return worker.resolve_output_path()
        finally:
            if token is not None:
                exit_run(token)

    async def _cross_flow_depart_if_tagged(self, worker) -> None:
        """Depart the cross-flow step barrier for ``worker`` if it is a coordinated flow.

        No-op for ordinary BTA workers (untagged) or when no rendezvous is active. The
        :class:`MultiFlowInferencer` subclass tags coordinated flow workers with
        ``_cross_flow_index`` and provides ``_resolve_rendezvous``. The rendezvous's
        ``leave`` is idempotent, so this worker-boundary safety net coexists with the
        LWI-level depart (which fires only when the worker actually runs its ``_ainfer``).
        """
        idx = getattr(worker, "_cross_flow_index", None)
        if idx is None:
            return
        resolver = getattr(self, "_resolve_rendezvous", None)
        rdv = resolver() if resolver is not None else None
        if rdv is not None:
            await rdv.leave(idx)

    def _is_worker_child_name(self, name: str) -> bool:
        """Return True if ``name`` matches a worker child directory name.

        Default: ``name.startswith("worker_")``. Subclasses that override
        ``_worker_child_name`` should also override this to match.

        Used by ``_finalize_response`` for deliverable boundary collection
        and worker deliverable promotion.
        """
        return name.startswith("worker_")

    def __attrs_post_init__(self):
        # InferencerBase.__attrs_post_init__ syncs self.workspace → self._workspace.
        # The legacy `workspace_root` shorthand was removed 2026-05-05.
        super().__attrs_post_init__()
        self._validate_worker_inference_args()

        if self._workspace is not None:
            self._workspace.ensure_dirs()

        # Auto-default output_path: derive from aggregator's output_path
        # (since the BTA's canonical output IS the aggregator's output,
        # symlinked via _symlink_child_output). Falls back to generic name
        # when no aggregator or no output_path is set on it.
        if not self.output_path:
            agg_out = (
                getattr(self.aggregator_inferencer, "output_path", None)
                if self.aggregator_inferencer
                else None
            )
            self.output_path = agg_out or "aggregation_report.md"

        if self.breakdown_inferencer is not None and hasattr(
            self.breakdown_inferencer, "template_extra_feed"
        ):
            for key in ("max_breakdown", "max_worker_query_chars"):
                if getattr(self, key) is not None:
                    self.breakdown_inferencer.template_extra_feed.setdefault(
                        key, getattr(self, key)
                    )

        # Re-resolve deferred "auto" logger now that workspace is available
        if isinstance(self.logger, str) and self.logger == "auto" and self._workspace:
            self._normalize_loggers()

        # BTA is an orchestrator — it does NOT render its own inference_input.
        # After the TemplatedInferencerBase refactor, BTA inherits InferencerBase
        # directly (no template_manager / template_key fields, and InferencerBase's
        # `_render_prompt` is a no-op stub returning input unchanged). The previous
        # `template_key = ""` line and `_render_prompt` override are no longer
        # needed. `_finalize_output` is now gated on output_path + has_local_access
        # (workspace concern, not template concern), so BTA's role_document.md /
        # aggregation_report.md output still gets written.

    def _validate_worker_inference_args(self) -> None:
        clashes = self._RESERVED_WORKER_INFERENCE_ARGS.intersection(
            self.worker_inference_args
        )
        if clashes:
            raise ValueError(
                f"{type(self).__name__}.worker_inference_args may not set "
                f"{sorted(clashes)}; the BTA passes them to every worker itself"
            )

    @staticmethod
    def _sub_query_text(sub_query) -> str:
        if isinstance(sub_query, dict):
            return sub_query.get("query", str(sub_query))
        return sub_query

    def _check_worker_query_sizes(self, shard_chars: List[int]) -> None:
        """Raise when any worker query exceeds ``max_worker_query_chars``."""
        limit = self.max_worker_query_chars
        if limit is None:
            return
        oversized = {i: n for i, n in enumerate(shard_chars) if n > limit}
        if oversized:
            raise ValueError(
                f"BTA[{self._effective('bta_node_name') or type(self).__name__}] worker "
                f"queries exceed max_worker_query_chars={limit} "
                f"(index: chars) {oversized}"
            )

    def _log_input_stats(
        self, shard_chars: List[int], aggregator_input: Any = None
    ) -> None:
        """Log an ``InferenceInputStats`` record: per-worker query sizes and,
        once built, the aggregator input size (before its own rendering)."""
        self.log_info(
            {
                "stage": "dispatch" if aggregator_input is None else "aggregate",
                "shard_count": len(shard_chars),
                "shard_chars": list(shard_chars),
                "max_shard_chars": max(shard_chars, default=0),
                "aggregator_input_chars": (
                    None if aggregator_input is None else len(str(aggregator_input))
                ),
            },
            "InferenceInputStats",
        )

    async def adisconnect(self):
        """Disconnect the stages this definition holds: the breakdown, the
        aggregator and a static worker list's instances. Workers a factory built
        were closed when their attempt ended. Every child is attempted; the first
        failure is re-raised."""
        workers = self.worker_inferencers
        children = [
            self.breakdown_inferencer,
            self.aggregator_inferencer,
            *(workers if isinstance(workers, list) else ()),
        ]
        unique = {id(c): c for c in children if hasattr(c, "adisconnect")}
        outcomes = await asyncio.gather(
            *(c.adisconnect() for c in unique.values()), return_exceptions=True
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome

    def _enforce_worker_quorum(self, failures: Mapping[int, str], n: int) -> None:
        """Log failed workers and raise when fewer than the required number
        succeeded: ``min_successful_workers``, or all ``n`` when it is 0."""
        survivors = n - len(failures)
        required = self.min_successful_workers or n
        if failures:
            self.log_info(
                {
                    "event": "QUORUM_FILTER",
                    "received": n,
                    "survivors": survivors,
                    "failed_workers": [
                        {"index": i, "failure": f[:200]}
                        for i, f in sorted(failures.items())
                    ],
                    "min_required": required,
                },
                "QuorumFilter",
            )
        if survivors >= required:
            return
        self.log_warning(
            {
                "event": "QUORUM_BELOW_THRESHOLD_FAIL_LOUD",
                "survivors": survivors,
                "min_required": required,
            },
            "QuorumFailLoud",
        )
        raise RuntimeError(
            f"BTA quorum unmet: only {survivors}/{required} required workers "
            "succeeded — refusing to synthesize a degraded aggregation. Failing loud."
        )

    def _order_worker_outcomes(
        self, outcomes: Sequence[Any], n: int
    ) -> Tuple[List[Any], Dict[int, str]]:
        """Index-align worker outcomes (see ``_ordered_outcomes``) and enforce
        the success quorum."""
        slots, failures = _ordered_outcomes(outcomes, n)
        self._enforce_worker_quorum(failures, n)
        return slots, failures

    def _resolve_worker_output_paths(
        self, workers: Sequence[Any], failures: Mapping[int, str]
    ) -> Tuple[List[Optional[str]], List[Optional[str]]]:
        """Resolve each worker's canonical output file and multi-artifact
        ``outputs/`` folder by declaration index. Entries are ``None`` for
        failed workers, and all ``None`` when this BTA has no workspace.

        The filename is each worker's own ``output_path``, not this BTA's
        (which names the aggregated deliverable)."""
        paths: List[Optional[str]] = [None] * len(workers)
        dirs: List[Optional[str]] = [None] * len(workers)
        ws = self._bta_call().effective_workspace
        if ws is None:
            return paths, dirs
        diag = []
        for idx, worker in enumerate(workers):
            if idx in failures:
                continue
            paths[idx], dirs[idx], info = self._resolve_worker_output_path(
                ws, idx, worker
            )
            diag.append(info)
        self.log_info(
            {
                "bta_name": self._effective("bta_node_name"),
                "bta_type": type(self).__name__,
                "bta_id": getattr(self, "id", None),
                "paths": [str(p) if p else None for p in paths],
                "deliverable_dirs": [str(d) if d else None for d in dirs],
                "workers": diag,
            },
            log_type="AggInputPaths",
        )
        return paths, dirs

    def _resolve_worker_output_path(self, ws, idx: int, worker: Any):
        from agent_foundation.common.inferencers.inferencer_workspace import (
            resolve_canonical_output_path,
        )

        child_name = self._worker_child_name(idx)
        child_ws = ws.child(child_name)
        filename = getattr(worker, "output_path", None)
        path = resolve_canonical_output_path(
            child_ws, filename=filename, deliverables_fallback="none"
        )
        # ``outputs/`` IS the worker's deliverable set; reference the folder only
        # when it holds more than the single canonical output file.
        out_dir = getattr(child_ws, "outputs_dir", None)
        folder = (
            os.path.abspath(out_dir)
            if out_dir and os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 1
            else None
        )
        info = {
            "idx": idx,
            "child_name": child_name,
            "ws_root": child_ws.root,
            "worker_output_path": filename,
            "resolved": str(path) if path else None,
        }
        if path is None and filename is not None and child_ws.root is not None:
            info.update(self._missing_output_diag(child_ws.root, filename))
        return path, folder, info

    @staticmethod
    def _missing_output_diag(ws_root: str, filename: str) -> Dict[str, Any]:
        out = os.path.join(ws_root, "outputs", filename)
        diag: Dict[str, Any] = {
            "diag_outputs_exists": os.path.exists(out),
            "diag_outputs_islink": os.path.islink(out),
        }
        if diag["diag_outputs_islink"]:
            target = os.readlink(out)
            diag["diag_link_target"] = target
            diag["diag_target_exists"] = os.path.exists(target)
        return diag

    def _build_aggregator_input(
        self, prompt_builder, results, failures, paths, dirs, original_query
    ):
        """Build the aggregator's ``inference_input`` from index-aligned worker
        results.

        A custom ``prompt_builder`` is fully responsible for the input; it gets
        the surviving results with their paired paths plus ``worker_failures``
        (see ``align_worker_results``). Otherwise the upstream artifacts are
        published as the aggregator's feed and the original query is returned,
        or, with ``inject_upstream_artifacts_to_aggregator`` off, formatted as
        the input itself (for wrappers that render workers into ``{{ input }}``).
        """
        if prompt_builder is not None:
            ok = [i for i in range(len(results)) if i not in failures]
            return _call_prompt_builder(
                prompt_builder,
                [results[i] for i in ok],
                original_query=original_query,
                worker_output_paths=[paths[i] for i in ok],
                bta=self,
                worker_failures=dict(failures),
            )
        if self.inject_upstream_artifacts_to_aggregator:
            self._inject_aggregator_extra_feed(
                results, paths, worker_deliverable_dirs=dirs, worker_failures=failures
            )
            return original_query or ""
        return self._format_worker_results_text(
            results, paths, worker_failures=failures
        )

    def _prepare_aggregator_input(
        self,
        agg_inf,
        prompt_builder,
        outcomes,
        workers,
        original_query,
        shard_chars=(),
    ):
        """Index-align the worker outcomes (raising when the quorum is unmet),
        re-bind the aggregator to its canonical slot, and build its input.

        Returns ``(agg_input, results, failures, paths)``."""
        results, failures = self._order_worker_outcomes(outcomes, len(workers))
        self._rebind_aggregator_workspace(agg_inf)
        paths, dirs = self._resolve_worker_output_paths(workers, failures)
        agg_input = self._build_aggregator_input(
            prompt_builder, results, failures, paths, dirs, original_query
        )
        self._log_input_stats(list(shard_chars), aggregator_input=agg_input)
        return agg_input, results, failures, paths

    def _rebind_aggregator_workspace(self, agg_inf) -> None:
        # Drift on a reused instance (or a fresh instance on resume, where the
        # current root is None): durably re-bind the aggregator to its canonical
        # slot. The durable set is resume-safe — see _bind_rebuilt_child_ws.
        ws = self._bta_call().effective_workspace
        if ws is None:
            return
        expected = ws.child("aggregator")
        current = getattr(getattr(agg_inf, "_workspace", None), "root", None)
        if current != expected.root:
            self._bind_rebuilt_child_ws(
                agg_inf, "aggregator", expected, owned=self._aggregator_stage().owned
            )

    def _format_worker_results_text(
        self,
        worker_results,
        worker_output_paths=None,
        worker_deliverable_dirs=None,
        *,
        worker_failures=None,
    ) -> str:
        """Format index-aligned worker results as ``### Upstream Outcome N``
        sections, joined by blank lines; a failed worker's section states its
        failure instead of an output.

        Used by both the legacy default aggregator-input path (where the
        formatted text becomes the aggregator's ``inference_input`` directly)
        and the ``inject_upstream_artifacts_to_aggregator`` path (where the
        formatted text becomes the ``upstream_artifacts`` feed variable).

        When the aggregator has local file access AND a worker output path is
        available, the result is referenced by path rather than inlined
        (avoids OS ARG_MAX limits when piping large outputs to subprocess).

        When ``worker_deliverable_dirs`` is provided and a worker has a
        non-empty deliverables folder, both the folder path and the report
        file are referenced — giving the aggregator a directory to explore
        AND a summary to read.
        """
        aggregator = self._current_aggregator()
        agg_has_local = aggregator is not None and getattr(
            aggregator, "has_local_access", False
        )
        paths = list(worker_output_paths or [])
        fd_dirs = list(worker_deliverable_dirs or [])
        failures = worker_failures or {}
        self.log_info(
            {
                "bta_name": self._effective("bta_node_name"),
                "bta_type": type(self).__name__,
                "agg_has_local": agg_has_local,
                "agg_type": type(aggregator).__name__ if aggregator else None,
                "paths": [str(p) if p else None for p in paths],
                "deliverable_dirs": [str(d) if d else None for d in fd_dirs],
                "num_results": len(worker_results) if worker_results else 0,
                "failed_indexes": sorted(failures),
            },
            log_type="AggFormatDecision",
        )
        parts = []
        for idx, res in enumerate(worker_results):
            if idx in failures:
                parts.append(
                    f"### Upstream Outcome {idx + 1}\n(failed: {failures[idx]})"
                )
                continue
            parts.append(
                self._format_worker_result_item(
                    idx,
                    res,
                    paths[idx] if idx < len(paths) else None,
                    fd_dirs[idx] if idx < len(fd_dirs) else None,
                    agg_has_local,
                )
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_worker_result_item(idx, res, path, fd_dir, agg_has_local) -> str:
        heading = f"### Upstream Outcome {idx + 1}"
        if agg_has_local and fd_dir:
            # Part 2: worker deliverables live directly in ``outputs/``.
            lines = [heading, f"(See outputs folder: `{fd_dir}`)"]
            if path:
                lines.append(f"(See file: `{path}`)")
            return "\n".join(lines)
        if agg_has_local and path:
            return f"{heading}\n(See file: `{path}`)"
        # Non-local aggregator: inline the FULL file content (not just the
        # <Response> summary) so the aggregator sees the complete upstream
        # artifact. Falls back to summary if file read fails.
        content = str(res)
        if path and os.path.isfile(path):
            try:
                content = open(path, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                pass
        return f"{heading}\n{content}"

    def _build_synthetic_aggregation(
        self,
        worker_results,
        original_query,
        *,
        worker_output_paths=None,
        worker_failures=None,
    ) -> str:
        """Produce a synthetic aggregation when the LLM aggregator fails.

        Lists the index-aligned upstream worker outputs (and each failed
        worker's reason) so downstream review/fix can still consume them.
        Written to the aggregator's output.md so the pipeline continues
        rather than crashing.
        """
        parts = [
            "# Synthetic Aggregation (automatic fallback)\n",
            "**Warning:** The LLM aggregator failed to produce a valid consolidated "
            "output after exhausting all retries. This synthetic aggregation lists "
            "the upstream worker outputs for manual review or downstream processing.\n",
            f"**Original task:** {original_query}\n",
        ]
        paths = list(worker_output_paths or [])
        failures = worker_failures or {}
        for idx, res in enumerate(worker_results):
            parts.append(f"## Upstream Outcome {idx + 1}")
            if idx in failures:
                parts.append(f"(failed: {failures[idx]})")
                continue
            path = paths[idx] if idx < len(paths) else None
            if path:
                parts.append(f"(Full output at: `{path}`)\n")
            parts.append(str(res)[:500] if res else "(empty)")
        return "\n\n".join(parts)

    def _inject_aggregator_extra_feed(
        self,
        worker_results,
        worker_output_paths=None,
        worker_deliverable_dirs=None,
        *,
        worker_failures=None,
    ) -> None:
        """Publish formatted upstream artifacts (and breakdown-captured
        aggregation_guidance, if any) as the aggregator inferencer's template
        feed. Used when ``inject_upstream_artifacts_to_aggregator=True`` AND
        no custom ``aggregator_prompt_builder`` is set.

        Under a RunContext the feed is published ctx-scoped at the
        aggregator's slot, composed over any ancestor override so keys this
        BTA does not own survive; otherwise it is merged into the
        aggregator's instance ``template_extra_feed``. The
        ``aggregation_guidance`` key is DROPPED when the breakdown didn't
        produce one this attempt (avoids stale guidance from a previous call
        leaking into the current prompt).
        """
        target = self.build_aggregator()
        if target is None:
            return
        if not hasattr(target, "template_extra_feed"):
            return
        self.log_info(
            {
                "bta_name": self._effective("bta_node_name"),
                "bta_type": type(self).__name__,
                "agg_type": type(target).__name__,
                "num_results": len(worker_results) if worker_results else 0,
                "paths": [str(p) if p else None for p in (worker_output_paths or [])],
            },
            log_type="AggInjectFeed",
        )
        feed = {
            "upstream_artifacts": self._format_worker_results_text(
                worker_results,
                worker_output_paths,
                worker_deliverable_dirs=worker_deliverable_dirs,
                worker_failures=worker_failures,
            )
        }
        # NOTE: Per-worker output paths are embedded inline within
        # ``upstream_artifacts`` via ``_format_worker_results_text`` (see
        # ``(See file: <path>)`` markers). No structured ``worker_output_paths``
        # variable is injected because no aggregator template currently
        # consumes it; speculative injection would be infrastructure with
        # no consumer. If a future template needs the structured list,
        # add the injection at that time.

        # Resume: the breakdown parse that normally records the attempt's guidance
        # ran in a prior process, so restore it from the promoted breakdown
        # checkpoint before publishing (this method owns the guidance channel).
        attempt = self._bta_attempt()
        if attempt.aggregation_guidance is None:
            attempt.aggregation_guidance = self._load_promoted_breakdown()[1]

        guidance = attempt.aggregation_guidance
        if guidance:
            feed["aggregation_guidance"] = guidance
        publish_child_template_feed(
            target,
            "aggregator",
            feed,
            drop_keys=() if guidance else ("aggregation_guidance",),
        )

    def _parse_json_subtasks(self, raw_output: str) -> List:
        """Parse JSON subtask format from the task_breakdown template.

        Extracts subtasks from ``<Response>`` tags or raw text, parses JSON
        with a ``subtasks`` array, and builds structured sub_queries for BTA.
        Falls back to ``parse_numbered_list`` if JSON extraction fails.

        This is the built-in parser for ``breakdown_format="json_subtasks"``,
        consolidating the parsing logic previously duplicated across tools.
        """
        return self._parse_json_breakdown(raw_output)[0]

    def _parse_json_breakdown(self, raw_output: str) -> Tuple[List, Optional[str]]:
        """:meth:`_parse_json_subtasks` plus the breakdown's
        ``aggregation_guidance`` (``None`` when the JSON is malformed, carries no
        subtasks, or has no such field), which the breakdown records on its
        attempt for the aggregator's feed."""
        from agent_foundation.common.response_parsers import extract_delimited

        response_text = extract_delimited(str(raw_output))
        if response_text is None:
            response_text = str(raw_output)

        # Try to extract JSON from ```json ... ``` code fence.
        # Allow optional newline between ```json and { (standard markdown).
        json_match = re.search(
            r"```json[^\n]*\n\s*(\{[\s\S]*?\})\s*\n\s*```", response_text
        )
        if not json_match:
            json_match = re.search(r'\{[\s\S]*"subtasks"[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                _logger.warning(
                    "No JSON in breakdown output, falling back to numbered list"
                )
                return parse_numbered_list(response_text), None
        else:
            json_str = json_match.group(1)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            repaired = re.sub(
                r"`(\{[^}]*\})`",
                lambda m: "`" + m.group(1).replace('"', "'") + "`",
                json_str,
            )
            try:
                data = json.loads(repaired)
                _logger.info("JSON parsed after repairing backtick-quoted code blocks")
            except json.JSONDecodeError as e:
                _logger.warning(
                    "JSON parse failed (%s), falling back to numbered list", e
                )
                return parse_numbered_list(response_text), None

        parsed = self._subtasks_from_fence_dict(data)
        if parsed is None:
            # No subtasks at all: no guidance either.
            return parse_numbered_list(response_text), None
        queries, guidance = parsed
        # The guidance survives the empty-queries fallback.
        if not queries:
            return parse_numbered_list(response_text), guidance

        _logger.info("Parsed %d subtasks from JSON breakdown", len(queries))
        return queries, guidance

    def _subtasks_from_fence_dict(
        self, data: dict
    ) -> Optional[Tuple[List, Optional[str]]]:
        """``decomposed_subtasks`` fence dict → ``(sub_queries, aggregation_guidance)``.

        The TRANSFORM half of ``_parse_json_subtasks``, split out at the ``data``
        boundary so one implementation serves every consumer of the block: this
        parser (text → dict → here), and anything that already holds the parsed
        fence (e.g. the persisted ``decomposed_subtasks.json``) — with no
        re-serializing a dict back to text just to re-run the extraction regex.

        Deliberately PURE with respect to instance state: it *reads*
        ``worker_query_fields`` but writes nothing, so a caller holding a shared or
        re-roled inferencer cannot be polluted by it. Recording the guidance stays
        with the caller.

        Returns ``None`` only when the dict carries **no subtasks at all** — the
        caller then falls back with no guidance, matching the original early return.
        When subtasks exist the pair is returned even if every one of them yielded
        an empty query string (``queries == []``): the original captured guidance
        *before* that check, so the caller must still record it before falling back.
        """
        subtasks = data.get("subtasks") or data.get("decomposed_subtasks") or []
        if not subtasks:
            return None

        # Tolerate missing/empty guidance — the aggregator prompt's ``{% if %}``
        # branch gates the whole section.
        _raw_guidance = data.get("aggregation_guidance")
        guidance = (
            _raw_guidance.strip()
            if isinstance(_raw_guidance, str) and _raw_guidance.strip()
            else None
        )

        queries = []
        for subtask in subtasks:
            desc = subtask.get("description", "")
            todos = subtask.get("todos") or []
            args = subtask.get("args", {})

            # Build query from selected fields
            parts = []
            if "description" in self.worker_query_fields and desc:
                parts.append(f"**Description**: {desc}")
            if "scope" in self.worker_query_fields and subtask.get("scope"):
                parts.append(f"**Scope**: {subtask['scope']}")
            if "priority" in self.worker_query_fields and subtask.get("priority"):
                parts.append(f"**Priority**: {subtask['priority']}")
            if "todos" in self.worker_query_fields and todos:
                todo_lines = "\n".join(f"- {t}" for t in todos)
                parts.append(f"**Todos**:\n{todo_lines}")
            query_text = "\n\n".join(parts)

            if query_text.strip():
                query_args = dict(args)
                if todos:
                    query_args["todos"] = todos
                if desc:
                    query_args["description"] = desc
                queries.append({"query": query_text.strip(), "args": query_args})

        return queries, guidance

    # _resolve_graph_reporter() is inherited from InferencerBase (Part F / GT#13).
    # Uniform graph_reporter propagation lives on the base now (seed into the
    # shared Tier-2 sink + path-namespaced child_reporter), so PTI/Dual/etc.
    # participate in graph viz too — not just BTA. Behavior is byte-identical.

    def _make_graph_status_callback(self):
        """Build the async status callback for WorkGraphNode event propagation.

        Returns a coroutine function that forwards NodeStatusEvent from WorkGraph
        nodes to the graph_reporter with resolved output_path. Set on the WorkGraph
        BEFORE _arun() so _propagate_settings_to_subgraph() copies it to expansion
        nodes (workers, aggregator) when they're created.

        Output paths resolve under this call's workspace (``_bta_call``), which a
        parent has bound by the time the call starts.
        """
        # Part F: resolve the shared Tier-2 sink (seeded from the instance) — this
        # runs inside ``_ainfer`` under the active ctx; with no ctx it is the
        # instance attrib (byte-identical).
        reporter = self._resolve_graph_reporter()
        _bta_self = self
        _ws_obj = self._bta_call().effective_workspace

        async def _async_status_cb(event):
            output_path = ""
            if event.status in ("completed", "error") and _ws_obj:
                from pathlib import Path as _P

                _ws = _P(str(_ws_obj.root))
                nid = event.node_id
                if nid == "breakdown":
                    for candidate in [
                        _ws
                        / "children"
                        / "breakdown"
                        / "outputs"
                        / "breakdown_output.md",
                    ]:
                        if candidate.exists():
                            output_path = str(candidate)
                            break
                elif _bta_self._is_worker_child_name(nid):
                    for fn in ("facet.md", "result.md", "output.md", "response.md"):
                        candidate = _ws / "children" / nid / "outputs" / fn
                        if candidate.exists():
                            output_path = str(candidate)
                            break
                elif nid == "aggregator":
                    for fn in (
                        "role_document.md",
                        "output.md",
                        "result.md",
                        "response.md",
                    ):
                        candidate = _ws / "children" / "aggregator" / "outputs" / fn
                        if candidate.exists():
                            output_path = str(candidate)
                            break
            await reporter.on_node_status(
                event.node_id,
                event.status,
                getattr(event, "error", ""),
                output_path=output_path,
            )

        return _async_status_cb

    async def _emit_pending_graph_topology(self, attempt: _BtaAttempt) -> None:
        """Emit the attempt's pending graph topology to the frontend.

        Called from _ainfer() — either early (from _breakdown_fn after expansion
        spec is built) or as a fallback after _arun() completes.

        The status callback is set separately in _ainfer() via
        _make_graph_status_callback() + set_graph_event_callback() BEFORE _arun(),
        so it propagates to expansion nodes automatically.
        """
        pending_topo = attempt.pending_topology
        _reporter = self._resolve_graph_reporter()  # Part F: shared Tier-2 sink
        _logger.info(
            "[BTA] _emit_pending_graph_topology: has_reporter=%s has_pending=%s",
            _reporter is not None,
            pending_topo is not None,
        )
        if pending_topo is not None:
            _logger.info(
                "[BTA] topology: %d nodes, %d edges",
                len(pending_topo.nodes),
                len(pending_topo.edges),
            )
        if _reporter is None or pending_topo is None:
            return
        attempt.pending_topology = None  # clear before await (re-entrant safety)
        try:
            await _reporter.on_graph_topology(pending_topo)

            # Breakdown is a VIRTUAL node — manually prepended to the topology, NOT a real
            # WorkGraphNode. So the graph event callback never fires for it.
            # Emit an explicit completion event with the resolved output_path so
            # the UI can fetch breakdown_output.md when the breakdown node is clicked.
            _ws_obj_b = self._bta_call().effective_workspace
            if _ws_obj_b:
                from pathlib import Path as _PB

                _ws_b = _PB(str(_ws_obj_b.root))
                _bd_output = ""
                for _cand in [
                    _ws_b
                    / "children"
                    / "breakdown"
                    / "outputs"
                    / "breakdown_output.md",
                ]:
                    if _cand.exists():
                        _bd_output = str(_cand)
                        break
                try:
                    await _reporter.on_node_status(
                        "breakdown",
                        "completed",
                        output_path=_bd_output,
                    )
                except Exception as _ebd:
                    _logger.warning(
                        "[BTA] breakdown node_status emit failed (visualization only): %s",
                        _ebd,
                    )
        except Exception as _e:
            _logger.warning(
                "[BTA] graph topology emit failed (visualization only): %s", _e
            )

    @staticmethod
    def _configure_child_workspace(inferencer, workspace):
        """Deprecated: workspace setter on InferencerBase auto-configures.

        Kept for backward compatibility. Equivalent to:
            inferencer._workspace = workspace
        """
        inferencer._workspace = workspace

    def _bta_call(self) -> _BtaCallRecord:
        """This invocation's call record, taken once, from the invocation's own ctx.

        ``_ainfer`` / ``_infer`` take it before any stage runs; a private hook that
        a test drives inside ``open_invocation`` takes it on first use. Taking it
        leases the checkpoint root for the whole invocation (§5.11): registered
        first in the frame ledger, the lease is released last, after every stage
        the call owned has closed.
        """
        frame = invocation_of(self)
        record = frame.get(self._BTA_CALL)
        if record is None:
            workspace = self._workspace_under(frame.ctx)
            record = _BtaCallRecord(
                effective_workspace=workspace,
                checkpoint_dir=self.checkpoint_dir,
                checkpoint_root=(
                    workspace.checkpoints_dir
                    if workspace is not None
                    else self.checkpoint_dir or None
                ),
            )
            if record.checkpoint_root is not None:
                lease = CheckpointLease(record.checkpoint_root).acquire()
                frame.ledger.register(lease, "checkpoint lease")
            frame.put(self._BTA_CALL, record)
        return record

    def _retry_archive_keeps(self) -> FrozenSet[str]:
        """The lease and the manifest stay put when a retry archives the attempt's
        checkpoints: they belong to the whole call (moving the lease would free the
        root mid-call)."""
        return frozenset({LEASE_FILE, MANIFEST_FILE})

    def _open_manifest(self, inference_input, inference_config, kwargs) -> None:
        """Open the call's resume manifest once, under the lease and before any
        checkpoint is read (§5.12): write the identity header for a fresh run, or
        verify it for a resume."""
        frame = invocation_of(self)
        root = self._bta_call().checkpoint_root
        if root is None or frame.get(self._BTA_MANIFEST) is not None:
            return
        header = build_header(
            inference_input,
            self,
            {"inference_config": inference_config, "kwargs": kwargs},
        )
        frame.put(self._BTA_MANIFEST, self._resolve_manifest(root, header))

    def _resolve_manifest(self, root: str, header) -> _BtaManifest:
        artifacts = has_root_artifacts(root) or has_worker_results(
            self._worker_checkpoint_dirs()
        )
        if not self.resume_with_saved_results and artifacts:
            return self._unvouched_manifest(root, header)
        if not self.resume_with_saved_results or not (
            artifacts or os.path.exists(os.path.join(root, MANIFEST_FILE))
        ):
            if self.resume_with_saved_results and header["unverifiable"]:
                _logger.warning(
                    "[BTA] checkpoints at %r can't be resumed: %s",
                    root,
                    header["unverifiable"],
                )
            write_manifest(root, header, None)
            return _BtaManifest(root=root, mode="fresh", header=header)
        stored = read_manifest(root)
        if stored is None:
            return self._legacy_manifest(root, header)
        unverifiable = header["unverifiable"] or stored["header"].get("unverifiable")
        if unverifiable:
            self._trust_or_raise(
                ResumeIdentityUnavailableError(
                    f"cannot verify a resume at {root!r}: {unverifiable}"
                )
            )
        else:
            mismatched = header_mismatches(stored["header"], header)
            if mismatched:
                raise BtaResumeIdentityMismatch(
                    f"checkpoints at {root!r} were written for another "
                    f"{', '.join(mismatched)}; use a fresh workspace or turn off "
                    f"resume_with_saved_results"
                )
        plan = stored["plan"]
        if plan is None and has_worker_results(self._worker_checkpoint_dirs()):
            raise BtaResumeCorruptionError(
                f"worker results under {root!r} without a committed plan"
            )
        if plan is not None and "unavailable" in plan:
            self._trust_or_raise(
                ResumeIdentityUnavailableError(
                    f"the plan at {root!r} can't be replayed: {plan['unavailable']}"
                )
            )
            plan = None
        return _BtaManifest(
            root=root, mode="resume", header=stored["header"], plan=plan
        )

    def _unvouched_manifest(self, root: str, header) -> _BtaManifest:
        """A call that doesn't resume overwrites only the checkpoints it reaches,
        so a tree holding an earlier call's is left without a manifest: resuming
        it later fails closed like a legacy tree. The plan stays in memory for
        this call's own rebuilds."""
        remove_manifest(root)
        _logger.info(
            "[BTA] %r holds checkpoints of an earlier call that this call does not "
            "resume; it is left without a resume manifest",
            root,
        )
        return _BtaManifest(root=root, mode="unvouched", header=header)

    def _legacy_manifest(self, root: str, header) -> _BtaManifest:
        self._trust_or_raise(
            UnverifiedLegacyBtaResumeError(
                f"checkpoints at {root!r} have no resume manifest (written before "
                f"manifests, or over an earlier call's checkpoints by a call that "
                f"did not resume them), so nothing proves they belong to this call; "
                f"resume them once with resume_identity_policy={TRUST_LEGACY!r} or "
                f"certify them with certify_resume_manifest(), or use a fresh "
                f"workspace"
            )
        )
        return _BtaManifest(root=root, mode="legacy", header=header)

    def _trust_or_raise(self, error: Exception) -> None:
        """Raise ``error`` unless the policy trusts unverified checkpoints, which is
        logged on every use."""
        if self.resume_identity_policy != TRUST_LEGACY:
            raise error
        _logger.warning("[BTA] resume_identity_policy=trust_legacy: %s", error)

    def _worker_checkpoint_dirs(self) -> List[str]:
        workspace = self._bta_call().effective_workspace
        if workspace is None or not os.path.isdir(workspace.children_dir):
            return []
        return [
            os.path.join(workspace.children_dir, name, "checkpoints")
            for name in sorted(os.listdir(workspace.children_dir))
            if self._is_worker_child_name(name)
        ]

    def _bta_manifest(self) -> Optional[_BtaManifest]:
        frame = frame_for(self)
        return None if frame is None else frame.get(self._BTA_MANIFEST)

    def _committed_sub_queries(self) -> Optional[List]:
        """The sub-queries of the plan committed for this call — by a resumed run,
        or earlier in this one (a retry rebuilds its saved expansion): the list
        after truncation and selection, which the workers are rebuilt from."""
        manifest = self._bta_manifest()
        if (
            manifest is None
            or manifest.mode == "legacy"
            or manifest.plan is None
            or "unavailable" in manifest.plan
        ):
            return None
        return list(manifest.plan["sub_queries"])

    def _commit_plan(self, sub_queries, spec) -> None:
        """The commit point (§5.12): record the effective plan once
        ``_build_subgraph_spec`` returned, before any worker result can persist; a
        replayed plan must rebuild the same workers. A plan that can't be replayed
        is recorded again."""
        manifest = self._bta_manifest()
        if manifest is None or manifest.mode == "legacy":
            return
        workers = [node.name for node in spec.entry_nodes]
        if manifest.plan is not None and "unavailable" not in manifest.plan:
            if manifest.plan.get("workers") != workers:
                raise BtaResumeCorruptionError(
                    f"the committed plan at {manifest.root!r} names workers "
                    f"{manifest.plan.get('workers')}, the rebuild {workers}"
                )
            return
        manifest.plan = plan_record(sub_queries, workers)
        if manifest.mode != "unvouched":
            write_manifest(manifest.root, manifest.header, manifest.plan)

    def _open_attempt(self, inference_input, *, use_async: bool) -> _BtaAttempt:
        """Start an attempt of this run: no summary, a fresh ``_BtaAttempt`` in the
        frame, then the ``_begin_attempt`` hook."""
        attempt = _BtaAttempt(original_query=inference_input, use_async=use_async)
        discard_result(self, self._BTA_SUMMARY)
        invocation_of(self).put(self._BTA_ATTEMPT, attempt)
        self._begin_attempt(attempt, inference_input)
        return attempt

    def _bta_attempt(self) -> _BtaAttempt:
        """The current attempt of this invocation's run."""
        return invocation_of(self).require(self._BTA_ATTEMPT)

    @property
    def bta_node_name(self) -> Optional[str]:
        """The node name a call runs under when no parent BTA passes one."""
        return self.name

    def _log_display_name(self) -> Optional[str]:
        return self._effective("bta_node_name")

    def _begin_attempt(self, attempt: _BtaAttempt, inference_input) -> None:
        """Hook run at the start of every attempt; identity in BTA."""

    def _conclude_attempt(self, result):
        """Hook run once per run, on the returned attempt's result after
        ``_finalize_response``; identity in BTA."""
        return result

    def _run_summary(self, attempt: _BtaAttempt) -> BtaCallSummary:
        """The returned attempt's workers and aggregator, as finalize reads them."""
        ws = self._bta_call().effective_workspace
        agg_inf = self._current_aggregator()
        # M7: resolve the aggregator's PUBLISHED workspace from its child ctx node
        # (override-aware), not the bare property which under the parent ctx misses
        # the child's mailbox and returns None/the BTA's own ws — orphaning the plan.
        agg_ws = self._read_child_workspace(agg_inf, "aggregator") if agg_inf else None
        names = attempt.worker_child_names
        selected_index, selected = self._select_contract(attempt)
        return BtaCallSummary(
            worker_child_names=names,
            worker_workspace_roots=[
                ws.child(name).root if ws is not None else None for name in names
            ],
            aggregator_output_name=getattr(agg_inf, "output_path", None),
            aggregator_workspace_root=agg_ws.root if agg_ws is not None else None,
            disable_aggregator=self.disable_aggregator,
            selected_contract_index=selected_index,
            selected_contract=selected,
        )

    def _harvest_worker_contract(
        self, attempt: _BtaAttempt, index: int, worker, worker_rc
    ) -> None:
        """Record the task contract a successful worker published for its call.
        A bookkeeping failure never fails the worker."""
        try:
            contract = self._task_contract_state_at(worker, worker_rc)
        except Exception as err:
            _logger.debug(
                "BTA[%s]: task_instructions publish-up skipped: %s",
                getattr(self, "name", "?"),
                err,
            )
            return
        if contract is not None and contract.text:
            attempt.worker_contracts[index] = contract

    def _select_contract(
        self, attempt: _BtaAttempt
    ) -> Tuple[Optional[int], Optional[RenderedTaskContractState]]:
        """The contract this run relays: the lowest worker index that reported one
        (B13), whatever order the workers finished in. The contract is the same
        across workers up to actor-scoped values, so the choice only has to be
        deterministic."""
        if not attempt.worker_contracts:
            return None, None
        index = min(attempt.worker_contracts)
        return index, attempt.worker_contracts[index]

    def _conclude_run(self, attempt: _BtaAttempt, result):
        """The run's tail after ``_finalize_response``: the ``_conclude_attempt``
        hook, then the summary, as the run's last statement."""
        result = self._conclude_attempt(result)
        publish_result(self, self._BTA_SUMMARY, self._run_summary(attempt))
        return result

    @property
    def last_call_summary(self) -> Optional[BtaCallSummary]:
        """The summary of this instance's last call outside a host ctx. Under a
        ctx, a parent reads the node's outcome instead."""
        return self._last_call_summary

    def _outcome_for(self, frame) -> Optional[NodeOutcomeState]:
        """A BTA publishes the summary of the run that produced its result, and
        the worker contract that run relays."""
        summary = frame.get(self._BTA_SUMMARY)
        if summary is None:
            return None
        return NodeOutcomeState(
            task_contract=summary.selected_contract, summary=summary
        )

    def _rebuild_subgraph(self):
        """The resume factory: the fan-out rebuilt from the committed plan (§5.12;
        B35, B37), with the attempt's original request. A ``trust_legacy`` resume
        of checkpoints without a manifest rebuilds from ``_legacy_sub_queries``."""
        committed = self._committed_sub_queries()
        queries = committed if committed is not None else self._legacy_sub_queries()
        if queries is None:
            raise BtaResumeCorruptionError(
                "a saved expansion with no committed plan and no saved sub-queries"
            )
        spec = self._build_subgraph_spec(
            queries, _original_query=self._bta_attempt().original_query
        )
        if committed is not None:
            self._commit_plan(queries, spec)
        return spec

    def _legacy_sub_queries(self) -> Optional[List]:
        """The worker sub-queries of checkpoints without a committed plan: the
        breakdown node's saved result (the list after truncation and selection),
        else the promoted breakdown (before them), else the predefined
        sub-queries, capped like a fresh run."""
        queries = self._saved_breakdown_result()
        if queries is None:
            queries = self._load_promoted_breakdown()[0]
        if queries is None and self.predefined_sub_queries is not None:
            queries = self._resolve_predefined_sub_queries()[: self.max_breakdown]
        return queries

    def _saved_breakdown_result(self) -> Optional[List]:
        """The sub-queries the attempt's breakdown node saved as its result, read
        the way the graph reloads it; ``None`` when it saved none."""
        graph = self._bta_attempt().graph
        if graph is None or not graph.start_nodes:
            return None
        node = graph.start_nodes[0]
        try:
            path = node._resolve_result_path(node.name)
            if not node._exists_result(node.name, path):
                return None
            saved = node._load_result(node.name, path)
        except NotImplementedError:
            return None
        except Exception as exc:
            raise BtaResumeCorruptionError(
                f"unreadable breakdown result in {self._bta_call().checkpoint_root!r}: "
                f"{exc}"
            ) from exc
        return saved if isinstance(saved, list) and saved else None

    def certify_resume_manifest(
        self, inference_input, inference_config=None, **kwargs
    ) -> str:
        """Write a verified resume manifest for this BTA's existing checkpoints
        (§5.12), for the call ``ainfer(inference_input, inference_config,
        **kwargs)`` would make; return its path.

        Only when identity is provable: an input, definition or argument without a
        resume identity raises ``ResumeIdentityUnavailableError``. A replayable plan
        already committed is kept, whichever header it was committed under (a
        replaced header is logged); otherwise the plan is rebuilt like a
        ``trust_legacy`` resume would (``_legacy_sub_queries``). Worker results with
        no plan to rebuild raise ``BtaResumeCorruptionError``. Holds the checkpoint
        root's lease while it works.
        """
        with open_invocation(self, entry="certify_resume_manifest"):
            root = self._bta_call().checkpoint_root
            if root is None:
                raise ValueError(
                    "certify_resume_manifest needs a workspace or a checkpoint_dir"
                )
            header = build_header(
                inference_input,
                self,
                {"inference_config": inference_config, "kwargs": kwargs},
            )
            if header["unverifiable"]:
                raise ResumeIdentityUnavailableError(header["unverifiable"])
            stored = read_manifest(root)
            plan = None if stored is None else stored["plan"]
            if stored is not None and header_mismatches(stored["header"], header):
                _logger.warning(
                    "[BTA] certify_resume_manifest: the manifest at %r was written "
                    "for another %s; certifying the checkpoints for this call",
                    root,
                    ", ".join(header_mismatches(stored["header"], header)),
                )
            if plan is None or "unavailable" in plan:
                plan = self._certified_plan(
                    root, inference_input, inference_config, kwargs
                )
            write_manifest(root, header, plan)
        return os.path.join(root, MANIFEST_FILE)

    def _certified_plan(
        self, root: str, inference_input, inference_config, kwargs
    ) -> Optional[Dict[str, Any]]:
        attempt = self._open_attempt(inference_input, use_async=False)
        try:
            self._install_breakdown_node(
                attempt, inference_input, inference_config, **kwargs
            )
            queries = self._legacy_sub_queries()
            if queries is None:
                if has_worker_results(self._worker_checkpoint_dirs()):
                    raise BtaResumeCorruptionError(
                        f"worker results under {root!r} with no plan to rebuild"
                    )
                return None
            spec = self._build_subgraph_spec(queries, _original_query=inference_input)
            return plan_record(queries, [node.name for node in spec.entry_nodes])
        finally:
            invocation_of(self).cleanup_errors.extend(attempt.ledger.close_joined())

    def _checkpoint_path(self, name: str) -> Optional[str]:
        """``<checkpoint root>/<name>`` for this call, else ``None`` (no workspace
        and no ``checkpoint_dir``)."""
        root = self._bta_call().checkpoint_root
        return os.path.join(root, name) if root is not None else None

    def _get_result_path(self, result_id, *args, **kwargs):
        """Provide result path for WorkGraph-level result saving."""
        ext = ".json" if self.checkpoint_mode == "jsonfy" else ".pkl"
        path = self._checkpoint_path(f"{result_id}_result{ext}")
        if path is None:
            raise NotImplementedError(
                "checkpoint_dir or workspace must be set for result saving"
            )
        return path

    def _get_effective_predefined_sub_queries(self):
        """The sub-queries this attempt runs: those a subclass derived for it
        (``_BtaAttempt.effective_sub_queries``), else ``predefined_sub_queries``."""
        frame = frame_for(self)
        attempt = None if frame is None else frame.get(self._BTA_ATTEMPT)
        if attempt is not None and attempt.effective_sub_queries is not None:
            return attempt.effective_sub_queries
        return self.predefined_sub_queries

    def _resolve_predefined_sub_queries(self) -> List:
        """Resolve predefined_sub_queries into a list of sub-queries.

        Called only when self.predefined_sub_queries is not None.

        Returns:
            List of sub-queries (strings or dicts) to pass to _build_diamond_graph.
            - If predefined_sub_queries is a list: returned directly (copy).
            - If predefined_sub_queries is a str: replicated N times where
              N = max_breakdown or max_concurrency or 1 (auto-repeat mode).
            - Any other type: coerced to str with a warning (single-item list).
        """
        psq = self._get_effective_predefined_sub_queries()
        if isinstance(psq, str):
            # Auto-repeat mode: replicate single query N times
            n = self.max_breakdown or self.max_concurrency or 1
            _logger.info(
                "predefined_sub_queries: auto-repeating single query x%d "
                "(max_breakdown=%s, max_concurrency=%s)",
                n,
                self.max_breakdown,
                self.max_concurrency,
            )
            return [psq] * n
        elif isinstance(psq, list):
            _logger.info(
                "predefined_sub_queries: using caller-supplied list of %d sub_queries",
                len(psq),
            )
            return list(psq)
        else:
            # Unexpected type — coerce to single-item list with warning
            _logger.warning(
                "predefined_sub_queries: unexpected type %s, coercing to string",
                type(psq).__name__,
            )
            return [str(psq)]

    def _load_promoted_breakdown(self):
        """Read the promoted breakdown checkpoint -> ``(sub_queries, aggregation_guidance)``.

        Resume-side counterpart of the breakdown's ``checkpoint_scope="parent"``
        promotion: reads the ``decomposed_subtasks.json`` that
        ``InferencerBase._promote_child_checkpoints`` published into this parent's
        ``checkpoints/breakdown/`` and runs the SAME ``_subtasks_from_fence_dict``
        transform the fresh-run parse uses — so the one file feeds BOTH resume
        consumers: the worker fan-out (``sub_queries``, via the
        ``subgraph_registry`` factories) and the aggregator
        (``aggregation_guidance``, via ``_inject_aggregator_extra_feed``). This
        retires the hand-rolled ``breakdown_result.json``.

        Returns ``(None, None)`` when the file is absent (a fresh run, or a resume
        of a run that never reached breakdown-complete), unparseable, or carries
        no sub_queries. Memoizes only a usable read (on the attempt), so a fresh
        run's early absent-check at Step 0 cannot pin a stale ``None``.
        """
        attempt = self._bta_attempt()
        if attempt.promoted_breakdown is not None:
            return attempt.promoted_breakdown
        ws = self._bta_call().effective_workspace
        if ws is None:
            return (None, None)
        path = ws.checkpoint_path(os.path.join("breakdown", "decomposed_subtasks.json"))
        if not os.path.isfile(path):
            return (None, None)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            _logger.warning("Failed to load promoted breakdown checkpoint: %s", e)
            return (None, None)
        parsed = self._subtasks_from_fence_dict(data)
        if not parsed or not parsed[0]:
            # No usable sub_queries — behave exactly as "no checkpoint" so Step 0
            # runs a fresh breakdown instead of short-circuiting into an error.
            return (None, None)
        attempt.promoted_breakdown = parsed
        _logger.info(
            "Loaded promoted breakdown checkpoint (%d sub_queries)", len(parsed[0])
        )
        return parsed

    async def _emit_graph_reconcile(self, attempt: _BtaAttempt):
        """Emit the attempt graph's final node statuses so the frontend can correct
        any stale UI state."""
        _reporter = self._resolve_graph_reporter()  # Part F: shared Tier-2 sink
        if _reporter is None:
            return
        try:
            statuses = {n.name: "completed" for n in attempt.graph._all_nodes()}
            statuses["breakdown"] = "completed"
            await _reporter.on_graph_reconcile(statuses)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Output finalization (orchestrator override)
    # ------------------------------------------------------------------

    def _finalize_output(self, response):
        """BTA override: symlink to aggregator's output as canonical.

        When aggregator is enabled, the aggregator's output IS the BTA's
        canonical output.  When aggregator is disabled, workers' outputs
        are the deliverables (organized under ``workers/``). Both read the
        summary of the run that produced ``response``. A response no BTA run
        produced (every attempt raised, then an external ``fallback_inferencer``
        or a non-exception ``default_return_or_raise`` answered) is finalized as
        a leaf's: no aggregator output is linked (B36).
        """
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        summary = read_result(self, self._BTA_SUMMARY)
        if summary is None:
            return super()._finalize_output(response)
        agg_root = summary.aggregator_workspace_root

        # U3b: an aggregator IS configured but its workspace never published →
        # silently taking the no-aggregator branch would DROP the aggregated
        # deliverable. Fail loud instead. (The no-aggregator else-branch below
        # stays valid for the legitimate disable_aggregator / no-aggregator case.)
        if (
            not summary.disable_aggregator
            and self.aggregator_inferencer is not None
            and agg_root is None
        ):
            raise RuntimeError(
                "BTA aggregator is configured but its workspace could not be "
                "resolved (agg_ws is None) — refusing to fall back to the "
                "no-aggregator path and drop the aggregated output."
            )

        if agg_root is not None:
            self._symlink_child_output(
                InferencerWorkspace(root=agg_root),
                child_output_name=summary.aggregator_output_name,
            )
            resolved = self.resolve_output_path()
            if resolved and os.path.isfile(resolved):
                self._emit_output_manifest(resolved)
            # Proposal-index PRODUCTION moved to the task executor's finalize
            # (Fix 3) — the generic BTA no longer writes proposals.json.
            return response
        # No aggregator: workers' outputs ARE the deliverables. Part 2
        # (two-axis model): deliverables live directly in ``outputs/`` — the
        # BTA surfaces each worker's ``outputs/`` under its own
        # ``outputs/workers/<worker>/``.
        ws = self._bta_call().effective_workspace
        if ws and ws.outputs_dir:
            workers_dir = os.path.join(ws.outputs_dir, "workers")
            for name, root in zip(
                summary.worker_child_names, summary.worker_workspace_roots
            ):
                worker_ws = InferencerWorkspace(root=root)
                if worker_ws.has_deliverables:
                    self._symlink_or_copy(
                        worker_ws.outputs_dir, os.path.join(workers_dir, name)
                    )
        # Fall through to base for outputs/output.md summary write
        return super()._finalize_output(response)

    def build_aggregator(self) -> Any:
        """This call's aggregator (``None`` without one). The slot is never
        written: see ``_aggregator_stage``."""
        stage = self._aggregator_stage()
        return None if stage is None else stage.inferencer

    def _aggregator_stage(self) -> Optional[ResolvedStage]:
        """This call's aggregator stage, resolved from the slot once per call: a
        factory's product is owned by the call and closes when the call ends; an
        instance is borrowed."""
        frame = invocation_of(self)
        if frame.has(self._BTA_AGGREGATOR):
            return frame.get(self._BTA_AGGREGATOR)
        slot = self.aggregator_inferencer
        stage = None if slot is None else resolve_stage(slot)
        frame.put(self._BTA_AGGREGATOR, stage)
        if (
            stage is not None
            and stage.owned
            and hasattr(stage.inferencer, "adisconnect")
        ):
            frame.ledger.register(stage.inferencer, "aggregator")
        return stage

    def _current_aggregator(self) -> Any:
        """The aggregator this call resolved, else the slot: what a read before
        resolution, or outside a call, sees."""
        frame = frame_for(self)
        if frame is not None and frame.has(self._BTA_AGGREGATOR):
            stage = frame.get(self._BTA_AGGREGATOR)
            return None if stage is None else stage.inferencer
        return self.aggregator_inferencer

    @staticmethod
    def response_text(result) -> str:
        """The text of a BTA result: the aggregator's response, or the last
        worker's result when there is no aggregator."""
        if isinstance(result, tuple):
            result = result[-1] if result else None
        if result is None:
            return ""
        if hasattr(result, "output"):
            return result.output or ""
        if isinstance(result, dict):
            return result.get("output", "")
        if hasattr(result, "text"):
            return result.text or ""
        return str(result)

    def _finalize_response(self, result):
        """BTA audit bookkeeping (surfacing moved to _finalize_output).

        Only the aggregation report fallback remains: when no deliverables
        exist, write the aggregator's text response as a report.
        """
        ws = self._bta_call().effective_workspace
        if ws is None or not self.output_path:
            return

        # Check if aggregator produced deliverables (written by leaf's
        # _finalize_output which runs before this via __ainfer_single_impl)
        agg_inf = self._current_aggregator()
        # M7: resolve the aggregator's PUBLISHED workspace from its child ctx node
        # (override-aware), not the bare property which under the parent ctx misses
        # the child's mailbox and returns None/the BTA's own ws — orphaning the plan.
        agg_ws = self._read_child_workspace(agg_inf, "aggregator") if agg_inf else None
        has_deliverables = agg_ws is not None and getattr(
            agg_ws, "has_deliverables", False
        )

        if has_deliverables:
            _logger.info(
                "Skipping pipeline report — aggregator deliverables handled by _finalize_output",
            )
        else:
            report_dst = ws.output_path(self.output_path)
            os.makedirs(os.path.dirname(report_dst), exist_ok=True)
            try:
                text = self.response_text(result)
                # Explicit utf-8 encoding required: on Windows the default
                # file encoding is cp1252 which can't encode common Unicode
                # characters (e.g., the arrow '→' that LLMs frequently produce
                # in architectural docs). A UnicodeEncodeError here would
                # propagate up through the retry layers and look like a
                # "transient" failure when it's actually a determinstic
                # encoding bug.
                with open(report_dst, "w", encoding="utf-8") as f:
                    f.write(text)
                _logger.info("Wrote pipeline report -> %s", report_dst)
            except (OSError, UnicodeEncodeError) as e:
                _logger.warning(
                    "Failed to write pipeline report to %s: %s", report_dst, e
                )

    def _configure_for_workspace(self, workspace):
        super()._configure_for_workspace(workspace)
        ctx = active_run_context()
        if ctx is not None and not ctx.legacy_mint:
            # Under a host ctx the breakdown, which may be shared, is bound per call
            # (``_bind_breakdown_workspace``) instead of written here.
            return
        if self.breakdown_inferencer is not None:
            bd_ws = workspace.child("breakdown")
            bd_ws.ensure_dirs()
            self.breakdown_inferencer._workspace = bd_ws

    def _bind_breakdown_workspace(self) -> None:
        """Under a host ctx, publish this call's breakdown workspace to the
        ``breakdown`` child ctx; without one the setter has bound it already."""
        workspace = self._bta_call().effective_workspace
        if workspace is None or self.breakdown_inferencer is None:
            return
        if invocation_of(self).mode == "host":
            self._bind_rebuilt_child_ws(
                self.breakdown_inferencer,
                "breakdown",
                workspace.child("breakdown"),
                owned=False,
            )

    def _iter_child_inferencers(self):
        """The aggregator inferencer.

        Workers are factory-created per-run in generic BTA (via
        ``worker_inferencers``), so this base implementation does not yield
        them — subclasses with declarative worker references (e.g.,
        :class:`MultiFlowInferencer`) extend this to include them.
        """
        aggregator = self._current_aggregator()
        if aggregator is not None:
            yield aggregator

    def _iter_child_slots(self):
        """§9.3/N-Major1: semantic slots for the static children (breakdown,
        aggregator) matching the ``ctx.child(slot)`` used in ``_ainfer``. Workers
        are dynamic (per-run via ``worker_inferencers`` — callable/dict/list) and are
        bound to ``ctx.child(worker_node_name)`` at spawn time, not here."""
        seen_ids = set()
        for slot, inf in (
            ("breakdown", self.breakdown_inferencer),
            ("aggregator", self._current_aggregator()),
        ):
            if inf is not None and id(inf) not in seen_ids:
                seen_ids.add(id(inf))
                yield (slot, inf)

    def _validate_worker_isolation(self, workers):
        """Check that no two workers share a sub-inferencer instance.

        Scans each worker's full descendant tree (via
        ``_collect_all_descendant_inferencers``) and warns when the same
        Python id appears under different worker indices. Shared instances
        cause cross-worker state pollution (workspace, session, prompt
        history).

        No-op when ``worker_isolation_check=False`` or when workers are
        not InferencerBase instances (e.g., plain callables).
        """
        if not self.worker_isolation_check:
            return
        seen = {}
        for i, w in enumerate(workers):
            if not isinstance(w, InferencerBase):
                continue
            for inf in w._collect_all_descendant_inferencers():
                iid = id(inf)
                if iid in seen and seen[iid] != i:
                    _logger.warning(
                        "BTA[%s] workers %d and %d share inferencer %s (id=0x%x). "
                        "This causes cross-worker state contamination. "
                        "Ensure the worker_inferencers field uses LazyConfigFactory "
                        "(auto-applied for *_factory attrs with _target_:) to "
                        "produce independent sub-trees per call.",
                        getattr(self, "name", "?"),
                        seen[iid],
                        i,
                        type(inf).__name__,
                        iid,
                    )
                else:
                    seen[iid] = i

    def _unwrap_workgraph_result(self, result):
        """Unwrap expansion-driven WorkGraph results.

        WorkGraph returns a tuple of start-node results. In the expansion-driven
        implementation, the breakdown node is the sole start node, so the outer
        tuple is always length-1. The inner value may be a tuple of worker results
        with None entries. Unwrap both levels and filter out Nones.

        With no aggregator the inner values are worker outcomes; they are
        quorum-checked, then reduced to their values in declaration order.
        """
        if isinstance(result, tuple) and len(result) == 1:
            result = result[0]
        worker_count = len(self._bta_attempt().worker_child_names)
        if worker_count and (
            self.disable_aggregator or self.aggregator_inferencer is None
        ):
            items = result if isinstance(result, tuple) else (result,)
            # WorkGraph stores leaf results in declaration-order slots, so a
            # slot's position is its worker's index.
            values, _ = self._order_worker_outcomes(
                [_as_outcome(x, i, None) for i, x in enumerate(items)],
                worker_count,
            )
            result = tuple(values)
        if isinstance(result, tuple):
            non_none = tuple(x for x in result if x is not None)
            if len(non_none) == 1:
                result = non_none[0]
            elif len(non_none) == 0:
                result = None
            else:
                result = non_none
        return result

    def _infer(self, inference_input, inference_config=None, **kwargs):
        """Expansion-driven sync inference: single breakdown node → GraphExpansionResult → diamond.

        Exactly one attempt: the interactive results review is async-only. The
        workers it built close before it returns (``ResourceLedger.close_joined``).
        """
        self._bta_call()
        self._open_manifest(inference_input, inference_config, kwargs)
        attempt = self._open_attempt(inference_input, use_async=False)
        try:
            self._install_breakdown_node(
                attempt, inference_input, inference_config, **kwargs
            )

            # Run the graph — expansion handles the rest
            result = attempt.graph._run(inference_input, **kwargs)

            result = self._unwrap_workgraph_result(result)
        finally:
            invocation_of(self).cleanup_errors.extend(attempt.ledger.close_joined())

        self._finalize_response(result)
        return self._conclude_run(attempt, result)

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        """Expansion-driven async inference: single breakdown node → GraphExpansionResult → diamond.

        One attempt, plus one more per interactive "rerun" review; each closes the
        workers it built when it ends, in this loop. The tail runs once, for the
        attempt whose result is returned.
        """
        self._bta_call()
        self._open_manifest(inference_input, inference_config, kwargs)
        while True:
            attempt = self._open_attempt(inference_input, use_async=True)
            try:
                result = await self._arun_attempt(
                    attempt, inference_input, inference_config, **kwargs
                )
                rerun = await self._review_requests_rerun(result)
            finally:
                invocation_of(self).cleanup_errors.extend(await attempt.ledger.aclose())
            if not rerun:
                break

        await self._emit_graph_reconcile(attempt)
        self._finalize_response(result)
        return self._conclude_run(attempt, result)

    def _install_breakdown_node(
        self, attempt: _BtaAttempt, inference_input, inference_config, **kwargs
    ) -> None:
        """Build the attempt's graph with its breakdown node as the sole start
        node, expansion configured. The order mirrors a graph built empty and then
        given its start node, so the start node is not parented to the graph."""
        attempt.graph = _bta_graph(self, attempt)

        # Create the breakdown node as the sole start node
        breakdown_node = WorkGraphNode(
            name="breakdown",
            value=self._make_breakdown_fn(inference_input, inference_config, **kwargs),
            result_pass_down_mode=ResultPassDownMode.NoPassDown,
            enable_result_save=self.enable_result_save,
            resume_with_saved_results=self.resume_with_saved_results,
            retry_on_exceptions=TRANSIENT_RETRY_EXCEPTIONS,
        )
        # Assign _get_result_path so expansion infrastructure can persist records
        _ext = ".json" if self.checkpoint_mode == "jsonfy" else ".pkl"
        _bd_ckpt = self._checkpoint_path("breakdown")
        if _bd_ckpt:
            breakdown_node._get_result_path = (
                lambda rid, *a, _d=_bd_ckpt, _e=_ext, **kw: os.path.join(
                    _d, f"{rid}_result{_e}"
                )
            )
        attempt.graph.start_nodes = [breakdown_node]

        # Propagate expansion settings to the breakdown node (and any future nodes).
        attempt.graph._propagate_expansion_settings()

    async def _arun_attempt(
        self, attempt: _BtaAttempt, inference_input, inference_config, **kwargs
    ):
        """One async attempt: the initial topology, the graph, the full topology."""
        # Emit initial single-node "Breakdown: Running" topology immediately.
        # This shows the user that something is happening before breakdown completes
        # (which can take 30-60s). The full diamond topology replaces it later.
        # Part F: resolve (and seed) the shared Tier-2 reporter once — this runs
        # under the active ctx (public ``ainfer`` bridge); no ctx -> instance attrib.
        _reporter = self._resolve_graph_reporter()
        if _reporter is not None:
            try:
                from agent_foundation.common.inferencers.graph_events import (
                    GraphTopologyEvent,
                    NodeStatus,
                )

                initial_topo = GraphTopologyEvent(
                    nodes=[
                        {
                            "id": "breakdown",
                            "label": "Breakdown",
                            "group": None,
                            "status": NodeStatus.RUNNING,
                        }
                    ],
                    edges=[],
                    layout="horizontal",
                )
                await _reporter.on_graph_topology(initial_topo)
            except Exception as _e:
                _logger.warning("[BTA] initial topology emit failed: %s", _e)

        self._install_breakdown_node(
            attempt, inference_input, inference_config, **kwargs
        )

        # Wire graph event callback BEFORE _arun() so it propagates to
        # expansion nodes via _propagate_settings_to_subgraph() (workgraph.py:670).
        # When breakdown returns GraphExpansionResult, _handle_graph_expansion
        # copies _graph_event_callback from the breakdown node to all worker
        # nodes — so they emit Running/Completed status events in real-time.
        if _reporter is not None:  # Part F: shared Tier-2 sink (resolved above)
            attempt.graph.set_graph_event_callback(self._make_graph_status_callback())

        # Run the graph — expansion handles the rest
        result = await attempt.graph._arun(inference_input, **kwargs)

        result = self._unwrap_workgraph_result(result)

        # Emit full topology after expansion attaches subgraph
        if _reporter is not None and not attempt.topology_emitted:  # Part F sink
            await self._emit_full_topology(attempt)
        return result

    async def _emit_full_topology(self, attempt: _BtaAttempt) -> None:
        """The diamond topology read off the expanded graph, when the breakdown
        did not emit it early."""
        try:
            from agent_foundation.common.inferencers.graph_events import (
                GraphTopologyEvent,
                NodeStatus,
            )

            worker_agg_topology = GraphTopologyEvent.from_work_graph(attempt.graph)
            # Prepend breakdown as virtual node (already completed)
            breakdown_vnode = {
                "id": "breakdown",
                "label": "Breakdown",
                "group": None,
                "status": NodeStatus.COMPLETED,
            }
            worker_agg_topology.nodes.insert(0, breakdown_vnode)
            # Add edges from breakdown to all worker entry nodes
            for n in worker_agg_topology.nodes:
                nid = n["id"]
                # Skip breakdown and aggregator nodes — only add edges to worker entry nodes
                if nid == "breakdown" or nid.endswith("aggregator"):
                    continue
                # Only add edge if not already present
                has_parent_edge = any(
                    e["target"] == nid
                    for e in worker_agg_topology.edges
                    if e["source"] == "breakdown"
                )
                if not has_parent_edge:
                    worker_agg_topology.edges.insert(
                        0, {"source": "breakdown", "target": nid}
                    )
            attempt.pending_topology = worker_agg_topology
            await self._emit_pending_graph_topology(attempt)
            attempt.topology_emitted = True
        except Exception as _e:
            _logger.warning("[BTA] full topology emit failed: %s", _e)

    async def _review_requests_rerun(self, result) -> bool:
        """The interactive results review checkpoint, when enabled: whether the
        user asked to run again."""
        interactive = self._effective("interactive")
        if not (self.enable_checkpoint_results_review and interactive):
            return False
        from agent_foundation.ui.interactive_checkpoint import checkpoint_results_review

        result_str = str(result)[:2000]
        cp_result = await checkpoint_results_review(
            interactive, result_str, default_action="approve"
        )
        return cp_result.action == "rerun"

    def _resolve_worker(
        self, index: int, query_str, sq_args
    ) -> Tuple[Optional[ResolvedStage], Optional[str]]:
        """The worker for subtask ``index`` and its task type, from the single
        ``worker_inferencers`` source. Accepted shapes (see the field docstring):
        list -> static-K round-robin (a borrowed instance); dict -> heterogeneous
        dispatch by task_type; LazyConfigFactory/partial/_FreshCloneFactory -> call
        no-args; callable -> call (sub_query, index). A factory's product is owned.
        """
        from rich_python_utils.config_utils._lazy_config_factory import (
            LazyConfigFactory,
        )

        wi = self.worker_inferencers
        if isinstance(wi, list):
            if not wi:
                return None, None
            return ResolvedStage(wi[index % len(wi)], "borrowed"), None
        task_type = None
        if isinstance(wi, dict):
            task_type = (
                sq_args.get(self.task_type_arg_name, "_default")
                if self.task_type_arg_name
                else "_default"
            )
            factory_entry = wi.get(task_type, wi.get("_default"))
            if isinstance(factory_entry, str):
                factory_entry = wi.get(factory_entry)
            if factory_entry is None:
                raise ValueError(
                    f"No worker factory for task type '{task_type}' "
                    f"and no '_default' fallback"
                )
            if isinstance(factory_entry, dict) and "factory" in factory_entry:
                factory = factory_entry["factory"]
            else:
                factory = factory_entry
        elif wi is not None:
            factory = wi
        else:
            return None, None
        if isinstance(factory, functools.partial) and not isinstance(
            factory, LazyConfigFactory
        ):
            _logger.error(
                "BTA[%s] worker_inferencers%s is a functools.partial, not a "
                "LazyConfigFactory. This causes cross-worker instance sharing. "
                "Ensure factory recipes use LazyConfigFactory (auto-applied by the "
                "config walker for _target_: entries).",
                getattr(self, "name", "?"),
                f"[{task_type}]" if task_type else "",
            )
        no_args = (functools.partial, LazyConfigFactory, _FreshCloneFactory)
        if isinstance(factory, no_args):
            worker = factory()
        else:
            worker = factory(sub_query=query_str, index=index)
        return ResolvedStage(worker, "owned"), task_type

    @staticmethod
    def _own_stage(attempt: _BtaAttempt, stage, label: str) -> None:
        """Take ownership of a stage the attempt built: it closes (``adisconnect``,
        when it has one) when the attempt ends."""
        if id(stage) in attempt.owned_ids:
            raise StageOwnershipError(
                f"The worker factory returned the same {type(stage).__name__} for "
                f"{label} as for an earlier worker of this attempt; a factory must "
                f"build a fresh stage per call."
            )
        attempt.owned_ids.add(id(stage))
        if hasattr(stage, "adisconnect"):
            attempt.ledger.register(stage, label)

    def _refuse_concurrent_duck_stages(
        self, attempt: _BtaAttempt, stages: List[Optional[ResolvedStage]]
    ) -> None:
        """In host mode, one borrowed duck-typed instance can't serve two workers
        that may run at once: it has no invocation the single-flight guard could
        see. Sequential sharing (the sync path, ``max_concurrency=1``) stays valid."""
        if (
            invocation_of(self).mode != "host"
            or not attempt.use_async
            or self.max_concurrency == 1
        ):
            return
        first_index: Dict[int, int] = {}
        for index, stage in enumerate(stages):
            if (
                stage is None
                or stage.owned
                or isinstance(stage.inferencer, InferencerBase)
            ):
                continue
            first = first_index.setdefault(id(stage.inferencer), index)
            if first != index:
                raise UncertifiedConcurrentUseError(
                    f"{type(stage.inferencer).__name__} is borrowed by workers {first} "
                    f"and {index}, which may run concurrently. Give "
                    f"worker_inferencers a factory that builds one per worker."
                )

    def _build_subgraph_spec(self, sub_queries, inference_config=None, **kwargs):
        """Build a SubgraphSpec from parsed sub-queries.

        Returns a SubgraphSpec, which the breakdown node expands into the
        attempt's graph.

        Preserves ALL worker construction logic (homogeneous, heterogeneous,
        todo expansion, per-worker workspace assignment) and ALL aggregator
        construction logic (prompt builder, workspace, checkpoint paths).

        Returns:
            SubgraphSpec with worker nodes (and optional aggregator node).
            entry_nodes = worker nodes.
        """
        from rich_python_utils.common_objects.workflow.common.expansion import (
            SubgraphSpec,
        )

        # Pre-process: expand sub_queries with todos into individual worker queries
        expanded_queries = []
        for sq in sub_queries:
            if isinstance(sq, dict):
                sq_args = sq.get("args", {})
                query_str = sq.get("query", str(sq))
            else:
                sq_args = {}
                query_str = sq

            # Determine if this task type should expand todos
            task_type = None
            if isinstance(self.worker_inferencers, dict) and self.task_type_arg_name:
                task_type = sq_args.get(self.task_type_arg_name, "_default")

            if isinstance(self.expand_todos_to_workers, dict):
                should_expand = (
                    self.expand_todos_to_workers.get(task_type, False)
                    if task_type
                    else False
                )
            else:
                should_expand = self.expand_todos_to_workers

            factory_entry = None
            if task_type and isinstance(self.worker_inferencers, dict):
                factory_entry = self.worker_inferencers.get(
                    task_type, self.worker_inferencers.get("_default")
                )
                if isinstance(factory_entry, dict):
                    should_expand = factory_entry.get("expand_todos", should_expand)

            todos = sq_args.get("todos") if isinstance(sq, dict) else None
            if should_expand and todos and len(todos) > 1:
                desc = sq_args.get("description", query_str)
                for todo in todos:
                    expanded_sq = dict(sq) if isinstance(sq, dict) else {"query": sq}
                    expanded_sq["query"] = (
                        f"**Description**: {desc}\n\n**Todo**:\n- {todo}"
                    )
                    expanded_sq["args"] = dict(sq_args)
                    expanded_queries.append(expanded_sq)
            else:
                expanded_queries.append(sq)

        if len(expanded_queries) != len(sub_queries):
            _logger.info(
                "Expanded %d sub_queries → %d workers (expand_todos_to_workers)",
                len(sub_queries),
                len(expanded_queries),
            )
        shard_chars = [len(self._sub_query_text(sq)) for sq in expanded_queries]
        self._check_worker_query_sizes(shard_chars)
        self._log_input_stats(shard_chars)

        worker_nodes = []
        workers = []  # Fix #5: collect for isolation check
        stages: List[Optional[ResolvedStage]] = []
        _bta_name = self._effective("bta_node_name")
        _bta_prefix = f"{_bta_name}." if _bta_name else ""
        attempt = self._bta_attempt()
        use_async = attempt.use_async
        ws = self._bta_call().effective_workspace

        for i, sq in enumerate(expanded_queries):
            if isinstance(sq, dict):
                query_str = sq.get("query", str(sq))
                sq_args = sq.get("args", {})
            else:
                query_str = sq
                sq_args = {}

            stage, task_type = self._resolve_worker(i, query_str, sq_args)
            worker = None if stage is None else stage.inferencer
            stages.append(stage)
            if stage is not None and stage.owned:
                self._own_stage(attempt, worker, self._worker_child_name(i))

            # Assign child workspace to worker
            if ws is not None and isinstance(worker, InferencerBase):
                worker_ws = ws.child(self._worker_child_name(i))
                # Durably re-bind the worker (a NON-SHARED, per-subtask node) to its
                # child slot — publishes to the worker's child ctx AND sets the
                # durable instance backing so the binding survives resume. The slot
                # MUST match the worker's threaded run_context (``_node_name`` =
                # ``{_bta_prefix}{worker_child_name}``). See _bind_rebuilt_child_ws.
                _worker_node_name = f"{_bta_prefix}{self._worker_child_name(i)}"
                self._bind_rebuilt_child_ws(
                    worker, _worker_node_name, worker_ws, owned=stage.owned
                )
                self.log_info(
                    {
                        "bta_name": _bta_name,
                        "bta_type": type(self).__name__,
                        "worker_idx": i,
                        "worker_type": type(worker).__name__,
                        "worker_child_name": self._worker_child_name(i),
                        "worker_ws_root": worker_ws.root,
                        "bta_ws_root": ws.root,
                    },
                    log_type="WorkerWsAssigned",
                )
                # Part 2 (two-axis model): workers are consumed inputs and are
                # NOT promoted — the aggregator is what promotes its ``outputs/``
                # up to this BTA's ``outputs/``. Output paths are resolved at
                # aggregation time by _resolve_worker_output_paths (after workers
                # finish, so files and LWI symlinks exist).

            workers.append(worker)  # Fix #5: track for isolation check
            _node_name = f"{_bta_prefix}{self._worker_child_name(i)}"
            # A nested BTA runs under this worker's node name for this call only,
            # so two workers sharing one nested BTA keep their own names (B31).
            _worker_values: dict = {}
            if isinstance(worker, BreakdownThenAggregateInferencer):
                _worker_values["bta_node_name"] = _node_name

            from rich_python_utils.common_objects.workflow.common.resumable import (
                Resumable,
            )

            _worker_manages_resume = isinstance(worker, Resumable) and bool(
                getattr(worker, "resume_with_saved_results", False)
            )

            def _make_worker_fn(
                w,
                q,
                is_async,
                manages_resume,
                index,
                _reporter=None,
                _node_id=None,
                _worker_ws=None,
                _stage_kw=None,
            ):
                def _try_load_from_output(worker_rc):
                    if manages_resume:
                        return None
                    output_path = self._worker_backup_output_path(w, worker_rc)
                    if not output_path:
                        return None
                    try:
                        if (
                            os.path.isfile(output_path)
                            and os.path.getsize(output_path) > 0
                        ):
                            with open(output_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            _logger.info(
                                "Backup resume: output file exists, skipping worker: %s (%d bytes)",
                                output_path,
                                len(content),
                            )
                            return content
                        if os.path.isdir(output_path) and os.listdir(output_path):
                            _logger.info(
                                "Backup resume: output dir exists, skipping worker: %s",
                                output_path,
                            )
                            return output_path
                    except OSError:
                        pass
                    return None

                if is_async and hasattr(w, "ainfer"):

                    async def async_worker_fn(*_args, **_kwargs):
                        try:
                            _worker_rc = self._rc_child(
                                _node_id or "worker", workspace=_worker_ws
                            )
                            cached = _try_load_from_output(_worker_rc)
                            if cached is not None:
                                if _reporter is not None and _node_id is not None:
                                    try:
                                        await _reporter.on_node_stream(
                                            _node_id, str(cached), is_final=True
                                        )
                                    except Exception:
                                        pass
                                return _wrap_outcome(index, _node_id, value=cached)
                            self._check_cancelled()  # §2.1/P-#6: halt at worker boundary
                            # v4 Phase 3.1 quorum wrap: when min_successful_workers > 0,
                            # catch terminal failures and return a failure outcome
                            # rather than raising. The aggregation step raises when
                            # fewer than min_successful_workers succeeded.
                            # Default (min_successful_workers == 0) preserves the
                            # historical raise-on-failure behavior used by all
                            # non-MultiFlow BTAs.
                            _quorum_active = (
                                getattr(self, "min_successful_workers", 0) > 0
                            )
                            try:
                                result = await w.ainfer(
                                    q,
                                    inference_config=inference_config,
                                    run_context=_worker_rc,
                                    **self.worker_inference_args,
                                    **(_stage_kw or {}),
                                )
                                # WorkGraph awaits only a bare awaitable node
                                # result; inside an outcome it would never run.
                                result = await maybe_await(result)
                                # Publish-up: workers are factory-ephemeral (never
                                # stored on ``self``), so a parent reviewing THIS
                                # node's output cannot reach an author leaf to learn
                                # the task contract.
                                self._harvest_worker_contract(
                                    attempt, index, w, _worker_rc
                                )
                            except BaseException as _werr:
                                # CancelledError + KeyboardInterrupt must NOT be
                                # swallowed (BaseException-not-Exception); pass them
                                # straight through. Everything else (HopelessOutputError,
                                # InferencerExecutionError, NodeExecutionFailed, generic
                                # Exception after retry exhaustion) becomes a failure
                                # outcome iff quorum is active.
                                import asyncio as _asyncio

                                if isinstance(
                                    _werr,
                                    (
                                        _asyncio.CancelledError,
                                        KeyboardInterrupt,
                                        SystemExit,
                                        MemoryError,
                                    ),
                                ) or not isinstance(_werr, Exception):
                                    # U4-B guard 1: never-contain floor — cooperative
                                    # cancellation + any non-Exception BaseException
                                    # ALWAYS propagate, regardless of quorum.
                                    raise
                                if isinstance(
                                    _werr, getattr(self, "surfaceable_exceptions", ())
                                ):
                                    # U4-B: per-inferencer surfaceable allowlist — always
                                    # propagate (never contain), like the floor.
                                    raise
                                if not _quorum_active:
                                    raise
                                # Quorum mode: mark this worker failed, emit
                                # node_status(error) for UI red-render, return a
                                # failure outcome.
                                _failure = f"{type(_werr).__name__}: {str(_werr)[:500]}"
                                if _reporter is not None and _node_id is not None:
                                    try:
                                        await _reporter.on_node_status(
                                            _node_id, "error", error=_failure
                                        )
                                    except Exception:
                                        pass
                                _logger.warning(
                                    "[BTA quorum] worker %r failed (%s); returning "
                                    "a failure outcome — aggregation proceeds if "
                                    "survivors >= min_successful_workers=%d",
                                    _node_id,
                                    type(_werr).__name__,
                                    getattr(self, "min_successful_workers", 0),
                                )
                                return _wrap_outcome(index, _node_id, failure=_failure)
                            if _reporter is not None and _node_id is not None:
                                try:
                                    await _reporter.on_node_stream(
                                        _node_id,
                                        str(result) if result else "",
                                        is_final=True,
                                    )
                                except Exception:
                                    pass
                            return _wrap_outcome(index, _node_id, value=result)
                        finally:
                            # Cross-flow barrier safety net (no-op unless this is a
                            # coordinated MultiFlow worker, tagged with ``_cross_flow_index``).
                            # Covers the paths that satisfy the worker WITHOUT running its
                            # ``_ainfer`` (backup-resume cache hit, cancel at the boundary),
                            # where the LWI-level depart can't fire. ``leave`` is idempotent,
                            # so the normal "ran _ainfer" double-call is harmless.
                            await self._cross_flow_depart_if_tagged(w)

                    return async_worker_fn
                else:

                    def worker_fn(*_args, **_kwargs):
                        _worker_rc = self._rc_child(
                            _node_id or "worker", workspace=_worker_ws
                        )
                        cached = _try_load_from_output(_worker_rc)
                        if cached is not None:
                            return _wrap_outcome(index, _node_id, value=cached)
                        self._check_cancelled()  # §2.1/P-#6: sync worker boundary
                        _quorum_active = getattr(self, "min_successful_workers", 0) > 0
                        try:
                            if hasattr(w, "infer"):
                                result = w.infer(
                                    q,
                                    inference_config=inference_config,
                                    run_context=_worker_rc,
                                    **self.worker_inference_args,
                                    **(_stage_kw or {}),
                                )
                            else:
                                result = w(q)
                        except BaseException as _werr:
                            # U4-B: mirror the async worker containment on the SYNC
                            # path. Never-contain floor (KeyboardInterrupt/SystemExit/
                            # MemoryError + any non-Exception BaseException) always
                            # propagates; otherwise, under an active quorum, mark the
                            # worker failed and return a failure outcome instead of
                            # sinking the whole fan-out.
                            if isinstance(
                                _werr,
                                (KeyboardInterrupt, SystemExit, MemoryError),
                            ) or not isinstance(_werr, Exception):
                                raise
                            if not _quorum_active:
                                raise
                            return _wrap_outcome(
                                index,
                                _node_id,
                                failure=f"{type(_werr).__name__}: {str(_werr)[:500]}",
                            )
                        self._harvest_worker_contract(attempt, index, w, _worker_rc)
                        return _wrap_outcome(index, _node_id, value=result)

                    return worker_fn

            worker_group = (
                task_type if isinstance(self.worker_inferencers, dict) else None
            )

            # Graph visualization (Part F / GT#13): node identity = the worker's
            # ctx path, NOT its Python instance. ``_reporter`` is THIS
            # orchestrator's sink, already namespaced by its own ctx path; tagging
            # it with the worker's flat node id (``_node_name``) places the worker
            # at ``<this-path>/<node_name>`` — concurrency-safe because the sink is
            # resolved from the shared Tier-2 binding and the node id is a value
            # captured per WorkGraphNode (no shared mutable instance state).
            _reporter = self._resolve_graph_reporter()
            if _reporter is not None:
                if isinstance(worker, BreakdownThenAggregateInferencer):
                    # Nested BTA: do NOT push a reporter onto the instance — it
                    # self-resolves from the shared Tier-2 sink + its OWN ctx path
                    # (so the instance stays definition-only, safe to share). Drop
                    # the legacy ``_bta_prefix`` disambiguation name only when it
                    # has no explicitly pre-wired reporter; path nesting (ctx.path)
                    # now provides node-id uniqueness.
                    if getattr(worker, "graph_reporter", None) is None:
                        _worker_values["bta_node_name"] = None
                else:
                    # Leaf worker: the per-token streaming niceties
                    # (interactive / stream_observer) are derived from this
                    # path-namespaced sink and tagged with the worker's flat node
                    # id.
                    _worker_values.update(
                        self._reporter_stage_values(worker, _reporter, _node_name)
                    )
            # Per-call values reach the worker as this call's invocation keywords
            # (B18), never written onto it.
            _worker_kw = self._stage_keywords(worker, **_worker_values)

            # A worker is a "container" for the graph viz if it will emit its
            # own sub-topology — broader than just BTA. After v4 Phase 5.1,
            # LWI ALSO emits its step graph (per-round in dynamic mode), so
            # it joins Dual + MFDual + MultiFlow + BTA as a container.
            try:
                from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
                    DualInferencer,
                )
            except Exception:
                DualInferencer = None  # type: ignore
            try:
                from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
                    LinearWorkflowInferencer,
                )
            except Exception:
                LinearWorkflowInferencer = None  # type: ignore
            _is_container = (
                isinstance(worker, BreakdownThenAggregateInferencer)
                or (DualInferencer is not None and isinstance(worker, DualInferencer))
                or (
                    LinearWorkflowInferencer is not None
                    and isinstance(worker, LinearWorkflowInferencer)
                )
            ) and (_reporter is not None)
            # The output-text emit (in the worker_fn closure) tags the worker's
            # result with its flat node id on THIS orchestrator's path-namespaced
            # sink — a per-closure captured value, concurrency-safe.
            _w_reporter = _reporter
            node = WorkGraphNode(
                name=_node_name,
                value=_make_worker_fn(
                    worker,
                    query_str,
                    use_async,
                    _worker_manages_resume,
                    i,
                    _reporter=_w_reporter,
                    _node_id=_node_name,
                    # Root-cause fix: give the worker context its INTENDED workspace
                    # (worker_<i>) verbatim instead of path-mirroring its namespaced
                    # ctx-node name (plan_bta.worker_<i>). Descendants then root the
                    # whole subtree under worker_<i>; the node name never leaks on
                    # disk. Guarded so worker_ws is only referenced when it was set.
                    _worker_ws=(
                        worker_ws
                        if ws is not None and isinstance(worker, InferencerBase)
                        else None
                    ),
                    _stage_kw=_worker_kw,
                ),
                result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
                group=worker_group,
                enable_result_save=StepResultSaveOptions.SkipResumable,
                resume_with_saved_results=ResumeMode.SkipResumable,
                checkpoint_mode=self.checkpoint_mode,
                retry_on_exceptions=TRANSIENT_RETRY_EXCEPTIONS,
            )
            _wname = self._worker_child_name(i)
            if ws is not None:
                _w_ckpt = os.path.join(
                    str(ws.root),
                    "children",
                    _wname,
                    "checkpoints",
                )
                _w_ext = ".json" if self.checkpoint_mode == "jsonfy" else ".pkl"
                node._get_result_path = (
                    lambda rid, *a, _d=_w_ckpt, _e=_w_ext, **kw: os.path.join(
                        _d, f"{rid}_result{_e}"
                    )
                )
            node.load_result = _rerun_failed_loader(node.load_result)
            node._get_args_for_downstream = _outcome_passdown(
                node._get_args_for_downstream, i, _node_name
            )
            _raw_label = (
                str(query_str)[:120]
                if isinstance(query_str, str)
                else query_str.get("description", query_str.get("query", _wname))[:120]
                if isinstance(query_str, dict)
                else _wname
            )
            _raw_label = _raw_label.replace("**", "").replace("__", "").strip()
            for _prefix in (
                "Description:",
                "description:",
                "Task:",
                "task:",
                "Query:",
                "query:",
            ):
                if _raw_label.startswith(_prefix):
                    _raw_label = _raw_label[len(_prefix) :].strip()
                    break
            node._viz_label = _raw_label[:80]
            if _is_container:
                node._is_container = True

            worker_nodes.append(node)

        agg_node = None
        if not self.disable_aggregator and self.aggregator_inferencer is not None:
            _bta_self = self

            def _make_agg_fn(
                agg_inf,
                prompt_builder,
                original_query,
                is_async,
                _reporter=None,
                _inference_args=None,
                _stage_kw=None,
            ):
                _agg_extra = {**(_inference_args or {}), **(_stage_kw or {})}
                if is_async and hasattr(agg_inf, "ainfer"):

                    async def async_agg_fn(*worker_results, **_kwargs):
                        # U4-B: an unmet quorum raises (fail loud) rather than
                        # degrading to a synthetic-aggregation stub that would flow
                        # into review/fix as if it succeeded.
                        agg_input, results, failures, paths = (
                            _bta_self._prepare_aggregator_input(
                                agg_inf,
                                prompt_builder,
                                worker_results,
                                workers,
                                original_query,
                                shard_chars,
                            )
                        )
                        try:
                            result = await agg_inf.ainfer(
                                agg_input,
                                inference_config=inference_config,
                                run_context=self._rc_child("aggregator"),
                                **_agg_extra,
                            )
                        except Exception as _agg_exc:
                            _bta_self.log_warning(
                                {
                                    "event": "AGGREGATOR_FAILED",
                                    "message": (
                                        "All aggregator retries exhausted. Producing "
                                        "synthetic aggregation with upstream worker "
                                        "paths so downstream review/fix can still consume."
                                    ),
                                    "exception": str(_agg_exc),
                                    "num_workers": len(results),
                                },
                                "AggregatorFallback",
                            )
                            result = _bta_self._build_synthetic_aggregation(
                                results,
                                original_query,
                                worker_output_paths=paths,
                                worker_failures=failures,
                            )
                        if _reporter is not None:
                            try:
                                await _reporter.on_node_stream(
                                    "aggregator",
                                    str(result) if result else "",
                                    is_final=True,
                                )
                            except Exception:
                                pass
                        return result

                    return async_agg_fn
                else:

                    def agg_fn(*worker_results, **_kwargs):
                        agg_input = _bta_self._prepare_aggregator_input(
                            agg_inf,
                            prompt_builder,
                            worker_results,
                            workers,
                            original_query,
                            shard_chars,
                        )[0]
                        if hasattr(agg_inf, "infer"):
                            return agg_inf.infer(
                                agg_input,
                                inference_config=inference_config,
                                run_context=self._rc_child("aggregator"),
                                **(_stage_kw or {}),
                            )
                        return agg_inf(agg_input)

                    return agg_fn

            original_query = kwargs.get("_original_query", "")

            agg_stage = self._aggregator_stage()
            agg_inf = agg_stage.inferencer
            if ws is not None and isinstance(agg_inf, InferencerBase):
                # Durably re-bind the aggregator (a NON-SHARED, dispatched-once node)
                # so its workspace survives resume, where the active ctx is not
                # guaranteed stable across the recovery gate. Also publishes to the
                # child ctx for fresh-path readers. See _bind_rebuilt_child_ws.
                self._bind_rebuilt_child_ws(
                    agg_inf,
                    "aggregator",
                    ws.child("aggregator"),
                    owned=agg_stage.owned,
                )

            _agg_node_name = f"{_bta_prefix}aggregator" if _bta_prefix else "aggregator"
            # Part F: the aggregator's per-token observer + output emit go through
            # THIS orchestrator's path-namespaced sink, tagged with the flat
            # aggregator node id (``_bta_prefix`` is empty under a ctx — node
            # identity comes from the ctx path). Byte-identical with no ctx.
            _agg_reporter = self._resolve_graph_reporter()
            _agg_kw = self._stage_keywords(
                agg_inf,
                **self._reporter_stage_values(
                    agg_inf, _agg_reporter, _agg_node_name, interactive=False
                ),
            )

            agg_node = WorkGraphNode(
                name=_agg_node_name,
                value=_make_agg_fn(
                    agg_inf,
                    self.aggregator_prompt_builder,
                    original_query,
                    use_async,
                    _reporter=_agg_reporter,
                    _inference_args=kwargs.get("_inference_args"),
                    _stage_kw=_agg_kw,
                ),
                result_pass_down_mode=ResultPassDownMode.NoPassDown,
                enable_result_save=self.enable_result_save,
                resume_with_saved_results=self.resume_with_saved_results,
                checkpoint_mode=self.checkpoint_mode,
                retry_on_exceptions=TRANSIENT_RETRY_EXCEPTIONS,
            )
            _ext = ".json" if self.checkpoint_mode == "jsonfy" else ".pkl"
            _agg_ckpt = self._checkpoint_path("aggregator_result")
            if _agg_ckpt:
                agg_node._get_result_path = (
                    lambda rid, *a, _d=_agg_ckpt, _e=_ext, **kw: os.path.join(
                        _d, f"{rid}_result{_e}"
                    )
                )

            # Wire all workers → aggregation
            for wn in worker_nodes:
                wn.add_next(agg_node)

        all_nodes = list(worker_nodes)
        if agg_node is not None:
            all_nodes.append(agg_node)

        attempt.worker_child_names = [
            self._worker_child_name(i) for i in range(len(workers))
        ]
        self._refuse_concurrent_duck_stages(attempt, stages)
        # Fix #5: check for shared sub-inferencer instances across workers
        self._validate_worker_isolation(workers)

        return SubgraphSpec(
            nodes=all_nodes,
            entry_nodes=list(worker_nodes),
        )

    @staticmethod
    def _reporter_stage_values(stage, reporter, node_name, *, interactive=True) -> dict:
        """The per-call visualization values ``stage`` gets from this BTA's graph
        reporter, tagged with its node id (none without a reporter). The reporter
        is asked for an observer only for a stage that has one."""
        values: dict = {}
        if reporter is None:
            return values
        if interactive and hasattr(reporter, "node_interactive"):
            values["interactive"] = reporter.node_interactive(node_name)
        if hasattr(stage, "stream_observer") and hasattr(
            reporter, "node_stream_observer"
        ):
            values["stream_observer"] = reporter.node_stream_observer(node_name)
        return values

    @staticmethod
    def _stage_keywords(stage, **values) -> dict:
        """B18: the per-call ``values`` a stage receives as the invocation keywords
        its class declares; nothing is written onto it. A duck-typed stage (not an
        ``InferencerBase``) has no invocation to receive them, so it keeps the
        documented attribute protocol: each value is set on an attribute it has."""
        if isinstance(stage, InferencerBase):
            declared = stage._invocation_keywords()
            return {name: value for name, value in values.items() if name in declared}
        for name, value in values.items():
            if hasattr(stage, name):
                setattr(stage, name, value)
        return {}

    def _make_breakdown_fn(
        self, inference_input, inference_config=None, **_inference_args
    ):
        """Create the breakdown node callable that returns GraphExpansionResult.

        Handles predefined sub-queries, breakdown_only, interactive selection,
        and resume via the committed plan (_committed_sub_queries) or the promoted
        breakdown checkpoint (_load_promoted_breakdown). On a fresh breakdown it
        promotes the decomposition into the parent's checkpoints/
        (_promote_child_checkpoints), which restores the aggregator's guidance on
        resume, and commits the plan once the workers are built (_commit_plan),
        which resume rebuilds the fan-out from.
        """
        from rich_python_utils.common_objects.workflow.common.expansion import (
            GraphExpansionResult,
        )

        _bta = self
        _inf_input = inference_input
        _inf_config = inference_config
        attempt = self._bta_attempt()

        if attempt.use_async:

            async def _breakdown_fn(*args, **kwargs):
                # Step 0: resume short-circuit — start from the committed plan
                # (§5.12; already truncated and selected), else reuse the promoted
                # breakdown (checkpoints/breakdown/decomposed_subtasks.json) instead
                # of re-running the breakdown LLM. The latter is gated on
                # resume_with_saved_results (mirroring the expansion-record
                # reconstruction gate) so a fresh run never short-circuits; the
                # promoted file does not exist yet at this point on a fresh run
                # anyway.
                committed = _bta._committed_sub_queries()
                sub_queries = (
                    _bta._load_promoted_breakdown()[0]
                    if committed is None and _bta.resume_with_saved_results
                    else committed
                )
                raw_output = None
                _from_predefined = False

                if sub_queries is not None:
                    # Resuming from checkpoint
                    pass
                elif _bta.predefined_sub_queries is not None:
                    _from_predefined = True
                    if _bta.breakdown_only:
                        _logger.warning(
                            "predefined_sub_queries is set but breakdown_only=True — "
                            "breakdown_only ignored (no LLM breakdown to stop after)."
                        )
                    sub_queries = _bta._resolve_predefined_sub_queries()
                else:
                    # Run breakdown inferencer
                    if _bta.breakdown_inferencer is None:
                        raise ValueError(
                            "breakdown_inferencer must be set when predefined_sub_queries is None. "
                            "Either provide a breakdown_inferencer or set predefined_sub_queries."
                        )
                    # Wire stream_observer for live breakdown streaming (Part F:
                    # through THIS BTA's path-namespaced shared sink, tagged with
                    # the flat "breakdown" node id; byte-identical with no ctx).
                    _bd_kw = _bta._stage_keywords(
                        _bta.breakdown_inferencer,
                        **_bta._reporter_stage_values(
                            _bta.breakdown_inferencer,
                            _bta._resolve_graph_reporter(),
                            "breakdown",
                            interactive=False,
                        ),
                    )

                    _bta._bind_breakdown_workspace()
                    if hasattr(_bta.breakdown_inferencer, "ainfer"):
                        raw_output = await _bta.breakdown_inferencer.ainfer(
                            _inf_input,
                            inference_config=_inf_config,
                            run_context=_bta._rc_child("breakdown"),
                            **_inference_args,
                            **_bd_kw,
                        )
                    else:
                        raw_output = _bta.breakdown_inferencer.infer(
                            _inf_input,
                            inference_config=_inf_config,
                            run_context=_bta._rc_child("breakdown"),
                            **_bd_kw,
                        )

                    # Guard: detect API error responses
                    _raw_str = str(raw_output).strip()
                    _ERROR_PATTERNS = [
                        "An unknown error occurred",
                        "RECONNECT_SUPPORTED",
                        "peer closed connection",
                        "Internal Server Error",
                    ]
                    if (
                        any(p in _raw_str for p in _ERROR_PATTERNS)
                        and len(_raw_str) < 200
                    ):
                        raise RuntimeError(
                            f"Breakdown returned API error instead of subtasks: {_raw_str[:100]}"
                        )

                    # Parse
                    if _bta.breakdown_parser is not None:
                        sub_queries = _bta.breakdown_parser(raw_output)
                    elif _bta.breakdown_format == "json_subtasks":
                        sub_queries, attempt.aggregation_guidance = (
                            _bta._parse_json_breakdown(raw_output)
                        )
                    elif _bta.breakdown_format == "numbered_list":
                        sub_queries = parse_numbered_list(str(raw_output))
                    elif isinstance(raw_output, list):
                        sub_queries = raw_output
                    else:
                        sub_queries = parse_numbered_list(str(raw_output))

                    # Promote the breakdown's decomposition into the parent's
                    # checkpoints/ so resume rebuilds the fan-out and restores the
                    # aggregator's guidance from it (see _promote_child_checkpoints
                    # / _load_promoted_breakdown).
                    _bta._promote_child_checkpoints(
                        _bta.breakdown_inferencer,
                        "breakdown",
                        parent_ws=_bta._bta_call().effective_workspace,
                    )

                # Apply max_breakdown cap
                if (
                    committed is None
                    and _bta.max_breakdown is not None
                    and len(sub_queries) > _bta.max_breakdown
                ):
                    sub_queries = sub_queries[: _bta.max_breakdown]

                if not sub_queries:
                    if (
                        not _bta.disable_aggregator
                        and _bta.aggregator_inferencer is not None
                    ):
                        raise RuntimeError(
                            "BTA breakdown produced zero sub_queries but an "
                            "aggregator is configured — nothing to fan out or "
                            "aggregate; refusing to silently return the raw "
                            f"breakdown output. raw_output={str(raw_output)[:200]!r}"
                        )
                    return raw_output if raw_output is not None else ""

                # Breakdown-only mode (skip when predefined_sub_queries — already warned)
                if _bta.breakdown_only and not _from_predefined:
                    return raw_output if raw_output is not None else sub_queries

                # Interactive sub-query selection
                if (
                    committed is None
                    and _bta.enable_checkpoint_sub_query_selection
                    and _bta._effective("interactive")
                ):
                    from agent_foundation.ui.interactive_checkpoint import (
                        checkpoint_breakdown_review,
                    )

                    cp_result = await checkpoint_breakdown_review(
                        _bta._effective("interactive"),
                        sub_queries,
                        default_action="approve",
                    )
                    if cp_result.action == "select" and cp_result.selected_indices:
                        sub_queries = [
                            sub_queries[i]
                            for i in cp_result.selected_indices
                            if i < len(sub_queries)
                        ]
                    if not sub_queries:
                        return raw_output if raw_output is not None else ""

                # Emit breakdown result as node_stream (Part F: path-namespaced sink)
                _bd_emit_reporter = _bta._resolve_graph_reporter()
                if _bd_emit_reporter is not None:
                    try:
                        _summary = []
                        for _i, _sq in enumerate(
                            sub_queries
                            if isinstance(sub_queries, list)
                            else [sub_queries]
                        ):
                            if isinstance(_sq, dict):
                                _desc = _sq.get("query", str(_sq))
                            else:
                                _desc = str(_sq)
                            if len(_desc) > 300:
                                _desc = _desc[:297] + "..."
                            _summary.append(f"**{_i + 1}.** {_desc}")
                        _bd_content = "\n\n".join(_summary)
                        await _bd_emit_reporter.on_node_stream(
                            "breakdown", _bd_content, is_final=True
                        )
                    except Exception as _e:
                        _logger.warning(
                            "[BTA] breakdown node_stream emit failed: %s", _e
                        )

                # On resume, _reconstruct_graph_expansions already attached workers
                # to the breakdown node. WorkGraphNode._run resets _expansion_applied
                # to False, so we can't rely on that flag. Instead, check if the
                # breakdown node already has downstream nodes (workers) attached.
                _graph = attempt.graph
                _bd_node = (
                    _graph.start_nodes[0]
                    if _graph is not None and _graph.start_nodes
                    else None
                )
                if _bd_node and _bd_node.next:
                    # Workers already attached from reconstruction — skip expansion
                    return sub_queries

                # Build SubgraphSpec
                subgraph = _bta._build_subgraph_spec(
                    sub_queries,
                    inference_config=_inf_config,
                    _original_query=_inf_input,
                    _inference_args=_inference_args,
                )
                _bta._commit_plan(sub_queries, subgraph)

                # Emit full topology IMMEDIATELY so the UI shows worker nodes
                # before they start running. Without this, the UI shows only
                # "Breakdown: Running" until _arun() returns (after all workers
                # and aggregator finish), which can be minutes.
                if (
                    _bta._resolve_graph_reporter() is not None
                    and not attempt.topology_emitted
                ):
                    try:
                        from agent_foundation.common.inferencers.graph_events import (
                            GraphTopologyEvent,
                            NodeStatus,
                        )

                        topo_nodes = [
                            {
                                "id": "breakdown",
                                "label": "Breakdown",
                                "group": None,
                                "status": NodeStatus.COMPLETED,
                            },
                        ]
                        topo_edges = []
                        worker_names = []
                        for n in subgraph.entry_nodes:
                            viz_label = getattr(n, "_viz_label", n.name)
                            topo_nodes.append(
                                {
                                    "id": n.name,
                                    "label": viz_label,
                                    "group": getattr(n, "group", None),
                                    "status": NodeStatus.PENDING,
                                }
                            )
                            topo_edges.append({"source": "breakdown", "target": n.name})
                            worker_names.append(n.name)
                        for n in subgraph.nodes:
                            if n not in subgraph.entry_nodes:
                                viz_label = getattr(n, "_viz_label", n.name)
                                topo_nodes.append(
                                    {
                                        "id": n.name,
                                        "label": viz_label,
                                        "group": getattr(n, "group", None),
                                        "status": NodeStatus.PENDING,
                                    }
                                )
                                for wn in worker_names:
                                    topo_edges.append({"source": wn, "target": n.name})

                        topo = GraphTopologyEvent(
                            nodes=topo_nodes,
                            edges=topo_edges,
                            layout="horizontal",
                        )
                        attempt.pending_topology = topo
                        await _bta._emit_pending_graph_topology(attempt)
                        attempt.topology_emitted = True
                    except Exception as _e:
                        _logger.warning("[BTA] early topology emit failed: %s", _e)

                # Determine expansion_id
                has_aggregator = (
                    not _bta.disable_aggregator
                    and _bta.aggregator_inferencer is not None
                )
                expansion_id = "bta_diamond" if has_aggregator else "bta_workers"

                return GraphExpansionResult(
                    result=sub_queries,
                    subgraph=subgraph,
                    expansion_id=expansion_id,
                    seed=None,
                    reconstruct_from_seed=None,
                    attach_mode="insert",
                )

            return _breakdown_fn
        else:

            def _breakdown_fn_sync(*args, **kwargs):
                # Step 0: resume short-circuit — start from the committed plan
                # (§5.12; already truncated and selected), else reuse the promoted
                # breakdown (checkpoints/breakdown/decomposed_subtasks.json) instead
                # of re-running the breakdown LLM. The latter is gated on
                # resume_with_saved_results (mirroring the expansion-record
                # reconstruction gate) so a fresh run never short-circuits; the
                # promoted file does not exist yet at this point on a fresh run
                # anyway.
                committed = _bta._committed_sub_queries()
                sub_queries = (
                    _bta._load_promoted_breakdown()[0]
                    if committed is None and _bta.resume_with_saved_results
                    else committed
                )
                raw_output = None
                _from_predefined = False

                if sub_queries is not None:
                    pass
                elif _bta.predefined_sub_queries is not None:
                    _from_predefined = True
                    if _bta.breakdown_only:
                        _logger.warning(
                            "predefined_sub_queries is set but breakdown_only=True — "
                            "breakdown_only ignored (no LLM breakdown to stop after)."
                        )
                    sub_queries = _bta._resolve_predefined_sub_queries()
                else:
                    if _bta.breakdown_inferencer is None:
                        raise ValueError(
                            "breakdown_inferencer must be set when predefined_sub_queries is None. "
                            "Either provide a breakdown_inferencer or set predefined_sub_queries."
                        )
                    _bta._bind_breakdown_workspace()
                    raw_output = _bta.breakdown_inferencer.infer(
                        _inf_input,
                        inference_config=_inf_config,
                        run_context=_bta._rc_child("breakdown"),
                    )

                    if _bta.breakdown_parser is not None:
                        sub_queries = _bta.breakdown_parser(raw_output)
                    elif _bta.breakdown_format == "json_subtasks":
                        sub_queries, attempt.aggregation_guidance = (
                            _bta._parse_json_breakdown(raw_output)
                        )
                    elif _bta.breakdown_format == "numbered_list":
                        sub_queries = parse_numbered_list(str(raw_output))
                    elif isinstance(raw_output, list):
                        sub_queries = raw_output
                    else:
                        sub_queries = parse_numbered_list(str(raw_output))

                    # Promote the breakdown's decomposition into the parent's
                    # checkpoints/ so resume rebuilds the fan-out and restores the
                    # aggregator's guidance from it (see _promote_child_checkpoints
                    # / _load_promoted_breakdown).
                    _bta._promote_child_checkpoints(
                        _bta.breakdown_inferencer,
                        "breakdown",
                        parent_ws=_bta._bta_call().effective_workspace,
                    )

                if (
                    committed is None
                    and _bta.max_breakdown is not None
                    and len(sub_queries) > _bta.max_breakdown
                ):
                    sub_queries = sub_queries[: _bta.max_breakdown]

                if not sub_queries:
                    if (
                        not _bta.disable_aggregator
                        and _bta.aggregator_inferencer is not None
                    ):
                        raise RuntimeError(
                            "BTA breakdown produced zero sub_queries but an "
                            "aggregator is configured — nothing to fan out or "
                            "aggregate; refusing to silently return the raw "
                            f"breakdown output. raw_output={str(raw_output)[:200]!r}"
                        )
                    return raw_output if raw_output is not None else ""

                # Breakdown-only mode (skip when predefined_sub_queries — already warned)
                if _bta.breakdown_only and not _from_predefined:
                    return raw_output if raw_output is not None else sub_queries

                # On resume, _reconstruct_graph_expansions already attached workers
                # to the breakdown node. WorkGraphNode._run resets _expansion_applied
                # to False, so we can't rely on that flag. Instead, check if the
                # breakdown node already has downstream nodes (workers) attached.
                _graph = attempt.graph
                _bd_node = (
                    _graph.start_nodes[0]
                    if _graph is not None and _graph.start_nodes
                    else None
                )
                if _bd_node and _bd_node.next:
                    # Workers already attached from reconstruction — skip expansion
                    return sub_queries

                subgraph = _bta._build_subgraph_spec(
                    sub_queries,
                    inference_config=_inf_config,
                    _original_query=_inf_input,
                    _inference_args=_inference_args,
                )
                _bta._commit_plan(sub_queries, subgraph)

                has_aggregator = (
                    not _bta.disable_aggregator
                    and _bta.aggregator_inferencer is not None
                )
                expansion_id = "bta_diamond" if has_aggregator else "bta_workers"

                return GraphExpansionResult(
                    result=sub_queries,
                    subgraph=subgraph,
                    expansion_id=expansion_id,
                    seed=None,
                    reconstruct_from_seed=None,
                    attach_mode="insert",
                )

            return _breakdown_fn_sync

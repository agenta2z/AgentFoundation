"""BreakdownThenAggregateInferencer — diamond-shaped WorkGraph-based inferencer.

Breaks a query into sub-queries, runs workers in parallel via WorkGraph,
and optionally aggregates results. Uses dual inheritance pattern
(InferencerBase, WorkGraph) following DualInferencer/PTI precedent.
"""

import asyncio
import functools
import json
import logging
import os
import re
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.template_defaults import (
    AGGREGATION_DEFAULTS,
    BREAKDOWN_TEMPLATE_DEFAULTS,
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
    candidate_subdirs=("outputs/final_deliverables", "outputs"),
):
    """Detect deliverable conflicts across workers and auto-promote agreed files.

    Thin wrapper around the generic ``find_conflicting_and_agreed_files``.
    Resolves each worker's output root (preferring final_deliverables/)
    and delegates to the generic diff + copy helpers.
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

    Both produce identical behavior: worker outputs pushed into
    ``aggregator_inferencer.template_extra_feed["upstream_artifacts"]``,
    breakdown's ``aggregation_guidance`` (if captured) forwarded to
    ``template_extra_feed["aggregation_guidance"]``, and the original BTA
    query returned as the aggregator's ``inference_input`` (rendered into
    ``{{ input }}`` by the wrapper).

    The class-level flag is the canonical pattern (mirrors MFDual's
    ``inject_upstream_artifacts``); this factory is kept for backward
    compatibility with YAMLs that still wire
    ``aggregator_prompt_builder: UpstreamInjectingAggregatorPromptBuilder``.
    """

    def _builder(
        worker_results, original_query=None, worker_output_paths=None, bta=None
    ):
        if bta is not None:
            bta._inject_aggregator_extra_feed(
                worker_results,
                worker_output_paths,
            )
        return original_query or ""

    return _builder


def make_conflict_aware_prompt_builder(
    conflict_resolution_mode="delegate_to_aggregator",
    candidate_subdirs=("outputs/final_deliverables", "outputs"),
    deliverables_subdir="final_deliverables",
):
    """Factory: returns an aggregator_prompt_builder that detects conflicts.

    The closure accepts BTA's hook signature (with optional ``bta=`` kwarg):
        (worker_results, original_query=..., worker_output_paths=..., bta=...) -> str

    In ``delegate_to_aggregator`` mode:
    1. Walks worker outputs, hashes files, categorizes as agreed/conflicting
    2. Auto-promotes agreed files to deliverables_dst
    3. Injects structured data (``deliverables_promoted``, ``deliverables_with_conflicts``,
       ``deliverables_dst``, ``worker_summaries``) into ``bta.aggregator_inferencer.template_extra_feed``
       so the aggregator template gets them as top-level ``{{ vars }}``
    4. Returns worker summaries text as ``{{ input }}``
    """

    def _builder(
        worker_results, original_query=None, worker_output_paths=None, bta=None
    ):
        worker_summaries = [str(r) for r in worker_results]

        # Build "summary text" — for aggregators with local file access, pass
        # paths only (avoids inlining 100KB+ of worker text per result, which
        # would exceed OS ARG_MAX when the prompt is passed to subprocess).
        # For non-local aggregators (e.g., RovoChat), inline full text.
        agg_has_local = (
            bta is not None
            and getattr(bta, "aggregator_inferencer", None) is not None
            and getattr(bta.aggregator_inferencer, "has_local_access", False)
        )

        def _format_result(idx, res, path):
            if agg_has_local and path:
                return f"### Upstream Outcome {idx + 1}\n(See file: `{path}`)"
            return f"### Upstream Outcome {idx + 1}\n{res}"

        paths = worker_output_paths or [None] * len(worker_results)
        default_text = "\n\n".join(
            _format_result(i, r, paths[i] if i < len(paths) else None)
            for i, r in enumerate(worker_results)
        )

        # Inject upstream_artifacts EARLY — before any return path — so the
        # aggregation preamble template always has worker content available.
        if bta is not None and getattr(bta, "aggregator_inferencer", None) is not None:
            agg_inf = bta.aggregator_inferencer
            if hasattr(agg_inf, "template_extra_feed"):
                agg_inf.template_extra_feed["upstream_artifacts"] = default_text

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
        fd_path = os.path.join(ws_root, "outputs", deliverables_subdir)
        deliverables_dst = (
            fd_path if os.path.isdir(fd_path) else os.path.join(ws_root, "outputs")
        )
        os.makedirs(deliverables_dst, exist_ok=True)

        promoted, conflicts = _detect_conflicts_and_promote(
            deliverables_dst,
            children_dir,
            candidate_subdirs,
        )

        conflicts_grouped = group_conflicts_by_parent(conflicts, depth=2)

        if bta is not None and getattr(bta, "aggregator_inferencer", None) is not None:
            agg_inf = bta.aggregator_inferencer
            if hasattr(agg_inf, "template_extra_feed"):
                agg_inf.template_extra_feed.update(
                    {
                        "deliverables_promoted": promoted,
                        "deliverables_with_conflicts": [
                            {"path": rp, "candidates": cands}
                            for rp, cands in conflicts.items()
                        ],
                        "conflicts_grouped_by_parent": conflicts_grouped,
                        "deliverables_dst": deliverables_dst,
                        "worker_summaries": worker_summaries,
                    }
                )

        return original_query or ""

    return _builder


@attrs(slots=False)
class BreakdownThenAggregateInferencer(InferencerBase, WorkGraph):
    """Diamond-shaped inferencer: breakdown → parallel workers → aggregate.

    Follows the dual inheritance pattern from DualInferencer(InferencerBase, Workflow)
    and PlanThenImplementInferencer(InferencerBase, Workflow), but uses WorkGraph
    instead of Workflow for parallel fan-out/fan-in execution.

    MRO: InferencerBase.__call__() -> infer() wins over WorkNodeBase.__call__() -> run().
    run()/arun() are blocked — callers must use infer()/ainfer().

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
          means unlimited parallelism.

    .. warning::
        ``max_concurrency`` with an ``aggregator_inferencer`` can deadlock because
        the downstream aggregator propagation acquires the same semaphore while
        start-node slots are still held. Use ``max_concurrency`` only without an
        aggregator, or leave it as ``None``.

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
    # Every subtask gets its own FRESH worker (object identity = worker ID); created workers
    # are ephemeral per-subtask locals, never stored on ``self`` (per-run state lives in the
    # RunContext node). MFI sets this internally to its flow-builder closure.
    worker_inferencers: Any = attrib(
        default=None, kw_only=True, metadata={"lazy_config_factory": True}
    )

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
    # forwards ``_last_aggregation_guidance`` (captured at breakdown-parse
    # time) to ``template_extra_feed["aggregation_guidance"]``, and returns
    # the original BTA query as the aggregator's ``inference_input`` (which
    # the wrapper template renders into ``{{ input }}``).
    #
    # When False (legacy opt-out): BTA formats worker outputs as
    # ``### Result N\n<output>`` and returns that text as the aggregator's
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

    # === Workspace support (opt-in, overrides checkpoint_dir when set) ===
    # workspace: InferencerWorkspace — inherited from InferencerBase.
    #   Configure workspace layout (e.g., use_final_deliverables_folder) on the
    #   InferencerWorkspace object directly, keeping workspace concerns out of BTA.
    # The legacy `workspace_root: Optional[str]` shorthand was removed
    # 2026-05-05; pass `workspace=InferencerWorkspace(root="/path", ...)`.

    # === Concurrency ===
    # Maximum number of worker nodes to run in parallel during the fan-out
    # layer of the diamond graph. When set, creates an asyncio.Semaphore to
    # throttle concurrent worker execution (sliding window, not batched).
    # Only applies to the async path (ainfer). None means unlimited parallelism.
    # Inherited from WorkGraph but surfaced here for discoverability.
    #
    # IMPORTANT: When an aggregator_inferencer is set, the semaphore is also
    # acquired for the downstream aggregator propagation *while* the start-node
    # semaphore slot is still held. This means the effective concurrency budget
    # must account for the aggregator slot. In practice, with N workers and
    # max_concurrency=M, the Mth worker to reach the aggregator will need an
    # (M+1)th slot. To avoid deadlock, either:
    #   - Use max_concurrency only without an aggregator, or
    #   - Set max_concurrency >= num_workers + 1 (which effectively means
    #     unlimited for the workers), or
    #   - Leave max_concurrency as None (default, unlimited).
    # A future fix could exclude the aggregator from semaphore gating.
    max_concurrency: Optional[int] = attrib(default=None)

    # === Interactive support ===
    interactive: Optional[Any] = attrib(default=None)
    enable_checkpoint_sub_query_selection: bool = attrib(default=False)
    enable_checkpoint_results_review: bool = attrib(default=False)
    breakdown_only: bool = attrib(default=False)  # Stop after breakdown phase
    disable_aggregator: bool = attrib(default=False)  # Run workers but skip aggregation
    promote_worker_deliverables: bool = attrib(default=False, kw_only=True)

    # === v1.7 Deliverable Boundary Semantics (Phase 2) ===
    # BTA is a boundary by default. The boundary mechanism only ACTIVATES
    # when use_final_deliverables_folder=True on the workspace; existing
    # callers that don't set that flag get a no-op (backward compatible).
    is_deliverable_boundary: bool = attrib(default=True, kw_only=True)
    # When True, BTA's own response (aggregator's text output) is published
    # to outputs/final_deliverables/<output_path>; when False, it stays in
    # outputs/ as a report only. Default True since BTA's response IS the
    # canonical aggregated artifact in most workflows.
    publishes_response_as_deliverable: bool = attrib(default=True, kw_only=True)
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

    # Guard: emit topology only once per _ainfer() call.
    # Reset at the top of _infer/_ainfer so reused BTA instances work correctly.
    _graph_topology_emitted: bool = attrib(default=False, init=False, repr=False)

    # Captured at breakdown-parse time from the breakdown response's
    # ``aggregation_guidance`` field. Plumbed forward to the aggregator's
    # ``template_extra_feed["aggregation_guidance"]`` so the aggregator
    # prompt can render the breakdown's reconstruction guidance via
    # ``{{ aggregation_guidance }}``. Reset to None at the top of
    # ``_parse_json_subtasks`` so stale values don't leak across calls.
    _last_aggregation_guidance: Optional[str] = attrib(
        default=None, init=False, repr=False
    )

    # Suppress WorkGraph's start_nodes requirement at construction time
    # (graph is built dynamically in _infer/_ainfer)
    start_nodes = attrib(factory=list)

    # ------------------------------------------------------------------
    # Worker naming hook (overridable by subclasses)
    # ------------------------------------------------------------------

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

        if (
            self.breakdown_inferencer is not None
            and self.max_breakdown is not None
            and hasattr(self.breakdown_inferencer, "template_extra_feed")
        ):
            self.breakdown_inferencer.template_extra_feed.setdefault(
                "max_breakdown", self.max_breakdown
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

        # --- Expansion-driven (new) implementation support ---
        # Cache for sub-queries used by subgraph_registry on resume.
        # Populated by _make_breakdown_fn before returning GraphExpansionResult.
        self._cached_sub_queries = None

        # Register subgraph factories for WorkGraph's registry-based expansion
        # reconstruction. On resume, _reconstruct_graph_expansions() looks up
        # expansion_id in subgraph_registry and calls the factory. The closure
        # captures `self` so it can access worker_inferencers, aggregator_inferencer, etc.
        self.subgraph_registry = self.subgraph_registry or {}
        self.subgraph_registry["bta_diamond"] = (
            lambda exp_id: self._build_subgraph_spec(self._cached_sub_queries)
        )
        self.subgraph_registry["bta_workers"] = (
            lambda exp_id: self._build_subgraph_spec(self._cached_sub_queries)
        )

    # === MRO safety: block run()/arun() ===

    def run(self, *args, **kwargs):
        raise NotImplementedError(
            "Use infer()/ainfer() instead of run()/arun(). "
            "run() would bypass graph setup in _infer()."
        )

    async def arun(self, *args, **kwargs):
        raise NotImplementedError(
            "Use ainfer() instead of arun(). "
            "arun() would bypass graph setup in _ainfer()."
        )

    def _format_worker_results_text(
        self,
        worker_results,
        worker_output_paths=None,
        worker_deliverable_dirs=None,
    ) -> str:
        """Format worker_results as ``### Result N\\n<output>`` lines, joined
        by blank lines.

        Used by both the legacy default ``_build_agg_input`` path (where the
        formatted text becomes the aggregator's ``inference_input`` directly)
        and the new ``inject_upstream_artifacts_to_aggregator`` path (where
        the formatted text becomes the value of
        ``template_extra_feed["upstream_artifacts"]``).

        When the aggregator has local file access AND a worker output path is
        available, the result is referenced by path rather than inlined
        (avoids OS ARG_MAX limits when piping large outputs to subprocess).

        When ``worker_deliverable_dirs`` is provided and a worker has a
        non-empty deliverables folder, both the folder path and the report
        file are referenced — giving the aggregator a directory to explore
        AND a summary to read.
        """
        agg_has_local = self.aggregator_inferencer is not None and getattr(
            self.aggregator_inferencer, "has_local_access", False
        )
        paths = list(worker_output_paths or [])
        fd_dirs = list(worker_deliverable_dirs or [])
        self.log_info(
            {
                "bta_name": getattr(self, "name", None),
                "bta_type": type(self).__name__,
                "agg_has_local": agg_has_local,
                "agg_type": type(self.aggregator_inferencer).__name__
                if self.aggregator_inferencer
                else None,
                "paths": [str(p) if p else None for p in paths],
                "deliverable_dirs": [str(d) if d else None for d in fd_dirs],
                "num_results": len(worker_results) if worker_results else 0,
            },
            log_type="AggFormatDecision",
        )
        parts = []
        for idx, res in enumerate(worker_results):
            path = paths[idx] if idx < len(paths) else None
            fd_dir = fd_dirs[idx] if idx < len(fd_dirs) else None
            if agg_has_local and fd_dir:
                lines = [f"### Upstream Outcome {idx + 1}"]
                label = (
                    "See deliverables"
                    if "final_deliverables" in str(fd_dir)
                    else "See outputs folder"
                )
                lines.append(f"({label}: `{fd_dir}`)")
                if path:
                    lines.append(f"(See file: `{path}`)")
                parts.append("\n".join(lines))
            elif agg_has_local and path:
                parts.append(f"### Upstream Outcome {idx + 1}\n(See file: `{path}`)")
            else:
                # Non-local aggregator: inline the FULL file content (not just
                # the <Response> summary) so the aggregator sees the complete
                # upstream artifact. Falls back to summary if file read fails.
                content = str(res)
                if path and os.path.isfile(path):
                    try:
                        content = open(path, encoding="utf-8").read()
                    except (OSError, UnicodeDecodeError):
                        pass
                parts.append(f"### Upstream Outcome {idx + 1}\n{content}")
        return "\n\n".join(parts)

    def _build_synthetic_aggregation(self, worker_results, original_query) -> str:
        """Produce a synthetic aggregation when the LLM aggregator fails.

        Lists the upstream worker outputs so downstream review/fix can still
        consume them. Written to the aggregator's output.md so the pipeline
        continues rather than crashing.
        """
        parts = [
            "# Synthetic Aggregation (automatic fallback)\n",
            "**Warning:** The LLM aggregator failed to produce a valid consolidated "
            "output after exhausting all retries. This synthetic aggregation lists "
            "the upstream worker outputs for manual review or downstream processing.\n",
            f"**Original task:** {original_query}\n",
        ]
        worker_paths = getattr(self, "_last_worker_output_paths", None) or []
        for idx, res in enumerate(worker_results):
            path = worker_paths[idx] if idx < len(worker_paths) else None
            parts.append(f"## Upstream Outcome {idx + 1}")
            if path:
                parts.append(f"(Full output at: `{path}`)\n")
            summary = str(res)[:500] if res else "(empty)"
            parts.append(summary)
        return "\n\n".join(parts)

    def _inject_aggregator_extra_feed(
        self,
        worker_results,
        worker_output_paths=None,
        worker_deliverable_dirs=None,
    ) -> None:
        """Push formatted upstream artifacts (and breakdown-captured
        aggregation_guidance, if any) into the aggregator inferencer's
        ``template_extra_feed``. Used when
        ``inject_upstream_artifacts_to_aggregator=True`` AND no custom
        ``aggregator_prompt_builder`` is set.

        Idempotent within a single BTA call: ``template_extra_feed`` is a
        plain dict mutated in place. The ``aggregation_guidance`` key is
        DROPPED when the breakdown didn't produce one this call (avoids stale
        guidance from a previous call leaking into the current prompt).
        """
        if self.aggregator_inferencer is None:
            return
        target = self.aggregator_inferencer
        if not hasattr(target, "template_extra_feed"):
            return
        self.log_info(
            {
                "bta_name": getattr(self, "name", None),
                "bta_type": type(self).__name__,
                "agg_type": type(target).__name__,
                "num_results": len(worker_results) if worker_results else 0,
                "paths": [str(p) if p else None for p in (worker_output_paths or [])],
            },
            log_type="AggInjectFeed",
        )
        if target.template_extra_feed is None:
            target.template_extra_feed = {}

        upstream_text = self._format_worker_results_text(
            worker_results,
            worker_output_paths,
            worker_deliverable_dirs=worker_deliverable_dirs,
        )
        target.template_extra_feed["upstream_artifacts"] = upstream_text
        # NOTE: Per-worker output paths are embedded inline within
        # ``upstream_text`` via ``_format_worker_results_text`` (see
        # ``(See file: <path>)`` markers). No structured ``worker_output_paths``
        # variable is injected because no aggregator template currently
        # consumes it; speculative injection would be infrastructure with
        # no consumer. If a future template needs the structured list,
        # add the injection at that time.

        guidance = getattr(self, "_last_aggregation_guidance", None)
        if guidance:
            target.template_extra_feed["aggregation_guidance"] = guidance
        else:
            target.template_extra_feed.pop("aggregation_guidance", None)

    def _parse_json_subtasks(self, raw_output: str) -> List:
        """Parse JSON subtask format from the task_breakdown template.

        Extracts subtasks from ``<Response>`` tags or raw text, parses JSON
        with a ``subtasks`` array, and builds structured sub_queries for BTA.
        Falls back to ``parse_numbered_list`` if JSON extraction fails.

        Side effect: also captures the breakdown's ``aggregation_guidance``
        field (when present) into ``self._last_aggregation_guidance`` so the
        downstream aggregator prompt builder can plumb it forward into the
        aggregator's ``template_extra_feed["aggregation_guidance"]``. Reset
        to ``None`` at the top so stale values from previous calls don't leak.

        This is the built-in parser for ``breakdown_format="json_subtasks"``,
        consolidating the parsing logic previously duplicated across tools.
        """
        from agent_foundation.common.response_parsers import extract_delimited

        # Reset breakdown-derived aggregation guidance for this call. Set
        # below if the parsed JSON includes the field; remains None if the
        # JSON is malformed or the field is absent.
        self._last_aggregation_guidance = None

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
                return parse_numbered_list(response_text)
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
                return parse_numbered_list(response_text)

        subtasks = data.get("subtasks") or data.get("decomposed_subtasks") or []
        if not subtasks:
            return parse_numbered_list(response_text)

        # Capture aggregation_guidance for the downstream aggregator. Tolerate
        # missing/empty values — the aggregator prompt's ``{% if %}`` branch
        # gates the whole section.
        guidance = data.get("aggregation_guidance")
        if isinstance(guidance, str) and guidance.strip():
            self._last_aggregation_guidance = guidance.strip()

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

        if not queries:
            return parse_numbered_list(response_text)

        _logger.info("Parsed %d subtasks from JSON breakdown", len(queries))
        return queries

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

        NOTE: captures self (not workspace_root) — for inner BTAs created by
        _ImportFactory, workspace_root is None at construction; _workspace is
        assigned later at runtime via worker._workspace = worker_ws.
        """
        # Part F: resolve the shared Tier-2 sink (seeded from the instance) — this
        # runs inside ``_ainfer`` under the active ctx; with no ctx it is the
        # instance attrib (byte-identical).
        reporter = self._resolve_graph_reporter()
        _bta_self = self

        async def _async_status_cb(event):
            output_path = ""
            _ws_obj = getattr(_bta_self, "_workspace", None)
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
                        _ws / "checkpoints" / "breakdown_result.json",
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

    async def _emit_pending_graph_topology(self) -> None:
        """Emit the pending graph topology to the frontend.

        Called from _ainfer() — either early (from _breakdown_fn after expansion
        spec is built) or as a fallback after _arun() completes.

        The status callback is set separately in _ainfer() via
        _make_graph_status_callback() + set_graph_event_callback() BEFORE _arun(),
        so it propagates to expansion nodes automatically.
        """
        pending_topo = getattr(self, "_pending_topology", None)
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
        self._pending_topology = None  # clear before await (re-entrant safety)
        try:
            await _reporter.on_graph_topology(pending_topo)

            # Breakdown is a VIRTUAL node — manually prepended to the topology, NOT a real
            # WorkGraphNode. So the graph event callback never fires for it.
            # Emit an explicit completion event with the resolved output_path so
            # the UI can fetch breakdown_output.md when the breakdown node is clicked.
            _ws_obj_b = getattr(self, "_workspace", None)
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
                    _ws_b / "checkpoints" / "breakdown_result.json",
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

    def _get_result_path(self, result_id, *args, **kwargs):
        """Provide result path for WorkGraph-level result saving."""
        if self._workspace is not None:
            ext = ".json" if self.checkpoint_mode == "jsonfy" else ".pkl"
            return self._workspace.checkpoint_path(f"{result_id}_result{ext}")
        if self.checkpoint_dir:
            ext = ".json" if self.checkpoint_mode == "jsonfy" else ".pkl"
            return os.path.join(self.checkpoint_dir, f"{result_id}_result{ext}")
        raise NotImplementedError(
            "checkpoint_dir or workspace must be set for result saving"
        )

    def _get_effective_predefined_sub_queries(self):
        """M5: read-through accessor for the runtime sub-queries.

        Prefers a value stored in the active ``ctx.node.call`` (runtime-propagated
        per-call input — written without mutating the definition) and falls back
        to the instance field ``self.predefined_sub_queries``. **Byte-identical**
        when no context override is present (returns the instance field exactly).
        """
        from agent_foundation.common.inferencers.run_context import active_run_context

        ctx = active_run_context()
        if ctx is not None:
            node = ctx.node(creator=(type(self).__qualname__, ctx.path))
            call = node.call
            override = None
            if isinstance(call, dict):
                override = call.get("predefined_sub_queries")
            else:
                override = getattr(call, "effective_sub_queries", None)
            if override is not None:
                return override
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

    def _load_breakdown_checkpoint(self):
        """Load saved breakdown result if resuming and checkpoint exists."""
        if not self.resume_with_saved_results:
            return None
        if self._workspace is not None:
            ckpt = self._workspace.checkpoint_path("breakdown_result.json")
        elif self.checkpoint_dir:
            ckpt = os.path.join(self.checkpoint_dir, "breakdown_result.json")
        else:
            return None
        if not os.path.exists(ckpt):
            return None
        try:
            with open(ckpt) as f:
                saved = json.load(f)
            sub_queries = saved.get("sub_queries", [])
            if sub_queries:
                _logger.info(
                    "Resuming from saved breakdown checkpoint (%d sub_queries)",
                    len(sub_queries),
                )
                return sub_queries
        except (json.JSONDecodeError, KeyError, OSError) as e:
            _logger.warning("Failed to load breakdown checkpoint: %s", e)
        return None

    def _save_breakdown_checkpoint(self, raw_output, sub_queries):
        """Save breakdown result and parsed sub_queries to checkpoint."""
        if self._workspace is not None:
            ckpt = self._workspace.checkpoint_path("breakdown_result.json")
        elif self.checkpoint_dir:
            ckpt = os.path.join(self.checkpoint_dir, "breakdown_result.json")
        else:
            return
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        try:
            # Explicit utf-8 + ensure_ascii=False: raw_output may contain
            # LLM-generated Unicode (arrows, em-dashes). Default encoding
            # is cp1252 on Windows which would raise UnicodeEncodeError.
            with open(ckpt, "w", encoding="utf-8") as f:
                json.dump(
                    {"raw_output": str(raw_output), "sub_queries": sub_queries},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            _logger.info(
                "Saved breakdown checkpoint with %d sub_queries", len(sub_queries)
            )
        except (OSError, UnicodeEncodeError) as e:
            _logger.warning("Failed to save breakdown checkpoint: %s", e)

    async def _emit_graph_reconcile(self):
        """Emit final node statuses so the frontend can correct any stale UI state."""
        _reporter = self._resolve_graph_reporter()  # Part F: shared Tier-2 sink
        if _reporter is None:
            return
        try:
            statuses = {n.name: "completed" for n in self._all_nodes()}
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
        are the deliverables (organized under ``workers/``).
        """
        agg_inf = self.aggregator_inferencer
        # M7: resolve the aggregator's PUBLISHED workspace from its child ctx node
        # (override-aware), not the bare property which under the parent ctx misses
        # the child's mailbox and returns None/the BTA's own ws — orphaning the plan.
        agg_ws = self._read_child_workspace(agg_inf, "aggregator") if agg_inf else None

        if agg_ws is not None:
            self._symlink_child_output(
                agg_ws, child_output_name=getattr(agg_inf, "output_path", None)
            )
            resolved = self.resolve_output_path()
            if resolved and os.path.isfile(resolved):
                self._emit_output_manifest(resolved)
            self._try_extract_proposal_index(response)
            return response
        else:
            # No aggregator: workers' outputs ARE the deliverables
            if self._workspace and self._workspace.deliverables_dir:
                workers_dir = os.path.join(self._workspace.deliverables_dir, "workers")
                for i, worker in enumerate(getattr(self, "_worker_instances", [])):
                    # M7: read the worker's canonical workspace from the orchestrator's
                    # own layout (self._workspace.child(<dir>)), not the bare property
                    # which under the parent ctx misses the worker's published mailbox.
                    worker_ws = (
                        self._workspace.child(self._worker_child_name(i))
                        if self._workspace is not None
                        else None
                    )
                    if worker_ws and worker_ws.has_deliverables:
                        self._symlink_or_copy(
                            worker_ws.deliverables_dir,
                            os.path.join(workers_dir, self._worker_child_name(i)),
                        )
            # Fall through to base for outputs/output.md summary write
            return super()._finalize_output(response)

    def _try_extract_proposal_index(self, response) -> None:
        """Extract ``proposal_index`` JSON fence from aggregator output and write sidecar.

        Runs unconditionally after every aggregation. Silently skips when no
        fence is present (expected for non-research aggregators). Cost: one
        regex search — microseconds.
        """
        if self._workspace is None:
            return
        text = str(response) if response is not None else ""
        if "proposal_index" not in text:
            # (a) Text-source fallback for truncated responses. The response
            # object surfaced to this inferencer can be truncated at the model's
            # token limit while the full ``proposal_index`` fence is present in
            # the aggregator's output file on disk. Re-read from that
            # source-of-truth file and re-attempt extraction. The fast path
            # (fence already in ``text``) is unchanged.
            fallback = self._read_aggregator_output_text()
            if fallback:
                text = fallback
        if not text or "proposal_index" not in text:
            return
        try:
            from agent_foundation.common.data_models.proposal.model import ProposalIndex
            from agent_foundation.common.data_models.proposal.parser import (
                make_empty_index,
                parse_proposal_index_from_text,
                write_proposal_index,
            )

            idx = parse_proposal_index_from_text(text)
            if idx is not None:
                from datetime import datetime, timezone

                idx.created_at = datetime.now(timezone.utc).isoformat()
                idx.source_workspace = str(self._workspace.root)
                from pathlib import Path as _PP

                # INVARIANT (depended on by model_optimization/SOP.md Phase 3b
                # and Phase 4): proposals.json lives at
                #   <research-propose workspace>/outputs/proposals.json
                # BTA._finalize_output early-returns when an aggregator is
                # present, so the base class's outputs->final_deliverables move
                # never runs for this file; this location is therefore stable
                # and discoverable by convention. Any future refactor that
                # relocates this file MUST also update:
                #   - resources/sops/model_optimization/SOP.md Phase 3b/4
                #   - the Phase 4 ``task --use-proposal`` path
                #   - test_breakdown_then_aggregate's proposals.json assertions
                sidecar = _PP(self._workspace.root) / "outputs" / "proposals.json"
                write_proposal_index(sidecar, idx)
                _logger.info(
                    "Wrote proposal index (%d proposals) to %s",
                    idx.total_count,
                    sidecar,
                )
            # else: no fence found — silently skip
        except Exception as exc:
            _logger.warning("Proposal index extraction failed (non-fatal): %s", exc)

    def _read_aggregator_output_text(self) -> "str | None":
        """Read the aggregator's output file as the proposal_index source.

        Returns the file contents when the aggregator inferencer exposes a
        workspace and its output file exists on disk, else ``None``. Used as a
        fallback when the in-memory response is truncated past the
        ``proposal_index`` fence. All I/O is guarded; failure is non-fatal.
        """
        agg_inf = self.aggregator_inferencer
        if agg_inf is None:
            return None
        # M7: override-aware read (see _finalize_output) — the truncated-response
        # proposal-index fallback must also resolve the published aggregator workspace.
        agg_ws = self._read_child_workspace(agg_inf, "aggregator")
        if agg_ws is None:
            return None
        out_name = getattr(agg_inf, "output_path", None) or "output.md"
        candidates: list[str] = []
        for getter_name in ("output_path", "deliverable_path"):
            getter = getattr(agg_ws, getter_name, None)
            if callable(getter):
                try:
                    path = getter(out_name)
                except Exception:  # noqa: BLE001 — path resolution is best-effort
                    path = None
                if path:
                    candidates.append(path)
        for path in candidates:
            try:
                if os.path.isfile(path):
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        return fh.read()
            except (OSError, UnicodeError) as exc:
                _logger.warning(
                    "BTA proposal_index file-fallback read failed for %s: %s",
                    path,
                    exc,
                )
        return None

    def _finalize_response(self, result):
        """BTA audit bookkeeping (surfacing moved to _finalize_output).

        Only the aggregation report fallback remains: when no deliverables
        exist, write the aggregator's text response as a report.
        """
        if self._workspace is None or not self.output_path:
            return

        # Check if aggregator produced deliverables (written by leaf's
        # _finalize_output which runs before this via __ainfer_single_impl)
        agg_inf = self.aggregator_inferencer
        # M7: resolve the aggregator's PUBLISHED workspace from its child ctx node
        # (override-aware), not the bare property which under the parent ctx misses
        # the child's mailbox and returns None/the BTA's own ws — orphaning the plan.
        agg_ws = self._read_child_workspace(agg_inf, "aggregator") if agg_inf else None
        has_deliverables = agg_ws is not None and getattr(
            agg_ws, "has_deliverables", False
        )

        if has_deliverables:
            self._deliverables_copied = True
            _logger.info(
                "Skipping pipeline report — aggregator deliverables handled by _finalize_output",
            )
        else:
            report_dst = self._workspace.output_path(self.output_path)
            os.makedirs(os.path.dirname(report_dst), exist_ok=True)
            try:
                if isinstance(result, tuple):
                    result = result[-1] if result else None
                if result is None:
                    text = ""
                elif hasattr(result, "output"):
                    text = result.output or ""
                elif isinstance(result, dict):
                    text = result.get("output", "")
                elif hasattr(result, "text"):
                    text = result.text or ""
                else:
                    text = str(result)
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
        if self.breakdown_inferencer is not None:
            bd_ws = workspace.child("breakdown")
            bd_ws.ensure_dirs()
            self.breakdown_inferencer._workspace = bd_ws

    def _iter_child_inferencers(self):
        """The aggregator inferencer.

        Workers are factory-created per-run in generic BTA (via
        ``worker_inferencers``), so this base implementation does not yield
        them — subclasses with declarative worker references (e.g.,
        :class:`MultiFlowInferencer`) extend this to include them.
        """
        if self.aggregator_inferencer is not None:
            yield self.aggregator_inferencer

    def _iter_child_slots(self):
        """§9.3/N-Major1: semantic slots for the static children (breakdown,
        aggregator) matching the ``ctx.child(slot)`` used in ``_ainfer``. Workers
        are dynamic (per-run via ``worker_inferencers`` — callable/dict/list) and are
        bound to ``ctx.child(worker_node_name)`` at spawn time, not here."""
        seen_ids = set()
        for slot, inf in (
            ("breakdown", self.breakdown_inferencer),
            ("aggregator", self.aggregator_inferencer),
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

    @staticmethod
    def _unwrap_workgraph_result(result):
        """Unwrap expansion-driven WorkGraph results.

        WorkGraph returns a tuple of start-node results. In the expansion-driven
        implementation, the breakdown node is the sole start node, so the outer
        tuple is always length-1. The inner value may be a tuple of worker results
        with None entries. Unwrap both levels and filter out Nones.
        """
        if isinstance(result, tuple) and len(result) == 1:
            result = result[0]
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
        """Expansion-driven sync inference: single breakdown node → GraphExpansionResult → diamond."""
        # Reset per-call state to prevent cross-run leakage on reused instances
        self._cached_sub_queries = None
        # Bootstrap _cached_sub_queries on resume
        if self._cached_sub_queries is None:
            saved = self._load_breakdown_checkpoint()
            if saved is not None:
                if self.max_breakdown is not None and len(saved) > self.max_breakdown:
                    saved = saved[: self.max_breakdown]
                self._cached_sub_queries = saved

        # Reset topology emission guard
        self._graph_topology_emitted = False
        self._clear_all_node_queues()

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
        _bd_ckpt = None
        if self._workspace is not None:
            _bd_ckpt = self._workspace.checkpoint_path("breakdown")
        elif self.checkpoint_dir:
            _bd_ckpt = os.path.join(self.checkpoint_dir, "breakdown")
        if _bd_ckpt:
            breakdown_node._get_result_path = (
                lambda rid, *a, _d=_bd_ckpt, _e=_ext, **kw: os.path.join(
                    _d, f"{rid}_result{_e}"
                )
            )
        self.start_nodes = [breakdown_node]

        # Configure expansion on WorkGraph
        self.max_expansion_depth = 1
        self.max_total_nodes = max(self.max_breakdown or 100, 100) + 2

        # Propagate expansion settings to the breakdown node.
        self._propagate_expansion_settings()

        # Run the graph — expansion handles the rest
        result = WorkGraph._run(self, inference_input, **kwargs)

        result = self._unwrap_workgraph_result(result)

        self._finalize_response(result)
        return result

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        """Expansion-driven async inference: single breakdown node → GraphExpansionResult → diamond."""
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

        # Reset per-call state to prevent cross-run leakage on reused instances
        self._cached_sub_queries = None
        # Bootstrap _cached_sub_queries on resume: load breakdown_result.json
        # BEFORE WorkGraph._arun() because _reconstruct_graph_expansions() runs
        # BEFORE start_nodes execute, and the subgraph_registry lambda calls
        # self._build_subgraph_spec(self._cached_sub_queries).
        if self._cached_sub_queries is None:
            saved = self._load_breakdown_checkpoint()
            if saved is not None:
                if self.max_breakdown is not None and len(saved) > self.max_breakdown:
                    saved = saved[: self.max_breakdown]
                self._cached_sub_queries = saved

        # Reset topology emission guard
        self._graph_topology_emitted = False
        self._clear_all_node_queues()

        # Force async mode for worker fns
        old_use_async = getattr(self, "use_async", False)
        self.use_async = True

        try:
            # Create the breakdown node as the sole start node
            breakdown_node = WorkGraphNode(
                name="breakdown",
                value=self._make_breakdown_fn(
                    inference_input, inference_config, **kwargs
                ),
                result_pass_down_mode=ResultPassDownMode.NoPassDown,
                enable_result_save=self.enable_result_save,
                resume_with_saved_results=self.resume_with_saved_results,
                retry_on_exceptions=TRANSIENT_RETRY_EXCEPTIONS,
            )
            # Assign _get_result_path so expansion infrastructure can persist records
            _ext = ".json" if self.checkpoint_mode == "jsonfy" else ".pkl"
            _bd_ckpt = None
            if self._workspace is not None:
                _bd_ckpt = self._workspace.checkpoint_path("breakdown")
            elif self.checkpoint_dir:
                _bd_ckpt = os.path.join(self.checkpoint_dir, "breakdown")
            if _bd_ckpt:
                breakdown_node._get_result_path = (
                    lambda rid, *a, _d=_bd_ckpt, _e=_ext, **kw: os.path.join(
                        _d, f"{rid}_result{_e}"
                    )
                )
            self.start_nodes = [breakdown_node]

            # Configure expansion on WorkGraph
            self.max_expansion_depth = 1
            self.max_total_nodes = max(self.max_breakdown or 100, 100) + 2

            # Propagate expansion settings to the breakdown node (and any future nodes).
            # __attrs_post_init__ already called this, but with the default max_expansion_depth=0.
            # We must re-propagate now that we've set max_expansion_depth=1 and new start_nodes.
            self._propagate_expansion_settings()

            # Wire graph event callback BEFORE _arun() so it propagates to
            # expansion nodes via _propagate_settings_to_subgraph() (workgraph.py:670).
            # When breakdown returns GraphExpansionResult, _handle_graph_expansion
            # copies _graph_event_callback from the breakdown node to all worker
            # nodes — so they emit Running/Completed status events in real-time.
            if _reporter is not None:  # Part F: shared Tier-2 sink (resolved above)
                self.set_graph_event_callback(self._make_graph_status_callback())

            # Run the graph — expansion handles the rest
            result = await WorkGraph._arun(self, inference_input, **kwargs)
        finally:
            self.use_async = old_use_async

        result = self._unwrap_workgraph_result(result)

        # Emit full topology after expansion attaches subgraph
        if _reporter is not None and not self._graph_topology_emitted:  # Part F sink
            try:
                from agent_foundation.common.inferencers.graph_events import (
                    GraphTopologyEvent,
                    NodeStatus,
                )

                worker_agg_topology = GraphTopologyEvent.from_work_graph(self)
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
                self._pending_topology = worker_agg_topology
                await self._emit_pending_graph_topology()
                self._graph_topology_emitted = True
            except Exception as _e:
                _logger.warning("[BTA] full topology emit failed: %s", _e)

        # Interactive results review checkpoint
        if self.enable_checkpoint_results_review and self.interactive:
            from agent_foundation.ui.interactive_checkpoint import (
                checkpoint_results_review,
            )

            result_str = str(result)[:2000]
            cp_result = await checkpoint_results_review(
                self.interactive, result_str, default_action="approve"
            )
            if cp_result.action == "rerun":
                return await self._ainfer(inference_input, inference_config, **kwargs)

        await self._emit_graph_reconcile()
        self._finalize_response(result)
        return result

    def _build_subgraph_spec(self, sub_queries, inference_config=None, **kwargs):
        """Build a SubgraphSpec from parsed sub-queries.

        Refactored from _build_diamond_graph(). Returns a SubgraphSpec
        instead of setting self.start_nodes directly.

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

        worker_nodes = []
        worker_output_paths = []
        _worker_instances = []  # Fix #5: collect for isolation check
        _bta_prefix = f"{self.name}." if getattr(self, "name", None) else ""
        use_async = getattr(self, "use_async", False)

        for i, sq in enumerate(expanded_queries):
            if isinstance(sq, dict):
                query_str = sq.get("query", str(sq))
                sq_args = sq.get("args", {})
            else:
                query_str = sq
                sq_args = {}

            # Resolve a FRESH worker for this subtask from the single
            # ``worker_inferencers`` source. Accepted shapes (see the field docstring):
            #   list -> static-K round-robin; dict -> heterogeneous dispatch by task_type;
            #   LazyConfigFactory/partial -> call no-args; callable -> call (sub_query, index).
            from rich_python_utils.config_utils._lazy_config_factory import (
                LazyConfigFactory,
            )

            worker = None
            task_type = None
            wi = self.worker_inferencers
            if isinstance(wi, list):
                if wi:
                    worker = wi[i % len(wi)]
            elif isinstance(wi, dict):
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
                if isinstance(factory, functools.partial) and not isinstance(
                    factory, LazyConfigFactory
                ):
                    _logger.error(
                        "BTA[%s] worker_inferencers[%s] is a functools.partial, not a "
                        "LazyConfigFactory. This causes cross-worker instance "
                        "sharing. Ensure factory recipes use LazyConfigFactory "
                        "(auto-applied by the config walker for _target_: entries).",
                        getattr(self, "name", "?"),
                        task_type or "_default",
                    )
                if isinstance(factory, (functools.partial, LazyConfigFactory)):
                    worker = factory()
                else:
                    worker = factory(sub_query=query_str, index=i)
            elif wi is not None:
                if isinstance(wi, functools.partial) and not isinstance(
                    wi, LazyConfigFactory
                ):
                    _logger.error(
                        "BTA[%s] worker_inferencers is a functools.partial, not a "
                        "LazyConfigFactory. This causes cross-worker instance sharing.",
                        getattr(self, "name", "?"),
                    )
                if isinstance(wi, (functools.partial, LazyConfigFactory)):
                    worker = wi()
                else:
                    worker = wi(sub_query=query_str, index=i)

            # Assign child workspace to worker
            if self._workspace is not None and isinstance(worker, InferencerBase):
                prev_ws = getattr(worker, "_workspace", None)
                use_fdl = (
                    getattr(prev_ws, "use_final_deliverables_folder", False)
                    if prev_ws
                    else False
                )
                worker_ws = self._workspace.child(self._worker_child_name(i))
                if use_fdl:
                    from agent_foundation.common.inferencers.inferencer_workspace import (
                        InferencerWorkspace,
                    )

                    worker_ws = InferencerWorkspace(
                        root=worker_ws.root,
                        use_final_deliverables_folder=use_fdl,
                    )
                worker_ws.ensure_dirs()
                # M7 workspace virtualization (fan-out case): publish the per-worker
                # workspace into the worker's child context so its ``_workspace``
                # getter resolves it from the context (§2.12 option-b). The slot
                # MUST match the worker's threaded run_context (``_node_name`` =
                # ``{_bta_prefix}{worker_child_name}``). The instance assignment
                # stays as the byte-identical fallback.
                _worker_node_name = f"{_bta_prefix}{self._worker_child_name(i)}"
                _worker_child = self._rc_child(_worker_node_name)
                self._publish_workspace_to_ctx(_worker_child, worker_ws)
                if _worker_child is None:
                    # Legacy (no context): mutate the worker instance (byte-identical);
                    # under a context the workspace is published above -> write-pure.
                    worker._workspace = worker_ws
                self.log_info(
                    {
                        "bta_name": getattr(self, "name", None),
                        "bta_type": type(self).__name__,
                        "worker_idx": i,
                        "worker_type": type(worker).__name__,
                        "worker_child_name": self._worker_child_name(i),
                        "worker_ws_root": worker_ws.root,
                        "bta_ws_root": self._workspace.root,
                    },
                    log_type="WorkerWsAssigned",
                )
                # === v1.7 Phase 3: Wire worker as deliverable boundary ===
                # Mark each worker as a boundary so its deliverables surface
                # to this BTA's collect step. Workers may also be flow
                # inferencers (MFDual, Dual) which default to is_deliverable_boundary=False;
                # we explicitly promote them here because they're acting as
                # the per-task work-unit. This is purely additive and only
                # affects behavior when workspace.use_final_deliverables_folder=True.
                worker.is_deliverable_boundary = True
                # Output paths computed later in _build_agg_input (after
                # workers finish, so files and LWI symlinks exist).
                worker_output_paths.append(None)
            else:
                worker_output_paths.append(None)

            _worker_instances.append(worker)  # Fix #5: track for isolation check
            _node_name = f"{_bta_prefix}{self._worker_child_name(i)}"
            if isinstance(worker, BreakdownThenAggregateInferencer):
                worker.name = _node_name

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
                _reporter=None,
                _node_id=None,
                _worker_ws=None,
            ):
                def _try_load_from_output():
                    if manages_resume:
                        return None
                    output_path = (
                        w.resolve_output_path()
                        if hasattr(w, "resolve_output_path")
                        else None
                    )
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
                            cached = _try_load_from_output()
                            if cached is not None:
                                if _reporter is not None and _node_id is not None:
                                    try:
                                        await _reporter.on_node_stream(
                                            _node_id, str(cached), is_final=True
                                        )
                                    except Exception:
                                        pass
                                return cached
                            self._check_cancelled()  # §2.1/P-#6: halt at worker boundary
                            result = await w.ainfer(
                                q,
                                inference_config=inference_config,
                                run_context=self._rc_child(
                                    _node_id or "worker", workspace=_worker_ws
                                ),
                            )
                            if _reporter is not None and _node_id is not None:
                                try:
                                    await _reporter.on_node_stream(
                                        _node_id,
                                        str(result) if result else "",
                                        is_final=True,
                                    )
                                except Exception:
                                    pass
                            return result
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
                        cached = _try_load_from_output()
                        if cached is not None:
                            return cached
                        self._check_cancelled()  # §2.1/P-#6: sync worker boundary
                        if hasattr(w, "infer"):
                            return w.infer(
                                q,
                                inference_config=inference_config,
                                run_context=self._rc_child(
                                    _node_id or "worker", workspace=_worker_ws
                                ),
                            )
                        return w(q)

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
                    # (so the instance stays definition-only, safe to share). Clear
                    # the legacy ``_bta_prefix`` disambiguation-name hack only when it
                    # has no explicitly pre-wired reporter; path nesting (ctx.path)
                    # now provides node-id uniqueness.
                    if getattr(worker, "graph_reporter", None) is None:
                        worker.name = None
                else:
                    # Leaf worker: the per-token streaming niceties
                    # (interactive / stream_observer) are derived from this
                    # path-namespaced sink and tagged with the worker's flat node
                    # id. Byte-identical on the fresh-per-subtask path (distinct
                    # instances); under shared reuse these per-token handles are the
                    # one visualization-only surface still on the instance (R7).
                    if hasattr(_reporter, "node_interactive"):
                        worker.interactive = _reporter.node_interactive(_node_name)
                    if hasattr(worker, "stream_observer") and hasattr(
                        _reporter, "node_stream_observer"
                    ):
                        worker.stream_observer = _reporter.node_stream_observer(
                            _node_name
                        )

            _is_container = isinstance(worker, BreakdownThenAggregateInferencer) and (
                _reporter is not None
            )
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
                    _reporter=_w_reporter,
                    _node_id=_node_name,
                    # Root-cause fix: give the worker context its INTENDED workspace
                    # (worker_<i>) verbatim instead of path-mirroring its namespaced
                    # ctx-node name (plan_bta.worker_<i>). Descendants then root the
                    # whole subtree under worker_<i>; the node name never leaks on
                    # disk. Guarded so worker_ws is only referenced when it was set.
                    _worker_ws=(
                        worker_ws
                        if (
                            self._workspace is not None
                            and isinstance(worker, InferencerBase)
                        )
                        else None
                    ),
                ),
                result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
                group=worker_group,
                enable_result_save=StepResultSaveOptions.SkipResumable,
                resume_with_saved_results=ResumeMode.SkipResumable,
                checkpoint_mode=self.checkpoint_mode,
                retry_on_exceptions=TRANSIENT_RETRY_EXCEPTIONS,
            )
            _wname = self._worker_child_name(i)
            if self._workspace is not None:
                _w_ckpt = os.path.join(
                    str(self._workspace.root),
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
            _captured_paths = list(worker_output_paths)
            _bta_self = self

            def _build_agg_input(prompt_builder, worker_results, original_query):
                nonlocal _captured_paths
                # Resolve each worker's output path.  Two sources:
                #   Workspace: from BTA's workspace tree (deterministic)
                #   Filename:  from the worker's own output_path (each worker
                #              declares its output filename — NOT _bta_self's,
                #              which is the BTA's aggregated deliverable name)
                from agent_foundation.common.inferencers.inferencer_workspace import (
                    resolve_canonical_output_path,
                )

                _bta_ws = _bta_self._workspace
                _captured_deliverable_dirs = []
                if _bta_ws is not None and worker_results:
                    _captured_paths = []
                    _diag_workers = []
                    for idx in range(len(worker_results)):
                        child_ws = _bta_ws.child(_bta_self._worker_child_name(idx))
                        _w = _worker_instances[idx]
                        _w_filename = _w.output_path
                        p = resolve_canonical_output_path(
                            child_ws,
                            filename=_w_filename,
                            deliverables_fallback="none",
                        )
                        _captured_paths.append(p)
                        # Capture deliverables folder for two-reference format.
                        _fd = getattr(child_ws, "deliverables_dir", None)
                        if _fd and os.path.isdir(_fd) and os.listdir(_fd):
                            _captured_deliverable_dirs.append(os.path.abspath(_fd))
                        else:
                            _out_dir = getattr(child_ws, "outputs_dir", None)
                            if _out_dir and os.path.isdir(_out_dir):
                                from agent_foundation.common.workspace.layout import (
                                    FINAL_DELIVERABLES_DIR,
                                )

                                _out_entries = [
                                    e
                                    for e in os.listdir(_out_dir)
                                    if e != FINAL_DELIVERABLES_DIR
                                ]
                                if len(_out_entries) > 1:
                                    _captured_deliverable_dirs.append(
                                        os.path.abspath(_out_dir)
                                    )
                                else:
                                    _captured_deliverable_dirs.append(None)
                            else:
                                _captured_deliverable_dirs.append(None)
                        _winfo = {
                            "idx": idx,
                            "child_name": _bta_self._worker_child_name(idx),
                            "ws_root": child_ws.root,
                            "worker_output_path": _w_filename,
                            "resolved": str(p) if p else None,
                        }
                        if (
                            p is None
                            and _w_filename is not None
                            and child_ws.root is not None
                        ):
                            _out = os.path.join(child_ws.root, "outputs", _w_filename)
                            _winfo["diag_outputs_exists"] = os.path.exists(_out)
                            _winfo["diag_outputs_islink"] = os.path.islink(_out)
                            if os.path.islink(_out):
                                _winfo["diag_link_target"] = os.readlink(_out)
                                _winfo["diag_target_exists"] = os.path.exists(
                                    os.readlink(_out)
                                )
                        _diag_workers.append(_winfo)
                    _bta_self.log_info(
                        {
                            "bta_name": getattr(_bta_self, "name", None),
                            "bta_type": type(_bta_self).__name__,
                            "bta_id": getattr(_bta_self, "id", None),
                            "paths": [str(p) if p else None for p in _captured_paths],
                            "deliverable_dirs": [
                                str(d) if d else None
                                for d in _captured_deliverable_dirs
                            ],
                            "workers": _diag_workers,
                        },
                        log_type="AggInputPaths",
                    )
                # Custom prompt_builder takes precedence — fully responsible
                # for building the aggregator's input.
                if prompt_builder is not None:
                    try:
                        return prompt_builder(
                            worker_results,
                            original_query=original_query,
                            worker_output_paths=_captured_paths,
                            bta=_bta_self,
                        )
                    except TypeError:
                        try:
                            return prompt_builder(
                                worker_results,
                                original_query=original_query,
                                worker_output_paths=_captured_paths,
                            )
                        except TypeError:
                            return prompt_builder(
                                worker_results, original_query=original_query
                            )

                # Modern path: opt-in via inject_upstream_artifacts_to_aggregator.
                # Worker outputs go into template_extra_feed["upstream_artifacts"];
                # the aggregator's inference_input becomes the original BTA query
                # (which the wrapper's {{ input }} slot renders correctly under
                # "Original User Request"). aggregation_guidance from the
                # breakdown is also forwarded (see _inject_aggregator_extra_feed).
                if _bta_self.inject_upstream_artifacts_to_aggregator:
                    _bta_self._inject_aggregator_extra_feed(
                        worker_results,
                        _captured_paths,
                        worker_deliverable_dirs=_captured_deliverable_dirs,
                    )
                    return original_query or ""

                # Legacy default: format worker outputs as agg input directly.
                # The wrapper's {{ input }} slot ends up containing the worker
                # outputs (and {{ upstream_artifacts }} is undefined). Older
                # aggregator wrappers expect this shape.
                return _bta_self._format_worker_results_text(
                    worker_results,
                    _captured_paths,
                )

            def _make_agg_fn(
                agg_inf,
                prompt_builder,
                original_query,
                is_async,
                _reporter=None,
                _inference_args=None,
            ):
                _agg_extra = _inference_args or {}
                if is_async and hasattr(agg_inf, "ainfer"):

                    async def async_agg_fn(*worker_results, **_kwargs):
                        # Fix #1: verify aggregator workspace points to canonical
                        # aggregator/ slot before invocation (guards against drift
                        # from prior calls on reused instances).
                        if _bta_self._workspace is not None:
                            expected = _bta_self._workspace.child("aggregator").root
                            current = getattr(
                                getattr(agg_inf, "_workspace", None), "root", None
                            )
                            if current != expected:
                                from agent_foundation.common.inferencers.inferencer_workspace import (
                                    InferencerWorkspace,
                                )

                                agg_ws = _bta_self._workspace.child("aggregator")
                                agg_ws.ensure_dirs()
                                # M7 write-purity: only re-assign the instance under
                                # legacy (no ctx); the aggregator resolves its
                                # workspace from the context otherwise.
                                _agg_child = _bta_self._rc_child("aggregator")
                                _bta_self._publish_workspace_to_ctx(_agg_child, agg_ws)
                                if _agg_child is None:
                                    agg_inf._workspace = agg_ws
                        agg_input = _build_agg_input(
                            prompt_builder, worker_results, original_query
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
                                    "num_workers": len(worker_results),
                                },
                                "AggregatorFallback",
                            )
                            result = _bta_self._build_synthetic_aggregation(
                                worker_results, original_query
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
                        # Fix #1: verify aggregator workspace (sync path)
                        if _bta_self._workspace is not None:
                            expected = _bta_self._workspace.child("aggregator").root
                            current = getattr(
                                getattr(agg_inf, "_workspace", None), "root", None
                            )
                            if current != expected:
                                from agent_foundation.common.inferencers.inferencer_workspace import (
                                    InferencerWorkspace,
                                )

                                agg_ws = _bta_self._workspace.child("aggregator")
                                agg_ws.ensure_dirs()
                                # M7 write-purity: only re-assign the instance under
                                # legacy (no ctx); the aggregator resolves its
                                # workspace from the context otherwise.
                                _agg_child = _bta_self._rc_child("aggregator")
                                _bta_self._publish_workspace_to_ctx(_agg_child, agg_ws)
                                if _agg_child is None:
                                    agg_inf._workspace = agg_ws
                        agg_input = _build_agg_input(
                            prompt_builder, worker_results, original_query
                        )
                        if hasattr(agg_inf, "infer"):
                            return agg_inf.infer(
                                agg_input,
                                inference_config=inference_config,
                                run_context=self._rc_child("aggregator"),
                            )
                        return agg_inf(agg_input)

                    return agg_fn

            original_query = kwargs.get("_original_query", "")

            agg_inf = self.aggregator_inferencer
            if callable(agg_inf) and not isinstance(agg_inf, InferencerBase):
                agg_inf = agg_inf()
                self.aggregator_inferencer = agg_inf
            if self._workspace is not None and isinstance(agg_inf, InferencerBase):
                agg_ws = self._workspace.child("aggregator")
                agg_ws.ensure_dirs()
                # M7 workspace write-purity: publish to the aggregator's child ctx;
                # only mutate the instance under legacy (no context). Byte-identical.
                _agg_child = self._rc_child("aggregator")
                self._publish_workspace_to_ctx(_agg_child, agg_ws)
                if _agg_child is None:
                    agg_inf._workspace = agg_ws

            _agg_node_name = f"{_bta_prefix}aggregator" if _bta_prefix else "aggregator"
            # Part F: the aggregator's per-token observer + output emit go through
            # THIS orchestrator's path-namespaced sink, tagged with the flat
            # aggregator node id (``_bta_prefix`` is empty under a ctx — node
            # identity comes from the ctx path). Byte-identical with no ctx.
            _agg_reporter = self._resolve_graph_reporter()
            if (
                _agg_reporter is not None
                and hasattr(agg_inf, "stream_observer")
                and (hasattr(_agg_reporter, "node_stream_observer"))
            ):
                agg_inf.stream_observer = _agg_reporter.node_stream_observer(
                    _agg_node_name
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
                ),
                result_pass_down_mode=ResultPassDownMode.NoPassDown,
                enable_result_save=self.enable_result_save,
                resume_with_saved_results=self.resume_with_saved_results,
                checkpoint_mode=self.checkpoint_mode,
                retry_on_exceptions=TRANSIENT_RETRY_EXCEPTIONS,
            )
            _ext = ".json" if self.checkpoint_mode == "jsonfy" else ".pkl"
            _agg_ckpt = None
            if self._workspace is not None:
                _agg_ckpt = self._workspace.checkpoint_path("aggregator_result")
            elif self.checkpoint_dir:
                _agg_ckpt = os.path.join(self.checkpoint_dir, "aggregator_result")
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

        # Fix #5: check for shared sub-inferencer instances across workers
        self._validate_worker_isolation(_worker_instances)

        return SubgraphSpec(
            nodes=all_nodes,
            entry_nodes=list(worker_nodes),
        )

    def _make_breakdown_fn(
        self, inference_input, inference_config=None, **_inference_args
    ):
        """Create the breakdown node callable that returns GraphExpansionResult.

        Handles predefined sub-queries, breakdown_only, interactive selection,
        legacy checkpoint loading. Saves self._cached_sub_queries = sub_queries
        BEFORE returning GraphExpansionResult.
        """
        from rich_python_utils.common_objects.workflow.common.expansion import (
            GraphExpansionResult,
        )

        _bta = self
        _inf_input = inference_input
        _inf_config = inference_config
        _extra_args = _inference_args
        use_async = getattr(self, "use_async", False)

        if use_async:

            async def _breakdown_fn(*args, **kwargs):
                # Step 0: Check for saved breakdown checkpoint (legacy compat)
                sub_queries = _bta._load_breakdown_checkpoint()
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
                    _bd_reporter = _bta._resolve_graph_reporter()
                    if (
                        _bd_reporter is not None
                        and hasattr(_bta.breakdown_inferencer, "stream_observer")
                        and (hasattr(_bd_reporter, "node_stream_observer"))
                    ):
                        _bta.breakdown_inferencer.stream_observer = (
                            _bd_reporter.node_stream_observer("breakdown")
                        )

                    if hasattr(_bta.breakdown_inferencer, "ainfer"):
                        raw_output = await _bta.breakdown_inferencer.ainfer(
                            _inf_input,
                            inference_config=_inf_config,
                            run_context=_bta._rc_child("breakdown"),
                            **_inference_args,
                        )
                    else:
                        raw_output = _bta.breakdown_inferencer.infer(
                            _inf_input,
                            inference_config=_inf_config,
                            run_context=_bta._rc_child("breakdown"),
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
                        sub_queries = _bta._parse_json_subtasks(raw_output)
                    elif _bta.breakdown_format == "numbered_list":
                        sub_queries = parse_numbered_list(str(raw_output))
                    elif isinstance(raw_output, list):
                        sub_queries = raw_output
                    else:
                        sub_queries = parse_numbered_list(str(raw_output))

                    # Save breakdown checkpoint
                    _bta._save_breakdown_checkpoint(raw_output, sub_queries)

                # Apply max_breakdown cap
                if (
                    _bta.max_breakdown is not None
                    and len(sub_queries) > _bta.max_breakdown
                ):
                    sub_queries = sub_queries[: _bta.max_breakdown]

                if not sub_queries:
                    return raw_output if raw_output is not None else ""

                # Breakdown-only mode (skip when predefined_sub_queries — already warned)
                if _bta.breakdown_only and not _from_predefined:
                    return raw_output if raw_output is not None else sub_queries

                # Interactive sub-query selection
                if _bta.enable_checkpoint_sub_query_selection and _bta.interactive:
                    from agent_foundation.ui.interactive_checkpoint import (
                        checkpoint_breakdown_review,
                    )

                    cp_result = await checkpoint_breakdown_review(
                        _bta.interactive, sub_queries, default_action="approve"
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

                # Cache sub_queries BEFORE returning GraphExpansionResult
                _bta._cached_sub_queries = sub_queries

                # On resume, _reconstruct_graph_expansions already attached workers
                # to the breakdown node. WorkGraphNode._run resets _expansion_applied
                # to False, so we can't rely on that flag. Instead, check if the
                # breakdown node already has downstream nodes (workers) attached.
                _bd_node = _bta.start_nodes[0] if _bta.start_nodes else None
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

                # Emit full topology IMMEDIATELY so the UI shows worker nodes
                # before they start running. Without this, the UI shows only
                # "Breakdown: Running" until _arun() returns (after all workers
                # and aggregator finish), which can be minutes.
                if (
                    _bta._resolve_graph_reporter() is not None
                    and not _bta._graph_topology_emitted
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
                        _bta._pending_topology = topo
                        await _bta._emit_pending_graph_topology()
                        _bta._graph_topology_emitted = True
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
                    seed=sub_queries,
                    reconstruct_from_seed=None,
                    attach_mode="insert",
                )

            return _breakdown_fn
        else:

            def _breakdown_fn_sync(*args, **kwargs):
                # Step 0: Check for saved breakdown checkpoint (legacy compat)
                sub_queries = _bta._load_breakdown_checkpoint()
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
                    raw_output = _bta.breakdown_inferencer.infer(
                        _inf_input,
                        inference_config=_inf_config,
                        run_context=_bta._rc_child("breakdown"),
                    )

                    if _bta.breakdown_parser is not None:
                        sub_queries = _bta.breakdown_parser(raw_output)
                    elif _bta.breakdown_format == "json_subtasks":
                        sub_queries = _bta._parse_json_subtasks(raw_output)
                    elif _bta.breakdown_format == "numbered_list":
                        sub_queries = parse_numbered_list(str(raw_output))
                    elif isinstance(raw_output, list):
                        sub_queries = raw_output
                    else:
                        sub_queries = parse_numbered_list(str(raw_output))

                    _bta._save_breakdown_checkpoint(raw_output, sub_queries)

                if (
                    _bta.max_breakdown is not None
                    and len(sub_queries) > _bta.max_breakdown
                ):
                    sub_queries = sub_queries[: _bta.max_breakdown]

                if not sub_queries:
                    return raw_output if raw_output is not None else ""

                # Breakdown-only mode (skip when predefined_sub_queries — already warned)
                if _bta.breakdown_only and not _from_predefined:
                    return raw_output if raw_output is not None else sub_queries

                # Cache sub_queries BEFORE returning GraphExpansionResult
                _bta._cached_sub_queries = sub_queries

                # On resume, _reconstruct_graph_expansions already attached workers
                # to the breakdown node. WorkGraphNode._run resets _expansion_applied
                # to False, so we can't rely on that flag. Instead, check if the
                # breakdown node already has downstream nodes (workers) attached.
                _bd_node = _bta.start_nodes[0] if _bta.start_nodes else None
                if _bd_node and _bd_node.next:
                    # Workers already attached from reconstruction — skip expansion
                    return sub_queries

                subgraph = _bta._build_subgraph_spec(
                    sub_queries,
                    inference_config=_inf_config,
                    _original_query=_inf_input,
                    _inference_args=_inference_args,
                )

                has_aggregator = (
                    not _bta.disable_aggregator
                    and _bta.aggregator_inferencer is not None
                )
                expansion_id = "bta_diamond" if has_aggregator else "bta_workers"

                return GraphExpansionResult(
                    result=sub_queries,
                    subgraph=subgraph,
                    expansion_id=expansion_id,
                    seed=sub_queries,
                    reconstruct_from_seed=None,
                    attach_mode="insert",
                )

            return _breakdown_fn_sync

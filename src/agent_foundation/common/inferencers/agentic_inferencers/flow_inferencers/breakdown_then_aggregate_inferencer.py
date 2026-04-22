11

"""BreakdownThenAggregateInferencer — diamond-shaped WorkGraph-based inferencer.

Breaks a query into sub-queries, runs workers in parallel via WorkGraph,
and optionally aggregates results. Uses dual inheritance pattern
(InferencerBase, WorkGraph) following DualInferencer/PTI precedent.
"""

import functools
import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Union

from attr import attrib, attrs
from agent_foundation.common.inferencers.inferencer_base import (
    InferencerBase,
)
from rich_python_utils.common_objects.workflow.common.result_pass_down_mode import (
    ResultPassDownMode,
)
from rich_python_utils.common_objects.workflow.common.step_result_save_options import (
    StepResultSaveOptions,
    ResumeMode,
)
from rich_python_utils.common_objects.workflow.workgraph import (
    WorkGraph,
    WorkGraphNode,
)


_logger = logging.getLogger(__name__)


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
    hash_file_canonical as _sha256_of_file_canonical,
    find_conflicting_and_agreed_files,
    safe_copy_agreed,
    group_conflicts_by_parent,
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
        deliverables_promoted.append({
            "path": entry["path"],
            "size": entry["size"],
            "sha256": entry["sha256"],
            "source_workers": entry["source_roots"],
        })
        if entry["path"] in copied:
            _logger.info(
                "Auto-promoted %s (%d bytes, agreed by %d worker(s))",
                entry["path"], entry["size"], len(entry["source_roots"]),
            )

    # Remap conflict field names (root_name → worker)
    deliverables_with_conflicts = {}
    for rel_path, instances in conflicts.items():
        deliverables_with_conflicts[rel_path] = [
            {**inst, "worker": inst.pop("root_name")} for inst in instances
        ]
        _logger.warning(
            "Conflict detected on %s — %d distinct versions",
            rel_path, len({i["sha256"] for i in deliverables_with_conflicts[rel_path]}),
        )

    return deliverables_promoted, deliverables_with_conflicts


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
    def _builder(worker_results, original_query=None, worker_output_paths=None, bta=None):
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
                return f"### Result {idx+1}\n(See file: `{path}`)"
            return f"### Result {idx+1}\n{res}"

        paths = worker_output_paths or [None] * len(worker_results)
        default_text = "\n\n".join(
            _format_result(i, r, paths[i] if i < len(paths) else None)
            for i, r in enumerate(worker_results)
        )

        if conflict_resolution_mode == "last_writer_wins":
            return default_text

        if not worker_output_paths or not any(worker_output_paths):
            return default_text

        first_path = next((p for p in worker_output_paths if p), None)
        if first_path is None:
            return default_text

        cur = os.path.abspath(first_path)
        while cur and os.path.basename(cur) != "children":
            parent = os.path.dirname(cur)
            if parent == cur:
                cur = None
                break
            cur = parent
        if cur is None:
            return default_text

        children_dir = cur
        ws_root = os.path.dirname(children_dir)
        fd_path = os.path.join(ws_root, "outputs", deliverables_subdir)
        deliverables_dst = fd_path if os.path.isdir(fd_path) else os.path.join(ws_root, "outputs")
        os.makedirs(deliverables_dst, exist_ok=True)

        promoted, conflicts = _detect_conflicts_and_promote(
            deliverables_dst, children_dir, candidate_subdirs,
        )

        conflicts_grouped = group_conflicts_by_parent(conflicts, depth=2)

        # Option 2: inject structured data into aggregator's template_extra_feed
        if bta is not None and hasattr(bta, "aggregator_inferencer") and bta.aggregator_inferencer is not None:
            agg_inf = bta.aggregator_inferencer
            if hasattr(agg_inf, "template_extra_feed"):
                agg_inf.template_extra_feed.update({
                    "deliverables_promoted": promoted,
                    "deliverables_with_conflicts": [
                        {"path": rp, "candidates": cands}
                        for rp, cands in conflicts.items()
                    ],
                    "conflicts_grouped_by_parent": conflicts_grouped,
                    "deliverables_dst": deliverables_dst,
                    "worker_summaries": worker_summaries,
                })

        return default_text

    return _builder


@attrs(slots=False)
class BreakdownThenAggregateInferencer(InferencerBase, WorkGraph):
    """Diamond-shaped inferencer: breakdown → parallel workers → aggregate.

    Follows the dual inheritance pattern from DualInferencer(InferencerBase, Workflow)
    and PlanThenImplementInferencer(InferencerBase, Workflow), but uses WorkGraph
    instead of Workflow for parallel fan-out/fan-in execution.

    MRO: InferencerBase.__call__() -> infer() wins over WorkNodeBase.__call__() -> run().
    run()/arun() are blocked — callers must use infer()/ainfer().

    The graph is built DYNAMICALLY in _infer()/_ainfer() each time,
    similar to how DualInferencer builds self._steps in _ainfer().

    Graph structure (2-layer diamond)::

        Layer 1 (start_nodes):  worker_0, worker_1, ..., worker_N   (parallel fan-out)
                                     \        |            /
        Layer 2:                        aggregator                  (fan-in)

    The breakdown step runs *before* the graph is constructed (since the number
    of worker nodes depends on its output) and is not itself a graph node.

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
    # worker_factory can be:
    #   - Callable(sub_query, index) -> InferencerBase  (homogeneous, all same type)
    #   - dict[str, Callable | functools.partial]: maps task type -> factory.
    #     functools.partial entries are called with no args to create fresh instances.
    #     "__default__" can be a string referencing another key.
    #     Requires task_type_arg_name and parser returning List[dict] with "args".
    worker_factory: Any = attrib(default=None)

    # When set, enables heterogeneous workers. Each sub_query item can be a dict
    # {"query": str, "args": {...}}. The value of args[task_type_arg_name] selects
    # which worker factory to use from a dict-typed worker_factory.
    task_type_arg_name: Optional[str] = attrib(default=None, kw_only=True)
    # Controls whether subtasks with multiple "todos" are expanded into one
    # worker per todo. Accepts bool (all types) or dict {task_type: bool}
    # for per-type control.
    expand_todos_to_workers: Union[bool, Dict[str, bool]] = attrib(default=False, kw_only=True)

    # === Aggregation ===
    aggregator_inferencer: Optional[InferencerBase] = attrib(default=None)
    aggregator_prompt_builder: Optional[Callable] = attrib(default=None)

    # === Checkpoint ===
    checkpoint_dir: Optional[str] = attrib(default=None)

    # === Workspace support (opt-in, overrides checkpoint_dir when set) ===
    # workspace_root: str — convenience shorthand to create a plain InferencerWorkspace.
    # workspace: InferencerWorkspace — inherited from InferencerBase; takes precedence.
    #   Configure workspace layout (e.g., use_final_deliverables_folder) on the
    #   InferencerWorkspace object directly, keeping workspace concerns out of BTA.
    workspace_root: Optional[str] = attrib(default=None)

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
    predefined_sub_queries: Optional[Union[str, List]] = attrib(default=None, kw_only=True)

    # Optional graph reporter for UI visualization (WebSocketGraphReporter or similar).
    # Protocol: must implement on_graph_topology(event), on_node_status(node_id, status, error).
    # Set by the executor after instantiation. BTA never imports WebSocket code directly.
    graph_reporter: Optional[Any] = attrib(default=None, kw_only=True)
    # Guard: _build_diamond_graph() is called from 6+ paths — emit topology only once per call.
    # Reset at the top of _build_diamond_graph so reused BTA instances work correctly.
    _graph_topology_emitted: bool = attrib(default=False, init=False, repr=False)

    # Suppress WorkGraph's start_nodes requirement at construction time
    # (graph is built dynamically in _infer/_ainfer)
    start_nodes = attrib(factory=list)

    def __attrs_post_init__(self):
        # InferencerBase.__attrs_post_init__ syncs self.workspace → self._workspace.
        # We override here to also handle workspace_root (convenience shorthand)
        # and to call ensure_dirs().
        super().__attrs_post_init__()

        if self._workspace is not None:
            # workspace was provided directly via InferencerBase.workspace — use as-is.
            self._workspace.ensure_dirs()
        elif self.workspace_root is not None:
            # Convenience: create a plain InferencerWorkspace from the root path.
            from agent_foundation.common.inferencers.inferencer_workspace import (
                InferencerWorkspace,
            )
            self._workspace = InferencerWorkspace(root=self.workspace_root)
            self._workspace.ensure_dirs()
        else:
            self._workspace = None

        # Auto-default output_path for the pipeline report.
        # This is the fallback filename used when the aggregator produces no
        # file outputs (written to outputs/). The default name makes it clear
        # this is the pipeline's aggregation report, not a final deliverable.
        if not self.output_path:
            self.output_path = "aggregation_report.md"

        # Re-resolve deferred "auto" logger now that workspace is available
        if isinstance(self.logger, str) and self.logger == "auto" and self._workspace:
            self._normalize_loggers()

        # Configure breakdown_inferencer with the BTA's workspace.
        # breakdown_inferencer runs BEFORE _build_diamond_graph(), so
        # _configure_child_workspace() is never called on it there.
        # We must configure it here, at init time, using a dedicated child workspace.
        if self._workspace is not None and self.breakdown_inferencer is not None:
            breakdown_ws = self._workspace.child("breakdown")
            breakdown_ws.ensure_dirs()
            self.breakdown_inferencer._workspace = breakdown_ws  # setter auto-configures

        # BTA is an orchestrator — it should NOT render its own inference_input
        # through a template (_render_prompt override below handles this).
        # Keep template_manager set so _finalize_output can write the BTA's
        # output file (e.g., role_document.md) to workspace/outputs/.
        self.template_key = ""

    def _render_prompt(self, inference_input: Any) -> Any:
        """BTA is an orchestrator — it never renders its own inference_input.

        The inference_input (task description) is passed directly to the
        breakdown inferencer, which applies its own template_manager.
        """
        return inference_input

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

    def _parse_json_subtasks(self, raw_output: str) -> List:
        """Parse JSON subtask format from the task_breakdown template.

        Extracts subtasks from ``<Response>`` tags or raw text, parses JSON
        with a ``subtasks`` array, and builds structured sub_queries for BTA.
        Falls back to ``parse_numbered_list`` if JSON extraction fails.

        This is the built-in parser for ``breakdown_format="json_subtasks"``,
        consolidating the parsing logic previously duplicated across tools.
        """
        from agent_foundation.common.response_parsers import extract_delimited

        response_text = extract_delimited(str(raw_output))

        # Try to extract JSON from ```json ... ``` code fence
        json_match = re.search(
            r"```json[^\n{]*(\{[\s\S]*\})\s*```", response_text
        )
        if not json_match:
            json_match = re.search(
                r'\{[\s\S]*"subtasks"[\s\S]*\}', response_text
            )
            if json_match:
                json_str = json_match.group(0)
            else:
                _logger.warning("No JSON in breakdown output, falling back to numbered list")
                return parse_numbered_list(response_text)
        else:
            json_str = json_match.group(1)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            _logger.warning("JSON parse failed (%s), falling back to numbered list", e)
            return parse_numbered_list(response_text)

        subtasks = data.get("subtasks") or data.get("decomposed_subtasks") or []
        if not subtasks:
            return parse_numbered_list(response_text)

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

    async def _emit_pending_graph_topology(self) -> None:
        """Emit the pending graph topology and wire status callbacks to all nodes.

        Called from _ainfer() after _build_diamond_graph() for ALL 3 execution paths:
        1. Checkpoint-resume path
        2. Predefined sub-queries path
        3. Normal breakdown path (the main first-run case)

        Previously only called in path 1 — that was the critical bug causing graph
        visualization to never appear on normal runs.
        """
        pending_topo = getattr(self, '_pending_topology', None)
        _logger.info(
            "[BTA] _emit_pending_graph_topology: has_reporter=%s has_pending=%s",
            self.graph_reporter is not None, pending_topo is not None
        )
        if pending_topo is not None:
            _logger.info(
                "[BTA] topology: %d nodes, %d edges",
                len(pending_topo.nodes), len(pending_topo.edges)
            )
        if self.graph_reporter is None or pending_topo is None:
            return
        self._pending_topology = None  # clear before await (re-entrant safety)
        try:
            await self.graph_reporter.on_graph_topology(pending_topo)
            # Wire async status callback to all WorkGraph nodes
            reporter = self.graph_reporter
            # Resolve output paths DYNAMICALLY at completion time (not at topology time).
            # Files don't exist when topology is emitted — they're written during execution.
            # When a node completes, its output file exists and can be resolved.
            _workspace = getattr(self, 'workspace_root', None) or getattr(self, '_workspace_root', None)

            async def _async_status_cb(event):
                output_path = ""
                if event.status in ("completed", "error") and _workspace:
                    from pathlib import Path as _P
                    _ws = _P(str(_workspace))
                    nid = event.node_id
                    if nid == "breakdown":
                        for candidate in [
                            _ws / "children" / "breakdown" / "outputs" / "breakdown_output.md",
                            _ws / "checkpoints" / "breakdown_result.json",
                        ]:
                            if candidate.exists():
                                output_path = str(candidate)
                                break
                    elif nid.startswith("worker_"):
                        for fn in ("facet.md", "result.md", "output.md", "response.md"):
                            candidate = _ws / "children" / nid / "outputs" / fn
                            if candidate.exists():
                                output_path = str(candidate)
                                break
                    elif nid == "aggregator":
                        for fn in ("role_document.md", "output.md", "result.md", "response.md"):
                            candidate = _ws / "children" / "aggregator" / "outputs" / fn
                            if candidate.exists():
                                output_path = str(candidate)
                                break
                await reporter.on_node_status(
                    event.node_id, event.status,
                    getattr(event, 'error', ''),
                    output_path=output_path,
                )
            self.set_graph_event_callback(_async_status_cb)

            # Breakdown is a VIRTUAL node — manually prepended to the topology, NOT a real
            # WorkGraphNode. So _async_status_cb (set on real WorkGraph nodes) never fires
            # for it. Emit an explicit completion event with the resolved output_path so
            # the UI can fetch breakdown_output.md when the breakdown node is clicked.
            if _workspace:
                from pathlib import Path as _PB
                _ws_b = _PB(str(_workspace))
                _bd_output = ""
                for _cand in [
                    _ws_b / "children" / "breakdown" / "outputs" / "breakdown_output.md",
                    _ws_b / "checkpoints" / "breakdown_result.json",
                ]:
                    if _cand.exists():
                        _bd_output = str(_cand)
                        break
                try:
                    await reporter.on_node_status(
                        "breakdown", "completed",
                        output_path=_bd_output,
                    )
                except Exception as _ebd:
                    import logging as _logbd
                    _logbd.getLogger(__name__).warning(
                        "[BTA] breakdown node_status emit failed (visualization only): %s", _ebd
                    )
        except Exception as _e:
            import logging as _log
            _log.getLogger(__name__).warning(
                "[BTA] graph topology emit failed (visualization only): %s", _e
            )

    def _build_diamond_graph(self, sub_queries, inference_config=None, **kwargs):
        """Build the diamond-shaped WorkGraph dynamically from sub-queries.

        Creates N worker nodes (one per sub-query) and optionally an
        aggregation node that collects all worker results.
        """
        # Reset topology emission guard — allows reused BTA instances to re-emit
        # correctly when called again with different sub-queries.
        self._graph_topology_emitted = False
        # Clear stale graph state from any prior _infer() call (must come after flag reset)
        self._clear_all_node_queues()
        self.start_nodes = []

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
            if isinstance(self.worker_factory, dict) and self.task_type_arg_name:
                task_type = sq_args.get(self.task_type_arg_name, "__default__")

            # expand_todos_to_workers: bool applies to all, dict gives per-type control
            if isinstance(self.expand_todos_to_workers, dict):
                should_expand = self.expand_todos_to_workers.get(task_type, False) if task_type else False
            else:
                should_expand = self.expand_todos_to_workers

            # Legacy: per-type override from dict-style factory entry
            factory_entry = None
            if task_type and isinstance(self.worker_factory, dict):
                factory_entry = self.worker_factory.get(
                    task_type, self.worker_factory.get("__default__")
                )
                if isinstance(factory_entry, dict):
                    should_expand = factory_entry.get("expand_todos", should_expand)

            todos = sq_args.get("todos") if isinstance(sq, dict) else None
            if should_expand and todos and len(todos) > 1:
                # Expand: one worker per todo
                desc = sq_args.get("description", query_str)
                for todo in todos:
                    expanded_sq = dict(sq) if isinstance(sq, dict) else {"query": sq}
                    expanded_sq["query"] = f"**Description**: {desc}\n\n**Todo**:\n- {todo}"
                    expanded_sq["args"] = dict(sq_args)  # preserve task_preamble etc.
                    expanded_queries.append(expanded_sq)
            else:
                expanded_queries.append(sq)

        if len(expanded_queries) != len(sub_queries):
            _logger.info(
                "Expanded %d sub_queries → %d workers (expand_todos_to_workers)",
                len(sub_queries), len(expanded_queries),
            )

        worker_nodes = []
        worker_output_paths = []  # for aggregator prompt closure
        _bta_prefix = f"{self.name}." if getattr(self, "name", None) else ""
        for i, sq in enumerate(expanded_queries):
            # Extract query string and args (backward compat: sq can be str or dict)
            if isinstance(sq, dict):
                query_str = sq.get("query", str(sq))
                sq_args = sq.get("args", {})
            else:
                query_str = sq
                sq_args = {}

            # Select and invoke worker factory
            task_type = None
            if isinstance(self.worker_factory, dict):
                # Heterogeneous workers: look up factory by task type
                task_type = (
                    sq_args.get(self.task_type_arg_name, "__default__")
                    if self.task_type_arg_name
                    else "__default__"
                )
                factory_entry = self.worker_factory.get(
                    task_type, self.worker_factory.get("__default__")
                )
                # Resolve __default__ string references (e.g., "__default__": "research")
                if isinstance(factory_entry, str):
                    factory_entry = self.worker_factory.get(factory_entry)
                if factory_entry is None:
                    raise ValueError(
                        f"No worker factory for task type '{task_type}' "
                        f"and no '__default__' fallback"
                    )
                # Support: dict with "factory" key, functools.partial, or callable
                if isinstance(factory_entry, dict) and "factory" in factory_entry:
                    factory = factory_entry["factory"]
                else:
                    factory = factory_entry
                # Partials create fresh instances with no extra args
                if isinstance(factory, functools.partial):
                    worker = factory()
                else:
                    worker = factory(sub_query=query_str, index=i)
            else:
                # Homogeneous workers
                if isinstance(self.worker_factory, functools.partial):
                    worker = self.worker_factory()
                else:
                    worker = self.worker_factory(sub_query=query_str, index=i)

            # Assign child workspace to worker (full composition mode)
            if self._workspace is not None and isinstance(worker, InferencerBase):
                prev_ws = getattr(worker, "_workspace", None)
                use_fdl = (
                    getattr(prev_ws, "use_final_deliverables_folder", False)
                    if prev_ws
                    else False
                )
                worker_ws = self._workspace.child(f"worker_{i}")
                if use_fdl:
                    from agent_foundation.common.inferencers.inferencer_workspace import (
                        InferencerWorkspace,
                    )
                    worker_ws = InferencerWorkspace(
                        root=worker_ws.root,
                        use_final_deliverables_folder=use_fdl,
                    )
                worker_ws.ensure_dirs()
                worker._workspace = worker_ws  # setter auto-configures
                worker_output_paths.append(worker.resolve_output_path())
            else:
                worker_output_paths.append(None)

            # Hierarchical naming: prefix worker node names with BTA name
            # so nested BTAs produce e.g. "outer_bta.worker_0.worker_3"
            _node_name = f"{_bta_prefix}worker_{i}"
            if isinstance(worker, BreakdownThenAggregateInferencer):
                worker.name = _node_name

            # Create a callable that captures this specific sub-query.
            # It intentionally ignores args from WorkGraph (which passes
            # the same args to all start nodes).
            # When the worker supports ainfer() AND the graph is in async mode,
            # returns an async coroutine for true parallel I/O.
            # NOTE: uses query_str (not sq) so ainfer always receives a string.
            use_async = getattr(self, "use_async", False)

            # Detect if the worker manages its own resume (e.g., PTI, nested BTA).
            from rich_python_utils.common_objects.workflow.common.resumable import Resumable
            _worker_manages_resume = isinstance(worker, Resumable) and bool(
                getattr(worker, "resume_with_saved_results", False)
            )

            def _make_worker_fn(w, q, is_async, manages_resume, _reporter=None, _node_id=None):
                """Create a WorkGraphNode function for a single worker.

                _reporter and _node_id: graph visualization hooks (Fix 3).
                After w.ainfer() completes, emit result as node_stream so
                clicking the node in TaskPanel shows its output.
                NOTE: _reporter is captured as a parameter (not via `self`) because
                this function is NOT a method — `self` would be unbound here.
                """
                def _try_load_from_output():
                    """Backup resume: if worker is non-resumable and has no
                    checkpoint, check if its output file/dir already exists."""
                    if manages_resume:
                        return None
                    output_path = (
                        w.resolve_output_path()
                        if hasattr(w, "resolve_output_path") else None
                    )
                    if not output_path:
                        return None
                    try:
                        if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                            with open(output_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            _logger.info(
                                "Backup resume: output file exists, skipping worker: %s (%d bytes)",
                                output_path, len(content),
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
                        cached = _try_load_from_output()
                        if cached is not None:
                            # Emit cached output so NodeDetailPanel shows it on click
                            if _reporter is not None and _node_id is not None:
                                try:
                                    await _reporter.on_node_stream(
                                        _node_id, str(cached), is_final=True
                                    )
                                except Exception:
                                    pass
                            return cached
                        result = await w.ainfer(q, inference_config=inference_config)
                        # Fix 3: Emit result as node_stream after completion.
                        # RovoChatInferencer doesn't stream through interactive, so this
                        # is the only way to get content into NodeDetailPanel.
                        if _reporter is not None and _node_id is not None:
                            try:
                                await _reporter.on_node_stream(
                                    _node_id, str(result) if result else "", is_final=True
                                )
                            except Exception:
                                pass
                        return result

                    return async_worker_fn
                else:

                    def worker_fn(*_args, **_kwargs):
                        cached = _try_load_from_output()
                        if cached is not None:
                            return cached
                        if hasattr(w, "infer"):
                            return w.infer(q, inference_config=inference_config)
                        return w(q)

                    return worker_fn

            # Determine group for per-group concurrency limiting
            worker_group = task_type if isinstance(self.worker_factory, dict) else None

            # Graph visualization: inject per-node interactive BEFORE _make_worker_fn
            # captures `worker` in a closure. w.interactive is read at call time by
            # w.ainfer(q) — so setting it here (before closure creation) ensures correct routing.
            # NOTE: RovoChatInferencer doesn't use self.interactive for streaming,
            # so NodeStreamInteractive won't intercept live tokens. Instead we emit the
            # final result via the worker function wrapper below (Fix 3).
            if self.graph_reporter is not None:
                worker.interactive = self.graph_reporter.node_interactive(_node_name)
                if hasattr(worker, 'stream_observer'):
                    worker.stream_observer = self.graph_reporter.node_stream_observer(_node_name)
                # Propagate graph_reporter to nested WorkGraph-based inferencers
                # so their internal topology is visible in the UI as a sub-graph.
                if hasattr(worker, 'graph_reporter') and worker.graph_reporter is None:
                    worker.graph_reporter = self.graph_reporter.child_reporter(_node_name)
                    # Clear name to prevent double-prefixing: _bta_prefix (dot-based)
                    # would conflict with NamespacedGraphReporter (slash-based).
                    if isinstance(worker, BreakdownThenAggregateInferencer):
                        worker.name = None

            _is_container = (
                self.graph_reporter is not None
                and hasattr(worker, 'graph_reporter')
                and worker.graph_reporter is not None
            )
            _w_reporter = self.graph_reporter if self.graph_reporter is not None else None
            node = WorkGraphNode(
                name=_node_name,
                value=_make_worker_fn(
                    worker, query_str, use_async, _worker_manages_resume,
                    _reporter=_w_reporter, _node_id=_node_name,
                ),
                result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
                group=worker_group,
                enable_result_save=StepResultSaveOptions.SkipResumable,
                resume_with_saved_results=ResumeMode.SkipResumable,
                worker_manages_resume=_worker_manages_resume,
                retry_on_exceptions=(Exception,),
            )
            # Store subtask description as display label for graph visualization UI.
            # _viz_label is read by GraphTopologyEvent.from_work_graph() via node_map.
            # Strip markdown bold markers (**text**, __text__) and common prefixes
            # from labels for clean graph node display.
            # Some breakdown JSON formats include "Description: Define the..." as the label.
            _raw_label = (
                str(query_str)[:120] if isinstance(query_str, str)
                else query_str.get("description", query_str.get("query", f"worker_{i}"))[:120]
                if isinstance(query_str, dict) else f"worker_{i}"
            )
            _raw_label = _raw_label.replace("**", "").replace("__", "").strip()
            # Strip common field prefixes added by some LLM breakdown formats
            for _prefix in ("Description:", "description:", "Task:", "task:", "Query:", "query:"):
                if _raw_label.startswith(_prefix):
                    _raw_label = _raw_label[len(_prefix):].strip()
                    break
            node._viz_label = _raw_label[:80]  # cap after stripping
            if _is_container:
                node._is_container = True

            worker_nodes.append(node)

        agg_node = None
        if self.disable_aggregator or self.aggregator_inferencer is None:
            # No aggregation — workers are terminal nodes
            self.start_nodes = worker_nodes
            return

        if self.aggregator_inferencer is not None:
            # Create aggregation node that receives all worker results.
            # worker_output_paths is captured by closure for single source
            # of truth — same paths workers write to.
            _captured_paths = list(worker_output_paths)

            _bta_self = self  # capture for closure

            def _build_agg_input(prompt_builder, worker_results, original_query):
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
                # Default: if aggregator has local file access, pass paths only
                # (avoids sending 100K+ of worker text inline).
                # Otherwise embed full worker text.
                agg_has_local = getattr(agg_inf, "has_local_access", False)
                parts = []
                for idx, res in enumerate(worker_results):
                    path = (
                        _captured_paths[idx]
                        if idx < len(_captured_paths)
                        else None
                    )
                    if agg_has_local and path:
                        # Aggregator can read files directly — pass path only
                        parts.append(
                            f"### Result {idx + 1}\n(See file: `{path}`)"
                        )
                    else:
                        # No local access — embed full text with path hint
                        path_ref = (
                            f"\n(Full output at: `{path}`)" if path else ""
                        )
                        parts.append(
                            f"### Result {idx + 1}\n{res}{path_ref}"
                        )
                return "\n\n".join(parts)

            def _make_agg_fn(agg_inf, prompt_builder, original_query, is_async, _reporter=None):
                """Create the aggregator function for the WorkGraph.

                _reporter: graph visualization hook (Fix 4).
                After agg_inf.ainfer() completes, emit result as node_stream so
                clicking "Aggregator" in TaskPanel shows the synthesized output.
                NOTE: _reporter captured as parameter (not via `self`) — not a method.
                """
                if is_async and hasattr(agg_inf, "ainfer"):

                    async def async_agg_fn(*worker_results, **_kwargs):
                        agg_input = _build_agg_input(
                            prompt_builder, worker_results, original_query
                        )
                        result = await agg_inf.ainfer(
                            agg_input, inference_config=inference_config
                        )
                        # Fix 4: Emit aggregator output as node_stream on completion.
                        if _reporter is not None:
                            try:
                                await _reporter.on_node_stream(
                                    "aggregator", str(result) if result else "", is_final=True
                                )
                            except Exception:
                                pass
                        return result

                    return async_agg_fn
                else:

                    def agg_fn(*worker_results, **_kwargs):
                        agg_input = _build_agg_input(
                            prompt_builder, worker_results, original_query
                        )
                        if hasattr(agg_inf, "infer"):
                            return agg_inf.infer(
                                agg_input, inference_config=inference_config
                            )
                        return agg_inf(agg_input)

                    return agg_fn

            original_query = kwargs.get("_original_query", "")

            # Give the aggregator its own child workspace so its logs,
            # outputs, and artifacts are organized under children/aggregator/
            # (same pattern as workers get children/worker_*/).
            agg_inf = self.aggregator_inferencer
            if callable(agg_inf) and not isinstance(agg_inf, InferencerBase):
                agg_inf = agg_inf()
                self.aggregator_inferencer = agg_inf
            if self._workspace is not None and isinstance(agg_inf, InferencerBase):
                agg_ws = self._workspace.child("aggregator")
                agg_ws.ensure_dirs()
                agg_inf._workspace = agg_ws  # setter auto-configures

            # Wire stream_observer for live aggregator streaming in graph visualization
            _agg_node_name = f"{_bta_prefix}aggregator" if _bta_prefix else "aggregator"
            if self.graph_reporter is not None and hasattr(agg_inf, 'stream_observer'):
                agg_inf.stream_observer = self.graph_reporter.node_stream_observer(_agg_node_name)

            agg_node = WorkGraphNode(
                name=_agg_node_name,
                value=_make_agg_fn(
                    agg_inf,
                    self.aggregator_prompt_builder,
                    original_query,
                    use_async,
                    _reporter=self.graph_reporter,  # Fix 4: emit aggregator result as node_stream
                ),
                result_pass_down_mode=ResultPassDownMode.NoPassDown,
                enable_result_save=self.enable_result_save,
                resume_with_saved_results=self.resume_with_saved_results,
                checkpoint_mode=self.checkpoint_mode,
                retry_on_exceptions=(Exception,),
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

        self.start_nodes = worker_nodes

        # Graph visualization: store topology for async emit in _ainfer().
        # _build_diamond_graph is sync — we can't await here, so defer to _ainfer.
        # Use getattr/None pattern (safer than hasattr after exceptions).
        if self.graph_reporter is not None and not self._graph_topology_emitted:
            self._graph_topology_emitted = True
            try:
                from agent_foundation.common.inferencers.graph_events import (
                    GraphTopologyEvent, NodeStatus,
                )
                # Build topology from the actual WorkGraph nodes (workers + aggregator).
                # Breakdown runs BEFORE this graph is built — manually prepend it as virtual node.
                worker_agg_topology = GraphTopologyEvent.from_work_graph(self)
                breakdown_node = {
                    "id": "breakdown", "label": "Breakdown",
                    "group": None, "status": NodeStatus.COMPLETED,
                }
                worker_agg_topology.nodes.insert(0, breakdown_node)
                for wn in worker_nodes:
                    worker_agg_topology.edges.insert(
                        0, {"source": "breakdown", "target": wn.name}
                    )
                self._pending_topology = worker_agg_topology
            except Exception as _e:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "graph topology build failed: %s", _e, exc_info=True
                )
                self._pending_topology = None

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
            "checkpoint_dir or workspace_root must be set for result saving"
        )

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
        psq = self.predefined_sub_queries
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
            with open(ckpt, "w") as f:
                json.dump(
                    {"raw_output": str(raw_output), "sub_queries": sub_queries},
                    f,
                    indent=2,
                )
            _logger.info(
                "Saved breakdown checkpoint with %d sub_queries", len(sub_queries)
            )
        except OSError as e:
            _logger.warning("Failed to save breakdown checkpoint: %s", e)

    async def _emit_graph_reconcile(self):
        """Emit final node statuses so the frontend can correct any stale UI state."""
        if self.graph_reporter is None:
            return
        try:
            statuses = {n.name: "completed" for n in self._all_nodes()}
            statuses["breakdown"] = "completed"
            await self.graph_reporter.on_graph_reconcile(statuses)
        except Exception:
            pass

    def _finalize_response(self, result):
        """Route aggregator deliverables to root workspace.

        Deliverables are copied from the aggregator's child workspace outputs/
        to either:
          - ``root/final_deliverables/`` (when workspace.use_final_deliverables_folder is set)
          - ``root/outputs/``            (fallback when no deliverables_dir configured)

        The pipeline report (aggregator's text response) is always written to
        ``root/outputs/<output_path>`` (default: ``aggregation_report.md``).

        Only runs in workspace mode with output_path set. Idempotent.
        """
        if self._workspace is None or not self.output_path:
            return

        import shutil

        agg_inf = self.aggregator_inferencer
        agg_ws = getattr(agg_inf, "_workspace", None) if agg_inf else None

        # Determine where to copy deliverables: deliverables_dir or outputs_dir
        deliverables_dst = self._workspace.deliverables_dir or str(self._workspace.outputs_dir)

        copied_any_deliverable = False
        if agg_ws is not None:
            agg_outputs = os.path.join(str(agg_ws.root), "outputs")
            if os.path.isdir(agg_outputs) and os.listdir(agg_outputs):
                shutil.copytree(agg_outputs, deliverables_dst, dirs_exist_ok=True)
                _logger.info(
                    "Copied aggregator deliverables (recursive) → %s",
                    deliverables_dst,
                )
                copied_any_deliverable = True

        if self.promote_worker_deliverables and self._workspace is not None:
            from rich_python_utils.path_utils.path_listing import (
                find_conflicting_and_agreed_files,
                safe_copy_per_file,
            )
            children_dir = self._workspace.children_dir
            if os.path.isdir(children_dir):
                roots = []
                root_names = []
                for child_name in sorted(os.listdir(children_dir)):
                    if not child_name.startswith("worker_"):
                        continue
                    child_root = os.path.join(children_dir, child_name)
                    child_fd = os.path.join(child_root, "outputs", "final_deliverables")
                    child_out = os.path.join(child_root, "outputs")
                    src = child_fd if os.path.isdir(child_fd) else child_out
                    if os.path.isdir(src) and os.listdir(src):
                        roots.append(src)
                        root_names.append(child_name)
                if roots:
                    diff = find_conflicting_and_agreed_files(roots, root_names)
                    # skip_existing=True protects files already written by the
                    # aggregator (merged conflict resolutions) or auto-promoted
                    # by the prompt builder. conflict_fallback="largest" handles
                    # the case where the aggregator failed to write a merged version.
                    copied = safe_copy_per_file(
                        diff, deliverables_dst,
                        skip_existing=True,
                        conflict_fallback="largest",
                    )
                    if copied:
                        _logger.info(
                            "Promoted %d worker deliverable(s) → %s",
                            len(copied), deliverables_dst,
                        )
                        copied_any_deliverable = True

        if copied_any_deliverable:
            self._deliverables_copied = True
            _logger.info(
                "Skipping pipeline report — aggregator deliverables copied to %s",
                deliverables_dst,
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
                with open(report_dst, "w") as f:
                    f.write(text)
                _logger.info("Wrote pipeline report → %s", report_dst)
            except OSError as e:
                _logger.warning(
                    "Failed to write pipeline report to %s: %s", report_dst, e
                )

    def _finalize_output(self, response):
        if getattr(self, "_deliverables_copied", False):
            return response
        return super()._finalize_output(response)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        """Core inference: breakdown → build graph → run graph."""
        # Step 0: Check for saved breakdown checkpoint
        sub_queries = self._load_breakdown_checkpoint()
        if sub_queries is not None:
            # Skip breakdown, jump to cap + graph
            if self.max_breakdown is not None and len(sub_queries) > self.max_breakdown:
                sub_queries = sub_queries[: self.max_breakdown]
            if not sub_queries:
                return ""
            self._build_diamond_graph(
                sub_queries,
                inference_config=inference_config,
                _original_query=inference_input,
                **kwargs,
            )
            result = WorkGraph._run(self, inference_input, **kwargs)
            if isinstance(result, tuple) and len(result) == 1:
                result = result[0]
            self._finalize_response(result)
            return result

        # Step 0b: Predefined sub-queries — skip breakdown entirely
        if self.predefined_sub_queries is not None:
            if self.breakdown_only:
                _logger.warning(
                    "predefined_sub_queries is set but breakdown_only=True — "
                    "breakdown_only ignored (no LLM breakdown to stop after)."
                )
            sub_queries = self._resolve_predefined_sub_queries()
            # Apply max_breakdown cap (consistent with checkpoint resume path)
            if self.max_breakdown is not None and len(sub_queries) > self.max_breakdown:
                sub_queries = sub_queries[: self.max_breakdown]
            if not sub_queries:
                return ""
            self._build_diamond_graph(
                sub_queries,
                inference_config=inference_config,
                _original_query=inference_input,
                **kwargs,
            )
            result = WorkGraph._run(self, inference_input, **kwargs)
            if isinstance(result, tuple) and len(result) == 1:
                result = result[0]
            self._finalize_response(result)
            return result

        # Step 1: Breakdown
        if self.breakdown_inferencer is None:
            raise ValueError(
                "breakdown_inferencer must be set when predefined_sub_queries is None. "
                "Either provide a breakdown_inferencer or set predefined_sub_queries."
            )
        raw_output = self.breakdown_inferencer.infer(
            inference_input, inference_config=inference_config
        )

        # Step 2: Parse breakdown output
        if self.breakdown_parser is not None:
            sub_queries = self.breakdown_parser(raw_output)
        elif self.breakdown_format == "json_subtasks":
            sub_queries = self._parse_json_subtasks(raw_output)
        elif self.breakdown_format == "numbered_list":
            sub_queries = parse_numbered_list(str(raw_output))
        elif isinstance(raw_output, list):
            sub_queries = raw_output
        else:
            sub_queries = parse_numbered_list(str(raw_output))

        # Step 2b: Save breakdown checkpoint
        self._save_breakdown_checkpoint(raw_output, sub_queries)

        # Step 3: Apply max_breakdown cap
        if self.max_breakdown is not None and len(sub_queries) > self.max_breakdown:
            sub_queries = sub_queries[: self.max_breakdown]

        if not sub_queries:
            return raw_output  # No sub-queries, return breakdown output as-is

        # Step 4: Build diamond graph
        self._build_diamond_graph(
            sub_queries,
            inference_config=inference_config,
            _original_query=inference_input,
            **kwargs,
        )

        # Step 5: Run the diamond via WorkGraph._run
        result = WorkGraph._run(self, inference_input, **kwargs)
        # Unwrap single-element tuples from WorkGraph's post_process
        if isinstance(result, tuple) and len(result) == 1:
            result = result[0]
        self._finalize_response(result)
        return result

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        """Async core inference: breakdown → build graph → run graph."""
        # Fix 1: Emit initial single-node "Breakdown: Running" topology immediately.
        # This shows the user that something is happening before breakdown completes
        # (which can take 30-60s). The full diamond topology replaces it later via
        # _emit_pending_graph_topology() (which overrides task.graph in the UI).
        if self.graph_reporter is not None:
            try:
                from agent_foundation.common.inferencers.graph_events import (
                    GraphTopologyEvent, NodeStatus,
                )
                initial_topo = GraphTopologyEvent(
                    nodes=[{"id": "breakdown", "label": "Breakdown", "group": None,
                            "status": NodeStatus.RUNNING}],
                    edges=[],
                    layout="horizontal",
                )
                await self.graph_reporter.on_graph_topology(initial_topo)
            except Exception as _e:
                _logger.warning("[BTA] initial topology emit failed: %s", _e)

        # Step 0: Check for saved breakdown checkpoint
        sub_queries = self._load_breakdown_checkpoint()
        if sub_queries is not None:
            if self.max_breakdown is not None and len(sub_queries) > self.max_breakdown:
                sub_queries = sub_queries[: self.max_breakdown]
            if not sub_queries:
                return ""
            old_use_async = getattr(self, "use_async", False)
            self.use_async = True
            try:
                self._build_diamond_graph(
                    sub_queries,
                    inference_config=inference_config,
                    _original_query=inference_input,
                    **kwargs,
                )
            finally:
                self.use_async = old_use_async
            # Graph visualization: emit topology + wire status callbacks (async, safe here)
            await self._emit_pending_graph_topology()
            result = await WorkGraph._arun(self, inference_input, **kwargs)
            if isinstance(result, tuple) and len(result) == 1:
                result = result[0]
            await self._emit_graph_reconcile()
            self._finalize_response(result)
            return result

        # Step 0b: Predefined sub-queries — skip breakdown entirely
        if self.predefined_sub_queries is not None:
            if self.breakdown_only:
                _logger.warning(
                    "predefined_sub_queries is set but breakdown_only=True — "
                    "breakdown_only ignored (no LLM breakdown to stop after)."
                )
            sub_queries = self._resolve_predefined_sub_queries()
            # Apply max_breakdown cap (consistent with checkpoint resume path)
            if self.max_breakdown is not None and len(sub_queries) > self.max_breakdown:
                sub_queries = sub_queries[: self.max_breakdown]
            if not sub_queries:
                return ""
            # NOTE: skip enable_checkpoint_sub_query_selection — user already chose sub-queries.
            # CRITICAL: set use_async=True so _build_diamond_graph creates async worker fns
            old_use_async = getattr(self, "use_async", False)
            self.use_async = True
            try:
                self._build_diamond_graph(
                    sub_queries,
                    inference_config=inference_config,
                    _original_query=inference_input,
                    **kwargs,
                )
            finally:
                self.use_async = old_use_async
            await self._emit_pending_graph_topology()  # Fix: was missing in predefined path
            result = await WorkGraph._arun(self, inference_input, **kwargs)
            if isinstance(result, tuple) and len(result) == 1:
                result = result[0]
            # Step 5b: Interactive results review — keep this even in predefined mode.
            # A re-run will correctly re-use the same predefined_sub_queries.
            if self.enable_checkpoint_results_review and self.interactive:
                # TODO: interactive_checkpoint module does not exist at agent_foundation.ui — needs separate migration
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

        # Step 1: Breakdown
        if self.breakdown_inferencer is None:
            raise ValueError(
                "breakdown_inferencer must be set when predefined_sub_queries is None. "
                "Either provide a breakdown_inferencer or set predefined_sub_queries."
            )
        # Wire stream_observer for live breakdown streaming in graph visualization
        if self.graph_reporter is not None and hasattr(self.breakdown_inferencer, 'stream_observer'):
            self.breakdown_inferencer.stream_observer = self.graph_reporter.node_stream_observer("breakdown")
        if hasattr(self.breakdown_inferencer, "ainfer"):
            raw_output = await self.breakdown_inferencer.ainfer(
                inference_input, inference_config=inference_config
            )
        else:
            raw_output = self.breakdown_inferencer.infer(
                inference_input, inference_config=inference_config
            )

        # Guard: detect API error responses masquerading as valid output.
        # Rovo/AI Gateway sometimes returns error strings (e.g., "An unknown error
        # occurred", "RECONNECT_SUPPORTED...") as 200 OK — the inferencer treats
        # them as valid. Detect and raise so the retry mechanism can handle it.
        _raw_str = str(raw_output).strip()
        _ERROR_PATTERNS = [
            "An unknown error occurred",
            "RECONNECT_SUPPORTED",
            "peer closed connection",
            "Internal Server Error",
        ]
        if any(p in _raw_str for p in _ERROR_PATTERNS) and len(_raw_str) < 200:
            raise RuntimeError(f"Breakdown returned API error instead of subtasks: {_raw_str[:100]}")

        # Step 2: Parse
        if self.breakdown_parser is not None:
            sub_queries = self.breakdown_parser(raw_output)
        elif self.breakdown_format == "json_subtasks":
            sub_queries = self._parse_json_subtasks(raw_output)
        elif self.breakdown_format == "numbered_list":
            sub_queries = parse_numbered_list(str(raw_output))
        elif isinstance(raw_output, list):
            sub_queries = raw_output
        else:
            sub_queries = parse_numbered_list(str(raw_output))

        # Step 2b: Save breakdown checkpoint
        self._save_breakdown_checkpoint(raw_output, sub_queries)

        # Step 3: Cap
        if self.max_breakdown is not None and len(sub_queries) > self.max_breakdown:
            sub_queries = sub_queries[: self.max_breakdown]

        if not sub_queries:
            return raw_output

        # Step 3b: Breakdown-only mode — return after breakdown phase
        if self.breakdown_only:
            return raw_output

        # Step 3c: Interactive sub-query selection checkpoint
        if self.enable_checkpoint_sub_query_selection and self.interactive:
            # TODO: interactive_checkpoint module does not exist at agent_foundation.ui — needs separate migration
            from agent_foundation.ui.interactive_checkpoint import (
                checkpoint_breakdown_review,
            )

            cp_result = await checkpoint_breakdown_review(
                self.interactive, sub_queries, default_action="approve"
            )
            if cp_result.action == "select" and cp_result.selected_indices:
                sub_queries = [
                    sub_queries[i]
                    for i in cp_result.selected_indices
                    if i < len(sub_queries)
                ]
            if not sub_queries:
                return raw_output

        # Fix 2: Emit breakdown result as node_stream so clicking "Breakdown" in the
        # TaskPanel graph shows the parsed subtask list.
        # Uses reporter.on_node_stream() public API — NOT reporter._ws directly.
        if self.graph_reporter is not None:
            try:
                _summary = []
                for _i, _sq in enumerate(sub_queries if isinstance(sub_queries, list) else [sub_queries]):
                    if isinstance(_sq, dict):
                        _desc = _sq.get("query", str(_sq))
                    else:
                        _desc = str(_sq)
                    if len(_desc) > 300:
                        _desc = _desc[:297] + "..."
                    _summary.append(f"**{_i+1}.** {_desc}")
                _bd_content = "\n\n".join(_summary)
                await self.graph_reporter.on_node_stream("breakdown", _bd_content, is_final=True)
            except Exception as _e:
                _logger.warning("[BTA] breakdown node_stream emit failed: %s", _e)

        # Step 4: Build diamond graph (force async mode for async worker fns)
        old_use_async = getattr(self, "use_async", False)
        self.use_async = True
        try:
            self._build_diamond_graph(
                sub_queries,
                inference_config=inference_config,
                _original_query=inference_input,
                **kwargs,
            )
        finally:
            self.use_async = old_use_async

        # Step 5: Run the diamond via WorkGraph._arun
        await self._emit_pending_graph_topology()  # Fix: was missing in normal breakdown path
        result = await WorkGraph._arun(self, inference_input, **kwargs)
        # Unwrap single-element tuples from WorkGraph's post_process
        if isinstance(result, tuple) and len(result) == 1:
            result = result[0]

        # Step 5b: Interactive results review checkpoint
        if self.enable_checkpoint_results_review and self.interactive:
            # TODO: interactive_checkpoint module does not exist at agent_foundation.ui — needs separate migration
            from agent_foundation.ui.interactive_checkpoint import (
                checkpoint_results_review,
            )

            result_str = str(result)[:2000]
            cp_result = await checkpoint_results_review(
                self.interactive, result_str, default_action="approve"
            )
            if cp_result.action == "rerun":
                # Re-run the entire graph
                return await self._ainfer(inference_input, inference_config, **kwargs)

        await self._emit_graph_reconcile()
        self._finalize_response(result)
        return result

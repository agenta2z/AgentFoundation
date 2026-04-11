11

"""BreakdownThenAggregateInferencer — diamond-shaped WorkGraph-based inferencer.

Breaks a query into sub-queries, runs workers in parallel via WorkGraph,
and optionally aggregates results. Uses dual inheritance pattern
(InferencerBase, WorkGraph) following DualInferencer/PTI precedent.
"""

import json
import logging
import os
from typing import Any, Callable, List, Optional

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
    """

    # === Breakdown ===
    breakdown_inferencer: InferencerBase = attrib(default=None)
    max_breakdown: Optional[int] = attrib(default=None)
    breakdown_parser: Optional[Callable] = attrib(default=None)

    # === Per-query worker ===
    # worker_factory can be:
    #   - Callable(sub_query, index) -> InferencerBase  (homogeneous, all same type)
    #   - dict[str, Callable]: maps task type -> factory. "__default__" is fallback.
    #     Requires task_type_arg_name and parser returning List[dict] with "args".
    worker_factory: Any = attrib(default=None)  # Callable | dict[str, Callable]

    # When set, enables heterogeneous workers. Each sub_query item can be a dict
    # {"query": str, "args": {...}}. The value of args[task_type_arg_name] selects
    # which worker factory to use from a dict-typed worker_factory.
    task_type_arg_name: Optional[str] = attrib(default=None, kw_only=True)
    # When True, subtasks with multiple "todos" in their args are expanded
    # into one worker per todo (each receiving description + single todo).
    # Can be overridden per task_type when worker_factory is a dict:
    #   {"type_name": {"factory": callable, "expand_todos": True}}
    expand_todos_to_workers: bool = attrib(default=False, kw_only=True)

    # === Aggregation ===
    aggregator_inferencer: Optional[InferencerBase] = attrib(default=None)
    aggregator_prompt_builder: Optional[Callable] = attrib(default=None)

    # === Checkpoint ===
    checkpoint_dir: Optional[str] = attrib(default=None)

    # === Workspace support (opt-in, overrides checkpoint_dir when set) ===
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

    # Suppress WorkGraph's start_nodes requirement at construction time
    # (graph is built dynamically in _infer/_ainfer)
    start_nodes = attrib(factory=list)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.workspace_root is not None:
            from agent_foundation.common.inferencers.inferencer_workspace import (
                InferencerWorkspace,
            )

            self._workspace = InferencerWorkspace(root=self.workspace_root)
            self._workspace.ensure_dirs()
        else:
            self._workspace = None

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

    def _build_diamond_graph(self, sub_queries, inference_config=None, **kwargs):
        """Build the diamond-shaped WorkGraph dynamically from sub-queries.

        Creates N worker nodes (one per sub-query) and optionally an
        aggregation node that collects all worker results.
        """
        # Clear stale graph state from any prior _infer() call
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
            should_expand = self.expand_todos_to_workers
            task_type = None
            factory_entry = None
            if isinstance(self.worker_factory, dict) and self.task_type_arg_name:
                task_type = sq_args.get(self.task_type_arg_name, "__default__")
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
                if factory_entry is None:
                    raise ValueError(
                        f"No worker factory for task type '{task_type}' "
                        f"and no '__default__' fallback"
                    )
                # Support both callable and dict with "factory" key
                factory = factory_entry["factory"] if isinstance(factory_entry, dict) else factory_entry
                worker = factory(sub_query=query_str, index=i)
            else:
                # Homogeneous workers (backward compat)
                worker = self.worker_factory(sub_query=query_str, index=i)

            # Assign child workspace to worker (full composition mode)
            if self._workspace is not None and isinstance(worker, InferencerBase):
                worker_ws = self._workspace.child(f"worker_{i}")
                worker_ws.ensure_dirs()
                worker._workspace = worker_ws
                worker_output_paths.append(worker.resolve_output_path())
            else:
                worker_output_paths.append(None)

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

            def _make_worker_fn(w, q, is_async, manages_resume):
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
                            return cached
                        return await w.ainfer(q, inference_config=inference_config)

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

            node = WorkGraphNode(
                name=f"worker_{i}",
                value=_make_worker_fn(worker, query_str, use_async, _worker_manages_resume),
                result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
                group=worker_group,
                enable_result_save=StepResultSaveOptions.SkipResumable,
                resume_with_saved_results=ResumeMode.SkipResumable,
                worker_manages_resume=_worker_manages_resume,
                retry_on_exceptions=(Exception,),
            )

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

            def _build_agg_input(prompt_builder, worker_results, original_query):
                if prompt_builder is not None:
                    try:
                        return prompt_builder(
                            worker_results,
                            original_query=original_query,
                            worker_output_paths=_captured_paths,
                        )
                    except TypeError:
                        # Builder doesn't accept worker_output_paths
                        return prompt_builder(
                            worker_results, original_query=original_query
                        )
                # Default: join all worker results with path references
                parts = []
                for idx, res in enumerate(worker_results):
                    path_ref = ""
                    if idx < len(_captured_paths) and _captured_paths[idx]:
                        path_ref = (
                            f"\n(Full output at: `{_captured_paths[idx]}`)"
                        )
                    parts.append(f"### Result {idx + 1}\n{res}{path_ref}")
                return "\n\n".join(parts)

            def _make_agg_fn(agg_inf, prompt_builder, original_query, is_async):
                if is_async and hasattr(agg_inf, "ainfer"):

                    async def async_agg_fn(*worker_results, **_kwargs):
                        agg_input = _build_agg_input(
                            prompt_builder, worker_results, original_query
                        )
                        return await agg_inf.ainfer(
                            agg_input, inference_config=inference_config
                        )

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

            agg_node = WorkGraphNode(
                name="aggregator",
                value=_make_agg_fn(
                    self.aggregator_inferencer,
                    self.aggregator_prompt_builder,
                    original_query,
                    use_async,
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

    def _finalize_response(self, result):
        """Copy aggregator output to workspace outputs/ as final deliverable.

        Only runs in workspace mode with output_path set. Idempotent.
        """
        if self._workspace is None or not self.output_path:
            return
        dst = self._workspace.output_path(self.output_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            with open(dst, "w") as f:
                f.write(str(result))
        except OSError as e:
            _logger.warning("Failed to write final output to %s: %s", dst, e)

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

        # Step 1: Breakdown
        raw_output = self.breakdown_inferencer.infer(
            inference_input, inference_config=inference_config
        )

        # Step 2: Parse breakdown output
        if self.breakdown_parser is not None:
            sub_queries = self.breakdown_parser(raw_output)
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
            result = await WorkGraph._arun(self, inference_input, **kwargs)
            if isinstance(result, tuple) and len(result) == 1:
                result = result[0]
            self._finalize_response(result)
            return result

        # Step 1: Breakdown
        if hasattr(self.breakdown_inferencer, "ainfer"):
            raw_output = await self.breakdown_inferencer.ainfer(
                inference_input, inference_config=inference_config
            )
        else:
            raw_output = self.breakdown_inferencer.infer(
                inference_input, inference_config=inference_config
            )

        # Step 2: Parse
        if self.breakdown_parser is not None:
            sub_queries = self.breakdown_parser(raw_output)
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

        self._finalize_response(result)
        return result

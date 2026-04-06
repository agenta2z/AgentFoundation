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
    worker_factory: Callable[..., Any] = attrib(default=None)

    # === Aggregation ===
    aggregator_inferencer: Optional[InferencerBase] = attrib(default=None)
    aggregator_prompt_builder: Optional[Callable] = attrib(default=None)

    # === Checkpoint ===
    checkpoint_dir: Optional[str] = attrib(default=None)

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

    # Suppress WorkGraph's start_nodes requirement at construction time
    # (graph is built dynamically in _infer/_ainfer)
    start_nodes = attrib(factory=list)

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

        worker_nodes = []
        for i, sq in enumerate(sub_queries):
            worker = self.worker_factory(sub_query=sq, index=i)

            # Create a callable that captures this specific sub-query.
            # It intentionally ignores args from WorkGraph (which passes
            # the same args to all start nodes).
            # When the worker supports ainfer() AND the graph is in async mode,
            # returns an async coroutine for true parallel I/O.
            use_async = getattr(self, "use_async", False)

            def _make_worker_fn(w, q, is_async):
                if is_async and hasattr(w, "ainfer"):

                    async def async_worker_fn(*_args, **_kwargs):
                        return await w.ainfer(q, inference_config=inference_config)

                    return async_worker_fn
                else:

                    def worker_fn(*_args, **_kwargs):
                        if hasattr(w, "infer"):
                            return w.infer(q, inference_config=inference_config)
                        return w(q)

                    return worker_fn

            node = WorkGraphNode(
                name=f"worker_{i}",
                value=_make_worker_fn(worker, sq, use_async),
                result_pass_down_mode=ResultPassDownMode.ResultAsFirstArg,
                # Workers handle their own checkpointing internally (e.g., PTI
                # saves plan/implement steps separately). Disabling node-level
                # result save enables recursive resume: BTA always calls the
                # worker, and the worker decides what to skip based on its own
                # internal checkpoints.
                enable_result_save=False,
                resume_with_saved_results=False,
                retry_on_exceptions=(Exception,),
            )

            worker_nodes.append(node)

        agg_node = None
        if self.aggregator_inferencer is not None:
            # Create aggregation node that receives all worker results
            def _build_agg_input(prompt_builder, worker_results, original_query):
                if prompt_builder is not None:
                    return prompt_builder(worker_results, original_query=original_query)
                # Default: join all worker results
                parts = []
                for idx, res in enumerate(worker_results):
                    parts.append(f"### Result {idx + 1}\n{res}")
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
            if self.checkpoint_dir:
                _ext = ".json" if self.checkpoint_mode == "jsonfy" else ".pkl"
                _agg_ckpt = os.path.join(self.checkpoint_dir, "aggregator_result")
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
        if self.checkpoint_dir:
            ext = ".json" if self.checkpoint_mode == "jsonfy" else ".pkl"
            return os.path.join(self.checkpoint_dir, f"{result_id}_result{ext}")
        raise NotImplementedError("checkpoint_dir must be set for result saving")

    def _load_breakdown_checkpoint(self):
        """Load saved breakdown result if resuming and checkpoint exists."""
        if not self.resume_with_saved_results or not self.checkpoint_dir:
            return None
        ckpt = os.path.join(self.checkpoint_dir, "breakdown_result.json")
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
        if not self.checkpoint_dir:
            return
        ckpt = os.path.join(self.checkpoint_dir, "breakdown_result.json")
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
                return result[0]
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
            return result[0]
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
                return result[0]
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

        # Step 3b: Interactive sub-query selection checkpoint
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
            return result[0]

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

        return result

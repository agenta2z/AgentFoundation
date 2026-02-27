"""
Example: Meta Agent Pipeline — end-to-end demonstration.

Shows how to configure a SyntheticDataProvider, run the MetaAgentPipeline,
inspect the SynthesisReport, and serialize the resulting ActionGraph to JSON.

Because this example runs without a real browser agent or action executor,
we use lightweight mock objects that simulate the pipeline's external
dependencies.

Requirements demonstrated: 10.1, 10.3
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agent_foundation.automation.meta_agent import (
    ExecutionTrace,
    MetaAgentPipeline,
    PipelineConfig,
    PipelineResult,
    SynthesisReport,
    TraceStep,
)
from agent_foundation.automation.meta_agent.synthetic_data import (
    SyntheticDataProvider,
)


# ---------------------------------------------------------------------------
# Mock agent and action executor
# ---------------------------------------------------------------------------

class MockAgent:
    """Minimal agent that records a fixed sequence of actions per run.

    A real agent would use an LLM to decide actions.  Here we return a
    canned three-step workflow so the pipeline has something to process.
    """

    def __call__(self, task: str, **kwargs: Any) -> None:
        # In production the agent writes structured logs via SessionLogger.
        # The TraceCollector reads those logs to build ExecutionTrace objects.
        # For this demo the collector is patched to return pre-built traces.
        pass


class MockActionExecutor:
    """Minimal action executor that accepts any action and returns success.

    A real executor (e.g. WebDriver) would drive a browser.
    """

    def execute(self, action_type: str, target: Any = None, **kwargs: Any) -> dict:
        return {"success": True}


# ---------------------------------------------------------------------------
# Helper: build a fake trace (simulates what TraceCollector would produce)
# ---------------------------------------------------------------------------

def _build_demo_trace(trace_id: str, search_query: str) -> ExecutionTrace:
    """Create a simple three-step trace that mimics a Google-search workflow."""
    return ExecutionTrace(
        trace_id=trace_id,
        task_description="Search Google for a query",
        success=True,
        steps=[
            TraceStep(action_type="visit_url", target="https://www.google.com"),
            TraceStep(
                action_type="input_text",
                target="//textarea[@title='Search']",
                args={"text": search_query},
            ),
            TraceStep(
                action_type="click",
                target="//input[@value='Google Search']",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Main example
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Configure synthetic data ──────────────────────────────────────────
    #    The SyntheticDataProvider generates varied inputs so each agent run
    #    exercises slightly different data paths.
    synthetic_data = SyntheticDataProvider(
        parameter_schema={"search_query": "str"},
    )

    # 2. Pipeline configuration ────────────────────────────────────────────
    config = PipelineConfig(
        run_count=3,                    # number of agent runs
        validate=False,                 # skip validation (no real browser)
        synthesis_strategy="rule_based",
        evaluation_strategy="exception_only",
        min_success_traces=1,
    )

    # 3. Create the pipeline ───────────────────────────────────────────────
    agent = MockAgent()
    executor = MockActionExecutor()

    pipeline = MetaAgentPipeline(
        agent=agent,
        action_executor=executor,
        config=config,
        synthetic_data_provider=synthetic_data,
    )

    # 4. Run the pipeline ──────────────────────────────────────────────────
    #    In a real scenario pipeline.run() drives the agent N times, collects
    #    traces, normalizes, aligns, extracts patterns, and synthesizes an
    #    ActionGraph.  Here we patch the collector so the demo works without
    #    a real browser.
    from unittest.mock import patch as mock_patch

    from agent_foundation.automation.meta_agent.collector import TraceCollector

    def _mock_collect(
        self: TraceCollector,
        task_description: str,
        run_count: int,
        input_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ExecutionTrace]:
        """Return pre-built traces instead of running a real agent."""
        if run_count < 1:
            raise ValueError("run_count must be >= 1")
        queries = ["python tutorial", "rust async", "typescript generics"]
        return [
            _build_demo_trace(f"trace_{i}", queries[i % len(queries)])
            for i in range(run_count)
        ]

    with mock_patch.object(TraceCollector, "collect", _mock_collect):
        result: PipelineResult = pipeline.run(
            task_description="Search Google for a query",
        )

    # 5. Inspect the result ────────────────────────────────────────────────
    if result.failed_stage:
        print(f"Pipeline failed at stage: {result.failed_stage}")
        print(f"Error: {result.error}")
        return

    print("Pipeline completed successfully!\n")

    # 5a. Traces collected
    print(f"Traces collected: {len(result.traces)}")
    for trace in result.traces:
        status = "OK" if trace.success else "FAILED"
        print(f"  {trace.trace_id}: {len(trace.steps)} steps [{status}]")

    # 5b. Evaluation results
    print(f"\nEvaluation results: {len(result.evaluation_results)}")
    for er in result.evaluation_results:
        print(f"  {er.trace_id}: passed={er.passed}  score={er.score}")

    # 5c. Synthesis report
    report: Optional[SynthesisReport] = result.synthesis_report
    if report:
        print("\nSynthesis Report:")
        print(f"  Total steps:          {report.total_steps}")
        print(f"  Deterministic:        {report.deterministic_count}")
        print(f"  Parameterizable:      {report.parameterizable_count}")
        print(f"  Agent nodes:          {report.agent_node_count}")
        print(f"  Optional:             {report.optional_count}")
        print(f"  User input gates:     {report.user_input_boundary_count}")
        print(f"  Branches:             {report.branch_count}")
        print(f"  Loops:                {report.loop_count}")
        print(f"  Strategy:             {report.synthesis_strategy}")
        if report.template_variables:
            print(f"  Template variables:   {report.template_variables}")
        if report.warnings:
            print(f"  Warnings:             {report.warnings}")

    # 6. Serialize the ActionGraph to JSON ─────────────────────────────────
    graph = result.graph
    if graph is not None:
        graph_json = graph.serialize(output_format="json")
        parsed = json.loads(graph_json)
        print("\nActionGraph JSON (pretty-printed):")
        print(json.dumps(parsed, indent=2)[:2000])  # truncate for readability

        # The report is also serializable
        if report:
            print("\nSynthesisReport JSON:")
            print(json.dumps(report.to_dict(), indent=2))
    else:
        print("\nNo ActionGraph was produced.")


if __name__ == "__main__":
    main()

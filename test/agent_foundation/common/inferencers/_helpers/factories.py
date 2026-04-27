"""Per-inferencer test factories — single source of truth for test wiring.

Without these, every Layer 1 test file would reinvent ~30 lines of factory
boilerplate. With them, tests focus on assertions.

Pattern matches test_breakdown_then_aggregate.py and test_multi_flow_inferencer.py
plus extends with PTI / Dual / LWI factories.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
    WorkflowStepConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
    PlanThenImplementInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase

from .mock_inferencer import MockInferencer, SequentialMockInferencer


# ---------------------------------------------------------------------------
# Mock factories (convenience wrappers)
# ---------------------------------------------------------------------------

def _make_mock_inferencer(
    response: Union[str, Callable, List, None] = "mock response",
) -> MockInferencer:
    """Static / callable / list response."""
    return MockInferencer(response=response)


def _make_sequential_mock(responses: List) -> SequentialMockInferencer:
    """Different response per call."""
    return SequentialMockInferencer(responses=responses)


# ---------------------------------------------------------------------------
# Flow inferencer factories — provide sensible test defaults
# ---------------------------------------------------------------------------

def _make_lwi(
    step_configs: Optional[List[WorkflowStepConfig]] = None,
    response_builder: Optional[Callable] = None,
    workspace_path: Optional[str] = None,
    **kwargs,
) -> LinearWorkflowInferencer:
    """Standard LWI with defaults useful in tests."""
    return LinearWorkflowInferencer(
        step_configs=step_configs or [],
        response_builder=response_builder,
        workspace_path=workspace_path,
        **kwargs,
    )


def _make_dual(
    base_response: Union[str, Callable, List] = "base proposal",
    review_response: Union[str, Callable, List] = "## Review\nSeverity: COSMETIC\nApproved: true",
    fixer_response: Union[str, Callable, List, None] = None,
    consensus_config: Optional[ConsensusConfig] = None,
    base_inferencer: Optional[InferencerBase] = None,
    review_inferencer: Optional[InferencerBase] = None,
    fixer_inferencer: Optional[InferencerBase] = None,
    **kwargs,
) -> DualInferencer:
    """DualInferencer with mocked children.

    Three ways to provide children:
    - As response strings (default behavior — wrapped in MockInferencer/SequentialMock)
    - As pre-built InferencerBase instances (override the corresponding param)
    """
    # Resolve base_inferencer
    if base_inferencer is None:
        base_inferencer = (
            _make_sequential_mock(base_response)
            if isinstance(base_response, list)
            else _make_mock_inferencer(base_response)
        )
    # Resolve review_inferencer
    if review_inferencer is None:
        review_inferencer = (
            _make_sequential_mock(review_response)
            if isinstance(review_response, list)
            else _make_mock_inferencer(review_response)
        )
    # Resolve fixer_inferencer (optional — None means 2-agent mode)
    if fixer_inferencer is None and fixer_response is not None:
        fixer_inferencer = (
            _make_sequential_mock(fixer_response)
            if isinstance(fixer_response, list)
            else _make_mock_inferencer(fixer_response)
        )

    return DualInferencer(
        base_inferencer=base_inferencer,
        review_inferencer=review_inferencer,
        fixer_inferencer=fixer_inferencer,
        consensus_config=consensus_config or ConsensusConfig(),
        **kwargs,
    )


def _make_pti(
    planner_inferencer: Optional[InferencerBase] = None,
    executor_inferencer: Optional[InferencerBase] = None,
    analyzer_inferencer: Optional[InferencerBase] = None,
    planner_response: Union[str, Callable, List] = "## Plan\n1. Step one\n2. Step two",
    executor_response: Union[str, Callable, List] = "Implementation complete.",
    analyzer_response: Optional[Union[str, Callable, List]] = None,
    **kwargs,
) -> PlanThenImplementInferencer:
    """PTI with caller-provided or auto-generated mock children."""
    if planner_inferencer is None:
        planner_inferencer = (
            _make_sequential_mock(planner_response)
            if isinstance(planner_response, list)
            else _make_mock_inferencer(planner_response)
        )
    if executor_inferencer is None:
        executor_inferencer = (
            _make_sequential_mock(executor_response)
            if isinstance(executor_response, list)
            else _make_mock_inferencer(executor_response)
        )
    if analyzer_inferencer is None and analyzer_response is not None:
        analyzer_inferencer = (
            _make_sequential_mock(analyzer_response)
            if isinstance(analyzer_response, list)
            else _make_mock_inferencer(analyzer_response)
        )

    return PlanThenImplementInferencer(
        planner_inferencer=planner_inferencer,
        executor_inferencer=executor_inferencer,
        analyzer_inferencer=analyzer_inferencer,
        **kwargs,
    )


def _make_bta_worker_factory(
    worker_responses: Optional[List] = None,
    response_fn: Optional[Callable] = None,
):
    """Build a worker_factory closure that creates MockInferencers.

    Two modes:
    - worker_responses=[r0, r1, ...]: index-keyed responses
    - response_fn=lambda(sub_query, index): str: dynamic responses
    """
    created_workers = []

    def factory(sub_query, index):
        if response_fn is not None:
            resp = response_fn(sub_query, index)
        elif worker_responses is not None:
            resp = (
                worker_responses[index]
                if index < len(worker_responses)
                else f"result_for_{sub_query}"
            )
        else:
            resp = f"result_for_{sub_query}"
        worker = MockInferencer(response=resp)
        created_workers.append(worker)
        return worker

    factory.created_workers = created_workers
    return factory


def _make_bta(
    breakdown_response: Union[str, Callable, List, None] = "1. Q1\n2. Q2\n3. Q3",
    worker_responses: Optional[List] = None,
    aggregator_response: Union[str, Callable, List] = "aggregated",
    breakdown_inferencer: Optional[InferencerBase] = None,
    aggregator_inferencer: Optional[InferencerBase] = None,
    worker_factory: Optional[Callable] = None,
    **kwargs,
) -> BreakdownThenAggregateInferencer:
    """BTA wired with mock breakdown + worker_factory + aggregator."""
    if breakdown_inferencer is None and breakdown_response is not None:
        breakdown_inferencer = _make_mock_inferencer(breakdown_response)
    if aggregator_inferencer is None:
        aggregator_inferencer = _make_mock_inferencer(aggregator_response)
    if worker_factory is None:
        worker_factory = _make_bta_worker_factory(worker_responses=worker_responses)

    return BreakdownThenAggregateInferencer(
        breakdown_inferencer=breakdown_inferencer,
        worker_factory=worker_factory,
        aggregator_inferencer=aggregator_inferencer,
        **kwargs,
    )


def _make_mfi(
    flow_configs: Optional[List[dict]] = None,
    aggregator_response: Optional[Union[str, Callable, List]] = "aggregated flows",
    aggregator_inferencer: Optional[InferencerBase] = None,
    **kwargs,
) -> MultiFlowInferencer:
    """MFI with caller-provided flow_configs."""
    if aggregator_inferencer is None and aggregator_response is not None:
        aggregator_inferencer = _make_mock_inferencer(aggregator_response)

    return MultiFlowInferencer(
        flow_configs=flow_configs or [],
        aggregator_inferencer=aggregator_inferencer,
        **kwargs,
    )

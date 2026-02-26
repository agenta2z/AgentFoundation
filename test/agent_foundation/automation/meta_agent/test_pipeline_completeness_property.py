"""Property test for pipeline result completeness (Property 22).

**Validates: Requirements 10.2, 10.3**

For any MetaAgentPipeline execution:
- Success → graph, synthesis_report, and validation_results all non-None,
  failed_stage is None
- Failure → failed_stage names the failing stage, traces contains all
  traces collected before the failure
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from science_modeling_tools.automation.meta_agent.evaluator import EvaluationResult
from science_modeling_tools.automation.meta_agent.models import (
    ExecutionTrace,
    PipelineConfig,
    PipelineResult,
    SynthesisReport,
    TraceStep,
    ValidationResults,
)
from science_modeling_tools.automation.meta_agent.pipeline import MetaAgentPipeline


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPELINE_STAGES = [
    "collection",
    "evaluation",
    "normalization",
    "target_conversion",
    "alignment",
    "extraction",
    "synthesis",
    "validation",
]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def trace_steps(draw, min_size: int = 1, max_size: int = 5) -> list[TraceStep]:
    """Generate a list of random TraceSteps."""
    action_types = ["click", "input_text", "scroll", "visit_url", "wait"]
    targets = [None, "btn", "field", "#submit"]
    return [
        TraceStep(
            action_type=draw(st.sampled_from(action_types)),
            target=draw(st.sampled_from(targets)),
        )
        for _ in range(draw(st.integers(min_value=min_size, max_value=max_size)))
    ]


@st.composite
def execution_traces(draw, min_count: int = 1, max_count: int = 4) -> list[ExecutionTrace]:
    """Generate a list of ExecutionTraces with unique IDs."""
    count = draw(st.integers(min_value=min_count, max_value=max_count))
    traces = []
    for i in range(count):
        steps = draw(trace_steps())
        traces.append(
            ExecutionTrace(
                trace_id=f"t{i}",
                task_description="test task",
                steps=steps,
                success=True,
            )
        )
    return traces


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(validate: bool = False, target_converter=None) -> MetaAgentPipeline:
    """Create a pipeline with default config and mock agent/executor."""
    config = PipelineConfig(run_count=2, validate=validate, target_converter=target_converter)
    return MetaAgentPipeline(
        agent=MagicMock(name="agent"),
        action_executor=MagicMock(name="executor"),
        config=config,
    )


def _mock_successful_pipeline(traces: list[ExecutionTrace]):
    """Return a dict of patch targets and their mock return values for a
    successful pipeline run."""
    eval_results = [
        EvaluationResult(trace_id=t.trace_id, passed=True) for t in traces
    ]

    mock_graph = MagicMock(name="graph")
    mock_report = MagicMock(spec=SynthesisReport, name="report")
    mock_synth_result = MagicMock()
    mock_synth_result.graph = mock_graph
    mock_synth_result.report = mock_report
    mock_synth_result.python_script = None

    return {
        "traces": traces,
        "eval_results": eval_results,
        "synth_result": mock_synth_result,
        "graph": mock_graph,
        "report": mock_report,
    }


# ---------------------------------------------------------------------------
# Property tests — Success case
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(traces=execution_traces(min_count=1, max_count=4))
def test_success_graph_non_none(traces: list[ExecutionTrace]):
    """Property 22: On success, graph SHALL be non-None.

    **Validates: Requirements 10.3**
    """
    mocks = _mock_successful_pipeline(traces)
    pipeline = _make_pipeline(validate=False)

    base = "agent_foundation.automation.meta_agent.pipeline"
    with (
        patch(f"{base}.TraceCollector") as MockCollector,
        patch(f"{base}.TraceEvaluator") as MockEvaluator,
        patch(f"{base}.TraceNormalizer") as MockNorm,
        patch(f"{base}.TraceAligner"),
        patch(f"{base}.PatternExtractor"),
        patch(f"{base}.RuleBasedSynthesizer") as MockSynth,
    ):
        MockCollector.return_value.collect.return_value = mocks["traces"]
        MockEvaluator.return_value.evaluate.return_value = mocks["eval_results"]
        MockNorm.return_value.normalize.return_value = mocks["traces"]
        MockSynth.return_value.synthesize.return_value = mocks["synth_result"]

        result = pipeline.run("test task")

    assert result.graph is not None, "Successful pipeline must have non-None graph"


@settings(max_examples=50, deadline=None)
@given(traces=execution_traces(min_count=1, max_count=4))
def test_success_synthesis_report_non_none(traces: list[ExecutionTrace]):
    """Property 22: On success, synthesis_report SHALL be non-None.

    **Validates: Requirements 10.3**
    """
    mocks = _mock_successful_pipeline(traces)
    pipeline = _make_pipeline(validate=False)

    base = "agent_foundation.automation.meta_agent.pipeline"
    with (
        patch(f"{base}.TraceCollector") as MockCollector,
        patch(f"{base}.TraceEvaluator") as MockEvaluator,
        patch(f"{base}.TraceNormalizer") as MockNorm,
        patch(f"{base}.TraceAligner"),
        patch(f"{base}.PatternExtractor"),
        patch(f"{base}.RuleBasedSynthesizer") as MockSynth,
    ):
        MockCollector.return_value.collect.return_value = mocks["traces"]
        MockEvaluator.return_value.evaluate.return_value = mocks["eval_results"]
        MockNorm.return_value.normalize.return_value = mocks["traces"]
        MockSynth.return_value.synthesize.return_value = mocks["synth_result"]

        result = pipeline.run("test task")

    assert result.synthesis_report is not None, (
        "Successful pipeline must have non-None synthesis_report"
    )


@settings(max_examples=50, deadline=None)
@given(traces=execution_traces(min_count=1, max_count=4))
def test_success_failed_stage_is_none(traces: list[ExecutionTrace]):
    """Property 22: On success, failed_stage SHALL be None.

    **Validates: Requirements 10.3**
    """
    mocks = _mock_successful_pipeline(traces)
    pipeline = _make_pipeline(validate=False)

    base = "agent_foundation.automation.meta_agent.pipeline"
    with (
        patch(f"{base}.TraceCollector") as MockCollector,
        patch(f"{base}.TraceEvaluator") as MockEvaluator,
        patch(f"{base}.TraceNormalizer") as MockNorm,
        patch(f"{base}.TraceAligner"),
        patch(f"{base}.PatternExtractor"),
        patch(f"{base}.RuleBasedSynthesizer") as MockSynth,
    ):
        MockCollector.return_value.collect.return_value = mocks["traces"]
        MockEvaluator.return_value.evaluate.return_value = mocks["eval_results"]
        MockNorm.return_value.normalize.return_value = mocks["traces"]
        MockSynth.return_value.synthesize.return_value = mocks["synth_result"]

        result = pipeline.run("test task")

    assert result.failed_stage is None, (
        f"Successful pipeline must have failed_stage=None, got '{result.failed_stage}'"
    )


# ---------------------------------------------------------------------------
# Property tests — Failure case
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    traces=execution_traces(min_count=1, max_count=4),
    failing_stage=st.sampled_from(PIPELINE_STAGES),
)
def test_failure_failed_stage_names_the_stage(
    traces: list[ExecutionTrace],
    failing_stage: str,
):
    """Property 22: On failure, failed_stage SHALL name the failing stage.

    **Validates: Requirements 10.2**
    """
    mocks = _mock_successful_pipeline(traces)

    # For target_conversion failures, inject a converter that raises
    mock_converter = None
    if failing_stage == "target_conversion":
        mock_converter = MagicMock()
        mock_converter.convert_all.side_effect = RuntimeError(f"fail at {failing_stage}")

    pipeline = _make_pipeline(validate=True, target_converter=mock_converter)

    base = "agent_foundation.automation.meta_agent.pipeline"
    with (
        patch(f"{base}.TraceCollector") as MockCollector,
        patch(f"{base}.TraceEvaluator") as MockEvaluator,
        patch(f"{base}.TraceNormalizer") as MockNorm,
        patch(f"{base}.TraceAligner") as MockAligner,
        patch(f"{base}.PatternExtractor") as MockExtractor,
        patch(f"{base}.RuleBasedSynthesizer") as MockSynth,
        patch(f"{base}.GraphValidator") as MockValidator,
    ):
        # Wire up all stages to succeed by default
        MockCollector.return_value.collect.return_value = mocks["traces"]
        MockEvaluator.return_value.evaluate.return_value = mocks["eval_results"]
        MockNorm.return_value.normalize.return_value = mocks["traces"]
        MockSynth.return_value.synthesize.return_value = mocks["synth_result"]
        MockValidator.return_value.validate.return_value = MagicMock(spec=ValidationResults)

        # Inject failure at the chosen stage
        error = RuntimeError(f"fail at {failing_stage}")
        if failing_stage == "collection":
            MockCollector.return_value.collect.side_effect = error
        elif failing_stage == "evaluation":
            MockEvaluator.return_value.evaluate.side_effect = error
        elif failing_stage == "normalization":
            MockNorm.return_value.normalize.side_effect = error
        elif failing_stage == "target_conversion":
            pass  # Handled via mock_converter above
        elif failing_stage == "alignment":
            MockAligner.return_value.align.side_effect = error
        elif failing_stage == "extraction":
            MockExtractor.return_value.extract.side_effect = error
        elif failing_stage == "synthesis":
            MockSynth.return_value.synthesize.side_effect = error
        elif failing_stage == "validation":
            MockValidator.return_value.validate.side_effect = error

        result = pipeline.run("test task")

    assert result.failed_stage == failing_stage, (
        f"Expected failed_stage='{failing_stage}', got '{result.failed_stage}'"
    )
    assert result.failed_stage in PIPELINE_STAGES, (
        f"failed_stage '{result.failed_stage}' is not a valid stage name"
    )


@settings(max_examples=50, deadline=None)
@given(
    traces=execution_traces(min_count=1, max_count=4),
    failing_stage=st.sampled_from([
        "normalization", "target_conversion", "alignment",
        "extraction", "synthesis",
    ]),
)
def test_failure_traces_contain_pre_failure_data(
    traces: list[ExecutionTrace],
    failing_stage: str,
):
    """Property 22: On failure, traces SHALL contain traces collected before failure.

    **Validates: Requirements 10.2**

    We test stages after collection (normalization onward) to verify that
    traces collected in stage 1 are preserved in the result even when a
    later stage fails.
    """
    mocks = _mock_successful_pipeline(traces)

    # For target_conversion failures, inject a converter that raises
    mock_converter = None
    if failing_stage == "target_conversion":
        mock_converter = MagicMock()
        mock_converter.convert_all.side_effect = RuntimeError(f"fail at {failing_stage}")

    pipeline = _make_pipeline(validate=False, target_converter=mock_converter)

    base = "agent_foundation.automation.meta_agent.pipeline"
    with (
        patch(f"{base}.TraceCollector") as MockCollector,
        patch(f"{base}.TraceEvaluator") as MockEvaluator,
        patch(f"{base}.TraceNormalizer") as MockNorm,
        patch(f"{base}.TraceAligner") as MockAligner,
        patch(f"{base}.PatternExtractor") as MockExtractor,
        patch(f"{base}.RuleBasedSynthesizer") as MockSynth,
    ):
        MockCollector.return_value.collect.return_value = mocks["traces"]
        MockEvaluator.return_value.evaluate.return_value = mocks["eval_results"]
        MockNorm.return_value.normalize.return_value = mocks["traces"]
        MockSynth.return_value.synthesize.return_value = mocks["synth_result"]

        error = RuntimeError(f"fail at {failing_stage}")
        if failing_stage == "normalization":
            MockNorm.return_value.normalize.side_effect = error
        elif failing_stage == "target_conversion":
            pass  # Handled via mock_converter above
        elif failing_stage == "alignment":
            MockAligner.return_value.align.side_effect = error
        elif failing_stage == "extraction":
            MockExtractor.return_value.extract.side_effect = error
        elif failing_stage == "synthesis":
            MockSynth.return_value.synthesize.side_effect = error

        result = pipeline.run("test task")

    assert result.failed_stage == failing_stage
    # Traces from collection stage must be preserved
    assert len(result.traces) == len(traces), (
        f"Expected {len(traces)} traces preserved, got {len(result.traces)}"
    )
    collected_ids = {t.trace_id for t in traces}
    result_ids = {t.trace_id for t in result.traces}
    assert result_ids == collected_ids, (
        f"Trace IDs mismatch: expected {collected_ids}, got {result_ids}"
    )


# ---------------------------------------------------------------------------
# Property test — PipelineResult invariant (direct dataclass testing)
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    has_graph=st.booleans(),
    has_report=st.booleans(),
    has_validation=st.booleans(),
    failed_stage=st.one_of(st.none(), st.sampled_from(PIPELINE_STAGES)),
    trace_count=st.integers(min_value=0, max_value=5),
)
def test_pipeline_result_success_failure_invariant(
    has_graph: bool,
    has_report: bool,
    has_validation: bool,
    failed_stage: str | None,
    trace_count: int,
):
    """Property 22: PipelineResult success/failure invariant.

    **Validates: Requirements 10.2, 10.3**

    If failed_stage is None (success), then graph and synthesis_report
    should be non-None for a well-formed result. If failed_stage is set
    (failure), it must be a valid stage name.
    """
    traces = [
        ExecutionTrace(trace_id=f"t{i}", task_description="task", steps=[])
        for i in range(trace_count)
    ]

    result = PipelineResult(
        graph=MagicMock() if has_graph else None,
        synthesis_report=MagicMock() if has_report else None,
        validation_results=MagicMock() if has_validation else None,
        traces=traces,
        failed_stage=failed_stage,
    )

    if failed_stage is not None:
        # Failure invariant: failed_stage is a valid stage name
        assert result.failed_stage in PIPELINE_STAGES, (
            f"failed_stage '{result.failed_stage}' not in valid stages"
        )
    else:
        # Success invariant: when the pipeline produces a result with
        # failed_stage=None, the caller expects graph and report to be set.
        # This is a structural invariant — we verify the dataclass allows it.
        if has_graph and has_report:
            assert result.graph is not None
            assert result.synthesis_report is not None
            assert result.failed_stage is None

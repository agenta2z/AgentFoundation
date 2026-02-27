"""
Tests verifying that MetaAgentPipeline is now a proper Workflow subclass
and that the Workflow-based mechanisms (steps, state, loop, abort) work
correctly through the pipeline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rich_python_utils.common_objects.workflow.workflow import Workflow
from rich_python_utils.common_objects.workflow.common.exceptions import WorkflowAborted
from rich_python_utils.common_objects.workflow.common.result_pass_down_mode import (
    ResultPassDownMode,
)

from agent_foundation.automation.meta_agent.errors import (
    InsufficientSuccessTracesError,
    PipelineAborted,
)
from agent_foundation.automation.meta_agent.evaluator import EvaluationResult
from agent_foundation.automation.meta_agent.models import (
    ExecutionTrace,
    PipelineConfig,
    PipelineResult,
    TraceStep,
)
from agent_foundation.automation.meta_agent.pipeline import (
    MetaAgentPipeline,
    _StepWrapper,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trace(trace_id="t1", success=True):
    return ExecutionTrace(
        trace_id=trace_id,
        task_description="test task",
        steps=[TraceStep(action_type="click", target="btn")],
        success=success,
    )


def _make_pipeline(**overrides):
    defaults = dict(
        agent=MagicMock(name="agent"),
        action_executor=MagicMock(name="action_executor"),
    )
    defaults.update(overrides)
    return MetaAgentPipeline(**defaults)


BASE = "agent_foundation.automation.meta_agent.pipeline"


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

class TestPipelineIsWorkflow:
    """Verify structural properties of the Workflow subclass."""

    def test_isinstance_workflow(self):
        pipeline = _make_pipeline()
        assert isinstance(pipeline, Workflow)

    def test_has_8_steps(self):
        pipeline = _make_pipeline()
        assert len(pipeline._steps) == 8

    def test_step_names(self):
        pipeline = _make_pipeline()
        names = [getattr(s, 'name', None) for s in pipeline._steps]
        assert names == [
            "collection", "evaluation", "normalization",
            "target_conversion", "alignment", "extraction",
            "synthesis", "validation",
        ]

    def test_steps_are_step_wrappers(self):
        pipeline = _make_pipeline()
        for step in pipeline._steps:
            assert isinstance(step, _StepWrapper)

    def test_result_pass_down_mode_is_no_pass_down(self):
        pipeline = _make_pipeline()
        assert pipeline.result_pass_down_mode == ResultPassDownMode.NoPassDown

    def test_enable_result_save_is_false(self):
        pipeline = _make_pipeline()
        assert pipeline.enable_result_save is False

    def test_unpack_single_result_is_false(self):
        pipeline = _make_pipeline()
        assert pipeline.unpack_single_result is False


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------

class TestPipelineState:
    """Verify flow state initialization and shape."""

    def test_init_state_returns_expected_keys(self):
        pipeline = _make_pipeline()
        state = pipeline._init_state()
        expected_keys = {
            'traces', 'evaluation_results', 'filtered_traces',
            'normalized', 'aligned', 'patterns',
            'synthesis_result', 'validation_results', 'python_script',
            '_config_generate_script',
        }
        assert set(state.keys()) == expected_keys

    def test_init_state_traces_empty(self):
        pipeline = _make_pipeline()
        state = pipeline._init_state()
        assert state['traces'] == []
        assert state['evaluation_results'] == []
        assert state['filtered_traces'] == []

    def test_pre_populated_state_returned(self):
        pipeline = _make_pipeline()
        pre = {'traces': [1, 2, 3], 'evaluation_results': [4, 5, 6]}
        pipeline._pre_populated_state = pre
        assert pipeline._init_state() is pre
        pipeline._pre_populated_state = None


# ---------------------------------------------------------------------------
# Loop mechanism tests
# ---------------------------------------------------------------------------

class TestEvaluationLoop:
    """Verify the retry loop through Workflow's loop mechanism."""

    def test_evaluation_step_has_loop_back_to_collection(self):
        pipeline = _make_pipeline()
        eval_step = pipeline._steps[1]
        assert getattr(eval_step, 'loop_back_to') == "collection"

    def test_evaluation_step_max_loop_iterations_from_config(self):
        config = PipelineConfig(max_retry_rounds=5)
        pipeline = _make_pipeline(config=config)
        eval_step = pipeline._steps[1]
        assert getattr(eval_step, 'max_loop_iterations') == 5

    def test_insufficient_traces_condition_true(self):
        pipeline = _make_pipeline(
            config=PipelineConfig(min_success_traces=3),
        )
        state = {'filtered_traces': [1, 2]}
        assert pipeline._insufficient_traces(state, None) is True

    def test_insufficient_traces_condition_false(self):
        pipeline = _make_pipeline(
            config=PipelineConfig(min_success_traces=2),
        )
        state = {'filtered_traces': [1, 2]}
        assert pipeline._insufficient_traces(state, None) is False


# ---------------------------------------------------------------------------
# Abort / error handling tests
# ---------------------------------------------------------------------------

class TestAbortHandling:
    """Verify abort and error handling through Workflow mechanisms."""

    @patch(f"{BASE}.TraceCollector")
    def test_pipeline_aborted_from_hook_returns_result(self, MockCollector):
        """When stage_hook raises PipelineAborted, result has _aborted suffix."""
        traces = [_make_trace("t1")]
        MockCollector.return_value.collect.return_value = traces

        def aborting_hook(stage, data):
            if stage == "collection":
                raise PipelineAborted("collection", "user abort")

        pipeline = _make_pipeline(stage_hook=aborting_hook)
        result = pipeline.run("test task")

        assert isinstance(result, PipelineResult)
        assert result.failed_stage == "collection_aborted"
        assert "user abort" in result.error

    def test_handle_abort_with_partial_result(self):
        pipeline = _make_pipeline()
        partial = PipelineResult(error="boom", failed_stage="synthesis")
        exc = WorkflowAborted(
            message="boom", step_name="synthesis", partial_result=partial,
        )
        result = pipeline._handle_abort(exc, None, {})
        assert result is partial

    def test_handle_abort_pipeline_aborted_without_partial(self):
        pipeline = _make_pipeline()
        exc = PipelineAborted("extraction", "stopped")
        result = pipeline._handle_abort(exc, None, {'traces': [1], 'evaluation_results': [2]})
        assert isinstance(result, PipelineResult)
        assert result.failed_stage == "extraction_aborted"

    def test_handle_abort_generic_workflow_aborted(self):
        pipeline = _make_pipeline()
        exc = WorkflowAborted(message="error", step_name="alignment")
        result = pipeline._handle_abort(exc, None, {'traces': [], 'evaluation_results': []})
        assert isinstance(result, PipelineResult)
        assert result.failed_stage == "alignment"


# ---------------------------------------------------------------------------
# Pipeline-as-Workflow integration test
# ---------------------------------------------------------------------------

class TestWorkflowIntegration:
    """End-to-end test through Workflow._run() machinery."""

    @patch(f"{BASE}.PatternExtractor")
    @patch(f"{BASE}.TraceAligner")
    @patch(f"{BASE}.TraceNormalizer")
    @patch(f"{BASE}.TraceEvaluator")
    @patch(f"{BASE}.TraceCollector")
    @patch(f"{BASE}.RuleBasedSynthesizer")
    def test_state_accumulated_through_run(
        self, MockSynth, MockCollector, MockEval, MockNorm, MockAlign, MockExt,
    ):
        """Verify that _state is populated after a successful run."""
        traces = [_make_trace("t1")]
        MockCollector.return_value.collect.return_value = traces
        MockEval.return_value.evaluate.return_value = [
            EvaluationResult(trace_id="t1", passed=True),
        ]
        MockNorm.return_value.normalize.return_value = traces

        mock_synth_result = MagicMock()
        mock_synth_result.graph = MagicMock()
        mock_synth_result.report = MagicMock()
        mock_synth_result.python_script = None
        MockSynth.return_value.synthesize.return_value = mock_synth_result

        config = PipelineConfig(run_count=1, validate=False)
        pipeline = _make_pipeline(config=config)
        result = pipeline.run("test task")

        assert result.failed_stage is None
        # State should have been accumulated
        state = pipeline._state
        assert len(state['traces']) == 1
        assert len(state['evaluation_results']) == 1
        assert state['synthesis_result'] is mock_synth_result

    @patch(f"{BASE}.TraceEvaluator")
    @patch(f"{BASE}.TraceCollector")
    def test_pipeline_aborted_is_workflow_aborted(self, MockCollector, MockEval):
        """PipelineAborted inherits from WorkflowAborted."""
        exc = PipelineAborted("test_stage")
        assert isinstance(exc, WorkflowAborted)
        assert exc.step_name == "test_stage"
        assert exc.step_index is None

    def test_step_error_handlers_set(self):
        """Every step should have an error_handler attribute."""
        pipeline = _make_pipeline()
        for step in pipeline._steps:
            assert hasattr(step, 'error_handler')
            assert step.error_handler is not None

    def test_step_update_state_set(self):
        """Every step should have an update_state attribute."""
        pipeline = _make_pipeline()
        for step in pipeline._steps:
            assert hasattr(step, 'update_state')
            assert step.update_state is not None

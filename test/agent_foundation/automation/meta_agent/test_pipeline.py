"""
Unit tests for MetaAgentPipeline with configurable strategy support.

Tests cover:
- Constructor validation (inferencer requirements for LLM/HYBRID/LLM_JUDGE)
- Full pipeline run with mocked components
- Evaluation stage filtering traces before normalization
- InsufficientSuccessTracesError when too few traces pass
- Stage failure produces partial results with correct failed_stage
- PipelineResult includes evaluation_results and python_script
- Synthesizer subclass selection based on config.synthesis_strategy
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_foundation.automation.meta_agent.errors import (
    InsufficientSuccessTracesError,
)
from agent_foundation.automation.meta_agent.evaluator import (
    EvaluationResult,
    EvaluationRule,
)
from agent_foundation.automation.meta_agent.models import (
    ExecutionTrace,
    PipelineConfig,
    PipelineResult,
    TraceStep,
)
from agent_foundation.automation.meta_agent.pipeline import MetaAgentPipeline
from agent_foundation.automation.meta_agent.synthesizer import (
    SynthesisStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trace(trace_id: str = "t1", success: bool = True) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id,
        task_description="test task",
        steps=[
            TraceStep(action_type="click", target="btn"),
        ],
        success=success,
    )


def _make_agent() -> MagicMock:
    return MagicMock(name="agent")


def _make_executor() -> MagicMock:
    return MagicMock(name="action_executor")


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    """Validate that the constructor enforces inferencer requirements."""

    def test_llm_synthesis_without_inferencer_raises(self):
        config = PipelineConfig(synthesis_strategy="llm")
        with pytest.raises(ValueError, match="llm synthesis strategy"):
            MetaAgentPipeline(
                agent=_make_agent(),
                action_executor=_make_executor(),
                config=config,
            )

    def test_hybrid_synthesis_without_inferencer_raises(self):
        config = PipelineConfig(synthesis_strategy="hybrid")
        with pytest.raises(ValueError, match="hybrid synthesis strategy"):
            MetaAgentPipeline(
                agent=_make_agent(),
                action_executor=_make_executor(),
                config=config,
            )

    def test_llm_judge_evaluation_without_inferencer_raises(self):
        config = PipelineConfig(evaluation_strategy="llm_judge")
        with pytest.raises(ValueError, match="LLM_JUDGE evaluation strategy"):
            MetaAgentPipeline(
                agent=_make_agent(),
                action_executor=_make_executor(),
                config=config,
            )

    def test_rule_based_defaults_ok(self):
        """Default config (rule_based / exception_only) needs no inferencer."""
        pipeline = MetaAgentPipeline(
            agent=_make_agent(),
            action_executor=_make_executor(),
        )
        assert pipeline._synthesis_strategy == SynthesisStrategy.RULE_BASED

    def test_llm_synthesis_with_inferencer_ok(self):
        config = PipelineConfig(synthesis_strategy="llm")
        pipeline = MetaAgentPipeline(
            agent=_make_agent(),
            action_executor=_make_executor(),
            config=config,
            inferencer=MagicMock(name="inferencer"),
        )
        assert pipeline._synthesis_strategy == SynthesisStrategy.LLM

    def test_unrecognised_synthesis_strategy_raises(self):
        config = PipelineConfig(synthesis_strategy="bogus")
        with pytest.raises(ValueError, match="Unrecognised synthesis_strategy"):
            MetaAgentPipeline(
                agent=_make_agent(),
                action_executor=_make_executor(),
                config=config,
            )

    def test_unrecognised_evaluation_strategy_raises(self):
        config = PipelineConfig(evaluation_strategy="bogus")
        with pytest.raises(ValueError, match="Unrecognised evaluation_strategy"):
            MetaAgentPipeline(
                agent=_make_agent(),
                action_executor=_make_executor(),
                config=config,
            )


# ---------------------------------------------------------------------------
# Pipeline run tests
# ---------------------------------------------------------------------------


class TestPipelineRun:
    """Test the full pipeline run with mocked internal components."""

    @patch("agent_foundation.automation.meta_agent.pipeline.PatternExtractor")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceAligner")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceNormalizer")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceEvaluator")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceCollector")
    @patch("agent_foundation.automation.meta_agent.pipeline.RuleBasedSynthesizer")
    def test_full_pipeline_success(
        self,
        MockSynthesizer,
        MockCollector,
        MockEvaluator,
        MockNormalizer,
        MockAligner,
        MockExtractor,
    ):
        traces = [_make_trace("t1"), _make_trace("t2")]
        MockCollector.return_value.collect.return_value = traces

        eval_results = [
            EvaluationResult(trace_id="t1", passed=True),
            EvaluationResult(trace_id="t2", passed=True),
        ]
        MockEvaluator.return_value.evaluate.return_value = eval_results

        MockNormalizer.return_value.normalize.return_value = traces

        mock_aligned = MagicMock()
        MockAligner.return_value.align.return_value = mock_aligned

        mock_patterns = MagicMock()
        MockExtractor.return_value.extract.return_value = mock_patterns

        mock_graph = MagicMock()
        mock_report = MagicMock()
        mock_synth_result = MagicMock()
        mock_synth_result.graph = mock_graph
        mock_synth_result.report = mock_report
        mock_synth_result.python_script = None
        MockSynthesizer.return_value.synthesize.return_value = mock_synth_result

        config = PipelineConfig(run_count=2, validate=False)
        pipeline = MetaAgentPipeline(
            agent=_make_agent(),
            action_executor=_make_executor(),
            config=config,
        )
        result = pipeline.run("test task")

        assert result.graph is mock_graph
        assert result.synthesis_report is mock_report
        assert result.failed_stage is None
        assert result.error is None
        assert len(result.traces) == 2
        assert len(result.evaluation_results) == 2

    @patch("agent_foundation.automation.meta_agent.pipeline.TraceEvaluator")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceCollector")
    def test_evaluation_filters_failed_traces(
        self,
        MockCollector,
        MockEvaluator,
    ):
        """Traces that fail evaluation should not reach normalization."""
        traces = [_make_trace("t1", success=True), _make_trace("t2", success=False)]
        MockCollector.return_value.collect.return_value = traces

        eval_results = [
            EvaluationResult(trace_id="t1", passed=True),
            EvaluationResult(trace_id="t2", passed=False),
        ]
        MockEvaluator.return_value.evaluate.return_value = eval_results

        config = PipelineConfig(run_count=2, validate=False, min_success_traces=1)
        pipeline = MetaAgentPipeline(
            agent=_make_agent(),
            action_executor=_make_executor(),
            config=config,
        )

        with patch(
            "agent_foundation.automation.meta_agent.pipeline.TraceNormalizer"
        ) as MockNorm, patch(
            "agent_foundation.automation.meta_agent.pipeline.TraceAligner"
        ), patch(
            "agent_foundation.automation.meta_agent.pipeline.PatternExtractor"
        ), patch(
            "agent_foundation.automation.meta_agent.pipeline.RuleBasedSynthesizer"
        ) as MockSynth:
            MockNorm.return_value.normalize.return_value = [traces[0]]
            MockSynth.return_value.synthesize.return_value = MagicMock(
                graph=MagicMock(), report=MagicMock(), python_script=None
            )
            result = pipeline.run("test task")

        # Normalizer should have received only the passing trace
        MockNorm.return_value.normalize.assert_called_once()
        normalized_arg = MockNorm.return_value.normalize.call_args[0][0]
        assert len(normalized_arg) == 1
        assert normalized_arg[0].trace_id == "t1"

    @patch("agent_foundation.automation.meta_agent.pipeline.TraceEvaluator")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceCollector")
    def test_insufficient_traces_raises(self, MockCollector, MockEvaluator):
        traces = [_make_trace("t1", success=False)]
        MockCollector.return_value.collect.return_value = traces

        eval_results = [EvaluationResult(trace_id="t1", passed=False)]
        MockEvaluator.return_value.evaluate.return_value = eval_results

        config = PipelineConfig(
            run_count=1, min_success_traces=1, max_retry_rounds=0,
        )
        pipeline = MetaAgentPipeline(
            agent=_make_agent(),
            action_executor=_make_executor(),
            config=config,
        )

        with pytest.raises(InsufficientSuccessTracesError) as exc_info:
            pipeline.run("test task")

        assert exc_info.value.required == 1
        assert exc_info.value.actual == 0
        assert exc_info.value.total == 1


class TestPipelineStageFailure:
    """Test that stage failures produce partial results."""

    @patch("agent_foundation.automation.meta_agent.pipeline.TraceCollector")
    def test_collection_failure(self, MockCollector):
        MockCollector.return_value.collect.side_effect = RuntimeError("agent crash")

        pipeline = MetaAgentPipeline(
            agent=_make_agent(),
            action_executor=_make_executor(),
        )
        result = pipeline.run("test task")

        assert result.failed_stage == "collection"
        assert "agent crash" in result.error
        assert result.graph is None

    @patch("agent_foundation.automation.meta_agent.pipeline.TraceNormalizer")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceEvaluator")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceCollector")
    def test_normalization_failure(self, MockCollector, MockEvaluator, MockNorm):
        traces = [_make_trace()]
        MockCollector.return_value.collect.return_value = traces
        MockEvaluator.return_value.evaluate.return_value = [
            EvaluationResult(trace_id="t1", passed=True)
        ]
        MockNorm.return_value.normalize.side_effect = RuntimeError("bad step")

        pipeline = MetaAgentPipeline(
            agent=_make_agent(),
            action_executor=_make_executor(),
        )
        result = pipeline.run("test task")

        assert result.failed_stage == "normalization"
        assert len(result.traces) == 1
        assert len(result.evaluation_results) == 1


class TestPipelineResultFields:
    """Verify PipelineResult includes evaluation_results and python_script."""

    def test_pipeline_result_has_evaluation_results_field(self):
        result = PipelineResult()
        assert hasattr(result, "evaluation_results")
        assert result.evaluation_results == []

    def test_pipeline_result_has_python_script_field(self):
        result = PipelineResult()
        assert hasattr(result, "python_script")
        assert result.python_script is None

    def test_pipeline_config_has_strategy_fields(self):
        config = PipelineConfig()
        assert config.evaluation_strategy == "exception_only"
        assert config.synthesis_strategy == "rule_based"
        assert config.min_success_traces == 1
        assert config.max_retry_rounds == 3
        assert config.generate_script is False


class TestRefine:
    """Test that refine() merges new traces with existing ones."""

    @patch("agent_foundation.automation.meta_agent.pipeline.PatternExtractor")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceAligner")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceNormalizer")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceEvaluator")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceCollector")
    @patch("agent_foundation.automation.meta_agent.pipeline.RuleBasedSynthesizer")
    def test_refine_merges_new_traces_with_existing(
        self,
        MockSynthesizer,
        MockCollector,
        MockEvaluator,
        MockNormalizer,
        MockAligner,
        MockExtractor,
    ):
        """refine() should collect new traces, evaluate only the new ones,
        merge with existing results, and proceed to synthesis."""
        existing_traces = [_make_trace("existing_1"), _make_trace("existing_2")]
        existing_result = PipelineResult(
            traces=existing_traces,
            evaluation_results=[
                EvaluationResult(trace_id="existing_1", passed=True),
                EvaluationResult(trace_id="existing_2", passed=True),
            ],
        )

        new_traces = [_make_trace("new_1"), _make_trace("new_2")]
        MockCollector.return_value.collect.return_value = new_traces

        # Only the new traces are evaluated
        new_eval = [
            EvaluationResult(trace_id="new_1", passed=True),
            EvaluationResult(trace_id="new_2", passed=True),
        ]
        MockEvaluator.return_value.evaluate.return_value = new_eval

        combined = existing_traces + new_traces
        MockNormalizer.return_value.normalize.return_value = combined

        mock_aligned = MagicMock()
        MockAligner.return_value.align.return_value = mock_aligned

        mock_patterns = MagicMock()
        MockExtractor.return_value.extract.return_value = mock_patterns

        mock_synth_result = MagicMock()
        mock_synth_result.graph = MagicMock()
        mock_synth_result.report = MagicMock()
        mock_synth_result.python_script = None
        MockSynthesizer.return_value.synthesize.return_value = mock_synth_result

        config = PipelineConfig(run_count=2, validate=False)
        pipeline = MetaAgentPipeline(
            agent=_make_agent(),
            action_executor=_make_executor(),
            config=config,
        )
        result = pipeline.refine(existing_result, additional_run_count=2)

        # Collector should have been asked for 2 new traces
        collect_call = MockCollector.return_value.collect.call_args
        assert collect_call[1].get("run_count") == 2 or (
            len(collect_call[0]) > 1 and collect_call[0][1] == 2
        )

        # Evaluator should receive ONLY the 2 new traces (not all 4)
        eval_call_traces = MockEvaluator.return_value.evaluate.call_args[0][0]
        assert len(eval_call_traces) == 2
        trace_ids = {t.trace_id for t in eval_call_traces}
        assert trace_ids == {"new_1", "new_2"}

        # Result should include all 4 traces
        assert len(result.traces) == 4
        assert result.failed_stage is None
        assert result.graph is not None


class TestSynthesizerSelection:
    """Verify the pipeline selects the correct synthesizer subclass."""

    def test_rule_based_selected_by_default(self):
        pipeline = MetaAgentPipeline(
            agent=_make_agent(),
            action_executor=_make_executor(),
        )
        synth = pipeline._create_synthesizer()
        from agent_foundation.automation.meta_agent.synthesizer import (
            RuleBasedSynthesizer,
        )
        assert isinstance(synth, RuleBasedSynthesizer)

    def test_llm_selected_when_configured(self):
        config = PipelineConfig(synthesis_strategy="llm")
        pipeline = MetaAgentPipeline(
            agent=_make_agent(),
            action_executor=_make_executor(),
            config=config,
            inferencer=MagicMock(name="inferencer"),
        )
        synth = pipeline._create_synthesizer()
        from agent_foundation.automation.meta_agent.synthesizer import (
            LLMSynthesizer,
        )
        assert isinstance(synth, LLMSynthesizer)

    def test_hybrid_selected_when_configured(self):
        config = PipelineConfig(synthesis_strategy="hybrid")
        pipeline = MetaAgentPipeline(
            agent=_make_agent(),
            action_executor=_make_executor(),
            config=config,
            inferencer=MagicMock(name="inferencer"),
        )
        synth = pipeline._create_synthesizer()
        from agent_foundation.automation.meta_agent.synthesizer import (
            HybridSynthesizer,
        )
        assert isinstance(synth, HybridSynthesizer)


class TestValidationStage:
    """Verify that Stage 8 (validation) is wired up to GraphValidator."""

    @patch("agent_foundation.automation.meta_agent.pipeline.PatternExtractor")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceAligner")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceNormalizer")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceEvaluator")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceCollector")
    @patch("agent_foundation.automation.meta_agent.pipeline.RuleBasedSynthesizer")
    @patch("agent_foundation.automation.meta_agent.pipeline.GraphValidator")
    def test_validation_calls_graph_validator(
        self,
        MockValidator,
        MockSynthesizer,
        MockCollector,
        MockEvaluator,
        MockNormalizer,
        MockAligner,
        MockExtractor,
    ):
        """When validate=True, the pipeline should call GraphValidator.validate()."""
        traces = [_make_trace("t1")]
        MockCollector.return_value.collect.return_value = traces

        eval_results = [EvaluationResult(trace_id="t1", passed=True)]
        MockEvaluator.return_value.evaluate.return_value = eval_results
        MockNormalizer.return_value.normalize.return_value = traces

        mock_graph = MagicMock()
        mock_report = MagicMock()
        mock_synth_result = MagicMock()
        mock_synth_result.graph = mock_graph
        mock_synth_result.report = mock_report
        mock_synth_result.python_script = None
        MockSynthesizer.return_value.synthesize.return_value = mock_synth_result

        from agent_foundation.automation.meta_agent.models import ValidationResults
        mock_val_results = MagicMock(spec=ValidationResults)
        MockValidator.return_value.validate.return_value = mock_val_results

        config = PipelineConfig(run_count=1, validate=True, validation_runs=2)
        pipeline = MetaAgentPipeline(
            agent=_make_agent(),
            action_executor=_make_executor(),
            config=config,
        )
        result = pipeline.run("test task")

        MockValidator.return_value.validate.assert_called_once()
        assert result.validation_results is mock_val_results
        assert result.failed_stage is None

    @patch("agent_foundation.automation.meta_agent.pipeline.PatternExtractor")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceAligner")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceNormalizer")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceEvaluator")
    @patch("agent_foundation.automation.meta_agent.pipeline.TraceCollector")
    @patch("agent_foundation.automation.meta_agent.pipeline.RuleBasedSynthesizer")
    @patch("agent_foundation.automation.meta_agent.pipeline.GraphValidator")
    def test_validation_failure_sets_failed_stage(
        self,
        MockValidator,
        MockSynthesizer,
        MockCollector,
        MockEvaluator,
        MockNormalizer,
        MockAligner,
        MockExtractor,
    ):
        """When validation raises, failed_stage should be 'validation'."""
        traces = [_make_trace("t1")]
        MockCollector.return_value.collect.return_value = traces

        eval_results = [EvaluationResult(trace_id="t1", passed=True)]
        MockEvaluator.return_value.evaluate.return_value = eval_results
        MockNormalizer.return_value.normalize.return_value = traces

        mock_synth_result = MagicMock()
        mock_synth_result.graph = MagicMock()
        mock_synth_result.report = MagicMock()
        mock_synth_result.python_script = None
        MockSynthesizer.return_value.synthesize.return_value = mock_synth_result

        MockValidator.return_value.validate.side_effect = RuntimeError("validation boom")

        config = PipelineConfig(run_count=1, validate=True)
        pipeline = MetaAgentPipeline(
            agent=_make_agent(),
            action_executor=_make_executor(),
            config=config,
        )
        result = pipeline.run("test task")

        assert result.failed_stage == "validation"
        assert "validation boom" in result.error


# ---------------------------------------------------------------------------
# Iterative collection (retry) tests
# ---------------------------------------------------------------------------


BASE = "agent_foundation.automation.meta_agent.pipeline"


class TestIterativeCollection:
    """Tests for the automatic retry mechanism in _evaluate_with_retries."""

    @patch(f"{BASE}.PatternExtractor")
    @patch(f"{BASE}.TraceAligner")
    @patch(f"{BASE}.TraceNormalizer")
    @patch(f"{BASE}.TraceEvaluator")
    @patch(f"{BASE}.TraceCollector")
    @patch(f"{BASE}.RuleBasedSynthesizer")
    def test_succeeds_on_first_try_no_retries(
        self, MockSynth, MockCollector, MockEval, MockNorm, MockAlign, MockExt,
    ):
        """When all traces pass initially, no retries should occur."""
        traces = [_make_trace("t1"), _make_trace("t2")]
        MockCollector.return_value.collect.return_value = traces
        MockEval.return_value.evaluate.return_value = [
            EvaluationResult(trace_id="t1", passed=True),
            EvaluationResult(trace_id="t2", passed=True),
        ]
        MockNorm.return_value.normalize.return_value = traces
        mock_synth_result = MagicMock(
            graph=MagicMock(), report=MagicMock(), python_script=None,
        )
        MockSynth.return_value.synthesize.return_value = mock_synth_result

        config = PipelineConfig(
            run_count=2, validate=False, min_success_traces=2,
        )
        pipeline = MetaAgentPipeline(
            agent=_make_agent(), action_executor=_make_executor(), config=config,
        )
        result = pipeline.run("test task")

        assert result.failed_stage is None
        # Collector called once (initial), no retries
        assert MockCollector.return_value.collect.call_count == 1

    @patch(f"{BASE}.PatternExtractor")
    @patch(f"{BASE}.TraceAligner")
    @patch(f"{BASE}.TraceNormalizer")
    @patch(f"{BASE}.TraceEvaluator")
    @patch(f"{BASE}.TraceCollector")
    @patch(f"{BASE}.RuleBasedSynthesizer")
    def test_collects_shortfall_on_retry(
        self, MockSynth, MockCollector, MockEval, MockNorm, MockAlign, MockExt,
    ):
        """When initial eval has insufficient passes, pipeline retries with
        shortfall count and evaluates only the new traces."""
        initial_traces = [_make_trace("t1"), _make_trace("t2"), _make_trace("t3")]
        retry_trace = [_make_trace("t4")]

        MockCollector.return_value.collect.side_effect = [
            initial_traces,  # initial collection: 3 traces
            retry_trace,     # retry: 1 trace (shortfall)
        ]

        # Initial eval: 2 of 3 pass; need 3
        initial_eval = [
            EvaluationResult(trace_id="t1", passed=True),
            EvaluationResult(trace_id="t2", passed=True),
            EvaluationResult(trace_id="t3", passed=False),
        ]
        # Retry eval: new trace passes
        retry_eval = [EvaluationResult(trace_id="t4", passed=True)]
        MockEval.return_value.evaluate.side_effect = [initial_eval, retry_eval]

        MockNorm.return_value.normalize.return_value = [
            _make_trace("t1"), _make_trace("t2"), _make_trace("t4"),
        ]
        mock_synth_result = MagicMock(
            graph=MagicMock(), report=MagicMock(), python_script=None,
        )
        MockSynth.return_value.synthesize.return_value = mock_synth_result

        config = PipelineConfig(
            run_count=3, validate=False, min_success_traces=3,
            max_retry_rounds=3,
        )
        pipeline = MetaAgentPipeline(
            agent=_make_agent(), action_executor=_make_executor(), config=config,
        )
        result = pipeline.run("test task")

        assert result.failed_stage is None
        assert result.graph is not None

        # Collector: 2 calls — initial (3) + retry (1 = shortfall)
        assert MockCollector.return_value.collect.call_count == 2
        retry_call = MockCollector.return_value.collect.call_args_list[1]
        assert retry_call[1]["run_count"] == 1

        # Evaluator: 2 calls — initial (3 traces) + retry (1 trace)
        assert MockEval.return_value.evaluate.call_count == 2
        # Second eval call should only have the new trace
        retry_eval_arg = MockEval.return_value.evaluate.call_args_list[1][0][0]
        assert len(retry_eval_arg) == 1
        assert retry_eval_arg[0].trace_id == "t4"

        # Result includes all 4 traces and all 4 eval results
        assert len(result.traces) == 4
        assert len(result.evaluation_results) == 4

    @patch(f"{BASE}.TraceEvaluator")
    @patch(f"{BASE}.TraceCollector")
    def test_max_rounds_exhausted_raises(self, MockCollector, MockEval):
        """When all retries are exhausted, InsufficientSuccessTracesError
        is raised with traces and evaluation_results attached."""
        initial_traces = [_make_trace("t1")]
        retry_traces = [_make_trace("t2"), _make_trace("t3")]

        MockCollector.return_value.collect.side_effect = [
            initial_traces,
            [retry_traces[0]],
            [retry_traces[1]],
        ]
        # All traces fail evaluation
        MockEval.return_value.evaluate.side_effect = [
            [EvaluationResult(trace_id="t1", passed=False)],
            [EvaluationResult(trace_id="t2", passed=False)],
            [EvaluationResult(trace_id="t3", passed=False)],
        ]

        config = PipelineConfig(
            run_count=1, min_success_traces=1, max_retry_rounds=2,
        )
        pipeline = MetaAgentPipeline(
            agent=_make_agent(), action_executor=_make_executor(), config=config,
        )

        with pytest.raises(InsufficientSuccessTracesError) as exc_info:
            pipeline.run("test task")

        err = exc_info.value
        assert err.required == 1
        assert err.actual == 0
        assert err.total == 3
        assert len(err.traces) == 3
        assert len(err.evaluation_results) == 3

    @patch(f"{BASE}.TraceEvaluator")
    @patch(f"{BASE}.TraceCollector")
    def test_max_retry_rounds_zero_disables_retries(
        self, MockCollector, MockEval,
    ):
        """With max_retry_rounds=0, pipeline fails immediately without retrying."""
        traces = [_make_trace("t1")]
        MockCollector.return_value.collect.return_value = traces
        MockEval.return_value.evaluate.return_value = [
            EvaluationResult(trace_id="t1", passed=False),
        ]

        config = PipelineConfig(
            run_count=1, min_success_traces=1, max_retry_rounds=0,
        )
        pipeline = MetaAgentPipeline(
            agent=_make_agent(), action_executor=_make_executor(), config=config,
        )

        with pytest.raises(InsufficientSuccessTracesError):
            pipeline.run("test task")

        # Only one collect call, no retries
        assert MockCollector.return_value.collect.call_count == 1
        assert MockEval.return_value.evaluate.call_count == 1

    @patch(f"{BASE}.TraceEvaluator")
    @patch(f"{BASE}.TraceCollector")
    def test_retry_uses_none_input_data(self, MockCollector, MockEval):
        """On retry rounds, collector.collect should be called with
        input_data=None, not the original input_data."""
        initial_traces = [_make_trace("t1")]
        retry_trace = [_make_trace("t2")]

        MockCollector.return_value.collect.side_effect = [
            initial_traces,
            retry_trace,
        ]
        MockEval.return_value.evaluate.side_effect = [
            [EvaluationResult(trace_id="t1", passed=False)],
            [EvaluationResult(trace_id="t2", passed=True)],
        ]

        config = PipelineConfig(
            run_count=1, validate=False, min_success_traces=1,
            max_retry_rounds=1,
        )
        pipeline = MetaAgentPipeline(
            agent=_make_agent(), action_executor=_make_executor(), config=config,
        )

        with patch(f"{BASE}.TraceNormalizer") as MockNorm, \
             patch(f"{BASE}.TraceAligner"), \
             patch(f"{BASE}.PatternExtractor"), \
             patch(f"{BASE}.RuleBasedSynthesizer") as MockSynth:
            MockNorm.return_value.normalize.return_value = [retry_trace[0]]
            MockSynth.return_value.synthesize.return_value = MagicMock(
                graph=MagicMock(), report=MagicMock(), python_script=None,
            )
            pipeline.run("test task", input_data=[{"key": "val"}])

        # Initial call uses original input_data
        initial_call = MockCollector.return_value.collect.call_args_list[0]
        assert initial_call[1]["input_data"] == [{"key": "val"}]

        # Retry call uses input_data=None
        retry_call = MockCollector.return_value.collect.call_args_list[1]
        assert retry_call[1]["input_data"] is None

    @patch(f"{BASE}.TraceEvaluator")
    @patch(f"{BASE}.TraceCollector")
    def test_evaluation_error_returns_failed_stage_evaluation(
        self, MockCollector, MockEval,
    ):
        """When the evaluator raises, failed_stage should be 'evaluation',
        not 'collection'."""
        traces = [_make_trace("t1")]
        MockCollector.return_value.collect.return_value = traces
        MockEval.return_value.evaluate.side_effect = RuntimeError("eval boom")

        config = PipelineConfig(run_count=1, min_success_traces=1)
        pipeline = MetaAgentPipeline(
            agent=_make_agent(), action_executor=_make_executor(), config=config,
        )
        result = pipeline.run("test task")

        assert result.failed_stage == "evaluation"
        assert "eval boom" in result.error

    @patch(f"{BASE}.TraceEvaluator")
    @patch(f"{BASE}.TraceCollector")
    def test_retry_collection_error_returns_failed_stage_collection(
        self, MockCollector, MockEval,
    ):
        """When collector raises during a retry round, failed_stage should
        be 'collection'."""
        initial_traces = [_make_trace("t1")]
        MockCollector.return_value.collect.side_effect = [
            initial_traces,
            RuntimeError("retry collect fail"),
        ]
        MockEval.return_value.evaluate.return_value = [
            EvaluationResult(trace_id="t1", passed=False),
        ]

        config = PipelineConfig(
            run_count=1, min_success_traces=1, max_retry_rounds=1,
        )
        pipeline = MetaAgentPipeline(
            agent=_make_agent(), action_executor=_make_executor(), config=config,
        )
        result = pipeline.run("test task")

        assert result.failed_stage == "collection"
        assert "retry collect fail" in result.error

    @patch(f"{BASE}.TraceEvaluator")
    @patch(f"{BASE}.TraceCollector")
    def test_partial_results_preserved_on_mid_retry_failure(
        self, MockCollector, MockEval,
    ):
        """When collector fails during retry round 2, traces from initial
        + round 1 are preserved in the result."""
        initial = [_make_trace("t1")]
        round1 = [_make_trace("t2")]

        MockCollector.return_value.collect.side_effect = [
            initial,
            round1,
            RuntimeError("round 2 fail"),
        ]
        MockEval.return_value.evaluate.side_effect = [
            [EvaluationResult(trace_id="t1", passed=False)],
            [EvaluationResult(trace_id="t2", passed=False)],
        ]

        config = PipelineConfig(
            run_count=1, min_success_traces=1, max_retry_rounds=3,
        )
        pipeline = MetaAgentPipeline(
            agent=_make_agent(), action_executor=_make_executor(), config=config,
        )
        result = pipeline.run("test task")

        assert result.failed_stage == "collection"
        # Traces from initial + round 1 are preserved
        assert len(result.traces) == 2
        assert {t.trace_id for t in result.traces} == {"t1", "t2"}
        # Evaluation results from initial + round 1 are preserved
        assert len(result.evaluation_results) == 2


class TestCollectionHooks:
    """Tests for hook timing with retries."""

    @patch(f"{BASE}.PatternExtractor")
    @patch(f"{BASE}.TraceAligner")
    @patch(f"{BASE}.TraceNormalizer")
    @patch(f"{BASE}.TraceEvaluator")
    @patch(f"{BASE}.TraceCollector")
    @patch(f"{BASE}.RuleBasedSynthesizer")
    def test_collection_hook_fires_before_evaluation(
        self, MockSynth, MockCollector, MockEval, MockNorm, MockAlign, MockExt,
    ):
        """The collection hook should fire with the initial trace count,
        before evaluation begins."""
        traces = [_make_trace("t1"), _make_trace("t2")]
        MockCollector.return_value.collect.return_value = traces
        MockEval.return_value.evaluate.return_value = [
            EvaluationResult(trace_id="t1", passed=True),
            EvaluationResult(trace_id="t2", passed=True),
        ]
        MockNorm.return_value.normalize.return_value = traces
        MockSynth.return_value.synthesize.return_value = MagicMock(
            graph=MagicMock(), report=MagicMock(), python_script=None,
        )

        hook = MagicMock()
        config = PipelineConfig(run_count=2, validate=False, min_success_traces=1)
        pipeline = MetaAgentPipeline(
            agent=_make_agent(), action_executor=_make_executor(),
            config=config, stage_hook=hook,
        )
        pipeline.run("test task")

        # Find the collection hook call
        collection_calls = [
            c for c in hook.call_args_list if c[0][0] == "collection"
        ]
        assert len(collection_calls) == 1
        data = collection_calls[0][0][1]
        assert data["trace_count"] == 2

    @patch(f"{BASE}.PatternExtractor")
    @patch(f"{BASE}.TraceAligner")
    @patch(f"{BASE}.TraceNormalizer")
    @patch(f"{BASE}.TraceEvaluator")
    @patch(f"{BASE}.TraceCollector")
    @patch(f"{BASE}.RuleBasedSynthesizer")
    def test_evaluation_hook_fires_once_with_all_results(
        self, MockSynth, MockCollector, MockEval, MockNorm, MockAlign, MockExt,
    ):
        """Even with retries, the evaluation hook should fire once with
        all combined results."""
        initial = [_make_trace("t1"), _make_trace("t2")]
        retry = [_make_trace("t3")]
        MockCollector.return_value.collect.side_effect = [initial, retry]

        MockEval.return_value.evaluate.side_effect = [
            [
                EvaluationResult(trace_id="t1", passed=True),
                EvaluationResult(trace_id="t2", passed=False),
            ],
            [EvaluationResult(trace_id="t3", passed=True)],
        ]
        MockNorm.return_value.normalize.return_value = [
            _make_trace("t1"), _make_trace("t3"),
        ]
        MockSynth.return_value.synthesize.return_value = MagicMock(
            graph=MagicMock(), report=MagicMock(), python_script=None,
        )

        hook = MagicMock()
        config = PipelineConfig(
            run_count=2, validate=False, min_success_traces=2,
            max_retry_rounds=1,
        )
        pipeline = MetaAgentPipeline(
            agent=_make_agent(), action_executor=_make_executor(),
            config=config, stage_hook=hook,
        )
        pipeline.run("test task")

        # Evaluation hook should fire once
        eval_calls = [
            c for c in hook.call_args_list if c[0][0] == "evaluation"
        ]
        assert len(eval_calls) == 1
        data = eval_calls[0][0][1]
        assert data["passed_count"] == 2
        assert data["total_count"] == 3


class TestRefineUpdated:
    """Tests for the updated refine() behavior."""

    @patch(f"{BASE}.PatternExtractor")
    @patch(f"{BASE}.TraceAligner")
    @patch(f"{BASE}.TraceNormalizer")
    @patch(f"{BASE}.TraceEvaluator")
    @patch(f"{BASE}.TraceCollector")
    @patch(f"{BASE}.RuleBasedSynthesizer")
    def test_refine_fires_evaluation_hook(
        self, MockSynth, MockCollector, MockEval, MockNorm, MockAlign, MockExt,
    ):
        """refine() should fire the evaluation hook before synthesis."""
        existing = PipelineResult(
            traces=[_make_trace("t1")],
            evaluation_results=[
                EvaluationResult(trace_id="t1", passed=True),
            ],
        )
        new_traces = [_make_trace("t2")]
        MockCollector.return_value.collect.return_value = new_traces
        MockEval.return_value.evaluate.return_value = [
            EvaluationResult(trace_id="t2", passed=True),
        ]
        MockNorm.return_value.normalize.return_value = [
            _make_trace("t1"), _make_trace("t2"),
        ]
        MockSynth.return_value.synthesize.return_value = MagicMock(
            graph=MagicMock(), report=MagicMock(), python_script=None,
        )

        hook = MagicMock()
        config = PipelineConfig(run_count=1, validate=False)
        pipeline = MetaAgentPipeline(
            agent=_make_agent(), action_executor=_make_executor(),
            config=config, stage_hook=hook,
        )
        pipeline.refine(existing, additional_run_count=1)

        eval_calls = [
            c for c in hook.call_args_list if c[0][0] == "evaluation"
        ]
        assert len(eval_calls) == 1
        data = eval_calls[0][0][1]
        assert data["passed_count"] == 2
        assert data["total_count"] == 2

    @patch(f"{BASE}.TraceEvaluator")
    @patch(f"{BASE}.TraceCollector")
    def test_refine_handles_mismatched_traces_and_eval_results(
        self, MockCollector, MockEval,
    ):
        """refine() should truncate mismatched traces/eval_results safely."""
        # 3 traces but only 2 eval results (simulating mid-eval crash)
        existing = PipelineResult(
            traces=[
                _make_trace("t1"), _make_trace("t2"), _make_trace("t3"),
            ],
            evaluation_results=[
                EvaluationResult(trace_id="t1", passed=True),
                EvaluationResult(trace_id="t2", passed=False),
            ],
        )
        new_traces = [_make_trace("t4")]
        MockCollector.return_value.collect.return_value = new_traces
        MockEval.return_value.evaluate.return_value = [
            EvaluationResult(trace_id="t4", passed=True),
        ]

        config = PipelineConfig(
            run_count=1, validate=False, min_success_traces=2,
        )
        pipeline = MetaAgentPipeline(
            agent=_make_agent(), action_executor=_make_executor(), config=config,
        )

        with patch(f"{BASE}.TraceNormalizer") as MockNorm, \
             patch(f"{BASE}.TraceAligner"), \
             patch(f"{BASE}.PatternExtractor"), \
             patch(f"{BASE}.RuleBasedSynthesizer") as MockSynth:
            MockNorm.return_value.normalize.return_value = [
                _make_trace("t1"), _make_trace("t4"),
            ]
            MockSynth.return_value.synthesize.return_value = MagicMock(
                graph=MagicMock(), report=MagicMock(), python_script=None,
            )
            result = pipeline.refine(existing, additional_run_count=1)

        # Should succeed: t1 passed (from existing), t4 passed (new) = 2
        assert result.failed_stage is None
        # 2 (truncated existing) + 1 (new) = 3 total traces
        assert len(result.traces) == 3

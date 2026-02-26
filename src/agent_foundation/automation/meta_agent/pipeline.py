"""
Meta Agent Pipeline — end-to-end orchestration.

Runs the full trace-to-ActionGraph pipeline:

1. Trace Collection
2. Trace Evaluation & Filtering
3. Trace Normalization
4. Target Conversion
5. Trace Alignment
6. Pattern Extraction
7. Graph Synthesis
8. (optional) Validation

The appropriate :class:`GraphSynthesizer` subclass is selected based on
``config.synthesis_strategy``.  LLM and Hybrid strategies require an
``InferencerBase`` instance.  Similarly, the ``LLM_JUDGE`` evaluation
strategy requires an inferencer.

Now implemented as a :class:`Workflow` subclass, using per-step attributes
for named steps, flow state, loop-back retries, and error handling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from attr import attrs, attrib

from rich_python_utils.common_objects.workflow.common.result_pass_down_mode import (
    ResultPassDownMode,
)
from rich_python_utils.common_objects.workflow.common.exceptions import (
    WorkflowAborted,
)
from rich_python_utils.common_objects.workflow.workflow import Workflow
from rich_python_utils.datetime_utils.common import timestamp

from science_modeling_tools.automation.meta_agent.aligner import TraceAligner
from science_modeling_tools.automation.meta_agent.collector import TraceCollector
from science_modeling_tools.automation.meta_agent.errors import (
    InsufficientSuccessTracesError,
    PipelineAborted,
    PipelineStageError,
)
from science_modeling_tools.automation.meta_agent.evaluator import (
    EvaluationResult,
    EvaluationRule,
    EvaluationStrategy,
    TraceEvaluator,
)
from science_modeling_tools.automation.meta_agent.models import (
    ExecutionTrace,
    PipelineConfig,
    PipelineResult,
)
from science_modeling_tools.automation.meta_agent.normalizer import TraceNormalizer
from science_modeling_tools.automation.meta_agent.pattern_extractor import PatternExtractor
from science_modeling_tools.automation.meta_agent.synthesizer import (
    GraphSynthesizer,
    HybridSynthesizer,
    LLMSynthesizer,
    RuleBasedSynthesizer,
    SynthesisStrategy,
)
from science_modeling_tools.automation.meta_agent.synthetic_data import (
    SyntheticDataProvider,
)
from science_modeling_tools.automation.meta_agent.validator import GraphValidator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy string → enum helpers
# ---------------------------------------------------------------------------

_EVAL_STRATEGY_MAP = {s.value: s for s in EvaluationStrategy}
_SYNTH_STRATEGY_MAP = {s.value: s for s in SynthesisStrategy}


# ---------------------------------------------------------------------------
# Step wrapper
# ---------------------------------------------------------------------------

class _StepWrapper:
    """Wraps a callable so arbitrary per-step attributes can be assigned."""

    def __init__(self, fn):
        self._fn = fn
        self.__name__ = getattr(fn, '__name__', str(fn))
        self.__module__ = getattr(fn, '__module__', None)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@attrs(slots=False)
class MetaAgentPipeline(Workflow):
    """End-to-end pipeline: collection → evaluation → normalization →
    target conversion → alignment → extraction → synthesis → validation.

    Inherits from :class:`Workflow` using per-step attributes for named
    steps, flow state, the evaluation/collection retry loop, and error
    handling.
    """

    # Pipeline-specific config (kw_only so attrs doesn't collide with
    # Workflow positional attrs whose defaults come first).
    _agent = attrib(kw_only=True)
    _action_executor = attrib(kw_only=True)
    _config = attrib(factory=PipelineConfig, kw_only=True)
    _synthetic_data_provider = attrib(default=None, kw_only=True)
    _action_metadata = attrib(default=None, kw_only=True)
    _inferencer = attrib(default=None, kw_only=True)
    _evaluation_rules = attrib(factory=list, kw_only=True)
    _output_dir = attrib(default=None, kw_only=True)
    _stage_hook = attrib(default=None, kw_only=True)

    # Derived (init=False)
    _synthesis_strategy = attrib(init=False)
    _evaluation_strategy = attrib(init=False)
    _prompt_formatter = attrib(init=False)
    _pre_populated_state = attrib(default=None, init=False)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __attrs_post_init__(self):
        self._validate_strategies()
        self._build_prompt_formatter()
        self._steps = self._build_pipeline_steps()
        self.result_pass_down_mode = ResultPassDownMode.NoPassDown
        self.enable_result_save = False
        self.unpack_single_result = False
        super().__attrs_post_init__()

    def _validate_strategies(self):
        """Resolve strategy strings and validate inferencer requirements."""
        synth_str = self._config.synthesis_strategy
        if synth_str not in _SYNTH_STRATEGY_MAP:
            raise ValueError(
                f"Unrecognised synthesis_strategy '{synth_str}'. "
                f"Valid values: {sorted(_SYNTH_STRATEGY_MAP)}"
            )
        synth = _SYNTH_STRATEGY_MAP[synth_str]

        eval_str = self._config.evaluation_strategy
        if eval_str not in _EVAL_STRATEGY_MAP:
            raise ValueError(
                f"Unrecognised evaluation_strategy '{eval_str}'. "
                f"Valid values: {sorted(_EVAL_STRATEGY_MAP)}"
            )
        eval_strat = _EVAL_STRATEGY_MAP[eval_str]

        if synth in (SynthesisStrategy.LLM, SynthesisStrategy.HYBRID):
            if self._inferencer is None:
                raise ValueError(
                    f"{synth.value} synthesis strategy requires an "
                    f"InferencerBase instance"
                )
        if eval_strat == EvaluationStrategy.LLM_JUDGE:
            if self._inferencer is None:
                raise ValueError(
                    "LLM_JUDGE evaluation strategy requires an "
                    "InferencerBase instance"
                )

        self._synthesis_strategy = synth
        self._evaluation_strategy = eval_strat

    def _build_prompt_formatter(self):
        from science_modeling_tools.automation.meta_agent.prompt_templates import (
            create_prompt_formatter,
        )
        self._prompt_formatter = create_prompt_formatter(
            self._config.prompt_templates,
        )

    # ------------------------------------------------------------------
    # Step construction
    # ------------------------------------------------------------------

    def _build_pipeline_steps(self) -> list:
        """Build the 8 pipeline steps as _StepWrapper instances."""
        steps = []

        # 0 — collection
        s = _StepWrapper(self._step_collect)
        s.name = "collection"
        s.update_state = self._update_state_collection
        s.error_handler = self._stage_error_handler
        steps.append(s)

        # 1 — evaluation (with loop back to collection)
        s = _StepWrapper(self._step_evaluate)
        s.name = "evaluation"
        s.update_state = self._update_state_evaluation
        s.error_handler = self._evaluation_error_handler
        s.loop_back_to = "collection"
        s.loop_condition = self._insufficient_traces
        s.max_loop_iterations = self._config.max_retry_rounds
        s.on_loop_exhausted = self._on_evaluation_loop_exhausted
        steps.append(s)

        # 2 — normalization
        s = _StepWrapper(self._step_normalize)
        s.name = "normalization"
        s.update_state = self._update_state_normalization
        s.error_handler = self._stage_error_handler
        steps.append(s)

        # 3 — target conversion
        s = _StepWrapper(self._step_target_convert)
        s.name = "target_conversion"
        s.update_state = self._update_state_target_conversion
        s.error_handler = self._stage_error_handler
        steps.append(s)

        # 4 — alignment
        s = _StepWrapper(self._step_align)
        s.name = "alignment"
        s.update_state = self._update_state_alignment
        s.error_handler = self._stage_error_handler
        steps.append(s)

        # 5 — extraction
        s = _StepWrapper(self._step_extract)
        s.name = "extraction"
        s.update_state = self._update_state_extraction
        s.error_handler = self._stage_error_handler
        steps.append(s)

        # 6 — synthesis
        s = _StepWrapper(self._step_synthesize)
        s.name = "synthesis"
        s.update_state = self._update_state_synthesis
        s.error_handler = self._stage_error_handler
        steps.append(s)

        # 7 — validation
        s = _StepWrapper(self._step_validate)
        s.name = "validation"
        s.update_state = self._update_state_validation
        s.error_handler = self._stage_error_handler
        steps.append(s)

        return steps

    # ------------------------------------------------------------------
    # Workflow overrides
    # ------------------------------------------------------------------

    def _init_state(self) -> dict:
        if getattr(self, '_pre_populated_state', None) is not None:
            return self._pre_populated_state
        return {
            'traces': [],
            'evaluation_results': [],
            'filtered_traces': [],
            'normalized': [],
            'aligned': None,
            'patterns': None,
            'synthesis_result': None,
            'validation_results': None,
            'python_script': None,
            '_config_generate_script': self._config.generate_script,
        }

    def _on_step_complete(self, result, step_name, step_index, state,
                          *args, **kwargs):
        if step_name == "collection":
            # Guard: only fire on the FIRST collection pass.
            if not state.get('evaluation_results'):
                self._invoke_hook("collection", {
                    "traces": state['traces'],
                    "trace_count": len(state['traces']),
                })
        elif step_name == "evaluation":
            self._invoke_hook("evaluation", {
                "traces": state['traces'],
                "evaluation_results": state['evaluation_results'],
                "filtered_traces": state['filtered_traces'],
                "passed_count": len(state['filtered_traces']),
                "total_count": len(state['traces']),
            })
        elif step_name == "synthesis":
            sr = state.get('synthesis_result')
            if sr:
                self._invoke_hook("synthesis", {
                    "graph": sr.graph,
                    "synthesis_report": sr.report,
                    "python_script": state.get('python_script'),
                })
        elif step_name == "validation":
            self._invoke_hook("validation", {
                "validation_results": state.get('validation_results'),
            })

    def _handle_abort(self, abort_exc, step_result, state):
        if abort_exc.partial_result is not None:
            return abort_exc.partial_result
        if isinstance(abort_exc, PipelineAborted):
            failed_stage = f"{abort_exc.stage}_aborted"
        else:
            failed_stage = abort_exc.step_name
        return PipelineResult(
            traces=state.get('traces', []) if state else [],
            evaluation_results=state.get('evaluation_results', []) if state else [],
            error=str(abort_exc),
            failed_stage=failed_stage,
        )

    # ------------------------------------------------------------------
    # Step methods
    # ------------------------------------------------------------------

    def _step_collect(self, task_description, input_data=None):
        """Stage 1: Trace collection."""
        state = self._state
        collector = TraceCollector(
            agent=self._agent,
            synthetic_data_provider=self._synthetic_data_provider,
        )

        if state.get('evaluation_results'):
            # Retry mode: collect shortfall
            shortfall = self._config.min_success_traces - len(
                state['filtered_traces']
            )
            logger.info(
                "Retry: %d/%d passed (need %d), collecting %d more",
                len(state['filtered_traces']),
                len(state['traces']),
                self._config.min_success_traces,
                shortfall,
            )
            new_traces = collector.collect(
                task_description=task_description,
                run_count=shortfall,
                input_data=None,
            )
        else:
            # Initial collection
            new_traces = collector.collect(
                task_description=task_description,
                run_count=self._config.run_count,
                input_data=input_data,
            )
        return new_traces

    def _step_evaluate(self, task_description, input_data=None):
        """Stage 2: Evaluate only new (unevaluated) traces."""
        state = self._state
        evaluator = TraceEvaluator(
            strategy=self._evaluation_strategy,
            rules=self._evaluation_rules or None,
            inferencer=self._inferencer,
            prompt_formatter=self._prompt_formatter,
        )
        unevaluated = state['traces'][len(state['evaluation_results']):]
        new_eval = evaluator.evaluate(list(unevaluated), task_description)
        return new_eval

    def _step_normalize(self, task_description, input_data=None):
        """Stage 3: Normalize filtered traces."""
        normalizer = TraceNormalizer(
            action_metadata=self._action_metadata,
            custom_type_map=self._config.custom_type_map,
        )
        return normalizer.normalize(self._state['filtered_traces'])

    def _step_target_convert(self, task_description, input_data=None):
        """Stage 4: Target conversion (optional, may be no-op)."""
        normalized = self._state['normalized']
        if self._config.target_converter is not None:
            converter = self._config.target_converter
            for trace in normalized:
                converter.convert_all(trace.steps)
        return normalized

    def _step_align(self, task_description, input_data=None):
        """Stage 5: Alignment."""
        aligner = TraceAligner()
        return aligner.align(self._state['normalized'])

    def _step_extract(self, task_description, input_data=None):
        """Stage 6: Pattern extraction."""
        extractor = PatternExtractor()
        return extractor.extract(self._state['aligned'])

    def _step_synthesize(self, task_description, input_data=None):
        """Stage 7: Graph synthesis."""
        synthesizer = self._create_synthesizer()
        synthesis_result = synthesizer.synthesize(
            self._state['patterns'], task_description,
        )
        return synthesis_result

    def _step_validate(self, task_description, input_data=None):
        """Stage 8: Validation (optional)."""
        state = self._state
        sr = state['synthesis_result']

        if not self._config.validate:
            return None

        validator = GraphValidator()
        filtered = state['filtered_traces']
        test_data = [
            t.input_data or {} for t in filtered
        ][:self._config.validation_runs]
        expected = filtered[:self._config.validation_runs]

        if not test_data:
            test_data = [{}]
            expected = [None]  # type: ignore[list-item]

        return validator.validate(
            graph=sr.graph,
            task_description=task_description,
            test_data=test_data,
            expected_traces=expected,
        )

    # ------------------------------------------------------------------
    # State updaters (static-like, called by Workflow._update_state)
    # ------------------------------------------------------------------

    @staticmethod
    def _update_state_collection(state, result):
        state['traces'].extend(result)
        return state

    @staticmethod
    def _update_state_evaluation(state, result):
        state['evaluation_results'].extend(result)
        state['filtered_traces'] = [
            t for t, r in zip(state['traces'], state['evaluation_results'])
            if r.passed
        ]
        return state

    @staticmethod
    def _update_state_normalization(state, result):
        state['normalized'] = result
        return state

    @staticmethod
    def _update_state_target_conversion(state, result):
        state['normalized'] = result
        return state

    @staticmethod
    def _update_state_alignment(state, result):
        state['aligned'] = result
        return state

    @staticmethod
    def _update_state_extraction(state, result):
        state['patterns'] = result
        return state

    @staticmethod
    def _update_state_synthesis(state, result):
        state['synthesis_result'] = result
        python_script = result.python_script
        if state.get('_config_generate_script') and python_script is None:
            try:
                python_script = result.graph._generate_python_script()
            except Exception:
                logger.warning(
                    "Python script generation failed", exc_info=True,
                )
        state['python_script'] = python_script
        return state

    @staticmethod
    def _update_state_validation(state, result):
        state['validation_results'] = result
        return state

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------

    def _stage_error_handler(self, error, step_result_so_far, state,
                             step_name, step_index):
        """Generic error handler — wraps error in WorkflowAborted with a
        PipelineResult as partial_result."""
        raise WorkflowAborted(
            message=str(error),
            step_name=step_name,
            step_index=step_index,
            partial_result=PipelineResult(
                traces=state.get('traces', []) if state else [],
                evaluation_results=state.get('evaluation_results', []) if state else [],
                error=str(error),
                failed_stage=step_name,
            ),
        )

    def _evaluation_error_handler(self, error, step_result_so_far, state,
                                  step_name, step_index):
        """Evaluation error handler — propagates InsufficientSuccessTracesError
        directly, wraps everything else."""
        if isinstance(error, InsufficientSuccessTracesError):
            raise error
        self._stage_error_handler(
            error, step_result_so_far, state, step_name, step_index,
        )

    # ------------------------------------------------------------------
    # Loop helpers
    # ------------------------------------------------------------------

    def _insufficient_traces(self, state, result):
        """Loop condition: True when fewer traces passed than required."""
        return len(state['filtered_traces']) < self._config.min_success_traces

    def _on_evaluation_loop_exhausted(self, state, result):
        """Called when max retry rounds exhausted."""
        raise InsufficientSuccessTracesError(
            required=self._config.min_success_traces,
            actual=len(state['filtered_traces']),
            total=len(state['traces']),
            traces=list(state['traces']),
            evaluation_results=list(state['evaluation_results']),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        task_description: str,
        input_data: Optional[List[Dict[str, Any]]] = None,
    ) -> PipelineResult:
        """Run the full pipeline.

        Raises:
            InsufficientSuccessTracesError: If fewer than
                ``config.min_success_traces`` pass evaluation after all
                retry rounds are exhausted.
        """
        try:
            result = super().run(task_description, input_data)
        except InsufficientSuccessTracesError:
            raise
        except PipelineAborted as exc:
            return PipelineResult(
                traces=self._state.get('traces', []) if self._state else [],
                evaluation_results=(
                    self._state.get('evaluation_results', [])
                    if self._state else []
                ),
                error=str(exc),
                failed_stage=f"{exc.stage}_aborted",
            )

        if isinstance(result, PipelineResult):
            return result

        # Build final PipelineResult from state
        state = self._state
        sr = state.get('synthesis_result')
        return PipelineResult(
            graph=sr.graph if sr else None,
            synthesis_report=sr.report if sr else None,
            validation_results=state.get('validation_results'),
            traces=state.get('traces', []),
            evaluation_results=state.get('evaluation_results', []),
            python_script=state.get('python_script'),
        )

    def refine(
        self,
        existing_result: PipelineResult,
        additional_run_count: int = 3,
    ) -> PipelineResult:
        """Refine an existing result with additional traces.

        Collects ``additional_run_count`` new traces, evaluates only
        the new ones, merges with existing results, and proceeds to
        synthesis if enough traces pass.

        Unlike ``run()``, this method does **not** retry internally —
        it is itself a manual retry mechanism.
        """
        task_desc = (
            existing_result.traces[0].task_description
            if existing_result.traces
            else ""
        )
        traces = list(existing_result.traces)
        evaluation_results = list(existing_result.evaluation_results)

        # Guard: ensure traces and evaluation_results are in sync.
        min_len = min(len(traces), len(evaluation_results))
        if min_len < max(len(traces), len(evaluation_results)):
            logger.warning(
                "Truncating mismatched traces (%d) / evaluation_results "
                "(%d) to %d",
                len(traces), len(evaluation_results), min_len,
            )
        traces = traces[:min_len]
        evaluation_results = evaluation_results[:min_len]

        try:
            try:
                collector = TraceCollector(
                    agent=self._agent,
                    synthetic_data_provider=self._synthetic_data_provider,
                )
                new_traces = collector.collect(
                    task_description=task_desc,
                    run_count=additional_run_count,
                )
            except Exception as exc:
                return PipelineResult(
                    traces=traces,
                    evaluation_results=evaluation_results,
                    error=str(exc),
                    failed_stage="collection",
                )

            try:
                evaluator = TraceEvaluator(
                    strategy=self._evaluation_strategy,
                    rules=self._evaluation_rules or None,
                    inferencer=self._inferencer,
                    prompt_formatter=self._prompt_formatter,
                )
                new_eval = evaluator.evaluate(new_traces, task_desc)
            except Exception as exc:
                traces.extend(new_traces)
                return PipelineResult(
                    traces=traces,
                    evaluation_results=evaluation_results,
                    error=str(exc),
                    failed_stage="evaluation",
                )

            traces.extend(new_traces)
            evaluation_results.extend(new_eval)
            filtered = [
                t for t, r in zip(traces, evaluation_results) if r.passed
            ]

            if len(filtered) < self._config.min_success_traces:
                raise InsufficientSuccessTracesError(
                    required=self._config.min_success_traces,
                    actual=len(filtered),
                    total=len(traces),
                    traces=traces,
                    evaluation_results=evaluation_results,
                )

            self._invoke_hook("evaluation", {
                "traces": traces,
                "evaluation_results": evaluation_results,
                "filtered_traces": filtered,
                "passed_count": len(filtered),
                "total_count": len(traces),
            })

            return self._run_from_synthesis(
                task_desc, traces, evaluation_results, filtered,
            )
        except PipelineAborted as exc:
            return PipelineResult(
                traces=traces,
                evaluation_results=evaluation_results,
                error=str(exc),
                failed_stage=f"{exc.stage}_aborted",
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_synthesizer(self) -> GraphSynthesizer:
        """Instantiate the appropriate synthesizer subclass."""
        kwargs = dict(
            action_executor=self._action_executor,
            action_metadata=self._action_metadata,
            agent_action_type=self._config.agent_action_type,
            prompt_formatter=self._prompt_formatter,
        )
        if self._synthesis_strategy == SynthesisStrategy.LLM:
            return LLMSynthesizer(inferencer=self._inferencer, **kwargs)
        if self._synthesis_strategy == SynthesisStrategy.HYBRID:
            return HybridSynthesizer(inferencer=self._inferencer, **kwargs)
        return RuleBasedSynthesizer(**kwargs)

    def _invoke_hook(self, stage_name: str, data: dict) -> None:
        """Save stage checkpoint (always) and call stage_hook (if set)."""
        self._save_stage_checkpoint(stage_name, data)
        if self._stage_hook is not None:
            self._stage_hook(stage_name, data)

    def _save_stage_checkpoint(self, stage_name: str, data: dict) -> None:
        """Write checkpoint.json to stage subfolder. No-op if output_dir not set."""
        if self._output_dir is None:
            return
        stage_dir = self._output_dir / f"stage_{stage_name}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        summary = self._build_checkpoint_summary(stage_name, data)
        checkpoint = {"stage": stage_name, "timestamp": timestamp(), **summary}
        (stage_dir / "checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2, default=str)
        )

        # For synthesis stage, also write action_graph.json and synthesis_report.json
        if stage_name == "synthesis":
            graph = data.get("graph")
            if graph is not None:
                try:
                    graph_dict = graph.to_dict()
                    (stage_dir / "action_graph.json").write_text(
                        json.dumps(graph_dict, indent=2, default=str)
                    )
                except Exception:
                    logger.warning(
                        "Failed to serialize action graph to checkpoint",
                        exc_info=True,
                    )
            report = data.get("synthesis_report")
            if report is not None:
                try:
                    report_dict = (
                        report.to_dict() if hasattr(report, "to_dict") else str(report)
                    )
                    (stage_dir / "synthesis_report.json").write_text(
                        json.dumps(report_dict, indent=2, default=str)
                    )
                except Exception:
                    logger.warning(
                        "Failed to serialize synthesis report to checkpoint",
                        exc_info=True,
                    )

        # For validation stage, write validation_results.json
        if stage_name == "validation":
            vr = data.get("validation_results")
            if vr is not None:
                try:
                    results_list = []
                    for r in vr.results:
                        rd: Dict[str, Any] = {
                            "input_data": r.input_data,
                            "passed": r.passed,
                        }
                        if r.divergence_point is not None:
                            rd["divergence_point"] = r.divergence_point
                        if r.expected_outcome is not None:
                            rd["expected_outcome"] = r.expected_outcome
                        if r.actual_outcome is not None:
                            rd["actual_outcome"] = r.actual_outcome
                        if r.error is not None:
                            rd["error"] = r.error
                        results_list.append(rd)
                    vr_dict = {
                        "results": results_list,
                        "success_rate": vr.success_rate,
                        "all_passed": vr.all_passed,
                    }
                    (stage_dir / "validation_results.json").write_text(
                        json.dumps(vr_dict, indent=2, default=str)
                    )
                except Exception:
                    logger.warning(
                        "Failed to serialize validation results to checkpoint",
                        exc_info=True,
                    )

    def _build_checkpoint_summary(self, stage_name: str, data: dict) -> dict:
        """Extract serializable summary fields from stage data."""
        if stage_name == "collection":
            return {"trace_count": data.get("trace_count", 0)}
        elif stage_name == "evaluation":
            return {
                "passed_count": data.get("passed_count", 0),
                "total_count": data.get("total_count", 0),
            }
        elif stage_name == "synthesis":
            return {"has_graph": data.get("graph") is not None}
        elif stage_name == "validation":
            vr = data.get("validation_results")
            if vr is None:
                return {"skipped": True}
            return {
                "success_rate": vr.success_rate,
                "all_passed": vr.all_passed,
                "result_count": len(vr.results),
            }
        return {}

    def _run_from_synthesis(
        self,
        task_description: str,
        traces: List[ExecutionTrace],
        evaluation_results: List[EvaluationResult],
        filtered_traces: List[ExecutionTrace],
    ) -> PipelineResult:
        """Run stages 3-8: normalization → synthesis → validation.

        Used by refine() which handles collection/evaluation itself.
        """
        try:
            # --- Stage 3: Normalization ---
            try:
                normalizer = TraceNormalizer(
                    action_metadata=self._action_metadata,
                    custom_type_map=self._config.custom_type_map,
                )
                normalized = normalizer.normalize(filtered_traces)
            except Exception as exc:
                return PipelineResult(
                    traces=traces,
                    evaluation_results=evaluation_results,
                    error=str(exc),
                    failed_stage="normalization",
                )

            # --- Stage 4: Target Conversion (optional) ---
            if self._config.target_converter is not None:
                try:
                    converter = self._config.target_converter
                    for trace in normalized:
                        converter.convert_all(trace.steps)
                except Exception as exc:
                    return PipelineResult(
                        traces=traces,
                        evaluation_results=evaluation_results,
                        error=str(exc),
                        failed_stage="target_conversion",
                    )

            # --- Stage 5: Alignment ---
            try:
                aligner = TraceAligner()
                aligned = aligner.align(normalized)
            except Exception as exc:
                return PipelineResult(
                    traces=traces,
                    evaluation_results=evaluation_results,
                    error=str(exc),
                    failed_stage="alignment",
                )

            # --- Stage 6: Pattern Extraction ---
            try:
                extractor = PatternExtractor()
                patterns = extractor.extract(aligned)
            except Exception as exc:
                return PipelineResult(
                    traces=traces,
                    evaluation_results=evaluation_results,
                    error=str(exc),
                    failed_stage="extraction",
                )

            # --- Stage 7: Synthesis ---
            try:
                synthesizer = self._create_synthesizer()
                synthesis_result = synthesizer.synthesize(
                    patterns, task_description,
                )

                python_script = synthesis_result.python_script
                if self._config.generate_script and python_script is None:
                    try:
                        python_script = (
                            synthesis_result.graph._generate_python_script()
                        )
                    except Exception:
                        logger.warning(
                            "Python script generation failed", exc_info=True,
                        )
            except Exception as exc:
                return PipelineResult(
                    traces=traces,
                    evaluation_results=evaluation_results,
                    error=str(exc),
                    failed_stage="synthesis",
                )

            self._invoke_hook("synthesis", {
                "graph": synthesis_result.graph,
                "synthesis_report": synthesis_result.report,
                "python_script": python_script,
            })

            # --- Stage 8: Validation (optional) ---
            validation_results = None
            if self._config.validate:
                try:
                    validator = GraphValidator()
                    test_data = [
                        t.input_data or {} for t in filtered_traces
                    ][:self._config.validation_runs]
                    expected = filtered_traces[:self._config.validation_runs]

                    if not test_data:
                        test_data = [{}]
                        expected = [None]  # type: ignore[list-item]

                    validation_results = validator.validate(
                        graph=synthesis_result.graph,
                        task_description=task_description,
                        test_data=test_data,
                        expected_traces=expected,
                    )
                except Exception as exc:
                    return PipelineResult(
                        graph=synthesis_result.graph,
                        synthesis_report=synthesis_result.report,
                        traces=traces,
                        evaluation_results=evaluation_results,
                        python_script=python_script,
                        error=str(exc),
                        failed_stage="validation",
                    )

            self._invoke_hook("validation", {
                "validation_results": validation_results,
            })

            return PipelineResult(
                graph=synthesis_result.graph,
                synthesis_report=synthesis_result.report,
                validation_results=validation_results,
                traces=traces,
                evaluation_results=evaluation_results,
                python_script=python_script,
            )
        except PipelineAborted as exc:
            return PipelineResult(
                traces=traces,
                evaluation_results=evaluation_results,
                error=str(exc),
                failed_stage=f"{exc.stage}_aborted",
            )

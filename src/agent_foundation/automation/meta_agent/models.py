"""
Core data models for the Meta Agent Workflow pipeline.

All models use Python dataclasses. They cover the full pipeline:
trace collection, evaluation, alignment, pattern extraction, graph
synthesis, validation, and pipeline orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from science_modeling_tools.automation.meta_agent.evaluator import EvaluationResult
    from science_modeling_tools.automation.meta_agent.target_converter import TargetConverterBase


# ---------------------------------------------------------------------------
# Trace Models
# ---------------------------------------------------------------------------

@dataclass
class TraceActionResult:
    """Structured result of a single action execution.

    Mirrors the key fields from WebDriverActionResult.
    """

    success: bool = True
    action_skipped: bool = False
    skip_reason: Optional[str] = None
    value: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d: Dict[str, Any] = {"success": self.success}
        if self.action_skipped:
            d["action_skipped"] = self.action_skipped
        if self.skip_reason is not None:
            d["skip_reason"] = self.skip_reason
        if self.value is not None:
            d["value"] = self.value
        if self.error is not None:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceActionResult":
        """Deserialize from a JSON-compatible dict."""
        return cls(
            success=data.get("success", True),
            action_skipped=data.get("action_skipped", False),
            skip_reason=data.get("skip_reason"),
            value=data.get("value"),
            error=data.get("error"),
        )


@dataclass
class TraceStep:
    """A single action within an execution trace.

    Corresponds to a single AgentAction from AgentResponse.next_actions.
    """

    action_type: str
    target: Optional[Any] = None
    args: Optional[Dict[str, Any]] = None
    result: Optional[TraceActionResult] = None
    timestamp: Optional[datetime] = None
    html_before: Optional[str] = None
    html_after: Optional[str] = None
    source_url: Optional[str] = None
    action_group_index: int = 0
    parallel_index: int = 0
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d: Dict[str, Any] = {"action_type": self.action_type}
        if self.target is not None:
            d["target"] = self.target
        if self.args is not None:
            d["args"] = self.args
        if self.result is not None:
            d["result"] = self.result.to_dict()
        if self.timestamp is not None:
            d["timestamp"] = self.timestamp.isoformat()
        if self.html_before is not None:
            d["html_before"] = self.html_before
        if self.html_after is not None:
            d["html_after"] = self.html_after
        if self.source_url is not None:
            d["source_url"] = self.source_url
        if self.action_group_index != 0:
            d["action_group_index"] = self.action_group_index
        if self.parallel_index != 0:
            d["parallel_index"] = self.parallel_index
        if self.reasoning is not None:
            d["reasoning"] = self.reasoning
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceStep":
        """Deserialize from a JSON-compatible dict."""
        result_data = data.get("result")
        result = TraceActionResult.from_dict(result_data) if result_data is not None else None

        ts_raw = data.get("timestamp")
        timestamp = datetime.fromisoformat(ts_raw) if ts_raw is not None else None

        return cls(
            action_type=data["action_type"],
            target=data.get("target"),
            args=data.get("args"),
            result=result,
            timestamp=timestamp,
            html_before=data.get("html_before"),
            html_after=data.get("html_after"),
            source_url=data.get("source_url"),
            action_group_index=data.get("action_group_index", 0),
            parallel_index=data.get("parallel_index", 0),
            reasoning=data.get("reasoning"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExecutionTrace:
    """Complete trace of a single agent run.

    Parsed from a session directory containing JSONL logs and .parts/
    artifacts written by SessionLogger + JsonLogger.
    """

    trace_id: str
    task_description: str
    steps: List[TraceStep]
    input_data: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    session_dir: Optional[str] = None
    turn_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "task_description": self.task_description,
            "steps": [s.to_dict() for s in self.steps],
            "success": self.success,
        }
        if self.input_data is not None:
            d["input_data"] = self.input_data
        if self.error is not None:
            d["error"] = self.error
        if self.start_time is not None:
            d["start_time"] = self.start_time.isoformat()
        if self.end_time is not None:
            d["end_time"] = self.end_time.isoformat()
        if self.session_dir is not None:
            d["session_dir"] = self.session_dir
        if self.turn_count != 0:
            d["turn_count"] = self.turn_count
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionTrace":
        """Deserialize from a JSON-compatible dict."""
        steps = [TraceStep.from_dict(s) for s in data.get("steps", [])]

        start_raw = data.get("start_time")
        start_time = datetime.fromisoformat(start_raw) if start_raw is not None else None

        end_raw = data.get("end_time")
        end_time = datetime.fromisoformat(end_raw) if end_raw is not None else None

        return cls(
            trace_id=data["trace_id"],
            task_description=data.get("task_description", ""),
            steps=steps,
            input_data=data.get("input_data"),
            success=data.get("success", True),
            error=data.get("error"),
            start_time=start_time,
            end_time=end_time,
            session_dir=data.get("session_dir"),
            turn_count=data.get("turn_count", 0),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Alignment Models
# ---------------------------------------------------------------------------

class AlignmentType(Enum):
    """Classification of an aligned position across traces."""

    DETERMINISTIC = "deterministic"
    PARAMETERIZABLE = "parameterizable"
    VARIABLE = "variable"
    OPTIONAL = "optional"
    BRANCH_POINT = "branch_point"


@dataclass
class AlignedPosition:
    """A single position in the aligned trace set."""

    index: int
    alignment_type: AlignmentType
    steps: Dict[str, Optional[TraceStep]]  # trace_id -> step (None if absent)
    confidence: float = 1.0


@dataclass
class AlignedTraceSet:
    """Result of aligning multiple execution traces."""

    positions: List[AlignedPosition]
    trace_ids: List[str]
    alignment_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pattern Models
# ---------------------------------------------------------------------------

@dataclass
class LoopPattern:
    """A detected loop pattern in the aligned traces."""

    body_start: int
    body_end: int
    min_iterations: int
    max_iterations: int
    body_steps: List[AlignedPosition]


@dataclass
class BranchPattern:
    """A detected branch pattern where traces diverge."""

    branch_point_index: int
    branches: Dict[str, List[AlignedPosition]]
    condition_description: str
    condition_source: Optional[str] = None


@dataclass
class ParameterizableInfo:
    """Info about a parameterizable step's template variables."""

    variable_args: Dict[str, str]   # arg_key -> template_variable_name
    constant_args: Dict[str, Any]   # arg_key -> constant_value


@dataclass
class ExtractedPatterns:
    """All patterns extracted from an aligned trace set."""

    deterministic_steps: List[AlignedPosition]
    parameterizable_steps: List[Tuple[AlignedPosition, ParameterizableInfo]]
    variable_steps: List[AlignedPosition]
    optional_steps: List[AlignedPosition]
    branch_patterns: List[BranchPattern]
    loop_patterns: List[LoopPattern]
    user_input_boundaries: List[int]
    step_order: List[int]


# ---------------------------------------------------------------------------
# Synthesis Models
# ---------------------------------------------------------------------------

@dataclass
class SynthesisReport:
    """Report documenting the synthesis results."""

    total_steps: int
    deterministic_count: int
    parameterizable_count: int
    agent_node_count: int
    optional_count: int
    user_input_boundary_count: int
    branch_count: int
    loop_count: int
    synthesis_strategy: str = "rule_based"
    target_strategy_coverage: Dict[str, int] = field(default_factory=dict)
    template_variables: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "total_steps": self.total_steps,
            "deterministic_count": self.deterministic_count,
            "parameterizable_count": self.parameterizable_count,
            "agent_node_count": self.agent_node_count,
            "optional_count": self.optional_count,
            "user_input_boundary_count": self.user_input_boundary_count,
            "branch_count": self.branch_count,
            "loop_count": self.loop_count,
            "synthesis_strategy": self.synthesis_strategy,
            "target_strategy_coverage": dict(self.target_strategy_coverage),
            "template_variables": list(self.template_variables),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynthesisReport":
        """Deserialize from a JSON-compatible dict."""
        return cls(
            total_steps=data["total_steps"],
            deterministic_count=data["deterministic_count"],
            parameterizable_count=data["parameterizable_count"],
            agent_node_count=data["agent_node_count"],
            optional_count=data["optional_count"],
            user_input_boundary_count=data["user_input_boundary_count"],
            branch_count=data["branch_count"],
            loop_count=data["loop_count"],
            synthesis_strategy=data.get("synthesis_strategy", "rule_based"),
            target_strategy_coverage=data.get("target_strategy_coverage", {}),
            template_variables=data.get("template_variables", []),
            warnings=data.get("warnings", []),
        )


# ---------------------------------------------------------------------------
# Validation Models
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of validating a single execution."""

    input_data: Optional[Dict[str, Any]]
    passed: bool
    divergence_point: Optional[int] = None
    expected_outcome: Optional[Any] = None
    actual_outcome: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class ValidationResults:
    """Aggregate validation results."""

    results: List[ValidationResult]

    @property
    def success_rate(self) -> float:
        """Fraction of results that passed. Returns 0.0 for empty list."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def all_passed(self) -> bool:
        """True when every result passed."""
        return all(r.passed for r in self.results)


# ---------------------------------------------------------------------------
# Pipeline Models
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for the meta agent pipeline.

    The ``evaluation_strategy`` and ``synthesis_strategy`` fields are stored
    as plain strings to avoid a circular import with the evaluator and
    synthesizer modules.  The pipeline maps them to the corresponding enum
    values at runtime.
    """

    run_count: int = 5
    validate: bool = True
    validation_runs: int = 3
    agent_action_type: str = "meta_workflow_agent"
    evaluation_strategy: str = "exception_only"
    synthesis_strategy: str = "rule_based"
    min_success_traces: int = 1
    max_retry_rounds: int = 3
    generate_script: bool = False
    prompt_templates: Optional[Dict[str, str]] = None
    custom_type_map: Optional[Dict[str, str]] = None
    target_converter: Optional["TargetConverterBase"] = None


@dataclass
class PipelineResult:
    """Result of the full pipeline execution."""

    graph: Optional[Any] = None
    synthesis_report: Optional[SynthesisReport] = None
    validation_results: Optional[ValidationResults] = None
    traces: List[ExecutionTrace] = field(default_factory=list)
    evaluation_results: List[Any] = field(default_factory=list)  # List[EvaluationResult]
    python_script: Optional[str] = None
    error: Optional[str] = None
    failed_stage: Optional[str] = None

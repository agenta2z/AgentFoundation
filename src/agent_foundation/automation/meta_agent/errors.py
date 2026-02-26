"""
Exception hierarchy for the Meta Agent Workflow pipeline.

All exceptions inherit from MetaAgentError so callers can catch
pipeline errors at any granularity — from a specific stage failure
down to the base class for a blanket handler.
"""

from typing import List, Optional

from rich_python_utils.common_objects.workflow.common.exceptions import WorkflowAborted


class MetaAgentError(Exception):
    """Base exception for meta agent workflow errors."""

    pass


class TraceCollectionError(MetaAgentError):
    """Error during trace collection phase."""

    def __init__(self, run_index: int, original_error: Exception):
        self.run_index = run_index
        self.original_error = original_error
        super().__init__(
            f"Trace collection failed at run {run_index}: {original_error}"
        )


class TraceNormalizationError(MetaAgentError):
    """Error during trace normalization."""

    def __init__(self, trace_id: str, step_index: int, message: str):
        self.trace_id = trace_id
        self.step_index = step_index
        super().__init__(
            f"Normalization failed for trace '{trace_id}' at step {step_index}: {message}"
        )


class TraceAlignmentError(MetaAgentError):
    """Error during trace alignment."""

    def __init__(self, message: str, trace_ids: Optional[List[str]] = None):
        self.trace_ids = trace_ids or []
        super().__init__(f"Alignment failed: {message}")


class PatternExtractionError(MetaAgentError):
    """Error during pattern extraction."""

    pass


class TraceEvaluationError(MetaAgentError):
    """Error during trace evaluation phase."""

    def __init__(self, trace_id: str, message: str):
        self.trace_id = trace_id
        super().__init__(
            f"Trace evaluation failed for trace '{trace_id}': {message}"
        )


class InsufficientSuccessTracesError(MetaAgentError):
    """Raised when too few traces pass evaluation to proceed with the pipeline."""

    def __init__(self, required: int, actual: int, total: int,
                 traces=None, evaluation_results=None):
        self.required = required
        self.actual = actual
        self.total = total
        self.traces = traces or []
        self.evaluation_results = evaluation_results or []
        super().__init__(
            f"Insufficient successful traces: {actual}/{total} passed evaluation, "
            f"but {required} required"
        )


class GraphSynthesisError(MetaAgentError):
    """Error during graph synthesis."""

    def __init__(self, message: str, pattern_type: Optional[str] = None):
        self.pattern_type = pattern_type
        super().__init__(f"Synthesis failed: {message}")


class PipelineStageError(MetaAgentError):
    """Error with pipeline stage context."""

    def __init__(self, stage: str, original_error: Exception):
        self.stage = stage
        self.original_error = original_error
        super().__init__(f"Pipeline failed at stage '{stage}': {original_error}")


class PipelineAborted(MetaAgentError, WorkflowAborted):
    """Raised when the pipeline is aborted via stage_hook.

    Inherits from both MetaAgentError (for pipeline callers) and
    WorkflowAborted (for Workflow's _handle_abort to catch it).
    Uses direct attribute assignment to avoid MRO conflicts from
    calling both parent __init__ methods.
    """

    def __init__(self, stage: str, message: str = "Pipeline aborted by user"):
        # Single __init__ call through MRO (MetaAgentError → WorkflowAborted)
        super().__init__(f"Pipeline aborted at stage '{stage}': {message}")
        # Override WorkflowAborted defaults AFTER super().__init__
        self.stage = stage
        self.step_name = stage
        self.step_index = None
        self.partial_result = None

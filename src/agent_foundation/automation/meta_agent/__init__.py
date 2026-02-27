"""
Meta Agent Workflow package.

Automates the creation of deterministic ActionGraph workflows from agent
execution traces. The pipeline collects traces from multiple agent runs,
normalizes and aligns them, extracts common patterns, and synthesizes an
ActionGraph using the existing fluent API.
"""

from agent_foundation.automation.meta_agent.errors import (
    GraphSynthesisError,
    InsufficientSuccessTracesError,
    MetaAgentError,
    PatternExtractionError,
    PipelineStageError,
    TraceAlignmentError,
    TraceCollectionError,
    TraceEvaluationError,
    TraceNormalizationError,
)
from agent_foundation.automation.meta_agent.collector import (
    TraceCollector,
)
from agent_foundation.automation.meta_agent.normalizer import (
    TraceNormalizer,
)
from agent_foundation.automation.meta_agent.synthetic_data import (
    SyntheticDataProvider,
)
from agent_foundation.automation.meta_agent.aligner import (
    TraceAligner,
)
from agent_foundation.automation.meta_agent.target_converter import (
    TargetConverterBase,
    TargetSpec,
    TargetSpecWithFallback,
)
from agent_foundation.automation.meta_agent.pattern_extractor import (
    PatternExtractor,
)
from agent_foundation.automation.meta_agent.evaluator import (
    EvaluationResult,
    EvaluationRule,
    EvaluationStrategy,
    TraceEvaluator,
)
from agent_foundation.automation.meta_agent.synthesizer import (
    ActionDecision,
    GraphSynthesizer,
    HybridSynthesizer,
    LLMSynthesizer,
    RuleBasedSynthesizer,
    SynthesisResult,
    SynthesisStrategy,
)
from agent_foundation.automation.meta_agent.validator import (
    GraphValidator,
)
from agent_foundation.automation.meta_agent.pipeline import (
    MetaAgentPipeline,
)
from agent_foundation.automation.meta_agent.prompt_templates import (
    DEFAULT_PROMPT_TEMPLATES,
    EVALUATION_TEMPLATE_KEY,
    SYNTHESIS_TEMPLATE_KEY,
    build_evaluation_feed,
    build_synthesis_feed,
    create_prompt_formatter,
)
from agent_foundation.automation.meta_agent.models import (
    AlignedPosition,
    AlignedTraceSet,
    AlignmentType,
    ExecutionTrace,
    ExtractedPatterns,
    PipelineConfig,
    PipelineResult,
    SynthesisReport,
    TraceActionResult,
    TraceStep,
    ValidationResult,
    ValidationResults,
)

__all__ = [
    "ActionDecision",
    "DEFAULT_PROMPT_TEMPLATES",
    "AlignedPosition",
    "AlignedTraceSet",
    "AlignmentType",
    "EVALUATION_TEMPLATE_KEY",
    "EvaluationResult",
    "EvaluationRule",
    "EvaluationStrategy",
    "ExecutionTrace",
    "ExtractedPatterns",
    "GraphSynthesisError",
    "GraphSynthesizer",
    "GraphValidator",
    "HybridSynthesizer",
    "InsufficientSuccessTracesError",
    "LLMSynthesizer",
    "MetaAgentError",
    "MetaAgentPipeline",
    "PatternExtractionError",
    "PatternExtractor",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStageError",
    "RuleBasedSynthesizer",
    "SyntheticDataProvider",
    "SynthesisReport",
    "SynthesisResult",
    "SYNTHESIS_TEMPLATE_KEY",
    "SynthesisStrategy",
    "TargetConverterBase",
    "TargetSpec",
    "TargetSpecWithFallback",
    "TargetStrategyConverter",
    "TraceActionResult",
    "TraceAligner",
    "TraceAlignmentError",
    "TraceCollectionError",
    "TraceCollector",
    "TraceEvaluationError",
    "TraceEvaluator",
    "TraceNormalizationError",
    "TraceNormalizer",
    "TraceStep",
    "ValidationResult",
    "ValidationResults",
    "build_evaluation_feed",
    "build_synthesis_feed",
    "create_prompt_formatter",
]


def __getattr__(name: str):
    """Lazy backward-compatible import of TargetStrategyConverter."""
    if name == "TargetStrategyConverter":
        from agent_foundation.automation.meta_agent.target_converter import (
            TargetStrategyConverter,
        )

        return TargetStrategyConverter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

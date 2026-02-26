"""
Trace evaluation for the Meta Agent Workflow pipeline.

Evaluates collected execution traces for quality and correctness using
configurable strategies before they enter the analysis pipeline. Supports
three strategies:

- EXCEPTION_ONLY: Filter traces by success flag (default).
- RULE_BASED: Apply configurable predicate rules with error/warning severity.
- LLM_JUDGE: Use InferencerBase to assess trace quality with a score threshold.

The evaluator does NOT modify traces — it produces EvaluationResult objects
that the pipeline uses to filter traces before normalization.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rich_python_utils.string_utils.formatting.template_manager import (
        TemplateManager,
    )

from science_modeling_tools.automation.meta_agent.models import ExecutionTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------


class EvaluationStrategy(Enum):
    """Strategy for evaluating trace quality."""

    EXCEPTION_ONLY = "exception_only"
    RULE_BASED = "rule_based"
    LLM_JUDGE = "llm_judge"


@dataclass
class EvaluationRule:
    """A single rule for evaluating trace quality.

    Rules are predicates over ExecutionTrace that return True if the trace
    passes the rule, False otherwise.
    """

    name: str
    description: str
    predicate: Callable[[ExecutionTrace], bool]
    severity: str = "error"  # "error" (trace rejected) or "warning" (trace kept, flagged)


@dataclass
class EvaluationResult:
    """Result of evaluating a single trace."""

    trace_id: str
    passed: bool
    score: float = 1.0
    failed_rules: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TraceEvaluator
# ---------------------------------------------------------------------------


class TraceEvaluator:
    """Evaluates execution traces for quality and correctness.

    Supports three evaluation strategies:

    - **EXCEPTION_ONLY** — Filters traces where ``success=False``.
      Default strategy; requires no additional configuration.

    - **RULE_BASED** — Applies a list of :class:`EvaluationRule` predicates.
      Rules with ``severity="error"`` cause the trace to fail.
      Rules with ``severity="warning"`` flag the trace but don't reject it.

    - **LLM_JUDGE** — Uses an ``InferencerBase`` instance to evaluate trace
      quality. The inferencer receives a structured prompt with the trace
      steps and task description, and returns a quality assessment.

    The evaluator does NOT modify traces — it produces :class:`EvaluationResult`
    objects that the pipeline uses to filter traces before normalization.
    """

    def __init__(
        self,
        strategy: EvaluationStrategy = EvaluationStrategy.EXCEPTION_ONLY,
        rules: Optional[List[EvaluationRule]] = None,
        inferencer: Optional[Any] = None,  # InferencerBase
        min_score: float = 0.5,
        prompt_formatter: Optional["TemplateManager"] = None,
    ):
        """
        Args:
            strategy: Evaluation strategy to use.
            rules: Evaluation rules (required for RULE_BASED).
            inferencer: InferencerBase instance (required for LLM_JUDGE).
            min_score: Minimum quality score for LLM_JUDGE pass threshold.
            prompt_formatter: Optional TemplateManager for rendering prompts.
                When *None*, the legacy inline f-string prompt is used.

        Raises:
            ValueError: If RULE_BASED without rules or LLM_JUDGE without inferencer.
        """
        self._strategy = strategy
        self._rules = rules or []
        self._inferencer = inferencer
        self._min_score = min_score
        self._prompt_formatter = prompt_formatter

        if strategy == EvaluationStrategy.RULE_BASED and not self._rules:
            raise ValueError(
                "RULE_BASED strategy requires at least one evaluation rule"
            )
        if strategy == EvaluationStrategy.LLM_JUDGE and self._inferencer is None:
            raise ValueError(
                "LLM_JUDGE strategy requires an InferencerBase instance"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        traces: List[ExecutionTrace],
        task_description: str = "",
    ) -> List[EvaluationResult]:
        """Evaluate all traces and return results in the same order.

        Args:
            traces: Execution traces to evaluate.
            task_description: Task description (used by LLM_JUDGE for context).

        Returns:
            One :class:`EvaluationResult` per trace, same order as input.
        """
        results: List[EvaluationResult] = []
        for trace in traces:
            if self._strategy == EvaluationStrategy.EXCEPTION_ONLY:
                results.append(self._evaluate_exception_only(trace))
            elif self._strategy == EvaluationStrategy.RULE_BASED:
                results.append(self._evaluate_rule_based(trace))
            elif self._strategy == EvaluationStrategy.LLM_JUDGE:
                results.append(self._evaluate_llm_judge(trace, task_description))
        return results

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _evaluate_exception_only(self, trace: ExecutionTrace) -> EvaluationResult:
        """Pass traces with ``success=True``, fail those with ``success=False``."""
        return EvaluationResult(
            trace_id=trace.trace_id,
            passed=trace.success,
            score=1.0 if trace.success else 0.0,
        )

    def _evaluate_rule_based(self, trace: ExecutionTrace) -> EvaluationResult:
        """Apply all configured rules. Error-severity failures reject the trace."""
        failed_rules: List[str] = []
        warnings: List[str] = []

        for rule in self._rules:
            try:
                passed = rule.predicate(trace)
            except Exception:
                # Treat predicate exceptions as rule failure.
                logger.warning(
                    "Rule '%s' raised an exception for trace '%s'; treating as failure",
                    rule.name,
                    trace.trace_id,
                )
                passed = False

            if not passed:
                if rule.severity == "error":
                    failed_rules.append(rule.name)
                else:
                    warnings.append(rule.name)

        has_errors = len(failed_rules) > 0
        return EvaluationResult(
            trace_id=trace.trace_id,
            passed=not has_errors,
            score=0.0 if has_errors else 1.0,
            failed_rules=failed_rules,
            warnings=warnings,
        )

    def _evaluate_llm_judge(
        self,
        trace: ExecutionTrace,
        task_description: str,
    ) -> EvaluationResult:
        """Use InferencerBase to assess trace quality and assign a score."""
        prompt = self._build_llm_prompt(trace, task_description)

        try:
            response = self._inferencer.infer(prompt)
            score = self._parse_score(response)
        except Exception as exc:
            logger.warning(
                "LLM judge failed for trace '%s': %s; marking as failed",
                trace.trace_id,
                exc,
            )
            return EvaluationResult(
                trace_id=trace.trace_id,
                passed=False,
                score=0.0,
                metadata={"llm_error": str(exc)},
            )

        passed = score >= self._min_score
        return EvaluationResult(
            trace_id=trace.trace_id,
            passed=passed,
            score=score,
            metadata={"llm_response": str(response)},
        )

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _build_llm_prompt(self, trace: ExecutionTrace, task_description: str) -> str:
        """Build a structured prompt for the LLM judge.

        When a ``prompt_formatter`` (TemplateManager) was provided at
        construction time, renders the evaluation template with it.
        Otherwise falls back to the original inline f-string for
        bit-identical backward compatibility.
        """
        from science_modeling_tools.automation.meta_agent.prompt_templates import (
            build_evaluation_feed,
            EVALUATION_TEMPLATE_KEY,
        )

        feed = build_evaluation_feed(trace, task_description)

        if self._prompt_formatter is not None:
            return self._prompt_formatter(EVALUATION_TEMPLATE_KEY, **feed)

        # Legacy fallback — exact original f-string output.
        return (
            "Evaluate the quality of the following execution trace.\n"
            f"Task: {task_description}\n"
            f"Trace ID: {trace.trace_id}\n"
            f"Success: {trace.success}\n"
            f"Steps ({len(trace.steps)}):\n{feed['steps_text']}\n\n"
            "Respond with a JSON object containing a single key 'score' "
            "with a float value between 0.0 and 1.0, where 1.0 is perfect quality.\n"
            'Example: {"score": 0.85}'
        )

    @staticmethod
    def _parse_score(response: Any) -> float:
        """Extract a quality score from the LLM response.

        Handles string responses (JSON or plain float) and dict responses.
        Falls back to 0.0 on parse failure.
        """
        if isinstance(response, (int, float)):
            return max(0.0, min(1.0, float(response)))

        if isinstance(response, dict):
            raw = response.get("score", 0.0)
            return max(0.0, min(1.0, float(raw)))

        text = str(response).strip()

        # Try JSON parse first.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "score" in parsed:
                return max(0.0, min(1.0, float(parsed["score"])))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Try plain float.
        try:
            return max(0.0, min(1.0, float(text)))
        except (ValueError, TypeError):
            pass

        logger.warning("Could not parse LLM score from response: %s", text)
        return 0.0

"""
Prompt templates for the Meta Agent pipeline.

Defines default Jinja2 templates for LLM-based evaluation and synthesis
prompts.  Templates can be overridden via ``PipelineConfig.prompt_templates``
for experimentation without code changes.

The module also provides *feed builder* functions that convert pipeline
domain objects (traces, aligned positions) into flat dicts of template
variables, separating dynamic data construction from static template
structure.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_foundation.automation.meta_agent.models import (
        AlignedPosition,
        ExecutionTrace,
    )

# ---------------------------------------------------------------------------
# Template key constants
# ---------------------------------------------------------------------------

EVALUATION_TEMPLATE_KEY = "evaluation"
SYNTHESIS_TEMPLATE_KEY = "synthesis"

# ---------------------------------------------------------------------------
# Jinja2 template strings
# ---------------------------------------------------------------------------
# Templates use explicit ``\n`` concatenation (not triple-quoted strings) to
# guarantee character-exact whitespace matching with the original inline
# f-strings in evaluator.py and synthesizer.py.

_EVALUATION_TEMPLATE_JINJA2 = (
    "Evaluate the quality of the following execution trace.\n"
    "Task: {{ task_description }}\n"
    "Trace ID: {{ trace_id }}\n"
    "Success: {{ trace_success }}\n"
    "Steps ({{ step_count }}):\n"
    "{{ steps_text }}\n"
    "\n"
    "Respond with a JSON object containing a single key 'score' "
    "with a float value between 0.0 and 1.0, where 1.0 is perfect quality.\n"
    'Example: {"score": 0.85}'
)

_SYNTHESIS_TEMPLATE_JINJA2 = (
    "Synthesize an ActionGraph action for a {{ pattern_type }} pattern.\n"
    "Task: {{ task_description }}\n"
    "Position index: {{ position_index }}\n"
    "Observed steps across traces:\n"
    "{{ steps_text }}\n"
    "{{ param_section }}\n"
    "Respond with a JSON object containing:\n"
    '  "action_type": string (the action type to use),\n'
    '  "target": string or null (the target element or description),\n'
    '  "args": object or null (action arguments),\n'
    '  "confidence": float between 0.0 and 1.0,\n'
    '  "reasoning": string (explanation of your decision)\n'
    "\n"
    'Example: {"action_type": "click", "target": "btn-submit", '
    '"args": null, "confidence": 0.9, '
    '"reasoning": "All traces show a click on submit button"}'
)

# Default aliases — swap to switch the template engine project-wide.
_EVALUATION_TEMPLATE = _EVALUATION_TEMPLATE_JINJA2
_SYNTHESIS_TEMPLATE = _SYNTHESIS_TEMPLATE_JINJA2

# ---------------------------------------------------------------------------
# Default template dict
# ---------------------------------------------------------------------------

DEFAULT_PROMPT_TEMPLATES: Dict[str, str] = {
    EVALUATION_TEMPLATE_KEY: _EVALUATION_TEMPLATE,
    SYNTHESIS_TEMPLATE_KEY: _SYNTHESIS_TEMPLATE,
}

# ---------------------------------------------------------------------------
# Feed builders
# ---------------------------------------------------------------------------


def build_evaluation_feed(
    trace: "ExecutionTrace",
    task_description: str,
) -> Dict[str, Any]:
    """Build the template variable dict for the evaluation prompt.

    Extracts dynamic data from *trace* that was previously computed
    inline in ``TraceEvaluator._build_llm_prompt()``.

    Returns:
        Dict with keys: ``task_description``, ``trace_id``,
        ``trace_success``, ``step_count``, ``steps_text``.
    """
    steps_summary = []
    for i, step in enumerate(trace.steps):
        entry = f"  Step {i + 1}: action_type={step.action_type}"
        if step.target is not None:
            entry += f", target={step.target}"
        if step.result is not None:
            entry += f", success={step.result.success}"
        steps_summary.append(entry)

    steps_text = "\n".join(steps_summary) if steps_summary else "  (no steps)"

    return {
        "task_description": task_description,
        "trace_id": trace.trace_id,
        "trace_success": trace.success,
        "step_count": len(trace.steps),
        "steps_text": steps_text,
    }


def build_synthesis_feed(
    position: "AlignedPosition",
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the template variable dict for the synthesis prompt.

    Extracts dynamic data from *position* and *context* that was
    previously computed inline in
    ``GraphSynthesizer._build_synthesis_prompt()``.

    Returns:
        Dict with keys: ``pattern_type``, ``task_description``,
        ``position_index``, ``steps_text``, ``param_section``.
    """
    pattern_type = context.get("pattern_type", "unknown")
    task_description = context.get("task_description", "")

    steps_summary = []
    for trace_id, step in position.steps.items():
        if step is None:
            steps_summary.append(f"  Trace '{trace_id}': (absent)")
        else:
            entry = f"  Trace '{trace_id}': action_type={step.action_type}"
            if step.target is not None:
                entry += f", target={step.target}"
            if step.args is not None:
                entry += f", args={step.args}"
            steps_summary.append(entry)

    steps_text = "\n".join(steps_summary) if steps_summary else "  (no steps)"

    param_section = ""
    param_info = context.get("param_info")
    if param_info is not None:
        param_section = (
            f"\nParameterizable info:\n"
            f"  Variable args: {param_info.variable_args}\n"
            f"  Constant args: {param_info.constant_args}\n"
        )

    return {
        "pattern_type": pattern_type,
        "task_description": task_description,
        "position_index": position.index,
        "steps_text": steps_text,
        "param_section": param_section,
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_prompt_formatter(
    templates: Optional[Dict[str, str]] = None,
) -> Any:
    """Create a ``TemplateManager`` for meta_agent prompts.

    Uses ``active_template_type=None`` for flat-dict resolution (no
    namespace hierarchy).  Template keys are looked up directly in the
    dict without any prefix.

    Args:
        templates: Custom template dict.  Defaults to
            :data:`DEFAULT_PROMPT_TEMPLATES`.

    Returns:
        A ``TemplateManager`` instance.
    """
    from rich_python_utils.string_utils.formatting.template_manager import (
        TemplateManager,
    )

    return TemplateManager(
        templates=templates or DEFAULT_PROMPT_TEMPLATES,
        active_template_type=None,  # flat namespace — direct key lookup
    )

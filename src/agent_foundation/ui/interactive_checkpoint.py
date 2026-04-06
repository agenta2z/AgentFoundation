

"""Interactive checkpoint utilities for flow inferencers.

Shared utility module (not mixin) to avoid MRO issues with dual inheritance
patterns like PlanThenImplementInferencer(InferencerBase, Workflow).

Provides checkpoint functions that:
  - Present structured options to the user via interactive transport
  - Fall through to default action when interactive is None
  - Return CheckpointResult with the user's decision
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from agent_foundation.ui.input_modes import (
    ChoiceOption,
    InputMode,
    InputModeConfig,
    single_choice,
    multiple_choices,
)
from agent_foundation.ui.interactive_base import (
    InteractionFlags,
    InteractiveBase,
)

logger = logging.getLogger(__name__)


@dataclass
class CheckpointResult:
    """Result of an interactive checkpoint."""

    action: str  # "approve", "modify", "reject", "skip", "timeout", "select"
    user_input: str = ""
    selected_indices: list[int] = field(default_factory=list)
    modified_items: list[Any] = field(default_factory=list)


async def run_checkpoint(
    interactive: Optional[InteractiveBase],
    prompt: str,
    options: list[ChoiceOption],
    default_action: str = "approve",
    allow_custom: bool = True,
) -> CheckpointResult:
    """Run a generic interactive checkpoint.

    If interactive is None, returns the default action immediately.

    Args:
        interactive: The interactive transport (or None).
        prompt: The prompt to display to the user.
        options: The choice options to present.
        default_action: The action to return if no interactive.
        allow_custom: Whether to allow custom text input.

    Returns:
        CheckpointResult with the user's decision.
    """
    if interactive is None:
        return CheckpointResult(action=default_action)

    input_mode = single_choice(options, allow_custom=allow_custom, prompt=prompt)
    await interactive.asend_response(
        prompt,
        flag=InteractionFlags.PendingInput,
        input_mode=input_mode,
    )

    user_input = await interactive.aget_input()
    if user_input is None:
        return CheckpointResult(action=default_action)

    # Extract value
    if isinstance(user_input, dict):
        value = str(user_input.get("user_input", user_input.get("content", "")))
    else:
        value = str(user_input)

    # Map value to action
    for opt in options:
        if value == opt.value:
            return CheckpointResult(action=opt.value, user_input=value)

    # Custom input
    return CheckpointResult(action="custom", user_input=value)


async def checkpoint_plan_review(
    interactive: Optional[InteractiveBase],
    plan_summary: str,
    default_action: str = "approve",
) -> CheckpointResult:
    """Enhanced plan review checkpoint with approve/modify/reject options.

    Args:
        interactive: The interactive transport (or None).
        plan_summary: A text summary of the plan to review.
        default_action: Default action if no interactive.

    Returns:
        CheckpointResult with action in ("approve", "modify", "reject").
    """
    prompt = f"## Plan Review\n\n{plan_summary}\n\nWhat would you like to do?"
    options = [
        ChoiceOption(
            label="Approve",
            value="approve",
            follow_up_prompt="",
        ),
        ChoiceOption(
            label="Modify",
            value="modify",
            follow_up_prompt="Describe the modifications you'd like:",
        ),
        ChoiceOption(
            label="Reject",
            value="reject",
            follow_up_prompt="Reason for rejection (optional):",
        ),
    ]

    result = await run_checkpoint(
        interactive, prompt, options,
        default_action=default_action,
        allow_custom=True,
    )

    # If modify was selected, collect the modification details
    if result.action == "modify" and interactive is not None:
        modify_mode = InputModeConfig(
            mode=InputMode.FREE_TEXT,
            prompt="Describe the modifications you'd like:",
        )
        await interactive.asend_response(
            "Describe the modifications you'd like:",
            flag=InteractionFlags.PendingInput,
            input_mode=modify_mode,
        )
        mod_input = await interactive.aget_input()
        if mod_input is not None:
            if isinstance(mod_input, dict):
                result.user_input = str(
                    mod_input.get("user_input", mod_input.get("content", ""))
                )
            else:
                result.user_input = str(mod_input)

    return result


async def checkpoint_breakdown_review(
    interactive: Optional[InteractiveBase],
    sub_queries: list[str],
    default_action: str = "approve",
) -> CheckpointResult:
    """Sub-query selection checkpoint for BreakdownThenAggregate.

    Presents sub-queries as multiple-choice, allowing the user to select
    which ones to execute.

    Args:
        interactive: The interactive transport (or None).
        sub_queries: List of sub-query strings.
        default_action: Default action if no interactive.

    Returns:
        CheckpointResult with selected_indices.
    """
    if interactive is None:
        return CheckpointResult(
            action=default_action,
            selected_indices=list(range(len(sub_queries))),
        )

    options = [
        ChoiceOption(label=q[:80], value=str(i))
        for i, q in enumerate(sub_queries)
    ]
    prompt = (
        "## Sub-query Selection\n\n"
        "Select which sub-queries to execute:\n\n"
        + "\n".join(f"  {i+1}. {q}" for i, q in enumerate(sub_queries))
    )

    input_mode = multiple_choices(
        options, allow_custom=False, prompt=prompt,
    )
    await interactive.asend_response(
        prompt,
        flag=InteractionFlags.PendingInput,
        input_mode=input_mode,
    )

    user_input = await interactive.aget_input()
    if user_input is None:
        return CheckpointResult(
            action=default_action,
            selected_indices=list(range(len(sub_queries))),
        )

    # Parse selections
    if isinstance(user_input, dict):
        value = user_input.get("user_input", "")
    else:
        value = str(user_input)

    # value is pipe-delimited indices from _resolve_multiple_choices
    if isinstance(value, str) and "|" in value:
        indices = []
        for part in value.split("|"):
            try:
                indices.append(int(part))
            except ValueError:
                pass
        return CheckpointResult(
            action="select",
            selected_indices=indices,
        )

    # Single value
    try:
        return CheckpointResult(
            action="select",
            selected_indices=[int(value)],
        )
    except (ValueError, TypeError):
        return CheckpointResult(
            action=default_action,
            selected_indices=list(range(len(sub_queries))),
        )


async def checkpoint_results_review(
    interactive: Optional[InteractiveBase],
    results_summary: str,
    default_action: str = "approve",
) -> CheckpointResult:
    """Results review checkpoint — approve or request re-run.

    Args:
        interactive: The interactive transport (or None).
        results_summary: Text summary of results.
        default_action: Default action if no interactive.

    Returns:
        CheckpointResult with action in ("approve", "rerun", "modify").
    """
    prompt = f"## Results Review\n\n{results_summary}\n\nWhat would you like to do?"
    options = [
        ChoiceOption(label="Approve", value="approve"),
        ChoiceOption(label="Re-run", value="rerun"),
        ChoiceOption(label="Modify and re-run", value="modify"),
    ]
    return await run_checkpoint(
        interactive, prompt, options,
        default_action=default_action,
    )

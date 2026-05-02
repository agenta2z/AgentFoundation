# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Effects re-exports.

Each effect class is a small dataclass whose `apply(inferencer)` performs ONE
explicit mutation. Adding a new effect kind is a one-file change here; the
dispatcher needs zero changes.
"""

from agent_foundation.common.inferencers.agentic_inferencers.conversational.effects.apply_context_updates import (
    ApplyContextUpdates,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.effects.override_next_action_tool_args import (
    OverrideNextActionToolArgs,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.effects.set_prompt_variable import (
    SetPromptVariable,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.effects.set_turn_variables import (
    SetTurnVariables,
)

__all__ = [
    "ApplyContextUpdates",
    "OverrideNextActionToolArgs",
    "SetPromptVariable",
    "SetTurnVariables",
]

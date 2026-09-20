# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""PublishSessionVariablesEffect — writes to prior_context + variable_manager.

Delegates to CI's ``set_session_variables(variables, tool_type=tool_type)`` —
AF's richer wrapper that publishes to both prior_context and variable_manager
with tool-type namespacing (``<tool_type>__<var>`` aliases per A1.b).

No merge check: publish is idempotent by name. Handlers emit one effect per
variable-group; the loop replays them in order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
        ConversationalInferencer,
    )


@dataclass
class PublishSessionVariablesEffect:
    variables: dict[str, Any]
    # Accept the raw enum or a string — `set_session_variables` unwraps via
    # ``getattr(tool_type, "value", tool_type)`` and normalizes hyphens.
    # DO NOT pre-str() the enum: Python's default ``str(Enum)`` returns
    # ``"ClassName.MEMBER"`` (e.g. ``"ConversationToolType.PROPOSAL_SELECTION"``),
    # breaking the ``<tool_type>__<var>`` alias convention.
    tool_type: Any = None

    async def apply(self, inferencer: ConversationalInferencer) -> None:
        inferencer.set_session_variables(self.variables, tool_type=self.tool_type)

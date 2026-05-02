# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""SetTurnVariables effect — replace _pending_variables dynamic mailbox.

The action-tool processing block consumes these via `vm.set` + `add_message`
on the next iteration. Whole-dict-replace semantics; raises on collision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    HandlerResultMergeConflict,
)

if TYPE_CHECKING:
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
        ConversationalInferencer,
    )


@dataclass
class SetTurnVariables:
    variables: dict[str, str]

    async def apply(self, inferencer: ConversationalInferencer) -> None:
        existing = inferencer._next_turn_variables
        if existing is not None:
            raise HandlerResultMergeConflict(
                field_name="_next_turn_variables",
                key=None,
                existing=existing,
                new=self.variables,
            )
        inferencer._next_turn_variables = self.variables

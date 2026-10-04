# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""SetTurnVariables effect — sets the ``turn_variables`` mailbox.

The action-tool processing block consumes these via `vm.set` + `add_message`
on the next iteration. Whole-dict-replace semantics; raises on collision.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
    effect_target,
    EffectTarget,
    HandlerResultMergeConflict,
)


@dataclass
class SetTurnVariables:
    variables: dict[str, str]

    async def apply(self, inferencer: EffectTarget) -> None:
        mailboxes = effect_target(inferencer).mailboxes
        existing = mailboxes.turn_variables
        if existing is not None:
            raise HandlerResultMergeConflict(
                field_name="turn_variables",
                key=None,
                existing=existing,
                new=self.variables,
            )
        mailboxes.turn_variables = self.variables
